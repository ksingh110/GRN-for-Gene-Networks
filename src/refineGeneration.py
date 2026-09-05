import copy
import json
import os
import random
import time

from fastsimulate import simulate_network
from mutateGenerationFurther import apply_mutation
from randomNetwork import prune_unconnected_genes

def compute_refinement_fitness(
    net,
    result,
    target_pattern,
):
    n_points = len(result)
    per_gene_scores = {}

    for i, col in enumerate(result.colnames):
        if col == "time":
            continue

        trace = [
            result[t_idx][i]
            for t_idx in range(n_points)
        ]

        sse = (sum((value - target) ** 2
                for value, target in zip(
                    trace,
                    target_pattern,
                )
            )
            / n_points
        )

        per_gene_scores[col] = float(sse)

    best_gene = min(
        per_gene_scores,
        key=per_gene_scores.get,
    )

    best_score = per_gene_scores[best_gene]

    network_edge_count = len(net["edges"])
    network_gene_count = len(net["genes"])

    network_edge_normalized = (
        (network_edge_count - 4)
        / (12 - 4)
    )

    network_gene_normalized = (
        (network_gene_count - 3)
        / (15 - 3)
    )

    combined_score = 10 * (
        best_score
        + 5 * network_edge_normalized**2
        + 5 * network_gene_normalized**2
    )

    return (
        combined_score,
        best_gene,
        per_gene_scores,
        best_score,
        network_edge_count,
    )


def score_refinement_entry(
    net,
    entry_id,
    target_pattern,
):
    try:
        result = simulate_network(net)

        (
            combined_score,
            best_gene,
            per_gene_scores,
            raw_sse,
            network_size,
        ) = compute_refinement_fitness(
            net,
            result,
            target_pattern,
        )

        fitness = {
            "score": round(combined_score, 6),
            "raw_sse": round(raw_sse, 6),
            "network_size": network_size,
            "best_gene": best_gene,
            "per_gene_scores": {
                gene: round(score, 6)
                for gene, score
                in per_gene_scores.items()
            },
        }

    except Exception as error:
        fitness = {
            "score": 10000000,
            "raw_sse": 10000000,
            "network_size": len(net["edges"]),
            "best_gene": None,
            "per_gene_scores": {},
            "error": str(error),
        }

    return {
        "id": entry_id,
        "genes": net["genes"],
        "edges": net["edges"],
        "degradation_rates": net[
            "degradation_rates"
        ],
        "y0": net["y0"],
        "fitness": fitness,
    }


def compute_refinement_generation_from_seed(
    seed_network,
    generation_number,
    target_pattern,
    mp_pool=None,
    logic_gates=None,
):
    number_of_networks = 100

    prepared = [
        (
            copy.deepcopy(seed_network),
            0,
            target_pattern,
        )
    ]

    for entry_id in range(
        1,
        number_of_networks,
    ):
        mutant = apply_mutation(
            copy.deepcopy(seed_network),
            logic_gates,
        )

        prepared.append(
            (
                mutant,
                entry_id,
                target_pattern,
            )
        )

    if mp_pool is not None:
        networks_out = mp_pool.starmap(
            score_refinement_entry,
            prepared,
        )
    else:
        networks_out = [
            score_refinement_entry(
                network,
                entry_id,
                pattern,
            )
            for network, entry_id, pattern
            in prepared
        ]

    networks_out.sort(
        key=lambda entry:
        entry["fitness"]["score"]
        if entry["fitness"]["score"] is not None
        else float("inf")
    )

    return {
        "generation": generation_number,
        "logic_gates": logic_gates,
        "networks": networks_out,
    }


def compute_refinement_elites(
    networks,
    elite_count,
):
    return copy.deepcopy(
        networks[:elite_count]
    )


def generate_refinement_children(
    population,
    number_of_children,
    start_id,
    target_pattern,
    mp_pool=None,
    logic_gates=None,
):
    def fitness_of(entry):
        score = entry["fitness"]["score"]

        if score is None:
            return float("inf")

        return score

    prepared = []

    for offset in range(number_of_children):
        parent_a, parent_b = random.sample(
            population,
            2,
        )

        if (
            fitness_of(parent_a)
            <= fitness_of(parent_b)
        ):
            parent = parent_a
        else:
            parent = parent_b

        genome = {
            "genes": parent["genes"],
            "edges": parent["edges"],
            "degradation_rates": parent[
                "degradation_rates"
            ],
            "y0": parent["y0"],
        }

        child = apply_mutation(genome, logic_gates)

        prepared.append(
            (
                child,
                start_id + offset,
                target_pattern,
            )
        )

    if mp_pool is not None:
        return mp_pool.starmap(
            score_refinement_entry,
            prepared,
        )

    return [
        score_refinement_entry(
            network,
            entry_id,
            pattern,
        )
        for network, entry_id, pattern
        in prepared
    ]


def create_refinement_generation(
    elite_count,
    generation,
    generation_number,
    target_pattern,
    mp_pool=None,
    logic_gates=None,
):
    elites = compute_refinement_elites(
        generation["networks"],
        elite_count,
    )

    elite_clones = []

    for entry_id, entry in enumerate(elites):
        clone = copy.deepcopy(entry)
        clone["id"] = entry_id
        elite_clones.append(clone)

    number_of_children = 100 - elite_count

    children = generate_refinement_children(
        generation["networks"],
        number_of_children,
        start_id=elite_count,
        target_pattern=target_pattern,
        mp_pool=mp_pool,
        logic_gates=logic_gates,
    )

    next_generation = (
        elite_clones + children
    )

    next_generation.sort(
        key=lambda entry:
        entry["fitness"]["score"]
        if entry["fitness"]["score"] is not None
        else float("inf")
    )

    return {
        "generation": generation_number + 1,
        "logic_gates": logic_gates,
        "networks": next_generation,
    }


def extract_seed_network(seed_data):
    if (
        isinstance(seed_data, dict)
        and "networks" in seed_data
    ):
        seed_entry = seed_data["networks"][0]

    elif (
        isinstance(seed_data, dict)
        and "best_network" in seed_data
    ):
        seed_entry = seed_data["best_network"]

    else:
        seed_entry = seed_data

    required_keys = {
        "genes",
        "edges",
        "degradation_rates",
        "y0",
    }

    missing_keys = (
        required_keys - set(seed_entry)
    )

    if missing_keys:
        raise ValueError(
            "Seed network is missing: "
            + ", ".join(sorted(missing_keys))
        )

    return {
        "genes": copy.deepcopy(
            seed_entry["genes"]
        ),
        "edges": copy.deepcopy(
            seed_entry["edges"]
        ),
        "degradation_rates": copy.deepcopy(
            seed_entry["degradation_rates"]
        ),
        "y0": copy.deepcopy(
            seed_entry["y0"]
        ),
    }


def load_seed_network(path):
    if not path:
        raise ValueError(
            "No refine_seed_path was provided."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Refinement seed not found: {path}"
        )

    with open(path) as file:
        seed_data = json.load(file)

    return extract_seed_network(seed_data)


def run_refinement(
    seed_network,
    target_pattern,
    pool,
    random_seed,
    target=100.0,
    max_generations=300,
    extend_threshold=120.0,
    extend_by=300,
    elite_count=10,
    print_every=20,
    silent=False,
    logic_gates=None,
):
    random.seed(random_seed)

    if logic_gates is not None:
        disallowed_gates = sorted({
            edge["rate_law"]
            for edge in seed_network["edges"]
            if edge["regulator2"] is not None
            and edge["rate_law"] not in logic_gates
        })
        if disallowed_gates:
            raise ValueError(
                "Refinement seed contains excluded logic gates: "
                + ", ".join(disallowed_gates)
            )

    start_time = time.perf_counter()

    data = (
        compute_refinement_generation_from_seed(
            seed_network,
            generation_number=1,
            target_pattern=target_pattern,
            mp_pool=pool,
            logic_gates=logic_gates,
        )
    )

    best_fitness = (
        data["networks"][0]["fitness"]["score"]
    )

    generation_number = 1
    generation_limit = max_generations
    limit_extended = False

    while (
        best_fitness > target
        and generation_number
        <= generation_limit
    ):
        data = create_refinement_generation(
            elite_count,
            data,
            generation_number,
            target_pattern,
            mp_pool=pool,
            logic_gates=logic_gates,
        )

        new_best = (
            data["networks"][0]
            ["fitness"]["score"]
        )

        best_fitness = min(
            best_fitness,
            new_best,
        )

        if (
            not silent
            and generation_number
            % print_every == 0
        ):
            best_entry = data["networks"][0]

            print(
                "Refinement generation: "
                f"{generation_number}, "
                f"fitness: {best_fitness:.3f}, "
                "size: "
                f"{best_entry['fitness']['network_size']}, "
                "raw_sse: "
                f"{best_entry['fitness']['raw_sse']:.3f}"
            )

        if (
            best_fitness < extend_threshold
            and not limit_extended
        ):
            generation_limit += extend_by
            limit_extended = True

        generation_number += 1

    elapsed = (
        time.perf_counter() - start_time
    )

    best_network = {
        **data["networks"][0],
        **prune_unconnected_genes(data["networks"][0]),
    }
    return {
        "converged": best_fitness <= target,
        "seed": random_seed,
        "generations_used": generation_number - 1,
        "elapsed_seconds": elapsed,
        "final_fitness": best_fitness,
        "raw_sse": best_network[
            "fitness"
        ]["raw_sse"],
        "network_size": best_network[
            "fitness"
        ]["network_size"],
        "best_network": best_network,
        "generation": data,
    }
