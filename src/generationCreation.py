import json
import time
import copy
import random
import os
import multiprocessing as mp
from randomNetwork import generate_random_network, prune_unconnected_genes
from fastsimulate import simulate_network
from mutateGeneration import apply_mutation
from visualize import visualize_network
OSCILLATOR = [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0]
SWITCH = [30.0, 30.0, 30.0, 30.0, 30.0,5.0, 5.0, 5.0, 5.0, 5.0]
DIP = [30.0, 30.0, 30.0, 30.0, 5.0,5.0, 30.0, 30.0, 30.0, 30.0]


def compute_oscillator_fitness(result, target_pattern):
    n_points = len(result)
    per_gene_scores = {}

    for i, col in enumerate(result.colnames):
        if col == "time":
            continue

        trace = [result[t_idx][i] for t_idx in range(n_points)]

        sse = sum(
            (value - target) ** 2
            for value, target in zip(trace, target_pattern)
        )

        per_gene_scores[col] = float(sse)

    best_gene = min(per_gene_scores, key=per_gene_scores.get)
    best_score = per_gene_scores[best_gene]

    return best_score, best_gene, per_gene_scores
def score_entry(net, entry_id, target_pattern):
    try:
        result = simulate_network(net)

        best_score, best_gene, per_gene_scores = compute_oscillator_fitness(
            result,
            target_pattern
        )

        fitness = {
            "score": round(best_score, 6),
            "best_gene": best_gene,
            "per_gene_scores": {
                gene: round(score, 6)
                for gene, score in per_gene_scores.items()
            },
        }
    except Exception as e:
        fitness = {
            "score": 10000000,
            "best_gene": None,
            "per_gene_scores": {},
            "error": str(e),
        }

    return {
        "id": entry_id,
        "genes": net["genes"],
        "edges": net["edges"],
        "degradation_rates": net["degradation_rates"],
        "y0": net["y0"],
        "fitness": fitness,
    }

def compute_generation_initial(genNumber, target_pattern, mp_pool=None, logic_gates=None):
    N_NETWORKS = 100

    prepared = []

    for i in range(N_NETWORKS):
        net = generate_random_network(logic_gates)
        net = prune_unconnected_genes(net)

        prepared.append((net, i, target_pattern))

    if mp_pool is not None:
        networks_out = mp_pool.starmap(score_entry, prepared)
    else:
        networks_out = [
            score_entry(net, entry_id, pattern)
            for net, entry_id, pattern in prepared
        ]

    networks_out = sorted(
        networks_out,
        key=lambda x: x["fitness"]["score"]
        if x["fitness"]["score"] is not None
        else float("inf"),
    )

    output = {
        "generation": genNumber,
        "logic_gates": logic_gates,
        "networks": networks_out,
    }

    return output
def compute_elite_initial(networks, elite_count):
    return copy.deepcopy(networks[:elite_count])


def generate_children_from_rejected(
    pool,
    n_children,
    start_id,
    target_pattern,
    mp_pool=None,
    logic_gates=None,
):
    def fitness_of(entry):
        score = entry["fitness"]["score"]
        return score if score is not None else float("inf")

    prepared = []

    for i in range(n_children):
        a, b = random.sample(pool, 2)
        parent = a if fitness_of(a) <= fitness_of(b) else b

        genome = {
            "genes": parent["genes"],
            "edges": parent["edges"],
            "degradation_rates": parent["degradation_rates"],
            "y0": parent["y0"],
        }

        child = apply_mutation(genome, logic_gates)
        prepared.append((child, start_id + i, target_pattern))

    if mp_pool is not None:
        children = mp_pool.starmap(score_entry, prepared)
    else:
        children = [
            score_entry(net, entry_id, pattern)
            for net, entry_id, pattern in prepared
        ]

    return children

def generationCreate(
    elite_percent,
    network,
    i,
    target_pattern,
    mp_pool=None,
    logic_gates=None,
):
    elites = compute_elite_initial(network["networks"], elite_percent)

    elite_clones = []

    for idx, entry in enumerate(elites):
        clone = copy.deepcopy(entry)
        clone["id"] = idx
        elite_clones.append(clone)

    n_bred = 100 - elite_percent

    children = generate_children_from_rejected(
        network["networks"],
        n_bred,
        start_id=elite_percent,
        target_pattern=target_pattern,
        mp_pool=mp_pool,
        logic_gates=logic_gates,
    )

    next_gen = elite_clones + children

    next_gen = sorted(
        next_gen,
        key=lambda x: x["fitness"]["score"]
        if x["fitness"]["score"] is not None
        else float("inf"),
    )

    return {
        "generation": i + 1,
        "logic_gates": logic_gates,
        "networks": next_gen,
    }
