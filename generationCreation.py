import json
import copy
import random
from randomNetwork import generate_random_network, prune_unconnected_genes, simulate_network
from mutateGeneration import apply_mutation

TARGET_PATTERN = [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0]
def compute_oscillator_fitness(result, target_pattern=TARGET_PATTERN):
    n_points = len(result)
    per_gene_scores = {}

    for i, col in enumerate(result.colnames):
        if col == "time":
            continue
        sse = 0.0
        for t_idx in range(n_points):
            value = result[t_idx][i]
            target = target_pattern[t_idx]
            sse += (value - target) ** 2
        per_gene_scores[col] = float(sse)

    best_gene = min(per_gene_scores, key=per_gene_scores.get)
    best_score = per_gene_scores[best_gene]
    return best_score, best_gene, per_gene_scores


def score_entry(net, entry_id):
    try:
        result = simulate_network(net)
        best_score, best_gene, per_gene_scores = compute_oscillator_fitness(result)
        fitness = {
            "score": round(best_score, 6),
            "best_gene": best_gene,
            "per_gene_scores": {g: round(s, 6) for g, s in per_gene_scores.items()},
        }
    except Exception as e:
        fitness = {
            "score": None,
            "best_gene": None,
            "per_gene_scores": {},
            "error": str(e),
        }
    return {"id": entry_id, "genes": net["genes"], "edges": net["edges"], "fitness": fitness}

def compute_generation_initial(genNumber):
    N_NETWORKS = 100
    OUT_PATH = "generations/generation_" + str(genNumber) + ".json"
    networks_out = []
    print("Generation:" + str(genNumber))
    for i in range(N_NETWORKS):
        print("     Network:" + str(i))
        net = generate_random_network()
        net = prune_unconnected_genes(net)

        entry = score_entry(net, i)
        networks_out.append(entry)

    networks_out = sorted(
        networks_out,
        key=lambda x: x["fitness"]["score"] if x["fitness"]["score"] is not None else float("inf"),
    )
    output = {"generation": genNumber, "networks": networks_out}

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    scored = [n["fitness"]["score"] for n in networks_out if n["fitness"]["score"] is not None]
    best_overall = min(scored) if scored else None
    print(f"Saved {N_NETWORKS} networks to {OUT_PATH} "
          f"(best score: {best_overall}, {N_NETWORKS - len(scored)} failed to simulate)")


def compute_elite_initial(networks, elite_count):
    return copy.deepcopy(networks[:elite_count])


def generate_children_from_rejected(pool, n_children, start_id):
    def fitness_of(entry):
        score = entry["fitness"]["score"]
        return score if score is not None else float("inf")

    children = []
    for i in range(n_children):
        a, b = random.sample(pool, 2)
        parent = a if fitness_of(a) <= fitness_of(b) else b
        genome = {"genes": parent["genes"], "edges": parent["edges"]}
        child = apply_mutation(genome)
        children.append(score_entry(child, start_id + i))

    return children


def generationCreate(elite_percent, i, immigrant_count=10):
    with open("generations/generation_"+str(i)+".json", "r") as file:
        network = json.load(file)
    elites = compute_elite_initial(network["networks"], elite_percent)
    elite_clones = []
    for idx, entry in enumerate(elites):
        clone = copy.deepcopy(entry)
        clone["id"] = idx
        elite_clones.append(clone)
    n_bred = 100 - elite_percent - immigrant_count
    children = generate_children_from_rejected(
        network["networks"], n_bred, start_id=elite_percent
    )

    immigrants = []
    for j in range(immigrant_count):
        net = generate_random_network()
        net = prune_unconnected_genes(net)
        immigrants.append(score_entry(net, elite_percent + n_bred + j))

    next_gen = elite_clones + children + immigrants
    next_gen = sorted(
        next_gen,
        key=lambda x: x["fitness"]["score"] if x["fitness"]["score"] is not None else float("inf"),
    )
    with open("generations/generation_"+str(i+1)+".json", "w") as f:
        json.dump({"generation": (i+1), "networks": next_gen}, f, indent=2)

    print(f"Saved generation {i+1}: {len(next_gen)} networks "
          f"({elite_percent} clones, {n_bred} bred, {immigrant_count} immigrants)")


if __name__ == "__main__":
    compute_generation_initial(1)
    with open("generations/generation_1.json") as f:
        best_fitness = json.load(f)["networks"][0]["fitness"]["score"]
    iteration = 1
    while(best_fitness>100):
        generationCreate(20,iteration)
        with open(f"generations/generation_{iteration+1}.json") as f:
          data = json.load(f)
        new_best = data["networks"][0]["fitness"]["score"]
        best_fitness = min(best_fitness, new_best)
        print(f"Generation {iteration}: best fitness = {best_fitness}, current fitness = {new_best}")
        iteration +=1