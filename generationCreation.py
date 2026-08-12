import json
import copy
import random
from randomNetwork import generate_random_network, prune_unconnected_genes, simulate_network
from mutateGeneration import apply_mutation

def compute_oscillator_fitness(result, target_pattern=[5.0, 30.0,5.0, 30.0,5.0, 30.0,5.0, 30.0,5.0, 30.0]):
    n_points = len(result)
    per_gene_scores = {}

    for i, col in enumerate(result.colnames):
        if col == "time":
            continue
        sse = 0.0
        for t_idx in range(n_points):
            value = result[t_idx][i]
            target = target_pattern[t_idx]
            sse += (value - target)**2
        per_gene_scores[col] = float(sse)

    best_gene = min(per_gene_scores, key=per_gene_scores.get)
    best_score = per_gene_scores[best_gene]
    return best_score, best_gene, per_gene_scores

def compute_generation_initial(genNumber):
    N_NETWORKS = 100
    OUT_PATH = "generation_" + str(genNumber) + ".json"
    networks_out = []
    print("Generation:" + str(genNumber))
    for i in range(N_NETWORKS):
        print("     Network:" + str(i))
        net = generate_random_network()
        net = prune_unconnected_genes(net)

        entry = {"id": i, "genes": net["genes"], "edges": net["edges"]}

        try:
            result = simulate_network(net)
            best_score, best_gene, per_gene_scores = compute_oscillator_fitness(result)
            entry["fitness"] = {
                "score": round(best_score, 6),
                "best_gene": best_gene,
                "per_gene_scores": {g: round(s, 6) for g, s in per_gene_scores.items()},
            }
        except Exception as e:
            entry["fitness"] = {
                "score": None,
                "best_gene": None,
                "per_gene_scores": {},
                "error": str(e),
            }
        networks_out.append(entry)
    networks_out = sorted(networks_out, key = lambda x: x["fitness"]["score"] if x["fitness"]["score"] is not None else float("inf"))
    output = {"generation": genNumber, "networks": networks_out}

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    scored = [n["fitness"]["score"] for n in networks_out if n["fitness"]["score"] is not None]
    best_overall = min(scored) if scored else None
    print(f"Saved {N_NETWORKS} networks to {OUT_PATH} "
        f"(best score: {best_overall}, {N_NETWORKS - len(scored)} failed to simulate)")
def compute_elite_initial(networks, elite_count):
    networkElite = copy.deepcopy(networks[:elite_count])
    return networkElite
def generateMutation(network):
    networkMutated = apply_mutation(network)
    return networkMutated    

if __name__ == "__main__":
    compute_generation_initial(1)
    with open("generation_1.json", "r") as file:
        data = json.load(file)
    initialNetwork = compute_elite_initial(data["networks"], 20)
    with open("mutated_elite_1.json", "w") as f:
        json.dump({"generation":1, "networks":generateMutation(initialNetwork)})
