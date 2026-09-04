import random


def generate_random_network(logic_gates=None):
    single_input_laws = ["activation", "inhibition"]
    two_input_laws = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"] if logic_gates is None else logic_gates
    rate_laws = single_input_laws + two_input_laws

    number_genes = random.randint(3, 15)
    number_connections = random.randint(4, 12)

    genes = [f"Gene{i}" for i in range(number_genes)]
    degradation_rates = {g: round(random.uniform(0.05, 1.0), 2) for g in genes}
    y0 = {g: round(random.uniform(0, 30), 2) for g in genes}
    edges = []
    gene_max = {g: 0 for g in genes}

    for _ in range(number_connections):
        law = random.choice(rate_laws)

        if law in single_input_laws:
            valid_targets = [g for g in genes if gene_max[g] <= 1]
        else:
            valid_targets = [g for g in genes if gene_max[g] <= 0]

        if not valid_targets:
            continue

        target_gene = random.choice(valid_targets)
        non_target_gene = [g for g in genes if g != target_gene]

        if law in single_input_laws:
            regulator = random.choice(non_target_gene)
            edges.append({
                "regulator": regulator,
                "regulator2": None,
                "target": target_gene,
                "rate_law": law,
                "Vf": round(random.random(), 2),
                "Ks": round(random.uniform(0.01, 1.0), 2),
                "n": round(random.uniform(0, 8.0), 2),
            })
            gene_max[target_gene] += 1
        else:
            reg_a, reg_b = random.sample(non_target_gene, 2)
            edges.append({
                "regulator": reg_a,
                "regulator2": reg_b,
                "target": target_gene,
                "rate_law": law,
                "Vf": round(random.random(), 2),
                "K1": round(random.random(), 2),
                "K2": round(random.random(), 2),
                "K3": round(random.random(), 2),
                "n1": round(random.uniform(0, 8.0), 2),
                "n2": round(random.uniform(0, 8.0), 2),
            })
            gene_max[target_gene] += 2

    return {
        "genes": genes,
        "edges": edges,
        "degradation_rates": degradation_rates,
        "y0": y0,
    }

def prune_unconnected_genes(network):
    genes = network["genes"]
    edges = network["edges"]
    degradation_rates = network["degradation_rates"]
    y0 = network["y0"]

    connected_genes = (
        {e["target"] for e in edges}
        | {e["regulator"] for e in edges}
        | {
            e["regulator2"]
            for e in edges
            if e["regulator2"] is not None
        }
    )

    kept_genes = [
        g for g in genes
        if g in connected_genes
    ]

    return {
        "genes": kept_genes,
        "edges": edges,
        "degradation_rates": {g: degradation_rates[g] for g in kept_genes},
        "y0": {g: y0[g] for g in kept_genes},
    }
