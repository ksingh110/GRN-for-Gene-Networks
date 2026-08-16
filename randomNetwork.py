import random


def generate_random_network():
    single_input_laws = ["activation", "inhibition"]
    two_input_laws = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]
    rate_laws = single_input_laws + two_input_laws

    number_genes = random.randint(3, 15)
    number_connections = random.randint(4, 12)

    genes = [f"Gene{i}" for i in range(number_genes)]
    degradation_rates = {g: round(random.uniform(0.05, 1.0), 2) for g in genes}
    y0 = {g: round(random.uniform(0, 30), 2) for g in genes}
    edges = []

    for _ in range(number_connections):
        law = random.choice(rate_laws)
        target_gene = random.choice(genes)
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