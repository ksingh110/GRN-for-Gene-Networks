import copy
import random

SINGLE_INPUT_LAWS = ["activation", "inhibition"]
TWO_INPUT_LAWS = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]

MOVE_DAMAGE = {
    "randomize_param": 2,
    "mutate_degradation": 2,
    "mutate_y0": 2,
    "change_law": 10,
    "remove_edge": 8,
    "add_edge": 6,
}


def mutate_add_edge(target, law, regulator, regulator2=None):
    if law in SINGLE_INPUT_LAWS:
        return {
            "regulator": regulator,
            "regulator2": None,
            "target": target,
            "rate_law": law,
            "Vf": round(random.random(), 2),
            "Ks": round(random.random(), 2),
            "n": round(random.uniform(0, 8.0), 2),
        }
    else:
        return {
            "regulator": regulator,
            "regulator2": regulator2,
            "target": target,
            "rate_law": law,
            "Vf": round(random.random(), 2),
            "K1": round(random.random(), 2),
            "K2": round(random.random(), 2),
            "K3": round(random.random(), 2),
            "n1": round(random.uniform(0, 8.0), 2),
            "n2": round(random.uniform(0, 8.0), 2),
        }


def mutate_law(edge, genes):
    is_two_input = edge["regulator2"] is not None
    all_laws = SINGLE_INPUT_LAWS + TWO_INPUT_LAWS
    law_pool = [l for l in all_laws if l != edge["rate_law"]]
    new_law = random.choice(law_pool)
    new_is_two_input = new_law in TWO_INPUT_LAWS

    if new_is_two_input and not is_two_input:
        candidates = [g for g in genes if g != edge["target"] and g != edge["regulator"]]
        if candidates:
            edge["regulator2"] = random.choice(candidates)
            edge["K1"] = round(random.random(), 2)
            edge["K2"] = round(random.random(), 2)
            edge["K3"] = round(random.random(), 2)
            edge["n1"] = round(random.uniform(0, 8.0), 2)
            edge["n2"] = round(random.uniform(0, 8.0), 2)
            edge.pop("Ks", None)
            edge.pop("n", None)
        else:
            new_law = random.choice([l for l in SINGLE_INPUT_LAWS if l != edge["rate_law"]])
            new_is_two_input = False

    elif not new_is_two_input and is_two_input:
        edge["regulator2"] = None
        edge["Ks"] = round(random.random(), 2)
        edge["n"] = round(random.uniform(0, 8.0), 2)
        for k in ("K1", "K2", "K3", "n1", "n2"):
            edge.pop(k, None)

    edge["rate_law"] = new_law

def mutate_parameters(edges):
    edge = random.choice(edges)

    if edge["regulator2"] is None:
        numeric_params = ["Vf", "Ks", "n"]
    else:
        numeric_params = ["Vf", "K1", "K2", "K3", "n1", "n2"]

    param = random.choice(numeric_params)

    if param.startswith("n"):
        edge[param] = round(random.uniform(0, 8.0), 2)
    else:
        edge[param] = round(random.random(), 2)


def mutate_degradation(degradation_rates, genes):
    gene = random.choice(genes)
    degradation_rates[gene] = round(random.uniform(0.05, 1.0), 2)


def mutate_y0(y0, genes):
    gene = random.choice(genes)
    y0[gene] = round(random.uniform(0, 30), 2)


def apply_mutation(network):
    net = copy.deepcopy(network)
    genes = net["genes"]
    edges = net["edges"]
    degradation_rates = net["degradation_rates"]
    y0 = net["y0"]

    possible_mutations = ["add_edge", "mutate_degradation", "mutate_y0"]
    if edges:
        possible_mutations += ["remove_edge", "randomize_param", "change_law"]
    weights = [(1.0 / MOVE_DAMAGE[m]) for m in possible_mutations]
    mutation = random.choices(possible_mutations, weights=weights, k=1)[0]

    if mutation == "mutate_degradation":
        mutate_degradation(degradation_rates, genes)

    elif mutation == "mutate_y0":
        mutate_y0(y0, genes)

    elif mutation == "randomize_param":
        mutate_parameters(edges)

    elif mutation == "remove_edge":
        edges.remove(random.choice(edges))

    elif mutation == "add_edge":
        target = random.choice(genes)
        candidates = [g for g in genes if g != target]
        p = random.random()
        if p > 0.5 and len(candidates) >= 2:
            law_pool = SINGLE_INPUT_LAWS + TWO_INPUT_LAWS
        else:
            law_pool = SINGLE_INPUT_LAWS
        law = random.choice(law_pool)
        if law in SINGLE_INPUT_LAWS:
            regulator = random.choice(candidates)
            edges.append(mutate_add_edge(target, law, regulator))
        else:
            reg_a, reg_b = random.sample(candidates, 2)
            edges.append(mutate_add_edge(target, law, reg_a, reg_b))

    elif mutation == "change_law":
        mutate_law(random.choice(edges), genes)

    return net