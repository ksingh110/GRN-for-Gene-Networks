import copy
import random

SINGLE_INPUT_LAWS = ["activation", "inhibition"]
TWO_INPUT_LAWS = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]

MOVE_DAMAGE = {
    "randomize_param": 2,
    "mutate_degradation": 2,
    "change_law": 3,
    "remove_edge": 3,
    "add_edge": 8,
}


def gene_slots_used(edges, gene, exclude_edge=None):
    total = 0
    for e in edges:
        if e is exclude_edge:
            continue
        if e["target"] != gene:
            continue
        total += 2 if e["regulator2"] is not None else 1
    return total


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


def mutate_law(edge, edges, genes, logic_gates=None):
    is_two_input = edge["regulator2"] is not None
    allowed_two_input_laws = TWO_INPUT_LAWS if logic_gates is None else logic_gates
    all_laws = SINGLE_INPUT_LAWS + allowed_two_input_laws

    if is_two_input:
        law_pool = [l for l in all_laws if l != edge["rate_law"]]
    else:
        others_used = gene_slots_used(edges, edge["target"], exclude_edge=edge)
        room_for_two = others_used == 0
        law_pool = [l for l in all_laws if l != edge["rate_law"]]
        if not room_for_two:
            law_pool = [l for l in law_pool if l in SINGLE_INPUT_LAWS]

    if not law_pool:
        return

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
            single_only = [l for l in SINGLE_INPUT_LAWS if l != edge["rate_law"]]
            new_law = random.choice(single_only) if single_only else edge["rate_law"]
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


def apply_mutation(network, logic_gates=None):
    net = copy.deepcopy(network)
    genes = net["genes"]
    edges = net["edges"]
    degradation_rates = net["degradation_rates"]
    y0 = net["y0"]

    possible_mutations = ["add_edge", "mutate_degradation"]
    if edges:
        possible_mutations += ["remove_edge", "randomize_param", "change_law"]
    weights = [(1.0 / MOVE_DAMAGE[m]) for m in possible_mutations]
    mutation = random.choices(possible_mutations, weights=weights, k=1)[0]

    if mutation == "mutate_degradation":
        mutate_degradation(degradation_rates, genes)

    elif mutation == "randomize_param":
        mutate_parameters(edges)

    elif mutation == "remove_edge":
        edges.remove(random.choice(edges))

    elif mutation == "add_edge":
        open_genes = [g for g in genes if gene_slots_used(edges, g) < 2]
        if open_genes:
            target = random.choice(open_genes)
            room = 2 - gene_slots_used(edges, target)
            candidates = [g for g in genes if g != target]

            allowed_two_input_laws = TWO_INPUT_LAWS if logic_gates is None else logic_gates
            law_pool = (SINGLE_INPUT_LAWS + allowed_two_input_laws) if room == 2 else SINGLE_INPUT_LAWS
            law = random.choice(law_pool)

            if law in SINGLE_INPUT_LAWS:
                regulator = random.choice(candidates)
                edges.append(mutate_add_edge(target, law, regulator))
            elif len(candidates) >= 2:
                reg_a, reg_b = random.sample(candidates, 2)
                edges.append(mutate_add_edge(target, law, reg_a, reg_b))

    elif mutation == "change_law":
        mutate_law(random.choice(edges), edges, genes, logic_gates)

    return net
