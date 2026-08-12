import copy
import random

SINGLE_INPUT_LAWS = ["activation", "inhibition"]
TWO_INPUT_LAWS = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]

MOVE_DAMAGE = {
    "randomize_param": 1,
    "change_law": 2,
    "remove_edge": 4,
    "add_edge": 4,
}
def mutate_add_edge(target, law, regulator, regulator2=None):
    if (law in SINGLE_INPUT_LAWS):
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
def mutate_law(edge):
    if edge["regulator2"] is None:
        law_pool = [l for l in SINGLE_INPUT_LAWS if l != edge["rate_law"]]
    else:
        law_pool = [l for l in (SINGLE_INPUT_LAWS + TWO_INPUT_LAWS)if l != edge["rate_law"]]
    law = random.choice(law_pool)
    edge["rate_law"] = law
def mutate_parameters(edge):
    if edge["regulator2"] is None:
        numeric_params = ["Vf", "Ks", "n"]
    else:
        numeric_params = ["Vf", "K1", "K2", "K3", "n1", "n2"]

    param = random.choice(numeric_params)
    for param in numeric_params:
            if random.random() < 0.15:
                if param.startswith("n"):
                    edge[param] = round(random.uniform(0, 8.0), 2)
                else:
                    edge[param] = round(random.random(), 2)
def apply_mutation(network):
    net = copy.deepcopy(network)
    genes = net["genes"]
    edges = net["edges"]
    possible_mutations = ["add_edge"]
    if edges:
        possible_mutations += ["remove_edge", "randomize_param", "change_law"]
    weights = [1.0 / MOVE_DAMAGE[m] for m in possible_mutations]
    mutation = random.choices(possible_mutations, weights=weights, k=1)[0]

    if (mutation == "randomize_param"):
        edge = random.choice(edges)
        mutate_parameters(edge)

    elif(mutation == "remove_edge"):
        edges.remove(random.choice(edges))
    elif mutation == "add_edge":
        target = random.choice(genes)
        candidates = [g for g in genes if g != target]
        p = random.random()
        if p > 0.25 and len(candidates) >= 2:   # <- this len() check is the fix
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
    elif(mutation == "change_law"):
        edge = random.choice(edges)
        mutate_law(edge)
    return net
