import random
from collections import defaultdict
import tellurium as te

#simple, no logic gates
single_input_laws = ["activation", "inhibition"]
#logic gates
two_input_laws = ["AND", "OR", "NOR", "NAND", "XOR", "EQ"]
rate_laws = single_input_laws + two_input_laws

def generate_random_network():

    number_genes = random.randint(2,19)
    number_connections = random.randint(1,15)
    genes = [f"Gene{i}" for i in range(number_genes)]
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
                "Ks": round(random.random(), 2),
                "n": round(random.uniform(0, 8.0),2),
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
                "n1": round(random.uniform(0, 8.0),2),
                "n2": round(random.uniform(0, 8.0),2),
            })

    return {"genes": genes, "edges": edges}
def prune_unconnected_genes(network):
    genes = network["genes"]
    edges = network["edges"]
    connected_genes = {e["target"] for e in edges} | {e["regulator"] for e in edges} | {
        e["regulator2"] for e in edges if e["regulator2"] is not None
    }
    kept_genes = [g for g in genes if g in connected_genes]
    return {"genes": kept_genes, "edges": edges}

def build_rate_term(e):
    law = e["rate_law"]

    if law == "activation":
        S, Vf, Ks, n = e["regulator"], e["Vf"], e["Ks"], e["n"]
        return f"({Vf}*{S}^{n})/({Ks} + {S}^{n})"

    if law == "inhibition":
        S, Vf, Ks, n = e["regulator"], e["Vf"], e["Ks"], e["n"]
        return f"({Vf})/({Ks} + {S}^{n})"

    A, B = e["regulator"], e["regulator2"]
    Vf, K1, K2, K3 = e["Vf"], e["K1"], e["K2"], e["K3"]
    n1, n2 = e["n1"], e["n2"]
    A_n, B_n = f"{A}^{n1}", f"{B}^{n2}"

    if law == "AND":
        return f"{Vf}*({K1}*{K2}*{A_n}*{B_n})/(1 + {K1}*{A_n} + {K2}*{B_n} + {K1}*{K2}*{A_n}*{B_n})"

    if law == "OR":
        return f"{Vf}*({K1}*{A_n} + {K2}*{B_n})/(1 + {K1}*{A_n} + {K2}*{B_n})"

    if law == "NOR":
        return f"{Vf}*(1)/(1 + {K1}*{A_n} + {K2}*{B_n} + {K3}*{A_n}*{B_n})"

    if law == "NAND":
        return f"{Vf}*(1 + {K1}*{A_n} + {K2}*{B_n})/(1 + {K1}*{A_n} + {K2}*{B_n} + {K3}*{A_n}*{B_n})"

    if law == "XOR":
        return f"{Vf}*({K1}*{A_n} + {K2}*{B_n})/(1 + {K1}*{A_n} + {K2}*{B_n} + {K3}*{A_n}*{B_n})"

    if law == "EQ":
        return f"{Vf}*(1 + {K1}*{A_n}*{B_n})/(1 + {K1}*{A_n} + {K2}*{B_n} + {K3}*{A_n}*{B_n})"
def network_to_antimony(network):

    genes = network["genes"]
    edges = network["edges"]
    lines = [""]

    for g in genes:
        lines.append(f"  {g} = {round(random.random(),2)};")
    lines.append("")

    regs_by_target = defaultdict(list)
    for e in edges:
        regs_by_target[e["target"]].append(e)

    for g in genes:
        regs = regs_by_target[g]
        if regs:
            terms = [build_rate_term(e) for e in regs]
            synth_rate = " + ".join(terms)
        else:
            synth_rate = "0"
        lines.append(f"  J_{g}_synth: -> {g}; {synth_rate};")
        lines.append(f"  J_{g}_deg: {g} -> ; {round(random.uniform(0.05, 1.0), 2)}*{g};")

    return "\n".join(lines)

def simulate_network(network):
    print("Network:")
    antimony_str = network_to_antimony(network)
    print(antimony_str)
    r = te.loada(antimony_str)
    result = r.simulate(0, 10, 10)
    r.plot(result, xlabel="Time", ylabel="Expression")
    return r, result

if __name__ == "__main__":
    net = generate_random_network()

    print(f"random genes {net['genes']}")

    print(f"edges ({len(net['edges'])}):")
    for e in net["edges"]:
            print(f" {e['target']},({e['regulator']}, {e['regulator2']})-{e['rate_law']}")

    net = prune_unconnected_genes(net)
    print(f"after removing disconnected genes: {net['genes']}")
    r, result = simulate_network(net)