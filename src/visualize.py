import json
import tellurium as te
from collections import defaultdict

# oscillator pattern: TARGET_PATTERN = [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0]
TARGET_PATTERN = [5.0, 5.0, 5.0, 5.0, 5.0, 30.0, 30.0, 30.0, 30.0, 30.0]


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

def saveAntimony(inputPath, savingPath):
    with open(inputPath) as f:
        data = json.load(f)

    network = data["best_network"]
    genes = network["genes"]
    edges = network["edges"]
    y0 = network["y0"]
    degradation_rates = network["degradation_rates"]
    lines = ["model network"]
    for g in genes:
        lines.append(f"  {g} = {y0[g]};")
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
        lines.append(f"  J_{g}_synth: -> {g}; {synth_rate};\n")
        lines.append(f"  J_{g}_deg: {g} -> ; {degradation_rates[g]}*{g};\n")
    lines.append("end")
    with open(savingPath, "w") as file:
        file.writelines(lines)

def network_to_antimony(network):
    genes = network["genes"]
    edges = network["edges"]
    y0 = network["y0"]
    degradation_rates = network["degradation_rates"]
    lines = ["model network"]
    for g in genes:
        lines.append(f"  {g} = {y0[g]};")
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
        lines.append(f"  J_{g}_deg: {g} -> ; {degradation_rates[g]}*{g};")
    lines.append("end")
    return "\n".join(lines)


def visualize_network(inputPath, savingPath):
    with open(inputPath) as f:
        data = json.load(f)

    network = data["best_network"]
    antimony_str = network_to_antimony(network)
    r = te.loada(antimony_str)
    result = r.simulate(0, 50, 100)
    r.plot(result, xlabel="time", ylabel="expression", title="Gene expression trajectories")
    import matplotlib.pyplot as plt
    plt.savefig(savingPath, dpi=300, bbox_inches="tight")
    
    

if __name__ == "__main__":
    with open("results_cli/refine_success_1.json") as f:
        data = json.load(f)

    best_network = data["best_network"]
    print("Best fitness:", best_network["fitness"]["score"], "gene:", best_network["fitness"]["best_gene"])

    visualize_network(best_network)