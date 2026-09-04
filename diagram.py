import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def diagram_network(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    net = data["best_network"]
    net_edges, net_genes = net["edges"], net["genes"]
    G = nx.MultiDiGraph()
    gene_nodes, gate_nodes = list(net_genes), []

    for gene in net_genes:
        G.add_node(gene, kind="gene", label=gene)

    for i, edge in enumerate(net_edges):
        regulator1, regulator2 = edge["regulator"], edge.get("regulator2")
        target, rate_law = edge["target"], edge["rate_law"]

        if regulator2 is None:
            law = rate_law.lower()
            color = "#D62728" if any(word in law for word in ["repress", "inhibit", "negative"]) else "#2CA02C"
            G.add_edge(regulator1, target, color=color, label=rate_law)
        else:
            gate_id = f"logic_gate_{i}"
            gate_nodes.append(gate_id)
            G.add_node(gate_id, kind="gate", label=rate_law)
            G.add_edge(regulator1, gate_id, color="#808080", label="")
            G.add_edge(regulator2, gate_id, color="#808080", label="")
            G.add_edge(gate_id, target, color="#1F77B4", label="")

    G.graph["graph"] = {
        "rankdir": "TB",
        "ranksep": "2.5",
        "nodesep": "1.5",
        "overlap": "false",
        "splines": "true",
    }

    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=4.0, iterations=1000, scale=3.0)

    plt.figure(figsize=(26, 20))
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=gene_nodes,
        node_color="#DCEEFF",
        edgecolors="#24557A",
        linewidths=2,
        node_shape="o",
        node_size=2500,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=gate_nodes,
        node_color="#D9E8F5",
        edgecolors="#1F77B4",
        linewidths=2,
        node_shape="D",
        node_size=1800,
    )

    edge_pairs = list(dict.fromkeys(
        (source, target)
        for source, target, _ in G.edges(keys=True)
    ))

    edge_radii = {}

    for source, target in edge_pairs:
        parallel_edges = list(G.get_edge_data(source, target).items())
        edge_count = len(parallel_edges)
        reciprocal = G.has_edge(target, source)

        for index, (key, attributes) in enumerate(parallel_edges):
            if edge_count > 1:
                radius = (
                    index - (edge_count - 1) / 2
                ) * 0.3
            elif reciprocal:
                radius = 0.18
            else:
                radius = 0

            edge_radii[(source, target, key)] = radius

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(source, target, key)],
                edge_color=attributes["color"],
                width=2.2,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=20,
                min_source_margin=22,
                min_target_margin=22,
                connectionstyle=f"arc3,rad={radius}",
                alpha=0.9,
            )

    gene_labels = {gene: gene for gene in gene_nodes}
    gate_labels = {
        gate: G.nodes[gate]["label"]
        for gate in gate_nodes
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=gene_labels,
        font_size=10,
        font_weight="bold",
        font_color="#17324D",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=gate_labels,
        font_size=8,
        font_weight="bold",
        font_color="#17324D",
    )

    for source, target, key, attributes in G.edges(keys=True, data=True):
        if not attributes["label"]:
            continue

        radius = edge_radii[(source, target, key)]

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (source, target, key): attributes["label"]
            },
            font_size=7,
            font_color="#333333",
            rotate=False,
            label_pos=0.5,
            connectionstyle=f"arc3,rad={radius}",
            bbox={
                "facecolor": "#FAFAFA",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.5,
            },
        )

    legend_items = [
        Patch(
            facecolor="#DCEEFF",
            edgecolor="#24557A",
            label="Gene",
        ),
        Patch(
            facecolor="#D9E8F5",
            edgecolor="#1F77B4",
            label="Logic gate",
        ),
        Line2D(
            [0],
            [0],
            color="#2CA02C",
            linewidth=2.5,
            label="Activation",
        ),
        Line2D(
            [0],
            [0],
            color="#D62728",
            linewidth=2.5,
            label="Repression",
        ),
        Line2D(
            [0],
            [0],
            color="#808080",
            linewidth=2.5,
            label="Logic-gate input",
        ),
        Line2D(
            [0],
            [0],
            color="#1F77B4",
            linewidth=2.5,
            label="Logic-gate output",
        ),
    ]

    plt.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=10,
        frameon=True,
        framealpha=0.95,
    )

    plt.axis("off")
    plt.margins(0.2)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="#FAFAFA",
    )
    plt.close()


if __name__ == "__main__":
    diagram_network(
        "results_cli/refine_success_0.json",
        "network_structure_88.png",
    )