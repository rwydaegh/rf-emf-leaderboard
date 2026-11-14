"""Co-authorship network of technical/dosimetric RF papers, 2016-2025 (a 2026 take on the BioEM 2016 Fig. 5)."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from rank import OUT, load

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TOP_N, LO, HI = 90, 2016, 2025


def short_name(n):
    parts = n.replace("‐", "-").split()
    return f"{parts[-1]} {parts[0][0]}" if len(parts) > 1 else n


def main():
    W, A = load()
    ws = W[W.rel & W.st.eq("tech") & W.year.between(LO, HI) & W.fb.apply(lambda f: bool(f & {"rf", "mmw"}))]
    a = A[A.work.isin(ws.work)].drop_duplicates(["work", "author"])
    papers = a.groupby("author").size().sort_values(ascending=False)
    keep = set(papers.head(TOP_N).index)
    names = a.drop_duplicates("author").set_index("author").name
    edges = Counter()
    for _, grp in a[a.author.isin(keep)].groupby("work"):
        for u, v in itertools.combinations(sorted(grp.author), 2):
            edges[(u, v)] += 1
    G = nx.Graph()
    for k in keep:
        G.add_node(k, papers=int(papers[k]), label=short_name(names[k]))
    for (u, v), w in edges.items():
        if w >= 2:
            G.add_edge(u, v, weight=w)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    comms = sorted(nx.community.greedy_modularity_communities(G, weight="weight"), key=len, reverse=True)
    color = {n: (PALETTE[i] if i < len(PALETTE) else "#8a8984") for i, c in enumerate(comms) for n in c}
    pos = nx.kamada_kawai_layout(G, weight=None)
    fig, ax = plt.subplots(figsize=(15, 12), dpi=150)
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#b9b8b1", lw=0.3 + 0.35 * d["weight"] ** 0.5, alpha=0.55, zorder=1)
    label_min = sorted((d["papers"] for _, d in G.nodes(data=True)), reverse=True)[min(44, len(G) - 1)]
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        focal = d["label"] == "Joseph W"
        ax.scatter([x], [y], s=12 * d["papers"], color=color[n], edgecolor="#0b0b0b" if focal else "white",
                   linewidth=2.2 if focal else 0.8, zorder=3)
        if d["papers"] >= label_min or focal:
            ax.annotate(d["label"], (x, y), xytext=(0, 0.9 * (12 * d["papers"]) ** 0.5 / 2 + 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5 + 0.09 * d["papers"], color="#0b0b0b",
                        weight="bold" if focal else "normal", zorder=4)
    ax.set_axis_off()
    ax.set_title(f"Co-authorship network, technical/dosimetric RF-EMF papers {LO}-{HI}\n"
                 f"{len(ws):,} papers; {G.number_of_nodes()} best-connected of the top {TOP_N} authors; circle area ~ papers; "
                 f"edges = 2 or more joint papers; colours = collaboration communities", fontsize=11, color="#52514e")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "network_rf_dosimetry_2016_2025.png", bbox_inches="tight")
    fig.savefig(OUT / "network_rf_dosimetry_2016_2025.svg", bbox_inches="tight")
    deg = {n: G.degree(n) for n in G}
    btw = nx.betweenness_centrality(G, weight=None)
    stats = sorted(({"name": names[n], "papers": int(papers[n]), "coauthors_in_top": deg[n], "betweenness": round(btw[n], 4)}
                    for n in G), key=lambda r: -r["betweenness"])
    (OUT / "network_centrality.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats[:10], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
