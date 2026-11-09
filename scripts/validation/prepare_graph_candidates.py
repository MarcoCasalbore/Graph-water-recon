"""Generate and screen homogeneous connected graph candidates for campaign v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
CAMPAIGN_ROOT = RESULTS_ROOT / "campaign_v2"
sys.path.insert(0, str(RESULTS_ROOT))

from scripts.plot_style import apply_style, save_figure


GRAPH_SEEDS = (101, 211, 307)
SIGNAL_SEEDS = (1101, 1211, 1307)


def canonical_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.T
        if int(u) != int(v)
    }


def generate_candidate(graph_seed: int, signal_seed: int, force: bool) -> Path:
    out = CAMPAIGN_ROOT / "datasets" / f"homogeneous_v2_g{graph_seed}_s{signal_seed}"
    if (out / "data.npz").exists() and (out / "meta.json").exists() and not force:
        return out
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/validation/generate_homogeneous_synthetic.py"),
        "--out_dir", str(out),
        "--num_nodes", "50",
        "--timesteps", "720",
        "--extra_edges", "30",
        "--chord_fraction", "0.25",
        "--pressure_smooth_tau", "2.2",
        "--initial_knn", "3",
        "--graph_seed", str(graph_seed),
        "--signal_seed", str(signal_seed),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out


def candidate_stats(directory: Path) -> tuple[dict, np.ndarray, set[tuple[int, int]]]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    data = np.load(directory / "data.npz")
    xy = data["node_xy"]
    edges = canonical_edges(data["edge_index"])
    graph = nx.Graph()
    graph.add_nodes_from(range(xy.shape[0]))
    graph.add_edges_from(edges)
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    lengths = np.asarray([np.linalg.norm(xy[u] - xy[v]) for u, v in edges], dtype=float)
    communities = list(nx.community.greedy_modularity_communities(graph))
    assortativity = nx.degree_assortativity_coefficient(graph)
    pressure = data["pressure"]
    demand = data["demand"]
    row = {
        "candidate": directory.name,
        "graph_seed": int(meta["graph_seed"]),
        "signal_seed": int(meta["signal_seed"]),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "mean_degree": degrees.mean(),
        "degree_std": degrees.std(),
        "degree_cv": degrees.std() / max(degrees.mean(), 1e-12),
        "min_degree": degrees.min(),
        "max_degree": degrees.max(),
        "components": nx.number_connected_components(graph),
        "clustering": nx.average_clustering(graph),
        "avg_shortest_path": nx.average_shortest_path_length(graph),
        "assortativity": float(assortativity) if np.isfinite(assortativity) else np.nan,
        "bridges": len(list(nx.bridges(graph))),
        "bridge_fraction": len(list(nx.bridges(graph))) / max(graph.number_of_edges(), 1),
        "greedy_modularity": nx.community.modularity(graph, communities),
        "greedy_communities": len(communities),
        "largest_community_fraction": max(map(len, communities)) / graph.number_of_nodes(),
        "edge_length_mean": lengths.mean(),
        "edge_length_std": lengths.std(),
        "edge_length_median": np.median(lengths),
        "edge_length_p95": np.quantile(lengths, 0.95),
        "edge_length_max": lengths.max(),
        "pressure_std": pressure.std(),
        "demand_std": demand.std(),
        "signals_finite": bool(np.isfinite(pressure).all() and np.isfinite(demand).all()),
    }
    return row, xy, edges


def selection_score(frame: pd.DataFrame) -> pd.Series:
    """Lower is better; uses structural/signal criteria only, never graph F1."""
    columns = ["bridge_fraction", "degree_cv", "greedy_modularity", "avg_shortest_path"]
    score = pd.Series(0.0, index=frame.index)
    for column in columns:
        values = frame[column].astype(float)
        span = values.max() - values.min()
        normalized = (values - values.min()) / (span if span > 0 else 1.0)
        score = score + normalized
    return score / len(columns)


def save_candidate_figure(rows_and_graphs: list[tuple[dict, np.ndarray, set[tuple[int, int]]]]) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, sharey=True)
    plot_rows = []
    for ax, (row, xy, edges) in zip(axes, rows_and_graphs):
        for u, v in sorted(edges):
            ax.plot(xy[[u, v], 0], xy[[u, v], 1], color="#4C78A8", lw=1.0, alpha=0.75, zorder=1)
            plot_rows.append({"candidate": row["candidate"], "u": u, "v": v,
                              "x_u": xy[u, 0], "y_u": xy[u, 1],
                              "x_v": xy[v, 0], "y_v": xy[v, 1]})
        ax.scatter(xy[:, 0], xy[:, 1], s=17, facecolor="white", edgecolor="#222222", lw=0.6, zorder=2)
        ax.set_xlabel(f"Graph seed {row['graph_seed']}")
        ax.set_aspect("equal")
        ax.grid(False)
    axes[0].set_ylabel("Normalized coordinate 2")
    for ax in axes:
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    data_dir = CAMPAIGN_ROOT / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(data_dir / "candidate_graphs.csv", index=False)
    save_figure(fig, CAMPAIGN_ROOT / "figures", "candidate_graphs")
    plt.close(fig)


def write_generator_protocol(selected: dict) -> None:
    text = f"""# Homogeneous synthetic generator protocol

## Selected scenario

The main campaign uses `{selected['candidate']}` (graph seed {selected['graph_seed']}, signal seed {selected['signal_seed']}). Selection used connectivity, bridge fraction, degree homogeneity, greedy modularity, path length, edge-length distribution and signal stability. No trained-model graph F1 entered selection.

## Construction

Nodes are sampled independently and uniformly in the unit square. Pairwise Euclidean distances define a complete weighted geometric support used only by the generator. Prim's algorithm constructs a global Euclidean minimum spanning tree, guaranteeing one connected component with N-1 edges. The generator then forms a pool of the shortest unused pairs, shuffles that local pool with the graph RNG and adds 24 local loop edges. Self-loops are never created. No district labels, block probabilities, pruning or rewiring are used. A separate geometry-only 3-nearest-neighbor graph is saved as the initial observable proxy.

## Pseudocode

```text
input N, E_extra, graph_seed, signal_seed
xy <- Uniform([0,1]^2; graph_seed), independently for each node
D_ij <- ||xy_i - xy_j||_2
E <- PrimMST(D)
C <- shortest unused unordered pairs in D
shuffle local prefix of C using graph_seed
E <- E union first E_extra pairs from the shuffled prefix
E_initial <- symmetrized geometric 3-NN(xy)
W <- row_normalized_adjacency(E)
L_G <- degree(E) - adjacency(E)
simulate demand and pressure with signal_seed using W and L_G
save xy, E, E_initial, signals and separate seeds
```

## Signal equations

For node-wise demand d_t and pressure p_t,

```text
d_t = 0.55 d_(t-1) + 0.20 W d_(t-1) + 0.25 u_t,
r_t = 66 + 3 b_source - 8 [0.65 d_t + 0.35 W d_t],
p_t = 0.72 p_(t-1) + 0.28 [(I + tau L_G)^(-1) r_t
      + 0.35 W p_(t-1) + epsilon_t].
```

The forcing u_t contains node-specific base, amplitude, phase, daily periodicity, weekly periodicity and Gaussian noise. The selected tau is 2.2. The graph affects both demand diffusion and pressure generation.

## Parameters

| Parameter | Value |
| --- | ---: |
| Nodes N | 50 |
| Timesteps | 720 hourly samples |
| Coordinate distribution | independent Uniform([0,1]^2) |
| Backbone | global Euclidean MST |
| Additional local edges | 24 |
| Total undirected edges | {selected['edges']} |
| Initial proxy | symmetrized geometric 3-NN |
| Graph seed | {selected['graph_seed']} |
| Signal seed | {selected['signal_seed']} |
| Pressure smoothing tau | 2.2 |
| Demand neighbor coefficient | 0.20 |
| Pressure neighbor-demand coefficient | 0.35 |
| Pressure neighbor-state coefficient | 0.35 |
| Demand noise standard deviation | 0.035 |
| Pressure noise standard deviation | 0.12 |
| Chronological split | 70% / 15% / 15% |
"""
    (CAMPAIGN_ROOT / "generator_protocol.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    rows_and_graphs = []
    for graph_seed, signal_seed in zip(GRAPH_SEEDS, SIGNAL_SEEDS):
        directory = generate_candidate(graph_seed, signal_seed, args.force)
        rows_and_graphs.append(candidate_stats(directory))
    frame = pd.DataFrame([item[0] for item in rows_and_graphs])
    frame["selection_score"] = selection_score(frame)
    frame["selected"] = False
    selected_idx = frame["selection_score"].idxmin()
    frame.loc[selected_idx, "selected"] = True
    selected = frame.loc[selected_idx].to_dict()

    tables = CAMPAIGN_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "candidate_graph_statistics.csv", index=False)
    (tables / "candidate_graph_statistics.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (tables / "candidate_graph_statistics.tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")
    (CAMPAIGN_ROOT / "selected_scenario.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    save_candidate_figure(rows_and_graphs)
    write_generator_protocol(selected)
    print(frame.to_string(index=False))
    print(f"[SELECTED] {selected['candidate']} without using reconstruction F1")


if __name__ == "__main__":
    main()
