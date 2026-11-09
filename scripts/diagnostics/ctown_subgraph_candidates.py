"""Freeze compact C-Town subnetworks before any DGM training."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import wntr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIZES = (50, 60, 75)


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical(int(u), int(v))
        for u, v in np.asarray(edge_index).T
        if int(u) != int(v)
    }


def edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    directed = [(u, v) for u, v in sorted(edges) for u, v in ((u, v), (v, u))]
    return np.asarray(directed, dtype=np.int64).T


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = xy.min(axis=0, keepdims=True)
    hi = xy.max(axis=0, keepdims=True)
    return ((xy - lo) / (hi - lo + 1e-12)).astype(np.float32)


def nearest_connected_region(
    graph: nx.Graph,
    xy: np.ndarray,
    anchor: int,
    size: int,
) -> set[int]:
    """Dijkstra prefix using geometric link length, hence always connected."""
    selected: set[int] = set()
    best = {anchor: 0.0}
    queue = [(0.0, anchor)]
    while queue and len(selected) < size:
        distance, node = heapq.heappop(queue)
        if node in selected or distance != best.get(node):
            continue
        selected.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor in selected:
                continue
            weight = float(np.linalg.norm(xy[node] - xy[neighbor]))
            candidate = distance + max(weight, 1e-9)
            if candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, int(neighbor)))
    if len(selected) != size:
        raise RuntimeError(f"Could not build size-{size} region from anchor {anchor}")
    return selected


def candidate_stats(
    graph: nx.Graph,
    xy: np.ndarray,
    nodes: set[int],
    anchor_name: str,
    size: int,
) -> dict:
    subgraph = graph.subgraph(nodes)
    internal = int(subgraph.number_of_edges())
    boundary = sum(1 for u, v in graph.edges() if (u in nodes) != (v in nodes))
    cycles = internal - len(nodes) + 1
    local_xy = xy[sorted(nodes)]
    span = np.ptp(local_xy, axis=0)
    compactness = float(np.sqrt(np.prod(np.maximum(span, 1e-12))))
    score = (
        boundary / max(1, internal)
        + 0.15 * compactness
        - 0.015 * min(cycles, 10)
    )
    return {
        "anchor": anchor_name,
        "nodes": size,
        "internal_edges": internal,
        "boundary_edges": boundary,
        "cycles": cycles,
        "average_degree": 2.0 * internal / size,
        "compactness": compactness,
        "selection_score": score,
    }


def build_knn_edges(xy: np.ndarray, k: int = 3) -> set[tuple[int, int]]:
    dist = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    idx = np.argsort(dist, axis=1)[:, : min(k, len(xy) - 1)]
    return {
        canonical(i, int(j))
        for i in range(len(xy))
        for j in idx[i]
    }


def draw_candidate(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    source_nodes: list[int],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for u, v in sorted(edges):
        ax.plot(
            [xy[u, 0], xy[v, 0]],
            [xy[u, 1], xy[v, 1]],
            color="#4b5563",
            linewidth=1.25,
            alpha=0.9,
            zorder=1,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=34,
        facecolor="#f8fafc",
        edgecolor="#1f2937",
        linewidth=0.8,
        zorder=2,
    )
    if source_nodes:
        ax.scatter(
            xy[source_nodes, 0],
            xy[source_nodes, 1],
            s=95,
            marker="s",
            facecolor="#2a9d8f",
            edgecolor="#0f172a",
            linewidth=0.9,
            zorder=3,
        )
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def freeze_candidate(
    full: np.lib.npyio.NpzFile,
    meta: dict,
    full_edges: set[tuple[int, int]],
    special_names: set[str],
    selected: set[int],
    stats: dict,
    name: str,
    output_root: Path,
) -> None:
    global_nodes = sorted(selected)
    remap = {old: new for new, old in enumerate(global_nodes)}
    local_edges = {
        canonical(remap[u], remap[v])
        for u, v in full_edges
        if u in selected and v in selected
    }
    local_xy = normalize_xy(full["node_xy"][global_nodes, :2].astype(np.float64))
    node_ids = [meta["node_ids"][index] for index in global_nodes]
    source_nodes = [
        local
        for local, node_id in enumerate(node_ids)
        if node_id in special_names
    ]
    initial = build_knn_edges(local_xy, k=3)

    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "data.npz",
        time_s=full["time_s"],
        pressure=full["pressure"][:, global_nodes],
        demand=full["demand"][:, global_nodes],
        node_xy=local_xy,
        edge_index=edge_index(local_edges),
        initial_edge_index=edge_index(initial),
        global_node_index=np.asarray(global_nodes, dtype=np.int64),
        source_nodes=np.asarray(source_nodes, dtype=np.int64),
    )
    candidate_meta = {
        "name": "ctown_fixed_induced_subgraph",
        "dataset_id": name,
        "source_dataset": "data/ctown_ts/data.npz",
        "selection_before_training": True,
        "selection_uses_model_metrics": False,
        "selection_rule": (
            "geometric-link Dijkstra region around a hydraulic source; "
            "ranked by boundary ratio, geographic compactness, and cycle count"
        ),
        "node_ids": node_ids,
        "global_node_index": global_nodes,
        "source_nodes": source_nodes,
        "source_node_ids": [node_ids[index] for index in source_nodes],
        "num_nodes": len(global_nodes),
        "num_gt_undirected_edges": len(local_edges),
        "num_initial_undirected_edges": len(initial),
        "splits": meta["splits"],
        "stats": stats,
        "allowed_for_training": ["node_xy", "pressure", "demand"],
        "posthoc_only": [
            "edge_index",
            "number of ground-truth edges",
            "global C-Town connectivity",
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(candidate_meta, indent=2),
        encoding="utf-8",
    )
    draw_candidate(
        local_xy,
        local_edges,
        source_nodes,
        (
            f"{name}: anchor={stats['anchor']}, N={len(global_nodes)}, "
            f"E={len(local_edges)}, cycles={stats['cycles']}, "
            f"boundary={stats['boundary_edges']}"
        ),
        out_dir / "ground_truth.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ctown_ts",
    )
    parser.add_argument(
        "--inp",
        type=Path,
        default=PROJECT_ROOT / "data" / "ctown.inp",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "ctown_subgraph_candidates",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    full = np.load(args.data_dir / "data.npz")
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    xy = normalize_xy(full["node_xy"][:, :2].astype(np.float64))
    edges = undirected_edges(full["edge_index"])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xy)))
    graph.add_edges_from(edges)

    wn = wntr.network.WaterNetworkModel(str(args.inp))
    source_names = set(wn.tank_name_list + wn.reservoir_name_list)
    node_lookup = {name: index for index, name in enumerate(meta["node_ids"])}
    anchors = [
        (name, node_lookup[name])
        for name in sorted(source_names)
        if name in node_lookup
    ]

    all_rows = []
    regions = {}
    for size in SIZES:
        for anchor_name, anchor in anchors:
            nodes = nearest_connected_region(graph, xy, anchor, size)
            stats = candidate_stats(graph, xy, nodes, anchor_name, size)
            key = (size, anchor_name)
            regions[key] = nodes
            all_rows.append(stats)

    # One frozen candidate per size. This choice is made before proxy/DGM tests.
    selected_rows = []
    used_anchors: set[str] = set()
    for size in SIZES:
        eligible = [
            row
            for row in all_rows
            if row["nodes"] == size and row["cycles"] >= 2
        ]
        eligible.sort(key=lambda row: (row["selection_score"], row["anchor"]))
        chosen = next(
            (row for row in eligible if row["anchor"] not in used_anchors),
            eligible[0],
        )
        selected_rows.append(chosen)
        used_anchors.add(str(chosen["anchor"]))

    labels = ("ctown_cut_a_n50", "ctown_cut_b_n60", "ctown_cut_c_n75")
    for label, row in zip(labels, selected_rows):
        freeze_candidate(
            full,
            meta,
            edges,
            source_names,
            regions[(int(row["nodes"]), str(row["anchor"]))],
            row,
            label,
            args.output_root,
        )

    manifest = {
        "full_network": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "selection_protocol": (
            "Candidates frozen before proxy or DGM evaluation. "
            "No reconstruction metric enters candidate selection."
        ),
        "all_source_centered_candidates": all_rows,
        "selected": dict(zip(labels, selected_rows)),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["selected"], indent=2))


if __name__ == "__main__":
    main()
