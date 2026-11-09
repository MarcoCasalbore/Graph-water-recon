"""Generate a connected, sparse, non-district infrastructure-like benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic.generate_synthetic_water import (
    build_knn_edges, build_mst_edges, canonical_edge, connected_components,
    make_splits, pairwise_dist, simulate_timeseries, to_edge_index,
)


def add_homogeneous_edges(edges, dist, extra_edges, rng, chord_fraction):
    """Add degree-balanced local links and a small set of medium chords."""
    n = dist.shape[0]
    target = len(edges) + max(0, int(extra_edges))
    chord_budget = int(round(max(0, extra_edges) * float(chord_fraction)))
    local_budget = max(0, int(extra_edges) - chord_budget)
    finite = dist[np.triu_indices(n, 1)]
    q35, q60 = np.quantile(finite, [0.35, 0.60])

    def degrees():
        value = np.zeros(n, dtype=float)
        for u, v in edges:
            value[u] += 1.0
            value[v] += 1.0
        return value

    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    nn = np.argsort(work, axis=1)[:, :8]
    local_candidates = {
        canonical_edge(u, int(v)) for u in range(n) for v in nn[u]
        if canonical_edge(u, int(v)) not in edges
    }
    for _ in range(local_budget):
        if not local_candidates:
            break
        deg = degrees()
        jitter = {edge: float(rng.uniform(0.0, 1e-5)) for edge in local_candidates}
        edge = min(
            local_candidates,
            key=lambda e: dist[e] / max(q35, 1e-12)
            + 0.45 * (deg[e[0]] + deg[e[1]]) / max(1.0, 2.0 * deg.max())
            + jitter[e],
        )
        edges.add(edge)
        local_candidates.remove(edge)

    for _ in range(chord_budget):
        deg = degrees()
        adjacency = [set() for _ in range(n)]
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        def hop_distance(source, target_node):
            frontier, seen = {source}, {source}
            for hops in range(1, n + 1):
                frontier = {v for u in frontier for v in adjacency[u] if v not in seen}
                if target_node in frontier:
                    return hops
                seen.update(frontier)
                if not frontier:
                    break
            return n

        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges or not (q35 <= dist[u, v] <= q60):
                    continue
                hops = hop_distance(u, v)
                if hops >= 4:
                    score = (deg[u] + deg[v]) - 0.35 * hops + float(rng.uniform(0.0, 1e-5))
                    candidates.append((score, (u, v)))
        if not candidates:
            break
        edges.add(min(candidates, key=lambda item: item[0])[1])

    if len(edges) < target:
        candidates = sorted(
            (float(dist[u, v]), (u, v)) for u in range(n) for v in range(u + 1, n)
            if (u, v) not in edges
        )
        for _, edge in candidates:
            edges.add(edge)
            if len(edges) >= target:
                break
    return edges


def generate(args):
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_graph = np.random.default_rng(args.graph_seed)
    rng_signal = np.random.default_rng(args.signal_seed)
    xy = rng_graph.uniform(0.0, 1.0, size=(args.num_nodes, 2)).astype(np.float32)
    dist = pairwise_dist(xy)
    edges = add_homogeneous_edges(
        build_mst_edges(dist), dist, args.extra_edges, rng_graph, args.chord_fraction
    )
    initial_edges = build_knn_edges(xy, args.initial_knn)
    pressure, demand = simulate_timeseries(
        xy, edges, args.timesteps, rng_signal, pressure_mode="smooth_laplacian",
        pressure_smooth_tau=args.pressure_smooth_tau,
    )
    np.savez_compressed(
        out_dir / "data.npz", pressure=pressure, demand=demand, node_xy=xy,
        edge_index=to_edge_index(edges, True),
        initial_edge_index=to_edge_index(initial_edges, True),
        time_s=np.arange(args.timesteps, dtype=np.int64) * 3600,
    )
    meta = {
        "name": "synthetic_homogeneous_infrastructure", "seed": args.graph_seed,
        "graph_seed": args.graph_seed, "signal_seed": args.signal_seed,
        "node_ids": [f"N{i:03d}" for i in range(args.num_nodes)],
        "num_nodes": args.num_nodes, "num_gt_undirected_edges": len(edges),
        "num_initial_undirected_edges": len(initial_edges), "splits": make_splits(args.timesteps),
        "generator": {"type": "mst_homogeneous", "coordinate_distribution": "uniform_unit_square",
                      "backbone": "euclidean_mst", "extra_edges": args.extra_edges,
                      "chord_fraction": args.chord_fraction, "local_candidate_k": 8,
                      "initial_knn": args.initial_knn, "pressure_mode": "smooth_laplacian",
                      "pressure_smooth_tau": args.pressure_smooth_tau},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir}")
    print(f"[INFO] nodes={args.num_nodes}, edges={len(edges)}, components={len(connected_components(args.num_nodes, edges))}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=720)
    parser.add_argument("--extra_edges", type=int, default=30)
    parser.add_argument("--chord_fraction", type=float, default=0.25)
    parser.add_argument("--initial_knn", type=int, default=3)
    parser.add_argument("--pressure_smooth_tau", type=float, default=2.2)
    parser.add_argument("--graph_seed", type=int, required=True)
    parser.add_argument("--signal_seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
