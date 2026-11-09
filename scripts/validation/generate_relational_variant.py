"""Generate the neighbor-dominant signal variant on the selected graph family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_homogeneous_synthetic import add_homogeneous_edges
from scripts.synthetic.generate_synthetic_water import (
    build_knn_edges, build_mst_edges, graph_laplacian, make_splits,
    normalized_adjacency, pairwise_dist, to_edge_index,
)


def simulate(xy, edges, timesteps, rng, tau):
    n = len(xy)
    w = normalized_adjacency(n, edges)
    lap = graph_laplacian(n, edges)
    smooth = np.linalg.inv(np.eye(n, dtype=np.float32) + tau * lap).astype(np.float32)
    time = np.arange(timesteps, dtype=np.float32)
    weekly = np.sin(2 * np.pi * time / 168.0)
    base = rng.uniform(0.7, 1.3, n).astype(np.float32)
    amp = rng.uniform(0.15, 0.45, n).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    demand = np.zeros((timesteps, n), np.float32)
    pressure = np.zeros((timesteps, n), np.float32)
    demand[0] = base + rng.normal(0, 0.03, n)
    pressure[0] = 65 - 7 * demand[0] + rng.normal(0, 0.15, n)
    source = int(np.argmin(xy[:, 0] + xy[:, 1]))
    boost = np.exp(-4 * np.linalg.norm(xy - xy[source], axis=1)).astype(np.float32)
    for step in range(1, timesteps):
        forcing = (base + amp * np.sin(2 * np.pi * step / 24 + phase)
                   + 0.12 * weekly[step] + rng.normal(0, 0.035, n))
        demand[step] = 0.15 * demand[step - 1] + 0.65 * (w @ demand[step - 1]) + 0.20 * forcing
        demand[step] = np.clip(demand[step], 0.05, None)
        load = 0.25 * demand[step] + 0.75 * (w @ demand[step])
        raw = 66 + 3 * boost - 8 * load
        target = smooth @ raw + 0.60 * (w @ pressure[step - 1]) + rng.normal(0, 0.12, n)
        pressure[step] = 0.30 * pressure[step - 1] + 0.70 * target
    return pressure, demand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graph_seed", type=int, default=211)
    parser.add_argument("--signal_seed", type=int, default=1211)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rg = np.random.default_rng(args.graph_seed)
    rs = np.random.default_rng(args.signal_seed)
    n, timesteps = 50, 720
    xy = rg.uniform(0, 1, (n, 2)).astype(np.float32)
    dist = pairwise_dist(xy)
    edges = add_homogeneous_edges(build_mst_edges(dist), dist, 30, rg, 0.25)
    initial = build_knn_edges(xy, 3)
    pressure, demand = simulate(xy, edges, timesteps, rs, 2.2)
    np.savez_compressed(args.out_dir / "data.npz", pressure=pressure, demand=demand,
                        node_xy=xy, edge_index=to_edge_index(edges, True),
                        initial_edge_index=to_edge_index(initial, True),
                        time_s=np.arange(timesteps, dtype=np.int64) * 3600)
    meta = {"name": "synthetic_homogeneous_relational", "seed": args.graph_seed,
            "graph_seed": args.graph_seed, "signal_seed": args.signal_seed,
            "node_ids": [f"N{i:03d}" for i in range(n)], "num_nodes": n,
            "num_gt_undirected_edges": len(edges), "num_initial_undirected_edges": len(initial),
            "splits": make_splits(timesteps),
            "generator": {"type": "mst_homogeneous", "signal_mode": "relational",
                          "demand_self": 0.15, "demand_neighbor": 0.65,
                          "pressure_self": 0.30, "pressure_neighbor_state": 0.60,
                          "pressure_neighbor_demand": 0.75, "pressure_smooth_tau": 2.2}}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {args.out_dir}; finite={np.isfinite(pressure).all() and np.isfinite(demand).all()}")


if __name__ == "__main__":
    main()
