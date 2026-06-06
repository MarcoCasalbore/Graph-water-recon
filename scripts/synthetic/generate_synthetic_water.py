"""Generate controlled water-like graph reconstruction benchmarks.

The saved files follow the same interface used by the EPANET loader:

data.npz:
  pressure, demand, node_xy, edge_index, initial_edge_index

meta.json:
  node_ids and temporal train/val/test splits

`edge_index` is the hidden ground-truth topology used only for evaluation.
`initial_edge_index` is a geometry-only proxy graph used by the model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pairwise_dist(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def build_mst_edges(dist: np.ndarray) -> set[tuple[int, int]]:
    n = dist.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: set[tuple[int, int]] = set()

    for _ in range(n - 1):
        best = None
        best_dist = np.inf
        for u in np.where(selected)[0]:
            candidates = np.where(~selected)[0]
            if candidates.size == 0:
                break
            v_idx = candidates[np.argmin(dist[u, candidates])]
            if dist[u, v_idx] < best_dist:
                best = (int(u), int(v_idx))
                best_dist = float(dist[u, v_idx])

        if best is None:
            raise RuntimeError("Could not build connected MST.")
        u, v = best
        edges.add(canonical_edge(u, v))
        selected[v] = True

    return edges


def add_local_edges(
    edges: set[tuple[int, int]],
    dist: np.ndarray,
    extra_edges: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    n = dist.shape[0]
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            e = (u, v)
            if e not in edges:
                candidates.append((float(dist[u, v]), e))

    candidates.sort(key=lambda item: item[0])
    local_pool = candidates[: max(extra_edges * 8, extra_edges)]
    rng.shuffle(local_pool)

    for _, edge in local_pool:
        if len(edges) >= n - 1 + extra_edges:
            break
        edges.add(edge)

    return edges


def build_knn_edges(xy: np.ndarray, k: int) -> set[tuple[int, int]]:
    dist = pairwise_dist(xy)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    edges = set()
    for u in range(xy.shape[0]):
        for v in nn[u]:
            edges.add(canonical_edge(int(u), int(v)))
    return edges


def build_erdos_renyi_edges(
    n: int,
    p: float,
    rng: np.random.Generator,
    ensure_connected: bool = True,
    dist: np.ndarray | None = None,
    geometric_sigma: float = 0.25,
) -> set[tuple[int, int]]:
    edges = set()
    probs = None
    if dist is not None:
        probs = calibrated_distance_probabilities(dist, base_prob=p, sigma=geometric_sigma)

    for u in range(n):
        for v in range(u + 1, n):
            edge_prob = float(probs[u, v]) if probs is not None else p
            if rng.random() < edge_prob:
                edges.add((u, v))

    if ensure_connected:
        edges = connect_components_by_distance(n, edges, dist, rng)
    return edges


def build_sbm_edges(
    n: int,
    num_blocks: int,
    p_intra: float,
    p_inter: float,
    rng: np.random.Generator,
    ensure_connected: bool = True,
    dist: np.ndarray | None = None,
    geometric_sigma: float = 0.25,
    labels: np.ndarray | None = None,
) -> tuple[set[tuple[int, int]], np.ndarray]:
    if labels is None:
        labels = np.arange(n) % int(num_blocks)
        rng.shuffle(labels)
    labels = labels.astype(np.int64)

    edges = set()
    decay = None
    same_block = labels[:, None] == labels[None, :]
    if dist is not None:
        decay = np.exp(-dist / max(float(geometric_sigma), 1e-6)).astype(np.float32)
        np.fill_diagonal(decay, 0.0)

    for u in range(n):
        for v in range(u + 1, n):
            p = p_intra if labels[u] == labels[v] else p_inter
            if decay is not None:
                mask = same_block if labels[u] == labels[v] else ~same_block
                mask = mask & ~np.eye(n, dtype=bool)
                mean_decay = float(decay[mask].mean()) if np.any(mask) else 1.0
                p = p * float(decay[u, v]) / max(mean_decay, 1e-6)
                p = min(max(p, 0.0), 1.0)
            if rng.random() < p:
                edges.add((u, v))

    if ensure_connected:
        edges = connect_components_by_distance(n, edges, dist, rng)
    return edges, labels.astype(np.int64)


def calibrated_distance_probabilities(
    dist: np.ndarray,
    base_prob: float,
    sigma: float,
) -> np.ndarray:
    """Distance-biased probabilities calibrated to preserve mean density."""
    n = dist.shape[0]
    mask = ~np.eye(n, dtype=bool)
    decay = np.exp(-dist / max(float(sigma), 1e-6)).astype(np.float32)
    mean_decay = float(decay[mask].mean())
    probs = float(base_prob) * decay / max(mean_decay, 1e-6)
    probs = np.clip(probs, 0.0, 1.0)
    np.fill_diagonal(probs, 0.0)
    return probs


def connected_components(n: int, edges: set[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    seen = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = set()
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(comp)
    return components


def connect_components_randomly(
    n: int,
    edges: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    components = connected_components(n, edges)
    if len(components) <= 1:
        return edges

    ordered = sorted(components, key=lambda c: min(c))
    for comp_a, comp_b in zip(ordered[:-1], ordered[1:]):
        u = int(rng.choice(list(comp_a)))
        v = int(rng.choice(list(comp_b)))
        edges.add(canonical_edge(u, v))
    return edges


def connect_components_by_distance(
    n: int,
    edges: set[tuple[int, int]],
    dist: np.ndarray | None,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    if dist is None:
        return connect_components_randomly(n, edges, rng)

    while True:
        components = connected_components(n, edges)
        if len(components) <= 1:
            return edges

        best = None
        best_dist = np.inf
        for i, comp_a in enumerate(components):
            for comp_b in components[i + 1:]:
                for u in comp_a:
                    for v in comp_b:
                        if dist[u, v] < best_dist:
                            best = (int(u), int(v))
                            best_dist = float(dist[u, v])

        if best is None:
            return connect_components_randomly(n, edges, rng)
        edges.add(canonical_edge(*best))


def sample_block_positions(
    n: int,
    num_blocks: int,
    rng: np.random.Generator,
    spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(n) % int(num_blocks)
    rng.shuffle(labels)

    centers = rng.uniform(0.15, 0.85, size=(num_blocks, 2)).astype(np.float32)
    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        xy[i] = centers[labels[i]] + rng.normal(0.0, spread, size=2)
    xy = np.clip(xy, 0.0, 1.0).astype(np.float32)
    return xy, labels.astype(np.int64)


def sample_water_branch_positions(
    n: int,
    num_districts: int,
    rng: np.random.Generator,
    spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sensor positions in compact service districts.

    This gives the synthetic benchmark a water-network flavour: most customers
    are geographically close to a local district, while districts themselves are
    separated enough to require a few bridge/backbone pipes.
    """
    labels = np.arange(n) % int(num_districts)
    rng.shuffle(labels)

    angles = np.linspace(0.0, 2.0 * np.pi, int(num_districts), endpoint=False)
    angles += rng.normal(0.0, 0.18, size=int(num_districts))
    radii = rng.uniform(0.22, 0.36, size=int(num_districts))
    centers = np.column_stack(
        [
            0.5 + radii * np.cos(angles),
            0.5 + radii * np.sin(angles),
        ]
    ).astype(np.float32)

    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        xy[i] = centers[labels[i]] + rng.normal(0.0, spread, size=2)
    xy = np.clip(xy, 0.03, 0.97).astype(np.float32)
    return xy, labels.astype(np.int64)


def build_water_branch_edges(
    xy: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    local_k: int,
    loops_per_district: int,
    bridge_edges: int,
) -> set[tuple[int, int]]:
    """Build a sparse water-like topology from geometry and districts.

    The graph is mostly a tree/backbone: each district has a local MST, district
    representatives are connected by an MST, then a small number of local loops
    and long bridges are added. This keeps the benchmark non-trivial while
    preserving the physical bias that most pipes are short and only a few longer
    pipes are justified.
    """
    n = xy.shape[0]
    dist = pairwise_dist(xy)
    edges: set[tuple[int, int]] = set()
    labels = labels.astype(np.int64)
    unique_labels = np.unique(labels)

    representatives: list[int] = []
    for label in unique_labels:
        nodes = np.where(labels == label)[0].astype(int)
        if nodes.size == 0:
            continue

        local_xy = xy[nodes]
        local_dist = pairwise_dist(local_xy)
        for u_local, v_local in build_mst_edges(local_dist):
            edges.add(canonical_edge(int(nodes[u_local]), int(nodes[v_local])))

        centroid = local_xy.mean(axis=0, keepdims=True)
        rep = int(nodes[np.argmin(np.linalg.norm(local_xy - centroid, axis=1))])
        representatives.append(rep)

        local_candidates = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                edge = canonical_edge(int(u), int(v))
                if edge in edges:
                    continue
                local_candidates.append((float(dist[u, v]), edge))
        local_candidates.sort(key=lambda item: item[0])

        loop_budget = max(0, int(loops_per_district))
        pool = local_candidates[: max(loop_budget * max(2, local_k), loop_budget)]
        rng.shuffle(pool)
        for _, edge in pool[:loop_budget]:
            edges.add(edge)

    if len(representatives) > 1:
        rep_dist = dist[np.ix_(representatives, representatives)]
        for u_local, v_local in build_mst_edges(rep_dist):
            edges.add(canonical_edge(representatives[u_local], representatives[v_local]))

    bridge_candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            edge = (i, j)
            if edge in edges:
                continue
            bridge_candidates.append((float(dist[i, j]), edge))
    bridge_candidates.sort(key=lambda item: item[0])

    pool = bridge_candidates[: max(int(bridge_edges) * 8, int(bridge_edges))]
    rng.shuffle(pool)
    for _, edge in pool[: max(0, int(bridge_edges))]:
        edges.add(edge)

    return connect_components_by_distance(n, edges, dist, rng)


def to_edge_index(edges: set[tuple[int, int]], bidirectional: bool = True) -> np.ndarray:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        if bidirectional:
            rows.append((v, u))
    return np.asarray(rows, dtype=np.int64).T


def normalized_adjacency(n: int, edges: set[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    return A / np.maximum(deg, 1.0)


def graph_laplacian(n: int, edges: set[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    deg = A.sum(axis=1)
    return np.diag(deg).astype(np.float32) - A


def simulate_timeseries(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    timesteps: int,
    rng: np.random.Generator,
    pressure_mode: str = "smooth_laplacian",
    pressure_smooth_tau: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    n = xy.shape[0]
    W = normalized_adjacency(n, edges)
    L = graph_laplacian(n, edges)
    smooth_operator = np.linalg.inv(
        np.eye(n, dtype=np.float32) + float(pressure_smooth_tau) * L
    ).astype(np.float32)

    t = np.arange(timesteps, dtype=np.float32)
    daily = np.sin(2.0 * np.pi * t / 24.0)
    slow = np.sin(2.0 * np.pi * t / (24.0 * 7.0))

    node_base = rng.uniform(0.7, 1.3, size=n).astype(np.float32)
    node_amp = rng.uniform(0.15, 0.45, size=n).astype(np.float32)
    node_phase = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)

    demand = np.zeros((timesteps, n), dtype=np.float32)
    pressure = np.zeros((timesteps, n), dtype=np.float32)

    demand[0] = node_base + rng.normal(0.0, 0.03, size=n)
    pressure[0] = 65.0 - 7.0 * demand[0] + rng.normal(0.0, 0.15, size=n)

    source_idx = int(np.argmin(xy[:, 0] + xy[:, 1]))
    source_boost = np.exp(-4.0 * np.linalg.norm(xy - xy[source_idx], axis=1)).astype(np.float32)

    for step in range(1, timesteps):
        local_daily = np.sin(2.0 * np.pi * step / 24.0 + node_phase)
        exogenous_demand = (
            node_base
            + node_amp * local_daily
            + 0.12 * slow[step]
            + rng.normal(0.0, 0.035, size=n)
        )
        demand[step] = (
            0.55 * demand[step - 1]
            + 0.20 * (W @ demand[step - 1])
            + 0.25 * exogenous_demand
        )
        demand[step] = np.clip(demand[step], 0.05, None)

        hydraulic_load = 0.65 * demand[step] + 0.35 * (W @ demand[step])
        raw_pressure = 66.0 + 3.0 * source_boost - 8.0 * hydraulic_load
        if pressure_mode == "smooth_laplacian":
            target_pressure = smooth_operator @ raw_pressure
            target_pressure += 0.35 * (W @ pressure[step - 1])
        else:
            target_pressure = raw_pressure + 0.8 * (W @ pressure[step - 1])
        target_pressure += rng.normal(0.0, 0.12, size=n)
        pressure[step] = 0.72 * pressure[step - 1] + 0.28 * target_pressure

    return pressure.astype(np.float32), demand.astype(np.float32)


def make_splits(timesteps: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    train_end = int(timesteps * train_ratio)
    val_end = int(timesteps * (train_ratio + val_ratio))
    return {
        "train": {"start": 0, "end": train_end},
        "val": {"start": train_end, "end": val_end},
        "test": {"start": val_end, "end": timesteps},
    }


def generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_block_labels = None
    if args.graph_type == "water_branch":
        node_xy, preset_block_labels = sample_water_branch_positions(
            n=args.num_nodes,
            num_districts=args.water_districts,
            rng=rng,
            spread=args.water_district_spread,
        )
    elif args.graph_type == "sbm" and args.geometric_edges and args.sbm_geometric_blocks:
        node_xy, preset_block_labels = sample_block_positions(
            n=args.num_nodes,
            num_blocks=args.sbm_blocks,
            rng=rng,
            spread=args.sbm_block_spread,
        )
    else:
        node_xy = rng.uniform(0.0, 1.0, size=(args.num_nodes, 2)).astype(np.float32)
    dist = pairwise_dist(node_xy)

    block_labels = None
    if args.graph_type == "mst_local":
        gt_edges = build_mst_edges(dist)
        gt_edges = add_local_edges(gt_edges, dist, args.extra_edges, rng)
    elif args.graph_type == "water_branch":
        block_labels = preset_block_labels
        gt_edges = build_water_branch_edges(
            xy=node_xy,
            labels=block_labels,
            rng=rng,
            local_k=args.water_local_k,
            loops_per_district=args.water_loops_per_district,
            bridge_edges=args.water_bridge_edges,
        )
    elif args.graph_type == "er":
        gt_edges = build_erdos_renyi_edges(
            n=args.num_nodes,
            p=args.er_p,
            rng=rng,
            ensure_connected=args.ensure_connected,
            dist=dist if args.geometric_edges else None,
            geometric_sigma=args.geometric_sigma,
        )
    elif args.graph_type == "sbm":
        gt_edges, block_labels = build_sbm_edges(
            n=args.num_nodes,
            num_blocks=args.sbm_blocks,
            p_intra=args.sbm_p_intra,
            p_inter=args.sbm_p_inter,
            rng=rng,
            ensure_connected=args.ensure_connected,
            dist=dist if args.geometric_edges else None,
            geometric_sigma=args.geometric_sigma,
            labels=preset_block_labels,
        )
    else:
        raise ValueError(f"Unknown graph_type={args.graph_type}")

    initial_edges = build_knn_edges(node_xy, args.initial_knn)

    pressure, demand = simulate_timeseries(
        xy=node_xy,
        edges=gt_edges,
        timesteps=args.timesteps,
        rng=rng,
        pressure_mode=args.pressure_mode,
        pressure_smooth_tau=args.pressure_smooth_tau,
    )

    save_kwargs = {}
    if block_labels is not None:
        save_kwargs["block_labels"] = block_labels

    np.savez_compressed(
        out_dir / "data.npz",
        pressure=pressure,
        demand=demand,
        node_xy=node_xy,
        edge_index=to_edge_index(gt_edges, bidirectional=True),
        initial_edge_index=to_edge_index(initial_edges, bidirectional=True),
        time_s=np.arange(args.timesteps, dtype=np.int64) * 3600,
        **save_kwargs,
    )

    meta = {
        "name": "synthetic_water",
        "seed": args.seed,
        "node_ids": [f"N{i:03d}" for i in range(args.num_nodes)],
        "num_nodes": args.num_nodes,
        "num_gt_undirected_edges": len(gt_edges),
        "num_initial_undirected_edges": len(initial_edges),
        "splits": make_splits(args.timesteps),
        "generator": {
            "type": args.graph_type,
            "extra_edges": args.extra_edges,
            "er_p": args.er_p,
            "sbm_blocks": args.sbm_blocks,
            "sbm_p_intra": args.sbm_p_intra,
            "sbm_p_inter": args.sbm_p_inter,
            "initial_knn": args.initial_knn,
            "ensure_connected": args.ensure_connected,
            "geometric_edges": args.geometric_edges,
            "geometric_sigma": args.geometric_sigma,
            "sbm_geometric_blocks": args.sbm_geometric_blocks,
            "sbm_block_spread": args.sbm_block_spread,
            "water_districts": args.water_districts,
            "water_district_spread": args.water_district_spread,
            "water_local_k": args.water_local_k,
            "water_loops_per_district": args.water_loops_per_district,
            "water_bridge_edges": args.water_bridge_edges,
            "pressure_mode": args.pressure_mode,
            "pressure_smooth_tau": args.pressure_smooth_tau,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote synthetic benchmark to: {out_dir}")
    print(f"[INFO] Nodes: {args.num_nodes}")
    print(f"[INFO] GT undirected edges: {len(gt_edges)}")
    print(f"[INFO] GT connected components: {len(connected_components(args.num_nodes, gt_edges))}")
    print(f"[INFO] Initial proxy undirected edges: {len(initial_edges)}")
    print(f"[INFO] Timesteps: {args.timesteps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic water-like benchmark.")
    parser.add_argument("--out_dir", default="data/synthetic/simple_water_n50")
    parser.add_argument("--graph_type", choices=["mst_local", "er", "sbm", "water_branch"], default="mst_local")
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=720)
    parser.add_argument("--extra_edges", type=int, default=12)
    parser.add_argument("--er_p", type=float, default=0.06)
    parser.add_argument("--sbm_blocks", type=int, default=4)
    parser.add_argument("--sbm_p_intra", type=float, default=0.18)
    parser.add_argument("--sbm_p_inter", type=float, default=0.025)
    parser.add_argument("--ensure_connected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geometric_edges", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometric_sigma", type=float, default=0.25)
    parser.add_argument("--sbm_geometric_blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sbm_block_spread", type=float, default=0.12)
    parser.add_argument("--water_districts", type=int, default=5)
    parser.add_argument("--water_district_spread", type=float, default=0.055)
    parser.add_argument("--water_local_k", type=int, default=3)
    parser.add_argument("--water_loops_per_district", type=int, default=2)
    parser.add_argument("--water_bridge_edges", type=int, default=4)
    parser.add_argument("--pressure_mode", choices=["smooth_laplacian", "autoregressive_neighbor"], default="smooth_laplacian")
    parser.add_argument("--pressure_smooth_tau", type=float, default=1.8)
    parser.add_argument("--initial_knn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
