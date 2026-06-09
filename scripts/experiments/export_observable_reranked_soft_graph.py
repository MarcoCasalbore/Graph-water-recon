"""Export a soft graph after GT-free observable re-ranking.

This is post-processing only. It combines the learned DGM soft probability
score with an observable proxy score built from coordinates and node time
series. Ground-truth edges are not used for ranking; optional comparison is
evaluation-only after the graph is exported.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_train_synthetic import _global_dynamic_score


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def edge_rows_from_pairs(edges: list[tuple[int, int]]) -> torch.Tensor:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        rows.append((v, u))
    return torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)


def component_labels(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        union(int(u), int(v))
    return [find(i) for i in range(num_nodes)]


def is_connected(num_nodes: int, edges: list[tuple[int, int]]) -> bool:
    return len(set(component_labels(num_nodes, edges))) <= 1


def connect_with_geometry(
    edges: list[tuple[int, int]],
    xy: np.ndarray,
) -> list[tuple[int, int]]:
    selected = {canonical(u, v) for u, v in edges}
    n = xy.shape[0]

    while True:
        labels = component_labels(n, sorted(selected))
        components: dict[int, list[int]] = {}
        for node, label in enumerate(labels):
            components.setdefault(label, []).append(node)
        if len(components) <= 1:
            return sorted(selected)

        comp_values = list(components.values())
        best_edge = None
        best_dist = float("inf")
        for idx, comp_a in enumerate(comp_values):
            for comp_b in comp_values[idx + 1 :]:
                a_xy = xy[comp_a]
                b_xy = xy[comp_b]
                dist = np.sqrt(((a_xy[:, None, :] - b_xy[None, :, :]) ** 2).sum(axis=-1))
                flat = int(np.argmin(dist))
                ia, ib = np.unravel_index(flat, dist.shape)
                if float(dist[ia, ib]) < best_dist:
                    best_dist = float(dist[ia, ib])
                    best_edge = canonical(int(comp_a[ia]), int(comp_b[ib]))
        if best_edge is None:
            return sorted(selected)
        selected.add(best_edge)


def prune_connected(
    edges: list[tuple[int, int]],
    combined_score: np.ndarray,
    top_edges: int,
) -> list[tuple[int, int]]:
    selected = {canonical(u, v) for u, v in edges}
    n = combined_score.shape[0]
    candidates = sorted(selected, key=lambda edge: (combined_score[edge[0], edge[1]], edge[0], edge[1]))
    for edge in candidates:
        if len(selected) <= top_edges:
            break
        selected.remove(edge)
        if not is_connected(n, sorted(selected)):
            selected.add(edge)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--top_edges", type=int, required=True)
    parser.add_argument("--proxy_mode", default="geo_global_dynamic_proxy")
    parser.add_argument("--blend_weight", type=float, default=0.5)
    parser.add_argument("--partial_ridge", type=float, default=0.05)
    parser.add_argument("--geo_tau", type=float, default=1.0)
    parser.add_argument("--score_name", default="dgm_soft_score.npy")
    parser.add_argument("--out_name", default=None)
    parser.add_argument("--ensure_connected", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    blend_weight = max(0.0, min(1.0, float(args.blend_weight)))
    run_dir = args.run_dir
    dgm_score = np.load(run_dir / args.score_name).astype(np.float64)
    data = np.load(args.npz)
    xy = data["node_xy"].astype(np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    obs_score = _global_dynamic_score(
        data_npz=str(args.npz),
        mode=args.proxy_mode,
        partial_ridge=float(args.partial_ridge),
        geo_tau=float(args.geo_tau),
    ).astype(np.float64)

    if dgm_score.shape != obs_score.shape:
        raise ValueError(f"score shape mismatch: dgm={dgm_score.shape}, observable={obs_score.shape}")

    combined = dgm_score * ((1.0 - blend_weight) + blend_weight * obs_score)
    np.fill_diagonal(combined, 0.0)
    combined = 0.5 * (combined + combined.T)

    n = combined.shape[0]
    score_items = [
        ((i, j), float(combined[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    edges = [edge for edge, _ in sorted(score_items, key=lambda item: item[1], reverse=True)[: args.top_edges]]
    if args.ensure_connected:
        edges = connect_with_geometry(edges, xy)
        if len(edges) > args.top_edges:
            edges = prune_connected(edges, combined, args.top_edges)

    mode_tag = args.proxy_mode.replace(".", "p")
    out_name = args.out_name or f"dgm_learned_edges_softprob_obsrerank_{mode_tag}_w{blend_weight:.2f}.pt".replace(".", "p")
    out_path = run_dir / out_name
    torch.save(edge_rows_from_pairs(edges), out_path)

    score_out = out_path.with_suffix(".npy")
    np.save(score_out, combined)

    print("========== OBSERVABLE RERANK EXPORT ==========")
    print(f"run_dir      : {run_dir}")
    print(f"DGM score    : {run_dir / args.score_name}")
    print(f"proxy mode   : {args.proxy_mode}")
    print(f"blend weight : {blend_weight:.3f}")
    print(f"formula      : S_export = P_soft * ((1-w) + w*Q_obs)")
    print(f"top_edges    : {args.top_edges}")
    print(f"connectivity : {'geometry repair, GT-free' if args.ensure_connected else 'none'}")
    print(f"saved edges  : {out_path}")
    print(f"saved scores : {score_out}")
    print("GT use       : none for export")

    if args.compare:
        outdir = run_dir / f"graph_comparison_{out_path.stem}"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "water" / "compare_graphs_epanet.py"),
            "--learned",
            str(out_path),
            "--npz",
            str(args.npz),
            "--meta",
            str(args.meta),
            "--outdir",
            str(outdir),
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"[OK] evaluation-only comparison written to: {outdir}")


if __name__ == "__main__":
    main()
