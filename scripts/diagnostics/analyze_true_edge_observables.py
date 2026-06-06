"""Analyze observable properties of true and candidate edges.

This script is diagnostic only. It uses ground-truth edges to measure which
observable, GT-free scores would make true edges available to DGM.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np

from diagnose_pseudo_adjacency import (
    _corr_score,
    _distance_matrix,
    _knn_mask,
    _load_dataset,
    _minmax_offdiag,
    _partial_corr_score,
    _rank01,
    _smooth_similarity,
    _top_edges,
    _undirected_edges,
)


def _edge_set_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    n = mask.shape[0]
    return {(i, j) for i in range(n) for j in range(i + 1, n) if bool(mask[i, j])}


def _score_knn_edges(score: np.ndarray, k: int) -> set[tuple[int, int]]:
    n = score.shape[0]
    work = score.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(max(1, int(k)), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in idx[i]:
            if np.isfinite(work[i, j]) and work[i, j] > 0:
                edges.add(tuple(sorted((int(i), int(j)))))
    return edges


def _coverage(name: str, candidates: set[tuple[int, int]], gt: set[tuple[int, int]]) -> dict:
    tp = len(candidates & gt)
    return {
        "candidate": name,
        "candidate_edges": len(candidates),
        "gt_covered": tp,
        "gt_total": len(gt),
        "gt_recall": tp / max(1, len(gt)),
    }


def _bridge_edges(gt_edges: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(gt_edges)
    return {tuple(sorted(edge)) for edge in nx.bridges(graph)}


def _edge_feature_rows(
    xy: np.ndarray,
    gt_edges: set[tuple[int, int]],
    scores: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    n = xy.shape[0]
    dist = _distance_matrix(xy)
    bridges = _bridge_edges(gt_edges, n)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            edge = (i, j)
            row = {
                "u": i,
                "v": j,
                "is_gt": int(edge in gt_edges),
                "is_gt_bridge": int(edge in bridges),
                "distance": float(dist[i, j]),
            }
            for name, score in scores.items():
                row[name] = float(score[i, j])
            rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_score_stats(gt_edges: set[tuple[int, int]], bridges: set[tuple[int, int]], scores: dict[str, np.ndarray]) -> None:
    n = next(iter(scores.values())).shape[0]
    gt = np.zeros((n, n), dtype=bool)
    bridge = np.zeros((n, n), dtype=bool)
    for u, v in gt_edges:
        gt[u, v] = gt[v, u] = True
    for u, v in bridges:
        bridge[u, v] = bridge[v, u] = True
    offdiag = ~np.eye(n, dtype=bool)
    non = offdiag & ~gt

    print("score,gt_mean,non_mean,bridge_mean,gt_p75,non_p75")
    for name, score in scores.items():
        gt_vals = score[gt]
        non_vals = score[non]
        bridge_vals = score[bridge]
        bridge_mean = float(bridge_vals.mean()) if bridge_vals.size else 0.0
        print(
            f"{name},"
            f"{float(gt_vals.mean()):.4f},"
            f"{float(non_vals.mean()):.4f},"
            f"{bridge_mean:.4f},"
            f"{float(np.quantile(gt_vals, 0.75)):.4f},"
            f"{float(np.quantile(non_vals, 0.75)):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--geo_k", type=int, default=12)
    parser.add_argument("--signal_k", type=int, default=4)
    parser.add_argument("--top_edges", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xy, pressure, demand, edge_index = _load_dataset(args.data_dir)
    n = xy.shape[0]
    gt_edges = _undirected_edges(edge_index)
    bridges = _bridge_edges(gt_edges, n)
    edge_budget = args.top_edges if args.top_edges > 0 else len(gt_edges)

    dist = _distance_matrix(xy)
    geo = _minmax_offdiag(np.exp(-dist / (np.median(dist[dist > 0]) + 1e-12)))
    pressure_corr = _corr_score(pressure)
    pressure_smooth = _smooth_similarity(pressure)
    pressure_partial = _partial_corr_score(pressure, ridge=args.ridge)
    demand_corr = _corr_score(demand)

    partial_smooth = _minmax_offdiag(_rank01(pressure_partial) * _rank01(pressure_smooth))
    geo_partial_smooth = _minmax_offdiag(_rank01(geo) * _rank01(pressure_partial) * _rank01(pressure_smooth))
    bridge_observable = _minmax_offdiag(
        _rank01(pressure_partial)
        * _rank01(pressure_smooth)
        * (1.0 - _rank01(geo))
    )
    hybrid_bridge = np.maximum(geo_partial_smooth, 0.85 * bridge_observable)

    scores = {
        "geo": geo,
        "pressure_corr": pressure_corr,
        "pressure_smooth": pressure_smooth,
        "pressure_partial": pressure_partial,
        "demand_corr": demand_corr,
        "partial_smooth": partial_smooth,
        "geo_partial_smooth": geo_partial_smooth,
        "bridge_observable": bridge_observable,
        "hybrid_bridge": hybrid_bridge,
    }

    coverages = []
    geo_edges = _edge_set_from_mask(_knn_mask(dist, args.geo_k))
    coverages.append(_coverage(f"geo_knn_{args.geo_k}", geo_edges, gt_edges))

    union_edges = set(geo_edges)
    for name in ["pressure_partial", "pressure_smooth", "partial_smooth", "bridge_observable", "hybrid_bridge"]:
        edges = _score_knn_edges(scores[name], args.signal_k)
        coverages.append(_coverage(f"{name}_knn_{args.signal_k}", edges, gt_edges))
        coverages.append(_coverage(f"geo_knn_{args.geo_k}_union_{name}_knn_{args.signal_k}", geo_edges | edges, gt_edges))
        union_edges |= edges
    coverages.append(_coverage("union_all_candidate_sets", union_edges, gt_edges))

    print("candidate,candidate_edges,gt_covered,gt_total,gt_recall")
    for row in coverages:
        print(
            f"{row['candidate']},{row['candidate_edges']},{row['gt_covered']},"
            f"{row['gt_total']},{row['gt_recall']:.4f}"
        )

    print()
    _print_score_stats(gt_edges, bridges, scores)

    print()
    print("method,f1_like_at_gt_budget,tp,fp,fn")
    for name, score in scores.items():
        pred = set(_top_edges(score, edge_budget))
        tp = len(pred & gt_edges)
        fp = len(pred - gt_edges)
        fn = len(gt_edges - pred)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        print(f"{name},{f1:.4f},{tp},{fp},{fn}")

    with (args.out_dir / "candidate_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverages[0].keys()))
        writer.writeheader()
        writer.writerows(coverages)
    _edge_feature_rows(xy, gt_edges, scores, args.out_dir / "edge_observable_features.csv")
    print(f"[OK] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
