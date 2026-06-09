"""Diagnostic observable edge proxies on the full C-Town dataset.

Ground-truth edges are used only to evaluate GT-free scores. None of these
metrics should be interpreted as training supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np


EPS = 1e-12


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def minmax_offdiag(score: np.ndarray) -> np.ndarray:
    out = np.asarray(score, dtype=np.float64).copy()
    n = out.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = out[mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        out[:] = 0.0
        return out
    lo = vals[finite].min()
    hi = vals[finite].max()
    if hi <= lo + EPS:
        out[:] = 0.0
    else:
        out = (out - lo) / (hi - lo + EPS)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + EPS)


def distance_matrix(xy: np.ndarray) -> np.ndarray:
    delta = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def gt_edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    edges = set()
    for u, v in edge_index.T.tolist():
        u = int(u)
        v = int(v)
        if u != v:
            edges.add(canonical(u, v))
    return edges


def top_edges(score: np.ndarray, budget: int) -> set[tuple[int, int]]:
    n = score.shape[0]
    pairs = [(float(score[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)
    return {(i, j) for _, i, j in pairs[:budget]}


def geometry_mst_edges(dist: np.ndarray) -> set[tuple[int, int]]:
    graph = nx.Graph()
    n = dist.shape[0]
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist[i, j]))
    return {canonical(int(u), int(v)) for u, v in nx.minimum_spanning_tree(graph).edges()}


def mst_plus_top(score: np.ndarray, dist: np.ndarray, budget: int) -> set[tuple[int, int]]:
    edges = set(geometry_mst_edges(dist))
    if len(edges) >= budget:
        return set(sorted(edges)[:budget])
    for edge in top_edges(score, budget * 2):
        edges.add(edge)
        if len(edges) >= budget:
            break
    return edges


def knn_edges(score: np.ndarray, k: int) -> set[tuple[int, int]]:
    n = score.shape[0]
    work = score.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(max(1, int(k)), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in idx[i]:
            if np.isfinite(work[i, j]):
                edges.add(canonical(i, int(j)))
    return edges


def pressure_smooth_score(pressure: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    n = pressure.shape[1]
    diff = np.empty((n, n), dtype=np.float64)
    p = pressure.astype(np.float64, copy=False)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff[start:stop] = np.mean(np.abs(p[:, start:stop, None] - p[:, None, :]), axis=0)
    positive = diff[diff > 0]
    scale = np.median(positive) if positive.size else 1.0
    score = np.exp(-diff / (scale + EPS))
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def positive_corr_score(x: np.ndarray) -> np.ndarray:
    z = standardize(x.astype(np.float64, copy=False))
    corr = (z.T @ z) / max(1, z.shape[0] - 1)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    return minmax_offdiag(corr)


def partial_corr_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = standardize(x.astype(np.float64, copy=False))
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    precision = np.linalg.pinv(cov + float(ridge) * np.eye(cov.shape[0]))
    d = np.sqrt(np.maximum(np.diag(precision), EPS))
    pcorr = -precision / (d[:, None] * d[None, :] + EPS)
    pcorr = np.maximum(pcorr, 0.0)
    np.fill_diagonal(pcorr, 0.0)
    return minmax_offdiag(pcorr)


def global_var_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = standardize(x.astype(np.float64, copy=False))
    source = z[:-1]
    target = z[1:]
    n = source.shape[1]
    coef = np.linalg.solve(source.T @ source + float(ridge) * np.eye(n), source.T @ target)
    score = np.maximum(np.abs(coef), np.abs(coef).T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def global_cross_var_score(source_x: np.ndarray, target_x: np.ndarray, ridge: float) -> np.ndarray:
    source = standardize(source_x.astype(np.float64, copy=False))
    target = standardize(target_x.astype(np.float64, copy=False))
    n = target.shape[1]
    design = np.concatenate([target[:-1], source[:-1]], axis=1)
    y = target[1:]
    coef = np.linalg.solve(design.T @ design + float(ridge) * np.eye(2 * n), design.T @ y)
    source_to_target = np.abs(coef[n:, :])
    score = np.maximum(source_to_target, source_to_target.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def metrics(pred: set[tuple[int, int]], gt: set[tuple[int, int]], n: int) -> dict[str, float | int]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(EPS, precision + recall)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(pred)
    return {
        "pred_edges": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "components": nx.number_connected_components(graph),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/ctown_ts"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ctown_proxy_diagnostics"))
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--geo_k", type=int, default=12)
    parser.add_argument("--q_k", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data_dir / "data.npz")
    xy = data["node_xy"].astype(np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    pressure = data["pressure"].astype(np.float64)
    demand = data["demand"].astype(np.float64)
    gt = gt_edge_set(data["edge_index"].astype(np.int64))
    n = xy.shape[0]
    budget = len(gt)

    dist = distance_matrix(xy)
    geo = minmax_offdiag(np.exp(-dist / (np.median(dist[dist > 0]) + EPS)))
    smooth = pressure_smooth_score(pressure)
    pressure_corr = positive_corr_score(pressure)
    demand_corr = positive_corr_score(demand)
    pressure_partial = partial_corr_score(pressure, ridge=args.ridge)
    demand_var = global_var_score(demand, ridge=args.ridge)
    demand_to_pressure = global_cross_var_score(demand, pressure, ridge=args.ridge)
    dynamic = minmax_offdiag(smooth * np.maximum(demand_var, demand_to_pressure))
    geo_dynamic = minmax_offdiag(geo * dynamic)

    scores = {
        "geo": geo,
        "pressure_smooth": smooth,
        "pressure_corr": pressure_corr,
        "pressure_partial": pressure_partial,
        "demand_corr": demand_corr,
        "demand_var": demand_var,
        "demand_to_pressure_var": demand_to_pressure,
        "global_dynamic_proxy": dynamic,
        "geo_global_dynamic_proxy": geo_dynamic,
    }

    rows = []
    for name, score in scores.items():
        rows.append({"method": f"{name}_top_budget", **metrics(top_edges(score, budget), gt, n)})
        rows.append({"method": f"{name}_mst_plus_top", **metrics(mst_plus_top(score, dist, budget), gt, n)})
        rows.append({"method": f"{name}_row_knn_{args.q_k}", **metrics(knn_edges(score, args.q_k), gt, n)})

    geo_candidates = knn_edges(geo, args.geo_k)
    rows.append({"method": f"geo_knn_{args.geo_k}", **metrics(geo_candidates, gt, n)})
    rows.append(
        {
            "method": f"geo_knn_{args.geo_k}_union_geo_dynamic_knn_{args.q_k}",
            **metrics(geo_candidates | knn_edges(geo_dynamic, args.q_k), gt, n),
        }
    )

    fieldnames = list(rows[0].keys())
    with (args.out_dir / "proxy_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_nodes": n,
        "gt_undirected_edges": budget,
        "best_by_f1": sorted(rows, key=lambda row: float(row["f1"]), reverse=True)[:10],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("method,f1,precision,recall,tp,fp,fn,pred_edges,components")
    for row in summary["best_by_f1"]:
        print(
            f"{row['method']},{row['f1']:.4f},{row['precision']:.4f},{row['recall']:.4f},"
            f"{row['tp']},{row['fp']},{row['fn']},{row['pred_edges']},{row['components']}"
        )
    print(f"[OK] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
