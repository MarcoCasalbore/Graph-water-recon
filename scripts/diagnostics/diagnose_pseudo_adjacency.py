"""Diagnose GT-free pseudo-adjacency scores against synthetic ground truth.

The scores computed here use only information allowed at training time:
node coordinates and node time series. Ground-truth edges are used only at the
end for evaluation, plotting and ranking the surrogate objectives.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


EPS = 1e-12


@dataclass(frozen=True)
class MethodResult:
    method: str
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    learned_edges: int
    gt_edges: int
    components: int
    mean_edge_length: float
    median_edge_length: float


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + EPS)


def _minmax_offdiag(score: np.ndarray) -> np.ndarray:
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
    if hi <= lo:
        out[:] = 0.0
    else:
        out = (out - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _rank01(score: np.ndarray) -> np.ndarray:
    n = score.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(score[i, j]), i, j))
    pairs.sort()
    out = np.zeros((n, n), dtype=np.float64)
    denom = max(1, len(pairs) - 1)
    for rank, (_, i, j) in enumerate(pairs):
        value = rank / denom
        out[i, j] = value
        out[j, i] = value
    return out


def _load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_dir / "data.npz")
    xy = np.asarray(data["node_xy"], dtype=np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]

    pressure = np.asarray(data["pressure"], dtype=np.float64)
    demand = np.asarray(data["demand"], dtype=np.float64)
    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    return xy, pressure, demand, edge_index


def _undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    for u, v in edge_index.T.tolist():
        if int(u) == int(v):
            continue
        a, b = sorted((int(u), int(v)))
        edges.add((a, b))
    return edges


def _distance_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _knn_mask(dist: np.ndarray, k: int) -> np.ndarray:
    n = dist.shape[0]
    k = min(max(1, int(k)), n - 1)
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True
    mask = np.logical_or(mask, mask.T)
    np.fill_diagonal(mask, False)
    return mask


def _corr_score(x: np.ndarray) -> np.ndarray:
    z = _standardize(x)
    score = (z.T @ z) / max(1, z.shape[0] - 1)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _lag_corr_score(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = _standardize(x)
    n = x.shape[1]
    best = np.zeros((n, n), dtype=np.float64)
    for lag in range(1, max(1, max_lag) + 1):
        a = x[lag:]
        b = x[:-lag]
        corr = (a.T @ b) / max(1, a.shape[0] - 1)
        best = np.maximum(best, np.maximum(corr, corr.T))
    best = np.maximum(best, 0.0)
    np.fill_diagonal(best, 0.0)
    return _minmax_offdiag(best)


def _smooth_similarity(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    chunk_size = 32
    diff = np.empty((n, n), dtype=np.float64)
    x64 = x.astype(np.float64, copy=False)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff[start:stop] = np.mean(
            np.abs(x64[:, start:stop, None] - x64[:, None, :]),
            axis=0,
        )
    scale = np.median(diff[diff > 0]) if np.any(diff > 0) else 1.0
    score = np.exp(-diff / (scale + EPS))
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _partial_corr_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = _standardize(x)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov = cov + float(ridge) * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.maximum(np.diag(precision), EPS))
    pcorr = -precision / (d[:, None] * d[None, :])
    pcorr = np.maximum(pcorr, 0.0)
    np.fill_diagonal(pcorr, 0.0)
    return _minmax_offdiag(pcorr)


def _mst_score(dist: np.ndarray) -> np.ndarray:
    graph = nx.Graph()
    n = dist.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist[i, j]))
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    out = np.zeros((n, n), dtype=np.float64)
    for u, v in tree.edges():
        out[u, v] = 1.0
        out[v, u] = 1.0
    return out


def _apply_mask(score: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = score.copy()
    if mask is not None:
        out[~mask] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _top_edges(score: np.ndarray, num_edges: int) -> list[tuple[int, int]]:
    pairs = []
    n = score.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(score[i, j]), i, j))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for value, i, j in pairs[:num_edges] if value > 0.0]


def _edge_lengths(edges: set[tuple[int, int]], dist: np.ndarray) -> tuple[float, float]:
    if not edges:
        return 0.0, 0.0
    vals = np.asarray([dist[u, v] for u, v in edges], dtype=np.float64)
    return float(vals.mean()), float(np.median(vals))


def _evaluate(method: str, pred_edges: list[tuple[int, int]], gt_edges: set[tuple[int, int]], dist: np.ndarray, n: int) -> MethodResult:
    pred = set(pred_edges)
    tp = len(pred & gt_edges)
    fp = len(pred - gt_edges)
    fn = len(gt_edges - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(EPS, precision + recall)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(pred)
    mean_len, median_len = _edge_lengths(pred, dist)
    return MethodResult(
        method=method,
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        tp=tp,
        fp=fp,
        fn=fn,
        learned_edges=len(pred),
        gt_edges=len(gt_edges),
        components=nx.number_connected_components(graph),
        mean_edge_length=mean_len,
        median_edge_length=median_len,
    )


def _save_edge_index(edges: list[tuple[int, int]], path: Path) -> None:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        rows.append((v, u))
    edge_index = torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)
    torch.save(edge_index, path)


def _plot_graphs(
    xy: np.ndarray,
    gt_edges: set[tuple[int, int]],
    pred_edges: list[tuple[int, int]],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, edges, name in [
        (axes[0], sorted(gt_edges), "Ground truth"),
        (axes[1], pred_edges, title),
    ]:
        for u, v in edges:
            ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], color="black", linewidth=0.6, alpha=0.75)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, color="#1f77b4", zorder=3)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _build_scores(
    xy: np.ndarray,
    pressure: np.ndarray,
    demand: np.ndarray,
    candidate_k: int,
    max_lag: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    dist = _distance_matrix(xy)
    finite = dist[~np.eye(dist.shape[0], dtype=bool)]
    sigma = np.median(finite[finite > 0]) if np.any(finite > 0) else 1.0
    geo = _minmax_offdiag(np.exp(-dist / (sigma + EPS)))
    mask = _knn_mask(dist, candidate_k)

    p_corr = _corr_score(pressure)
    d_corr = _corr_score(demand)
    p_lag = _lag_corr_score(pressure, max_lag=max_lag)
    p_smooth = _smooth_similarity(pressure)
    p_partial = _partial_corr_score(pressure, ridge=ridge)
    mst = _mst_score(dist)

    smooth_dynamic = _rank01(p_smooth) * _rank01(p_lag)
    pressure_evidence = _rank01(p_corr) * _rank01(p_smooth) * _rank01(p_lag)
    conditional_evidence = _rank01(p_partial) * _rank01(p_smooth)

    scores = {
        "geo_knn": _apply_mask(geo, mask),
        "pressure_corr_knn": _apply_mask(p_corr, mask),
        "pressure_lag_knn": _apply_mask(p_lag, mask),
        "pressure_smooth_knn": _apply_mask(p_smooth, mask),
        "pressure_partial_knn": _apply_mask(p_partial, mask),
        "demand_corr_knn": _apply_mask(d_corr, mask),
        "geo_x_pressure_corr": _apply_mask(_rank01(geo) * _rank01(p_corr), mask),
        "geo_x_pressure_lag": _apply_mask(_rank01(geo) * _rank01(p_lag), mask),
        "geo_x_pressure_smooth": _apply_mask(_rank01(geo) * _rank01(p_smooth), mask),
        "geo_x_smooth_dynamic": _apply_mask(_rank01(geo) * smooth_dynamic, mask),
        "geo_x_pressure_evidence": _apply_mask(_rank01(geo) * pressure_evidence, mask),
        "geo_x_partial_smooth": _apply_mask(_rank01(geo) * conditional_evidence, mask),
        "geo_mst_plus_pressure": _apply_mask(np.maximum(mst, _rank01(geo) * pressure_evidence), mask),
    }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--candidate_k", type=int, default=12)
    parser.add_argument("--top_edges", type=int, default=0, help="0 means use GT edge count for evaluation only.")
    parser.add_argument("--max_lag", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--plot_top", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xy, pressure, demand, edge_index = _load_dataset(args.data_dir)
    gt_edges = _undirected_edges(edge_index)
    n = xy.shape[0]
    dist = _distance_matrix(xy)
    edge_budget = int(args.top_edges) if args.top_edges > 0 else len(gt_edges)

    scores = _build_scores(
        xy=xy,
        pressure=pressure,
        demand=demand,
        candidate_k=args.candidate_k,
        max_lag=args.max_lag,
        ridge=args.ridge,
    )

    results = []
    selected_by_method = {}
    for method, score in scores.items():
        pred_edges = _top_edges(score, edge_budget)
        selected_by_method[method] = pred_edges
        results.append(_evaluate(method, pred_edges, gt_edges, dist, n))
        _save_edge_index(pred_edges, args.out_dir / f"{method}_edges.pt")
        np.save(args.out_dir / f"{method}_score.npy", score)

    results.sort(key=lambda item: item.f1, reverse=True)

    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MethodResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    config = {
        "data_dir": str(args.data_dir),
        "candidate_k": args.candidate_k,
        "top_edges_for_evaluation": edge_budget,
        "gt_edges": len(gt_edges),
        "max_lag": args.max_lag,
        "ridge": args.ridge,
        "note": "GT edges are used only for evaluation/ranking, never to build scores.",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for result in results[: max(0, args.plot_top)]:
        _plot_graphs(
            xy=xy,
            gt_edges=gt_edges,
            pred_edges=selected_by_method[result.method],
            title=f"{result.method} | F1={result.f1:.3f}",
            out_path=args.out_dir / f"compare_{result.method}.png",
        )

    print(f"[OK] Pseudo-adjacency diagnosis written to {args.out_dir}")
    print("method,f1,precision,recall,tp,fp,fn,components,mean_edge_length")
    for result in results:
        print(
            f"{result.method},{result.f1:.4f},{result.precision:.4f},{result.recall:.4f},"
            f"{result.tp},{result.fp},{result.fn},{result.components},{result.mean_edge_length:.4f}"
        )


if __name__ == "__main__":
    main()
