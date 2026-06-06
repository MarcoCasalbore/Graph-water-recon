"""Search observable edge scores against synthetic ground-truth graphs.

This is a diagnostic/oracle-analysis script: ground-truth edges are used only
to evaluate scores built from allowed observables: node positions and node
time series. The goal is to find mathematical edge properties worth turning
into DGM candidate masks or topology losses.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


EPS = 1e-12


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[tuple[int, int]]]:
    data = np.load(data_dir / "data.npz")
    xy = np.asarray(data["node_xy"], dtype=np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    pressure = np.asarray(data["pressure"], dtype=np.float64)
    demand = np.asarray(data["demand"], dtype=np.float64)
    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    gt_edges = set()
    for u, v in edge_index.T.tolist():
        if int(u) == int(v):
            continue
        gt_edges.add(tuple(sorted((int(u), int(v)))))
    return xy, pressure, demand, gt_edges


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + EPS)


def distance_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def minmax_offdiag(score: np.ndarray) -> np.ndarray:
    out = np.asarray(score, dtype=np.float64).copy()
    mask = ~np.eye(out.shape[0], dtype=bool)
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
        out = (out - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def rank01(score: np.ndarray) -> np.ndarray:
    n = score.shape[0]
    pairs = [(float(score[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort()
    out = np.zeros((n, n), dtype=np.float64)
    denom = max(1, len(pairs) - 1)
    for rank, (_, i, j) in enumerate(pairs):
        value = rank / denom
        out[i, j] = out[j, i] = value
    return out


def corr_score(x: np.ndarray, absolute: bool = False) -> np.ndarray:
    z = standardize(x)
    score = (z.T @ z) / max(1, z.shape[0] - 1)
    score = np.abs(score) if absolute else np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def lag_corr_score(x: np.ndarray, max_lag: int = 4, absolute: bool = False) -> np.ndarray:
    z = standardize(x)
    n = z.shape[1]
    best = np.zeros((n, n), dtype=np.float64)
    for lag in range(1, max(1, int(max_lag)) + 1):
        a = z[lag:]
        b = z[:-lag]
        c = (a.T @ b) / max(1, a.shape[0] - 1)
        c = np.maximum(c, c.T)
        c = np.abs(c) if absolute else np.maximum(c, 0.0)
        best = np.maximum(best, c)
    np.fill_diagonal(best, 0.0)
    return minmax_offdiag(best)


def smooth_similarity(x: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(x[:, :, None] - x[:, None, :]), axis=0)
    scale = np.median(diff[diff > 0]) if np.any(diff > 0) else 1.0
    score = np.exp(-diff / (scale + EPS))
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def partial_corr_score(x: np.ndarray, ridge: float = 0.05, absolute: bool = False) -> np.ndarray:
    z = standardize(x)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov = cov + float(ridge) * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.maximum(np.diag(precision), EPS))
    pcorr = -precision / (d[:, None] * d[None, :])
    pcorr = np.abs(pcorr) if absolute else np.maximum(pcorr, 0.0)
    np.fill_diagonal(pcorr, 0.0)
    return minmax_offdiag(pcorr)


def cross_partial_pressure_demand(pressure: np.ndarray, demand: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    z = standardize(np.concatenate([pressure, demand], axis=1))
    n = pressure.shape[1]
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov = cov + float(ridge) * np.eye(2 * n)
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.maximum(np.diag(precision), EPS))
    pcorr = -precision / (d[:, None] * d[None, :])
    score = np.maximum(np.abs(pcorr[:n, n:]), np.abs(pcorr[n:, :n]).T)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def lag_regression_scores(x: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise lag score: source j helps predict target i at t+1.

    Returns absolute source coefficient and relative MSE improvement over an
    own-lag-only baseline, symmetrized by max over directions.
    """
    z = standardize(x)
    y_all = z[1:]
    lag_all = z[:-1]
    n = z.shape[1]
    coef_score = np.zeros((n, n), dtype=np.float64)
    gain_score = np.zeros((n, n), dtype=np.float64)

    for target in range(n):
        y = y_all[:, target]
        own = lag_all[:, [target]]
        own_design = np.concatenate([np.ones((own.shape[0], 1)), own], axis=1)
        beta0 = np.linalg.solve(own_design.T @ own_design + ridge * np.eye(2), own_design.T @ y)
        base_resid = y - own_design @ beta0
        base_mse = float(np.mean(base_resid * base_resid)) + EPS

        for source in range(n):
            if source == target:
                continue
            design = np.concatenate([np.ones((own.shape[0], 1)), own, lag_all[:, [source]]], axis=1)
            beta = np.linalg.solve(design.T @ design + ridge * np.eye(3), design.T @ y)
            resid = y - design @ beta
            mse = float(np.mean(resid * resid))
            coef_score[target, source] = abs(float(beta[-1]))
            gain_score[target, source] = max(0.0, (base_mse - mse) / base_mse)

    coef_score = np.maximum(coef_score, coef_score.T)
    gain_score = np.maximum(gain_score, gain_score.T)
    np.fill_diagonal(coef_score, 0.0)
    np.fill_diagonal(gain_score, 0.0)
    return minmax_offdiag(coef_score), minmax_offdiag(gain_score)


def global_var_score(x: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    """Fit X[t+1] = X[t] B with ridge; score off-diagonal coefficients."""
    z = standardize(x)
    source = z[:-1]
    target = z[1:]
    xtx = source.T @ source
    xty = source.T @ target
    coef = np.linalg.solve(xtx + float(ridge) * np.eye(source.shape[1]), xty)
    score = np.abs(coef)
    score = np.maximum(score, score.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def global_cross_var_score(source_x: np.ndarray, target_x: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    """Fit target[t+1] from all source[t] plus own target[t]."""
    source = standardize(source_x)
    target = standardize(target_x)
    n = target.shape[1]
    design = np.concatenate([target[:-1], source[:-1]], axis=1)
    y = target[1:]
    coef = np.linalg.solve(design.T @ design + float(ridge) * np.eye(2 * n), design.T @ y)
    source_to_target = np.abs(coef[n:, :])
    score = np.maximum(source_to_target, source_to_target.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def cross_lag_regression_score(source_x: np.ndarray, target_x: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Pairwise cross-channel lag score: source channel j predicts target channel i."""
    source = standardize(source_x)
    target = standardize(target_x)
    y_all = target[1:]
    own_lag = target[:-1]
    source_lag = source[:-1]
    n = target.shape[1]
    score = np.zeros((n, n), dtype=np.float64)
    for target_i in range(n):
        y = y_all[:, target_i]
        own = own_lag[:, [target_i]]
        own_design = np.concatenate([np.ones((own.shape[0], 1)), own], axis=1)
        beta0 = np.linalg.solve(own_design.T @ own_design + ridge * np.eye(2), own_design.T @ y)
        base_resid = y - own_design @ beta0
        base_mse = float(np.mean(base_resid * base_resid)) + EPS
        for source_j in range(n):
            if source_j == target_i:
                continue
            design = np.concatenate([np.ones((own.shape[0], 1)), own, source_lag[:, [source_j]]], axis=1)
            beta = np.linalg.solve(design.T @ design + ridge * np.eye(3), design.T @ y)
            resid = y - design @ beta
            mse = float(np.mean(resid * resid))
            score[target_i, source_j] = max(abs(float(beta[-1])), max(0.0, (base_mse - mse) / base_mse))
    score = np.maximum(score, score.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def local_residual(x: np.ndarray, dist: np.ndarray, k: int) -> np.ndarray:
    n = dist.shape[0]
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    idx = np.argsort(work, axis=1)[:, : min(k, n - 1)]
    residual = x.copy()
    for i in range(n):
        neigh = idx[i]
        w = np.exp(-work[i, neigh] / (np.median(work[np.isfinite(work)]) + EPS))
        w = w / (w.sum() + EPS)
        residual[:, i] = x[:, i] - x[:, neigh] @ w
    return residual


def mst_score_from_weight(weight: np.ndarray) -> np.ndarray:
    n = weight.shape[0]
    graph = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(weight[i, j]))
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    out = np.zeros((n, n), dtype=np.float64)
    for u, v in tree.edges():
        out[u, v] = out[v, u] = 1.0
    return out


def top_edges(score: np.ndarray, m: int) -> list[tuple[int, int]]:
    pairs = [(float(score[i, j]), i, j) for i in range(score.shape[0]) for j in range(i + 1, score.shape[0])]
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for value, i, j in pairs[:m] if value > 0.0]


def mst_plus_top_edges(score: np.ndarray, dist: np.ndarray, m: int, mst_weight: np.ndarray) -> list[tuple[int, int]]:
    mst = set(top_edges(mst_score_from_weight(mst_weight), score.shape[0] - 1))
    pairs = [(float(score[i, j]), i, j) for i in range(score.shape[0]) for j in range(i + 1, score.shape[0])]
    pairs.sort(key=lambda item: item[0], reverse=True)
    edges = set(mst)
    for value, i, j in pairs:
        if len(edges) >= m:
            break
        if value > 0.0:
            edges.add((i, j))
    return sorted(edges)


def evaluate(pred_edges: list[tuple[int, int]], gt_edges: set[tuple[int, int]]) -> tuple[float, float, float, int, int, int]:
    pred = set(pred_edges)
    tp = len(pred & gt_edges)
    fp = len(pred - gt_edges)
    fn = len(gt_edges - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(EPS, precision + recall)
    return f1, precision, recall, tp, fp, fn


def plot_best(xy: np.ndarray, gt_edges: set[tuple[int, int]], pred_edges: list[tuple[int, int]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, edges, name in [(axes[0], gt_edges, "Ground truth"), (axes[1], pred_edges, title)]:
        for u, v in sorted(edges):
            ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], color="black", linewidth=0.65, alpha=0.75)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, color="#1f77b4", zorder=3)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_edges(edges: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["u", "v"])
        writer.writeheader()
        for u, v in sorted(edges):
            writer.writerow({"u": u, "v": v})


def build_scores(xy: np.ndarray, pressure: np.ndarray, demand: np.ndarray, ridge: float, lag: int) -> dict[str, np.ndarray]:
    dist = distance_matrix(xy)
    dist_norm = minmax_offdiag(dist)
    short = 1.0 - rank01(dist)
    np.fill_diagonal(short, 0.0)
    long = rank01(dist)
    p_res6 = local_residual(pressure, dist, k=6)
    p_res12 = local_residual(pressure, dist, k=12)
    dp = np.diff(pressure, axis=0)
    dd = np.diff(demand, axis=0)
    p_lag_coef, p_lag_gain = lag_regression_scores(pressure, ridge=ridge)
    d_lag_coef, d_lag_gain = lag_regression_scores(demand, ridge=ridge)

    scores = {
        "short_geo": short,
        "long_geo": long,
        "geo_exp": minmax_offdiag(np.exp(-dist / (np.median(dist[dist > 0]) + EPS))),
        "pressure_corr": corr_score(pressure),
        "pressure_abs_corr": corr_score(pressure, absolute=True),
        "pressure_lag": lag_corr_score(pressure, max_lag=lag),
        "pressure_smooth": smooth_similarity(pressure),
        "pressure_partial": partial_corr_score(pressure, ridge=ridge),
        "pressure_abs_partial": partial_corr_score(pressure, ridge=ridge, absolute=True),
        "demand_corr": corr_score(demand),
        "demand_abs_corr": corr_score(demand, absolute=True),
        "demand_partial": partial_corr_score(demand, ridge=ridge),
        "dp_corr": corr_score(dp, absolute=True),
        "dd_corr": corr_score(dd, absolute=True),
        "res6_corr": corr_score(p_res6, absolute=True),
        "res12_corr": corr_score(p_res12, absolute=True),
        "res6_partial": partial_corr_score(p_res6, ridge=ridge, absolute=True),
        "res12_partial": partial_corr_score(p_res12, ridge=ridge, absolute=True),
        "pd_cross_partial": cross_partial_pressure_demand(pressure, demand, ridge=ridge),
        "pressure_lag_coef": p_lag_coef,
        "pressure_lag_gain": p_lag_gain,
        "demand_lag_coef": d_lag_coef,
        "demand_lag_gain": d_lag_gain,
        "pressure_from_demand_lag": cross_lag_regression_score(demand, pressure, ridge=ridge),
        "demand_from_pressure_lag": cross_lag_regression_score(pressure, demand, ridge=ridge),
        "pressure_global_var": global_var_score(pressure, ridge=ridge),
        "demand_global_var": global_var_score(demand, ridge=ridge),
        "pressure_from_demand_global_var": global_cross_var_score(demand, pressure, ridge=ridge),
        "demand_from_pressure_global_var": global_cross_var_score(pressure, demand, ridge=ridge),
    }
    scores["short_pressure_smooth"] = minmax_offdiag(rank01(scores["short_geo"]) * rank01(scores["pressure_smooth"]))
    scores["geo_partial_smooth"] = minmax_offdiag(
        rank01(scores["short_geo"]) * rank01(scores["pressure_partial"]) * rank01(scores["pressure_smooth"])
    )
    scores["bridge_partial_smooth"] = minmax_offdiag(
        rank01(scores["long_geo"]) * rank01(scores["pressure_partial"]) * rank01(scores["pressure_smooth"])
    )
    scores["bridge_residual"] = minmax_offdiag(
        rank01(scores["long_geo"]) * rank01(scores["res6_corr"]) * rank01(scores["pressure_smooth"])
    )
    scores["hybrid_local_bridge"] = np.maximum(scores["geo_partial_smooth"], 0.85 * scores["bridge_partial_smooth"])
    scores["hydraulic_proxy"] = minmax_offdiag(
        rank01(scores["short_geo"])
        * rank01(scores["pressure_smooth"])
        * np.maximum(rank01(scores["pressure_partial"]), rank01(scores["pd_cross_partial"]))
    )
    scores["dynamic_proxy"] = minmax_offdiag(
        rank01(scores["pressure_smooth"])
        * np.maximum(rank01(scores["demand_lag_gain"]), rank01(scores["pressure_from_demand_lag"]))
    )
    scores["global_dynamic_proxy"] = minmax_offdiag(
        rank01(scores["pressure_smooth"])
        * np.maximum(rank01(scores["demand_global_var"]), rank01(scores["pressure_from_demand_global_var"]))
    )
    scores["geo_dynamic_proxy"] = minmax_offdiag(rank01(scores["short_geo"]) * rank01(scores["dynamic_proxy"]))
    scores["geo_global_dynamic_proxy"] = minmax_offdiag(rank01(scores["short_geo"]) * rank01(scores["global_dynamic_proxy"]))
    scores["distance"] = dist_norm
    return scores


def expression_search(base_scores: dict[str, np.ndarray], max_combo: int) -> dict[str, np.ndarray]:
    names = [
        "short_geo",
        "long_geo",
        "pressure_smooth",
        "pressure_partial",
        "pressure_abs_partial",
        "pd_cross_partial",
        "res6_corr",
        "res6_partial",
        "demand_abs_corr",
        "dp_corr",
        "demand_lag_gain",
        "pressure_from_demand_lag",
        "dynamic_proxy",
        "demand_global_var",
        "pressure_from_demand_global_var",
        "global_dynamic_proxy",
    ]
    out = {}
    for r in range(2, max_combo + 1):
        for combo in combinations(names, r):
            score = np.ones_like(next(iter(base_scores.values())))
            for name in combo:
                score *= rank01(base_scores[name])
            out["prod__" + "__".join(combo)] = minmax_offdiag(score)
    for a in names:
        for b in names:
            if a >= b:
                continue
            out[f"max__{a}__{b}"] = np.maximum(rank01(base_scores[a]), rank01(base_scores[b]))
            out[f"min__{a}__{b}"] = np.minimum(rank01(base_scores[a]), rank01(base_scores[b]))
    return out


def run_one(data_dir: Path, out_dir: Path, ridge: float, lag: int, max_combo: int, top_n: int) -> list[dict[str, object]]:
    xy, pressure, demand, gt_edges = load_dataset(data_dir)
    dist = distance_matrix(xy)
    edge_budget = len(gt_edges)
    scores = build_scores(xy, pressure, demand, ridge=ridge, lag=lag)
    scores.update(expression_search(scores, max_combo=max_combo))

    rows = []
    for name, score in scores.items():
        for selector, edges in [
            ("top_budget", top_edges(score, edge_budget)),
            ("geo_mst_plus_top", mst_plus_top_edges(score, dist, edge_budget, mst_weight=dist)),
            ("score_mst_plus_top", mst_plus_top_edges(score, dist, edge_budget, mst_weight=1.0 - rank01(score))),
        ]:
            f1, precision, recall, tp, fp, fn = evaluate(edges, gt_edges)
            pred_dist = np.asarray([dist[u, v] for u, v in edges], dtype=np.float64) if edges else np.asarray([])
            rows.append(
                {
                    "dataset": data_dir.name,
                    "method": name,
                    "selector": selector,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_edges": len(edges),
                    "gt_edges": edge_budget,
                    "pred_len_mean": float(pred_dist.mean()) if pred_dist.size else 0.0,
                    "gt_len_mean": float(np.mean([dist[u, v] for u, v in gt_edges])),
                }
            )

    rows.sort(key=lambda row: (float(row["f1"]), int(row["tp"])), reverse=True)
    ds_out = out_dir / data_dir.name
    ds_out.mkdir(parents=True, exist_ok=True)
    with (ds_out / "score_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_edges = top_edges(scores[str(best["method"])], edge_budget)
    if best["selector"] == "geo_mst_plus_top":
        best_edges = mst_plus_top_edges(scores[str(best["method"])], dist, edge_budget, mst_weight=dist)
    elif best["selector"] == "score_mst_plus_top":
        best_edges = mst_plus_top_edges(scores[str(best["method"])], dist, edge_budget, mst_weight=1.0 - rank01(scores[str(best["method"])]))
    plot_best(xy, gt_edges, best_edges, f"{best['method']} / {best['selector']}", ds_out / "best_score_graph.png")
    write_edges(best_edges, ds_out / "best_edges.csv")
    return rows[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--max_combo", type=int, default=3)
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for data_dir in args.data_dirs:
        top_rows = run_one(data_dir, args.out_dir, ridge=args.ridge, lag=args.lag, max_combo=args.max_combo, top_n=args.top_n)
        all_rows.extend(top_rows)
        print(f"\n== {data_dir.name} ==")
        for row in top_rows[: min(args.top_n, 10)]:
            print(
                f"{row['f1']:.4f}  {row['selector']:<18}  {row['method']}  "
                f"TP={row['tp']}/{row['gt_edges']} len={row['pred_len_mean']:.3f}/{row['gt_len_mean']:.3f}"
            )

    with (args.out_dir / "top_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
