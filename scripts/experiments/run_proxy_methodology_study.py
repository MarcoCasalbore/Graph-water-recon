"""Large GT-free proxy methodology study for static graph reconstruction.

Candidate proxies are constructed only from coordinates and observed node time
series. Observable criteria select the proxy configuration. Ground-truth edges
are loaded only after selection to audit topology recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "simple": PROJECT_ROOT / "data/synthetic/simple_water_n50",
    "water_base": PROJECT_ROOT / "data/synthetic/water_branch_n50_base",
    "water_bridges": PROJECT_ROOT / "data/synthetic/water_branch_n50_bridges",
    "water_loopy": PROJECT_ROOT / "data/synthetic/water_branch_n50_loopy",
}


@dataclass(frozen=True)
class ProxyConfig:
    dynamic: str
    geometry: str
    fusion: str
    row_k: int
    symmetry: str
    include_mst: bool


def minmax_offdiag(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).copy()
    eye = np.eye(score.shape[0], dtype=bool)
    values = score[~eye]
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(score)
    lo = values[finite].min()
    hi = values[finite].max()
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.clip((score - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    score[eye] = 0.0
    return score


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)


def corr_score(x: np.ndarray, positive: bool = False) -> np.ndarray:
    z = standardize(x.astype(np.float64))
    score = z.T @ z / max(1, z.shape[0] - 1)
    score = np.maximum(score, 0.0) if positive else np.abs(score)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def smooth_score(x: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(x[:, :, None] - x[:, None, :]), axis=0)
    scale = np.median(diff[diff > 0]) if np.any(diff > 0) else 1.0
    score = np.exp(-diff / (scale + 1e-12))
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def partial_corr_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = standardize(x.astype(np.float64))
    cov = z.T @ z / max(1, z.shape[0] - 1)
    precision = np.linalg.pinv(cov + ridge * np.eye(cov.shape[0]))
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    score = np.abs(-precision / (diag[:, None] * diag[None, :] + 1e-12))
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def var_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = standardize(x.astype(np.float64))
    source, target = z[:-1], z[1:]
    coef = np.linalg.solve(source.T @ source + ridge * np.eye(source.shape[1]), source.T @ target)
    score = np.maximum(np.abs(coef), np.abs(coef).T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def cross_var_score(source_x: np.ndarray, target_x: np.ndarray, ridge: float) -> np.ndarray:
    source = standardize(source_x.astype(np.float64))
    target = standardize(target_x.astype(np.float64))
    n = target.shape[1]
    design = np.concatenate([target[:-1], source[:-1]], axis=1)
    y = target[1:]
    coef = np.linalg.solve(design.T @ design + ridge * np.eye(2 * n), design.T @ y)
    block = np.abs(coef[n:, :])
    score = np.maximum(block, block.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def innovation_score(x: np.ndarray, ridge: float) -> np.ndarray:
    z = standardize(x.astype(np.float64))
    prev, nxt = z[:-1], z[1:]
    beta = np.sum(prev * nxt, axis=0) / (np.sum(prev * prev, axis=0) + ridge)
    residual = nxt - prev * beta[None, :]
    return partial_corr_score(residual, ridge=ridge)


def lag_corr_score(x: np.ndarray, max_lag: int = 4) -> np.ndarray:
    best = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for lag in range(1, min(max_lag, x.shape[0] - 1) + 1):
        a = standardize(x[lag:].astype(np.float64))
        b = standardize(x[:-lag].astype(np.float64))
        score = np.abs(a.T @ b / max(1, a.shape[0] - 1))
        score = np.maximum(score, score.T)
        np.fill_diagonal(score, 0.0)
        best = np.maximum(best, minmax_offdiag(score))
    return minmax_offdiag(best)


def geometry_scores(xy: np.ndarray) -> dict[str, np.ndarray]:
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    positive = dist[dist > 0]
    median = np.median(positive) if positive.size else 1.0
    out = {"none": np.ones_like(dist) - np.eye(dist.shape[0])}
    for scale in [0.5, 1.0, 2.0, 4.0]:
        score = np.exp(-dist / (scale * median + 1e-12))
        np.fill_diagonal(score, 0.0)
        out[f"geo_exp_{scale:g}"] = minmax_offdiag(score)
    ranks = np.argsort(np.argsort(dist + np.eye(dist.shape[0]) * 1e9, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(np.float64))
    np.fill_diagonal(rank_score, 0.0)
    out["geo_rank"] = minmax_offdiag(0.5 * (rank_score + rank_score.T))
    return out


def dynamic_scores(pressure: np.ndarray, demand: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    p_diff = np.diff(pressure, axis=0)
    d_diff = np.diff(demand, axis=0)
    p_smooth = smooth_score(pressure)
    d_var = var_score(demand, ridge)
    dp_var = cross_var_score(demand, pressure, ridge)
    p_innov = innovation_score(pressure, ridge)
    d_innov = innovation_score(demand, ridge)
    base = {
        "pressure_smooth": p_smooth,
        "pressure_corr": corr_score(pressure),
        "demand_corr": corr_score(demand),
        "pressure_diff_corr": corr_score(p_diff),
        "demand_diff_corr": corr_score(d_diff),
        "pressure_partial": partial_corr_score(pressure, ridge),
        "pressure_diff_partial": partial_corr_score(p_diff, ridge),
        "demand_var": d_var,
        "demand_to_pressure_var": dp_var,
        "pressure_lag": lag_corr_score(pressure),
        "pressure_innovation": p_innov,
        "demand_innovation": d_innov,
    }
    base.update(
        {
            "smooth_x_varmax": minmax_offdiag(p_smooth * np.maximum(d_var, dp_var)),
            "smooth_x_innovation": minmax_offdiag(p_smooth * np.maximum(p_innov, d_innov)),
            "varmax": minmax_offdiag(np.maximum(d_var, dp_var)),
            "innovation_max": minmax_offdiag(np.maximum(p_innov, d_innov)),
            "lag_x_smooth": minmax_offdiag(lag_corr_score(pressure) * p_smooth),
        }
    )
    return base


def fuse(dynamic: np.ndarray, geo: np.ndarray, mode: str) -> np.ndarray:
    if mode == "dynamic_only":
        return minmax_offdiag(dynamic)
    if mode == "product":
        return minmax_offdiag(dynamic * geo)
    if mode == "sqrt_product":
        return minmax_offdiag(np.sqrt(np.maximum(dynamic * geo, 0.0)))
    if mode == "geo_gate_25":
        return minmax_offdiag(dynamic * (0.25 + 0.75 * geo))
    if mode == "geo_gate_50":
        return minmax_offdiag(dynamic * (0.50 + 0.50 * geo))
    raise ValueError(mode)


def mst_edges(xy: np.ndarray) -> set[tuple[int, int]]:
    n = xy.shape[0]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: set[tuple[int, int]] = set()
    for _ in range(n - 1):
        work = dist.copy()
        work[~selected, :] = np.inf
        work[:, selected] = np.inf
        u, v = np.unravel_index(np.argmin(work), work.shape)
        if not np.isfinite(work[u, v]):
            break
        edges.add((min(int(u), int(v)), max(int(u), int(v))))
        selected[v] = True
    return edges


def graph_from_score(score: np.ndarray, row_k: int, symmetry: str, mst: set[tuple[int, int]] | None) -> set[tuple[int, int]]:
    n = score.shape[0]
    work = score.copy()
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(-work, kth=row_k - 1, axis=1)[:, :row_k]
    directed = {(i, int(j)) for i in range(n) for j in idx[i]}
    edges = set()
    for u, v in directed:
        if symmetry == "union" or (v, u) in directed:
            edges.add((min(u, v), max(u, v)))
    if mst:
        edges |= mst
    return edges


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.triu(np.ones_like(a, dtype=bool), 1)
    x, y = a[mask], b[mask]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry) + 1e-12))


def jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    return len(a & b) / max(1, len(a | b))


def forecast_bic(
    pressure_train: np.ndarray,
    demand_train: np.ndarray,
    pressure_val: np.ndarray,
    demand_val: np.ndarray,
    edges: set[tuple[int, int]],
    local_cache: dict[tuple[int, tuple[int, ...]], tuple[float, int, int]],
    ridge: float = 0.1,
) -> float:
    n = pressure_train.shape[1]
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    total_sse = 0.0
    total_obs = 0
    total_params = 0
    for target in range(n):
        nodes = [target] + sorted(neighbors[target])
        cache_key = (target, tuple(nodes))
        if cache_key in local_cache:
            sse, observations, parameters = local_cache[cache_key]
            total_sse += sse
            total_obs += observations
            total_params += parameters
            continue
        x_train = np.concatenate([pressure_train[:-1, nodes], demand_train[:-1, nodes]], axis=1)
        y_train = pressure_train[1:, target:target + 1]
        x_val = np.concatenate([pressure_val[:-1, nodes], demand_val[:-1, nodes]], axis=1)
        y_val = pressure_val[1:, target:target + 1]
        x_mean, x_std = x_train.mean(0), x_train.std(0) + 1e-12
        y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-12
        xt = (x_train - x_mean) / x_std
        yt = (y_train - y_mean) / y_std
        xv = (x_val - x_mean) / x_std
        yv = (y_val - y_mean) / y_std
        coef = np.linalg.solve(xt.T @ xt + ridge * np.eye(xt.shape[1]), xt.T @ yt)
        err = yv - xv @ coef
        result = (float(np.sum(err * err)), int(err.size), int(coef.size))
        local_cache[cache_key] = result
        total_sse += result[0]
        total_obs += result[1]
        total_params += result[2]
    mse = total_sse / max(1, total_obs)
    return float(total_obs * np.log(mse + 1e-12) + total_params * np.log(max(2, total_obs)))


def edge_metrics(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float]:
    tp, fp, fn = len(gt & pred), len(pred - gt), len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def average_precision(gt: set[tuple[int, int]], score: np.ndarray) -> float:
    ranked = sorted(
        [((i, j), float(score[i, j])) for i in range(score.shape[0]) for j in range(i + 1, score.shape[1])],
        key=lambda item: item[1], reverse=True,
    )
    hits, acc = 0, 0.0
    for rank, (edge, _) in enumerate(ranked, 1):
        if edge in gt:
            hits += 1
            acc += hits / rank
    return acc / max(1, len(gt))


def percentile_ranks(values: list[float], higher_is_better: bool) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr if higher_is_better else -arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(arr), endpoint=True)
    return ranks.tolist()


def config_key(row: dict) -> tuple:
    return tuple(row[name] for name in [
        "ridge", "dynamic", "geometry", "fusion", "row_k", "symmetry", "include_mst"
    ])


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_rows(path: Path, rows: list[dict]) -> None:
    public = [public_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
        writer.writeheader()
        writer.writerows(public)


def study_dataset(name: str, data_dir: Path, out_dir: Path, row_ks: list[int], ridges: list[float]) -> list[dict]:
    data = np.load(data_dir / "data.npz")
    meta = json.loads((data_dir / "meta.json").read_text())
    pressure, demand = data["pressure"], data["demand"]
    xy = data["node_xy"][0] if data["node_xy"].ndim == 3 else data["node_xy"]
    train_end = int(meta["splits"]["train"]["end"])
    val_end = int(meta["splits"]["val"]["end"])
    split = max(4, train_end // 2)
    p_a, d_a = pressure[:split], demand[:split]
    p_b, d_b = pressure[split:train_end], demand[split:train_end]
    p_train, d_train = pressure[:train_end], demand[:train_end]
    p_val, d_val = pressure[train_end:val_end], demand[train_end:val_end]
    geo = geometry_scores(xy)
    mst = mst_edges(xy)
    rows: list[dict] = []
    graph_cache: dict[tuple[tuple[int, int], ...], float] = {}
    local_forecast_cache: dict[tuple[int, tuple[int, ...]], tuple[float, int, int]] = {}
    for ridge in ridges:
        dyn_a = dynamic_scores(p_a, d_a, ridge)
        dyn_b = dynamic_scores(p_b, d_b, ridge)
        dyn_full = dynamic_scores(p_train, d_train, ridge)
        for dynamic_name in dyn_full:
            for geo_name, geo_score in geo.items():
                for fusion_name in ["dynamic_only", "product", "sqrt_product", "geo_gate_25", "geo_gate_50"]:
                    if geo_name == "none" and fusion_name != "dynamic_only":
                        continue
                    if geo_name != "none" and fusion_name == "dynamic_only":
                        continue
                    score_a = fuse(dyn_a[dynamic_name], geo_score, fusion_name)
                    score_b = fuse(dyn_b[dynamic_name], geo_score, fusion_name)
                    score_full = fuse(dyn_full[dynamic_name], geo_score, fusion_name)
                    score_consistency = rank_corr(score_a, score_b)
                    for row_k in row_ks:
                        for symmetry in ["union", "mutual"]:
                            for include_mst in [False, True]:
                                graph_a = graph_from_score(score_a, row_k, symmetry, mst if include_mst else None)
                                graph_b = graph_from_score(score_b, row_k, symmetry, mst if include_mst else None)
                                graph_full = graph_from_score(score_full, row_k, symmetry, mst if include_mst else None)
                                key = tuple(sorted(graph_full))
                                if key not in graph_cache:
                                    graph_cache[key] = forecast_bic(
                                        p_train, d_train, p_val, d_val, graph_full, local_forecast_cache
                                    )
                                rows.append({
                                    "dataset": name, "ridge": ridge, "dynamic": dynamic_name,
                                    "geometry": geo_name, "fusion": fusion_name, "row_k": row_k,
                                    "symmetry": symmetry, "include_mst": include_mst,
                                    "num_edges": len(graph_full), "score_consistency": score_consistency,
                                    "edge_consistency": jaccard(graph_a, graph_b),
                                    "forecast_bic": graph_cache[key],
                                    "_edges": graph_full, "_score": score_full,
                                })
    consistency_rank = percentile_ranks([r["score_consistency"] for r in rows], True)
    edge_rank = percentile_ranks([r["edge_consistency"] for r in rows], True)
    forecast_rank = percentile_ranks([r["forecast_bic"] for r in rows], False)
    for idx, row in enumerate(rows):
        row["observable_score"] = (consistency_rank[idx] + edge_rank[idx] + forecast_rank[idx]) / 3.0
    rows.sort(key=lambda row: row["observable_score"], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / f"{name}_observable_proxy_study.csv", rows)
    return rows


def add_posthoc_metrics(rows: list[dict], data_dir: Path) -> None:
    data = np.load(data_dir / "data.npz")
    gt = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(data["edge_index"][0], data["edge_index"][1])
        if int(u) != int(v)
    }
    ap_cache: dict[tuple, float] = {}
    for row in rows:
        score_key = tuple(row[name] for name in ["ridge", "dynamic", "geometry", "fusion"])
        if score_key not in ap_cache:
            ap_cache[score_key] = average_precision(gt, row["_score"])
        row["average_precision"] = ap_cache[score_key]
        row.update(edge_metrics(gt, row["_edges"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--dataset_dir", action="append", default=[], metavar="NAME=PATH",
        help="Add or override a dataset directory without changing the source file.",
    )
    parser.add_argument("--row_ks", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--ridges", nargs="+", type=float, default=[0.01, 0.05, 0.2])
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs/proxy_methodology_study")
    args = parser.parse_args()
    dataset_dirs = dict(DATASETS)
    for item in args.dataset_dir:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        dataset_dirs[name] = path if path.is_absolute() else PROJECT_ROOT / path
    rows_by_dataset: dict[str, list[dict]] = {}
    print("Observable proxy study")
    print("Q = fuse(dynamic evidence, geometric evidence)")
    print("Graph = row-top-k(Q), union/mutual symmetry, optional geometric MST")
    print("Validation forecast = next pressure from local/neighbor pressure and demand; complexity penalized by BIC")
    print("J_proxy = mean(rank(score stability), rank(edge stability), rank(validation pressure-forecast BIC))")
    print("Ground truth is loaded only after local and cross-dataset selections are frozen.\n")
    for dataset in args.datasets:
        rows = study_dataset(dataset, dataset_dirs[dataset], args.out_dir, args.row_ks, args.ridges)
        rows_by_dataset[dataset] = rows
        print(f"{dataset}: observable candidates={len(rows)}, unique graphs evaluated without GT")

    common_scores: dict[tuple, list[float]] = {}
    for rows in rows_by_dataset.values():
        for row in rows:
            common_scores.setdefault(config_key(row), []).append(row["observable_score"])
    complete_scores = {
        key: values for key, values in common_scores.items() if len(values) == len(rows_by_dataset)
    }
    common_key = max(complete_scores, key=lambda key: float(np.mean(complete_scores[key])))

    # The observable selections are now frozen. GT is used below only for audit metrics.
    selected = []
    for dataset, rows in rows_by_dataset.items():
        add_posthoc_metrics(rows, dataset_dirs[dataset])
        best_obs = rows[0]
        common_obs = next(row for row in rows if config_key(row) == common_key)
        best_f1 = max(rows, key=lambda row: row["f1"])
        selected.append({"selection": "dataset_observable", **public_row(best_obs)})
        selected.append({"selection": "cross_dataset_observable", **public_row(common_obs)})
        selected.append({"selection": "best_f1_evaluation_only", **public_row(best_f1)})
        write_rows(args.out_dir / f"{dataset}_posthoc_audit.csv", rows)
        print(
            f"{dataset}: local observable F1={best_obs['f1']:.4f}; "
            f"common observable F1={common_obs['f1']:.4f}; "
            f"best post-hoc F1={best_f1['f1']:.4f}"
        )
    write_rows(args.out_dir / "selected_summary.csv", selected)
    print("\nCommon observable configuration:")
    print(dict(zip(["ridge", "dynamic", "geometry", "fusion", "row_k", "symmetry", "include_mst"], common_key)))


if __name__ == "__main__":
    main()
