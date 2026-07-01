"""Oracle diagnostic of observable proxy landscapes around the true graph.

Ground-truth edges are used only to construct controlled perturbations and to
measure alignment. The tested scores themselves use coordinates and time series
only. This script must never be used for model or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_proxy_methodology_study import (  # noqa: E402
    DATASETS,
    geometry_scores,
    minmax_offdiag,
    partial_corr_score,
    smooth_score,
    standardize,
    var_score,
)


HELDOUT = {
    "heldout_base": PROJECT_ROOT / "data/synthetic/heldout_water_base_s101",
    "heldout_bridges": PROJECT_ROOT / "data/synthetic/heldout_water_bridges_s103",
    "heldout_loopy": PROJECT_ROOT / "data/synthetic/heldout_water_loopy_s107",
    "heldout_ar": PROJECT_ROOT / "data/synthetic/heldout_water_ar_s109",
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical(int(u), int(v))
        for u, v in zip(edge_index[0], edge_index[1])
        if int(u) != int(v)
    }


def conditional_pressure_gain(
    pressure_train: np.ndarray,
    demand_train: np.ndarray,
    pressure_val: np.ndarray,
    demand_val: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Pairwise validation gain beyond the target node's own state."""
    p_train = standardize(pressure_train.astype(np.float64))
    d_train = standardize(demand_train.astype(np.float64))
    p_val = (pressure_val - pressure_train.mean(0)) / (pressure_train.std(0) + 1e-12)
    d_val = (demand_val - demand_train.mean(0)) / (demand_train.std(0) + 1e-12)
    n = p_train.shape[1]
    directed = np.zeros((n, n), dtype=np.float64)

    def validation_sse(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
        coef = np.linalg.solve(
            x_train.T @ x_train + ridge * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )
        error = y_val - x_val @ coef
        return float(error @ error)

    for target in range(n):
        y_train = p_train[1:, target]
        y_val = p_val[1:, target]
        base_train = np.stack([p_train[:-1, target], d_train[:-1, target]], axis=1)
        base_val = np.stack([p_val[:-1, target], d_val[:-1, target]], axis=1)
        base_sse = validation_sse(base_train, y_train, base_val, y_val)
        for source in range(n):
            if source == target:
                continue
            source_train = np.stack([p_train[:-1, source], d_train[:-1, source]], axis=1)
            source_val = np.stack([p_val[:-1, source], d_val[:-1, source]], axis=1)
            full_sse = validation_sse(
                np.concatenate([base_train, source_train], axis=1),
                y_train,
                np.concatenate([base_val, source_val], axis=1),
                y_val,
            )
            directed[source, target] = max(0.0, (base_sse - full_sse) / (base_sse + 1e-12))
    score = np.maximum(directed, directed.T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def inverse_demand_pressure_response(pressure: np.ndarray, demand: np.ndarray, ridge: float) -> np.ndarray:
    """Estimate the sparse operator behind the dense demand-to-pressure response."""
    d = standardize(np.diff(demand.astype(np.float64), axis=0))
    p = standardize(np.diff(pressure.astype(np.float64), axis=0))
    response = np.linalg.solve(d.T @ d + ridge * np.eye(d.shape[1]), d.T @ p)
    inverse = np.linalg.pinv(response)
    score = np.maximum(np.abs(inverse), np.abs(inverse).T)
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def backbone_score(xy: np.ndarray) -> np.ndarray:
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.nan)
    finite = dist[np.isfinite(dist)]
    closeness = 1.0 / (np.nanmean(dist, axis=1) + 1e-12)
    closeness = (closeness - closeness.min()) / (closeness.max() - closeness.min() + 1e-12)
    moderate = np.exp(-np.abs(dist - np.nanquantile(dist, 0.35)) / (np.nanstd(dist) + 1e-12))
    score = np.maximum(closeness[:, None], closeness[None, :]) * moderate
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)
    return minmax_offdiag(score)


def score_family(xy: np.ndarray, pressure: np.ndarray, demand: np.ndarray, train_end: int, val_end: int, ridge: float) -> dict[str, np.ndarray]:
    geo_rank = geometry_scores(xy)["geo_rank"]
    smooth = smooth_score(pressure[:train_end])
    demand_var = var_score(demand[:train_end], ridge)
    pressure_partial = partial_corr_score(np.diff(pressure[:train_end], axis=0), ridge)
    inverse_response = inverse_demand_pressure_response(
        pressure[:train_end], demand[:train_end], ridge
    )
    backbone = backbone_score(xy)
    gain = conditional_pressure_gain(
        pressure[:train_end], demand[:train_end],
        pressure[train_end:val_end], demand[train_end:val_end], ridge,
    )
    eps = 1e-6

    def geometric_mix(*parts: tuple[np.ndarray, float]) -> np.ndarray:
        total = sum(weight for _, weight in parts)
        log_score = sum(weight * np.log(score + eps) for score, weight in parts) / total
        return minmax_offdiag(np.exp(log_score))

    return {
        "geometry": geo_rank,
        "pressure_smooth": smooth,
        "conditional_gain": gain,
        "demand_var": demand_var,
        "pressure_diff_partial": pressure_partial,
        "inverse_demand_pressure": inverse_response,
        "geo_x_smooth": geometric_mix((geo_rank, 1.0), (smooth, 1.0)),
        "geo_x_demand_var": geometric_mix((geo_rank, 1.0), (demand_var, 1.0)),
        "geo_x_inverse_response": geometric_mix((geo_rank, 1.0), (inverse_response, 1.0)),
        "geo_x_smooth_x_demand_var": geometric_mix(
            (geo_rank, 1.0), (smooth, 1.0), (demand_var, 1.0)
        ),
        "electric_voltage": geometric_mix((geo_rank, 1.0), (smooth, 1.0)),
        "electric_dynamic": geometric_mix((geo_rank, 1.0), (smooth, 1.0), (demand_var, 1.0)),
        "electric_grid": geometric_mix((geo_rank, 1.0), (smooth, 1.0), (demand_var, 1.0), (backbone, 0.7)),
        "geo_x_smooth_x_inverse": geometric_mix(
            (geo_rank, 1.0), (smooth, 1.0), (inverse_response, 1.0)
        ),
        "geo_x_gain": geometric_mix((geo_rank, 1.0), (gain, 1.0)),
        "smooth_x_gain": geometric_mix((smooth, 1.0), (gain, 1.0)),
        "geo_x_smooth_x_gain": geometric_mix((geo_rank, 1.0), (smooth, 1.0), (gain, 1.0)),
        "geo2_x_smooth_x_gain": geometric_mix((geo_rank, 2.0), (smooth, 1.0), (gain, 1.0)),
        "geo_x_smooth2_x_gain": geometric_mix((geo_rank, 1.0), (smooth, 2.0), (gain, 1.0)),
        "geo_x_smooth_x_gain2": geometric_mix((geo_rank, 1.0), (smooth, 1.0), (gain, 2.0)),
    }


def graph_value(edges: set[tuple[int, int]], score: np.ndarray) -> float:
    return float(np.mean([np.log(float(score[u, v]) + 1e-9) for u, v in edges]))


def top_graph(score: np.ndarray, budget: int) -> set[tuple[int, int]]:
    ranked = sorted(
        [(float(score[u, v]), (u, v)) for u in range(score.shape[0]) for v in range(u + 1, score.shape[1])],
        reverse=True,
    )
    return {edge for _, edge in ranked[:budget]}


def f1_score(gt: set[tuple[int, int]], graph: set[tuple[int, int]]) -> float:
    tp = len(gt & graph)
    precision = tp / max(1, len(graph))
    recall = tp / max(1, len(gt))
    return 2 * precision * recall / max(1e-12, precision + recall)


def perturb(gt: set[tuple[int, int]], nonedges: list[tuple[int, int]], swaps: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    graph = set(gt)
    removed = rng.choice(len(gt), size=swaps, replace=False)
    added = rng.choice(len(nonedges), size=swaps, replace=False)
    gt_list = sorted(gt)
    for index in removed:
        graph.remove(gt_list[int(index)])
    for index in added:
        graph.add(nonedges[int(index)])
    return graph


def analyze_score(
    dataset: str,
    name: str,
    score: np.ndarray,
    gt: set[tuple[int, int]],
    nonedges: list[tuple[int, int]],
    swap_levels: list[int],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    gt_value = graph_value(gt, score)
    top = top_graph(score, len(gt))
    better_single = 0
    best_single_value = -np.inf
    best_single_f1 = 0.0
    single_count = len(gt) * len(nonedges)
    for removed in gt:
        removed_value = np.log(float(score[removed]) + 1e-9)
        for added in nonedges:
            value = gt_value + (np.log(float(score[added]) + 1e-9) - removed_value) / len(gt)
            if value > gt_value + 1e-12:
                better_single += 1
            if value > best_single_value:
                best_single_value = value
                best_single_f1 = 1.0 - 1.0 / len(gt)

    curves = []
    means = []
    for swaps in swap_levels:
        values = []
        for _ in range(repeats):
            graph = perturb(gt, nonedges, min(swaps, len(gt)), rng)
            values.append(graph_value(graph, score))
        means.append(float(np.mean(values)))
        curves.append({
            "dataset": dataset, "score": name, "swaps": swaps,
            "mean_f1": 1.0 - swaps / len(gt),
            "mean_proxy_value": float(np.mean(values)),
            "std_proxy_value": float(np.std(values)),
            "gt_proxy_value": gt_value,
        })
    monotonic_steps = sum(means[i] > means[i + 1] for i in range(len(means) - 1))
    summary = {
        "dataset": dataset,
        "score": name,
        "gt_proxy_value": gt_value,
        "top_proxy_value": graph_value(top, score),
        "top_graph_f1": f1_score(gt, top),
        "single_swaps": single_count,
        "fraction_single_swaps_better_than_gt": better_single / max(1, single_count),
        "gt_local_optimum": int(better_single == 0),
        "best_single_swap_f1": best_single_f1,
        "monotonic_steps": monotonic_steps,
        "total_monotonic_steps": max(0, len(means) - 1),
    }
    return summary, curves


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS) + list(HELDOUT))
    parser.add_argument("--dataset_dir", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--ridge", type=float, default=0.2)
    parser.add_argument("--swap_levels", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs/proxy_landscape_test")
    args = parser.parse_args()
    paths = {**DATASETS, **HELDOUT}
    for item in args.dataset_dir:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        paths[name] = path if path.is_absolute() else PROJECT_ROOT / path
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries: list[dict] = []
    curves: list[dict] = []
    print("ORACLE DIAGNOSTIC ONLY: GT constructs perturbations; scores remain observable.\n")
    for dataset in args.datasets:
        data = np.load(paths[dataset] / "data.npz")
        meta = json.loads((paths[dataset] / "meta.json").read_text())
        xy = data["node_xy"][0] if data["node_xy"].ndim == 3 else data["node_xy"]
        gt = edge_set(data["edge_index"])
        all_edges = {(u, v) for u in range(xy.shape[0]) for v in range(u + 1, xy.shape[0])}
        nonedges = sorted(all_edges - gt)
        train_end = int(meta["splits"]["train"]["end"])
        val_end = int(meta["splits"]["val"]["end"])
        scores = score_family(xy, data["pressure"], data["demand"], train_end, val_end, args.ridge)
        print(f"{dataset}: |E|={len(gt)}, one-swap neighborhood={len(gt) * len(nonedges):,}")
        for name, score in scores.items():
            summary, score_curves = analyze_score(
                dataset, name, score, gt, nonedges, args.swap_levels, args.repeats, rng
            )
            summaries.append(summary)
            curves.extend(score_curves)
            print(
                f"  {name:28s} top-F1={summary['top_graph_f1']:.4f} "
                f"better-swaps={summary['fraction_single_swaps_better_than_gt']:.3f} "
                f"monotonic={summary['monotonic_steps']}/{summary['total_monotonic_steps']}"
            )
    write_csv(args.out_dir / "landscape_summary.csv", summaries)
    write_csv(args.out_dir / "perturbation_curves.csv", curves)
    print(f"\n[OK] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
