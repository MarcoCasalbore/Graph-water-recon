"""Explore when temporal history alone stops explaining the synthetic dynamics.

For each (lags, horizon) cell this diagnostic trains:

1. a node-only MLP (no neighbor information);
2. fixed-architecture GCN forecasters on GT, proxy, and random graphs;
3. the existing joint DGM workflow.

The ground-truth graph is used only in the fixed-graph oracle diagnostic and in
post-hoc graph metrics. The joint DGM remains GT-free. Eval_0.5 is explicitly
post-hoc because it contains reconstruction F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
DEFAULT_OUT = RESULTS_ROOT / "raw/temporal_explainability"
DATA_DIR = PROJECT_ROOT / "data/synthetic/water_branch_n50_base"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(cmd: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def persistence_mse(lags: int, horizon: int) -> float:
    data = np.load(DATA_DIR / "data.npz")
    meta = read_json(DATA_DIR / "meta.json")
    series = np.stack([data["pressure"], data["demand"]], axis=-1).astype(np.float32)
    train = meta["splits"]["train"]
    mean = series[train["start"]:train["end"]].mean(axis=(0, 1), keepdims=True)
    std = series[train["start"]:train["end"]].std(axis=(0, 1), keepdims=True) + 1e-8
    z = (series - mean) / std
    test = meta["splits"]["test"]
    errors = []
    last_start = test["end"] - lags - horizon
    for start in range(test["start"], last_start + 1):
        last = z[start + lags - 1]
        target = z[start + lags:start + lags + horizon]
        errors.append((target - last[None, ...]) ** 2)
    return float(np.mean(np.asarray(errors)))


def node_baseline(lags: int, horizon: int, seed: int, epochs: int, cell: Path, force: bool) -> tuple[float, float]:
    out = cell / "node_mlp"
    metrics_path = out / "forecast_baseline_metrics.json"
    elapsed_path = out / "elapsed_seconds.txt"
    if force or not metrics_path.exists():
        cmd = [
            sys.executable, "scripts/experiments/train_forecast_baseline.py",
            "--data_dir", str(DATA_DIR), "--out_dir", str(out),
            "--model", "node_mlp", "--lags", str(lags),
            "--prediction_window", str(horizon), "--epochs", str(epochs),
            "--batch_size", "64", "--hidden", "64", "--seed", str(seed),
        ]
        elapsed = run_logged(cmd, out / "training.log")
        elapsed_path.write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    metrics = read_json(metrics_path)
    elapsed = float(elapsed_path.read_text()) if elapsed_path.exists() else float("nan")
    return float(metrics["test"]["normalized"]["mse"]), elapsed


def fixed_graphs(lags: int, horizon: int, seed: int, epochs: int, cell: Path, force: bool) -> tuple[dict[str, float], float]:
    out = cell / "fixed_graphs"
    csv_path = out / "oracle_graph_loss_results.csv"
    elapsed_path = out / "elapsed_seconds.txt"
    if force or not csv_path.exists():
        cmd = [
            sys.executable, "scripts/diagnostics/oracle_graph_loss_test.py",
            "--data_dir", str(DATA_DIR), "--out_dir", str(out),
            "--epochs", str(epochs), "--check_every", str(max(1, epochs // 3)),
            "--batch_size", "32", "--hidden_dim", "64", "--lags", str(lags),
            "--prediction_window", str(horizon), "--disable_self_loops",
            "--knn_k", "3", "--random_repeats", "1", "--proxy_row_k", "3",
            "--proxy_symmetry", "mutual", "--proxy_include_mst", "--seed", str(seed),
        ]
        elapsed = run_logged(cmd, out / "training.log")
        elapsed_path.write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    frame = pd.read_csv(csv_path).set_index("graph_name")
    values = {
        "gt_graph_mse": float(frame.loc["ground_truth", "test_mse"]) / 2.0,
        "random_graph_mse": float(frame.loc["random_same_edges_1", "test_mse"]) / 2.0,
        "initial_proxy_mse": float(frame.loc["initial_proxy", "test_mse"]) / 2.0,
        "observable_proxy_mse": float(frame.loc["observable_proxy", "test_mse"]) / 2.0,
    }
    elapsed = float(elapsed_path.read_text()) if elapsed_path.exists() else float("nan")
    return values, elapsed


def joint_dgm(lags: int, horizon: int, seed: int, epochs: int, cell: Path, force: bool) -> tuple[dict[str, float], float]:
    out = cell / "joint_dgm"
    metrics_path = out / "forecast_metrics.json"
    elapsed_path = out / "elapsed_seconds.txt"
    if force or not metrics_path.exists():
        run_name = f"temporal_l{lags}_h{horizon}_s{seed}_e{epochs}"
        cmd = [
            sys.executable, "scripts/experiments/run_train_synthetic.py",
            "--data_dir", str(DATA_DIR), "--out_dir", str(out), "--run_name", run_name,
            "--num_nodes", "50", "--lags", str(lags), "--prediction_window", str(horizon),
            "--max_epochs", str(epochs), "--check_val_every_n_epoch", str(max(1, epochs // 5)),
            "--batch_size", "32", "--k", "1", "--static_top_edges", "0",
            "--static_row_k", "3", "--static_symmetry", "mutual", "--static_include_mst",
            "--use_complementary_loss", "--alpha_forecast", "0.70", "--lambda_graph", "0.0",
            "--lambda_soft_pseudo_edge", "1.0", "--lambda_stability", "1.0",
            "--pseudo_score_mode", "geo_global_dynamic_proxy", "--pseudo_use_static_target",
            "--pseudo_topk", "1", "--pseudo_include_mst", "--select_best_by", "observable_val",
            "--assign_candidate_mask", "--candidate_mode", "q_dynamic", "--candidate_k", "2",
            "--ensure_connected_consensus", "--export_soft_graph", "--save_soft_score", "--seed", str(seed),
        ]
        elapsed = run_logged(cmd, out / "training.log")
        elapsed_path.write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    forecast = read_json(metrics_path)
    comparison = read_json(out / "graph_comparison_softprob/comparison_summary.json")
    test = forecast["test"]
    rec = comparison["edge_recovery_undirected"]
    graph = comparison["learned_undirected"]
    reported_task_mse = float(test["mse"])
    values = {
        "joint_mse": 0.5 * (float(test["pressure_mse"]) + float(test["demand_mse"])),
        "joint_reported_task_mse": reported_task_mse,
        "joint_f1": float(rec["f1"]),
        "joint_precision": float(rec["precision"]),
        "joint_recall": float(rec["recall"]),
        "joint_edges": int(graph["num_edges"]),
        "joint_components": int(graph["num_connected_components_undirected"]),
        "task_score": 1.0 / (1.0 + reported_task_mse),
        "eval_0p5": 0.5 / (1.0 + reported_task_mse) + 0.5 * float(rec["f1"]),
    }
    elapsed = float(elapsed_path.read_text()) if elapsed_path.exists() else float("nan")
    return values, elapsed


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", nargs="+", type=int, default=[4, 12, 24])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node_epochs", type=int, default=30)
    parser.add_argument("--fixed_epochs", type=int, default=15)
    parser.add_argument("--joint_epochs", type=int, default=25)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for lags in args.lags:
        for horizon in args.horizons:
            cell = args.out_root / f"l{lags}_h{horizon}_seed{args.seed}"
            cell.mkdir(parents=True, exist_ok=True)
            print(f"[CELL] L={lags} H={horizon}", flush=True)
            persistence = persistence_mse(lags, horizon)
            node_mse, node_seconds = node_baseline(lags, horizon, args.seed, args.node_epochs, cell, args.force)
            fixed, fixed_seconds = fixed_graphs(lags, horizon, args.seed, args.fixed_epochs, cell, args.force)
            joint, joint_seconds = joint_dgm(lags, horizon, args.seed, args.joint_epochs, cell, args.force)
            node_gain_persistence = (persistence - node_mse) / max(persistence, 1e-12)
            neighbor_gain_node = (node_mse - fixed["gt_graph_mse"]) / max(node_mse, 1e-12)
            topology_selectivity = (fixed["random_graph_mse"] - fixed["gt_graph_mse"]) / max(fixed["random_graph_mse"], 1e-12)
            joint_gain_node = (node_mse - joint["joint_mse"]) / max(node_mse, 1e-12)
            row = {
                "lags": lags, "horizon": horizon, "horizon_lag_ratio": horizon / lags,
                "persistence_mse": persistence, "node_mlp_mse": node_mse, **fixed, **joint,
                "node_gain_vs_persistence": node_gain_persistence,
                "oracle_neighbor_gain_vs_node": neighbor_gain_node,
                "topology_selectivity_gt_vs_random": topology_selectivity,
                "joint_gain_vs_node": joint_gain_node,
                "node_seconds": node_seconds, "fixed_seconds": fixed_seconds,
                "joint_seconds": joint_seconds, "seed": args.seed,
            }
            rows.append(row)
            write_csv(args.out_root / "temporal_explainability_results.csv", rows)
            (cell / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(json.dumps(row, indent=2), flush=True)

    print(f"[OK] {len(rows)} cells -> {args.out_root / 'temporal_explainability_results.csv'}")


if __name__ == "__main__":
    main()
