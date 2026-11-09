"""Run the recovered final balanced joint configuration over the L/H grid."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
DATA_DIR = PROJECT_ROOT / "data/synthetic/water_branch_n50_base"
DEFAULT_OUT = RESULTS_ROOT / "raw/temporal_balanced_joint"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_cell(lags: int, horizon: int, seed: int, epochs: int, root: Path, force: bool) -> dict:
    out = root / f"l{lags}_h{horizon}_seed{seed}"
    metrics_path = out / "forecast_metrics.json"
    elapsed_path = out / "elapsed_seconds.txt"
    out.mkdir(parents=True, exist_ok=True)
    if force or not metrics_path.exists():
        run_name = f"balanced_l{lags}_h{horizon}_s{seed}_e{epochs}"
        cmd = [
            sys.executable, "scripts/experiments/run_train_synthetic.py",
            "--data_dir", str(DATA_DIR), "--out_dir", str(out), "--run_name", run_name,
            "--num_nodes", "50", "--lags", str(lags), "--prediction_window", str(horizon),
            "--max_epochs", str(epochs), "--check_val_every_n_epoch", "4", "--batch_size", "32",
            "--k", "2", "--ffun", "mlp", "--gfun", "gcn", "--lambda_graph", "0.0",
            "--use_complementary_loss", "--alpha_forecast", "0.8",
            "--lambda_soft_pseudo_edge", "0.5", "--lambda_stability", "0.0",
            "--pseudo_score_mode", "observable_expert_full", "--pseudo_topk", "2",
            "--pseudo_include_mst", "--select_best_by", "observable_val",
            "--assign_candidate_mask", "--candidate_mode", "q_dynamic", "--candidate_k", "3",
            "--static_top_edges", "0", "--static_row_k", "3", "--static_symmetry", "mutual",
            "--static_include_mst", "--ensure_connected_consensus", "--export_soft_graph", "--save_soft_score",
            "--task_loss_type", "huber", "--task_huber_delta", "1.0",
            "--task_feature_combine", "weighted_mean", "--task_feature_normalization", "target_std",
            "--pressure_weight", "1.0", "--demand_weight", "1.0", "--seed", str(seed),
        ]
        started = time.perf_counter()
        with (out / "training.log").open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        elapsed_path.write_text(f"{time.perf_counter()-started:.9f}\n", encoding="utf-8")

    forecast = read_json(metrics_path)
    comparison = read_json(out / "graph_comparison_softprob/comparison_summary.json")
    test = forecast["test"]
    val = forecast["best_observable_validation"]
    rec = comparison["edge_recovery_undirected"]
    graph = comparison["learned_undirected"]
    task_error = float(test["mse"])
    task_score = 1.0 / (1.0 + task_error)
    f1 = float(rec["f1"])
    return {
        "lags": lags, "horizon": horizon, "horizon_lag_ratio": horizon / lags,
        "seed": seed, "selected_epoch": forecast["selected_epoch"],
        "J_obs": float(val["observable_val_score"]),
        "test_mse": task_error,
        "test_mse_feature_mean": 0.5 * (float(test["pressure_mse"]) + float(test["demand_mse"])),
        "test_pressure_mse": float(test["pressure_mse"]),
        "test_demand_mse": float(test["demand_mse"]),
        "f1": f1, "precision": float(rec["precision"]), "recall": float(rec["recall"]),
        "edges": int(graph["num_edges"]), "components": int(graph["num_connected_components_undirected"]),
        "task_score": task_score, "eval_0p5": 0.5 * task_score + 0.5 * f1,
        "elapsed_seconds": float(elapsed_path.read_text()) if elapsed_path.exists() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", nargs="+", type=int, default=[4, 12, 24])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for lags in args.lags:
        for horizon in args.horizons:
            print(f"[CELL] balanced L={lags} H={horizon}", flush=True)
            rows.append(run_cell(lags, horizon, args.seed, args.epochs, args.out_root, args.force))
            write_csv(args.out_root / "balanced_joint_results.csv", rows)
            print(json.dumps(rows[-1], indent=2), flush=True)
    print(f"[OK] {len(rows)} balanced cells")


if __name__ == "__main__":
    main()
