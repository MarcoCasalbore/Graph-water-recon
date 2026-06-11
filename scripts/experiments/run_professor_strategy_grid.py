"""Run controlled coordinate-embedding and alpha-curriculum DGM experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "simple": "data/synthetic/simple_water_n50",
    "water_base": "data/synthetic/water_branch_n50_base",
    "water_bridges": "data/synthetic/water_branch_n50_bridges",
    "water_loopy": "data/synthetic/water_branch_n50_loopy",
}
STRATEGIES = {
    "baseline": {"coordinate_dim": 0, "curriculum": False},
    "coordinates": {"coordinate_dim": 8, "curriculum": False},
    "curriculum": {"coordinate_dim": 0, "curriculum": True},
    "coordinates_curriculum": {"coordinate_dim": 8, "curriculum": True},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect(dataset: str, strategy: str, out_dir: Path) -> dict:
    forecast = read_json(out_dir / "forecast_metrics.json")
    comparison = read_json(out_dir / "graph_comparison_softprob/comparison_summary.json")
    undirected = comparison.get("edge_recovery_undirected", {})
    learned = comparison.get("learned_undirected", {})
    best = forecast.get("best_observable_validation", {})
    test = forecast.get("test", {})
    return {
        "dataset": dataset,
        "strategy": strategy,
        "loss": "alpha(e)*L_forecast+(1-alpha(e))*(L_Q+lambda_stability*L_stability)",
        "coordinate_embedding_dim": STRATEGIES[strategy]["coordinate_dim"],
        "curriculum": STRATEGIES[strategy]["curriculum"],
        "val_mse": best.get("mse"),
        "val_lq": best.get("soft_pseudo_edge_loss"),
        "observable_val_score": best.get("observable_val_score"),
        "test_mse": test.get("mse"),
        "test_pressure_mse": test.get("pressure_mse"),
        "f1": undirected.get("f1"),
        "precision": undirected.get("precision"),
        "recall": undirected.get("recall"),
        "tp": undirected.get("tp"),
        "fp": undirected.get("fp"),
        "fn": undirected.get("fn"),
        "num_edges": learned.get("num_edges"),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["water_base", "water_bridges"])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha_start", type=float, default=0.05)
    parser.add_argument("--alpha_end", type=float, default=0.40)
    parser.add_argument("--alpha_hold_epochs", type=int, default=0)
    parser.add_argument("--alpha_ramp_epochs", type=int, default=0)
    parser.add_argument("--lambda_stability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=Path, default=PROJECT_ROOT / "outputs/professor_strategy_grid")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        for strategy in args.strategies:
            config = STRATEGIES[strategy]
            run_name = f"{dataset}_{strategy}_e{args.epochs}_seed{args.seed}"
            out_dir = args.out_root / run_name
            cmd = [
                sys.executable, "scripts/experiments/run_train_synthetic.py",
                "--data_dir", str(PROJECT_ROOT / DATASETS[dataset]),
                "--out_dir", str(out_dir), "--run_name", run_name,
                "--max_epochs", str(args.epochs), "--check_val_every_n_epoch", "2",
                "--batch_size", str(args.batch_size), "--seed", str(args.seed),
                "--k", "2", "--lambda_graph", "0", "--lambda_stability", str(args.lambda_stability),
                "--use_complementary_loss", "--alpha_forecast", str(args.alpha_end),
                "--lambda_soft_pseudo_edge", "1", "--pseudo_score_mode", "geo_global_dynamic_proxy",
                "--pseudo_use_static_target", "--pseudo_topk", "2", "--pseudo_include_mst",
                "--select_best_by", "observable_val", "--assign_candidate_mask",
                "--candidate_mode", "q_dynamic", "--candidate_k", "2",
                "--static_top_edges", "0", "--static_row_k", "3",
                "--static_symmetry", "mutual", "--static_include_mst",
                "--export_soft_graph", "--save_soft_score",
                "--coordinate_embedding_dim", str(config["coordinate_dim"]),
            ]
            if config["curriculum"]:
                ramp_epochs = args.alpha_ramp_epochs or args.epochs
                cmd.extend([
                    "--alpha_forecast_start", str(args.alpha_start),
                    "--alpha_forecast_end", str(args.alpha_end),
                    "--alpha_hold_epochs", str(args.alpha_hold_epochs),
                    "--alpha_schedule_epochs", str(ramp_epochs),
                ])
            print("\n[RUN]", dataset, strategy)
            print("Loss:", "alpha(e)*L_forecast + (1-alpha(e))*(L_Q + lambda_stability*L_stability)")
            print("Settings:", {
                "alpha_start": args.alpha_start if config["curriculum"] else args.alpha_end,
                "alpha_end": args.alpha_end,
                "alpha_hold_epochs": args.alpha_hold_epochs if config["curriculum"] else 0,
                "alpha_ramp_epochs": (args.alpha_ramp_epochs or args.epochs) if config["curriculum"] else 0,
                "coordinate_embedding_dim": config["coordinate_dim"],
                "lambda_stability": args.lambda_stability,
            })
            print("Command:", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
                rows.append(collect(dataset, strategy, out_dir))
                with (args.out_root / "strategy_results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    if rows:
        print("\nRESULTS")
        for row in rows:
            print(
                f"{row['dataset']:14s} {row['strategy']:24s} "
                f"F1={float(row['f1']):.4f} test_MSE={float(row['test_mse']):.4f} "
                f"E={row['num_edges']}"
            )


if __name__ == "__main__":
    main()
