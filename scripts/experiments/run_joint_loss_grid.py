"""Run structured GT-free joint-loss ablations for synthetic water benchmarks.

Training uses only observable quantities:

    L_train = alpha * L_task + (1-alpha) * lambda_Q * L_Q + lambda_stability * L_stability

Ground-truth edges are read only after each run is fixed, to report topology
metrics and the post-hoc Eval_gamma score.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    "reverse_base_211": "data/synthetic/reverse_base_s211",
    "reverse_bridges_211": "data/synthetic/reverse_bridges_s211",
}

PRESETS = {
    "smoke": {
        "priors": ["observable_geometry_plus", "observable_expert_full"],
        "alphas": [0.5],
        "lambda_qs": [1.0],
        "lambda_stabilities": [0.0, 0.1],
        "task_losses": ["mse", "mae", "huber"],
        "modules": [("gcn", "gcn")],
    },
    "task": {
        "priors": ["observable_expert_full"],
        "alphas": [0.5],
        "lambda_qs": [1.0],
        "lambda_stabilities": [0.1],
        "task_losses": ["mse", "mae", "huber"],
        "modules": [("gcn", "gcn")],
    },
    "weights": {
        "priors": ["observable_expert_full"],
        "alphas": [0.2, 0.5, 0.8],
        "lambda_qs": [0.5, 1.0, 2.0],
        "lambda_stabilities": [0.1],
        "task_losses": ["huber"],
        "modules": [("gcn", "gcn")],
    },
    "topo": {
        "priors": ["observable_geometry", "observable_geometry_plus", "observable_expert_pressure", "observable_expert_dynamic", "observable_expert_full"],
        "alphas": [0.5],
        "lambda_qs": [1.0],
        "lambda_stabilities": [0.1],
        "task_losses": ["huber"],
        "modules": [("gcn", "gcn")],
    },
    "loss": {
        "priors": ["observable_geometry", "observable_geometry_plus", "observable_expert_pressure", "observable_expert_dynamic", "observable_expert_full"],
        "alphas": [0.2, 0.5, 0.8],
        "lambda_qs": [0.5, 1.0, 2.0],
        "lambda_stabilities": [0.1],
        "task_losses": ["mse", "mae", "huber"],
        "modules": [("gcn", "gcn")],
    },
    "stability": {
        "priors": ["observable_expert_full"],
        "alphas": [0.8],
        "lambda_qs": [0.5],
        "lambda_stabilities": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        "task_losses": ["huber"],
        "modules": [("gcn", "gcn")],
    },
    "modules": {
        "priors": ["observable_expert_full"],
        "alphas": [0.8],
        "lambda_qs": [0.5],
        "lambda_stabilities": [0.0],
        "task_losses": ["huber"],
        "modules": [("gcn", "gcn"), ("mlp", "gcn"), ("gat", "gat"), ("gcn1d", "gcn1d")],
    },
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("\n[CMD]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_one(args: argparse.Namespace, dataset_name: str, prior: str, alpha: float, lambda_q: float,
            lambda_stability: float, task_loss: str, ffun: str, gfun: str) -> dict:
    data_dir = PROJECT_ROOT / DATASETS[dataset_name]
    module_name = f"{ffun}-{gfun}"
    run_name = (
        f"{dataset_name}_{prior}_a{fmt_float(alpha)}_lq{fmt_float(lambda_q)}_"
        f"stab{fmt_float(lambda_stability)}_{task_loss}_{module_name}_e{args.epochs}"
    )
    out_dir = args.out_root / run_name
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--run_name", run_name,
        "--num_nodes", str(args.num_nodes),
        "--lags", str(args.lags),
        "--prediction_window", str(args.prediction_window),
        "--max_epochs", str(args.epochs),
        "--check_val_every_n_epoch", str(args.check_every),
        "--batch_size", str(args.batch_size),
        "--k", str(args.k),
        "--ffun", ffun,
        "--gfun", gfun,
        "--lambda_graph", "0.0",
        "--use_complementary_loss",
        "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", str(lambda_q),
        "--lambda_stability", str(lambda_stability),
        "--pseudo_score_mode", prior,
        "--pseudo_topk", str(args.k),
        "--pseudo_include_mst",
        "--select_best_by", "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode", "q_dynamic",
        "--candidate_k", str(args.candidate_k),
        "--static_top_edges", "0",
        "--static_row_k", str(args.static_row_k),
        "--static_symmetry", args.static_symmetry,
        "--static_include_mst",
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
        "--task_loss_type", task_loss,
        "--task_feature_combine", args.task_feature_combine,
        "--task_feature_normalization", args.task_feature_normalization,
        "--pressure_weight", str(args.pressure_weight),
        "--demand_weight", str(args.demand_weight),
        "--seed", str(args.seed),
    ]
    if task_loss == "huber":
        cmd += ["--task_huber_delta", str(args.task_huber_delta)]
    if args.coordinate_embedding_dim > 0:
        cmd += ["--coordinate_embedding_dim", str(args.coordinate_embedding_dim)]
    run_cmd(cmd, args.dry_run)
    if args.dry_run:
        return {
            "run": run_name, "dataset": dataset_name, "prior": prior,
            "alpha": alpha, "lambda_Q": lambda_q, "lambda_stability": lambda_stability,
            "task_loss": task_loss, "ffun": ffun, "gfun": gfun,
            "val_mse": 0.0, "val_LQ": 0.0, "val_stability": 0.0,
            "val_topology": 0.0, "J_obs": 0.0, "test_mse": 0.0, "test_mae": 0.0,
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "components": 0,
            "out_dir": str(out_dir),
        }
    return collect_run(out_dir, dataset_name, prior, alpha, lambda_q, lambda_stability, task_loss, ffun, gfun)


def collect_run(run_dir: Path, dataset_name: str | None = None, prior: str | None = None,
                alpha: float | None = None, lambda_q: float | None = None,
                lambda_stability: float | None = None, task_loss: str | None = None,
                ffun: str | None = None, gfun: str | None = None) -> dict:
    forecast = read_json(run_dir / "forecast_metrics.json")
    comparison = read_json(run_dir / "graph_comparison_softprob" / "comparison_summary.json")
    val = forecast.get("best_observable_validation") or forecast.get("best_validation") or forecast.get("last_validation") or {}
    test = forecast.get("test") or {}
    undir = comparison.get("edge_recovery_undirected", {})
    learned_stats = comparison.get("learned_undirected", {})
    val_mse = float(val.get("mse", 0.0))
    val_lq = float(val.get("soft_pseudo_edge_loss", val.get("topology_loss", 0.0)))
    val_stab = float(val.get("stability_loss", 0.0))
    a = float(alpha if alpha is not None else 0.0)
    lq = float(lambda_q if lambda_q is not None else 1.0)
    ls = float(lambda_stability if lambda_stability is not None else 0.0)
    val_topology = lq * val_lq
    j_obs = a * val_mse + (1.0 - a) * val_topology + ls * val_stab
    return {
        "run": run_dir.name,
        "dataset": dataset_name or "",
        "prior": prior or "",
        "alpha": a,
        "lambda_Q": lq,
        "lambda_stability": ls,
        "task_loss": task_loss or "",
        "ffun": ffun or "",
        "gfun": gfun or "",
        "val_mse": val_mse,
        "val_LQ": val_lq,
        "val_stability": val_stab,
        "val_topology": val_topology,
        "val_pressure_mse": float(val.get("pressure_mse", 0.0)),
        "val_demand_mse": float(val.get("demand_mse", 0.0)),
        "J_obs": j_obs,
        "test_mse": float(test.get("mse", 0.0)),
        "test_mae": float(test.get("mae", 0.0)),
        "test_pressure_mse": float(test.get("pressure_mse", 0.0)),
        "test_demand_mse": float(test.get("demand_mse", 0.0)),
        "f1": float(undir.get("f1", 0.0)),
        "precision": float(undir.get("precision", 0.0)),
        "recall": float(undir.get("recall", 0.0)),
        "tp": int(undir.get("tp", 0)),
        "fp": int(undir.get("fp", 0)),
        "fn": int(undir.get("fn", 0)),
        "components": int(learned_stats.get("num_connected_components_undirected", 0)),
        "out_dir": str(run_dir),
    }


def add_eval_gamma(rows: list[dict], gamma: float, task_baseline_mse: float | None) -> None:
    """Simple compact post-hoc score.

    TaskScore uses the same task error family reported by the model: lower is
    better, mapped to [0, 1] with 1/(1+error). TopologyScore is F1. gamma=0.5
    means equal interest in forecast and topology.
    """
    for row in rows:
        task_error = max(0.0, float(row.get("test_mse", 0.0)))
        row["task_score_norm"] = 1.0 / (1.0 + task_error)
        row["task_norm_source"] = "1/(1+task_error)"
        row["topology_score_norm"] = float(row.get("f1", 0.0))
        row["Eval_gamma"] = gamma * row["task_score_norm"] + (1.0 - gamma) * row["topology_score_norm"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict], key: str, reverse: bool, title: str, limit: int) -> None:
    print(f"\n{title}")
    print("| rank | dataset | prior | alpha | lQ | stab | task | module | J_obs | test_mse | F1 | Eval |")
    print("| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(sorted(rows, key=lambda r: r[key], reverse=reverse)[:limit], start=1):
        module = f"{row['ffun']}/{row['gfun']}"
        print(
            f"| {idx} | {row['dataset']} | {row['prior']} | {row['alpha']:.2f} | "
            f"{row['lambda_Q']:.2f} | {row['lambda_stability']:.2f} | {row['task_loss']} | {module} | "
            f"{row['J_obs']:.4f} | {row['test_mse']:.4f} | {row['f1']:.3f} | {row['Eval_gamma']:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=Path, default=PROJECT_ROOT / "outputs" / "joint_loss_grid")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["water_base", "water_bridges", "water_loopy"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--check_every", type=int, default=3)
    parser.add_argument("--lags", type=int, default=6)
    parser.add_argument("--prediction_window", type=int, default=12)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--candidate_k", type=int, default=3)
    parser.add_argument("--static_row_k", type=int, default=3)
    parser.add_argument("--static_symmetry", choices=["union", "mutual"], default="mutual")
    parser.add_argument("--task_feature_combine", choices=["weighted_sum", "weighted_mean"], default="weighted_mean")
    parser.add_argument("--task_feature_normalization", choices=["none", "target_std", "target_mad"], default="target_std")
    parser.add_argument("--pressure_weight", type=float, default=1.0)
    parser.add_argument("--demand_weight", type=float, default=1.0)
    parser.add_argument("--task_huber_delta", type=float, default=1.0)
    parser.add_argument("--coordinate_embedding_dim", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--task_baseline_mse", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    preset = PRESETS[args.preset]

    if args.collect_only:
        for run_dir in sorted(p for p in args.out_root.iterdir() if p.is_dir()):
            if (run_dir / "forecast_metrics.json").exists():
                rows.append(collect_run(run_dir))
    else:
        combos = itertools.product(
            args.datasets,
            preset["priors"],
            preset["alphas"],
            preset["lambda_qs"],
            preset["lambda_stabilities"],
            preset["task_losses"],
            preset["modules"],
        )
        for dataset, prior, alpha, lambda_q, lambda_stability, task_loss, module in combos:
            ffun, gfun = module
            row = run_one(args, dataset, prior, alpha, lambda_q, lambda_stability, task_loss, ffun, gfun)
            rows.append(row)
            add_eval_gamma(rows, gamma=args.gamma, task_baseline_mse=args.task_baseline_mse)
            write_csv(args.out_root / "grid_results.csv", rows)

    if args.dry_run:
        print(f"\n[OK] dry-run commands: {len(rows)}")
        return
    add_eval_gamma(rows, gamma=args.gamma, task_baseline_mse=args.task_baseline_mse)
    write_csv(args.out_root / "grid_results.csv", rows)
    print(f"\n[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.out_root / 'grid_results.csv'}")
    print_table(rows, key="J_obs", reverse=False, title="Best by observable validation objective (GT-free selection)", limit=args.limit)
    print_table(rows, key="Eval_gamma", reverse=True, title="Best by final Eval_gamma (post-hoc evaluation)", limit=args.limit)


if __name__ == "__main__":
    main()