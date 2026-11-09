"""Execute the controlled main, L/H, alpha, and k_dgm campaign with resume support."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
ROOT = RESULTS_ROOT / "campaign_v2"
DATA_DIR = ROOT / "datasets/homogeneous_v2_g211_s1211"
GRAPH_SEED, SIGNAL_SEED = 211, 1211
TRAIN_SEEDS = (42, 43, 44)
MAIN_SEEDS = (42, 43, 44, 45, 46)
LH_LAGS = (6, 12, 24, 48)
LH_HORIZONS = (1, 3, 6, 12)
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 1.0)
K_DGM_VALUES = (1, 2, 3, 4, 5)

EPOCH_RE = re.compile(r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+).*alpha_forecast=(?P<alpha>[0-9.]+)")
TRAIN_RE = re.compile(
    r"Train - MSE: (?P<task>[0-9.eE+-]+).*P-MSE: (?P<pressure>[0-9.eE+-]+).*"
    r"D-MSE: (?P<demand>[0-9.eE+-]+).*GraphLoss: (?P<graph>[0-9.eE+-]+).*"
    r"Stability: (?P<stability>[0-9.eE+-]+).*L_Q: (?P<lq>[0-9.eE+-]+).*"
    r"L_topology: (?P<topology>[0-9.eE+-]+).*Total: (?P<total>[0-9.eE+-]+).*"
    r"WTask: (?P<w_task>[0-9.eE+-]+).*WGraph: (?P<w_graph>[0-9.eE+-]+).*"
    r"WTopo: (?P<w_topology>[0-9.eE+-]+).*WStab: (?P<w_stability>[0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"Val\s+- MSE: (?P<task>[0-9.eE+-]+).*P-MSE: (?P<pressure>[0-9.eE+-]+).*"
    r"D-MSE: (?P<demand>[0-9.eE+-]+).*L_Q: (?P<lq>[0-9.eE+-]+).*"
    r"L_topology: (?P<topology>[0-9.eE+-]+)"
)
OBJECTIVE_RE = re.compile(r"ValObjective - J_obs: (?P<objective>[0-9.eE+-]+)")


def slug_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def parse_history(log_path: Path, out_path: Path) -> None:
    rows, current = [], None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            current = {"epoch": int(match["epoch"]), "alpha": float(match["alpha"])}
            rows.append(current)
            continue
        if current is None:
            continue
        match = TRAIN_RE.search(line)
        if match:
            current.update({f"train_{k}": float(v) for k, v in match.groupdict().items()})
            continue
        match = VAL_RE.search(line)
        if match:
            current.update({f"val_{k}": float(v) for k, v in match.groupdict().items()})
            continue
        match = OBJECTIVE_RE.search(line)
        if match:
            current["val_J_obs"] = float(match["objective"])
    if rows:
        write_rows(out_path, rows)


def joint_command(out, run_id, lags, horizon, alpha, k_dgm, candidate_k, seed, epochs):
    return [
        sys.executable, "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(DATA_DIR), "--out_dir", str(out), "--run_name", run_id,
        "--num_nodes", "50", "--lags", str(lags), "--prediction_window", str(horizon),
        "--max_epochs", str(epochs), "--check_val_every_n_epoch", "5", "--batch_size", "32",
        "--lr", "0.001", "--k", str(k_dgm), "--ffun", "mlp", "--gfun", "gcn",
        "--lambda_graph", "0.0", "--use_complementary_loss", "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", "0.5", "--lambda_stability", "0.0",
        "--pseudo_score_mode", "observable_expert_full", "--pseudo_use_static_target",
        "--pseudo_topk", "2", "--pseudo_include_mst", "--select_best_by", "observable_val",
        "--assign_candidate_mask", "--candidate_mode", "q_dynamic", "--candidate_k", str(candidate_k),
        "--static_top_edges", "0", "--static_row_k", "3", "--static_symmetry", "mutual",
        "--static_include_mst", "--ensure_connected_consensus", "--export_soft_graph", "--save_soft_score",
        "--task_loss_type", "huber", "--task_huber_delta", "1.0",
        "--task_feature_combine", "weighted_mean", "--task_feature_normalization", "target_std",
        "--pressure_weight", "1.0", "--demand_weight", "1.0", "--seed", str(seed),
    ]


def summarize_joint(out: Path, config: dict, elapsed: float) -> dict:
    forecast = json.loads((out / "forecast_metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads((out / "graph_comparison_softprob/comparison_summary.json").read_text(encoding="utf-8"))
    test = forecast["test"]
    rec = comparison["edge_recovery_undirected"]
    graph = comparison["learned_undirected"]
    task_error = float(test["mse"])
    task_score = 1.0 / (1.0 + max(0.0, task_error))
    return {
        **config, "status": "completed", "runtime_seconds": elapsed,
        "best_epoch": forecast["selected_epoch"],
        "test_mse": task_error, "test_rmse": float(test["rmse"]), "test_mae": float(test["mae"]),
        "test_pressure_mse": float(test["pressure_mse"]), "test_demand_mse": float(test["demand_mse"]),
        "precision": float(rec["precision"]), "recall": float(rec["recall"]),
        "f1": float(rec["f1"]), "jaccard": float(rec["jaccard"]),
        "edge_count": int(graph["num_edges"]), "density": float(graph["density"]),
        "mean_degree": float(graph["degree_mean"]),
        "components": int(graph["num_connected_components_undirected"]),
        "task_score": task_score, "eval0p5": 0.5 * task_score + 0.5 * float(rec["f1"]),
        "output_dir": str(out.relative_to(PROJECT_ROOT)),
    }


def run_joint(group, lags, horizon, alpha, k_dgm, candidate_k, seed, epochs, force):
    run_id = (f"{group}_g{GRAPH_SEED}_s{SIGNAL_SEED}_t{seed}_l{lags}_h{horizon}_"
              f"a{slug_float(alpha)}_kd{k_dgm}_kc{candidate_k}_kq2_e{epochs}")
    out = ROOT / group / run_id
    config = {
        "run_id": run_id, "group": group, "graph_seed": GRAPH_SEED, "signal_seed": SIGNAL_SEED,
        "training_seed": seed, "generator": "mst_homogeneous", "nodes": 50,
        "reference_edges": 79, "lags": lags, "horizon": horizon, "alpha": alpha,
        "k_dgm": k_dgm, "k_candidate": candidate_k, "k_decoder": 3, "k_q": 2,
        "epochs": epochs, "selection_criterion": "observable_val", "lambda_graph": 0.0,
        "lambda_Q": 0.5, "lambda_stability": 0.0, "decoder": "row_k3_mutual_mst",
        "consensus": "mean_probability", "task_loss": "huber_target_std_weighted_mean",
    }
    marker = out / "forecast_metrics.json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if force or not marker.exists():
        started = time.perf_counter()
        with (out / "training.log").open("w", encoding="utf-8") as handle:
            subprocess.run(joint_command(out, run_id, lags, horizon, alpha, k_dgm, candidate_k, seed, epochs),
                           cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - started
        (out / "elapsed_seconds.txt").write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    else:
        elapsed = float((out / "elapsed_seconds.txt").read_text()) if (out / "elapsed_seconds.txt").exists() else 0.0
    parse_history(out / "training.log", out / "training_history.csv")
    return summarize_joint(out, config, elapsed)


def run_node(lags, horizon, seed, epochs, force):
    run_id = f"node_g{GRAPH_SEED}_s{SIGNAL_SEED}_t{seed}_l{lags}_h{horizon}_e{epochs}"
    out = ROOT / "node_only" / run_id
    marker = out / "forecast_baseline_metrics.json"
    out.mkdir(parents=True, exist_ok=True)
    if force or not marker.exists():
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/validation/train_node_only_campaign.py"),
               "--data_dir", str(DATA_DIR), "--out_dir", str(out), "--lags", str(lags),
               "--prediction_window", str(horizon), "--epochs", str(epochs), "--batch_size", "32",
               "--hidden", "64", "--loss_type", "huber", "--feature_normalization", "target_std",
               "--seed", str(seed)]
        started = time.perf_counter()
        with (out / "training.log").open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - started
        (out / "elapsed_seconds.txt").write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    else:
        elapsed = float((out / "elapsed_seconds.txt").read_text())
    result = json.loads(marker.read_text(encoding="utf-8")); test = result["test"]["normalized"]
    return {"run_id": run_id, "group": "node_only", "graph_seed": GRAPH_SEED,
            "signal_seed": SIGNAL_SEED, "training_seed": seed, "lags": lags, "horizon": horizon,
            "epochs": epochs, "best_epoch": result["best_epoch"], "parameter_count": result["parameter_count"],
            "test_mse": test["mse"], "test_rmse": test["rmse"], "test_mae": test["mae"],
            "test_pressure_mse": test["pressure_mse"], "test_demand_mse": test["demand_mse"],
            "runtime_seconds": elapsed, "status": "completed", "output_dir": str(out.relative_to(PROJECT_ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["main", "lh", "alpha", "k", "all"], default="all")
    parser.add_argument("--epochs_main", type=int, default=300)
    parser.add_argument("--epochs_sweep", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    groups = {"main", "lh", "alpha", "k"} if args.group == "all" else {args.group}
    manifest = []
    if "main" in groups:
        rows = []
        for seed in MAIN_SEEDS:
            print(f"[MAIN] seed={seed}", flush=True)
            rows.append(run_joint("main_experiment", 6, 12, 0.8, 2, 3, seed, args.epochs_main, args.force))
            write_rows(ROOT / "raw_metrics/main_runs.csv", rows)
        manifest.extend(rows)
    if "lh" in groups:
        joint_rows, node_rows = [], []
        for lags in LH_LAGS:
            for horizon in LH_HORIZONS:
                for seed in TRAIN_SEEDS:
                    print(f"[LH] L={lags} H={horizon} seed={seed}", flush=True)
                    joint_rows.append(run_joint("lh_sweep", lags, horizon, 0.8, 2, 3, seed, args.epochs_sweep, args.force))
                    node_rows.append(run_node(lags, horizon, seed, args.epochs_sweep, args.force))
                    write_rows(ROOT / "raw_metrics/lh_joint_runs.csv", joint_rows)
                    write_rows(ROOT / "raw_metrics/lh_node_runs.csv", node_rows)
        manifest.extend(joint_rows + node_rows)
    if "alpha" in groups:
        rows = []
        for alpha in ALPHAS:
            for seed in TRAIN_SEEDS:
                print(f"[ALPHA] alpha={alpha} seed={seed}", flush=True)
                rows.append(run_joint("alpha_sweep", 6, 12, alpha, 2, 3, seed, args.epochs_sweep, args.force))
                write_rows(ROOT / "raw_metrics/alpha_runs.csv", rows)
        manifest.extend(rows)
    if "k" in groups:
        rows = []
        for k_dgm in K_DGM_VALUES:
            for seed in TRAIN_SEEDS:
                print(f"[K] k_dgm={k_dgm} seed={seed}", flush=True)
                rows.append(run_joint("k_dgm_sweep", 6, 12, 0.8, k_dgm, 6, seed, args.epochs_sweep, args.force))
                write_rows(ROOT / "raw_metrics/k_dgm_runs.csv", rows)
        manifest.extend(rows)
    write_rows(ROOT / "experiment_manifest_latest.csv", manifest)
    print(f"[OK] completed_or_reused={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
