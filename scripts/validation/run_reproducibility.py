"""Run additional Section V seeds and parse printed loss components to CSV."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "results_section_v/raw"

EPOCH_RE = re.compile(r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+).*alpha_forecast=(?P<alpha>[0-9.]+)")
TRAIN_RE = re.compile(
    r"Train - MSE: (?P<mse>[0-9.eE+-]+).*P-MSE: (?P<pressure>[0-9.eE+-]+).*"
    r"D-MSE: (?P<demand>[0-9.eE+-]+).*GraphLoss: (?P<graph>[0-9.eE+-]+).*"
    r"Stability: (?P<stability>[0-9.eE+-]+).*L_Q: (?P<lq>[0-9.eE+-]+).*"
    r"L_topology: (?P<topology>[0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"Val\s+- MSE: (?P<mse>[0-9.eE+-]+).*P-MSE: (?P<pressure>[0-9.eE+-]+).*"
    r"D-MSE: (?P<demand>[0-9.eE+-]+).*L_Q: (?P<lq>[0-9.eE+-]+).*"
    r"L_topology: (?P<topology>[0-9.eE+-]+)"
)
OBJECTIVE_RE = re.compile(r"ValObjective - J_obs: (?P<objective>[0-9.eE+-]+)")


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            current = {"epoch": int(match["epoch"]), "alpha": float(match["alpha"])}
            rows.append(current)
            continue
        if current is None:
            continue
        match = TRAIN_RE.search(line)
        if match:
            current.update({f"train_{key}": float(value) for key, value in match.groupdict().items()})
            continue
        match = VAL_RE.search(line)
        if match:
            current.update({f"val_{key}": float(value) for key, value in match.groupdict().items()})
            continue
        match = OBJECTIVE_RE.search(line)
        if match:
            current["val_observable_objective"] = float(match["objective"])
    return rows


def command(seed: int, epochs: int, out_dir: Path) -> list[str]:
    run_name = f"section_v_main_seed{seed}_e{epochs}"
    return [
        sys.executable, "scripts/experiments/run_train_synthetic.py",
        "--data_dir", "data/synthetic/water_branch_n50_base",
        "--out_dir", str(out_dir), "--run_name", run_name,
        "--num_nodes", "50", "--max_epochs", str(epochs),
        "--check_val_every_n_epoch", "5", "--batch_size", "32", "--k", "1",
        "--static_top_edges", "0", "--static_row_k", "3",
        "--static_symmetry", "mutual", "--static_include_mst",
        "--use_complementary_loss", "--alpha_forecast", "0.70",
        "--lambda_graph", "0.0", "--lambda_soft_pseudo_edge", "1.0",
        "--lambda_stability", "1.0", "--pseudo_score_mode", "geo_global_dynamic_proxy",
        "--pseudo_use_static_target", "--pseudo_topk", "1", "--pseudo_include_mst",
        "--select_best_by", "observable_val", "--assign_candidate_mask",
        "--candidate_mode", "q_dynamic", "--candidate_k", "2",
        "--ensure_connected_consensus", "--export_soft_graph", "--save_soft_score",
        "--seed", str(seed),
    ]


def run_seed(seed: int, epochs: int, force: bool) -> None:
    out_dir = RAW_ROOT / f"section_v_main_seed{seed}_e{epochs}"
    marker = out_dir / "forecast_metrics.json"
    if marker.exists() and not force:
        print(f"[SKIP] completed output exists: {out_dir}")
        return
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise RuntimeError(f"Refusing to overwrite incomplete output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.log"
    cmd = command(seed, epochs, out_dir)
    print("[RUN]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)

    rows = parse_log(log_path)
    if not rows:
        raise RuntimeError(f"No epoch metrics parsed from {log_path}")
    columns = sorted({key for row in rows for key in row}, key=lambda key: (key != "epoch", key))
    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    print(f"[OK] seed={seed}: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        run_seed(seed, args.epochs, args.force)


if __name__ == "__main__":
    main()
