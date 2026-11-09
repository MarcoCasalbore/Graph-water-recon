"""Replicate the easy and hard temporal corners for seeds 43 and 44."""

from __future__ import annotations

import csv
from pathlib import Path

import run_temporal_balanced_joint as balanced
import run_temporal_explainability as diagnostic


ROOT = diagnostic.RESULTS_ROOT / "raw/temporal_corner_seeds"
CASES = [(24, 1, "easy"), (4, 12, "hard")]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (43, 44):
        for lags, horizon, regime in CASES:
            cell = ROOT / f"{regime}_l{lags}_h{horizon}_seed{seed}"
            node_mse, node_seconds = diagnostic.node_baseline(
                lags, horizon, seed, 30, cell, force=False)
            joint = balanced.run_cell(
                lags, horizon, seed, 16, ROOT / "balanced", force=False)
            gain = (node_mse - joint["test_mse_feature_mean"]) / node_mse
            rows.append({
                "regime": regime, "lags": lags, "horizon": horizon, "seed": seed,
                "node_mlp_mse": node_mse,
                "balanced_joint_mse": joint["test_mse_feature_mean"],
                "joint_gain_vs_node": gain,
                "f1": joint["f1"], "task_score": joint["task_score"],
                "eval_0p5": joint["eval_0p5"], "node_seconds": node_seconds,
                "joint_seconds": joint["elapsed_seconds"],
            })
            with (ROOT / "corner_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
