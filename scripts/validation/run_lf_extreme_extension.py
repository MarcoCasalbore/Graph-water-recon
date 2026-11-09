"""Extend the relational L/F sweep at the four requested boundary settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
sys.path.insert(0, str(PROJECT / "scripts/validation"))

import run_campaign_v2 as campaign  # noqa: E402


DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT = RESULTS / "campaign_v2/relational/lf_extreme_extension"
SPECS = (
    (6, 24), (6, 48), (2, 12), (4, 12),
    (6, 2), (6, 4), (6, 8), (1, 12), (8, 12),
)
SEEDS = (42, 43, 44)


def proposed_metrics(output_dir: Path) -> tuple[float, float, float]:
    forecast = json.loads((output_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    test = forecast["test"]
    legacy = 0.5 * (float(test["pressure_mse"]) + float(test["demand_mse"]))
    predictions = np.load(output_dir / "test_predictions.npz")
    error = predictions["y_pred"].astype(np.float64) - predictions["y_true"].astype(np.float64)
    return legacy, float(np.mean(error**2)), float(np.mean(np.abs(error)))


def main() -> None:
    campaign.DATA_DIR = DATA
    campaign.ROOT = ROOT
    joint_rows, node_rows, comparison = [], [], []
    for lags, horizon in SPECS:
        for seed in SEEDS:
            joint = campaign.run_joint("joint", lags, horizon, 0.8, 2, 3, seed, 120, False)
            node = campaign.run_node(lags, horizon, seed, 120, False)
            legacy, direct_mse, direct_mae = proposed_metrics(PROJECT / joint["output_dir"])
            comparison.append({
                "lags": lags,
                "horizon": horizon,
                "training_seed": seed,
                "node_mlp_mse": node["test_mse"],
                "proposed_legacy_quantity": legacy,
                "proposed_direct_mse": direct_mse,
                "proposed_direct_mae": direct_mae,
            })
            joint_rows.append(joint)
            node_rows.append(node)
            campaign.write_rows(ROOT / "raw_joint_runs.csv", joint_rows)
            campaign.write_rows(ROOT / "raw_node_runs.csv", node_rows)
            pd.DataFrame(comparison).to_csv(ROOT / "comparison_per_seed.csv", index=False)

    per_seed = pd.DataFrame(comparison)
    aggregate = per_seed.groupby(["lags", "horizon"], as_index=False).agg(
        node_mlp_mse_mean=("node_mlp_mse", "mean"),
        node_mlp_mse_std=("node_mlp_mse", "std"),
        proposed_legacy_mean=("proposed_legacy_quantity", "mean"),
        proposed_legacy_std=("proposed_legacy_quantity", "std"),
        proposed_direct_mse_mean=("proposed_direct_mse", "mean"),
        proposed_direct_mse_std=("proposed_direct_mse", "std"),
        proposed_direct_mae_mean=("proposed_direct_mae", "mean"),
        proposed_direct_mae_std=("proposed_direct_mae", "std"),
    )
    aggregate["legacy_gain"] = (
        aggregate.node_mlp_mse_mean - aggregate.proposed_legacy_mean
    ) / aggregate.node_mlp_mse_mean
    aggregate["direct_mse_gain"] = (
        aggregate.node_mlp_mse_mean - aggregate.proposed_direct_mse_mean
    ) / aggregate.node_mlp_mse_mean
    aggregate.to_csv(ROOT / "comparison_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
