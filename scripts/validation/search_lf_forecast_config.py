"""Validation-only search for a forecasting-oriented relational configuration."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
sys.path.insert(0, str(PROJECT / "scripts/validation"))

import run_campaign_v2 as campaign  # noqa: E402


DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT = RESULTS / "campaign_v2/relational/lf_forecast_config_search"
SCREEN_CELLS = ((2, 12), (6, 48))
FINAL_CELLS = ((2, 12), (4, 12), (6, 24), (6, 48))
CONFIGS = (
    {"name": "q3_k2_lr1e3", "k": 2, "candidate": 3, "mask": True, "lr": 1e-3},
    {"name": "q6_k2_lr1e3", "k": 2, "candidate": 6, "mask": True, "lr": 1e-3},
    {"name": "q6_k4_lr1e3", "k": 4, "candidate": 6, "mask": True, "lr": 1e-3},
    {"name": "q8_k6_lr1e3", "k": 6, "candidate": 8, "mask": True, "lr": 1e-3},
    {"name": "full_k2_lr1e3", "k": 2, "candidate": 49, "mask": False, "lr": 1e-3},
    {"name": "full_k4_lr1e3", "k": 4, "candidate": 49, "mask": False, "lr": 1e-3},
    {"name": "full_k6_lr1e3", "k": 6, "candidate": 49, "mask": False, "lr": 1e-3},
    {"name": "q3_k2_lr3e4", "k": 2, "candidate": 3, "mask": True, "lr": 3e-4},
)


def set_arg(command: list[str], name: str, value: object) -> None:
    command[command.index(name) + 1] = str(value)


def run(config: dict, lags: int, horizon: int, seed: int, epochs: int, stage: str) -> dict:
    name = f"{stage}_{config['name']}_t{seed}_l{lags}_f{horizon}_e{epochs}"
    output = ROOT / stage / config["name"] / name
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "forecast_metrics.json"
    if not marker.exists():
        command = campaign.joint_command(
            output, name, lags, horizon, 1.0, config["k"], config["candidate"], seed, epochs
        )
        set_arg(command, "--data_dir", DATA)
        set_arg(command, "--task_loss_type", "mse")
        set_arg(command, "--task_feature_normalization", "none")
        set_arg(command, "--lr", config["lr"])
        if not config["mask"]:
            command.remove("--assign_candidate_mask")
        started = time.perf_counter()
        with (output / "training.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        runtime = time.perf_counter() - started
        (output / "elapsed_seconds.txt").write_text(f"{runtime:.9f}\n", encoding="utf-8")
    else:
        runtime = float((output / "elapsed_seconds.txt").read_text())

    forecast = json.loads(marker.read_text(encoding="utf-8"))
    validation = forecast["best_observable_validation"]
    predictions = np.load(output / "test_predictions.npz")
    error = predictions["y_pred"].astype(np.float64) - predictions["y_true"].astype(np.float64)
    return {
        **config,
        "stage": stage,
        "lags": lags,
        "horizon": horizon,
        "seed": seed,
        "epochs": epochs,
        "selected_epoch": forecast["selected_epoch"],
        "validation_mse": float(validation["mse"]),
        "test_mse": float(np.mean(error**2)),
        "test_mae": float(np.mean(np.abs(error))),
        "runtime_seconds": runtime,
        "output_dir": str(output.relative_to(PROJECT)),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    screen_rows = []
    for config in CONFIGS:
        for lags, horizon in SCREEN_CELLS:
            screen_rows.append(run(config, lags, horizon, 41, 160, "screen"))
            pd.DataFrame(screen_rows).to_csv(ROOT / "screen_per_run.csv", index=False)

    screen = pd.DataFrame(screen_rows)
    ranking = screen.groupby("name", as_index=False).agg(
        validation_mse_mean=("validation_mse", "mean"),
        validation_mse_max=("validation_mse", "max"),
        runtime_seconds=("runtime_seconds", "sum"),
    ).sort_values(["validation_mse_mean", "validation_mse_max"])
    ranking.to_csv(ROOT / "screen_validation_ranking.csv", index=False)
    best_name = str(ranking.iloc[0]["name"])
    best = next(config for config in CONFIGS if config["name"] == best_name)
    (ROOT / "selected_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    final_rows = []
    for lags, horizon in FINAL_CELLS:
        for seed in (42, 43, 44):
            final_rows.append(run(best, lags, horizon, seed, 160, "final"))
            pd.DataFrame(final_rows).to_csv(ROOT / "final_per_seed.csv", index=False)
    final = pd.DataFrame(final_rows)
    aggregate = final.groupby(["lags", "horizon"], as_index=False).agg(
        proposed_mse_mean=("test_mse", "mean"),
        proposed_mse_std=("test_mse", "std"),
        proposed_mae_mean=("test_mae", "mean"),
        proposed_mae_std=("test_mae", "std"),
        validation_mse_mean=("validation_mse", "mean"),
    )
    aggregate.to_csv(ROOT / "final_aggregate.csv", index=False)
    print(ranking.to_string(index=False))
    print(json.dumps(best, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
