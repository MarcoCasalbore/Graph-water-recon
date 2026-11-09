#!/usr/bin/env python3
"""Summarize empirical campaign scaling and render N-runtime/memory plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmarks.summarize_screening import topology_size
from src.paper_protocol import ObservableElectricPrior, ObservableWaterPrior, load_observations


RUNTIME_KEYS = (
    "mean_epoch_seconds",
    "training_seconds_total",
    "inference_seconds_all_training_windows",
    "consensus_decoding_seconds",
    "peak_gpu_memory_mb",
)


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def runtime_values(campaign_dir: Path) -> dict[str, tuple[float, float]]:
    selected = list(csv.DictReader((campaign_dir / "selected_seed_results.csv").open(encoding="utf-8")))
    values = {key: [] for key in RUNTIME_KEYS}
    for row in selected:
        run_dir = Path(row["run_dir"])
        metrics = json.loads((run_dir / "metrics_gt_free.json").read_text(encoding="utf-8"))
        runtime = metrics["runtime"]
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        fallback = {
            "mean_epoch_seconds": float(np.mean([item["epoch_seconds"] for item in history])),
            "training_seconds_total": float(sum(item["epoch_seconds"] for item in history)),
        }
        for key in RUNTIME_KEYS:
            if key in runtime:
                values[key].append(float(runtime[key]))
            elif key in fallback:
                values[key].append(fallback[key])
    return {key: mean_std(items) if items else (0.0, 0.0) for key, items in values.items()}


def candidate_pairs(observation_dir: Path, config: dict) -> tuple[int, int]:
    bundle = load_observations(observation_dir)
    train_stop = int(bundle.manifest["splits"]["train"][1])
    if bundle.manifest.get("domain") == "electric":
        prior = ObservableElectricPrior(
            bundle.values[:train_stop],
            bundle.coordinates,
            int(config["candidate_size"]),
            float(config.get("electric_prior_ridge", 0.5)),
        )
    else:
        prior = ObservableWaterPrior(
            bundle.values[:train_stop],
            bundle.coordinates,
            int(config["candidate_size"]),
            float(config.get("prior_ridge", 0.05)),
        )
    undirected = prior.candidate_mask | prior.candidate_mask.T
    unique = int(torch.triu(undirected, diagonal=1).sum().item())
    return int(prior.candidate_mask.sum().item()), unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", nargs=3, action="append", metavar=("LABEL", "OBSERVATIONS", "CAMPAIGN"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label, observation_text, campaign_text in args.entry:
        observation_dir, campaign_dir = Path(observation_text), Path(campaign_text)
        selection = json.loads((campaign_dir / "selection_gt_free.json").read_text(encoding="utf-8"))
        config = selection["selected_config"]
        n, edges = topology_size(observation_dir)
        directed_slots, unique_pairs = candidate_pairs(observation_dir, config)
        row = {
            "Dataset": label,
            "N": n,
            "E": edges,
            "candidate_pairs_unique": unique_pairs,
            "candidate_slots_directed": directed_slots,
            "candidate_size": int(config["candidate_size"]),
        }
        for key, (average, deviation) in runtime_values(campaign_dir).items():
            row[key] = average
            row[f"{key}_std"] = deviation
        rows.append(row)
    rows.sort(key=lambda item: int(item["N"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    n_values = [row["N"] for row in rows]
    axes[0].plot(n_values, [row["mean_epoch_seconds"] for row in rows], "o-")
    axes[0].set(xlabel="Nodes N", ylabel="Seconds", title="Training time per epoch")
    axes[1].plot(n_values, [row["peak_gpu_memory_mb"] for row in rows], "o-")
    axes[1].set(xlabel="Nodes N", ylabel="MiB", title="Peak GPU memory")
    for axis in axes:
        axis.grid(alpha=0.25)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
