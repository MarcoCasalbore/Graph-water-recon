#!/usr/bin/env python3
"""Resumable GT-free grid/final runner; reference evaluation is a separate phase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_protocol import evaluate_reference, train_run


DEFAULTS: dict[str, Any] = {
    "alpha": 0.7,
    "candidate_size": 5,
    "k": 2,
    "learning_rate": 1e-3,
    "embedding_dim": 4,
    "hidden_dim": 32,
    "message_layers": 2,
    "batch_size": 16,
    "epochs": 40,
    "coordinate_dim": 4,
    "prior_ridge": 0.05,
    "weight_decay": 0.0,
    "gradient_clip": 5.0,
    "consensus_target_degree": 4.0,
    "decoder_row_k": 2,
    "device": "auto",
}


def trial_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def configurations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    base = {**DEFAULTS, **spec.get("base", {})}
    grid = spec.get("grid", {})
    if not grid:
        return [base]
    keys = sorted(grid)
    output = []
    for values in itertools.product(*(grid[key] for key in keys)):
        config = {**base, **dict(zip(keys, values))}
        if int(config["k"]) <= int(config["candidate_size"]):
            output.append(config)
    return output


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start, stop = map(int, text.split(":", 1))
        return list(range(start, stop))
    return [int(value) for value in text.split(",") if value.strip()]


def flatten_trial(config: dict[str, Any], seed: int, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"trial_id": trial_id(config), "seed": seed, "run_dir": str(run_dir)}
    row.update(config)
    row.update({
        "best_epoch": metrics["best_epoch"],
        "validation_joint": metrics["validation_joint"],
        "mean_epoch_seconds": metrics["runtime"]["mean_epoch_seconds"],
        "peak_gpu_memory_mb": metrics["runtime"]["peak_gpu_memory_mb"],
    })
    for key in ("training_seconds_total", "prior_precompute_seconds", "inference_seconds_all_training_windows", "consensus_decoding_seconds"):
        if key in metrics["runtime"]:
            row[key] = metrics["runtime"][key]
    if "forecast" in metrics:
        forecast = metrics["forecast"]
        row.update({
            "test_mse": forecast["mse"],
            "test_mae": forecast["mae"],
            "test_rmse": forecast["rmse"],
        })
        for group, prefix in (("normalized_per_variable", "normalized"), ("physical_units", "physical")):
            for variable, values in forecast.get(group, {}).items():
                for metric, value in values.items():
                    row[f"{prefix}_{variable}_{metric}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_selected(rows: list[dict[str, Any]], selected_id: str, posthoc: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["trial_id"] == selected_id]
    numeric: dict[str, list[float]] = {}
    for row in selected:
        runtime_keys = {
            "validation_joint", "mean_epoch_seconds", "training_seconds_total",
            "prior_precompute_seconds", "inference_seconds_all_training_windows",
            "consensus_decoding_seconds", "peak_gpu_memory_mb",
        }
        for key, value in row.items():
            if key in runtime_keys or key.startswith(("test_", "normalized_", "physical_")):
                if value not in ("", None):
                    numeric.setdefault(key, []).append(float(value))
        result = posthoc.get((selected_id, int(row["seed"])))
        if result:
            for key, value in result["topology"].items():
                numeric.setdefault(key, []).append(float(value))
            for key, value in result["joint"].items():
                numeric.setdefault(key, []).append(float(value))
    return {key: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "n": len(values)} for key, values in numeric.items()}


def run_campaign(args: argparse.Namespace) -> None:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    configs = configurations(spec)
    seeds = parse_seeds(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)) or not configs:
        raise ValueError("Provide nonempty configurations and unique seeds")
    if args.device:
        configs = [{**config, "device": args.device} for config in configs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config in configs:
        identifier = trial_id(config)
        for seed in seeds:
            run_dir = args.output_dir / "trials" / identifier / f"seed_{seed:03d}"
            metrics = train_run(args.observation_dir, run_dir, config, seed, resume=True, finalize=False)
            rows.append(flatten_trial(config, seed, run_dir, metrics))
            write_csv(args.output_dir / "gt_free_trials.csv", rows)

    grouped: dict[str, list[float]] = {}
    config_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        grouped.setdefault(row["trial_id"], []).append(float(row["validation_joint"]))
    for config in configs:
        config_by_id[trial_id(config)] = config
    selected_id = min(grouped, key=lambda key: float(np.mean(grouped[key])))
    selection = {
        "selection_rule": "minimum mean GT-free normalized joint validation objective across tuning seeds",
        "selected_trial_id": selected_id,
        "selected_config": config_by_id[selected_id],
        "validation_joint_mean": float(np.mean(grouped[selected_id])),
        "validation_joint_std": float(np.std(grouped[selected_id], ddof=1)) if len(grouped[selected_id]) > 1 else 0.0,
        "reference_topology_used": False,
    }
    (args.output_dir / "selection_gt_free.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    for row in rows:
        if row["trial_id"] == selected_id:
            run_dir = Path(row["run_dir"])
            finalized = train_run(args.observation_dir, run_dir, config_by_id[selected_id], int(row["seed"]), resume=True, finalize=True)
            row.update(flatten_trial(config_by_id[selected_id], int(row["seed"]), run_dir, finalized))
    write_csv(args.output_dir / "gt_free_trials.csv", rows)

    posthoc: dict[tuple[str, int], dict[str, Any]] = {}
    if args.evaluation_dir:
        for row in rows:
            if row["trial_id"] == selected_id:
                result = evaluate_reference(Path(row["run_dir"]), args.evaluation_dir)
                row.update(result["topology"])
                row.update(result["joint"])
                posthoc[(selected_id, int(row["seed"]))] = result
    write_csv(args.output_dir / "selected_seed_results.csv", [row for row in rows if row["trial_id"] == selected_id])
    aggregate = aggregate_selected(rows, selected_id, posthoc)
    (args.output_dir / "selected_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"configurations": len(configs), "seeds": seeds, **selection, "aggregate": aggregate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="GPU includes AMD with a ROCm PyTorch build")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--evaluation-dir", type=Path, help="Optional separate post-hoc phase; never passed into training")
    run_campaign(parser.parse_args())


if __name__ == "__main__":
    main()
