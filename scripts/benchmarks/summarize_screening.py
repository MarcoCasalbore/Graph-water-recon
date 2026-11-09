#!/usr/bin/env python3
"""Build a ranked screening table from completed, GT-free-selected campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


METRICS = ("precision", "accuracy", "recall", "f1", "test_mse", "test_mae", "S_0.5")


def scalar(aggregate: dict, key: str, field: str = "mean") -> float:
    return float(aggregate.get(key, {}).get(field, 0.0))


def topology_size(observation_dir: Path) -> tuple[int, int]:
    manifest = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_dir = observation_dir.parent / "evaluation_only"
    evaluation_manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    if "num_unique_undirected_edges" in evaluation_manifest:
        edges = int(evaluation_manifest["num_unique_undirected_edges"])
    else:
        reference = np.load(evaluation_dir / "reference_topology.npz")
        pairs = {tuple(sorted((int(u), int(v)))) for u, v in reference["edge_index"].T if int(u) != int(v)}
        edges = len(pairs)
    return int(manifest["num_nodes"]), edges


def training_times(campaign_dir: Path) -> list[float]:
    rows = list(csv.DictReader((campaign_dir / "selected_seed_results.csv").open(encoding="utf-8")))
    values = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        metrics = json.loads((run_dir / "metrics_gt_free.json").read_text(encoding="utf-8"))
        runtime = metrics.get("runtime", {})
        if "training_seconds_total" in runtime:
            values.append(float(runtime["training_seconds_total"]))
        else:
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            values.append(sum(float(item["epoch_seconds"]) for item in history))
    return values


def campaign_row(name: str, observation_dir: Path, campaign_dir: Path, note: str) -> dict[str, object]:
    manifest = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads((campaign_dir / "selected_aggregate.json").read_text(encoding="utf-8"))
    selection = json.loads((campaign_dir / "selection_gt_free.json").read_text(encoding="utf-8"))
    n, edges = topology_size(observation_dir)
    times = training_times(campaign_dir)
    row: dict[str, object] = {
        "Network": name,
        "N": n,
        "E": edges,
        "provenance": manifest.get("provenance", ""),
        "candidate_pairs": n * int(selection["selected_config"]["candidate_size"]),
        "candidate_size": int(selection["selected_config"]["candidate_size"]),
        "runtime_seconds": mean(times),
        "runtime_std": stdev(times) if len(times) > 1 else 0.0,
        "note": note,
    }
    for metric in METRICS:
        row[metric] = scalar(aggregate, metric)
        row[f"{metric}_std"] = scalar(aggregate, metric, "std")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", nargs=4, action="append", metavar=("NAME", "OBSERVATIONS", "CAMPAIGN", "NOTE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        campaign_row(name, Path(observations), Path(campaign), note)
        for name, observations, campaign, note in args.entry
    ]
    rows.sort(key=lambda row: float(row["S_0.5"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
