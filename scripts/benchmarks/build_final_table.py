#!/usr/bin/env python3
"""Build the compact mean ± std table requested for the paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = {
    "Precision": "precision",
    "Accuracy": "accuracy",
    "Recall": "recall",
    "F1": "f1",
    "MSE": "test_mse",
    "MAE": "test_mae",
    "S_0.5": "S_0.5",
}


def topology_size(observation_dir: Path) -> tuple[int, int]:
    observation = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_dir = observation_dir.parent / "evaluation_only"
    evaluation = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    if "num_unique_undirected_edges" in evaluation:
        edges = int(evaluation["num_unique_undirected_edges"])
    elif "num_internal_edges" in evaluation:
        edges = int(evaluation["num_internal_edges"])
    else:
        reference = np.load(evaluation_dir / "reference_topology.npz")
        pairs = {
            tuple(sorted((int(u), int(v))))
            for u, v in reference["edge_index"].T
            if int(u) != int(v)
        }
        edges = len(pairs)
    return int(observation["num_nodes"]), edges


def make_row(domain: str, observation_dir: Path, campaign_dir: Path) -> dict[str, object]:
    manifest = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads((campaign_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    nodes, edges = topology_size(observation_dir)
    row: dict[str, object] = {
        "Domain": domain,
        "Dataset": manifest["dataset"],
        "N": nodes,
        "E": edges,
    }
    for label, key in METRICS.items():
        item = aggregate[key]
        row[label] = f'{float(item["mean"]):.6f} ± {float(item["std"]):.6f}'
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        nargs=3,
        action="append",
        required=True,
        metavar=("DOMAIN", "OBSERVATION_DIR", "CAMPAIGN_DIR"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        make_row(domain, Path(observation_dir), Path(campaign_dir))
        for domain, observation_dir, campaign_dir in args.entry
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
