#!/usr/bin/env python3
"""Materialize the requested root-level artifacts for a final campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmarks.render_campaign_artifacts import render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.campaign_dir / "selected_seed_results.csv").open(encoding="utf-8")))
    representative = min(rows, key=lambda row: float(row["validation_joint"]))
    run_dir = Path(representative["run_dir"])
    render(args.observation_dir, args.evaluation_dir, run_dir)

    copies = {
        args.campaign_dir / "selection_gt_free.json": args.campaign_dir / "config_selection_gt_free.json",
        args.campaign_dir / "selected_seed_results.csv": args.campaign_dir / "seed_results.csv",
        args.campaign_dir / "selected_aggregate.json": args.campaign_dir / "aggregate_metrics.json",
        run_dir / "config.json": args.campaign_dir / "config.json",
        run_dir / "best.pt": args.campaign_dir / "best_checkpoint.pt",
        run_dir / "learned_graph.npz": args.campaign_dir / "learned_graph.npz",
        run_dir / "learned_edges.csv": args.campaign_dir / "learned_edges.csv",
        run_dir / "history.json": args.campaign_dir / "representative_loss_history.json",
        run_dir / "metrics_gt_free.json": args.campaign_dir / "representative_metrics_gt_free.json",
        run_dir / "metrics_posthoc.json": args.campaign_dir / "representative_metrics_posthoc.json",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)

    checkpoints = args.campaign_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    logs = args.campaign_dir / "logs"
    logs.mkdir(exist_ok=True)
    for row in rows:
        seed = int(row["seed"])
        seed_run = Path(row["run_dir"])
        shutil.copy2(seed_run / "best.pt", checkpoints / f"seed_{seed:03d}_best.pt")
        for name in ("history.json", "metrics_gt_free.json", "metrics_posthoc.json"):
            source = seed_run / name
            if source.exists():
                shutil.copy2(source, logs / f"seed_{seed:03d}_{name}")

    figures = args.campaign_dir / "figures"
    figures.mkdir(exist_ok=True)
    for source in (run_dir / "figures").glob("*.png"):
        shutil.copy2(source, figures / source.name)

    metadata = {
        "representative_seed": int(representative["seed"]),
        "selection_rule": "minimum GT-free validation_joint among final seeds",
        "representative_run_dir": str(run_dir),
        "reference_topology_used_for_selection": False,
    }
    (args.campaign_dir / "representative_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
