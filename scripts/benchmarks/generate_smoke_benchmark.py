"""Create a tiny deterministic fixture; this is not a physical benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate(output: Path, domain: str = "water") -> None:
    rng = np.random.default_rng(7)
    samples, nodes = 240, 6
    demand = rng.normal(size=(samples, nodes)).astype(np.float32)
    pressure = np.cumsum(0.1 * demand, axis=0) + np.sin(np.arange(samples)[:, None] / 12)
    observations, evaluation = output / "observations", output / "evaluation_only"
    observations.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(observations / "data.npz", pressure=pressure.astype(np.float32),
                        demand=demand, node_xy=rng.uniform(size=(nodes, 2)).astype(np.float32))
    ids = [f"node_{i}" for i in range(nodes)]
    manifest = {
        "schema_version": 1, "dataset": "smoke_fixture", "domain": domain,
        "provenance": "Deterministic software fixture; not a hydraulic or electric simulation",
        "observation_file": "data.npz", "variables": ["pressure", "demand"],
        "node_ids": ids, "num_nodes": nodes, "num_samples": samples,
        "splits": {"train": [0, 180], "val": [180, 204], "test": [204, 240]},
        "window": {"lags": 6, "forecast_horizon": 12, "cross_boundary_windows": False},
    }
    (observations / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(evaluation / "reference_topology.npz", node_ids=np.asarray(ids),
                        edge_index=np.asarray([np.arange(nodes - 1), np.arange(1, nodes)]))
    (evaluation / "manifest.json").write_text(json.dumps({
        "dataset": "smoke_fixture", "reference_file": "reference_topology.npz",
        "num_nodes": nodes, "num_unique_undirected_edges": nodes - 1,
        "policy": "post-hoc evaluation only",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke/data"))
    parser.add_argument("--domain", choices=["water", "electric"], default="water")
    args = parser.parse_args()
    generate(args.output_dir, args.domain)
