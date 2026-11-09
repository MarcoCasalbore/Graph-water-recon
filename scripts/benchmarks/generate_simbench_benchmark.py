"""Export an unmodified SimBench network using the paper-protocol contract.

This adapter intentionally reuses the established pandapower/SimBench helpers.
The observation package contains no connectivity; the reference graph is
written to ``evaluation_only`` and is consumed only by post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import simbench as sb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.electric.generate_simbench_electric import (
    active_topology_edges,
    bus_lookup,
    parse_bus_xy,
    simulate_timeseries,
)

NETWORKS = {
    "simbench_lv_semiurb4": "1-LV-semiurb4--0-sw",
    "simbench_lv_urban6": "1-LV-urban6--0-sw",
    "simbench_lv_rural2": "1-LV-rural2--0-sw",
    "simbench_mv_rural": "1-MV-rural--0-sw",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmarks/electric"))
    parser.add_argument("--samples", type=int, default=2880)
    parser.add_argument("--profile-start", type=int, default=0)
    parser.add_argument("--profile-stride", type=int, default=1)
    args = parser.parse_args()

    code = NETWORKS[args.network]
    net = sb.get_simbench_net(code)
    lookup = bus_lookup(net)
    coordinates = parse_bus_xy(net)
    edges = sorted(active_topology_edges(net, lookup))
    series = simulate_timeseries(
        net,
        timesteps=args.samples,
        stride=args.profile_stride,
        start=args.profile_start,
        lookup=lookup,
    )
    if not bool(series["powerflow_converged"].all()):
        raise RuntimeError("At least one SimBench power-flow solve did not converge")

    output = PROJECT_ROOT / args.output_root / args.network
    observations = output / "observations"
    evaluation = output / "evaluation_only"
    observations.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    step_seconds = 900 * args.profile_stride
    np.savez_compressed(
        observations / "data.npz",
        pressure=series["voltage_pu"].astype(np.float32),
        demand=series["net_p_mw"].astype(np.float32),
        node_xy=coordinates.astype(np.float32),
        time_s=np.arange(args.samples, dtype=np.int64) * step_seconds,
    )
    node_ids = np.asarray([str(value) for value in net.bus.index.tolist()])
    link_ids = np.asarray([f"edge_{index:04d}" for index in range(len(edges))])
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    np.savez_compressed(
        evaluation / "reference_topology.npz",
        edge_index=edge_index,
        node_ids=node_ids,
        link_ids=link_ids,
    )

    train_end = int(0.75 * args.samples)
    val_end = train_end + int(0.10 * args.samples)
    manifest = {
        "schema_version": 1,
        "dataset": args.network,
        "domain": "electric",
        "provenance": "SimBench representative synthetic benchmark; unmodified topology",
        "simbench_code": code,
        "observation_file": "data.npz",
        "variables": ["pressure", "demand"],
        "physical_variable_names": ["voltage_magnitude_pu", "p_net_mw"],
        "node_ids": node_ids.tolist(),
        "num_nodes": int(len(node_ids)),
        "num_samples": int(args.samples),
        "step_seconds": int(step_seconds),
        "splits": {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, args.samples],
        },
        "window": {"lags": 6, "forecast_horizon": 12, "cross_boundary_windows": False},
        "normalization_fit_split": "train",
        "candidate_statistics_fit_split": "train",
        "simulation": {
            "engine": "pandapower",
            "profile_start": int(args.profile_start),
            "profile_stride": int(args.profile_stride),
            "converged_samples": int(series["powerflow_converged"].sum()),
        },
        "reference_topology_exposed": False,
    }
    manifest_path = observations / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    evaluation_manifest = {
        "purpose": "post-hoc topology metrics only",
        "reference_file": "reference_topology.npz",
        "num_nodes": int(len(node_ids)),
        "num_unique_undirected_edges": int(len(edges)),
        "all_pairs_without_self_loops": int(len(node_ids) * (len(node_ids) - 1) // 2),
        "observation_manifest_sha256": sha256(manifest_path),
    }
    (evaluation / "manifest.json").write_text(json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": args.network, "N": len(node_ids), "E": len(edges), "T": args.samples}, indent=2))


if __name__ == "__main__":
    main()
