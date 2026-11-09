#!/usr/bin/env python3
"""Generate leakage-safe WDN observations from a public EPANET model.

The observation manifest deliberately has no path to the reference topology.
The reference graph is written to a separate evaluation-only directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_inp(source: Path, destination: Path) -> dict[str, object]:
    """Make a parser-safe copy without altering nodes or links."""
    raw = source.read_bytes()
    nul_count = raw.count(b"\x00")
    text = raw.replace(b"\x00", b"").decode("latin-1")
    lines = text.splitlines()

    node_ids: set[str] = set()
    section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.upper()
            continue
        if section in {"[JUNCTIONS]", "[RESERVOIRS]", "[TANKS]"} and stripped and not stripped.startswith(";"):
            node_ids.add(stripped.split()[0])

    dropped_coordinates: list[str] = []
    removed_default_patterns: list[str] = []
    out: list[str] = []
    section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.upper()
            out.append(line)
            continue
        if section == "[COORDINATES]" and stripped and not stripped.startswith(";"):
            node_id = stripped.split()[0]
            if node_id not in node_ids:
                dropped_coordinates.append(node_id)
                continue
        if section == "[OPTIONS]" and stripped.lower().startswith("pattern "):
            removed_default_patterns.append(stripped)
            out.append(f"; disabled invalid default {stripped}")
            continue
        out.append(line)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n", encoding="latin-1")
    return {
        "removed_nul_bytes": nul_count,
        "dropped_orphan_coordinate_ids": dropped_coordinates,
        "disabled_default_pattern_lines": removed_default_patterns,
        "topology_changed": False,
    }


def demand_pattern(samples: int, step_seconds: int, seed: int, node_index: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, node_index]))
    hours = np.arange(samples, dtype=np.float64) * step_seconds / 3600.0
    phase = rng.uniform(-0.75, 0.75)
    morning = np.exp(-0.5 * (((hours + phase) % 24.0 - 7.5) / 1.8) ** 2)
    evening = np.exp(-0.5 * (((hours + phase) % 24.0 - 19.0) / 2.4) ** 2)
    overnight = np.exp(-0.5 * (((hours + phase) % 24.0 - 2.5) / 2.8) ** 2)
    daily = 0.60 + rng.uniform(0.32, 0.44) * morning + rng.uniform(0.38, 0.52) * evening - 0.10 * overnight
    weekday = (np.floor(hours / 24.0).astype(int) % 7) < 5
    weekly = np.where(weekday, 1.02, 0.95)
    innovations = rng.normal(0.0, 0.018, size=samples)
    smooth = np.empty(samples, dtype=np.float64)
    smooth[0] = innovations[0]
    for t in range(1, samples):
        smooth[t] = 0.92 * smooth[t - 1] + innovations[t]
    pattern = daily * weekly * np.exp(smooth)
    relative = pattern / pattern.mean()
    hydraulically_safe = 1.0 + 0.30 * (relative - 1.0)
    return np.clip(hydraulically_safe, 0.70, 1.30)


def split_ranges(samples: int) -> dict[str, list[int]]:
    train_end = int(np.floor(0.75 * samples))
    val_end = int(np.floor(0.85 * samples))
    return {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, samples]}


def generate(args: argparse.Namespace) -> None:
    import wntr

    source = args.source_inp.resolve()
    root = args.output_root.resolve() / args.dataset
    observations_dir = root / "observations"
    evaluation_dir = root / "evaluation_only"
    observations_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    source_copy = root / "source_original.inp"
    shutil.copyfile(source, source_copy)
    simulation_inp = root / "simulation_ready.inp"
    repairs = sanitize_inp(source_copy, simulation_inp)
    wn = wntr.network.WaterNetworkModel(str(simulation_inp))

    wn.options.time.duration = int((args.samples - 1) * args.step_seconds)
    wn.options.time.hydraulic_timestep = int(args.step_seconds)
    wn.options.time.report_timestep = int(args.step_seconds)
    wn.options.time.pattern_timestep = int(args.step_seconds)
    wn.options.time.rule_timestep = int(args.step_seconds)
    wn.options.time.quality_timestep = int(args.step_seconds)

    pattern_names: list[str] = []
    for node_index, name in enumerate(wn.junction_name_list):
        pattern_name = f"campaign_{node_index:04d}"
        wn.add_pattern(pattern_name, demand_pattern(args.samples, args.step_seconds, args.seed, node_index))
        junction = wn.get_node(name)
        for demand_ts in junction.demand_timeseries_list:
            demand_ts.pattern_name = pattern_name
        pattern_names.append(pattern_name)

    simulator = wntr.sim.EpanetSimulator(wn) if args.simulator == "epanet" else wntr.sim.WNTRSimulator(wn)
    results = (simulator.run_sim(file_prefix=str(root / "epanet"))
               if args.simulator == "epanet" else simulator.run_sim())
    node_ids = list(wn.node_name_list)
    link_ids = list(wn.link_name_list)
    pressure_frame = results.node["pressure"].reindex(columns=node_ids)
    time_s = np.asarray(pressure_frame.index, dtype=np.int64)
    pressure = pressure_frame.to_numpy(dtype=np.float32, copy=True)
    demand = results.node["demand"].reindex(columns=node_ids).to_numpy(dtype=np.float32, copy=True)
    if len(time_s) != args.samples:
        raise RuntimeError(f"Expected {args.samples} samples, simulator returned {len(time_s)}")
    if not np.isfinite(pressure).all() or not np.isfinite(demand).all():
        raise RuntimeError("Simulation returned non-finite pressure or demand")

    node_xy = np.asarray([wn.get_node(name).coordinates for name in node_ids], dtype=np.float32)
    if node_xy.shape != (len(node_ids), 2) or not np.isfinite(node_xy).all():
        raise RuntimeError("Complete finite coordinates are required for this campaign")

    node_pos = {name: i for i, name in enumerate(node_ids)}
    edge_index = np.asarray(
        [[node_pos[wn.get_link(name).start_node_name], node_pos[wn.get_link(name).end_node_name]] for name in link_ids],
        dtype=np.int64,
    ).T
    np.savez_compressed(
        observations_dir / "data.npz",
        time_s=time_s,
        pressure=pressure,
        demand=demand,
        node_xy=node_xy,
        mask_pressure=np.ones_like(pressure, dtype=np.uint8),
        mask_demand=np.ones_like(demand, dtype=np.uint8),
    )
    np.savez_compressed(
        evaluation_dir / "reference_topology.npz",
        edge_index=edge_index,
        node_ids=np.asarray(node_ids),
        link_ids=np.asarray(link_ids),
    )

    splits = split_ranges(args.samples)
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "domain": "water",
        "provenance": args.provenance,
        "source_url": args.source_url,
        "source_revision": args.source_revision,
        "license": args.license,
        "observation_file": "data.npz",
        "variables": ["pressure", "demand"],
        "node_ids": node_ids,
        "num_nodes": len(node_ids),
        "num_samples": args.samples,
        "step_seconds": args.step_seconds,
        "splits": splits,
        "window": {"lags": 6, "forecast_horizon": 12, "cross_boundary_windows": False},
        "normalization_fit_split": "train",
        "candidate_statistics_fit_split": "train",
        "source_sha256": sha256(source_copy),
        "simulation_ready_sha256": sha256(simulation_inp),
        "simulation": {"engine": args.simulator, "seed": args.seed, "demand_pattern_model": "documented diurnal + weekday/weekend + smoothed log-noise", "pattern_count": len(pattern_names)},
        "non_topological_repairs": repairs,
        "reference_topology_exposed": False,
    }
    (observations_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    evaluation_manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "reference_file": "reference_topology.npz",
        "num_nodes": len(node_ids),
        "num_links": len(link_ids),
        "num_unique_undirected_edges": len({tuple(sorted((int(u), int(v)))) for u, v in edge_index.T if int(u) != int(v)}),
        "topology_representation": "binary undirected simple graph; parallel hydraulic links collapse to one node pair",
        "reference_sha256": sha256(evaluation_dir / "reference_topology.npz"),
        "policy": "post-hoc evaluation only; never import from training, selection, consensus, or decoding",
    }
    (evaluation_dir / "manifest.json").write_text(json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "nodes": len(node_ids), "links": len(link_ids), "samples": len(time_s), "splits": splits, "output": str(root)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-inp", type=Path, required=True)
    parser.add_argument("--provenance", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmarks/water"))
    parser.add_argument("--samples", type=int, default=2880, help="30 days at the default 15-minute step")
    parser.add_argument("--step-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--simulator", choices=["epanet", "wntr"], default="epanet")
    args = parser.parse_args()
    if args.samples < 64:
        parser.error("--samples must be at least 64")
    return args


if __name__ == "__main__":
    generate(parse_args())
