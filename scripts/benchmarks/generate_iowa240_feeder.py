#!/usr/bin/env python3
"""Generate an electric benchmark from the real Iowa 240-bus feeder data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import opendssdirect as dss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_matrix(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key], dtype=np.float64).T


def split_ranges(samples: int) -> dict[str, list[int]]:
    train_end = int(np.floor(0.75 * samples))
    val_end = int(np.floor(0.85 * samples))
    return {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, samples]}


def base_bus(bus_with_terminals: str) -> str:
    return bus_with_terminals.split(".", 1)[0].lower()


def internal_edges(bus_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    position = {name: index for index, name in enumerate(bus_ids)}
    edges: set[tuple[int, int]] = set()
    labels: list[str] = []
    index = dss.Lines.First()
    while index:
        name = dss.Lines.Name()
        buses = dss.CktElement.BusNames()
        if len(buses) >= 2:
            first, second = base_bus(buses[0]), base_bus(buses[1])
            if first in position and second in position and first != second and dss.CktElement.Enabled():
                edge = tuple(sorted((position[first], position[second])))
                if edge not in edges:
                    edges.add(edge)
                    labels.append(name)
        index = dss.Lines.Next()
    return np.asarray(sorted(edges), dtype=np.int64).T, labels


def feeder_coordinates(bus_ids: list[str]) -> np.ndarray:
    coordinates = []
    for name in bus_ids:
        dss.Circuit.SetActiveBus(name)
        coordinates.append((dss.Bus.X(), dss.Bus.Y()))
    xy = np.asarray(coordinates, dtype=np.float32)
    if not np.isfinite(xy).all() or np.ptp(xy, axis=0).min() <= 0:
        raise RuntimeError("Iowa feeder coordinates are missing or degenerate")
    return xy


def voltage_magnitude(bus_ids: list[str]) -> np.ndarray:
    values = np.empty(len(bus_ids), dtype=np.float32)
    for index, name in enumerate(bus_ids):
        dss.Circuit.SetActiveBus(name)
        magnitude_angle = np.asarray(dss.Bus.puVmagAngle(), dtype=np.float64)
        magnitudes = magnitude_angle[0::2]
        values[index] = float(magnitudes.mean())
    return values


def generate(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    qsts_dir = args.qsts_dir.resolve()
    output = args.output_root.resolve() / args.dataset
    observations = output / "observations"
    evaluation = output / "evaluation_only"
    observations.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feeder in "ABC":
        p = read_matrix(qsts_dir / f"Feeder{feeder}_P.mat", f"Feeder{feeder}_P")
        q = read_matrix(qsts_dir / f"Feeder{feeder}_Q.mat", f"Feeder{feeder}_Q")
        matrices[feeder] = (p, q)
    available = min(matrix.shape[0] for pair in matrices.values() for matrix in pair)
    samples = available if args.samples == 0 else min(args.samples, available)

    master = model_dir / "Master.dss"
    dss.Text.Command("Compile [" + str(master) + "]")
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS snapshot model did not converge")

    feeder = args.feeder.upper()
    feeder_offset = {"A": 1000, "B": 2000, "C": 3000}[feeder]
    feeder_width = matrices[feeder][0].shape[1]
    bus_ids = [f"bus{feeder_offset + index}" for index in range(1, feeder_width + 1)]
    coordinates = feeder_coordinates(bus_ids)
    edge_index, edge_labels = internal_edges(bus_ids)

    load_mapping: list[tuple[str, str, int]] = []
    load_index = dss.Loads.First()
    while load_index:
        name = dss.Loads.Name()
        match = re.search(r"(\d{4})$", name)
        if match:
            number = int(match.group(1))
            group = "A" if 1000 < number < 2000 else "B" if 2000 < number < 3000 else "C" if 3000 < number < 4000 else ""
            if group:
                load_mapping.append((name, group, number - {"A": 1000, "B": 2000, "C": 3000}[group] - 1))
        load_index = dss.Loads.Next()

    voltage = np.empty((samples, len(bus_ids)), dtype=np.float32)
    converged = np.zeros(samples, dtype=np.uint8)
    started = time.perf_counter()
    for timestep in range(samples):
        for load_name, group, column in load_mapping:
            dss.Loads.Name(load_name)
            dss.Loads.kW(float(matrices[group][0][timestep, column]))
            dss.Loads.kvar(float(matrices[group][1][timestep, column]))
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS did not converge at timestep {timestep}")
        converged[timestep] = 1
        voltage[timestep] = voltage_magnitude(bus_ids)

    p_net_mw = (matrices[feeder][0][:samples] / 1000.0).astype(np.float32)
    time_s = np.arange(samples, dtype=np.int64) * 3600
    np.savez_compressed(
        observations / "data.npz",
        time_s=time_s,
        pressure=voltage,
        demand=p_net_mw,
        node_xy=coordinates,
        mask_pressure=np.ones_like(voltage, dtype=np.uint8),
        mask_demand=np.ones_like(p_net_mw, dtype=np.uint8),
    )
    np.savez_compressed(
        evaluation / "reference_topology.npz",
        edge_index=edge_index,
        node_ids=np.asarray(bus_ids),
        link_ids=np.asarray(edge_labels),
    )

    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "domain": "electric",
        "provenance": "real utility grid, unmodified; public Iowa 240-bus time-series test system",
        "source_url": "https://wzy.ece.iastate.edu/testsystem.html",
        "paper_doi": "10.1109/NAPS46351.2019.8999982",
        "license": "publicly shared with utility permission; no explicit standard licence on the source page",
        "subnetwork_rule": f"pre-defined operational feeder {feeder}, primary buses {bus_ids[0]}..{bus_ids[-1]}",
        "boundary_condition": f"full three-feeder OpenDSS solve; upstream breaker CB_{feeder_offset // 10 + 1} and common source excluded from reconstructed internal graph",
        "observation_file": "data.npz",
        "variables": ["pressure", "demand"],
        "physical_variable_names": ["voltage_magnitude_pu", "p_net_mw"],
        "node_ids": bus_ids,
        "num_nodes": len(bus_ids),
        "num_samples": samples,
        "step_seconds": 3600,
        "splits": split_ranges(samples),
        "window": {"lags": 6, "forecast_horizon": 12, "cross_boundary_windows": False},
        "normalization_fit_split": "train",
        "candidate_statistics_fit_split": "train",
        "source_files": {
            "master_dss_sha256": sha256(master),
            **{f"feeder_{name.lower()}_{kind.lower()}_sha256": sha256(qsts_dir / f"Feeder{name}_{kind}.mat") for name in "ABC" for kind in ("P", "Q")},
        },
        "simulation": {"engine": "OpenDSSDirect", "samples": samples, "nonconverged": int((1 - converged).sum()), "elapsed_seconds": time.perf_counter() - started},
        "reference_topology_exposed": False,
    }
    (observations / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    eval_manifest = {
        "dataset": args.dataset,
        "reference_file": "reference_topology.npz",
        "num_nodes": len(bus_ids),
        "num_internal_edges": int(edge_index.shape[1]),
        "num_unique_undirected_edges": len({tuple(sorted((int(u), int(v)))) for u, v in edge_index.T if int(u) != int(v)}),
        "all_pairs_without_self_loops": len(bus_ids) * (len(bus_ids) - 1) // 2,
        "reference_sha256": sha256(evaluation / "reference_topology.npz"),
        "policy": "post-hoc evaluation only",
    }
    (evaluation / "manifest.json").write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "feeder": feeder, "nodes": len(bus_ids), "edges": int(edge_index.shape[1]), "samples": samples, "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="iowa240_feeder_b")
    parser.add_argument("--feeder", choices=["A", "B", "C", "a", "b", "c"], default="B")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qsts-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmarks/electric"))
    parser.add_argument("--samples", type=int, default=0, help="0 uses the full 8760-hour year")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
