"""Evaluate four fixed, structurally diverse electric distribution networks."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandapower as pp
import simbench as sb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.electric.generate_simbench_electric import (  # noqa: E402
    build_knn_edges,
    make_splits,
    to_edge_index,
)
from scripts.experiments import run_electric_final_four_extended as base  # noqa: E402
from scripts.experiments.run_section_vi_water_final import undirected_edges  # noqa: E402


SCENARIOS = {
    "electric_a_radial_feeder": {
        "topology": "electric_a_radial_feeder.npz",
        "profile_seed": 2101,
        "description": "Radial feeder with laterals",
    },
    "electric_b_ring_feeder": {
        "topology": "electric_b_ring_feeder.npz",
        "profile_seed": 2102,
        "description": "Ring feeder with service laterals",
    },
    "electric_c_urban_grid": {
        "topology": "electric_c_urban_grid.npz",
        "profile_seed": 2103,
        "description": "Urban weakly meshed street grid",
    },
    "electric_d_multi_feeder": {
        "topology": "electric_d_multi_feeder.npz",
        "profile_seed": 2104,
        "description": "Interconnected multi-feeder network",
    },
}

TUNING_SEED = 7
FINAL_SEEDS = (11, 22)
PROFILE_STARTS = {7: 0, 11: 4000, 22: 8000}
PROFILE_TEMPLATE = "1-LV-urban6--0-sw"


@lru_cache(maxsize=1)
def profile_bank() -> tuple[np.ndarray, np.ndarray]:
    """Aggregate real SimBench profiles by bus for reuse on fixed circuits."""
    net = sb.get_simbench_net(PROFILE_TEMPLATE)
    values = sb.get_absolute_values(net, True)
    p_load = values[("load", "p_mw")]
    q_load = values[("load", "q_mvar")]
    p_sgen = values.get(("sgen", "p_mw"))
    q_sgen = values.get(("sgen", "q_mvar"))

    buses = sorted(
        set(net.load["bus"].astype(int).tolist())
        | set(net.sgen["bus"].astype(int).tolist())
    )
    p_profiles = []
    q_profiles = []
    for bus in buses:
        load_ids = net.load.index[net.load["bus"].astype(int) == bus].tolist()
        sgen_ids = net.sgen.index[net.sgen["bus"].astype(int) == bus].tolist()
        p = p_load.loc[:, load_ids].sum(axis=1).to_numpy(dtype=np.float64)
        q = q_load.loc[:, load_ids].sum(axis=1).to_numpy(dtype=np.float64)
        if p_sgen is not None and sgen_ids:
            p -= p_sgen.loc[:, sgen_ids].sum(axis=1).to_numpy(dtype=np.float64)
        if q_sgen is not None and sgen_ids:
            q -= q_sgen.loc[:, sgen_ids].sum(axis=1).to_numpy(dtype=np.float64)
        if np.max(np.abs(p)) > 1e-10 or np.max(np.abs(q)) > 1e-10:
            p_profiles.append(p)
            q_profiles.append(q)

    return (
        np.stack(p_profiles, axis=1).astype(np.float32),
        np.stack(q_profiles, axis=1).astype(np.float32),
    )


def build_network(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    sources: list[int],
) -> tuple[object, list[int], np.ndarray]:
    """Build a physical LV circuit whose active branches equal the fixed graph."""
    net = pp.create_empty_network(name="fixed_diverse_lv_network")
    for node in range(len(xy)):
        pp.create_bus(net, vn_kv=0.4, name=f"N{node:03d}")
    for source in sources:
        pp.create_ext_grid(net, bus=source, vm_pu=1.0, name=f"source_{source}")

    distances = np.asarray(
        [np.linalg.norm(xy[u] - xy[v]) for u, v in sorted(edges)],
        dtype=np.float64,
    )
    median_distance = float(np.median(distances))
    lengths = []
    for edge_id, (u, v) in enumerate(sorted(edges)):
        distance = float(np.linalg.norm(xy[u] - xy[v]))
        length_km = float(np.clip(0.017 * distance / max(median_distance, 1e-8), 0.005, 0.08))
        pp.create_line_from_parameters(
            net,
            from_bus=u,
            to_bus=v,
            length_km=length_km,
            r_ohm_per_km=0.1267,
            x_ohm_per_km=0.0798,
            c_nf_per_km=870.0,
            max_i_ka=0.357,
            name=f"branch_{edge_id:03d}",
            type="cs",
            in_service=True,
        )
        lengths.append(length_km)

    load_nodes = [node for node in range(len(xy)) if node not in sources]
    for node in load_nodes:
        pp.create_load(net, bus=node, p_mw=0.0, q_mvar=0.0, name=f"load_{node:03d}")
    return net, load_nodes, np.asarray(lengths, dtype=np.float32)


def simulate_fixed_network(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    sources: list[int],
    scenario: str,
    signal_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    net, load_nodes, line_lengths = build_network(xy, edges, sources)
    p_bank, q_bank = profile_bank()
    rng = np.random.default_rng(int(SCENARIOS[scenario]["profile_seed"]))
    selected = rng.permutation(p_bank.shape[1])[: len(load_nodes)]

    timesteps = int(base.COMMON["timesteps"])
    stride = int(base.COMMON["profile_stride"])
    start = int(PROFILE_STARTS[signal_seed])
    profile_indices = (start + stride * np.arange(timesteps)) % p_bank.shape[0]
    n = len(xy)

    voltage = np.zeros((timesteps, n), dtype=np.float32)
    angle = np.zeros((timesteps, n), dtype=np.float32)
    p_load = np.zeros((timesteps, n), dtype=np.float32)
    q_load = np.zeros((timesteps, n), dtype=np.float32)
    converged = np.zeros(timesteps, dtype=bool)

    for t, profile_idx in enumerate(profile_indices):
        p_values = p_bank[int(profile_idx), selected]
        q_values = q_bank[int(profile_idx), selected]
        net.load.loc[:, "p_mw"] = p_values
        net.load.loc[:, "q_mvar"] = q_values
        try:
            pp.runpp(net, calculate_voltage_angles=False, init="auto", numba=False)
            converged[t] = bool(net.converged)
        except Exception:
            converged[t] = False

        if converged[t]:
            voltage[t] = net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
            angle[t] = net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        elif t:
            voltage[t] = voltage[t - 1]
            angle[t] = angle[t - 1]
        else:
            voltage[t] = 1.0

        p_load[t, load_nodes] = p_values
        q_load[t, load_nodes] = q_values

    zeros = np.zeros_like(p_load)
    return {
        "voltage_pu": voltage,
        "va_degree": angle,
        "p_load_mw": p_load,
        "q_load_mvar": q_load,
        "p_sgen_mw": zeros,
        "q_sgen_mvar": zeros,
        "net_p_mw": p_load,
        "net_q_mvar": q_load,
        "profile_index": profile_indices.astype(np.int64),
        "powerflow_converged": converged,
        "profile_assignment": selected.astype(np.int64),
    }, line_lengths


def prepare_dataset(
    topology_path: Path,
    out_dir: Path,
    scenario: str,
    signal_seed: int,
) -> None:
    data_path = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if data_path.exists() and meta_path.exists():
        return

    fixed = np.load(topology_path)
    xy = fixed["node_xy"].astype(np.float32)
    gt = undirected_edges(fixed["edge_index"])
    sources = fixed["source_nodes"].astype(int).tolist()
    series, line_lengths = simulate_fixed_network(
        xy,
        gt,
        sources,
        scenario,
        signal_seed,
    )
    if not np.all(series["powerflow_converged"]):
        failed = int((~series["powerflow_converged"]).sum())
        raise RuntimeError(f"{scenario} seed={signal_seed}: {failed} power-flow failures")

    initial = build_knn_edges(xy, int(base.COMMON["initial_knn"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        pressure=series["voltage_pu"],
        demand=series["net_p_mw"],
        node_xy=xy,
        edge_index=to_edge_index(gt, bidirectional=True),
        initial_edge_index=to_edge_index(initial, bidirectional=True),
        time_s=series["profile_index"] * int(base.COMMON["profile_step_seconds"]),
        voltage_pu=series["voltage_pu"],
        va_degree=series["va_degree"],
        p_load_mw=series["p_load_mw"],
        q_load_mvar=series["q_load_mvar"],
        p_sgen_mw=series["p_sgen_mw"],
        q_sgen_mvar=series["q_sgen_mvar"],
        net_p_mw=series["net_p_mw"],
        net_q_mvar=series["net_q_mvar"],
        profile_index=series["profile_index"],
        powerflow_converged=series["powerflow_converged"],
        source_nodes=np.asarray(sources, dtype=np.int64),
    )
    meta = {
        "name": "section_vi_fixed_diverse_electric",
        "dataset_id": scenario,
        "description": SCENARIOS[scenario]["description"],
        "profile_template": PROFILE_TEMPLATE,
        "topology_source": str(topology_path.relative_to(PROJECT_ROOT)),
        "topology_regenerated": False,
        "signal_seed": signal_seed,
        "num_nodes": len(xy),
        "num_gt_undirected_edges": len(gt),
        "num_initial_undirected_edges": len(initial),
        "node_ids": [f"N{node:03d}" for node in range(len(xy))],
        "source_nodes": sources,
        "splits": make_splits(int(base.COMMON["timesteps"])),
        "compatibility_mapping": {
            "pressure": "voltage_pu",
            "demand": "net_p_mw",
        },
        "allowed_for_training": [
            "node_xy",
            "voltage_pu / pressure",
            "net_p_mw / demand",
        ],
        "posthoc_only": [
            "edge_index",
            "branch connectivity",
            "line parameters and lengths",
            "number of ground-truth edges",
        ],
        "generator": {
            "power_flow": "pandapower AC",
            "voltage_level_kv": 0.4,
            "profile_start": int(PROFILE_STARTS[signal_seed]),
            "profile_stride": int(base.COMMON["profile_stride"]),
            "profile_assignment": series["profile_assignment"].tolist(),
            "timesteps": int(base.COMMON["timesteps"]),
            "converged_steps": int(series["powerflow_converged"].sum()),
            "line_length_km_min": float(line_lengths.min()),
            "line_length_km_median": float(np.median(line_lengths)),
            "line_length_km_max": float(line_lengths.max()),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[DATA] {scenario} seed={signal_seed}: nodes={len(xy)} edges={len(gt)} "
        f"converged={int(series['powerflow_converged'].sum())}/{base.COMMON['timesteps']}"
    )


def configure_base_module() -> None:
    base.SCENARIOS = SCENARIOS
    base.TUNING_SEED = TUNING_SEED
    base.FINAL_SEEDS = FINAL_SEEDS
    base.PROFILE_STARTS = PROFILE_STARTS
    base.COMMON["candidate_mode"] = "observable_union"
    base.COMMON["candidate_signal_k"] = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology_root",
        type=Path,
        default=PROJECT_ROOT
        / "results_section_vi"
        / "electric_topology_candidates_diverse",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_diverse_datasets",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_diverse_experiments",
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()
    configure_base_module()

    for scenario, cfg in SCENARIOS.items():
        topology_path = args.topology_root / cfg["topology"]
        if not topology_path.exists():
            raise FileNotFoundError(
                f"Missing fixed topology {topology_path}. Run "
                "scripts/visualization/generate_diverse_electric_topology_candidates.py first."
            )
        for seed in (TUNING_SEED, *FINAL_SEEDS):
            prepare_dataset(
                topology_path,
                args.data_root / scenario / f"seed_{seed}",
                scenario,
                seed,
            )

    if args.prepare_only:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    base.proxy_diagnostics(args.data_root, args.output_root)
    _, weights = base.stage1(args.data_root, args.output_root, args.collect_only)
    _, selected = base.stage2(
        args.data_root,
        args.output_root,
        weights,
        args.collect_only,
    )
    rows = base.final_runs(
        args.data_root,
        args.output_root,
        selected,
        args.collect_only,
    )
    base.aggregate(rows, selected, args.output_root)
    base.copy_figures(args.output_root)
    base.write_protocol(args.topology_root, args.output_root, selected)
    print(f"[OK] Results: {args.output_root}")


if __name__ == "__main__":
    main()
