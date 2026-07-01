"""Generate electric graph-reconstruction benchmarks from SimBench.

The saved files keep the current training contract:

data.npz:
  pressure, demand, node_xy, edge_index, initial_edge_index

For this electric benchmark:
  pressure := bus voltage magnitude vm_pu
  demand   := net active load at the bus, load_p_mw - sgen_p_mw

`edge_index` is the hidden ground-truth topology and must be used only for
post-hoc metrics. Training can use node coordinates and nodal time series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pandapower as pp
    import simbench as sb
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit(
        "This generator requires simbench and pandapower. "
        "Install them with: pip install simbench pandapower"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "electric_simple": "1-LV-semiurb4--0-sw",
    "electric_intermediate": "1-LV-urban6--0-sw",
    "electric_medium": "1-LV-rural2--0-sw",
    "electric_loopy": "1-MV-rural--0-sw",
    "electric_mesh_simple": "1-LV-semiurb4--0-sw",
    "electric_mesh_medium": "1-LV-urban6--0-sw",
}

MESH_PRESETS = {
    "electric_mesh_simple": {"extra_tie_lines": 12, "seed": 1101},
    "electric_mesh_medium": {"extra_tie_lines": 16, "seed": 1102},
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def to_edge_index(edges: set[tuple[int, int]], bidirectional: bool = True) -> np.ndarray:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        if bidirectional:
            rows.append((v, u))
    if not rows:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64).T


def parse_bus_xy(net) -> np.ndarray:
    coords = []
    for geo in net.bus["geo"].fillna(""):
        if not isinstance(geo, str) or "coordinates" not in geo:
            coords.append((np.nan, np.nan))
            continue
        data = json.loads(geo)
        lon, lat = data["coordinates"][:2]
        coords.append((float(lon), float(lat)))
    xy = np.asarray(coords, dtype=np.float64)
    if np.isnan(xy).any():
        rng = np.random.default_rng(0)
        missing = np.isnan(xy).any(axis=1)
        xy[missing] = rng.uniform(0.0, 1.0, size=(int(missing.sum()), 2))
    lo = xy.min(axis=0, keepdims=True)
    hi = xy.max(axis=0, keepdims=True)
    xy = (xy - lo) / (hi - lo + 1e-12)
    return xy.astype(np.float32)


def build_knn_edges(node_xy: np.ndarray, k: int) -> set[tuple[int, int]]:
    n = node_xy.shape[0]
    dist = np.sqrt(((node_xy[:, None, :] - node_xy[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    k = min(max(1, int(k)), n - 1)
    idx = np.argsort(dist, axis=1)[:, :k]
    edges = set()
    for u in range(n):
        for v in idx[u]:
            edges.add(canonical(int(u), int(v)))
    return edges


def closed_switches_by_element(net, et: str) -> set[int]:
    if "switch" not in net or len(net.switch) == 0:
        return set()
    sw = net.switch
    mask = (sw["et"] == et) & (sw["closed"].astype(bool))
    return set(sw.loc[mask, "element"].astype(int).tolist())


def open_switches_by_element(net, et: str) -> set[int]:
    if "switch" not in net or len(net.switch) == 0:
        return set()
    sw = net.switch
    mask = (sw["et"] == et) & (~sw["closed"].astype(bool))
    return set(sw.loc[mask, "element"].astype(int).tolist())


def bus_lookup(net) -> dict[int, int]:
    return {int(bus_id): pos for pos, bus_id in enumerate(net.bus.index.astype(int).tolist())}


def add_plausible_tie_lines(net, node_xy: np.ndarray, lookup: dict[int, int], count: int, seed: int) -> list[tuple[int, int]]:
    """Add a few same-voltage local tie-lines before running power flow."""
    if count <= 0 or len(net.line) == 0:
        return []

    rng = np.random.default_rng(seed)
    inverse_lookup = {pos: bus_id for bus_id, pos in lookup.items()}
    existing = active_topology_edges(net, lookup)
    n = node_xy.shape[0]
    dist = np.sqrt(((node_xy[:, None, :] - node_xy[None, :, :]) ** 2).sum(axis=-1))
    finite = dist[~np.eye(n, dtype=bool)]
    min_d = float(np.quantile(finite, 0.08)) if finite.size else 0.0
    max_d = float(np.quantile(finite, 0.55)) if finite.size else 1.0

    candidates = []
    vn = net.bus["vn_kv"].to_dict()
    for u in range(n):
        for v in range(u + 1, n):
            edge = canonical(u, v)
            if edge in existing:
                continue
            bus_u = inverse_lookup[u]
            bus_v = inverse_lookup[v]
            if abs(float(vn[bus_u]) - float(vn[bus_v])) > 1e-9:
                continue
            d = float(dist[u, v])
            if min_d <= d <= max_d:
                candidates.append((d, u, v))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return []
    local_pool = candidates[: max(count * 10, count)]
    rng.shuffle(local_pool)

    line_template = net.line[net.line["in_service"].astype(bool)].copy()
    if line_template.empty:
        line_template = net.line.copy()
    numeric = line_template.select_dtypes(include=[np.number])
    med = numeric.median(numeric_only=True)
    length_med = float(max(0.001, med.get("length_km", 0.03)))
    geo_med = float(np.median([x[0] for x in candidates[: max(1, min(len(candidates), 50))]]))

    added = []
    for d, u, v in local_pool:
        if len(added) >= count:
            break
        bus_u = inverse_lookup[u]
        bus_v = inverse_lookup[v]
        length_km = float(np.clip(length_med * d / max(geo_med, 1e-6), 0.001, max(0.002, 3.0 * length_med)))
        pp.create_line_from_parameters(
            net,
            from_bus=bus_u,
            to_bus=bus_v,
            length_km=length_km,
            r_ohm_per_km=float(med.get("r_ohm_per_km", 0.2)),
            x_ohm_per_km=float(med.get("x_ohm_per_km", 0.08)),
            c_nf_per_km=float(med.get("c_nf_per_km", 0.0)),
            max_i_ka=float(med.get("max_i_ka", 0.2)),
            name=f"synthetic_tie_{len(added):03d}",
            type=str(line_template.iloc[0].get("type", "cs")),
            in_service=True,
        )
        added.append(canonical(u, v))
    return added


def active_topology_edges(net, lookup: dict[int, int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    open_lines = open_switches_by_element(net, "l")
    open_trafos = open_switches_by_element(net, "t")

    for idx, row in net.line.iterrows():
        if bool(row.get("in_service", True)) and int(idx) not in open_lines:
            edges.add(canonical(lookup[int(row.from_bus)], lookup[int(row.to_bus)]))

    for idx, row in net.trafo.iterrows():
        if bool(row.get("in_service", True)) and int(idx) not in open_trafos:
            edges.add(canonical(lookup[int(row.hv_bus)], lookup[int(row.lv_bus)]))

    closed_bus_switches = net.switch[(net.switch["et"] == "b") & (net.switch["closed"].astype(bool))]
    for _, row in closed_bus_switches.iterrows():
        edges.add(canonical(lookup[int(row.bus)], lookup[int(row.element)]))

    return {e for e in edges if e[0] != e[1]}


def make_splits(timesteps: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    train_end = int(timesteps * train_ratio)
    val_end = int(timesteps * (train_ratio + val_ratio))
    return {
        "train": {"start": 0, "end": train_end},
        "val": {"start": train_end, "end": val_end},
        "test": {"start": val_end, "end": timesteps},
    }


def apply_profile_row(net, abs_values: dict, profile_idx: int) -> None:
    if ("load", "p_mw") in abs_values and len(net.load):
        row = abs_values[("load", "p_mw")].iloc[profile_idx]
        net.load.loc[row.index, "p_mw"] = row.to_numpy(dtype=float)
    if ("load", "q_mvar") in abs_values and len(net.load):
        row = abs_values[("load", "q_mvar")].iloc[profile_idx]
        net.load.loc[row.index, "q_mvar"] = row.to_numpy(dtype=float)
    if ("sgen", "p_mw") in abs_values and len(net.sgen):
        row = abs_values[("sgen", "p_mw")].iloc[profile_idx]
        net.sgen.loc[row.index, "p_mw"] = row.to_numpy(dtype=float)
    if ("sgen", "q_mvar") in abs_values and len(net.sgen):
        row = abs_values[("sgen", "q_mvar")].iloc[profile_idx]
        net.sgen.loc[row.index, "q_mvar"] = row.to_numpy(dtype=float)
    if ("storage", "p_mw") in abs_values and len(net.storage):
        row = abs_values[("storage", "p_mw")].iloc[profile_idx]
        net.storage.loc[row.index, "p_mw"] = row.to_numpy(dtype=float)


def aggregate_bus_power(net, n: int, lookup: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_load = np.zeros(n, dtype=np.float32)
    q_load = np.zeros(n, dtype=np.float32)
    p_sgen = np.zeros(n, dtype=np.float32)
    q_sgen = np.zeros(n, dtype=np.float32)

    if len(net.load):
        buses = np.asarray([lookup[int(b)] for b in net.load["bus"].to_numpy(dtype=int)], dtype=int)
        np.add.at(p_load, buses, net.load["p_mw"].to_numpy(dtype=np.float32))
        np.add.at(q_load, buses, net.load["q_mvar"].to_numpy(dtype=np.float32))
    if len(net.sgen):
        buses = np.asarray([lookup[int(b)] for b in net.sgen["bus"].to_numpy(dtype=int)], dtype=int)
        np.add.at(p_sgen, buses, net.sgen["p_mw"].to_numpy(dtype=np.float32))
        if "q_mvar" in net.sgen.columns:
            np.add.at(q_sgen, buses, net.sgen["q_mvar"].fillna(0.0).to_numpy(dtype=np.float32))
    return p_load, q_load, p_sgen, q_sgen


def simulate_timeseries(net, timesteps: int, stride: int, start: int, lookup: dict[int, int]) -> dict[str, np.ndarray]:
    abs_values = sb.get_absolute_values(net, True)
    max_steps = min(
        len(v)
        for v in abs_values.values()
        if hasattr(v, "__len__") and len(v) > 0
    )
    profile_indices = (start + stride * np.arange(timesteps)) % max_steps
    n = len(net.bus)

    voltage = np.zeros((timesteps, n), dtype=np.float32)
    va_degree = np.zeros((timesteps, n), dtype=np.float32)
    p_load = np.zeros((timesteps, n), dtype=np.float32)
    q_load = np.zeros((timesteps, n), dtype=np.float32)
    p_sgen = np.zeros((timesteps, n), dtype=np.float32)
    q_sgen = np.zeros((timesteps, n), dtype=np.float32)
    converged = np.zeros(timesteps, dtype=bool)

    for t, profile_idx in enumerate(profile_indices):
        apply_profile_row(net, abs_values, int(profile_idx))
        try:
            pp.runpp(net, calculate_voltage_angles=False, init="auto", numba=False)
            converged[t] = bool(net.converged)
        except Exception:
            converged[t] = False

        if converged[t]:
            voltage[t] = net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
            va_degree[t] = net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        elif t > 0:
            voltage[t] = voltage[t - 1]
            va_degree[t] = va_degree[t - 1]
        else:
            voltage[t] = 1.0
            va_degree[t] = 0.0

        p_l, q_l, p_g, q_g = aggregate_bus_power(net, n, lookup)
        p_load[t], q_load[t], p_sgen[t], q_sgen[t] = p_l, q_l, p_g, q_g

    return {
        "voltage_pu": voltage,
        "va_degree": va_degree,
        "p_load_mw": p_load,
        "q_load_mvar": q_load,
        "p_sgen_mw": p_sgen,
        "q_sgen_mvar": q_sgen,
        "net_p_mw": p_load - p_sgen,
        "net_q_mvar": q_load - q_sgen,
        "profile_index": profile_indices.astype(np.int64),
        "powerflow_converged": converged,
    }


def generate_one(name: str, code: str, args: argparse.Namespace) -> None:
    out_dir = PROJECT_ROOT / args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    net = sb.get_simbench_net(code)
    lookup = bus_lookup(net)
    node_xy = parse_bus_xy(net)
    mesh_cfg = MESH_PRESETS.get(name, {})
    added_tie_lines = add_plausible_tie_lines(
        net,
        node_xy=node_xy,
        lookup=lookup,
        count=int(mesh_cfg.get("extra_tie_lines", 0)),
        seed=int(mesh_cfg.get("seed", 0)),
    )
    gt_edges = active_topology_edges(net, lookup)
    initial_edges = build_knn_edges(node_xy, args.initial_knn)
    series = simulate_timeseries(net, args.timesteps, args.profile_stride, args.profile_start, lookup)

    np.savez_compressed(
        out_dir / "data.npz",
        pressure=series["voltage_pu"],
        demand=series["net_p_mw"],
        node_xy=node_xy,
        edge_index=to_edge_index(gt_edges, bidirectional=True),
        initial_edge_index=to_edge_index(initial_edges, bidirectional=True),
        time_s=series["profile_index"] * int(args.profile_step_seconds),
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
    )

    meta = {
        "name": "simbench_electric",
        "dataset_id": name,
        "simbench_code": code,
        "num_nodes": int(len(net.bus)),
        "num_gt_undirected_edges": int(len(gt_edges)),
        "num_initial_undirected_edges": int(len(initial_edges)),
        "node_ids": [str(x) for x in net.bus["name"].tolist()],
        "splits": make_splits(args.timesteps),
        "compatibility_mapping": {
            "pressure": "voltage_pu",
            "demand": "net_p_mw",
        },
        "allowed_for_training": [
            "node_xy",
            "voltage_pu / pressure",
            "net_p_mw / demand",
            "p_load_mw",
            "q_load_mvar",
            "p_sgen_mw",
            "q_sgen_mvar",
            "net_q_mvar",
        ],
        "posthoc_only": [
            "edge_index",
            "line / trafo / switch connectivity",
            "line impedances",
            "line lengths",
            "number of ground-truth edges",
        ],
        "generator": {
            "source": "SimBench + pandapower power flow",
            "synthetic_tie_lines": len(added_tie_lines),
            "synthetic_tie_line_edges": [[int(u), int(v)] for u, v in added_tie_lines],
            "timesteps": int(args.timesteps),
            "profile_stride": int(args.profile_stride),
            "profile_start": int(args.profile_start),
            "initial_knn": int(args.initial_knn),
            "converged_steps": int(series["powerflow_converged"].sum()),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {name}: {code}")
    print(f"     nodes={len(net.bus)} gt_edges={len(gt_edges)} initial_edges={len(initial_edges)}")
    if added_tie_lines:
        print(f"     synthetic_tie_lines={len(added_tie_lines)}")
    print(f"     converged={int(series['powerflow_converged'].sum())}/{args.timesteps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SimBench electric benchmarks.")
    parser.add_argument("--out_root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    parser.add_argument("--timesteps", type=int, default=720)
    parser.add_argument("--profile_stride", type=int, default=4, help="SimBench profiles are 15-min; stride=4 gives hourly samples.")
    parser.add_argument("--profile_start", type=int, default=0)
    parser.add_argument("--profile_step_seconds", type=int, default=900)
    parser.add_argument("--initial_knn", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = PRESETS if args.preset == "all" else {args.preset: PRESETS[args.preset]}
    for name, code in selected.items():
        generate_one(name, code, args)


if __name__ == "__main__":
    main()

