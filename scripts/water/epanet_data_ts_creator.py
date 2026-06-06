#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
epanet_data_ts_creator.py

Generate a node-level time-series dataset (pressure + demand) from an EPANET .inp
by running an Extended Period Simulation (EPS) with WNTR.

This version:
- uses in-code paths (no required CLI args)
- is robust across WNTR versions (results.time can be list)
- saves node coordinates if available in the INP ([COORDINATES]) as node_xy [N,2]

Dependencies:
  pip install wntr

Outputs (in OUTDIR):
  - data.npz (compressed):
      time_s          int64   [T]
      pressure        float32 [T, N]
      demand          float32 [T, N]
      mask_pressure   uint8   [T, N]  (1 observed, 0 missing)
      mask_demand     uint8   [T, N]
      node_xy         float32 [N, 2]  (X,Y coordinates; NaN if unavailable)
      mask_xy         uint8   [N]     (1 if coord available, else 0)
      edge_index      int64   [2, E]  (saved if SAVE_EDGES=True)
  - meta.json:
      node_ids, link_ids, splits, simulator/options, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np


# =========================
# USER CONFIG (EDIT HERE)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INP_PATH = PROJECT_ROOT / "data" / "ctown.inp"
OUTDIR = PROJECT_ROOT / "data" / "ctown_ts"

SIMULATOR = "wntr"          # "wntr" or "epanet" (requires EPANET engine)
SAVE_EDGES = True           # keep GT edges for post-hoc evaluation

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
SEED = 0

NOISE_STD_PRESSURE = 0.0
NOISE_STD_DEMAND = 0.0
MISSING_RATE_PRESSURE = 0.0
MISSING_RATE_DEMAND = 0.0
MISSING_VALUE = np.nan

# If None, use values from the INP [OPTIONS]
DURATION_S = 90 * 24 * 3600    # 90 days
REPORT_STEP_S = None
HYDRAULIC_STEP_S = None
# =========================


def _require_wntr() -> None:
    try:
        import wntr  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'wntr'. Install with: pip install wntr\n"
            f"Original import error: {e}"
        )


def _set_eps_options(wn, duration_s: Optional[int], report_step_s: Optional[int], hydraulic_step_s: Optional[int]) -> None:
    opts = wn.options
    if duration_s is not None:
        opts.time.duration = int(duration_s)
    if hydraulic_step_s is not None:
        opts.time.hydraulic_timestep = int(hydraulic_step_s)
    if report_step_s is not None:
        opts.time.report_timestep = int(report_step_s)


def _extract_node_xy(wn, node_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract node coordinates if present in INP ([COORDINATES]).

    Returns:
      node_xy: [N,2] float32 with NaN for missing coords
      mask_xy: [N] uint8, 1 if coord present else 0
    """
    node_xy = np.full((len(node_ids), 2), np.nan, dtype=np.float32)
    mask_xy = np.zeros((len(node_ids),), dtype=np.uint8)

    for i, nid in enumerate(node_ids):
        node = wn.get_node(nid)

        # WNTR nodes usually expose coordinates as a tuple (x, y) if available.
        # We keep it robust by checking multiple possibilities.
        xy = None
        if hasattr(node, "coordinates"):
            xy = node.coordinates
        elif hasattr(node, "x") and hasattr(node, "y"):
            xy = (node.x, node.y)

        if xy is not None:
            try:
                x, y = float(xy[0]), float(xy[1])
                if np.isfinite(x) and np.isfinite(y):
                    node_xy[i, 0] = x
                    node_xy[i, 1] = y
                    mask_xy[i] = 1
            except Exception:
                pass

    return node_xy, mask_xy


def _run_simulation(
    inp_path: Path,
    duration_s: Optional[int],
    report_step_s: Optional[int],
    hydraulic_step_s: Optional[int],
    simulator: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Returns:
      time_s:        [T]
      pressure:      [T, N]
      demand:        [T, N]
      node_ids:      list length N
      edge_index:    [2, E] (ground truth)
      link_ids:      list length E
      node_xy:       [N, 2]
      mask_xy:       [N]
    """
    _require_wntr()
    import wntr

    if not inp_path.exists():
        raise FileNotFoundError(f"INP file not found: {inp_path}")

    wn = wntr.network.WaterNetworkModel(str(inp_path))
    _set_eps_options(wn, duration_s, report_step_s, hydraulic_step_s)

    simulator = simulator.lower().strip()
    if simulator not in {"wntr", "epanet"}:
        raise ValueError("SIMULATOR must be 'wntr' or 'epanet'")

    if simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    else:
        sim = wntr.sim.WNTRSimulator(wn)

    results = sim.run_sim()

    # Robust across WNTR versions
    time_s = np.asarray(results.time, dtype=np.int64)

    node_ids = list(wn.node_name_list)

    # Coordinates
    node_xy, mask_xy = _extract_node_xy(wn, node_ids)

    pressure_df = results.node["pressure"].reindex(columns=node_ids)
    demand_df = results.node["demand"].reindex(columns=node_ids)

    pressure = pressure_df.to_numpy(dtype=np.float32, copy=True)
    demand = demand_df.to_numpy(dtype=np.float32, copy=True)

    # Ground-truth edges from links (saved optionally)
    link_ids = list(wn.link_name_list)
    node_pos = {nid: i for i, nid in enumerate(node_ids)}

    src = np.empty(len(link_ids), dtype=np.int64)
    dst = np.empty(len(link_ids), dtype=np.int64)
    for k, lid in enumerate(link_ids):
        link = wn.get_link(lid)
        src[k] = node_pos[link.start_node_name]
        dst[k] = node_pos[link.end_node_name]

    edge_index = np.vstack([src, dst])
    return time_s, pressure, demand, node_ids, edge_index, link_ids, node_xy, mask_xy


def _apply_noise_and_missing(
    rng: np.random.Generator,
    x: np.ndarray,
    noise_std: float,
    missing_rate: float,
    missing_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x2 = x.copy()

    if noise_std and noise_std > 0.0:
        x2 = x2 + rng.normal(0.0, noise_std, size=x2.shape).astype(x2.dtype)

    mask = np.ones_like(x2, dtype=bool)

    if missing_rate and missing_rate > 0.0:
        drop = rng.random(size=x2.shape) < float(missing_rate)
        mask[drop] = False
        x2[drop] = missing_value

    return x2, mask


def _make_splits(T: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("TRAIN_RATIO must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("VAL_RATIO must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be < 1")

    t_train_end = int(np.floor(T * train_ratio))
    t_val_end = int(np.floor(T * (train_ratio + val_ratio)))
    return {
        "train": (0, t_train_end),
        "val": (t_train_end, t_val_end),
        "test": (t_val_end, T),
    }


def _parse_optional_cli_overrides():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--duration_s", default=None, type=int)
    ap.add_argument("--report_step_s", default=None, type=int)
    ap.add_argument("--hydraulic_step_s", default=None, type=int)
    ap.add_argument("--simulator", default=None, type=str, choices=["wntr", "epanet"])
    ap.add_argument("--save_edges", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_optional_cli_overrides()

    inp_path = INP_PATH.expanduser().resolve()
    outdir = OUTDIR.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    simulator = args.simulator if args.simulator is not None else SIMULATOR
    duration_s = args.duration_s if args.duration_s is not None else DURATION_S
    report_step_s = args.report_step_s if args.report_step_s is not None else REPORT_STEP_S
    hydraulic_step_s = args.hydraulic_step_s if args.hydraulic_step_s is not None else HYDRAULIC_STEP_S
    save_edges = bool(SAVE_EDGES or args.save_edges)

    time_s, pressure, demand, node_ids, edge_index, link_ids, node_xy, mask_xy = _run_simulation(
        inp_path=inp_path,
        duration_s=duration_s,
        report_step_s=report_step_s,
        hydraulic_step_s=hydraulic_step_s,
        simulator=simulator,
    )

    rng = np.random.default_rng(SEED)

    pressure2, mask_p = _apply_noise_and_missing(
        rng, pressure, NOISE_STD_PRESSURE, MISSING_RATE_PRESSURE, MISSING_VALUE
    )
    demand2, mask_d = _apply_noise_and_missing(
        rng, demand, NOISE_STD_DEMAND, MISSING_RATE_DEMAND, MISSING_VALUE
    )

    T = int(time_s.shape[0])
    splits = _make_splits(T, TRAIN_RATIO, VAL_RATIO)

    npz_path = outdir / "data.npz"
    payload = {
        "time_s": time_s.astype(np.int64),
        "pressure": pressure2.astype(np.float32),
        "demand": demand2.astype(np.float32),
        "mask_pressure": mask_p.astype(np.uint8),
        "mask_demand": mask_d.astype(np.uint8),
        "node_xy": node_xy.astype(np.float32),
        "mask_xy": mask_xy.astype(np.uint8),
    }
    if save_edges:
        payload["edge_index"] = edge_index.astype(np.int64)

    np.savez_compressed(npz_path, **payload)

    coords_present = bool(int(mask_xy.sum()) > 0)

    meta = {
        "source_inp": str(inp_path),
        "simulator": simulator,
        "duration_s": duration_s,
        "report_step_s": report_step_s,
        "hydraulic_step_s": hydraulic_step_s,
        "n_time": T,
        "n_nodes": len(node_ids),
        "node_ids": node_ids,
        "coords": {
            "present": coords_present,
            "n_with_coords": int(mask_xy.sum()),
            "format": "XY",
            "units": "same as INP [COORDINATES] (typically planar drawing units; not necessarily meters)",
            "field_in_npz": "node_xy",
        },
        "units": {
            "time_s": "s",
            "pressure": "model units (EPANET/WNTR; see INP [OPTIONS])",
            "demand": "model units (EPANET/WNTR; see INP [OPTIONS])",
        },
        "splits": {k: {"start": int(v[0]), "end": int(v[1])} for k, v in splits.items()},
        "noise": {
            "seed": int(SEED),
            "noise_std_pressure": float(NOISE_STD_PRESSURE),
            "noise_std_demand": float(NOISE_STD_DEMAND),
            "missing_rate_pressure": float(MISSING_RATE_PRESSURE),
            "missing_rate_demand": float(MISSING_RATE_DEMAND),
            "missing_value": float(MISSING_VALUE) if np.isfinite(MISSING_VALUE) else "nan",
        },
        "ground_truth_edges_saved": bool(save_edges),
        "n_links": len(link_ids),
        "link_ids": link_ids,
        "files": {
            "npz": str(npz_path),
            "meta": str(outdir / "meta.json"),
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {npz_path}")
    print(f"[OK] Saved: {outdir / 'meta.json'}")
    print(f"    T={T}, N={len(node_ids)}, coords_present={coords_present}, edges_saved={save_edges}")


if __name__ == "__main__":
    main()
