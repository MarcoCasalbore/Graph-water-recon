"""Finalize the Section VI figures with a readable geometric ER base network."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic.generate_synthetic_water import (
    build_erdos_renyi_edges,
    pairwise_dist,
    to_edge_index,
)
from scripts.visualization import generate_water_topologies_section_vi as shared


def _polyline(start, end, count, bend, rng):
    t = np.linspace(0.08, 1.0, count, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    xy = start[None, :] + t[:, None] * (end - start)[None, :]
    xy[:, 1] += bend * np.sin(np.pi * t)
    return xy + rng.normal(0.0, 0.010, size=xy.shape)


def _base_candidate(seed: int):
    rng = np.random.default_rng(seed)
    trunk_x = np.linspace(0.06, 0.94, 20, dtype=np.float32)
    trunk = np.column_stack(
        [trunk_x, 0.50 + 0.045 * np.sin(2.0 * np.pi * trunk_x)]
    )
    trunk += rng.normal(0.0, 0.010, size=trunk.shape)

    upper_left = _polyline((0.30, 0.52), (0.10, 0.90), 10, 0.025, rng)
    upper_right = _polyline((0.58, 0.48), (0.72, 0.92), 10, -0.020, rng)
    lower_right = _polyline((0.72, 0.49), (0.91, 0.12), 10, -0.025, rng)
    node_xy = np.clip(
        np.vstack([trunk, upper_left, upper_right, lower_right]),
        0.035,
        0.965,
    ).astype(np.float32)

    distances = pairwise_dist(node_xy)
    edges = build_erdos_renyi_edges(
        n=shared.N_NODES,
        p=0.050,
        rng=rng,
        ensure_connected=True,
        dist=distances,
        geometric_sigma=0.075,
    )
    return node_xy, edges, None


def _edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.T
        if int(u) != int(v)
    }


def main() -> None:
    seed, xy, edges, _, stats = shared._select(
        _base_candidate,
        lambda s: (
            4.0 * abs(s["edges"] - 61)
            + 2.0 * s["crossings"]
            + abs(s["bridges"] - 12)
        ),
    )
    assert stats["components"] == 1
    assert 55 <= stats["edges"] <= 67

    shared._save_individual("base", xy, edges)
    np.savez_compressed(
        shared.OUT_DIR / "water_topology_base.npz",
        node_xy=xy.astype(np.float32),
        edge_index=to_edge_index(edges, bidirectional=True),
        block_labels=np.full(shared.N_NODES, -1, dtype=np.int64),
    )

    scenarios = {"base": (xy, edges)}
    for name in ("bridges", "loopy"):
        saved = np.load(shared.OUT_DIR / f"water_topology_{name}.npz")
        scenarios[name] = (saved["node_xy"], _edge_set(saved["edge_index"]))

    titles = {
        "base": "(a) Sparse branching network",
        "bridges": "(b) Bridged communities",
        "loopy": "(c) Loopy communities",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    for ax, name in zip(axes, ("base", "bridges", "loopy")):
        panel_xy, panel_edges = scenarios[name]
        shared._draw(ax, panel_xy, panel_edges)
        ax.text(
            0.5,
            -0.035,
            titles[name],
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=shared.TEXT_COLOR,
        )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.99, wspace=0.05)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            shared.OUT_DIR / f"water_topologies_section_vi.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)

    manifest_path = shared.OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenarios"]["base"] = {
        "seed": seed,
        "generator": "geometric_er",
        "statistics": stats,
        "coordinate_pattern": "main trunk with three branches",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
