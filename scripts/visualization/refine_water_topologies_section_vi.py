"""Refine the sparse base panel while preserving the accepted bridge/loop panels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic.generate_synthetic_water import (
    add_local_edges,
    build_mst_edges,
    pairwise_dist,
    to_edge_index,
)
from scripts.visualization import generate_water_topologies_section_vi as base


def _candidate(seed: int):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.06, 0.94, size=(base.N_NODES, 2)).astype(np.float32)
    dist = pairwise_dist(xy)
    edges = add_local_edges(
        build_mst_edges(dist),
        dist,
        extra_edges=12,
        rng=rng,
    )
    return xy, edges, None


def _edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.T
        if int(u) != int(v)
    }


def main() -> None:
    seed, xy, edges, _, stats = base._select(
        _candidate,
        lambda s: 0.8 * s["crossings"] + 1.5 * abs(s["bridges"] - 12),
    )
    assert stats["components"] == 1
    assert stats["edges"] == 61

    base._save_individual("base", xy, edges)
    np.savez_compressed(
        base.OUT_DIR / "water_topology_base.npz",
        node_xy=xy.astype(np.float32),
        edge_index=to_edge_index(edges, bidirectional=True),
        block_labels=np.full(base.N_NODES, -1, dtype=np.int64),
    )

    scenarios = {"base": (xy, edges)}
    for name in ("bridges", "loopy"):
        saved = np.load(base.OUT_DIR / f"water_topology_{name}.npz")
        scenarios[name] = (saved["node_xy"], _edge_set(saved["edge_index"]))

    titles = {
        "base": "(a) Sparse branching network",
        "bridges": "(b) Bridged communities",
        "loopy": "(c) Loopy communities",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    for ax, name in zip(axes, ("base", "bridges", "loopy")):
        panel_xy, panel_edges = scenarios[name]
        base._draw(ax, panel_xy, panel_edges)
        ax.text(
            0.5,
            -0.035,
            titles[name],
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=base.TEXT_COLOR,
        )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.99, wspace=0.05)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            base.OUT_DIR / f"water_topologies_section_vi.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)

    manifest_path = base.OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenarios"]["base"] = {
        "seed": seed,
        "generator": "mst_local",
        "statistics": stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
