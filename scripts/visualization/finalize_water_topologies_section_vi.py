"""Finalize the Section VI topologies with an Erdős-Rényi base network."""

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


def _base_candidate(seed: int):
    rng = np.random.default_rng(seed)
    parts = []
    for branch in range(5):
        t = np.linspace(0.12, 1.0, 10, dtype=np.float32)
        angle = 2.0 * np.pi * branch / 5.0 + 0.14
        angle = angle + 0.22 * np.sin(np.pi * t + branch)
        radius = 0.44 * t
        xy = np.column_stack(
            [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle),
            ]
        )
        parts.append(xy + rng.normal(0.0, 0.018, size=xy.shape))

    node_xy = np.clip(np.vstack(parts), 0.04, 0.96).astype(np.float32)
    distances = pairwise_dist(node_xy)
    edges = build_erdos_renyi_edges(
        n=shared.N_NODES,
        p=0.050,
        rng=rng,
        ensure_connected=True,
        dist=distances,
        geometric_sigma=0.15,
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
            3.0 * abs(s["edges"] - 61)
            + 0.8 * s["crossings"]
            + 1.5 * abs(s["bridges"] - 12)
        ),
    )
    assert stats["components"] == 1
    assert 57 <= stats["edges"] <= 65

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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
