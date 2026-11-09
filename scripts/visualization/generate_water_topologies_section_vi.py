"""Generate three deterministic reference topologies for Section VI.

The networks are selected only from structural and visual criteria. Model
performance and learned edges are never consulted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scripts.synthetic.generate_synthetic_water import (
    build_erdos_renyi_edges,
    build_sbm_edges,
    pairwise_dist,
    sample_water_branch_positions,
    to_edge_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results_section_vi" / "water_topologies"
N_NODES = 50

NODE_COLOR = "#1F6B82"
EDGE_COLOR = "#77848D"
TEXT_COLOR = "#263238"


def _segment_crosses(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return np.cross(q - p, r - p)

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return bool(o1 * o2 < 0 and o3 * o4 < 0)


def _crossing_count(xy: np.ndarray, edges: set[tuple[int, int]]) -> int:
    edge_list = sorted(edges)
    count = 0
    for i, (u, v) in enumerate(edge_list):
        for x, y in edge_list[i + 1 :]:
            if len({u, v, x, y}) < 4:
                continue
            count += _segment_crosses(xy[u], xy[v], xy[x], xy[y])
    return count


def _statistics(edges: set[tuple[int, int]], xy: np.ndarray) -> dict[str, int | float]:
    graph = nx.Graph()
    graph.add_nodes_from(range(N_NODES))
    graph.add_edges_from(edges)
    distances = pairwise_dist(xy)
    np.fill_diagonal(distances, np.inf)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "bridges": len(list(nx.bridges(graph))),
        "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + 1,
        "crossings": _crossing_count(xy, edges),
        "minimum_node_distance": float(distances.min()),
    }


def _base_candidate(seed: int):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.06, 0.94, size=(N_NODES, 2)).astype(np.float32)
    edges = build_erdos_renyi_edges(
        n=N_NODES,
        p=0.050,
        rng=rng,
        ensure_connected=True,
        dist=pairwise_dist(xy),
        geometric_sigma=0.22,
    )
    return xy, edges, None


def _bridge_candidate(seed: int):
    rng = np.random.default_rng(seed)
    xy, labels = sample_water_branch_positions(
        n=N_NODES,
        num_districts=4,
        rng=rng,
        spread=0.060,
    )
    edges, labels = build_sbm_edges(
        n=N_NODES,
        num_blocks=4,
        p_intra=0.18,
        p_inter=0.002,
        rng=rng,
        ensure_connected=True,
        dist=pairwise_dist(xy),
        geometric_sigma=0.18,
        labels=labels,
    )
    return xy, edges, labels


def _loopy_candidate(seed: int):
    rng = np.random.default_rng(seed)
    xy, labels = sample_water_branch_positions(
        n=N_NODES,
        num_districts=5,
        rng=rng,
        spread=0.065,
    )
    edges, labels = build_sbm_edges(
        n=N_NODES,
        num_blocks=5,
        p_intra=0.30,
        p_inter=0.012,
        rng=rng,
        ensure_connected=True,
        dist=pairwise_dist(xy),
        geometric_sigma=0.22,
        labels=labels,
    )
    return xy, edges, labels


def _select(generator, score_fn):
    best = None
    for seed in range(500):
        xy, edges, labels = generator(seed)
        stats = _statistics(edges, xy)
        if stats["components"] != 1 or stats["minimum_node_distance"] < 0.012:
            continue
        score = score_fn(stats)
        if best is None or score < best[0]:
            best = (score, seed, xy, edges, labels, stats)
    if best is None:
        raise RuntimeError("No valid topology candidate found.")
    _, seed, xy, edges, labels, stats = best
    return seed, xy, edges, labels, stats


def _draw(ax, xy: np.ndarray, edges: set[tuple[int, int]]) -> None:
    for u, v in sorted(edges):
        ax.plot(
            [xy[u, 0], xy[v, 0]],
            [xy[u, 1], xy[v, 1]],
            color=EDGE_COLOR,
            linewidth=0.72,
            alpha=0.72,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=14,
        c=NODE_COLOR,
        edgecolors="white",
        linewidths=0.32,
        zorder=2,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def _save_individual(name: str, xy, edges) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    _draw(ax, xy, edges)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            OUT_DIR / f"water_topology_{name}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 8,
        }
    )

    scenarios = {
        "base": _select(
            _base_candidate,
            lambda s: (
                3.0 * abs(s["edges"] - 60)
                + 0.25 * s["crossings"]
                + abs(s["bridges"] - 3)
            ),
        ),
        "bridges": _select(
            _bridge_candidate,
            lambda s: (
                3.0 * abs(s["edges"] - 62)
                + 0.20 * s["crossings"]
                + 4.0 * max(0, 3 - s["bridges"])
                + abs(s["bridges"] - 5)
            ),
        ),
        "loopy": _select(
            _loopy_candidate,
            lambda s: (
                3.0 * abs(s["edges"] - 75)
                + 0.18 * s["crossings"]
                + 4.0 * max(0, 24 - s["cycle_rank"])
                + 2.0 * max(0, s["bridges"] - 2)
            ),
        ),
    }

    assert scenarios["base"][4]["components"] == 1
    assert scenarios["bridges"][4]["bridges"] >= 3
    assert scenarios["loopy"][4]["cycle_rank"] > scenarios["bridges"][4]["cycle_rank"]

    titles = {
        "base": "(a) Sparse base",
        "bridges": "(b) Bridged communities",
        "loopy": "(c) Loopy communities",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    manifest = {
        "purpose": "Section VI water-network reference topologies",
        "selection": (
            "Seeds selected from structural and visual criteria only; "
            "no model outputs or performance metrics were consulted."
        ),
        "node_count": N_NODES,
        "scenarios": {},
    }

    for ax, (name, (seed, xy, edges, labels, stats)) in zip(axes, scenarios.items()):
        _draw(ax, xy, edges)
        ax.text(
            0.5,
            -0.035,
            titles[name],
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=TEXT_COLOR,
        )
        _save_individual(name, xy, edges)
        np.savez_compressed(
            OUT_DIR / f"water_topology_{name}.npz",
            node_xy=xy.astype(np.float32),
            edge_index=to_edge_index(edges, bidirectional=True),
            block_labels=(
                labels.astype(np.int64)
                if labels is not None
                else np.full(N_NODES, -1, dtype=np.int64)
            ),
        )
        manifest["scenarios"][name] = {
            "seed": seed,
            "generator": "geometric_er" if name == "base" else "geometric_sbm",
            "statistics": stats,
        }

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.99, wspace=0.05)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            OUT_DIR / f"water_topologies_section_vi.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
