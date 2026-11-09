"""Generate visually distinct, electrically plausible topology candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    directed = [(u, v) for u, v in sorted(edges) for u, v in ((u, v), (v, u))]
    return np.asarray(directed, dtype=np.int64).T


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = xy.min(axis=0, keepdims=True)
    hi = xy.max(axis=0, keepdims=True)
    return ((xy - lo) / (hi - lo + 1e-12)).astype(np.float32)


def radial_feeder() -> tuple[np.ndarray, set[tuple[int, int]], list[int]]:
    xy = [(0.0, 0.0)]
    edges: set[tuple[int, int]] = set()

    for node in range(1, 13):
        xy.append((float(node), 0.35 * np.sin(node / 1.8)))
        edges.add((node - 1, node))

    branch_specs = [
        (2, 5, 1),
        (3, 5, -1),
        (5, 6, 1),
        (6, 5, -1),
        (8, 6, 1),
        (9, 5, -1),
        (11, 5, 1),
    ]
    next_node = 13
    for anchor, length, direction in branch_specs:
        parent = anchor
        x0, y0 = xy[anchor]
        for step in range(1, length + 1):
            node = next_node
            next_node += 1
            bend = 0.18 * np.sin(step + anchor)
            xy.append((x0 + 0.24 * step + bend, y0 + direction * 0.72 * step))
            edges.add(canonical(parent, node))
            parent = node

    return normalize_xy(np.asarray(xy)), edges, [0]


def ring_with_laterals() -> tuple[np.ndarray, set[tuple[int, int]], list[int]]:
    xy = [(-1.35, 0.0)]
    edges: set[tuple[int, int]] = {(0, 1)}

    ring_size = 30
    for idx in range(ring_size):
        angle = np.pi + 2.0 * np.pi * idx / ring_size
        xy.append((1.18 * np.cos(angle), 0.76 * np.sin(angle)))
    for node in range(1, ring_size + 1):
        nxt = 1 if node == ring_size else node + 1
        edges.add(canonical(node, nxt))

    anchors = [2, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    next_node = ring_size + 1
    remaining = 19
    for branch_idx, anchor in enumerate(anchors):
        length = 2 if branch_idx < 9 else 1
        length = min(length, remaining)
        remaining -= length
        parent = anchor
        center = np.asarray(xy[anchor])
        direction = center / (np.linalg.norm(center) + 1e-12)
        for step in range(1, length + 1):
            node = next_node
            next_node += 1
            point = center + direction * (0.24 * step)
            xy.append(tuple(point))
            edges.add(canonical(parent, node))
            parent = node

    return normalize_xy(np.asarray(xy)), edges, [0]


def urban_grid() -> tuple[np.ndarray, set[tuple[int, int]], list[int]]:
    rng = np.random.default_rng(1303)
    xy = [(-1.0, 3.0)]
    for row in range(7):
        for col in range(7):
            xy.append(
                (
                    float(col) + rng.normal(0.0, 0.055),
                    float(6 - row) + rng.normal(0.0, 0.055),
                )
            )

    def node(row: int, col: int) -> int:
        return 1 + 7 * row + col

    edges: set[tuple[int, int]] = {(0, node(3, 0))}
    for row in range(7):
        for col in range(6):
            edges.add(canonical(node(row, col), node(row, col + 1)))
    for row in range(6):
        for col in range(7):
            edges.add(canonical(node(row, col), node(row + 1, col)))

    graph = nx.Graph(edges)
    removable = list(edges - {(0, node(3, 0))})
    rng.shuffle(removable)
    for candidate in removable:
        if len(edges) <= 70:
            break
        graph.remove_edge(*candidate)
        if nx.is_connected(graph) and min(dict(graph.degree()).values()) >= 1:
            edges.remove(candidate)
        else:
            graph.add_edge(*candidate)

    return normalize_xy(np.asarray(xy)), edges, [0]


def multi_feeder() -> tuple[np.ndarray, set[tuple[int, int]], list[int]]:
    xy = [(-0.8, 1.0), (-0.8, -1.0)]
    edges: set[tuple[int, int]] = set()
    rows: list[list[int]] = []
    next_node = 2

    for row_idx, y in enumerate((1.65, 0.55, -0.55, -1.65)):
        row = []
        source = 0 if row_idx < 2 else 1
        parent = source
        for step in range(12):
            node = next_node
            next_node += 1
            row.append(node)
            xy.append((float(step), y + 0.08 * np.sin(step + row_idx)))
            edges.add(canonical(parent, node))
            parent = node
        rows.append(row)

    ties = [
        (rows[0][2], rows[1][2]),
        (rows[0][7], rows[1][7]),
        (rows[0][10], rows[1][10]),
        (rows[1][1], rows[2][1]),
        (rows[1][5], rows[2][5]),
        (rows[1][9], rows[2][9]),
        (rows[2][3], rows[3][3]),
        (rows[2][7], rows[3][7]),
        (rows[2][11], rows[3][11]),
        (rows[0][11], rows[3][11]),
    ]
    edges.update(canonical(u, v) for u, v in ties)
    return normalize_xy(np.asarray(xy)), edges, [0, 1]


SCENARIOS = {
    "electric_a_radial_feeder": {
        "title": "A - Radial feeder with laterals",
        "family": "radial",
        "builder": radial_feeder,
    },
    "electric_b_ring_feeder": {
        "title": "B - Ring feeder with service laterals",
        "family": "ring",
        "builder": ring_with_laterals,
    },
    "electric_c_urban_grid": {
        "title": "C - Urban weakly meshed street grid",
        "family": "urban_grid",
        "builder": urban_grid,
    },
    "electric_d_multi_feeder": {
        "title": "D - Interconnected multi-feeder network",
        "family": "multi_feeder",
        "builder": multi_feeder,
    },
}


def draw(
    ax: plt.Axes,
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    sources: list[int],
    title: str,
) -> None:
    for u, v in sorted(edges):
        ax.plot(
            [xy[u, 0], xy[v, 0]],
            [xy[u, 1], xy[v, 1]],
            color="#4b5563",
            linewidth=1.35,
            alpha=0.9,
            zorder=1,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=32,
        facecolor="#f8fafc",
        edgecolor="#1f2937",
        linewidth=0.8,
        zorder=2,
    )
    ax.scatter(
        xy[sources, 0],
        xy[sources, 1],
        s=88,
        marker="s",
        facecolor="#2a9d8f",
        edgecolor="#0f172a",
        linewidth=0.9,
        zorder=3,
    )
    graph = nx.Graph(edges)
    cycles = len(edges) - len(xy) + nx.number_connected_components(graph)
    ax.set_title(f"{title}\nN={len(xy)}, E={len(edges)}, cycles={cycles}", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def generate(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    panels = []

    for name, cfg in SCENARIOS.items():
        xy, edges, sources = cfg["builder"]()
        graph = nx.Graph(edges)
        if len(xy) != 50 or not nx.is_connected(graph):
            raise RuntimeError(f"Invalid candidate topology: {name}")

        np.savez_compressed(
            output_root / f"{name}.npz",
            node_xy=xy,
            edge_index=edge_index(edges),
            source_nodes=np.asarray(sources, dtype=np.int64),
        )
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        draw(ax, xy, edges, sources, str(cfg["title"]))
        fig.tight_layout()
        fig.savefig(output_root / f"{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        cycles = len(edges) - len(xy) + 1
        manifest[name] = {
            "title": cfg["title"],
            "family": cfg["family"],
            "nodes": len(xy),
            "edges": len(edges),
            "average_degree": 2.0 * len(edges) / len(xy),
            "independent_cycles": cycles,
            "source_nodes": sources,
            "status": "topology_candidate_only",
        }
        panels.append((xy, edges, sources, str(cfg["title"])))

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    for ax, panel in zip(axes.flat, panels):
        draw(ax, *panel)
    fig.suptitle("Structurally diverse electric topology candidates", fontsize=17)
    fig.tight_layout()
    fig.savefig(output_root / "electric_topology_candidates_overview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT
        / "results_section_vi"
        / "electric_topology_candidates_diverse",
    )
    args = parser.parse_args()
    generate(args.output_root)


if __name__ == "__main__":
    main()
