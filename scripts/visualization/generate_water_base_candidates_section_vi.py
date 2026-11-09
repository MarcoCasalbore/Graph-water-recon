"""Generate visually distinct, water-like candidates for the Section VI base case."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scripts.synthetic.generate_synthetic_water import (
    build_mst_edges,
    pairwise_dist,
    to_edge_index,
)
from scripts.visualization import generate_water_topologies_section_vi as shared


OUT_DIR = shared.OUT_DIR / "base_candidates"
N = 50


def edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def chain(nodes: list[int]) -> set[tuple[int, int]]:
    return {edge(u, v) for u, v in zip(nodes[:-1], nodes[1:])}


def crosses_existing(
    xy: np.ndarray,
    candidate: tuple[int, int],
    edges: set[tuple[int, int]],
) -> bool:
    u, v = candidate
    for x, y in edges:
        if len({u, v, x, y}) < 4:
            continue
        if shared._segment_crosses(xy[u], xy[v], xy[x], xy[y]):
            return True
    return False


def add_short_planar_ties(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    target_edges: int,
    max_degree: int = 4,
) -> set[tuple[int, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xy)))
    graph.add_edges_from(edges)
    dist = pairwise_dist(xy)
    candidates = sorted(
        (float(dist[u, v]), edge(u, v))
        for u in range(len(xy))
        for v in range(u + 1, len(xy))
        if edge(u, v) not in edges
    )
    for _, candidate in candidates:
        if graph.number_of_edges() >= target_edges:
            break
        u, v = candidate
        if graph.degree[u] >= max_degree or graph.degree[v] >= max_degree:
            continue
        if crosses_existing(xy, candidate, set(graph.edges())):
            continue
        graph.add_edge(u, v)
    return {edge(int(u), int(v)) for u, v in graph.edges()}


def organic_tree_with_ties(seed: int = 101):
    """An irregular distribution tree with a few normally-open tie links."""
    rng = np.random.default_rng(seed)
    cols, rows = 10, 5
    points = []
    for row in range(rows):
        for col in range(cols):
            x = 0.07 + 0.095 * col + rng.normal(0.0, 0.017)
            y = 0.10 + 0.195 * row + 0.055 * np.sin(0.75 * col + row)
            y += rng.normal(0.0, 0.025)
            points.append((x, y))
    xy = np.clip(np.asarray(points, dtype=np.float32), 0.035, 0.965)
    edges = build_mst_edges(pairwise_dist(xy))
    edges = add_short_planar_ties(xy, edges, target_edges=57, max_degree=4)
    return xy, edges


def irregular_street_mesh(seed: int = 202):
    """A sparse street-following mesh with short branches and cul-de-sacs."""
    rng = np.random.default_rng(seed)
    xy = np.zeros((N, 2), dtype=np.float32)
    for row in range(5):
        for col in range(10):
            idx = row * 10 + col
            xy[idx] = (
                0.06 + 0.097 * col + rng.normal(0.0, 0.012),
                0.09 + 0.202 * row + 0.025 * np.sin(0.9 * col) + rng.normal(0.0, 0.015),
            )
    xy = np.clip(xy, 0.035, 0.965)

    edges: set[tuple[int, int]] = set()
    for row in range(5):
        row_nodes = list(range(row * 10, (row + 1) * 10))
        edges |= chain(row_nodes)
    for col in (1, 3, 6, 8):
        for row in range(4):
            if (col + row) % 3 != 0 or col in (3, 8):
                edges.add(edge(row * 10 + col, (row + 1) * 10 + col))
    edges = add_short_planar_ties(xy, edges, target_edges=61, max_degree=4)
    return xy, edges


def dual_main_with_crosslinks(seed: int = 303):
    """Two supply mains joined by cross-links, with short service laterals."""
    rng = np.random.default_rng(seed)
    x_main = np.linspace(0.07, 0.93, 17, dtype=np.float32)
    lower = np.column_stack([x_main, 0.36 + 0.035 * np.sin(2.5 * np.pi * x_main)])
    upper = np.column_stack([x_main, 0.65 + 0.040 * np.sin(2.2 * np.pi * x_main + 0.7)])
    lower += rng.normal(0.0, 0.009, size=lower.shape)
    upper += rng.normal(0.0, 0.009, size=upper.shape)

    laterals = []
    anchors = [(2, -1), (5, 1), (8, -1), (11, 1), (14, -1), (15, 1), (4, -1), (12, 1)]
    for anchor, direction in anchors:
        base = lower[anchor] if direction < 0 else upper[anchor]
        laterals.append(base + np.array([rng.normal(0.0, 0.012), 0.14 * direction]))
        laterals.append(base + np.array([rng.normal(0.0, 0.018), 0.23 * direction]))
    xy = np.clip(np.vstack([lower, upper, laterals]), 0.035, 0.965).astype(np.float32)

    edges = chain(list(range(17))) | chain(list(range(17, 34)))
    for col in (1, 4, 7, 10, 13, 16):
        edges.add(edge(col, 17 + col))
    for branch_idx, (anchor, direction) in enumerate(anchors):
        start = 34 + 2 * branch_idx
        source = anchor if direction < 0 else 17 + anchor
        edges.add(edge(source, start))
        edges.add(edge(start, start + 1))
    edges = add_short_planar_ties(xy, edges, target_edges=58, max_degree=4)
    return xy, edges


def chained_neighborhood_loops(seed: int = 404):
    """Small local loops connected by a sparse distribution backbone."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(
        [(0.18, 0.28), (0.40, 0.68), (0.62, 0.32), (0.82, 0.70)],
        dtype=np.float32,
    )
    sizes = (12, 13, 12, 13)
    xy_parts = []
    labels = []
    for district, (center, size) in enumerate(zip(centers, sizes)):
        angles = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
        radius_x = 0.105 + rng.normal(0.0, 0.008, size=size)
        radius_y = 0.125 + rng.normal(0.0, 0.010, size=size)
        points = np.column_stack(
            [center[0] + radius_x * np.cos(angles), center[1] + radius_y * np.sin(angles)]
        )
        points += rng.normal(0.0, 0.008, size=points.shape)
        xy_parts.append(points)
        labels.extend([district] * size)
    xy = np.clip(np.vstack(xy_parts), 0.035, 0.965).astype(np.float32)

    edges: set[tuple[int, int]] = set()
    offset = 0
    representatives = []
    for size in sizes:
        nodes = list(range(offset, offset + size))
        edges |= chain(nodes + [nodes[0]])
        representatives.append(nodes[0])
        offset += size
    edges |= chain(representatives)
    edges.add(edge(representatives[0], representatives[2]))
    edges = add_short_planar_ties(xy, edges, target_edges=60, max_degree=4)
    return xy, edges


def statistics(xy: np.ndarray, edges: set[tuple[int, int]]) -> dict[str, float | int]:
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    lengths = np.asarray([np.linalg.norm(xy[u] - xy[v]) for u, v in edges])
    degrees = np.asarray([degree for _, degree in graph.degree()])
    return {
        "nodes": N,
        "edges": graph.number_of_edges(),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "bridges": len(list(nx.bridges(graph))),
        "cycle_rank": graph.number_of_edges() - N + 1,
        "crossings": shared._crossing_count(xy, edges),
        "mean_edge_length": float(lengths.mean()),
        "max_edge_length": float(lengths.max()),
        "components": nx.number_connected_components(graph),
    }


def save_candidate(name: str, title: str, xy: np.ndarray, edges: set[tuple[int, int]]) -> dict:
    stats = statistics(xy, edges)
    if stats["components"] != 1:
        raise RuntimeError(f"{name} is disconnected")
    np.savez_compressed(
        OUT_DIR / f"{name}.npz",
        node_xy=xy,
        edge_index=to_edge_index(edges, bidirectional=True),
        block_labels=np.full(N, -1, dtype=np.int64),
    )
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    shared._draw(ax, xy, edges)
    ax.set_title(
        f"{title}\nE={stats['edges']}, cycles={stats['cycle_rank']}, "
        f"bridges={stats['bridges']}, max degree={stats['max_degree']}",
        fontsize=9,
        color=shared.TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(OUT_DIR / f"{name}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    return {"title": title, "statistics": stats}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 8,
        }
    )
    candidates = {
        "base_option_a_organic": ("Organic tree with tie links", organic_tree_with_ties()),
        "base_option_b_street_mesh": ("Irregular suburban street mesh", irregular_street_mesh()),
        "base_option_c_dual_main": ("Dual main with cross-links", dual_main_with_crosslinks()),
        "base_option_d_neighborhoods": ("Chained neighborhood loops", chained_neighborhood_loops()),
    }
    manifest = {
        "selection_protocol": (
            "Generated and selected from visual/structural criteria only. "
            "No DGM outputs, proxy scores, or GT-recovery metrics were used."
        ),
        "candidates": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))
    for ax, (name, (title, (xy, edges))) in zip(axes.flat, candidates.items()):
        manifest["candidates"][name] = save_candidate(name, title, xy, edges)
        shared._draw(ax, xy, edges)
        stats = manifest["candidates"][name]["statistics"]
        ax.set_title(
            f"{title}\nE={stats['edges']}, cycles={stats['cycle_rank']}, "
            f"bridges={stats['bridges']}",
            fontsize=8.5,
            color=shared.TEXT_COLOR,
        )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.96, hspace=0.18, wspace=0.08)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(OUT_DIR / f"water_base_options.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
