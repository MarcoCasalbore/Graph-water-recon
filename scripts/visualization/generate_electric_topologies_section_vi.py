"""Freeze four plausible SimBench-based electric topologies for Section VI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simbench as sb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.electric.generate_simbench_electric import (  # noqa: E402
    active_topology_edges,
    add_plausible_tie_lines,
    bus_lookup,
    parse_bus_xy,
    to_edge_index,
)


SCENARIOS = {
    "electric_a_semiurban_radial": {
        "code": "1-LV-semiurb4--0-sw",
        "tie_lines": 0,
        "tie_seed": 1101,
        "title": "A - Semi-urban radial LV grid",
    },
    "electric_b_urban_radial": {
        "code": "1-LV-urban6--0-sw",
        "tie_lines": 0,
        "tie_seed": 1102,
        "title": "B - Urban radial LV grid",
    },
    "electric_c_semiurban_tied": {
        "code": "1-LV-semiurb4--0-sw",
        "tie_lines": 12,
        "tie_seed": 1101,
        "title": "C - Semi-urban LV grid with tie-lines",
    },
    "electric_d_urban_tied": {
        "code": "1-LV-urban6--0-sw",
        "tie_lines": 16,
        "tie_seed": 1102,
        "title": "D - Urban LV grid with tie-lines",
    },
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def source_nodes(net, lookup: dict[int, int]) -> list[int]:
    buses = set()
    if len(net.ext_grid):
        buses.update(lookup[int(bus)] for bus in net.ext_grid["bus"].to_numpy(dtype=int))
    return sorted(buses)


def draw_topology(
    xy: np.ndarray,
    edges: set[tuple[int, int]],
    ties: set[tuple[int, int]],
    sources: list[int],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    for u, v in sorted(edges):
        is_tie = canonical(u, v) in ties
        ax.plot(
            [xy[u, 0], xy[v, 0]],
            [xy[u, 1], xy[v, 1]],
            color="#d95f02" if is_tie else "#4b5563",
            linewidth=2.0 if is_tie else 1.25,
            alpha=0.92,
            zorder=1,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=42,
        facecolor="#f8fafc",
        edgecolor="#1f2937",
        linewidth=0.9,
        zorder=2,
    )
    if sources:
        ax.scatter(
            xy[sources, 0],
            xy[sources, 1],
            s=115,
            marker="s",
            facecolor="#2a9d8f",
            edgecolor="#0f172a",
            linewidth=1.0,
            zorder=3,
            label="Grid connection",
        )
    ax.set_title(title, fontsize=15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    if sources:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for scenario, cfg in SCENARIOS.items():
        net = sb.get_simbench_net(str(cfg["code"]))
        lookup = bus_lookup(net)
        xy = parse_bus_xy(net)
        added = add_plausible_tie_lines(
            net,
            node_xy=xy,
            lookup=lookup,
            count=int(cfg["tie_lines"]),
            seed=int(cfg["tie_seed"]),
        )
        ties = {canonical(u, v) for u, v in added}
        edges = active_topology_edges(net, lookup)
        sources = source_nodes(net, lookup)

        np.savez_compressed(
            output_root / f"{scenario}.npz",
            node_xy=xy.astype(np.float32),
            edge_index=to_edge_index(edges, bidirectional=True),
            tie_edge_index=to_edge_index(ties, bidirectional=True),
            source_nodes=np.asarray(sources, dtype=np.int64),
            node_names=np.asarray(net.bus["name"].astype(str).tolist()),
        )
        draw_topology(
            xy,
            edges,
            ties,
            sources,
            str(cfg["title"]),
            figure_root / f"{scenario}_ground_truth.png",
        )
        manifest[scenario] = {
            **cfg,
            "nodes": int(len(net.bus)),
            "edges": int(len(edges)),
            "average_degree": float(2.0 * len(edges) / len(net.bus)),
            "source_nodes": sources,
            "tie_edges": [[int(u), int(v)] for u, v in sorted(ties)],
            "topology_regenerated_during_experiments": False,
        }
        print(
            f"[OK] {scenario}: nodes={len(net.bus)} edges={len(edges)} "
            f"ties={len(ties)}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_topologies",
    )
    args = parser.parse_args()
    generate(args.output_root)


if __name__ == "__main__":
    main()
