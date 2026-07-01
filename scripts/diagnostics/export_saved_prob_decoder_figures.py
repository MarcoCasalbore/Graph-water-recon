"""Export summary figures for GT-free saved-probability decoders.

The decoder is selected from a post-hoc metrics CSV, but the decoded graph itself
uses only saved DGM probability statistics: mean(P_ij) and std(P_ij). Ground truth
is used here only for metrics and error-overlay visualization.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "electric_simple": PROJECT_ROOT / "data/synthetic/electric_simple",
    "electric_intermediate": PROJECT_ROOT / "data/synthetic/electric_intermediate",
    "electric_medium": PROJECT_ROOT / "data/synthetic/electric_medium",
    "electric_loopy": PROJECT_ROOT / "data/synthetic/electric_loopy",
    "electric_mesh_simple": PROJECT_ROOT / "data/synthetic/electric_mesh_simple",
    "electric_mesh_medium": PROJECT_ROOT / "data/synthetic/electric_mesh_medium",
    "electric_synth_branch": PROJECT_ROOT / "data/synthetic/electric_synth_branch",
    "electric_synth_mesh": PROJECT_ROOT / "data/synthetic/electric_synth_mesh",
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def edge_set(edge_index: np.ndarray, n: int) -> set[tuple[int, int]]:
    src = edge_index[0].astype(int) % n
    dst = edge_index[1].astype(int) % n
    return {canonical(int(u), int(v)) for u, v in zip(src, dst) if int(u) != int(v)}


def mst_edges(xy: np.ndarray) -> set[tuple[int, int]]:
    xy = xy[:, :2]
    n = xy.shape[0]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    out: set[tuple[int, int]] = set()
    for _ in range(n - 1):
        work = dist.copy()
        work[~selected, :] = np.inf
        work[:, selected] = np.inf
        u, v = np.unravel_index(np.argmin(work), work.shape)
        if not np.isfinite(work[u, v]):
            break
        out.add(canonical(int(u), int(v)))
        selected[v] = True
    return out


def decode(score: np.ndarray, row_k: int, symmetry: str, include_mst: bool, xy: np.ndarray) -> set[tuple[int, int]]:
    n = score.shape[0]
    work = score.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(max(1, row_k), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    directed = {(int(i), int(j)) for i in range(n) for j in idx[i]}
    edges = set()
    for u, v in directed:
        if symmetry == "union" or (v, u) in directed:
            edges.add(canonical(u, v))
    if include_mst:
        edges |= mst_edges(xy)
    return edges


def parse_decoder(name: str) -> tuple[float, int, str, bool]:
    match = re.fullmatch(r"mean_minus_([0-9.]+)std_row([0-9]+)_(mutual|union)_(mst|nomst)", name)
    if not match:
        raise ValueError(f"Unsupported decoder name: {name}")
    return float(match.group(1)), int(match.group(2)), match.group(3), match.group(4) == "mst"


def draw_pair(pos: dict[int, tuple[float, float]], gt: set[tuple[int, int]], pred: set[tuple[int, int]], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, edges, name in [(axes[0], gt, "Ground truth"), (axes[1], pred, "Reconstructed")]:
        graph = nx.Graph()
        graph.add_nodes_from(pos)
        graph.add_edges_from(edges)
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="#334155", width=0.8, alpha=0.75)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=12, node_color="#111827")
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_overlay(pos: dict[int, tuple[float, float]], gt: set[tuple[int, int]], pred: set[tuple[int, int]], out: Path, title: str) -> None:
    tp = gt & pred
    fp = pred - gt
    fn = gt - pred
    fig, ax = plt.subplots(figsize=(7, 7))
    nodes = nx.Graph()
    nodes.add_nodes_from(pos)
    nx.draw_networkx_nodes(nodes, pos=pos, ax=ax, node_size=12, node_color="#111827")
    for edges, color, width, alpha in [(fn, "#dc2626", 1.0, 0.8), (fp, "#f59e0b", 0.9, 0.8), (tp, "#16a34a", 0.8, 0.65)]:
        if edges:
            graph = nx.Graph()
            graph.add_edges_from(edges)
            nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color=color, width=width, alpha=alpha)
    ax.set_title(title + "\nTP green, FN red, FP orange")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder_csv", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    rows = []
    with args.decoder_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in args.datasets:
                row["eval_gamma"] = float(row["eval_gamma"])
                rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        best = max((row for row in rows if row["dataset"] == dataset), key=lambda row: row["eval_gamma"])
        beta, row_k, symmetry, include_mst = parse_decoder(best["decoder"])
        run_dir = args.run_root / best["run"]
        mean = np.load(run_dir / "dgm_prob_mean.npy").astype(np.float64)
        std = np.load(run_dir / "dgm_prob_std.npy").astype(np.float64)
        score = np.maximum(mean - beta * std, 0.0)
        np.fill_diagonal(score, 0.0)

        data = np.load(DATASETS[dataset] / "data.npz")
        xy = data["node_xy"].astype(np.float64)
        if xy.ndim == 3:
            xy = xy[0]
        n = xy.shape[0]
        gt = edge_set(data["edge_index"], n)
        pred = decode(score, row_k=row_k, symmetry=symmetry, include_mst=include_mst, xy=xy)
        pos = {i: (float(xy[i, 0]), float(xy[i, 1])) for i in range(n)}
        title = f"{dataset}: {best['decoder']} | F1={float(best['f1']):.3f}"
        draw_pair(pos, gt, pred, args.out_dir / f"{dataset}_best_decoder_compare.png", title)
        draw_overlay(pos, gt, pred, args.out_dir / f"{dataset}_best_decoder_overlay.png", title)
        print(f"[OK] {dataset}: {best['decoder']} F1={float(best['f1']):.3f}")


if __name__ == "__main__":
    main()
