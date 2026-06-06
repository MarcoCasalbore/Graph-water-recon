#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example:
#   python scripts/water/compare_graphs_epanet.py \
#     --learned trials/ctown8/dgm_learned_edges.pt \
#     --npz data/ctown_ts/data.npz \
#     --meta data/ctown_ts/meta.json \
#     --outdir trials/ctown8/graph_comparison1 \
#     --show


"""
Compare an EPANET ground-truth graph against a learned DGM graph.

Inputs:
  - learned edges tensor: dgm_learned_edges.pt
  - dataset NPZ: contains edge_index, node_xy
  - meta JSON: contains node_ids

Outputs:
  - printed statistics for both graphs
  - recovery metrics (directed and undirected)
  - CSVs listing true positives / false positives / false negatives
  - side-by-side figure of GT vs learned graph
  - error-overlay figure highlighting missing and extra edges

Example:
  python scripts/water/compare_graphs_epanet.py \
      --learned ./dgm_learned_edges.pt \
      --npz ./data/ctown_ts/data.npz \
      --meta ./data/ctown_ts/meta.json \
      --outdir ./graph_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


Edge = Tuple[int, int]


def canonical_undirected(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def edge_set_from_edge_index(edge_index: np.ndarray, directed: bool = True) -> Set[Edge]:
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    if directed:
        return set(zip(src.tolist(), dst.tolist()))
    return set(canonical_undirected(int(u), int(v)) for u, v in zip(src, dst))


def remap_learned_edges_to_node_range(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Learned edges may be stored with batch-global indices. Reduce them modulo N.
    """
    out = edge_index.copy().astype(np.int64)
    out[0] %= num_nodes
    out[1] %= num_nodes
    return out


def deduplicate_edge_index(edge_index: np.ndarray, directed: bool) -> np.ndarray:
    s = edge_set_from_edge_index(edge_index, directed=directed)
    if not s:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(sorted(s), dtype=np.int64).T
    return arr


def safe_positions(node_xy: np.ndarray | None, num_nodes: int) -> Dict[int, Tuple[float, float]]:
    if node_xy is not None:
        xy = np.asarray(node_xy)
        if xy.ndim == 3:
            xy = xy[0]
        if xy.ndim == 2 and xy.shape[0] == num_nodes and xy.shape[1] >= 2:
            finite_mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            if finite_mask.all():
                return {i: (float(xy[i, 0]), float(xy[i, 1])) for i in range(num_nodes)}
    # fallback spring layout on empty graph
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    return nx.spring_layout(g, seed=42)


def build_nx_graph(num_nodes: int, edges: Iterable[Edge], directed: bool) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    return G


def graph_stats(edge_set: Set[Edge], num_nodes: int, pos: Dict[int, Tuple[float, float]], directed: bool) -> Dict[str, float]:
    if directed:
        G = build_nx_graph(num_nodes, edge_set, directed=True)
        out_deg = np.array([d for _, d in G.out_degree()], dtype=float)
        in_deg = np.array([d for _, d in G.in_degree()], dtype=float)
        deg = out_deg + in_deg
        density = len(edge_set) / max(1, num_nodes * (num_nodes - 1))
    else:
        G = build_nx_graph(num_nodes, edge_set, directed=False)
        deg = np.array([d for _, d in G.degree()], dtype=float)
        out_deg = None
        in_deg = None
        density = len(edge_set) / max(1, (num_nodes * (num_nodes - 1) / 2.0))

    lengths = []
    for u, v in edge_set:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lengths.append(float(np.hypot(x1 - x2, y1 - y2)))
    lengths = np.array(lengths, dtype=float) if lengths else np.array([], dtype=float)

    n_cc = nx.number_connected_components(G.to_undirected())

    stats = {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(edge_set)),
        "density": float(density),
        "degree_mean": float(deg.mean()) if deg.size else 0.0,
        "degree_std": float(deg.std()) if deg.size else 0.0,
        "degree_min": float(deg.min()) if deg.size else 0.0,
        "degree_max": float(deg.max()) if deg.size else 0.0,
        "num_connected_components_undirected": int(n_cc),
        "edge_length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "edge_length_median": float(np.median(lengths)) if lengths.size else 0.0,
        "edge_length_max": float(lengths.max()) if lengths.size else 0.0,
    }

    if directed and out_deg is not None and in_deg is not None:
        reciprocal = sum((v, u) in edge_set for (u, v) in edge_set)
        stats.update(
            {
                "out_degree_mean": float(out_deg.mean()) if out_deg.size else 0.0,
                "in_degree_mean": float(in_deg.mean()) if in_deg.size else 0.0,
                "bidirectional_ratio": float(reciprocal / max(1, len(edge_set))),
            }
        )

    return stats


def recovery_metrics(gt: Set[Edge], learned: Set[Edge]) -> Dict[str, float | int]:
    tp = len(gt & learned)
    fp = len(learned - gt)
    fn = len(gt - learned)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    jaccard = tp / max(1, tp + fp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "pct_gt_recovered": float(100.0 * recall),
        "pct_pred_correct": float(100.0 * precision),
    }


def write_edge_csv(path: Path, edge_set: Set[Edge], node_ids: Sequence[str], pos: Dict[int, Tuple[float, float]]):
    with path.open("w", encoding="utf-8") as f:
        f.write("src_idx,dst_idx,src_id,dst_id,length\n")
        for u, v in sorted(edge_set):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            d = float(np.hypot(x1 - x2, y1 - y2))
            f.write(f"{u},{v},{node_ids[u]},{node_ids[v]},{d}\n")


def print_stats_block(title: str, stats: Dict[str, float]):
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


def print_recovery_block(title: str, metrics: Dict[str, float | int]):
    print(f"\n=== {title} ===")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Jaccard:   {metrics['jaccard']:.4f}")
    print(f"% GT recovered: {metrics['pct_gt_recovered']:.2f}%")
    print(f"% predicted correct: {metrics['pct_pred_correct']:.2f}%")


def draw_two_graphs(
    pos: Dict[int, Tuple[float, float]],
    gt_edges: Set[Edge],
    learned_edges: Set[Edge],
    out_path: Path,
    directed: bool,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    G_gt = build_nx_graph(len(pos), gt_edges, directed=directed)
    G_lr = build_nx_graph(len(pos), learned_edges, directed=directed)

    nx.draw(G_gt, pos=pos, ax=axes[0], node_size=10, width=0.5, with_labels=False)
    axes[0].set_title("Ground-truth graph")

    nx.draw(G_lr, pos=pos, ax=axes[1], node_size=10, width=0.5, with_labels=False)
    axes[1].set_title("Learned graph")

    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_error_overlay(
    pos: Dict[int, Tuple[float, float]],
    tp_edges: Set[Edge],
    fp_edges: Set[Edge],
    fn_edges: Set[Edge],
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    G_nodes = nx.Graph()
    G_nodes.add_nodes_from(range(len(pos)))
    nx.draw_networkx_nodes(G_nodes, pos=pos, ax=ax, node_size=8, node_color="black")

    if fn_edges:
        G_fn = nx.Graph()
        G_fn.add_edges_from(fn_edges)
        nx.draw_networkx_edges(G_fn, pos=pos, ax=ax, edge_color="red", width=0.7, alpha=0.8)
    if fp_edges:
        G_fp = nx.Graph()
        G_fp.add_edges_from(fp_edges)
        nx.draw_networkx_edges(G_fp, pos=pos, ax=ax, edge_color="orange", width=0.7, alpha=0.8)
    if tp_edges:
        G_tp = nx.Graph()
        G_tp.add_edges_from(tp_edges)
        nx.draw_networkx_edges(G_tp, pos=pos, ax=ax, edge_color="green", width=0.7, alpha=0.6)

    ax.set_title("Overlay: correct edges (green), missing GT edges (red), extra learned edges (orange)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def maybe_show_images(paths: List[Path], do_show: bool):
    if not do_show:
        return
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(p.name)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare EPANET ground-truth graph vs learned DGM graph.")
    parser.add_argument("--learned", type=str, default="dgm_learned_edges.pt", help="Path to learned edge tensor [.pt]")
    parser.add_argument("--npz", type=str, default="data/ctown_ts/data.npz", help="Path to dataset .npz")
    parser.add_argument("--meta", type=str, default="data/ctown_ts/meta.json", help="Path to meta.json")
    parser.add_argument("--outdir", type=str, default="graph_comparison", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show figures on screen in addition to saving them")
    args = parser.parse_args()

    learned_path = Path(args.learned).resolve()
    npz_path = Path(args.npz).resolve()
    meta_path = Path(args.meta).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not learned_path.exists():
        raise FileNotFoundError(f"Learned graph file not found: {learned_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)

    if "edge_index" not in npz.files:
        raise ValueError("The NPZ file does not contain 'edge_index'.")

    node_ids = meta["node_ids"]
    num_nodes = len(node_ids)
    gt_edge_index = np.asarray(npz["edge_index"], dtype=np.int64)
    node_xy = np.asarray(npz["node_xy"], dtype=float) if "node_xy" in npz.files else None
    pos = safe_positions(node_xy, num_nodes)

    learned_obj = torch.load(learned_path, map_location="cpu")
    if not isinstance(learned_obj, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {learned_path}, got {type(learned_obj)}")
    learned_edge_index = learned_obj.detach().cpu().numpy().astype(np.int64)
    if learned_edge_index.ndim != 2 or learned_edge_index.shape[0] != 2:
        raise ValueError(f"Expected learned edge_index shape [2, E], got {learned_edge_index.shape}")

    learned_edge_index = remap_learned_edges_to_node_range(learned_edge_index, num_nodes)

    # Directed comparison
    gt_dir = edge_set_from_edge_index(gt_edge_index, directed=True)
    lr_dir = edge_set_from_edge_index(learned_edge_index, directed=True)

    # Undirected comparison, more meaningful for physical pipes
    gt_undir = edge_set_from_edge_index(gt_edge_index, directed=False)
    lr_undir = edge_set_from_edge_index(learned_edge_index, directed=False)

    # Optional deduplicated edge indexes saved for inspection
    np.save(outdir / "gt_edge_index_directed.npy", deduplicate_edge_index(gt_edge_index, directed=True))
    np.save(outdir / "learned_edge_index_directed.npy", deduplicate_edge_index(learned_edge_index, directed=True))
    np.save(outdir / "gt_edge_index_undirected.npy", deduplicate_edge_index(gt_edge_index, directed=False))
    np.save(outdir / "learned_edge_index_undirected.npy", deduplicate_edge_index(learned_edge_index, directed=False))

    gt_dir_stats = graph_stats(gt_dir, num_nodes, pos, directed=True)
    lr_dir_stats = graph_stats(lr_dir, num_nodes, pos, directed=True)
    gt_undir_stats = graph_stats(gt_undir, num_nodes, pos, directed=False)
    lr_undir_stats = graph_stats(lr_undir, num_nodes, pos, directed=False)

    rec_dir = recovery_metrics(gt_dir, lr_dir)
    rec_undir = recovery_metrics(gt_undir, lr_undir)

    print_stats_block("GROUND-TRUTH GRAPH (DIRECTED VIEW)", gt_dir_stats)
    print_stats_block("LEARNED GRAPH (DIRECTED VIEW)", lr_dir_stats)
    print_recovery_block("EDGE RECOVERY (DIRECTED)", rec_dir)

    print_stats_block("GROUND-TRUTH GRAPH (UNDIRECTED VIEW)", gt_undir_stats)
    print_stats_block("LEARNED GRAPH (UNDIRECTED VIEW)", lr_undir_stats)
    print_recovery_block("EDGE RECOVERY (UNDIRECTED)", rec_undir)

    tp_dir = gt_dir & lr_dir
    fp_dir = lr_dir - gt_dir
    fn_dir = gt_dir - lr_dir

    tp_undir = gt_undir & lr_undir
    fp_undir = lr_undir - gt_undir
    fn_undir = gt_undir - lr_undir

    write_edge_csv(outdir / "true_positives_directed.csv", tp_dir, node_ids, pos)
    write_edge_csv(outdir / "false_positives_directed.csv", fp_dir, node_ids, pos)
    write_edge_csv(outdir / "false_negatives_directed.csv", fn_dir, node_ids, pos)

    write_edge_csv(outdir / "true_positives_undirected.csv", tp_undir, node_ids, pos)
    write_edge_csv(outdir / "false_positives_undirected.csv", fp_undir, node_ids, pos)
    write_edge_csv(outdir / "false_negatives_undirected.csv", fn_undir, node_ids, pos)

    fig_compare = outdir / "graph_compare_undirected.png"
    fig_overlay = outdir / "graph_overlay_errors_undirected.png"
    draw_two_graphs(pos, gt_undir, lr_undir, fig_compare, directed=False)
    draw_error_overlay(pos, tp_undir, fp_undir, fn_undir, fig_overlay)

    summary = {
        "ground_truth_directed": gt_dir_stats,
        "learned_directed": lr_dir_stats,
        "edge_recovery_directed": rec_dir,
        "ground_truth_undirected": gt_undir_stats,
        "learned_undirected": lr_undir_stats,
        "edge_recovery_undirected": rec_undir,
        "counts_undirected": {
            "correct_edges": len(tp_undir),
            "missing_edges": len(fn_undir),
            "extra_edges": len(fp_undir),
        },
        "files": {
            "graph_compare_undirected_png": str(fig_compare),
            "graph_overlay_errors_undirected_png": str(fig_overlay),
            "tp_undirected_csv": str(outdir / 'true_positives_undirected.csv'),
            "fp_undirected_csv": str(outdir / 'false_positives_undirected.csv'),
            "fn_undirected_csv": str(outdir / 'false_negatives_undirected.csv'),
        },
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== OUTPUT FILES ===")
    print(f"Summary JSON: {outdir / 'comparison_summary.json'}")
    print(f"Compare figure: {fig_compare}")
    print(f"Overlay figure: {fig_overlay}")
    print(f"TP/FP/FN CSVs written in: {outdir}")

    maybe_show_images([fig_compare, fig_overlay], args.show)


if __name__ == "__main__":
    main()
