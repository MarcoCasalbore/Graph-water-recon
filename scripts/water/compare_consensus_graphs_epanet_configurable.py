#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_consensus_graphs_epanet_configurable.py

No CLI arguments: edit the CONFIG block at the top.
Run it from the project root:
    python scripts/water/compare_consensus_graphs_epanet_configurable.py

Expected trial layout (example):
    W-DGM/
      trials/
        ctown23/
          dgm_learned_edges.pt
          models/
            dgm_ctown_epanet_best.pt

The script:
  1) reloads the trained DGM model from the trial checkpoint
  2) rebuilds batch-local graphs on the chosen split
  3) constructs four static graphs:
       - union
       - consensus_50
       - consensus_80
       - consensus_100
  4) compares each of them against the EPANET ground truth
  5) saves figures, CSVs, JSON summaries, and one final recap CSV/JSON
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

# =====================================================================
# CONFIG: EDIT ONLY THIS BLOCK
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIAL_NAME = "ctown24"
TRIAL_DIR = PROJECT_ROOT / "trials" / TRIAL_NAME

# Model checkpoint inside the trial folder
MODEL_CKPT = TRIAL_DIR / "models" / "dgm_ctown_epanet_best.pt"

# Optional saved learned-edges file from the usual pipeline (not required here)
LEARNED_EDGES_PT = TRIAL_DIR / "dgm_learned_edges.pt"

# Dataset files
DATA_NPZ = PROJECT_ROOT / "data" / "ctown_ts" / "data.npz"
META_JSON = PROJECT_ROOT / "data" / "ctown_ts" / "meta.json"

# Where all consensus-comparison outputs will be written
OUTDIR = TRIAL_DIR / "consensus_comparison"

# Which split to inspect: "train", "val", "test"
SPLIT = "val"

# How many dataloader batches to inspect. Use None to inspect all.
MAX_BATCHES = 20

# Rebuild exactly the run configuration you want to inspect.
# Update these values to match the trial.
BATCH_SIZE = 32
NUM_NODES = 396
LAGS = 24
PREDICTION_WINDOW = 1
INPUT_FEATURES = 2
OUTPUT_FEATURES = 2

K = 3
K_GEO_CANDIDATES = 3
LAMBDA_GEO = 1e-4
LAMBDA_SMOOTH = 0.0
LAMBDA_GRAPH = 1e-2
LAMBDA_KNN = 0.0
LAMBDA_DEGREE = 0.0
MAX_DEGREE_TARGET = 3
PRESSURE_WEIGHT = 0.3
DEMAND_WEIGHT = 2.0

# Optional: show figures interactively
SHOW = False
# =====================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Parameters
from src.utils.utils import get_model, get_datamodule, set_params_wrt_dataset
from src.train_dgm import _reshape_ctown_epanet_batch, _flatten_for_dgm

Edge = Tuple[int, int]


def canonical_undirected(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def build_knn_candidate_mask(node_xy: np.ndarray, k: int) -> torch.Tensor:
    xy = torch.tensor(node_xy, dtype=torch.float32)
    n = xy.shape[0]

    dist = torch.cdist(xy, xy, p=2)
    eye = torch.eye(n, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    k = min(int(k), n - 1)
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices

    mask = torch.zeros((n, n), dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1).expand_as(nn_idx)
    mask[rows, nn_idx] = True

    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


def safe_positions(node_xy: np.ndarray | None, num_nodes: int) -> Dict[int, Tuple[float, float]]:
    if node_xy is not None:
        xy = np.asarray(node_xy)
        if xy.ndim == 3:
            xy = xy[0]
        if xy.ndim == 2 and xy.shape[0] == num_nodes and xy.shape[1] >= 2:
            finite_mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            if finite_mask.all():
                return {i: (float(xy[i, 0]), float(xy[i, 1])) for i in range(num_nodes)}
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


def draw_two_graphs(pos: Dict[int, Tuple[float, float]], gt_edges: Set[Edge], learned_edges: Set[Edge], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    G_gt = build_nx_graph(len(pos), gt_edges, directed=False)
    G_lr = build_nx_graph(len(pos), learned_edges, directed=False)

    nx.draw(G_gt, pos=pos, ax=axes[0], node_size=10, width=0.5, with_labels=False)
    axes[0].set_title("Ground-truth graph")

    nx.draw(G_lr, pos=pos, ax=axes[1], node_size=10, width=0.5, with_labels=False)
    axes[1].set_title("Learned graph")

    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_error_overlay(pos: Dict[int, Tuple[float, float]], tp_edges: Set[Edge], fp_edges: Set[Edge], fn_edges: Set[Edge], out_path: Path):
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


def edge_set_from_edge_index(edge_index: np.ndarray, directed: bool = True) -> Set[Edge]:
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    if directed:
        return set(zip(src.tolist(), dst.tolist()))
    return set(canonical_undirected(int(u), int(v)) for u, v in zip(src, dst))


def extract_local_graphs_from_edges(edges: torch.Tensor, num_nodes: int) -> List[Set[Edge]]:
    """
    edges: [2, E] with batch-global indexing
    returns one undirected edge-set per logical batch sample.
    """
    edges = edges.detach().cpu().numpy().astype(np.int64)
    src = edges[0]
    dst = edges[1]

    batch_ids = src // num_nodes
    B = int(batch_ids.max()) + 1 if src.size > 0 else 0

    out = []
    for b in range(B):
        mask = (src // num_nodes == b)
        src_b = src[mask] - b * num_nodes
        dst_b = dst[mask] - b * num_nodes

        edge_set = set()
        for u, v in zip(src_b.tolist(), dst_b.tolist()):
            if u == v:
                continue
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                continue
            edge_set.add(canonical_undirected(int(u), int(v)))
        out.append(edge_set)
    return out


def build_consensus_graphs(edge_sets: List[Set[Edge]]) -> Dict[str, Set[Edge]]:
    freq = Counter()
    union_edges: Set[Edge] = set()
    inter_edges = None

    for es in edge_sets:
        union_edges |= es
        if inter_edges is None:
            inter_edges = set(es)
        else:
            inter_edges &= es
        for e in es:
            freq[e] += 1

    if inter_edges is None:
        inter_edges = set()

    n_graphs = len(edge_sets)

    def consensus(thr: float) -> Set[Edge]:
        need = int(np.ceil(thr * n_graphs))
        return {e for e, c in freq.items() if c >= need}

    return {
        "union": union_edges,
        "consensus_50": consensus(0.50),
        "consensus_80": consensus(0.80),
        "consensus_100": inter_edges,
    }


def compare_one_graph(name: str, learned_undir: Set[Edge], gt_undir: Set[Edge], gt_dir: Set[Edge], node_ids, pos, outdir: Path):
    subdir = outdir / name
    subdir.mkdir(parents=True, exist_ok=True)

    learned_dir = set()
    for u, v in learned_undir:
        learned_dir.add((u, v))
        learned_dir.add((v, u))

    gt_dir_stats = graph_stats(gt_dir, len(node_ids), pos, directed=True)
    lr_dir_stats = graph_stats(learned_dir, len(node_ids), pos, directed=True)
    gt_undir_stats = graph_stats(gt_undir, len(node_ids), pos, directed=False)
    lr_undir_stats = graph_stats(learned_undir, len(node_ids), pos, directed=False)

    rec_dir = recovery_metrics(gt_dir, learned_dir)
    rec_undir = recovery_metrics(gt_undir, learned_undir)

    print("\n" + "#" * 90)
    print(f"CONSENSUS GRAPH: {name}")
    print("#" * 90)
    print_stats_block("GROUND-TRUTH GRAPH (DIRECTED VIEW)", gt_dir_stats)
    print_stats_block(f"LEARNED GRAPH (DIRECTED VIEW) [{name}]", lr_dir_stats)
    print_recovery_block(f"EDGE RECOVERY (DIRECTED) [{name}]", rec_dir)
    print_stats_block("GROUND-TRUTH GRAPH (UNDIRECTED VIEW)", gt_undir_stats)
    print_stats_block(f"LEARNED GRAPH (UNDIRECTED VIEW) [{name}]", lr_undir_stats)
    print_recovery_block(f"EDGE RECOVERY (UNDIRECTED) [{name}]", rec_undir)

    tp_undir = gt_undir & learned_undir
    fp_undir = learned_undir - gt_undir
    fn_undir = gt_undir - learned_undir

    write_edge_csv(subdir / "true_positives_undirected.csv", tp_undir, node_ids, pos)
    write_edge_csv(subdir / "false_positives_undirected.csv", fp_undir, node_ids, pos)
    write_edge_csv(subdir / "false_negatives_undirected.csv", fn_undir, node_ids, pos)

    fig_compare = subdir / "graph_compare_undirected.png"
    fig_overlay = subdir / "graph_overlay_errors_undirected.png"
    draw_two_graphs(pos, gt_undir, learned_undir, fig_compare)
    draw_error_overlay(pos, tp_undir, fp_undir, fn_undir, fig_overlay)

    summary = {
        "graph_name": name,
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
            "tp_undirected_csv": str(subdir / "true_positives_undirected.csv"),
            "fp_undirected_csv": str(subdir / "false_positives_undirected.csv"),
            "fn_undirected_csv": str(subdir / "false_negatives_undirected.csv"),
        },
    }
    (subdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    maybe_show_images([fig_compare, fig_overlay], SHOW)

    recap_row = {
        "graph_name": name,
        "num_edges_undirected": lr_undir_stats["num_edges"],
        "num_cc_undirected": lr_undir_stats["num_connected_components_undirected"],
        "degree_mean_undirected": lr_undir_stats["degree_mean"],
        "degree_max_undirected": lr_undir_stats["degree_max"],
        "precision_undirected": rec_undir["precision"],
        "recall_undirected": rec_undir["recall"],
        "f1_undirected": rec_undir["f1"],
        "jaccard_undirected": rec_undir["jaccard"],
        "tp_undirected": rec_undir["tp"],
        "fp_undirected": rec_undir["fp"],
        "fn_undirected": rec_undir["fn"],
    }
    return recap_row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_CKPT.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_CKPT}")
    if not DATA_NPZ.exists():
        raise FileNotFoundError(f"NPZ not found: {DATA_NPZ}")
    if not META_JSON.exists():
        raise FileNotFoundError(f"Meta JSON not found: {META_JSON}")

    print("=" * 100)
    print("CONSENSUS COMPARISON CONFIG")
    print("=" * 100)
    print(f"TRIAL_DIR          : {TRIAL_DIR}")
    print(f"MODEL_CKPT         : {MODEL_CKPT}")
    print(f"LEARNED_EDGES_PT   : {LEARNED_EDGES_PT}")
    print(f"DATA_NPZ           : {DATA_NPZ}")
    print(f"META_JSON          : {META_JSON}")
    print(f"OUTDIR             : {OUTDIR}")
    print(f"SPLIT              : {SPLIT}")
    print(f"MAX_BATCHES        : {MAX_BATCHES}")
    print(f"BATCH_SIZE         : {BATCH_SIZE}")
    print(f"K                  : {K}")
    print(f"K_GEO_CANDIDATES   : {K_GEO_CANDIDATES}")
    print(f"LAMBDA_GRAPH       : {LAMBDA_GRAPH}")
    print(f"LAMBDA_GEO         : {LAMBDA_GEO}")
    print(f"LAMBDA_DEGREE      : {LAMBDA_DEGREE}")
    print(f"PRESSURE_WEIGHT    : {PRESSURE_WEIGHT}")
    print(f"DEMAND_WEIGHT      : {DEMAND_WEIGHT}")
    print("=" * 100)

    class Args:
        batch_size = BATCH_SIZE
        verbose = False

    args = Args()
    run_params = Parameters(args)

    run_params.dataset_name = "ctown_epanet"
    run_params.dataset_path = ""
    run_params.data_npz = str(DATA_NPZ)
    run_params.meta_json = str(META_JSON)

    run_params.num_nodes = NUM_NODES
    run_params.lags = LAGS
    run_params.prediction_window = PREDICTION_WINDOW
    run_params.batch_size = BATCH_SIZE

    run_params.input_features = INPUT_FEATURES
    run_params.output_features = OUTPUT_FEATURES

    run_params.k = K
    run_params.k_geo_candidates = K_GEO_CANDIDATES
    run_params.lambda_geo = LAMBDA_GEO
    run_params.lambda_smooth = LAMBDA_SMOOTH
    run_params.lambda_graph = LAMBDA_GRAPH
    run_params.lambda_knn = LAMBDA_KNN
    run_params.lambda_degree = LAMBDA_DEGREE
    run_params.max_degree_target = MAX_DEGREE_TARGET
    run_params.pressure_weight = PRESSURE_WEIGHT
    run_params.demand_weight = DEMAND_WEIGHT

    train_l, val_l, test_l, run_params = get_datamodule(run_params)
    run_params = set_params_wrt_dataset(run_params, train_l)

    if SPLIT == "train":
        loader = train_l
    elif SPLIT == "val":
        loader = val_l
    elif SPLIT == "test":
        loader = test_l
    else:
        raise ValueError(f"Unknown split: {SPLIT}")

    model = get_model(run_params)

    state = torch.load(MODEL_CKPT, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(
            f"MODEL_CKPT must be a model state_dict. Got {type(state)} at {MODEL_CKPT}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()

    npz = np.load(DATA_NPZ)
    node_xy_np = npz["node_xy"].astype(np.float32)
    candidate_mask = build_knn_candidate_mask(node_xy_np, run_params.k_geo_candidates)

    assigned = 0
    for gf in model.graph_f:
        if hasattr(gf, "candidate_mask"):
            gf.candidate_mask = candidate_mask
            assigned += 1
    print(f"[INFO] Assigned hard kNN mask to {assigned} graph layer(s).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_local_graphs: List[Set[Edge]] = []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if MAX_BATCHES is not None and bidx >= MAX_BATCHES:
                break

            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

            x, y, batch = _reshape_ctown_epanet_batch(batch, run_params, device)
            x, y = _flatten_for_dgm(x, y)

            _ = model(x, edge_index, edge_weight)

            if not hasattr(model, "edges") or model.edges is None:
                raise RuntimeError("model.edges is missing after forward pass")

            local_graphs = extract_local_graphs_from_edges(model.edges, run_params.num_nodes)
            all_local_graphs.extend(local_graphs)
            print(f"[BATCH {bidx}] logical_graphs={len(local_graphs)}")

    if len(all_local_graphs) == 0:
        raise RuntimeError("No local graphs extracted.")

    consensus_graphs = build_consensus_graphs(all_local_graphs)

    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    node_ids = meta["node_ids"]
    num_nodes = len(node_ids)
    gt_edge_index = np.asarray(npz["edge_index"], dtype=np.int64)
    pos = safe_positions(np.asarray(npz["node_xy"], dtype=float), num_nodes)

    gt_dir = edge_set_from_edge_index(gt_edge_index, directed=True)
    gt_undir = edge_set_from_edge_index(gt_edge_index, directed=False)

    recap_rows = []
    for name, learned_undir in consensus_graphs.items():
        recap_rows.append(
            compare_one_graph(
                name=name,
                learned_undir=learned_undir,
                gt_undir=gt_undir,
                gt_dir=gt_dir,
                node_ids=node_ids,
                pos=pos,
                outdir=OUTDIR,
            )
        )

    recap_df = pd.DataFrame(recap_rows)
    recap_df = recap_df.sort_values(
        by=["jaccard_undirected", "f1_undirected", "precision_undirected", "recall_undirected"],
        ascending=[False, False, False, False],
        na_position="last",
    )

    recap_csv = OUTDIR / "consensus_recap.csv"
    recap_json = OUTDIR / "consensus_recap.json"
    recap_df.to_csv(recap_csv, index=False)
    recap_json.write_text(recap_df.to_json(orient="records", indent=2), encoding="utf-8")

    stability_summary = {
        "n_batch_graphs": len(all_local_graphs),
        "union_num_edges": len(consensus_graphs["union"]),
        "consensus_50_num_edges": len(consensus_graphs["consensus_50"]),
        "consensus_80_num_edges": len(consensus_graphs["consensus_80"]),
        "consensus_100_num_edges": len(consensus_graphs["consensus_100"]),
    }
    (OUTDIR / "stability_summary_short.json").write_text(json.dumps(stability_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("FINAL CONSENSUS RECAP")
    print(recap_df.to_string(index=False))
    print("=" * 100)
    print(f"[OK] Saved recap CSV : {recap_csv}")
    print(f"[OK] Saved recap JSON: {recap_json}")


if __name__ == "__main__":
    main()
