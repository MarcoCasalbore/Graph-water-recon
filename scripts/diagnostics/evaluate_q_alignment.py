"""Evaluate whether a learned DGM graph follows an observable Q target.

This diagnostic does two comparisons:
  1. Q top-budget graph vs ground truth.
  2. Learned graph vs Q top-budget graph and vs ground truth.

Ground truth is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from search_edge_scores import (  # noqa: E402
    build_scores,
    distance_matrix,
    load_dataset,
    mst_plus_top_edges,
    rank01,
    top_edges,
)


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def load_learned_edges(path: Path, num_nodes: int) -> set[tuple[int, int]]:
    edge_index = torch.load(path, map_location="cpu")
    if isinstance(edge_index, np.ndarray):
        arr = edge_index
    else:
        arr = edge_index.detach().cpu().numpy()
    if arr.shape[0] != 2:
        arr = arr.T
    edges = set()
    for u, v in arr.T.tolist():
        u = int(u) % num_nodes
        v = int(v) % num_nodes
        if u != v:
            edges.add(canonical(u, v))
    return edges


def metrics(pred: set[tuple[int, int]], target: set[tuple[int, int]]) -> dict[str, float]:
    tp = len(pred & target)
    fp = len(pred - target)
    fn = len(target - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def q_score(scores: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "global_dynamic_proxy":
        return scores["global_dynamic_proxy"]
    if mode == "geo_global_dynamic_proxy":
        return scores["geo_global_dynamic_proxy"]
    if mode in scores:
        return scores[mode]
    raise ValueError(f"Unknown mode {mode}. Available: {sorted(scores)}")


def select_q_edges(score: np.ndarray, xy: np.ndarray, selector: str, budget: int) -> set[tuple[int, int]]:
    dist = distance_matrix(xy)
    if selector == "top_budget":
        return set(top_edges(score, budget))
    if selector == "geo_mst_plus_top":
        return set(mst_plus_top_edges(score, dist, budget, mst_weight=dist))
    if selector == "score_mst_plus_top":
        return set(mst_plus_top_edges(score, dist, budget, mst_weight=1.0 - rank01(score)))
    raise ValueError(f"Unknown selector {selector}")


def graph_components(edges: set[tuple[int, int]], num_nodes: int) -> int:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    return nx.number_connected_components(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mode", default="geo_global_dynamic_proxy")
    parser.add_argument("--selector", choices=["top_budget", "geo_mst_plus_top", "score_mst_plus_top"], default="top_budget")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--lag", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xy, pressure, demand, gt_edges = load_dataset(args.data_dir)
    budget = args.budget if args.budget > 0 else len(gt_edges)
    num_nodes = xy.shape[0]

    scores = build_scores(xy, pressure, demand, ridge=args.ridge, lag=args.lag)
    score = q_score(scores, args.mode)
    q_edges = select_q_edges(score, xy, args.selector, budget)
    learned_edges = load_learned_edges(args.learned, num_nodes)

    rows = [
        {
            "comparison": "q_vs_gt",
            "pred_edges": len(q_edges),
            "target_edges": len(gt_edges),
            "components": graph_components(q_edges, num_nodes),
            **metrics(q_edges, gt_edges),
        },
        {
            "comparison": "learned_vs_gt",
            "pred_edges": len(learned_edges),
            "target_edges": len(gt_edges),
            "components": graph_components(learned_edges, num_nodes),
            **metrics(learned_edges, gt_edges),
        },
        {
            "comparison": "learned_vs_q",
            "pred_edges": len(learned_edges),
            "target_edges": len(q_edges),
            "components": graph_components(learned_edges, num_nodes),
            **metrics(learned_edges, q_edges),
        },
    ]

    with (args.out_dir / "q_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("comparison,f1,precision,recall,tp,fp,fn,pred_edges,target_edges,components")
    for row in rows:
        print(
            f"{row['comparison']},{row['f1']:.4f},{row['precision']:.4f},{row['recall']:.4f},"
            f"{row['tp']},{row['fp']},{row['fn']},{row['pred_edges']},{row['target_edges']},{row['components']}"
        )
    print(f"[OK] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
