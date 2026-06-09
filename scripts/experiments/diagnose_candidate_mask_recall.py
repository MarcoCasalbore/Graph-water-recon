"""Post-hoc candidate-mask diagnostics for synthetic water runs.

Ground-truth edges are used only for evaluation after a candidate rule is fixed.
The candidate masks themselves are built from coordinates and observed time
series only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_train_synthetic import (
    build_knn_candidate_mask,
    build_local_bridge_candidate_mask,
    build_observable_candidate_mask,
    build_q_dynamic_candidate_mask,
)


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def edge_set_from_edge_index(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical(int(u), int(v))
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if int(u) != int(v)
    }


def edge_length(edge: tuple[int, int], xy: np.ndarray) -> float:
    u, v = edge
    delta = xy[u] - xy[v]
    return float(np.sqrt((delta * delta).sum()))


def build_mask(args: argparse.Namespace, candidate_k: int) -> np.ndarray:
    if args.candidate_mode == "q_dynamic":
        mask = build_q_dynamic_candidate_mask(
            data_npz=str(args.npz),
            pseudo_score_mode=args.pseudo_score_mode,
            q_k=candidate_k,
            partial_ridge=args.partial_ridge,
            geo_tau=args.geo_tau,
            include_mst=args.include_mst,
        )
    elif args.candidate_mode == "observable_union":
        mask = build_observable_candidate_mask(
            data_npz=str(args.npz),
            geo_k=candidate_k,
            signal_k=args.candidate_signal_k,
            partial_ridge=args.partial_ridge,
        )
    elif args.candidate_mode == "local_bridge":
        mask = build_local_bridge_candidate_mask(
            data_npz=str(args.npz),
            geo_k=candidate_k,
            bridge_k=args.pseudo_bridge_candidate_k,
            partial_ridge=args.partial_ridge,
            bridge_quantile=args.pseudo_bridge_quantile,
            bridge_min_temporal=args.pseudo_bridge_min_temporal,
        )
    else:
        data = np.load(args.npz)
        mask = build_knn_candidate_mask(data["node_xy"].astype(np.float32), candidate_k)
    return mask.detach().cpu().numpy().astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--candidate_mode", choices=["geo_knn", "observable_union", "local_bridge", "q_dynamic"], default="q_dynamic")
    parser.add_argument("--pseudo_score_mode", default="geo_global_dynamic_proxy")
    parser.add_argument("--candidate_ks", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--candidate_signal_k", type=int, default=4)
    parser.add_argument("--partial_ridge", type=float, default=0.05)
    parser.add_argument("--geo_tau", type=float, default=1.0)
    parser.add_argument("--include_mst", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo_bridge_candidate_k", type=int, default=4)
    parser.add_argument("--pseudo_bridge_quantile", type=float, default=0.65)
    parser.add_argument("--pseudo_bridge_min_temporal", type=float, default=0.55)
    args = parser.parse_args()

    data = np.load(args.npz)
    gt_edges = edge_set_from_edge_index(data["edge_index"])
    xy = data["node_xy"].astype(np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]

    print("========== CANDIDATE MASK DIAGNOSTIC ==========")
    print(f"npz         : {args.npz}")
    print(f"mode        : {args.candidate_mode}")
    print(f"proxy mode  : {args.pseudo_score_mode}")
    print("GT use      : evaluation-only diagnostics; not for mask construction")
    print("")
    for candidate_k in args.candidate_ks:
        mask = build_mask(args, candidate_k=candidate_k)
        candidate_edges = {
            (i, j)
            for i in range(mask.shape[0])
            for j in range(i + 1, mask.shape[1])
            if bool(mask[i, j])
        }
        hit = gt_edges & candidate_edges
        miss = gt_edges - candidate_edges
        miss_lengths = [edge_length(edge, xy) for edge in miss]
        cand_degree = 2.0 * len(candidate_edges) / max(1, mask.shape[0])
        print(
            f"k={candidate_k:2d} | candidates={len(candidate_edges):3d} "
            f"avg_degree={cand_degree:.2f} | GT recall={len(hit):2d}/{len(gt_edges)} "
            f"= {len(hit) / max(1, len(gt_edges)):.4f} | "
            f"miss_len_median={np.median(miss_lengths) if miss_lengths else 0.0:.4f} "
            f"miss_len_max={max(miss_lengths) if miss_lengths else 0.0:.4f}"
        )


if __name__ == "__main__":
    main()
