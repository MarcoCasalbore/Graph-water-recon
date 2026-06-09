"""Post-hoc diagnostics for learned soft graph scores.

This script is evaluation-only: ground-truth edges are used only after a run is
fixed, to understand whether missed edges were near the learned score cutoff.
Do not use these diagnostics for checkpoint, proxy, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def edge_set_from_edge_index(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical(int(u), int(v))
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if int(u) != int(v)
    }


def load_xy(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    xy = data["node_xy"].astype(np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    return xy[:, :2]


def edge_length(edge: tuple[int, int], xy: np.ndarray) -> float:
    u, v = edge
    delta = xy[u] - xy[v]
    return float(np.sqrt((delta * delta).sum()))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--score_name", default="dgm_soft_score.npy")
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    score = np.load(run_dir / args.score_name)
    data = np.load(args.npz)
    gt_edges = edge_set_from_edge_index(data["edge_index"])
    xy = load_xy(args.npz)
    budget = len(gt_edges)

    learned_path = run_dir / "graph_comparison_softprob" / "learned_edge_index_undirected.npy"
    learned_edges = edge_set_from_edge_index(np.load(learned_path))

    pairs: list[dict] = []
    for i in range(score.shape[0]):
        for j in range(i + 1, score.shape[1]):
            edge = (i, j)
            pairs.append(
                {
                    "src": i,
                    "dst": j,
                    "score": float(score[i, j]),
                    "length": edge_length(edge, xy),
                    "is_gt": edge in gt_edges,
                    "is_learned": edge in learned_edges,
                }
            )
    pairs.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        if row["is_gt"] and row["is_learned"]:
            row["kind"] = "TP"
        elif row["is_gt"]:
            row["kind"] = "FN"
        elif row["is_learned"]:
            row["kind"] = "FP"
        else:
            row["kind"] = "TN"

    by_kind = {kind: [row for row in pairs if row["kind"] == kind] for kind in ["TP", "FP", "FN", "TN"]}
    print("========== SOFT SCORE RANK DIAGNOSTIC ==========")
    print(f"run_dir : {run_dir}")
    print(f"score   : {run_dir / args.score_name}")
    print(f"GT use  : evaluation-only diagnostics; not for model selection")
    print(f"budget  : {budget} undirected edges")
    print("")
    for kind in ["TP", "FP", "FN"]:
        rows = by_kind[kind]
        rank_stats = summarize([float(row["rank"]) for row in rows])
        score_stats = summarize([float(row["score"]) for row in rows])
        length_stats = summarize([float(row["length"]) for row in rows])
        print(
            f"{kind}: n={len(rows):2d} | "
            f"rank median={rank_stats['median']:.1f} mean={rank_stats['mean']:.1f} | "
            f"score median={score_stats['median']:.6f} | "
            f"length median={length_stats['median']:.4f} max={length_stats['max']:.4f}"
        )

    print("")
    for extra in [0, 5, 10, 20, 40, budget]:
        cutoff = budget + extra
        top = pairs[:cutoff]
        gt_in_top = sum(1 for row in top if row["is_gt"])
        print(f"top {cutoff:3d}: GT recall upper bound {gt_in_top}/{budget} = {gt_in_top / max(1, budget):.4f}")

    near_cutoff_fn = [row for row in by_kind["FN"] if row["rank"] <= budget + 20]
    print("")
    print(f"FN within top budget+20: {len(near_cutoff_fn)}")
    for row in near_cutoff_fn[:20]:
        print(
            f"  ({row['src']:02d},{row['dst']:02d}) "
            f"rank={row['rank']:3d} score={row['score']:.6f} length={row['length']:.4f}"
        )

    out_csv = args.out_csv or (run_dir / "soft_score_rank_diagnostics.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "src", "dst", "score", "length", "is_gt", "is_learned", "kind"],
        )
        writer.writeheader()
        writer.writerows(pairs)
    print("")
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
