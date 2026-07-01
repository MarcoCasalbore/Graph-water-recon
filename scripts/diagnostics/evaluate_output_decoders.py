"""Evaluate GT-free final graph decoders for trained DGM runs.

This is a post-hoc diagnostic. It does not train models and does not use GT to
select a decoder. GT edges are loaded only to report final metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "water_base": PROJECT_ROOT / "data/synthetic/water_branch_n50_base",
    "water_bridges": PROJECT_ROOT / "data/synthetic/water_branch_n50_bridges",
    "water_loopy": PROJECT_ROOT / "data/synthetic/water_branch_n50_loopy",
    "reverse_base_211": PROJECT_ROOT / "data/synthetic/reverse_base_s211",
    "reverse_bridges_211": PROJECT_ROOT / "data/synthetic/reverse_bridges_s211",
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def infer_dataset(run_name: str) -> str:
    for name in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(name + "_"):
            return name
    raise ValueError(f"Cannot infer dataset from run name: {run_name}")


def undirected_from_edge_index(edge_index: np.ndarray, num_nodes: int) -> set[tuple[int, int]]:
    if edge_index.size == 0:
        return set()
    src = edge_index[0].astype(int) % num_nodes
    dst = edge_index[1].astype(int) % num_nodes
    return {canonical(int(u), int(v)) for u, v in zip(src, dst) if int(u) != int(v)}


def mst_edges(xy: np.ndarray) -> set[tuple[int, int]]:
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    n = xy.shape[0]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: set[tuple[int, int]] = set()
    for _ in range(n - 1):
        work = dist.copy()
        work[~selected, :] = np.inf
        work[:, selected] = np.inf
        u, v = np.unravel_index(np.argmin(work), work.shape)
        if not np.isfinite(work[u, v]):
            break
        edges.add(canonical(int(u), int(v)))
        selected[v] = True
    return edges


def decode_row_topk(score: np.ndarray, row_k: int, symmetry: str, include_mst: bool, xy: np.ndarray) -> set[tuple[int, int]]:
    n = score.shape[0]
    work = score.copy().astype(float)
    np.fill_diagonal(work, -np.inf)
    k = min(max(1, int(row_k)), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    directed = {(int(i), int(j)) for i in range(n) for j in idx[i]}
    edges = set()
    for u, v in directed:
        if symmetry == "union" or (v, u) in directed:
            edges.add(canonical(u, v))
    if include_mst:
        edges |= mst_edges(xy)
    return edges


def components(num_nodes: int, edges: set[tuple[int, int]]) -> int:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        union(u, v)
    return len({find(i) for i in range(num_nodes)})


def edge_lengths(edges: set[tuple[int, int]], xy: np.ndarray) -> tuple[float, float, float]:
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    vals = [float(np.linalg.norm(xy[u] - xy[v])) for u, v in edges]
    if not vals:
        return 0.0, 0.0, 0.0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


def recovery(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    jaccard = tp / max(1, tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard}


def read_task_error(run_dir: Path) -> float:
    metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    return float((metrics.get("test") or {}).get("mse", 0.0))


def selected_score_stats(edges: set[tuple[int, int]], score: np.ndarray) -> tuple[float, float]:
    if not edges:
        return 0.0, 0.0
    vals = np.asarray([float(score[u, v]) for u, v in edges], dtype=float)
    return float(vals.mean()), float(vals.std())


def evaluate_run(run_dir: Path, gamma: float) -> list[dict]:
    dataset = infer_dataset(run_dir.name)
    data = np.load(DATASETS[dataset] / "data.npz")
    xy = data["node_xy"]
    if xy.ndim == 3:
        xy = xy[0]
    num_nodes = int(xy.shape[0])
    gt = undirected_from_edge_index(data["edge_index"], num_nodes)
    task_error = read_task_error(run_dir)
    task_score = 1.0 / (1.0 + max(0.0, task_error))
    soft_score = np.load(run_dir / "dgm_soft_score.npy")
    decoders: list[tuple[str, str, set[tuple[int, int]]]] = []

    selected_path = run_dir / "dgm_learned_edges.pt"
    if selected_path.exists():
        edge_index = torch.load(selected_path, map_location="cpu").numpy()
        decoders.append(("selected_edges", "checkpoint selected edge sample", undirected_from_edge_index(edge_index, num_nodes)))

    consensus_path = run_dir / "dgm_learned_edges_consensus.pt"
    if consensus_path.exists():
        edge_index = torch.load(consensus_path, map_location="cpu").numpy()
        decoders.append(("consensus_frequency", "edge appears frequently across validation DGM samples", undirected_from_edge_index(edge_index, num_nodes)))

    for row_k, symmetry, include_mst in [
        (2, "mutual", True),
        (3, "mutual", True),
        (3, "mutual", False),
        (2, "union", True),
        (3, "union", True),
        (3, "union", False),
    ]:
        name = f"soft_row{row_k}_{symmetry}_{'mst' if include_mst else 'nomst'}"
        desc = f"mean DGM P_ij, row-top-{row_k}, {symmetry}, MST={include_mst}"
        decoders.append((name, desc, decode_row_topk(soft_score, row_k, symmetry, include_mst, xy)))

    rows = []
    for name, desc, edges in decoders:
        rec = recovery(gt, edges)
        length_mean, length_median, length_max = edge_lengths(edges, xy)
        mean_prob, std_prob = selected_score_stats(edges, soft_score)
        eval_gamma = gamma * task_score + (1.0 - gamma) * float(rec["f1"])
        rows.append({
            "run": run_dir.name,
            "dataset": dataset,
            "decoder": name,
            "decoder_description": desc,
            "task_error": task_error,
            "task_score": task_score,
            "eval_gamma": eval_gamma,
            "num_edges": len(edges),
            "components": components(num_nodes, edges),
            "mean_selected_probability": mean_prob,
            "std_selected_probability": std_prob,
            "edge_length_mean": length_mean,
            "edge_length_median": length_median,
            "edge_length_max": length_max,
            **rec,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        if (run_dir / "forecast_metrics.json").exists() and (run_dir / "dgm_soft_score.npy").exists():
            rows.extend(evaluate_run(run_dir, args.gamma))

    write_csv(args.out_csv, rows)
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.out_csv}")
    print("\nBest decoder rows by Eval_gamma")
    print("| rank | dataset | decoder | task | F1 | P | R | edges | comp | Eval |")
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(sorted(rows, key=lambda r: r["eval_gamma"], reverse=True)[: args.limit], start=1):
        print(
            f"| {i} | {row['dataset']} | {row['decoder']} | {row['task_error']:.4f} | "
            f"{row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['num_edges']} | {row['components']} | {row['eval_gamma']:.3f} |"
        )


if __name__ == "__main__":
    main()
