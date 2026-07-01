"""Evaluate GT-free decoders from saved DGM probability mean/std matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def infer_dataset(run_name: str, datasets: dict[str, Path]) -> str:
    for name in sorted(datasets, key=len, reverse=True):
        if run_name.startswith(name + "_"):
            return name
    raise ValueError(f"Cannot infer dataset from run name: {run_name}")


def edge_set(edge_index: np.ndarray, n: int) -> set[tuple[int, int]]:
    if edge_index.size == 0:
        return set()
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


def recovery(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def components(n: int, edges: set[tuple[int, int]]) -> int:
    parent = list(range(n))

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
    return len({find(i) for i in range(n)})


def read_task_error(run_dir: Path) -> float:
    path = run_dir / "forecast_metrics.json"
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float((data.get("test") or {}).get("mse", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--dataset_dir", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--row_ks", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--symmetries", nargs="+", choices=["mutual", "union"], default=["mutual"])
    parser.add_argument("--mst_options", nargs="+", type=int, choices=[0, 1], default=[1])
    args = parser.parse_args()

    datasets = {}
    for item in args.dataset_dir:
        name, path = item.split("=", 1)
        datasets[name] = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not datasets:
        raise SystemExit("--dataset_dir NAME=PATH is required")

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        mean_path = run_dir / "dgm_prob_mean.npy"
        if not mean_path.exists():
            mean_path = run_dir / "dgm_soft_score.npy"
        if not mean_path.exists():
            continue
        std_path = run_dir / "dgm_prob_std.npy"
        mean = np.load(mean_path).astype(np.float64)
        std = np.load(std_path).astype(np.float64) if std_path.exists() else np.zeros_like(mean)

        dataset = infer_dataset(run_dir.name, datasets)
        data = np.load(datasets[dataset] / "data.npz")
        xy = data["node_xy"].astype(np.float64)
        n = xy.shape[0]
        gt = edge_set(data["edge_index"], n)
        task_error = read_task_error(run_dir)
        task_score = 1.0 / (1.0 + max(0.0, task_error))

        for beta in args.betas:
            score = mean - beta * std
            score = np.maximum(score, 0.0)
            np.fill_diagonal(score, 0.0)
            for row_k in args.row_ks:
                for symmetry in args.symmetries:
                    for mst in args.mst_options:
                        pred = decode(score, row_k=row_k, symmetry=symmetry, include_mst=bool(mst), xy=xy)
                        rec = recovery(gt, pred)
                        eval_gamma = args.gamma * task_score + (1.0 - args.gamma) * float(rec["f1"])
                        rows.append({
                            "run": run_dir.name,
                            "dataset": dataset,
                            "decoder": f"mean_minus_{beta:g}std_row{row_k}_{symmetry}_{'mst' if mst else 'nomst'}",
                            "beta": beta,
                            "row_k": row_k,
                            "symmetry": symmetry,
                            "mst": int(mst),
                            "task_error": task_error,
                            "task_score": task_score,
                            "eval_gamma": eval_gamma,
                            "num_edges": len(pred),
                            "components": components(n, pred),
                            **rec,
                        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.out_csv}")
    print("| rank | dataset | decoder | task | F1 | P | R | edges | comp | Eval |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(sorted(rows, key=lambda r: r["eval_gamma"], reverse=True)[:20], start=1):
        print(f"| {i} | {row['dataset']} | {row['decoder']} | {row['task_error']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['num_edges']} | {row['components']} | {row['eval_gamma']:.3f} |")


if __name__ == "__main__":
    main()
