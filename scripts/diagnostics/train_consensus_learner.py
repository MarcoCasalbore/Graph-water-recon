"""Train a small GT-free consensus learner on DGM probability statistics.

The learner is post-DGM and uses only quantities produced by the DGM itself:
mean/std of P_ij across validation batches plus ranks/uncertainty derived from
those probabilities. It never uses GT edges, coordinates, T_obs, pressure, or
demand as inputs or training targets. GT is loaded only after training to report
post-hoc metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "water_base": PROJECT_ROOT / "data/synthetic/water_branch_n50_base",
    "water_bridges": PROJECT_ROOT / "data/synthetic/water_branch_n50_bridges",
    "water_loopy": PROJECT_ROOT / "data/synthetic/water_branch_n50_loopy",
    "reverse_base_211": PROJECT_ROOT / "data/synthetic/reverse_base_s211",
    "reverse_bridges_211": PROJECT_ROOT / "data/synthetic/reverse_bridges_s211",
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


def row_rank_score(score: np.ndarray) -> np.ndarray:
    n = score.shape[0]
    out = np.zeros_like(score, dtype=np.float64)
    for i in range(n):
        order = np.argsort(-score[i])
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(n, dtype=np.float64)
        out[i] = 1.0 / (1.0 + rank)
    np.fill_diagonal(out, 0.0)
    return out


def minmax_offdiag(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    mask = ~np.eye(out.shape[0], dtype=bool)
    vals = out[mask]
    finite = np.isfinite(vals)
    if not finite.any():
        out[:] = 0.0
        return out
    lo, hi = vals[finite].min(), vals[finite].max()
    out = (out - lo) / (hi - lo + 1e-12)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, 1.0)


def build_pair_dataset(mean_p: np.ndarray, std_p: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]], np.ndarray]:
    n = mean_p.shape[0]
    mean_norm = minmax_offdiag(mean_p)
    std_norm = minmax_offdiag(std_p)
    stability = 1.0 - std_norm
    entropy = -(mean_norm * np.log(mean_norm + 1e-12) + (1.0 - mean_norm) * np.log(1.0 - mean_norm + 1e-12)) / np.log(2.0)
    confidence = 1.0 - np.clip(entropy, 0.0, 1.0)
    rank = row_rank_score(mean_norm)
    mutual_rank = np.minimum(rank, rank.T)
    mutual_mean = np.minimum(mean_norm, mean_norm.T)

    features = []
    targets = []
    stabilities = []
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            features.append([
                mean_norm[i, j],
                std_norm[i, j],
                stability[i, j],
                confidence[i, j],
                mutual_mean[i, j],
                mutual_rank[i, j],
            ])
            targets.append(mean_norm[i, j])
            stabilities.append(std_norm[i, j])
            pairs.append((i, j))
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
        torch.tensor(stabilities, dtype=torch.float32).unsqueeze(1),
        pairs,
        mean_norm,
    )


class ConsensusMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def initialize_output_density(model: ConsensusMLP, target_degree: float, num_nodes: int) -> None:
    """Start from a sparse graph density instead of the default 0.5 sigmoid output."""
    target_density = float(np.clip(target_degree / max(1, num_nodes - 1), 1e-4, 1.0 - 1e-4))
    bias = math.log(target_density / (1.0 - target_density))
    last = model.net[-1]
    if isinstance(last, nn.Linear):
        with torch.no_grad():
            nn.init.normal_(last.weight, mean=0.0, std=0.01)
            last.bias.fill_(bias)


@dataclass
class TrainResult:
    score: np.ndarray
    loss: float
    recon_loss: float
    stability_loss: float
    degree_loss: float
    entropy_loss: float


def train_consensus(features: torch.Tensor, target: torch.Tensor, std_norm: torch.Tensor, pairs: list[tuple[int, int]], n: int, args) -> TrainResult:
    torch.manual_seed(args.seed)
    model = ConsensusMLP(features.shape[1], hidden_dim=args.hidden_dim)
    target_degree = float(args.target_degree)
    initialize_output_density(model, target_degree=target_degree, num_nodes=n)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    last_parts = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(args.epochs):
        opt.zero_grad()
        s = model(features)
        recon = F.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), target)
        stability = torch.mean(s * std_norm)
        # Expected undirected degree = 2E/N from soft edge probabilities.
        expected_degree = 2.0 * s.sum() / float(n)
        degree = (expected_degree - target_degree).pow(2)
        entropy = -torch.mean(s * torch.log(s + 1e-6) + (1.0 - s) * torch.log(1.0 - s + 1e-6))
        loss = recon + args.lambda_stability * stability + args.lambda_degree * degree + args.lambda_entropy * entropy
        loss.backward()
        opt.step()
        last_parts = (float(loss.item()), float(recon.item()), float(stability.item()), float(degree.item()), float(entropy.item()))

    with torch.no_grad():
        pred = model(features).squeeze(1).numpy()
    score = np.zeros((n, n), dtype=np.float64)
    for value, (i, j) in zip(pred.tolist(), pairs):
        score[i, j] = score[j, i] = float(value)
    return TrainResult(score=score, loss=last_parts[0], recon_loss=last_parts[1], stability_loss=last_parts[2], degree_loss=last_parts[3], entropy_loss=last_parts[4])


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


def recovery(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    jaccard = tp / max(1, tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard}


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


def read_task_error(run_dir: Path) -> float:
    data = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    return float((data.get("test") or {}).get("mse", 0.0))


def evaluate_run(run_dir: Path, args) -> list[dict]:
    dataset = infer_dataset(run_dir.name)
    data = np.load(DATASETS[dataset] / "data.npz")
    xy = data["node_xy"]
    if xy.ndim == 3:
        xy = xy[0]
    n = int(xy.shape[0])
    gt = undirected_from_edge_index(data["edge_index"], n)
    mean_path = run_dir / "dgm_prob_mean.npy"
    std_path = run_dir / "dgm_prob_std.npy"
    if not mean_path.exists() or not std_path.exists():
        # Fallback to old mean-only export; std=0 means no learned stability signal.
        mean_p = np.load(run_dir / "dgm_soft_score.npy")
        std_p = np.zeros_like(mean_p)
    else:
        mean_p = np.load(mean_path)
        std_p = np.load(std_path)

    features, target, std_norm, pairs, _ = build_pair_dataset(mean_p, std_p)
    train = train_consensus(features, target, std_norm, pairs, n, args)
    task_error = read_task_error(run_dir)
    task_score = 1.0 / (1.0 + max(0.0, task_error))

    rows = []
    for row_k in args.row_ks:
        for symmetry in args.symmetries:
            for include_mst in args.mst_options:
                edges = decode(train.score, row_k=row_k, symmetry=symmetry, include_mst=include_mst, xy=xy)
                rec = recovery(gt, edges)
                eval_gamma = args.gamma * task_score + (1.0 - args.gamma) * float(rec["f1"])
                rows.append({
                    "run": run_dir.name,
                    "dataset": dataset,
                    "decoder": f"learner_row{row_k}_{symmetry}_{'mst' if include_mst else 'nomst'}",
                    "task_error": task_error,
                    "task_score": task_score,
                    "eval_gamma": eval_gamma,
                    "num_edges": len(edges),
                    "components": components(n, edges),
                    "train_loss": train.loss,
                    "recon_loss": train.recon_loss,
                    "stability_loss": train.stability_loss,
                    "degree_loss": train.degree_loss,
                    "entropy_loss": train.entropy_loss,
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
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lambda_stability", type=float, default=1.0)
    parser.add_argument("--lambda_degree", type=float, default=0.05)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--target_degree", type=float, default=2.6)
    parser.add_argument("--row_ks", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--symmetries", nargs="+", choices=["mutual", "union"], default=["mutual", "union"])
    parser.add_argument("--mst_options", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    args.mst_options = [bool(x) for x in args.mst_options]

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        if (run_dir / "forecast_metrics.json").exists() and ((run_dir / "dgm_prob_mean.npy").exists() or (run_dir / "dgm_soft_score.npy").exists()):
            rows.extend(evaluate_run(run_dir, args))
    write_csv(args.out_csv, rows)
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.out_csv}")
    print("\nBest consensus learner rows by Eval_gamma")
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
