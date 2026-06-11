"""Oracle diagnostic: does forecasting prefer the ground-truth graph?

This script trains the same fixed-graph forecaster on several candidate
topologies and compares forecasting metrics against topology recovery metrics.

It does not train DGM. It answers a more basic identifiability question:
if the graph is fixed externally, is the ground-truth topology better for the
forecasting objective than kNN/random/learned alternatives?
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from src.config import Parameters
from src.dataset.dataset_epanet import EpanetDataModule
from src.train_dgm import _flatten_for_dgm, _forecasting_loss_ctown, _reshape_ctown_epanet_batch
from run_proxy_methodology_study import geometry_scores, graph_from_score, minmax_offdiag, mst_edges, smooth_score


class FixedGraphForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.output(x)


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def undirected_set(edge_index: np.ndarray, num_nodes: int) -> set[tuple[int, int]]:
    edges = set()
    for u, v in edge_index.T.astype(int).tolist():
        u %= num_nodes
        v %= num_nodes
        if u != v:
            edges.add(canonical(u, v))
    return edges


def to_directed_edge_index(edges: set[tuple[int, int]]) -> torch.Tensor:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        rows.append((v, u))
    return torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)


def pairwise_dist(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def knn_edges(xy: np.ndarray, k: int) -> set[tuple[int, int]]:
    dist = pairwise_dist(xy)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    edges = set()
    for u in range(xy.shape[0]):
        for v in nn[u]:
            edges.add(canonical(int(u), int(v)))
    return edges


def random_edges(num_nodes: int, num_edges: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    all_edges = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
    idx = rng.choice(len(all_edges), size=num_edges, replace=False)
    return {all_edges[int(i)] for i in idx}


def batch_edge_index(local_edge_index: torch.Tensor, batch_size: int, num_nodes: int, device) -> torch.Tensor:
    local_edge_index = local_edge_index.to(device)
    chunks = []
    for b in range(batch_size):
        chunks.append(local_edge_index + b * num_nodes)
    return torch.cat(chunks, dim=1)


def recovery_metrics(gt_edges: set[tuple[int, int]], pred_edges: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    jaccard = tp / (tp + fp + fn + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def evaluate(model, loader, local_edge_index, params, device) -> dict[str, float]:
    model.eval()
    total = 0.0
    total_p = 0.0
    total_d = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x, y, _ = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            bsz = x.shape[0] // params.num_nodes
            edge_index = batch_edge_index(local_edge_index, bsz, params.num_nodes, device)
            pred = model(x, edge_index)
            loss, loss_p, loss_d = _forecasting_loss_ctown(pred, y, params)
            total += float(loss.detach().item())
            total_p += loss_p
            total_d += loss_d
            n_batches += 1

    return {
        "mse": total / max(1, n_batches),
        "pressure_mse": total_p / max(1, n_batches),
        "demand_mse": total_d / max(1, n_batches),
    }


def train_one_graph(name: str, edges: set[tuple[int, int]], loaders, params, args, device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_loader, val_loader, test_loader = loaders
    local_edge_index = to_directed_edge_index(edges)

    model = FixedGraphForecaster(
        input_dim=params.lags * params.input_features,
        hidden_dim=args.hidden_dim,
        output_dim=params.prediction_window * params.output_features,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y, _ = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            bsz = x.shape[0] // params.num_nodes
            edge_index = batch_edge_index(local_edge_index, bsz, params.num_nodes, device)
            pred = model(x, edge_index)
            loss, _, _ = _forecasting_loss_ctown(pred, y, params)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().item())
            n_batches += 1

        if epoch % args.check_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, local_edge_index, params, device)
            if best_val is None or val["mse"] < best_val["mse"]:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{name}] epoch {epoch:03d} train_mse={running/max(1,n_batches):.4f} val_mse={val['mse']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test = evaluate(model, test_loader, local_edge_index, params, device)
    return {
        "graph_name": name,
        "num_edges": len(edges),
        "best_val_mse": best_val["mse"] if best_val else float("nan"),
        "best_val_pressure_mse": best_val["pressure_mse"] if best_val else float("nan"),
        "best_val_demand_mse": best_val["demand_mse"] if best_val else float("nan"),
        "test_mse": test["mse"],
        "test_pressure_mse": test["pressure_mse"],
        "test_demand_mse": test["demand_mse"],
    }


def load_graph_variants(args, npz, meta: dict, num_nodes: int) -> dict[str, set[tuple[int, int]]]:
    rng = np.random.default_rng(args.seed)
    gt = undirected_set(npz["edge_index"], num_nodes)
    variants = {
        "ground_truth": gt,
        "no_edges": set(),
        "initial_proxy": undirected_set(npz["initial_edge_index"], num_nodes),
        f"knn_{args.knn_k}": knn_edges(npz["node_xy"].astype(np.float32), args.knn_k),
    }
    xy = npz["node_xy"][0] if npz["node_xy"].ndim == 3 else npz["node_xy"]
    train_end = int(meta["splits"]["train"]["end"])
    observable_score = minmax_offdiag(
        geometry_scores(xy)["geo_rank"] * smooth_score(npz["pressure"][:train_end])
    )
    variants["observable_proxy"] = graph_from_score(
        observable_score,
        row_k=args.proxy_row_k,
        symmetry=args.proxy_symmetry,
        mst=mst_edges(xy) if args.proxy_include_mst else None,
    )

    for i in range(1, args.random_repeats + 1):
        variants[f"random_same_edges_{i}"] = random_edges(num_nodes, len(gt), rng)

    learned_path = Path(args.learned_edges) if args.learned_edges else None
    if learned_path and learned_path.exists():
        learned_tensor = torch.load(learned_path, map_location="cpu")
        variants["dgm_learned_static"] = undirected_set(learned_tensor.numpy(), num_nodes)

    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle fixed-graph forecasting diagnostic.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic" / "simple_water_n50")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs" / "oracle_graph_loss_test")
    parser.add_argument("--learned_edges", type=Path, default=PROJECT_ROOT / "outputs" / "synthetic_water_knnmask_k6" / "dgm_learned_edges_consensus.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--check_every", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--knn_k", type=int, default=6)
    parser.add_argument("--random_repeats", type=int, default=2)
    parser.add_argument("--proxy_row_k", type=int, default=3)
    parser.add_argument("--proxy_symmetry", choices=["union", "mutual"], default="mutual")
    parser.add_argument("--proxy_include_mst", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_npz = args.data_dir / "data.npz"
    meta_json = args.data_dir / "meta.json"
    npz = np.load(data_npz)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    num_nodes = int(npz["pressure"].shape[1])

    class CliArgs:
        batch_size = args.batch_size
        verbose = False

    params = Parameters(CliArgs())
    params.dataset_name = "synthetic_water"
    params.data_npz = str(data_npz)
    params.meta_json = str(meta_json)
    params.num_nodes = num_nodes
    params.lags = args.lags
    params.prediction_window = 1
    params.input_features = 2
    params.output_features = 2
    params.pressure_weight = 1.0
    params.demand_weight = 1.0

    dm = EpanetDataModule(params)
    loaders = (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())

    variants = load_graph_variants(args, npz, meta, num_nodes)
    gt_edges = variants["ground_truth"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Dataset: {args.data_dir}")
    print(f"[INFO] Graph variants: {list(variants)}")

    rows = []
    for name, edges in variants.items():
        graph_metrics = recovery_metrics(gt_edges, edges)
        row = train_one_graph(name, edges, loaders, params, args, device)
        row.update({f"topology_{k}": v for k, v in graph_metrics.items()})
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["best_val_mse"])
    csv_path = args.out_dir / "oracle_graph_loss_results.csv"
    json_path = args.out_dir / "oracle_graph_loss_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== ORACLE GRAPH LOSS RESULTS ===")
    for row in rows:
        print(
            f"{row['graph_name']:>18} | val_mse={row['best_val_mse']:.4f} "
            f"| test_mse={row['test_mse']:.4f} | F1={row['topology_f1']:.4f} "
            f"| P={row['topology_precision']:.4f} | R={row['topology_recall']:.4f} "
            f"| E={row['num_edges']}"
        )
    print(f"\n[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {json_path}")


if __name__ == "__main__":
    main()
