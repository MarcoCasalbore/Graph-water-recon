"""Train the exact forecasting path on fixed, non-oracle graph baselines.

Graph construction never consults the reference topology. Reference edges are
loaded only after construction for post-hoc metrics and for the explicitly
labelled oracle reference-graph forecaster.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
DATASET = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT = RESULTS / "campaign_v2/relational/simple_topological_baselines"

import sys
sys.path.insert(0, str(PROJECT))
from src.config import Parameters  # noqa: E402
from src.dataset.dataset_epanet import EpanetDataModule  # noqa: E402
from src.model.dgm import MLP  # noqa: E402
from src.train_dgm import (  # noqa: E402
    _flatten_for_dgm, _forecasting_loss_ctown, _reshape_ctown_epanet_batch,
)


def canonical(u, v):
    return (u, v) if u < v else (v, u)


def edge_set(edge_index, n=50):
    return {canonical(int(u) % n, int(v) % n) for u, v in edge_index.T if int(u) % n != int(v) % n}


def directed(edges):
    rows = [(u, v) for u, v in sorted(edges)] + [(v, u) for u, v in sorted(edges)]
    return torch.tensor(rows, dtype=torch.long).T.contiguous()


def distance_matrix(xy):
    delta = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def mst_edges(xy):
    dist = distance_matrix(xy)
    g = nx.Graph()
    g.add_nodes_from(range(len(xy)))
    g.add_weighted_edges_from((i, j, float(dist[i, j])) for i in range(len(xy)) for j in range(i + 1, len(xy)))
    return {canonical(u, v) for u, v in nx.minimum_spanning_tree(g).edges()}


def topk_edges(score, k, symmetry):
    score = np.asarray(score, dtype=float).copy()
    np.fill_diagonal(score, -np.inf)
    selected = np.argsort(-score, axis=1)[:, :k]
    directed_pairs = {(i, int(j)) for i in range(len(score)) for j in selected[i]}
    if symmetry == "mutual":
        return {canonical(i, j) for i, j in directed_pairs if (j, i) in directed_pairs}
    return {canonical(i, j) for i, j in directed_pairs}


def rank_similarity_from_distance(dist):
    n = len(dist)
    rank = np.empty_like(dist, dtype=float)
    for i in range(n):
        order = np.argsort(dist[i])
        inverse = np.empty(n, dtype=float)
        inverse[order] = np.arange(n)
        rank[i] = 1.0 - inverse / max(1, n - 1)
    return .5 * (rank + rank.T)


def correlation_score(archive, train_end):
    scores = []
    for key in ["pressure", "demand"]:
        values = archive[key][:train_end].astype(float)
        corr = np.corrcoef(values.T)
        scores.append(np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0))
    score = .5 * (scores[0] + scores[1])
    np.fill_diagonal(score, 0.0)
    return score


def topology_metrics(reference, predicted, n=50):
    tp, fp, fn = len(reference & predicted), len(predicted - reference), len(reference - predicted)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    g = nx.Graph()
    g.add_nodes_from(range(n)); g.add_edges_from(predicted)
    degrees = np.array([d for _, d in g.degree()])
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        "f1": f1, "jaccard": tp / max(1, tp + fp + fn), "edge_count": len(predicted),
        "density": nx.density(g), "mean_degree": degrees.mean(), "min_degree": degrees.min(),
        "max_degree": degrees.max(), "components": nx.number_connected_components(g),
        "isolated_nodes": len(list(nx.isolates(g))), "bridge_count": len(list(nx.bridges(g))),
    }


class FixedDGMForecastPath(nn.Module):
    """DGM forecasting path with only its topology-learning branch removed."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.pre_fc = MLP([input_dim, 32], final_activation=True)
        self.convs = nn.ModuleList([GCNConv(32, 32), GCNConv(32, 16), GCNConv(16, 8)])
        self.fc = nn.Linear(8, output_dim)

    def forward(self, x, edges):
        x = self.pre_fc(x)
        for conv in self.convs:
            x = torch.relu(conv(x, edges))
        return self.fc(x)


def batch_edges(local, batch_size, nodes, device):
    return torch.cat([local.to(device) + b * nodes for b in range(batch_size)], dim=1)


def evaluate(model, loader, local_edges, params, device):
    model.eval()
    sq, absolute, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y, _ = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            bsz = x.shape[0] // params.num_nodes
            pred = model(x, batch_edges(local_edges, bsz, params.num_nodes, device))
            error = pred - y
            sq += float((error * error).sum().item())
            absolute += float(error.abs().sum().item())
            count += error.numel()
    mse = sq / count
    return {"mse": mse, "mae": absolute / count, "rmse": np.sqrt(mse)}


def train_one(method, edges, seed, params, loaders, epochs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedDGMForecastPath(params.lags * 2, params.prediction_window * 2).to(device)
    local = directed(edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=.8)
    train_loader, val_loader, test_loader = loaders
    best_val, best_epoch, best_state, history = float("inf"), 0, None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); total, batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y, _ = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            bsz = x.shape[0] // params.num_nodes
            pred = model(x, batch_edges(local, bsz, params.num_nodes, device))
            loss, _, _ = _forecasting_loss_ctown(pred, y, params)
            loss.backward(); optimizer.step()
            total += float(loss.detach()); batches += 1
        check_now = epoch % max(1, epochs // 20) == 0 or epoch == epochs
        val = evaluate(model, val_loader, local, params, device) if check_now else {"mse": np.nan}
        if check_now:
            scheduler.step(val["mse"])
        if check_now and val["mse"] < best_val:
            best_val, best_epoch = val["mse"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_task_loss": total / max(1, batches),
                        "val_mse": val["mse"], "lr": optimizer.param_groups[0]["lr"]})
    model.load_state_dict(best_state); model.to(device)
    test = evaluate(model, test_loader, local, params, device)
    metrics = {"method": method, "training_seed": seed, "best_epoch": best_epoch,
               "best_val_mse": best_val, **{f"test_{k}": v for k, v in test.items()},
               "runtime_seconds": time.perf_counter() - started, "status": "completed"}
    torch.save(best_state, out_dir / "best_forecaster.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_graphs(archive, meta, seed, learned_path):
    xy = archive["node_xy"].astype(float)
    distance = distance_matrix(xy)
    geo_score = rank_similarity_from_distance(distance)
    train_end = int(meta["splits"]["train"]["end"])
    corr_score = correlation_score(archive, train_end)
    learned = edge_set(np.load(learned_path))
    rng = np.random.default_rng(9000 + seed)
    all_pairs = [(u, v) for u in range(50) for v in range(u + 1, 50)]
    random_control = {all_pairs[int(i)] for i in rng.choice(len(all_pairs), len(learned), replace=False)}
    return {
        "euclidean_mst": mst_edges(xy),
        "geometric_knn3_mutual": topk_edges(geo_score, 3, "mutual"),
        "geometric_knn3_union": topk_edges(geo_score, 3, "union"),
        "correlation_abs_knn3_union": topk_edges(corr_score, 3, "union"),
        "distance_correlation_hybrid_knn3_union": topk_edges(.5 * geo_score + .5 * corr_score, 3, "union"),
        "proposed_final_decoded_graph": learned,
        "random_final_decoded_edge_count_control": random_control,
        "oracle_reference_graph": edge_set(archive["edge_index"]),
    }, corr_score, geo_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lags", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = np.load(DATASET / "data.npz")
    meta = json.loads((DATASET / "meta.json").read_text(encoding="utf-8"))

    class Cli: batch_size = 32; verbose = False
    params = Parameters(Cli())
    params.dataset_name = "synthetic_water"; params.data_npz = str(DATASET / "data.npz")
    params.meta_json = str(DATASET / "meta.json"); params.num_nodes = 50
    params.lags = args.lags; params.prediction_window = args.horizon
    params.input_features = 2; params.output_features = 2
    params.task_loss_type = "huber"; params.task_huber_delta = 1.0
    params.task_feature_normalization = "target_std"; params.task_feature_combine = "weighted_mean"
    params.task_target = "all"; params.pressure_weight = 1.0; params.demand_weight = 1.0
    dm = EpanetDataModule(params)
    loaders = (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())
    reference = edge_set(archive["edge_index"])
    rows, topology_rows = [], []
    for seed in args.seeds:
        learned_dir = ROOT.parent / "main_experiment" / f"main_experiment_g211_s1211_t{seed}_l6_h12_a0p80_kd2_kc3_kq2_e300"
        graphs, corr_score, geo_score = build_graphs(
            archive, meta, seed, learned_dir / "graph_comparison_softprob/learned_edge_index_undirected.npy")
        np.savez_compressed(ROOT / f"baseline_scores_seed{seed}.npz", correlation_abs=corr_score,
                            geometry_rank=geo_score, train_end=int(meta["splits"]["train"]["end"]))
        for method, edges in graphs.items():
            out = ROOT / "runs" / f"{method}_seed{seed}_l{args.lags}_h{args.horizon}_e{args.epochs}"
            config = {"method": method, "training_seed": seed, "lags": args.lags,
                      "horizon": args.horizon, "epochs": args.epochs, "graph_frozen": True,
                      "reference_used_for_construction": method == "oracle_reference_graph",
                      "correlation_fit_end": int(meta["splits"]["train"]["end"]),
                      "k": 3 if "knn3" in method else None,
                      "symmetry": "mutual" if "mutual" in method else "union" if "union" in method else None}
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            pd.DataFrame(sorted(edges), columns=["u", "v"]).to_csv(out / "edges.csv", index=False)
            metric = train_one(method, edges, seed, params, loaders, args.epochs, out)
            topo = topology_metrics(reference, edges)
            rows.append({**metric, **topo, "lags": args.lags, "horizon": args.horizon})
            topology_rows.append({"method": method, "training_seed": seed, **topo})
            pd.DataFrame(rows).to_csv(ROOT / "fixed_graph_forecasting_runs.csv", index=False)
            print(f"[OK] {method} seed={seed} mse={metric['test_mse']:.6f} f1={topo['f1']:.4f}", flush=True)
    pd.DataFrame(topology_rows).to_csv(ROOT / "topology_runs.csv", index=False)
    print(f"[DONE] {len(rows)} fixed-graph runs")


if __name__ == "__main__":
    main()
