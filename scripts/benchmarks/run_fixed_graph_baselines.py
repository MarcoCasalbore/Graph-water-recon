"""Fixed-graph construction + GCN forecasting baselines for the paper protocol.

Graphs are built from coordinates or training observations only. The withheld
reference topology is opened exclusively in posthoc_metrics().
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_protocol import DenseGCN, compute_forecast_metrics, load_observations, make_loaders, seed_all


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def pairwise_distance(xy: np.ndarray) -> np.ndarray:
    delta = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def mst_edges(xy: np.ndarray) -> set[tuple[int, int]]:
    distance = pairwise_distance(xy)
    n = len(xy)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: set[tuple[int, int]] = set()
    for _ in range(n - 1):
        work = distance.copy()
        work[~selected, :] = np.inf
        work[:, selected] = np.inf
        u, v = np.unravel_index(np.argmin(work), work.shape)
        if not np.isfinite(work[u, v]):
            raise RuntimeError("Could not construct a connected Euclidean MST")
        edges.add(canonical(int(u), int(v)))
        selected[v] = True
    return edges


def row_topk_graph(score: np.ndarray, k: int, symmetry: str) -> set[tuple[int, int]]:
    work = np.asarray(score, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)
    indices = np.argsort(-work, axis=1, kind="stable")[:, :int(k)]
    directed = {(u, int(v)) for u in range(len(work)) for v in indices[u]}
    return {
        canonical(u, v)
        for u, v in directed
        if symmetry == "union" or (v, u) in directed
    }


def correlation_score(train_values: np.ndarray) -> np.ndarray:
    """Mean absolute Pearson correlation across variables, fitted on train only."""
    scores = []
    for variable in range(train_values.shape[-1]):
        values = train_values[..., variable].astype(np.float64)
        values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-12)
        score = np.abs(values.T @ values / max(1, len(values) - 1))
        np.fill_diagonal(score, 0.0)
        scores.append(score)
    return np.mean(scores, axis=0)


def build_graph(method: str, options: dict[str, Any], coordinates: np.ndarray, train_values: np.ndarray) -> set[tuple[int, int]]:
    if method == "euclidean_mst":
        return mst_edges(coordinates)
    if method.startswith("geometric_knn"):
        return row_topk_graph(-pairwise_distance(coordinates), options["k"], options["symmetry"])
    if method == "correlation_graph":
        return row_topk_graph(correlation_score(train_values), options["k"], options["symmetry"])
    raise ValueError(f"Unknown method: {method}")


class FixedGraphForecaster(nn.Module):
    """Proposed network capacity with graph generation replaced by a fixed graph."""

    def __init__(self, n: int, lags: int, horizon: int, variables: int, adjacency: torch.Tensor, config: dict[str, Any]):
        super().__init__()
        hidden = int(config["hidden_dim"])
        embedding = int(config["embedding_dim"])
        layers = int(config["message_layers"])
        coordinate_dim = int(config.get("coordinate_dim", 4))
        self.n, self.horizon, self.variables = n, horizon, variables
        self.register_buffer("adjacency", adjacency.float())
        self.temporal = nn.Linear(lags * variables, hidden)
        self.coordinate = nn.Linear(2, coordinate_dim)
        graph_dims = [hidden + coordinate_dim] + [hidden] * max(1, layers - 1) + [embedding]
        self.graph_layers = nn.ModuleList(DenseGCN(graph_dims[i], graph_dims[i + 1]) for i in range(len(graph_dims) - 1))
        forecast_dims = [embedding] + [hidden] * layers
        self.forecast_layers = nn.ModuleList(DenseGCN(forecast_dims[i], forecast_dims[i + 1]) for i in range(layers))
        self.output = nn.Linear(hidden, horizon * variables)

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        temporal = F.relu(self.temporal(x.permute(0, 2, 1, 3).reshape(batch, self.n, -1)))
        coordinate = F.relu(self.coordinate(coordinates)).expand(batch, -1, -1)
        hidden = torch.cat([temporal, coordinate], dim=-1)
        for index, layer in enumerate(self.graph_layers):
            hidden = layer(hidden, self.adjacency)
            if index + 1 < len(self.graph_layers):
                hidden = F.relu(hidden)
        for layer in self.forecast_layers:
            hidden = F.relu(layer(hidden, self.adjacency))
        return self.output(hidden).reshape(batch, self.n, self.horizon, self.variables).permute(0, 2, 1, 3)


def adjacency_from_edges(n: int, edges: set[tuple[int, int]]) -> torch.Tensor:
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1.0
    return adjacency


@torch.no_grad()
def collect(model, loader, coordinates, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, y, _ in loader:
        predictions.append(model(x.to(device), coordinates).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def evaluate_mse(model, loader, coordinates, device) -> float:
    model.eval()
    squared_error = observations = 0.0
    for x, y, _ in loader:
        y = y.to(device)
        squared_error += float(F.mse_loss(model(x.to(device), coordinates), y, reduction="sum").item())
        observations += y.numel()
    return squared_error / observations


def train_run(bundle, edges, config, seed: int, run_dir: Path, force_epochs: int | None) -> dict[str, Any]:
    metrics_path = run_dir / "metrics_gt_free.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("test_evaluated"):
            return metrics
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = int(force_epochs if force_epochs is not None else config["epochs"])
    loaders, generator = make_loaders(bundle, int(config["batch_size"]), seed)
    coordinates = torch.as_tensor(bundle.coordinates, dtype=torch.float32, device=device).unsqueeze(0)
    model = FixedGraphForecaster(
        bundle.values.shape[1],
        bundle.manifest["window"]["lags"],
        bundle.manifest["window"]["forecast_horizon"],
        bundle.values.shape[-1],
        adjacency_from_edges(bundle.values.shape[1], edges),
        config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    best_value, best_epoch, start_epoch, history = float("inf"), -1, 0, []
    last_path = run_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_value, best_epoch = float(state["best_value"]), int(state["best_epoch"])
        start_epoch, history = int(state["epoch"]) + 1, state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        generator.set_state(state["loader_rng"].cpu())
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        total = samples = 0.0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x, coordinates), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 5.0)))
            optimizer.step()
            total += float(loss.item()) * x.shape[0]
            samples += x.shape[0]
        validation_mse = evaluate_mse(model, loaders["val"], coordinates, device)
        history.append({
            "epoch": epoch + 1,
            "train_mse": total / samples,
            "val_mse": validation_mse,
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if validation_mse < best_value:
            best_value, best_epoch = validation_mse, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_mse": best_value}, run_dir / "best.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best_value": best_value, "best_epoch": best_epoch, "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "loader_rng": generator.get_state(),
        }, last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    prediction, target = collect(model, loaders["test"], coordinates, device)
    forecast = compute_forecast_metrics(
        prediction, target, bundle.mean, bundle.std,
        bundle.manifest.get("physical_variable_names", bundle.manifest["variables"]),
    )
    metrics = {
        "seed": seed, "best_epoch": best_epoch, "validation_mse": best_value,
        "forecast": forecast, "num_edges": len(edges), "test_evaluated": True,
        "runtime": {
            "training_seconds_this_invocation": time.perf_counter() - started,
            "training_seconds_total": float(sum(row["epoch_seconds"] for row in history)),
            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_edges(path: Path, edges: set[tuple[int, int]]) -> None:
    path.write_text("source,target\n" + "".join(f"{u},{v}\n" for u, v in sorted(edges)), encoding="utf-8")


def posthoc_metrics(evaluation_dir: Path, predicted: set[tuple[int, int]]) -> dict[str, float | int]:
    """The only function that opens reference edges."""
    reference = np.load(evaluation_dir / "reference_topology.npz")
    n = len(reference["node_ids"])
    truth = {canonical(int(u), int(v)) for u, v in reference["edge_index"].T if int(u) != int(v)}
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    total = n * (n - 1) // 2
    tn = total - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "accuracy": (tp + tn) / total, "recall": recall, "f1": f1}


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_campaign(campaign, output_dir, requested_datasets, requested_methods, seeds, force_epochs):
    rows = []
    for dataset_name, dataset_config in campaign["datasets"].items():
        if requested_datasets and dataset_name not in requested_datasets:
            continue
        observation_dir = PROJECT_ROOT / dataset_config["observation_dir"]
        evaluation_dir = PROJECT_ROOT / dataset_config["evaluation_dir"]
        model_config = json.loads((PROJECT_ROOT / dataset_config["model_config"]).read_text(encoding="utf-8"))["base"]
        bundle = load_observations(observation_dir)
        train_stop = bundle.manifest["splits"]["train"][1]
        for method, options in campaign["methods"].items():
            if requested_methods and method not in requested_methods:
                continue
            method_dir = output_dir / dataset_name / method
            method_dir.mkdir(parents=True, exist_ok=True)
            edges = build_graph(method, options, bundle.coordinates, bundle.values[:train_stop])
            write_edges(method_dir / "graph_edges.csv", edges)
            (method_dir / "graph_config.json").write_text(json.dumps({
                "method": method, "options": options, "fit_split": "train",
                "reference_topology_used": False, "num_edges": len(edges), "model_config": model_config,
            }, indent=2), encoding="utf-8")
            run_metrics = []
            for seed in seeds:
                print(f"[{dataset_name}/{method}] seed={seed}", flush=True)
                run_metrics.append(train_run(bundle, edges, model_config, seed, method_dir / f"seed_{seed}", force_epochs))
            topology = posthoc_metrics(evaluation_dir, edges)
            seed_rows = []
            for metrics in run_metrics:
                test_mse = float(metrics["forecast"]["mse"])
                seed_rows.append({
                    "dataset": dataset_name, "method": method, "seed": metrics["seed"],
                    "mse": test_mse, "mae": float(metrics["forecast"]["mae"]),
                    "precision": topology["precision"], "recall": topology["recall"], "f1": topology["f1"],
                    "S_0.5": 0.5 / (1.0 + test_mse) + 0.5 * float(topology["f1"]),
                    "num_edges": len(edges), "best_epoch": metrics["best_epoch"],
                })
            with (method_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
                writer.writeheader()
                writer.writerows(seed_rows)
            row = {"dataset": dataset_name, "method": method, "num_nodes": bundle.values.shape[1], "num_edges": len(edges), **topology}
            for metric in ["mse", "mae", "precision", "recall", "f1", "S_0.5"]:
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std([float(item[metric]) for item in seed_rows])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/paper_campaign/fixed_graph_baselines.json")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "baseline_joint")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--epochs", type=int, help="Smoke-test override only.")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert mst_edges(xy) == {(0, 1), (1, 2), (2, 3)}
        assert len(row_topk_graph(-pairwise_distance(xy), 1, "union")) == 3
        print("self-check passed")
        return
    campaign = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else campaign["seeds"]
    rows = run_campaign(campaign, args.output_dir, set(args.datasets), set(args.methods), seeds, args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_dir / "baseline_joint_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output_dir / "baseline_joint_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
