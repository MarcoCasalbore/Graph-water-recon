"""Paper-compliant GT-free learning primitives for benchmark campaigns."""

from __future__ import annotations

import json
import hashlib
import platform
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.diagnostics.train_consensus_learner import build_pair_dataset, decode, train_consensus


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, start: int, stop: int, lags: int, horizon: int):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = np.arange(start, stop - lags - horizon + 1, dtype=np.int64)
        self.lags = lags
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        start = int(self.starts[index])
        split = start + self.lags
        return self.values[start:split], self.values[split:split + self.horizon], start


@dataclass
class ObservationBundle:
    manifest: dict[str, Any]
    values: np.ndarray
    physical_values: np.ndarray
    coordinates: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    datasets: dict[str, WindowDataset]


def load_observations(observation_dir: Path) -> ObservationBundle:
    """Load observations only. This function has no reference-topology argument."""
    manifest = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("domain") not in {"water", "electric"}:
        raise ValueError("domain must be water or electric")
    variables = manifest["variables"]
    if len(variables) != 2 or len(set(variables)) != 2:
        raise ValueError("Exactly two distinct variables are required, in smooth-signal/load order")
    with np.load(observation_dir / manifest["observation_file"], allow_pickle=False) as data:
        if any("edge" in key.lower() for key in data.files):
            raise ValueError("Observation artifacts must not contain reference edges")
        physical = np.stack([data[name].astype(np.float32) for name in variables], axis=-1)
        xy = data["node_xy"].astype(np.float32)
    if physical.ndim != 3 or physical.shape[1] < 2 or xy.shape != (physical.shape[1], 2):
        raise ValueError("Expected variables [time, nodes] and node_xy [nodes, 2], with at least two nodes")
    if not np.isfinite(physical).all() or not np.isfinite(xy).all():
        raise ValueError("Observations and coordinates must be finite; impute missing values before loading")
    if physical.shape[:2] != (manifest["num_samples"], manifest["num_nodes"]):
        raise ValueError("Manifest sample/node counts do not match observations")
    if len(manifest["node_ids"]) != physical.shape[1] or len(set(manifest["node_ids"])) != physical.shape[1]:
        raise ValueError("node_ids must uniquely identify every observation column")
    lags, horizon = int(manifest["window"]["lags"]), int(manifest["window"]["forecast_horizon"])
    if lags < 1 or horizon < 1:
        raise ValueError("lags and forecast_horizon must be positive")
    previous_stop = 0
    for split in ("train", "val", "test"):
        start, stop = manifest["splits"][split]
        if start != previous_stop or stop > len(physical) or stop - start < lags + horizon:
            raise ValueError(f"Invalid {split} split: ranges must partition time and each contain a complete window")
        previous_stop = stop
    if previous_stop != len(physical):
        raise ValueError("Splits must cover all observations")
    train_start, train_stop = manifest["splits"]["train"]
    train = physical[train_start:train_stop]
    train64 = train.astype(np.float64)
    mean = train64.mean(axis=(0, 1), keepdims=True)
    std = train64.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    values = ((physical - mean) / std).astype(np.float32)
    xy_min = xy.min(axis=0, keepdims=True)
    xy_span = np.maximum(xy.max(axis=0, keepdims=True) - xy_min, 1e-8)
    xy = (xy - xy_min) / xy_span
    lags = int(manifest["window"]["lags"])
    horizon = int(manifest["window"]["forecast_horizon"])
    datasets = {
        split: WindowDataset(values, int(bounds[0]), int(bounds[1]), lags, horizon)
        for split, bounds in manifest["splits"].items()
    }
    return ObservationBundle(manifest, values, physical, xy, mean.reshape(-1), std.reshape(-1), datasets)


def offdiag_minmax(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=torch.bool)
    values = x[..., ~eye]
    lo = values.amin(dim=-1, keepdim=True)
    hi = values.amax(dim=-1, keepdim=True)
    flat = (values - lo) / (hi - lo + 1e-12)
    out = torch.zeros_like(x)
    out[..., ~eye] = flat
    return out.clamp(0.0, 1.0)


def geometry_rank_affinity(distance: torch.Tensor) -> torch.Tensor:
    n = distance.shape[0]
    eye = torch.eye(n, device=distance.device, dtype=torch.bool)
    order = torch.argsort(distance.masked_fill(eye, float("inf")), dim=1)
    ranks = torch.empty_like(order, dtype=distance.dtype)
    ranks.scatter_(1, order, torch.arange(n, device=distance.device, dtype=distance.dtype).expand(n, n))
    score = 1.0 / (1.0 + ranks)
    return offdiag_minmax(0.5 * (score + score.T))


def nonmediated_geometry(distance: torch.Tensor) -> torch.Tensor:
    n = distance.shape[0]
    output = torch.full_like(distance, 0.25)
    for i in range(n):
        for j in range(i + 1, n):
            witness = torch.maximum(distance[i], distance[j]) < distance[i, j] - 1e-12
            witness[i] = witness[j] = False
            if not bool(witness.any()):
                output[i, j] = output[j, i] = 1.0
    output.fill_diagonal_(0.0)
    return output


def standardize_time(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True, unbiased=False) + 1e-6)


def ridge_lag_score(source: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    """Pairwise ridge lag coefficients; leading dimensions are preserved."""
    source = standardize_time(source)[..., :-1, :]
    target = standardize_time(target)[..., 1:, :]
    cross = torch.einsum("...ti,...tj->...ij", source, target)
    denom = source.square().sum(dim=-2)[..., :, None] + ridge
    coefficient = (cross / denom).abs()
    return torch.maximum(coefficient, coefficient.transpose(-1, -2))


def inverse_response_score(source: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    """Inverse operator of the observable increment response; supports batches."""
    if source.shape[-2] < 3:
        n = target.shape[-1]
        return torch.zeros((*target.shape[:-2], n, n), device=target.device, dtype=target.dtype)
    source_diff = standardize_time(source[..., 1:, :] - source[..., :-1, :])
    target_diff = standardize_time(target[..., 1:, :] - target[..., :-1, :])
    n = target.shape[-1]
    identity = torch.eye(n, device=target.device, dtype=target.dtype)
    gram = source_diff.transpose(-1, -2).matmul(source_diff) + float(ridge) * identity
    response = torch.linalg.pinv(gram).matmul(source_diff.transpose(-1, -2).matmul(target_diff))
    inverse = torch.linalg.pinv(response)
    score = torch.maximum(inverse.abs(), inverse.transpose(-1, -2).abs())
    diagonal = torch.eye(n, device=target.device, dtype=torch.bool)
    return score.masked_fill(diagonal, 0.0)


class ObservableWaterPrior:
    """Train-fitted observable prior and fixed row-wise candidate support."""
    def __init__(self, train_values: np.ndarray, coordinates: np.ndarray, candidate_size: int, ridge: float = 0.05):
        values = torch.as_tensor(train_values, dtype=torch.float32)
        xy = torch.as_tensor(coordinates, dtype=torch.float32)
        self.n = values.shape[1]
        self.ridge = float(ridge)
        self.distance = torch.cdist(xy, xy)
        geometry = geometry_rank_affinity(self.distance) * nonmediated_geometry(self.distance)
        pressure = values[..., 0]
        demand = values[..., 1]
        difference = (pressure[:, :, None] - pressure[:, None, :]).abs().mean(dim=0)
        positive = difference[difference > 0]
        self.pressure_scale = float(torch.median(positive).item()) if positive.numel() else 1.0
        pressure_score = torch.exp(-difference / max(self.pressure_scale, 1e-6))
        demand_lag = ridge_lag_score(demand, demand, self.ridge)
        cross_lag = ridge_lag_score(demand, pressure, self.ridge)
        coefficient_values = torch.cat([demand_lag.flatten(), cross_lag.flatten()])
        self.lag_scale = float(torch.quantile(coefficient_values, 0.99).item())
        dynamics = torch.maximum(demand_lag, cross_lag) / max(self.lag_scale, 1e-6)
        aggregate = offdiag_minmax(geometry * pressure_score * (0.1 + 0.9 * dynamics.clamp(0.0, 1.0)))
        eye = torch.eye(self.n, dtype=torch.bool)
        size = min(max(1, int(candidate_size)), self.n - 1)
        indices = torch.topk(aggregate.masked_fill(eye, -1.0), k=size, dim=1).indices
        mask = torch.zeros((self.n, self.n), dtype=torch.bool)
        mask.scatter_(1, indices, True)
        mask.fill_diagonal_(False)
        self.static_geometry = geometry
        self.training_support = aggregate
        self.candidate_mask = mask
        self.candidate_indices = indices

    def to(self, device: torch.device) -> "ObservableWaterPrior":
        self.distance = self.distance.to(device)
        self.static_geometry = self.static_geometry.to(device)
        self.training_support = self.training_support.to(device)
        self.candidate_mask = self.candidate_mask.to(device)
        self.candidate_indices = self.candidate_indices.to(device)
        return self

    def distribution(self, windows: torch.Tensor) -> torch.Tensor:
        pressure = windows[..., 0]
        demand = windows[..., 1]
        difference = (pressure[:, :, :, None] - pressure[:, :, None, :]).abs().mean(dim=1)
        pressure_score = torch.exp(-difference / max(self.pressure_scale, 1e-6))
        demand_lag = ridge_lag_score(demand, demand, self.ridge)
        cross_lag = ridge_lag_score(demand, pressure, self.ridge)
        dynamics = torch.maximum(demand_lag, cross_lag) / max(self.lag_scale, 1e-6)
        affinity = self.static_geometry[None] * pressure_score * (0.1 + 0.9 * dynamics.clamp(0.0, 1.0))
        affinity = affinity.masked_fill(~self.candidate_mask[None], 0.0)
        rows = affinity.sum(dim=-1, keepdim=True)
        uniform = self.candidate_mask.float()[None] / self.candidate_mask.float().sum(dim=-1, keepdim=True)[None]
        return torch.where(rows > 1e-12, affinity / (rows + 1e-12), uniform)


class ObservableElectricPrior:
    """Paper electric prior: geometry, voltage smoothness, inverse P-to-V response."""
    def __init__(self, train_values: np.ndarray, coordinates: np.ndarray, candidate_size: int, ridge: float = 0.5):
        values = torch.as_tensor(train_values, dtype=torch.float32)
        xy = torch.as_tensor(coordinates, dtype=torch.float32)
        self.n = values.shape[1]
        self.ridge = float(ridge)
        self.distance = torch.cdist(xy, xy)
        geometry = geometry_rank_affinity(self.distance)
        voltage, power = values[..., 0], values[..., 1]
        difference = (voltage[:, :, None] - voltage[:, None, :]).abs().mean(dim=0)
        positive = difference[difference > 0]
        self.voltage_scale = float(torch.median(positive).item()) if positive.numel() else 1.0
        voltage_score = torch.exp(-difference / max(self.voltage_scale, 1e-6))
        response = inverse_response_score(power, voltage, self.ridge)
        eye = torch.eye(self.n, dtype=torch.bool)
        response_values = response[~eye]
        self.response_scale = float(torch.quantile(response_values, 0.99).item()) if response_values.numel() else 1.0
        response_score = (response / max(self.response_scale, 1e-6)).clamp(0.0, 1.0)
        aggregate = offdiag_minmax(geometry * voltage_score * response_score)
        size = min(max(1, int(candidate_size)), self.n - 1)
        indices = torch.topk(aggregate.masked_fill(eye, -1.0), k=size, dim=1).indices
        mask = torch.zeros((self.n, self.n), dtype=torch.bool)
        mask.scatter_(1, indices, True)
        mask.fill_diagonal_(False)
        self.static_geometry = geometry
        self.training_support = aggregate
        self.candidate_mask = mask
        self.candidate_indices = indices

    def to(self, device: torch.device) -> "ObservableElectricPrior":
        self.distance = self.distance.to(device)
        self.static_geometry = self.static_geometry.to(device)
        self.training_support = self.training_support.to(device)
        self.candidate_mask = self.candidate_mask.to(device)
        self.candidate_indices = self.candidate_indices.to(device)
        return self

    def distribution(self, windows: torch.Tensor) -> torch.Tensor:
        voltage, power = windows[..., 0], windows[..., 1]
        difference = (voltage[:, :, :, None] - voltage[:, :, None, :]).abs().mean(dim=1)
        voltage_score = torch.exp(-difference / max(self.voltage_scale, 1e-6))
        response = inverse_response_score(power, voltage, self.ridge)
        response_score = (response / max(self.response_scale, 1e-6)).clamp(0.0, 1.0)
        affinity = self.static_geometry[None] * voltage_score * response_score
        affinity = affinity.masked_fill(~self.candidate_mask[None], 0.0)
        rows = affinity.sum(dim=-1, keepdim=True)
        uniform = self.candidate_mask.float()[None] / self.candidate_mask.float().sum(dim=-1, keepdim=True)[None]
        return torch.where(rows > 1e-12, affinity / (rows + 1e-12), uniform)


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[-1]
    identity = torch.eye(n, device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency + identity
    degree = adjacency.sum(dim=-1).clamp_min(1e-6)
    return degree.rsqrt().unsqueeze(-1) * adjacency * degree.rsqrt().unsqueeze(-2)


class DenseGCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() == 2:
            adjacency = adjacency.expand(x.shape[0], -1, -1)
        return self.linear(torch.bmm(normalized_adjacency(adjacency), x))


class PaperDGMForecaster(nn.Module):
    def __init__(self, n: int, lags: int, horizon: int, variables: int, candidate_mask: torch.Tensor, config: dict[str, Any]):
        super().__init__()
        hidden = int(config["hidden_dim"])
        dgm_dim = int(config["embedding_dim"])
        layers = int(config["message_layers"])
        self.n, self.horizon, self.variables = n, horizon, variables
        self.k = int(config["k"])
        self.register_buffer("candidate_mask", candidate_mask.bool())
        candidate_graph = (candidate_mask | candidate_mask.T).float()
        self.register_buffer("candidate_graph", candidate_graph)
        self.temporal = nn.Linear(lags * variables, hidden)
        self.coordinate = nn.Linear(2, int(config.get("coordinate_dim", 4)))
        graph_input = hidden + int(config.get("coordinate_dim", 4))
        graph_dims = [graph_input] + [hidden] * max(1, layers - 1) + [dgm_dim]
        self.graph_layers = nn.ModuleList(DenseGCN(graph_dims[i], graph_dims[i + 1]) for i in range(len(graph_dims) - 1))
        forecast_dims = [dgm_dim] + [hidden] * layers
        self.forecast_layers = nn.ModuleList(DenseGCN(forecast_dims[i], forecast_dims[i + 1]) for i in range(layers))
        self.output = nn.Linear(hidden, horizon * variables)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0)))

    def _sample(self, logits: torch.Tensor, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.masked_fill(~self.candidate_mask[None], float("-inf"))
        work = logits
        if stochastic:
            uniform = torch.rand_like(work).clamp_(1e-6, 1.0 - 1e-6)
            work = work - torch.log(-torch.log(uniform))
        indices = torch.topk(work, k=self.k, dim=-1).indices
        adjacency = torch.zeros_like(logits).scatter(-1, indices, 1.0)
        remaining = self.candidate_mask[None].expand(logits.shape[0], -1, -1).clone()
        log_probability = torch.zeros(logits.shape[:2], device=logits.device)
        for step in range(self.k):
            selected = indices[..., step:step + 1]
            conditional = F.log_softmax(logits.masked_fill(~remaining, -1e9), dim=-1)
            log_probability += conditional.gather(-1, selected).squeeze(-1)
            remaining.scatter_(-1, selected, False)
        return adjacency, log_probability / float(self.k)

    def probabilities(self, x: torch.Tensor, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        temporal = F.relu(self.temporal(x.permute(0, 2, 1, 3).reshape(batch, self.n, -1)))
        coord = F.relu(self.coordinate(coordinates)).expand(batch, -1, -1)
        hidden = torch.cat([temporal, coord], dim=-1)
        for index, layer in enumerate(self.graph_layers):
            hidden = layer(hidden, self.candidate_graph)
            if index + 1 < len(self.graph_layers):
                hidden = F.relu(hidden)
        distance = torch.cdist(hidden, hidden).square()
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        logits = (-distance / temperature).masked_fill(~self.candidate_mask[None], -1e9)
        return hidden, F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, probabilities = self.probabilities(x, coordinates)
        logits = torch.log(probabilities.clamp_min(1e-12))
        adjacency, log_probability = self._sample(logits, stochastic)
        hidden = embedding
        for layer in self.forecast_layers:
            hidden = F.relu(layer(hidden, adjacency))
        output = self.output(hidden).reshape(x.shape[0], self.n, self.horizon, self.variables).permute(0, 2, 1, 3)
        return output, probabilities, log_probability


def make_loaders(bundle: ObservationBundle, batch_size: int, seed: int) -> tuple[dict[str, DataLoader], torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", generator=generator if name == "train" else None, num_workers=0)
        for name, dataset in bundle.datasets.items()
    }
    return loaders, generator


@torch.no_grad()
def precompute_prior_cache(prior: Any, bundle: ObservationBundle, batch_size: int, device: torch.device) -> float:
    """Compute deterministic Q(t) once; cache only its candidate-set values."""
    started = time.perf_counter()
    candidate_count = prior.candidate_indices.shape[1]
    cache = torch.zeros((bundle.values.shape[0], prior.n, candidate_count), dtype=torch.float32)
    for dataset in bundle.datasets.values():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for x, _, starts in loader:
            q = prior.distribution(x.to(device))
            indices = prior.candidate_indices[None].expand(q.shape[0], -1, -1)
            cache[starts.long()] = q.gather(-1, indices).cpu()
    prior.q_cache = cache
    return time.perf_counter() - started


def structural_cross_entropy(probabilities: torch.Tensor, prior: Any, starts: torch.Tensor, device: torch.device) -> torch.Tensor:
    indices = prior.candidate_indices[None].expand(probabilities.shape[0], -1, -1)
    candidate_probabilities = probabilities.gather(-1, indices)
    q = prior.q_cache[starts.long()].to(device)
    return -(q * torch.log(candidate_probabilities.clamp_min(1e-12))).sum(dim=-1).mean()


@torch.no_grad()
def loss_pass(model: PaperDGMForecaster, loader: DataLoader, prior: Any, coordinates: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    forecast_total = structural_total = samples = 0.0
    for x, y, starts in loader:
        x, y = x.to(device), y.to(device)
        prediction, probabilities, _ = model(x, coordinates, stochastic=False)
        structural = structural_cross_entropy(probabilities, prior, starts, device)
        forecast_total += float(F.mse_loss(prediction, y, reduction="sum").item())
        structural_total += float(structural.item()) * x.shape[0]
        samples += x.shape[0]
    observations = samples * model.horizon * model.n * model.variables
    return {"forecast": forecast_total / observations, "structural": structural_total / samples}


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda (also used by PyTorch ROCm)")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU requested but unavailable; install the matching PyTorch build or use --device cpu")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def verify_run_identity(observation_dir: Path, output_dir: Path, config: dict[str, Any], seed: int) -> None:
    manifest_path = observation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest_path.read_bytes())
    with (observation_dir / manifest["observation_file"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    identity = {"observations_sha256": digest.hexdigest(), "config": config, "seed": seed}
    path = output_dir / "run_identity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run inputs/configuration changed; use a new output directory")
    elif any((output_dir / name).exists() for name in ("last.pt", "prior_q_cache.pt", "GRID_COMPLETE")):
        raise ValueError("Unverified legacy checkpoint/cache; use a new output directory")
    else:
        path.write_text(json.dumps(identity, indent=2), encoding="utf-8")


def train_run(observation_dir: Path, output_dir: Path, config: dict[str, Any], seed: int, resume: bool = True, finalize: bool = True) -> dict[str, Any]:
    device = resolve_device(config.get("device", "auto"))
    if int(config["epochs"]) < 1 or int(config["k"]) < 1 or int(config["candidate_size"]) < 1:
        raise ValueError("epochs, k and candidate_size must be positive")
    if not 0.0 <= float(config["alpha"]) <= 1.0:
        raise ValueError("alpha must be between zero and one")
    seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_run_identity(observation_dir, output_dir, config, seed)
    metrics_path = output_dir / "metrics_gt_free.json"
    if resume and metrics_path.exists():
        if (output_dir / ("COMPLETE" if finalize else "GRID_COMPLETE")).exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": str(torch.__version__), "numpy": str(np.__version__),
                   "device": str(device), "cuda": torch.version.cuda, "hip": torch.version.hip}
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    prior_peak_memory = 0.0
    if metrics_path.exists():
        previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        prior_peak_memory = float(previous_metrics.get("runtime", {}).get("peak_gpu_memory_mb", 0.0))
    (output_dir / "config.json").write_text(json.dumps({**config, "seed": seed}, indent=2), encoding="utf-8")
    bundle = load_observations(observation_dir)
    (output_dir / "node_ids.json").write_text(json.dumps(bundle.manifest["node_ids"]), encoding="utf-8")
    lags = int(bundle.manifest["window"]["lags"])
    horizon = int(bundle.manifest["window"]["forecast_horizon"])
    train_start, train_stop = map(int, bundle.manifest["splits"]["train"])
    if bundle.manifest.get("domain") == "electric":
        prior = ObservableElectricPrior(
            bundle.values[train_start:train_stop], bundle.coordinates, int(config["candidate_size"]), float(config.get("electric_prior_ridge", 0.5))
        )
    else:
        prior = ObservableWaterPrior(
            bundle.values[train_start:train_stop], bundle.coordinates, int(config["candidate_size"]), float(config.get("prior_ridge", 0.05))
        )
    if int(config["k"]) > prior.candidate_mask.sum(dim=1).min():
        raise ValueError("k cannot exceed candidate_size")
    prior.to(device)
    coordinates = torch.as_tensor(bundle.coordinates, dtype=torch.float32, device=device).unsqueeze(0)
    model = PaperDGMForecaster(bundle.values.shape[1], lags, horizon, bundle.values.shape[-1], prior.candidate_mask, config).to(device)
    loaders, generator = make_loaders(bundle, int(config["batch_size"]), seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    prior_cache_path = output_dir / "prior_q_cache.pt"
    if prior_cache_path.exists():
        cache_payload = torch.load(prior_cache_path, map_location="cpu", weights_only=True)
        prior.q_cache = cache_payload["q_cache"]
        prior_precompute_seconds = float(cache_payload["precompute_seconds"])
    else:
        prior_precompute_seconds = precompute_prior_cache(prior, bundle, int(config["batch_size"]), device)
        torch.save({"q_cache": prior.q_cache, "precompute_seconds": prior_precompute_seconds}, prior_cache_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    initial = loss_pass(model, loaders["train"], prior, coordinates, device)
    scales = {key: max(float(value), 1e-8) for key, value in initial.items()}
    start_epoch, best_value, best_epoch, baseline = 0, float("inf"), -1, scales["forecast"]
    history: list[dict[str, float]] = []
    last_checkpoint = output_dir / "last.pt"
    if resume and last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_value, best_epoch, baseline = float(state["best_value"]), int(state["best_epoch"]), float(state["baseline"])
        scales, history = state["scales"], state["history"]
        prior_peak_memory = max(prior_peak_memory, float(state.get("peak_gpu_memory_mb", 0.0)))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        generator.set_state(state["loader_rng"].cpu())

    alpha = float(config["alpha"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        forecast_sum = structural_sum = count = 0.0
        epoch_started = time.perf_counter()
        for x, y, starts in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, probabilities, log_probability = model(x, coordinates, stochastic=True)
            per_window = (prediction - y).square().mean(dim=(1, 2, 3))
            forecast = per_window.mean()
            structural = structural_cross_entropy(probabilities, prior, starts, device)
            advantage = per_window.detach() - baseline
            reinforce = (advantage[:, None] * log_probability).mean()
            joint = alpha * forecast / scales["forecast"] + (1.0 - alpha) * structural / scales["structural"]
            optimization_loss = joint + alpha * reinforce / scales["forecast"]
            optimization_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 5.0)))
            optimizer.step()
            baseline = 0.95 * baseline + 0.05 * float(per_window.mean().item())
            forecast_sum += float(forecast.item()) * x.shape[0]
            structural_sum += float(structural.item()) * x.shape[0]
            count += x.shape[0]

        validation = loss_pass(model, loaders["val"], prior, coordinates, device)
        val_joint = alpha * validation["forecast"] / scales["forecast"] + (1.0 - alpha) * validation["structural"] / scales["structural"]
        record = {
            "epoch": epoch + 1,
            "train_forecast": forecast_sum / count,
            "train_structural": structural_sum / count,
            "val_forecast": validation["forecast"],
            "val_structural": validation["structural"],
            "val_joint": val_joint,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if val_joint < best_value:
            best_value, best_epoch = val_joint, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_joint": best_value, "scales": scales}, output_dir / "best.pt")
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best_value": best_value, "best_epoch": best_epoch, "baseline": baseline, "scales": scales, "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None, "loader_rng": generator.get_state(),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        }
        torch.save(state, last_checkpoint)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    training_seconds_total = float(sum(row["epoch_seconds"] for row in history))
    current_peak_memory = float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    peak_gpu_memory_mb = max(prior_peak_memory, current_peak_memory)
    grid_metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_joint": best_value,
        "reference_scales": scales,
        "runtime": {
            "training_seconds_this_invocation": time.perf_counter() - started,
            "training_seconds_total": training_seconds_total,
            "prior_precompute_seconds": prior_precompute_seconds,
            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
        "test_evaluated": False,
    }
    (output_dir / "metrics_gt_free.json").write_text(json.dumps(grid_metrics, indent=2), encoding="utf-8")
    (output_dir / "GRID_COMPLETE").write_text("train-validation complete\n", encoding="utf-8")
    if not finalize:
        return grid_metrics

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_losses = loss_pass(model, loaders["test"], prior, coordinates, device)
    predictions, targets = collect_forecasts(model, loaders["test"], coordinates, device)
    forecast_metrics = compute_forecast_metrics(predictions, targets, bundle.mean, bundle.std, bundle.manifest.get("physical_variable_names", bundle.manifest["variables"]))
    probabilities, inference_seconds = collect_probabilities(model, loaders["train"], coordinates, device)
    mean_p, std_p = consensus_statistics(probabilities)
    consensus_started = time.perf_counter()
    features, target, std_norm, pairs, _ = build_pair_dataset(mean_p, std_p)
    consensus_args = SimpleNamespace(
        seed=seed, hidden_dim=16, target_degree=float(config.get("consensus_target_degree", 4.0)),
        lr=1e-2, weight_decay=1e-4, epochs=800, lambda_stability=0.25, lambda_degree=0.05, lambda_entropy=0.01,
    )
    consensus = train_consensus(features, target, std_norm, pairs, model.n, consensus_args)
    edges = decode(consensus.score, row_k=int(config.get("decoder_row_k", 2)), symmetry="mutual", include_mst=True, xy=bundle.coordinates)
    consensus_seconds = time.perf_counter() - consensus_started
    np.save(output_dir / "omega.npy", consensus.score)
    np.save(output_dir / "probability_mean.npy", mean_p)
    np.save(output_dir / "probability_std.npy", std_p)
    (output_dir / "learned_edges.csv").write_text("source,target\n" + "".join(f"{u},{v}\n" for u, v in sorted(edges)), encoding="utf-8")
    metrics = {
        "seed": seed, "best_epoch": best_epoch, "validation_joint": best_value,
        "reference_scales": scales, "test_objective_components": test_losses,
        "forecast": forecast_metrics, "num_learned_edges": len(edges),
        "runtime": {
            "training_seconds_this_invocation": time.perf_counter() - started,
            "training_seconds_total": training_seconds_total,
            "prior_precompute_seconds": prior_precompute_seconds,
            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
            "inference_seconds_all_training_windows": inference_seconds,
            "consensus_decoding_seconds": consensus_seconds,
            "peak_gpu_memory_mb": max(peak_gpu_memory_mb, float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0),
        },
        "test_evaluated": True,
    }
    (output_dir / "metrics_gt_free.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return metrics


@torch.no_grad()
def collect_forecasts(model: PaperDGMForecaster, loader: DataLoader, coordinates: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, y, _ in loader:
        prediction, _, _ = model(x.to(device), coordinates, stochastic=False)
        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def collect_probabilities(model: PaperDGMForecaster, loader: DataLoader, coordinates: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    values = []
    started = time.perf_counter()
    for x, _, _ in loader:
        _, probabilities = model.probabilities(x.to(device), coordinates)
        values.append(probabilities.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(values), time.perf_counter() - started


def consensus_statistics(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (probabilities + probabilities.transpose(0, 2, 1))
    return symmetric.mean(axis=0), symmetric.std(axis=0)


def compute_forecast_metrics(prediction: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray, variable_names: list[str]) -> dict[str, Any]:
    error = prediction - target
    mse, mae = float(np.mean(error**2)), float(np.mean(np.abs(error)))
    per_variable = {}
    physical = {}
    for index, name in enumerate(variable_names):
        e = error[..., index]
        per_variable[name] = {"mse": float(np.mean(e**2)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2)))}
        ep = e * std[index]
        physical[name] = {"mse": float(np.mean(ep**2)), "mae": float(np.mean(np.abs(ep))), "rmse": float(np.sqrt(np.mean(ep**2)))}
    return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "normalized_per_variable": per_variable, "physical_units": physical}


def load_learned_edges(path: Path) -> set[tuple[int, int]]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    return {tuple(sorted(map(int, row.split(",")))) for row in rows if row.strip()}


def evaluate_reference(run_dir: Path, evaluation_dir: Path) -> dict[str, Any]:
    """Post-hoc only: this is the sole function that opens reference edges."""
    reference = np.load(evaluation_dir / "reference_topology.npz")
    expected_ids = json.loads((run_dir / "node_ids.json").read_text(encoding="utf-8"))
    if reference["node_ids"].tolist() != expected_ids:
        raise ValueError("Reference node IDs/order do not match the run observations")
    n = len(expected_ids)
    edges = reference["edge_index"]
    if edges.ndim != 2 or edges.shape[0] != 2 or (edges < 0).any() or (edges >= n).any():
        raise ValueError("Reference edge_index must have shape [2,E] with valid node indices")
    gt = {tuple(sorted((int(u), int(v)))) for u, v in reference["edge_index"].T if int(u) != int(v)}
    predicted = load_learned_edges(run_dir / "learned_edges.csv")
    if any(u < 0 or v >= n or u == v for u, v in predicted):
        raise ValueError("Learned edges must use valid distinct node indices")
    tp, fp, fn = len(gt & predicted), len(predicted - gt), len(gt - predicted)
    total = n * (n - 1) // 2
    tn = total - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    topology = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "accuracy": (tp + tn) / total,
        "recall": recall, "f1": f1, "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }
    gt_free = json.loads((run_dir / "metrics_gt_free.json").read_text(encoding="utf-8"))
    mse = float(gt_free["forecast"]["mse"])
    joint = {"S_task": 1.0 / (1.0 + mse), "S_topo": f1}
    joint["S_0.5"] = 0.5 * joint["S_task"] + 0.5 * joint["S_topo"]
    result = {"topology": topology, "joint": joint}
    (run_dir / "metrics_posthoc.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
