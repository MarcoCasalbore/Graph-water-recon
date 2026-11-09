"""Train DGM on the controlled synthetic water-like benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import Parameters
from src.train_dgm import _flatten_for_dgm, _reshape_ctown_epanet_batch
from src.utils.utils import get_datamodule, get_model, set_params_wrt_dataset
from src.utils.utils import train_and_test


def build_knn_candidate_mask(node_xy: np.ndarray, k: int) -> torch.Tensor:
    xy = torch.tensor(node_xy, dtype=torch.float32)
    n = xy.shape[0]
    dist = torch.cdist(xy, xy, p=2)
    dist = dist.masked_fill(torch.eye(n, dtype=torch.bool), float("inf"))
    k = min(int(k), n - 1)
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
    mask = torch.zeros((n, n), dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1).expand_as(nn_idx)
    mask[rows, nn_idx] = True
    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)


def _minmax_offdiag(score: np.ndarray) -> np.ndarray:
    out = score.astype(np.float64, copy=True)
    n = out.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = out[mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        out[:] = 0.0
        return out
    lo = vals[finite].min()
    hi = vals[finite].max()
    out = (out - lo) / (hi - lo + 1e-12)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _score_knn_mask(score: np.ndarray, k: int) -> np.ndarray:
    n = score.shape[0]
    work = score.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(max(1, int(k)), n - 1)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return mask


def _geometry_mst_mask(node_xy: np.ndarray) -> np.ndarray:
    """Minimum spanning tree support from coordinates only, no GT edges."""
    xy = np.asarray(node_xy, dtype=np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    n = xy.shape[0]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    mask = np.zeros((n, n), dtype=bool)

    for _ in range(n - 1):
        best = None
        best_dist = float("inf")
        src_nodes = np.where(selected)[0]
        dst_nodes = np.where(~selected)[0]
        for u in src_nodes:
            local_idx = int(np.argmin(dist[u, dst_nodes]))
            v = int(dst_nodes[local_idx])
            d = float(dist[u, v])
            if d < best_dist:
                best_dist = d
                best = (int(u), v)
        if best is None:
            break
        u, v = best
        mask[u, v] = True
        mask[v, u] = True
        selected[v] = True

    np.fill_diagonal(mask, False)
    return mask


def _partial_corr_score(x: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    z = _standardize(x.astype(np.float64))
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    precision = np.linalg.pinv(cov + ridge * np.eye(cov.shape[0]))
    d = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    pcorr = -precision / (d[:, None] * d[None, :] + 1e-12)
    pcorr = np.maximum(pcorr, 0.0)
    np.fill_diagonal(pcorr, 0.0)
    return _minmax_offdiag(pcorr)


def _positive_corr_score(x: np.ndarray) -> np.ndarray:
    z = _standardize(x.astype(np.float64))
    score = (z.T @ z) / max(1, z.shape[0] - 1)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _pressure_smooth_score(pressure: np.ndarray) -> np.ndarray:
    """Pressure smoothness score without allocating a full [T, N, N] tensor."""
    n = pressure.shape[1]
    chunk_size = 32
    diff = np.empty((n, n), dtype=np.float64)
    pressure64 = pressure.astype(np.float64, copy=False)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff[start:stop] = np.mean(
            np.abs(pressure64[:, start:stop, None] - pressure64[:, None, :]),
            axis=0,
        )
    scale = np.median(diff[diff > 0]) if np.any(diff > 0) else 1.0
    score = np.exp(-diff / (scale + 1e-12))
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _rank01(score: np.ndarray) -> np.ndarray:
    n = score.shape[0]
    pairs = [(float(score[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort()
    out = np.zeros((n, n), dtype=np.float64)
    denom = max(1, len(pairs) - 1)
    for rank, (_, i, j) in enumerate(pairs):
        value = rank / denom
        out[i, j] = out[j, i] = value
    return out


def _global_var_score(x: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    z = _standardize(x.astype(np.float64))
    source = z[:-1]
    target = z[1:]
    n = source.shape[1]
    coef = np.linalg.solve(source.T @ source + float(ridge) * np.eye(n), source.T @ target)
    score = np.maximum(np.abs(coef), np.abs(coef).T)
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _global_cross_var_score(source_x: np.ndarray, target_x: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    source = _standardize(source_x.astype(np.float64))
    target = _standardize(target_x.astype(np.float64))
    n = target.shape[1]
    design = np.concatenate([target[:-1], source[:-1]], axis=1)
    y = target[1:]
    coef = np.linalg.solve(design.T @ design + float(ridge) * np.eye(2 * n), design.T @ y)
    source_to_target = np.abs(coef[n:, :])
    score = np.maximum(source_to_target, source_to_target.T)
    np.fill_diagonal(score, 0.0)
    return _minmax_offdiag(score)


def _global_dynamic_score(data_npz: str, mode: str, partial_ridge: float, geo_tau: float) -> np.ndarray:
    data = np.load(data_npz)
    xy = data["node_xy"].astype(np.float32)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    pressure = data["pressure"].astype(np.float32)
    demand = data["demand"].astype(np.float32)

    if mode == "pressure_corr":
        return _positive_corr_score(pressure)

    smooth = _pressure_smooth_score(pressure)
    if mode == "pressure_smooth":
        return smooth
    if mode == "geo_pressure_smooth":
        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
        geo = np.exp(-dist / max(float(geo_tau), 1e-6))
        np.fill_diagonal(geo, 0.0)
        return _minmax_offdiag(geo * smooth)

    demand_global = _global_var_score(demand, ridge=partial_ridge)
    pressure_from_demand = _global_cross_var_score(demand, pressure, ridge=partial_ridge)
    dynamic = _minmax_offdiag(smooth * np.maximum(demand_global, pressure_from_demand))

    if mode == "global_dynamic_proxy":
        return dynamic

    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    geo = np.exp(-dist / max(float(geo_tau), 1e-6))
    np.fill_diagonal(geo, 0.0)
    if mode == "geo_global_dynamic_bridge_proxy":
        finite = dist[~np.eye(dist.shape[0], dtype=bool)]
        threshold = np.quantile(finite, 0.65) if finite.size else 0.0
        scale = np.std(finite) + 1e-12 if finite.size else 1.0
        long_gate = 1.0 / (1.0 + np.exp(-(dist - threshold) / (0.15 * scale)))
        evidence_gate = np.maximum(0.0, dynamic - 0.55) / max(1e-12, 1.0 - 0.55)
        bridge = _minmax_offdiag(dynamic * evidence_gate * long_gate)
        local = _minmax_offdiag(geo * dynamic)
        return _minmax_offdiag(0.85 * local + 0.15 * bridge)
    return _minmax_offdiag(geo * dynamic)


def build_q_dynamic_candidate_mask(
    data_npz: str,
    pseudo_score_mode: str,
    q_k: int,
    partial_ridge: float,
    geo_tau: float,
    include_mst: bool,
) -> torch.Tensor:
    """Candidate support from the same observable Q score used by the loss."""
    data = np.load(data_npz)
    score = _global_dynamic_score(
        data_npz=data_npz,
        mode=pseudo_score_mode,
        partial_ridge=partial_ridge,
        geo_tau=geo_tau,
    )
    mask = _score_knn_mask(score, q_k)
    if include_mst:
        mask = mask | _geometry_mst_mask(data["node_xy"].astype(np.float32))
    np.fill_diagonal(mask, False)
    return torch.tensor(mask, dtype=torch.bool)


def build_observable_candidate_mask(
    data_npz: str,
    geo_k: int,
    signal_k: int,
    partial_ridge: float,
) -> torch.Tensor:
    """Candidate support from coordinates plus pressure observables, no GT edges."""
    data = np.load(data_npz)
    xy = data["node_xy"].astype(np.float32)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    pressure = data["pressure"].astype(np.float32)

    geo_mask = build_knn_candidate_mask(xy, geo_k).numpy()
    partial = _partial_corr_score(pressure, ridge=partial_ridge)
    smooth = _pressure_smooth_score(pressure)
    partial_smooth = _minmax_offdiag(partial * smooth)

    signal_mask = (
        _score_knn_mask(partial, signal_k)
        | _score_knn_mask(smooth, signal_k)
        | _score_knn_mask(partial_smooth, signal_k)
    )
    mask = geo_mask | signal_mask
    np.fill_diagonal(mask, False)
    return torch.tensor(mask, dtype=torch.bool)


def build_local_bridge_candidate_mask(
    data_npz: str,
    geo_k: int,
    bridge_k: int,
    partial_ridge: float,
    bridge_quantile: float,
    bridge_min_temporal: float,
) -> torch.Tensor:
    """Mostly geometric support plus a small pressure-driven long-edge budget."""
    data = np.load(data_npz)
    xy = data["node_xy"].astype(np.float32)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    pressure = data["pressure"].astype(np.float32)

    geo_mask = build_knn_candidate_mask(xy, geo_k).numpy()
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    partial = _partial_corr_score(pressure, ridge=partial_ridge)
    smooth = _pressure_smooth_score(pressure)
    temporal = _minmax_offdiag(partial * smooth)

    finite = dist[~np.eye(dist.shape[0], dtype=bool)]
    threshold = np.quantile(finite, max(0.0, min(1.0, bridge_quantile)))
    scale = np.std(finite) + 1e-12
    long_gate = 1.0 / (1.0 + np.exp(-(dist - threshold) / (0.15 * scale)))
    evidence_gate = np.maximum(0.0, temporal - bridge_min_temporal) / max(1e-12, 1.0 - bridge_min_temporal)
    bridge_score = temporal * evidence_gate * long_gate
    np.fill_diagonal(bridge_score, 0.0)

    bridge_mask = _score_knn_mask(bridge_score, bridge_k)
    mask = geo_mask | bridge_mask
    np.fill_diagonal(mask, False)
    return torch.tensor(mask, dtype=torch.bool)


def ensure_dataset(data_dir: Path, regenerate: bool) -> None:
    data_npz = data_dir / "data.npz"
    meta_json = data_dir / "meta.json"
    if data_npz.exists() and meta_json.exists() and not regenerate:
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "synthetic" / "generate_synthetic_water.py"),
        "--out_dir",
        str(data_dir),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def configure_run(args: argparse.Namespace) -> Parameters:
    params = Parameters(args)
    params.dataset_name = "synthetic_water"
    params.dataset_path = ""
    params.data_npz = str(args.data_dir / "data.npz")
    params.meta_json = str(args.data_dir / "meta.json")

    params.run_name = args.run_name
    params.export_dir = str(args.out_dir)
    params.loss_plot_path = str(args.out_dir / "loss_curve.png")
    params.diagnostic_graph_metrics = bool(args.diagnostic_graph_metrics)

    params.graph_detach_input = False
    params.model = f"dgm_{args.ffun}_{args.gfun}"
    params.ffun = args.ffun
    params.gfun = args.gfun
    params.distance = "euclidean"

    params.num_nodes = args.num_nodes
    params.lags = args.lags
    params.stride = args.stride
    params.prediction_window = args.prediction_window
    params.batch_size = args.batch_size
    params.max_epochs = args.max_epochs
    params.check_val_every_n_epoch = args.check_val_every_n_epoch
    params.lr = args.lr

    params.input_features = 2
    params.output_features = 2
    params.k = args.k

    params.emb_dim = 32
    params.pre_fc = [-1, 32]
    params.conv_layers = [[32, 32], [32, 16], [16, 8]]
    params.dgm_layers = [[32, 16, 4], [36, 16, 4], []]
    params.fc_layers = [8, 8, 2]

    # Baseline: stay close to current DGM training, no strong geometric/degree priors.
    params.lambda_graph = args.lambda_graph
    params.lambda_geo = 0.0
    params.lambda_smooth = 0.0
    params.lambda_knn = 0.0
    params.lambda_degree = 0.0
    params.lambda_stability = args.lambda_stability
    params.use_complementary_loss = args.use_complementary_loss
    params.alpha_forecast = args.alpha_forecast
    params.alpha_forecast_start = (
        args.alpha_forecast if args.alpha_forecast_start is None else args.alpha_forecast_start
    )
    params.alpha_forecast_end = (
        args.alpha_forecast if args.alpha_forecast_end is None else args.alpha_forecast_end
    )
    params.alpha_schedule_epochs = args.alpha_schedule_epochs
    params.alpha_hold_epochs = args.alpha_hold_epochs
    params.coordinate_embedding_dim = args.coordinate_embedding_dim
    params.coordinate_embedding_trainable = args.coordinate_embedding_trainable
    node_xy = np.load(params.data_npz)["node_xy"].astype(np.float32)
    params.node_xy = node_xy[0] if node_xy.ndim == 3 else node_xy
    params.lambda_soft_pseudo_edge = args.lambda_soft_pseudo_edge
    params.topology_loss_normalization = args.topology_loss_normalization
    params.pseudo_geo_tau = args.pseudo_geo_tau
    params.pseudo_temporal_weight = args.pseudo_temporal_weight
    params.pseudo_score_mode = args.pseudo_score_mode
    params.pseudo_use_static_target = args.pseudo_use_static_target
    params.pseudo_max_lag = args.pseudo_max_lag
    params.pseudo_partial_ridge = args.pseudo_partial_ridge
    params.pseudo_bridge_topk = args.pseudo_bridge_topk
    params.pseudo_bridge_mass = args.pseudo_bridge_mass
    params.pseudo_bridge_quantile = args.pseudo_bridge_quantile
    params.pseudo_bridge_temperature = args.pseudo_bridge_temperature
    params.pseudo_bridge_min_temporal = args.pseudo_bridge_min_temporal
    params.pseudo_bridge_candidate_k = args.pseudo_bridge_candidate_k
    params.pseudo_candidate_k = args.pseudo_candidate_k
    params.pseudo_topk = args.pseudo_topk
    params.pseudo_include_mst = args.pseudo_include_mst
    params.pseudo_mst_bonus = args.pseudo_mst_bonus
    params.select_best_by = args.select_best_by
    params.static_top_edges = args.static_top_edges
    params.proxy_length_candidate_k = args.proxy_length_candidate_k
    params.proxy_length_quantile = args.proxy_length_quantile
    params.proxy_length_weight = args.proxy_length_weight
    params.proxy_connectivity_weight = args.proxy_connectivity_weight
    params.proxy_frequency_weight = args.proxy_frequency_weight
    params.soft_degree_target = args.soft_degree_target
    params.max_degree_target = 4

    params.task_loss_type = args.task_loss_type
    params.task_huber_delta = args.task_huber_delta
    params.task_target = args.task_target
    params.task_feature_combine = args.task_feature_combine
    params.task_feature_normalization = args.task_feature_normalization
    params.pressure_weight = args.pressure_weight
    params.demand_weight = args.demand_weight
    return params


def _canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _local_undirected_graphs(edges: torch.Tensor, num_nodes: int) -> list[set[tuple[int, int]]]:
    edges_np = edges.detach().cpu().numpy().astype(np.int64)
    src = edges_np[0]
    dst = edges_np[1]
    if src.size == 0:
        return []

    batch_ids = src // num_nodes
    n_graphs = int(batch_ids.max()) + 1
    out: list[set[tuple[int, int]]] = []

    for batch_id in range(n_graphs):
        mask = batch_ids == batch_id
        src_b = src[mask] - batch_id * num_nodes
        dst_b = dst[mask] - batch_id * num_nodes
        edge_set = set()
        for u, v in zip(src_b.tolist(), dst_b.tolist()):
            if u == v:
                continue
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                edge_set.add(_canonical(int(u), int(v)))
        out.append(edge_set)
    return out


def export_consensus_graph(
    model,
    loader,
    params,
    out_path: Path,
    threshold: float,
    top_edges: int,
    ensure_connected: bool,
) -> Path:
    """Aggregate dynamic DGM edges into one static undirected graph."""
    device = next(model.parameters()).device
    model.eval()

    edge_counts: dict[tuple[int, int], int] = {}
    n_graphs = 0

    with torch.no_grad():
        for batch in loader:
            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None
            x, y, batch = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            _ = model(x, edge_index, edge_weight)

            for edge_set in _local_undirected_graphs(model.edges, params.num_nodes):
                n_graphs += 1
                for edge in edge_set:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

    if n_graphs == 0:
        raise RuntimeError("No local graphs were produced while exporting consensus graph.")

    min_count = max(1, int(np.ceil(threshold * n_graphs)))
    if top_edges > 0:
        consensus_edges = [
            edge
            for edge, _ in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)[:top_edges]
        ]
    else:
        consensus_edges = [edge for edge, count in edge_counts.items() if count >= min_count]

    if ensure_connected:
        consensus_edges = connect_components_with_geometry(
            consensus_edges=consensus_edges,
            edge_counts=edge_counts,
            params=params,
        )
        if top_edges > 0 and len(consensus_edges) > top_edges:
            consensus_edges = prune_to_edge_budget_connected(
                consensus_edges=consensus_edges,
                edge_counts=edge_counts,
                top_edges=top_edges,
                num_nodes=int(params.num_nodes),
            )

    rows = []
    for u, v in sorted(consensus_edges):
        rows.append((u, v))
        rows.append((v, u))

    edge_index = torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)
    torch.save(edge_index, out_path)

    print("[INFO] Consensus graph export")
    print(f"       logical graphs inspected: {n_graphs}")
    if top_edges > 0:
        print(f"       selection: top {top_edges} edges by validation frequency")
    else:
        print(f"       threshold: {threshold:.2f} -> min_count={min_count}")
    print(f"       undirected static edges: {len(consensus_edges)}")
    if ensure_connected:
        print("       postprocess: connected components bridged with geometry/frequency proxy")
    print(f"       saved to: {out_path}")
    return out_path


def _local_probability_blocks(logits: torch.Tensor, num_nodes: int) -> list[torch.Tensor]:
    """Convert DGM logits into per-graph row probabilities."""
    if logits is None:
        return []
    logits = logits.detach()
    if logits.dim() != 2:
        return []
    n_total = logits.shape[0]
    if n_total % num_nodes != 0:
        return []

    blocks = []
    n_graphs = n_total // num_nodes
    for batch_id in range(n_graphs):
        start = batch_id * num_nodes
        stop = start + num_nodes
        block_logits = logits[start:stop, start:stop]
        finite = torch.isfinite(block_logits)
        if not finite.any():
            continue
        safe_logits = block_logits.masked_fill(~finite, -1e9)
        prob = torch.softmax(safe_logits, dim=1)
        prob = prob.masked_fill(~finite, 0.0)
        prob.fill_diagonal_(0.0)
        blocks.append(prob)
    return blocks


def decode_observable_score(
    score: torch.Tensor,
    row_k: int,
    symmetry: str,
    include_mst: bool,
    data_npz: str,
) -> list[tuple[int, int]]:
    """Decode an edge score without a GT-derived global edge budget."""
    values = score.detach().cpu().numpy().astype(np.float64)
    n = values.shape[0]
    np.fill_diagonal(values, -np.inf)
    k = min(max(1, int(row_k)), n - 1)
    neighbors = np.argpartition(-values, kth=k - 1, axis=1)[:, :k]
    directed = {(u, int(v)) for u in range(n) for v in neighbors[u]}
    edges = {
        _canonical(u, v)
        for u, v in directed
        if symmetry == "union" or (v, u) in directed
    }
    if include_mst:
        npz = np.load(data_npz)
        mst_mask = _geometry_mst_mask(npz["node_xy"])
        mst_u, mst_v = np.where(np.triu(mst_mask, k=1))
        edges.update(_canonical(int(u), int(v)) for u, v in zip(mst_u, mst_v))
    return sorted(edges)


def export_soft_probability_graph(
    model,
    loader,
    params,
    out_path: Path,
    top_edges: int,
    ensure_connected: bool,
    score_out_path: Path | None = None,
    row_k: int = 3,
    symmetry: str = "mutual",
    include_mst: bool = True,
) -> Path:
    """Export a static graph from mean DGM soft probabilities P_ij."""
    device = next(model.parameters()).device
    model.eval()

    prob_sum = torch.zeros((params.num_nodes, params.num_nodes), dtype=torch.float64)
    prob_sq_sum = torch.zeros((params.num_nodes, params.num_nodes), dtype=torch.float64)
    n_blocks = 0

    with torch.no_grad():
        for batch in loader:
            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None
            x, y, batch = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            _ = model(x, edge_index, edge_weight)

            for gf in getattr(model, "graph_f", []):
                logits = getattr(gf, "last_logits", None)
                for prob in _local_probability_blocks(logits, int(params.num_nodes)):
                    prob_cpu = prob.detach().cpu().double()
                    prob_sum += prob_cpu
                    prob_sq_sum += prob_cpu * prob_cpu
                    n_blocks += 1

    if n_blocks == 0:
        raise RuntimeError("No DGM probability blocks were available for soft graph export.")

    prob = prob_sum / float(n_blocks)
    var = torch.clamp(prob_sq_sum / float(n_blocks) - prob * prob, min=0.0)
    prob_std = torch.sqrt(var)
    score = 0.5 * (prob + prob.T)
    score_std = 0.5 * (prob_std + prob_std.T)
    score.fill_diagonal_(0.0)
    score_std.fill_diagonal_(0.0)
    if score_out_path is not None:
        np.save(score_out_path, score.numpy())
        np.save(score_out_path.with_name("dgm_prob_mean.npy"), score.numpy())
        np.save(score_out_path.with_name("dgm_prob_std.npy"), score_std.numpy())

    edges: list[tuple[int, int]] = []
    score_dict: dict[tuple[int, int], float] = {}
    n = int(params.num_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            score_dict[(i, j)] = float(score[i, j].item())

    if top_edges > 0:
        edges = [
            edge for edge, _ in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:top_edges]
        ]
        selection_label = f"top {top_edges} edges by mean symmetrized P_ij"
    else:
        edges = decode_observable_score(
            score=score,
            row_k=row_k,
            symmetry=symmetry,
            include_mst=include_mst,
            data_npz=params.data_npz,
        )
        selection_label = (
            f"observable row-top-{row_k}, symmetry={symmetry}, geometric_mst={include_mst}"
        )

    if ensure_connected:
        edges = connect_components_with_geometry(
            consensus_edges=edges,
            edge_counts=score_dict,
            params=params,
        )
        if top_edges > 0 and len(edges) > top_edges:
            edges = prune_to_edge_budget_connected(
                consensus_edges=edges,
                edge_counts=score_dict,
                top_edges=top_edges,
                num_nodes=n,
            )

    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        rows.append((v, u))
    edge_index = torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)
    torch.save(edge_index, out_path)

    print("[INFO] Soft probability graph export")
    print(f"       probability blocks inspected: {n_blocks}")
    print(f"       selection: {selection_label}")
    print(f"       undirected static edges: {len(edges)}")
    if score_out_path is not None:
        print(f"       score matrix saved to: {score_out_path}")
    print(f"       saved to: {out_path}")
    return out_path


def connect_components_with_geometry(
    consensus_edges: list[tuple[int, int]],
    edge_counts: dict[tuple[int, int], int],
    params,
) -> list[tuple[int, int]]:
    """Force one connected static graph using only coordinates and DGM frequencies."""
    try:
        import networkx as nx
    except Exception:
        return consensus_edges

    num_nodes = int(params.num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(consensus_edges)

    npz = np.load(params.data_npz)
    xy = npz["node_xy"].astype(np.float32)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:num_nodes, :2]

    selected = set(_canonical(u, v) for u, v in consensus_edges)
    max_count = max(edge_counts.values()) if edge_counts else 1

    while nx.number_connected_components(graph) > 1:
        components = [set(c) for c in nx.connected_components(graph)]
        best_edge = None
        best_score = None

        for c_idx, comp_a in enumerate(components):
            for comp_b in components[c_idx + 1:]:
                for u in comp_a:
                    delta = xy[list(comp_b)] - xy[u]
                    dist = np.sqrt((delta * delta).sum(axis=1))
                    order = np.argsort(dist)[: min(3, len(dist))]
                    comp_b_list = list(comp_b)
                    for idx in order:
                        v = comp_b_list[int(idx)]
                        edge = _canonical(int(u), int(v))
                        if edge in selected:
                            continue
                        freq = edge_counts.get(edge, 0) / max(1, max_count)
                        score = float(dist[int(idx)]) - 0.25 * float(freq)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_edge = edge

        if best_edge is None:
            break
        selected.add(best_edge)
        graph.add_edge(*best_edge)

    return sorted(selected)


def prune_to_edge_budget_connected(
    consensus_edges: list[tuple[int, int]],
    edge_counts: dict[tuple[int, int], int],
    top_edges: int,
    num_nodes: int,
) -> list[tuple[int, int]]:
    try:
        import networkx as nx
    except Exception:
        return sorted(consensus_edges)[:top_edges]

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(consensus_edges)

    candidates = sorted(
        consensus_edges,
        key=lambda edge: (edge_counts.get(_canonical(*edge), 0), edge[0], edge[1]),
    )
    for edge in candidates:
        if graph.number_of_edges() <= top_edges:
            break
        graph.remove_edge(*edge)
        if not nx.is_connected(graph):
            graph.add_edge(*edge)

    return sorted(_canonical(int(u), int(v)) for u, v in graph.edges())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DGM on synthetic water-like data.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic" / "simple_water_n50")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs" / "synthetic_water_base")
    parser.add_argument("--run_name", default="base")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--prediction_window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--ffun",
        choices=["gcn", "gat", "mlp", "knn", "gcn1d", "gcn1d-big", "gConvLSTM", "gConvGRU"],
        default="gcn",
        help="Network used inside the DGM edge-probability module.",
    )
    parser.add_argument(
        "--gfun",
        choices=["gcn", "gat", "edgeconv", "gcn1d", "gcn1d-big", "gConvLSTM", "gConvGRU"],
        default="gcn",
        help="Message-passing module used after DGM edge sampling.",
    )
    parser.add_argument("--lambda_graph", type=float, default=1e-2)
    parser.add_argument("--lambda_stability", type=float, default=0.0)
    parser.add_argument("--use_complementary_loss", action="store_true")
    parser.add_argument("--alpha_forecast", type=float, default=1.0)
    parser.add_argument("--alpha_forecast_start", type=float, default=None)
    parser.add_argument("--alpha_forecast_end", type=float, default=None)
    parser.add_argument(
        "--alpha_schedule_epochs", type=int, default=0,
        help="Number of ramp epochs from alpha start to alpha end.",
    )
    parser.add_argument("--alpha_hold_epochs", type=int, default=0)
    parser.add_argument("--task_loss_type", choices=["mse", "mae", "huber"], default="mse")
    parser.add_argument("--task_huber_delta", type=float, default=1.0)
    parser.add_argument("--task_target", choices=["all", "pressure", "demand"], default="all")
    parser.add_argument("--task_feature_combine", choices=["weighted_sum", "weighted_mean"], default="weighted_sum")
    parser.add_argument("--task_feature_normalization", choices=["none", "target_std", "target_mad"], default="none")
    parser.add_argument("--pressure_weight", type=float, default=1.0)
    parser.add_argument("--demand_weight", type=float, default=1.0)
    parser.add_argument("--coordinate_embedding_dim", type=int, default=0)
    parser.add_argument(
        "--coordinate_embedding_trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lambda_soft_pseudo_edge", type=float, default=0.0)
    parser.add_argument(
        "--topology_loss_normalization",
        choices=["none", "uniform_baseline"],
        default="none",
    )
    parser.add_argument("--pseudo_geo_tau", type=float, default=1.0)
    parser.add_argument("--pseudo_temporal_weight", type=float, default=0.5)
    parser.add_argument(
        "--pseudo_score_mode",
        choices=[
            "geo_temporal",
            "geo_only",
            "temporal_only",
            "demand_corr",
            "pressure_corr",
            "pressure_smooth",
            "pressure_lag",
            "pressure_partial",
            "geo_pressure_smooth",
            "geo_pressure_evidence",
            "geo_partial_smooth",
            "geo_partial_bridge",
            "global_dynamic_proxy",
            "geo_global_dynamic_proxy",
            "geo_global_dynamic_bridge_proxy",
            "observable_geometry",
            "observable_geometry_plus",
            "observable_expert_pressure",
            "observable_expert_dynamic",
            "observable_expert_full",
            "observable_electric_voltage",
            "observable_electric_dynamic",
            "observable_electric_grid",
            "observable_electric_inverse",
            "observable_electric_full",
        ],
        default="geo_temporal",
    )
    parser.add_argument("--pseudo_max_lag", type=int, default=6)
    parser.add_argument("--pseudo_partial_ridge", type=float, default=0.05)
    parser.add_argument("--pseudo_use_static_target", action="store_true")
    parser.add_argument("--pseudo_bridge_topk", type=int, default=1)
    parser.add_argument("--pseudo_bridge_mass", type=float, default=0.15)
    parser.add_argument("--pseudo_bridge_quantile", type=float, default=0.65)
    parser.add_argument("--pseudo_bridge_temperature", type=float, default=0.08)
    parser.add_argument("--pseudo_bridge_min_temporal", type=float, default=0.55)
    parser.add_argument("--pseudo_bridge_candidate_k", type=int, default=4)
    parser.add_argument("--pseudo_candidate_k", type=int, default=6)
    parser.add_argument("--pseudo_topk", type=int, default=2)
    parser.add_argument("--pseudo_include_mst", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo_mst_bonus", type=float, default=1.0)
    parser.add_argument("--select_best_by", choices=["val_mse", "observable_val", "topology_proxy"], default="val_mse")
    parser.add_argument("--proxy_length_candidate_k", type=int, default=12)
    parser.add_argument("--proxy_length_quantile", type=float, default=0.65)
    parser.add_argument("--proxy_length_weight", type=float, default=1.0)
    parser.add_argument("--proxy_connectivity_weight", type=float, default=1.0)
    parser.add_argument("--proxy_frequency_weight", type=float, default=0.05)
    parser.add_argument("--soft_degree_target", type=float, default=1.0)
    parser.add_argument("--consensus_threshold", type=float, default=0.5)
    parser.add_argument(
        "--static_top_edges", type=int, default=0,
        help="Diagnostic fixed edge budget. Keep at 0 for the GT-free observable decoder.",
    )
    parser.add_argument("--static_row_k", type=int, default=3)
    parser.add_argument("--static_symmetry", choices=["union", "mutual"], default="mutual")
    parser.add_argument("--static_include_mst", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ensure_connected_consensus", action="store_true")
    parser.add_argument("--export_soft_graph", action="store_true")
    parser.add_argument("--save_soft_score", action="store_true")
    parser.add_argument("--assign_candidate_mask", action="store_true")
    parser.add_argument("--candidate_mode", choices=["geo_knn", "observable_union", "local_bridge", "q_dynamic"], default="geo_knn")
    parser.add_argument("--candidate_k", type=int, default=6)
    parser.add_argument("--candidate_signal_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic_graph_metrics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(args.data_dir, regenerate=args.regenerate)

    params = configure_run(args)
    train_l, val_l, test_l, params = get_datamodule(params)
    params = set_params_wrt_dataset(params, train_l)

    print("========== SYNTHETIC RUN CONFIG ==========")
    print(f"data_npz          : {params.data_npz}")
    print(f"meta_json         : {params.meta_json}")
    print(f"export_dir        : {params.export_dir}")
    print(f"num_nodes         : {params.num_nodes}")
    print(f"lags              : {params.lags}")
    print(f"stride            : {params.stride}")
    print(f"k                 : {params.k}")
    print(f"dgm_modules       : ffun={params.ffun}, gfun={params.gfun}")
    print(f"lambda_graph      : {params.lambda_graph}")
    print(f"lambda_stability  : {params.lambda_stability}")
    print(f"complementary_loss: {params.use_complementary_loss}")
    print(f"alpha_forecast    : {params.alpha_forecast}")
    print(
        "alpha_schedule    : "
        f"{params.alpha_forecast_start} -> {params.alpha_forecast_end} "
        f"after hold={params.alpha_hold_epochs}, ramp={params.alpha_schedule_epochs} epochs"
    )
    print(
        "coordinate_embed : "
        f"dim={params.coordinate_embedding_dim}, "
        f"trainable={params.coordinate_embedding_trainable}"
    )
    print(f"lambda_soft_pseudo_edge: {params.lambda_soft_pseudo_edge}")
    print(f"topology_loss_normalization: {params.topology_loss_normalization}")
    print(f"pseudo_edge_target: mode={params.pseudo_score_mode}, static_target={params.pseudo_use_static_target}, geo_tau={params.pseudo_geo_tau}, temporal_weight={params.pseudo_temporal_weight}, candidate_k={params.pseudo_candidate_k}, topk={params.pseudo_topk}, include_mst={params.pseudo_include_mst}, max_lag={params.pseudo_max_lag}, partial_ridge={params.pseudo_partial_ridge}")
    print(f"select_best_by   : {params.select_best_by}")
    print(f"initial edge_index: geometry proxy from initial_edge_index")
    print("==========================================")
    print("LOSS FORMULA")
    if params.use_complementary_loss:
        print(
            "  L = alpha * L_forecast + (1-alpha) * L_Q + lambda_stability * L_stability"
        )
    else:
        print(
            "  L = L_forecast + lambda_Q * L_Q + lambda_stability * L_stability"
        )
    print(
        "  L_Q = mean_i -sum_j Q_ij log(P_ij + eps), "
        "Q_ij built from coordinates + time series only, no GT edges"
    )
    if params.pseudo_score_mode == "geo_partial_bridge":
        print(
            "  Q_bridge strategy: Q = (1-r_bridge) Q_local + r_bridge Q_bridge, "
            "where Q_bridge uses pressure partial*smooth evidence gated by long relative distance."
        )
    if params.pseudo_score_mode in {"global_dynamic_proxy", "geo_global_dynamic_proxy"}:
        print(
            "  Q_global_dynamic strategy: fit observable VAR scores from full node series, no GT edges."
        )
        print(
            "    B_demand from D(t+1)=D(t)B; B_cross from P(t+1)=[P(t),D(t)]B."
        )
        print(
            "    S_dynamic = minmax(pressure_smooth * max(|B_demand|, |B_cross_demand_to_pressure|))."
        )
        if params.pseudo_score_mode == "geo_global_dynamic_proxy":
            print("    Q score = geo_affinity * S_dynamic, then row top-k normalization.")
        else:
            print("    Q score = S_dynamic, then row top-k normalization.")
    if params.pseudo_score_mode == "geo_global_dynamic_bridge_proxy":
        print(
            "  Q_global_dynamic_bridge strategy: same observable VAR score as geo_global_dynamic_proxy,"
        )
        print(
            "    but Q row mass is split into local geo-dynamic support plus a small long-edge bridge channel."
        )
        print(
            "    Bridge candidates use only S_dynamic evidence gated by relative geometric length; no GT edges."
        )
    print(
        "  Checkpoint selection uses the final alpha: "
        "J_obs = alpha_end*MSE_val + (1-alpha_end)*L_Q_val + lambda_stability*L_stability_val; no GT edges."
    )
    print("  Change in this strategy: minimal direct weak edge classification on observable pseudo-target Q.")
    print("==========================================")

    first_batch = next(iter(train_l))
    print("First batch x shape         :", first_batch.x.shape)
    print("First batch y shape         :", first_batch.y.shape)
    print("First batch edge_index shape:", first_batch.edge_index.shape)
    print("First batch node_xy shape   :", first_batch.node_xy.shape)

    model = get_model(params)

    if args.assign_candidate_mask:
        npz = np.load(params.data_npz)
        if args.candidate_mode == "observable_union":
            candidate_mask = build_observable_candidate_mask(
                data_npz=params.data_npz,
                geo_k=args.candidate_k,
                signal_k=args.candidate_signal_k,
                partial_ridge=args.pseudo_partial_ridge,
            )
        elif args.candidate_mode == "local_bridge":
            candidate_mask = build_local_bridge_candidate_mask(
                data_npz=params.data_npz,
                geo_k=args.candidate_k,
                bridge_k=args.pseudo_bridge_candidate_k,
                partial_ridge=args.pseudo_partial_ridge,
                bridge_quantile=args.pseudo_bridge_quantile,
                bridge_min_temporal=args.pseudo_bridge_min_temporal,
            )
        elif args.candidate_mode == "q_dynamic":
            candidate_mask = build_q_dynamic_candidate_mask(
                data_npz=params.data_npz,
                pseudo_score_mode=args.pseudo_score_mode,
                q_k=args.candidate_k,
                partial_ridge=args.pseudo_partial_ridge,
                geo_tau=args.pseudo_geo_tau,
                include_mst=args.pseudo_include_mst,
            )

        else:
            candidate_mask = build_knn_candidate_mask(npz["node_xy"].astype(np.float32), args.candidate_k)
        assigned = 0
        for gf in model.graph_f:
            if hasattr(gf, "candidate_mask"):
                gf.candidate_mask = candidate_mask
                assigned += 1
        print(f"[INFO] Assigned {args.candidate_mode} candidate mask to {assigned} DGM layer(s), geo_k={args.candidate_k}, signal_k={args.candidate_signal_k}.")

    train_and_test(model, train_l, val_l, test_l, params)

    # Post-hoc export only: never used for training or model selection.
    torch.save(model.state_dict(), args.out_dir / "model_state_selected.pt")
    model.eval()
    device = next(model.parameters()).device
    y_true_parts, y_pred_parts = [], []
    with torch.no_grad():
        for prediction_batch in test_l:
            edge_index = prediction_batch.edge_index.to(device)
            edge_weight = prediction_batch.edge_attr.to(device) if prediction_batch.edge_attr is not None else None
            x_pred, y_true, _ = _reshape_ctown_epanet_batch(prediction_batch, params, device)
            x_pred, y_true = _flatten_for_dgm(x_pred, y_true)
            y_pred, _ = model(x_pred, edge_index, edge_weight)
            batch_samples = y_true.shape[0] // int(params.num_nodes)
            prediction_shape = (
                batch_samples, int(params.num_nodes),
                int(params.prediction_window), int(params.output_features),
            )
            y_true_parts.append(y_true.reshape(prediction_shape).detach().cpu().numpy())
            y_pred_parts.append(y_pred.reshape(prediction_shape).detach().cpu().numpy())
    np.savez_compressed(
        args.out_dir / "test_predictions.npz",
        y_true=np.concatenate(y_true_parts, axis=0),
        y_pred=np.concatenate(y_pred_parts, axis=0),
        domain=np.asarray("train_standardized"),
    )

    consensus_edges = export_consensus_graph(
        model=model,
        loader=val_l,
        params=params,
        out_path=args.out_dir / "dgm_learned_edges_consensus.pt",
        threshold=args.consensus_threshold,
        top_edges=args.static_top_edges,
        ensure_connected=args.ensure_connected_consensus,
    )

    compare_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "water" / "compare_graphs_epanet.py"),
        "--learned",
        str(consensus_edges),
        "--npz",
        params.data_npz,
        "--meta",
        params.meta_json,
        "--outdir",
        str(args.out_dir / "graph_comparison"),
    ]
    subprocess.run(compare_cmd, check=True, cwd=PROJECT_ROOT)
    print(f"[OK] Synthetic comparison written to: {args.out_dir / 'graph_comparison'}")

    if args.export_soft_graph:
        soft_edges = export_soft_probability_graph(
            model=model,
            loader=val_l,
            params=params,
            out_path=args.out_dir / "dgm_learned_edges_softprob.pt",
            top_edges=args.static_top_edges,
            ensure_connected=args.ensure_connected_consensus,
            score_out_path=(args.out_dir / "dgm_soft_score.npy") if args.save_soft_score else None,
            row_k=args.static_row_k,
            symmetry=args.static_symmetry,
            include_mst=args.static_include_mst,
        )
        soft_compare_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "water" / "compare_graphs_epanet.py"),
            "--learned",
            str(soft_edges),
            "--npz",
            params.data_npz,
            "--meta",
            params.meta_json,
            "--outdir",
            str(args.out_dir / "graph_comparison_softprob"),
        ]
        subprocess.run(soft_compare_cmd, check=True, cwd=PROJECT_ROOT)
        print(f"[OK] Soft-probability comparison written to: {args.out_dir / 'graph_comparison_softprob'}")


if __name__ == "__main__":
    main()
