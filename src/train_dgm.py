import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import root_mean_squared_error

try:
    from einops import rearrange
except Exception:
    rearrange = None
import matplotlib.pyplot as plt

WATER_LIKE_DATASETS = {"ctown_epanet", "synthetic_water"}


def _is_water_like_dataset(hparams) -> bool:
    return getattr(hparams, "dataset_name", "") in WATER_LIKE_DATASETS


def _get_float_attr(hparams, name, default=0.0) -> float:
    if hasattr(hparams, name):
        v = getattr(hparams, name)
        try:
            return float(v)
        except Exception:
            return float(default)
    return float(default)


def _save_loss_curve(history, out_path):
    if history is None or len(history.get("epoch", [])) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_mse"], label="train_mse")

    if "val_mse" in history and len(history["val_mse"]) == len(history["epoch"]):
        plt.plot(history["epoch"], history["val_mse"], label="val_mse")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training / Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _compute_geo_and_smooth_losses(model, batch, y_predicted, device, hparams):
    """
    Optional losses:
      - L_geo: penalize long edges using node_xy
      - L_smooth: smoothness of predictions along learned edges

    Requirements:
      - model.edges exists and is [2, B*N*K] (batch-aware global indexing)
      - batch.node_xy exists as [B, N, 2] or [N, 2]

    If unavailable, returns zeros.
    """
    loss_geo = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)

    if not hasattr(model, "edges") or model.edges is None:
        return loss_geo, loss_smooth

    if not hasattr(batch, "node_xy") or batch.node_xy is None:
        return loss_geo, loss_smooth

    edges = model.edges.to(device)  # [2, E]
    src = edges[0]
    dst = edges[1]

    N = getattr(model, "num_nodes", None)
    if N is None:
        return loss_geo, loss_smooth

    node_xy = batch.node_xy.to(device)
    if node_xy.dim() == 2:
        node_xy = node_xy.unsqueeze(0)  # [1, N, 2]
    B = node_xy.shape[0]

    # global index -> batch id, node id
    b_id = torch.div(src, N, rounding_mode="floor")
    u = src % N
    v = dst % N

    valid = (b_id >= 0) & (b_id < B) & (u >= 0) & (u < N) & (v >= 0) & (v < N)
    b_id = b_id[valid]
    u = u[valid]
    v = v[valid]

    if b_id.numel() == 0:
        return loss_geo, loss_smooth

    duv = node_xy[b_id, u, :] - node_xy[b_id, v, :]
    dist_uv = torch.sqrt((duv * duv).sum(-1) + 1e-12)
    loss_geo = dist_uv.mean()

    # y_predicted comes as [(B*N), Dout] in your pipeline
    Dout = y_predicted.shape[-1]
    y_hat = y_predicted.reshape(-1, N, Dout)  # [B, N, Dout] if consistent
    if y_hat.shape[0] != B:
        # fallback: cannot align batches
        return loss_geo, loss_smooth

    diff_y = y_hat[b_id, u, :] - y_hat[b_id, v, :]
    loss_smooth = (diff_y * diff_y).mean()

    return loss_geo, loss_smooth


def _compute_dgm_stability_loss(model, device):
    losses = []
    for graph_layer in getattr(model, "graph_f", []):
        loss = getattr(graph_layer, "stability_loss", None)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack([loss.to(device) for loss in losses]).mean()


def _compute_soft_topology_loss(model, batch, device, hparams, x_flat=None):
    """
    Minimal differentiable topology prior over the DGM neighbor distribution.

    Main research loss:
        L_Q = - mean_i sum_j Q_ij log(P_ij + eps)

    Q is built only from observed coordinates/time series. Ground-truth edges
    are never used here.
    """
    if not hasattr(batch, "node_xy") or batch.node_xy is None:
        zero = torch.tensor(0.0, device=device)
        return zero, {"soft_pseudo_edge": 0.0}

    node_xy = batch.node_xy.to(device)
    if node_xy.dim() == 3:
        node_xy = node_xy[0]
    elif node_xy.dim() == 2:
        node_xy = node_xy[: int(hparams.num_nodes)]
    else:
        zero = torch.tensor(0.0, device=device)
        return zero, {"soft_pseudo_edge": 0.0}

    dist = torch.cdist(node_xy, node_xy, p=2)
    dist = dist / (dist[torch.isfinite(dist)].mean().detach() + 1e-12)
    eye = torch.eye(dist.shape[0], device=device, dtype=torch.bool)
    dist = dist.masked_fill(eye, 0.0)

    pseudo_edge_weight = _get_float_attr(hparams, "lambda_soft_pseudo_edge", 1.0)
    pseudo_edge_terms = []

    pseudo_target = None
    if pseudo_edge_weight > 0.0:
        pseudo_target = _build_pseudo_edge_target(
            dist=dist,
            x_flat=x_flat,
            hparams=hparams,
            device=device,
        )

    for graph_layer in getattr(model, "graph_f", []):
        logits = getattr(graph_layer, "last_logits", None)
        if logits is None:
            continue

        logits = logits.to(device)
        if logits.shape != dist.shape:
            continue

        probs = torch.softmax(logits, dim=1)
        probs = torch.where(torch.isfinite(logits), probs, torch.zeros_like(probs))
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)

        if pseudo_target is not None:
            row_loss = -(pseudo_target * torch.log(probs + 1e-12)).sum(dim=1)
            if str(getattr(hparams, "topology_loss_normalization", "none")) == "uniform_baseline":
                eligible = torch.isfinite(logits) & ~eye
                uniform_loss = torch.log(eligible.sum(dim=1).clamp_min(2).to(row_loss.dtype))
                row_loss = row_loss / uniform_loss
            pseudo_edge_terms.append(row_loss.mean())

    if not pseudo_edge_terms:
        zero = torch.tensor(0.0, device=device)
        return zero, {"soft_pseudo_edge": 0.0}

    soft_pseudo_edge = torch.stack(pseudo_edge_terms).mean()
    loss = pseudo_edge_weight * soft_pseudo_edge

    return loss, {
        "soft_pseudo_edge": float(soft_pseudo_edge.detach().item()),
    }


def _geometry_only_length_floor(dist: torch.Tensor, hparams) -> torch.Tensor:
    """Length floor from coordinates only, never from ground-truth edges."""
    n = dist.shape[0]
    k = int(getattr(hparams, "soft_length_candidate_k", getattr(hparams, "pseudo_candidate_k", 8)))
    q = float(getattr(hparams, "soft_length_quantile", 0.65))
    k = min(max(1, k), n - 1)
    q = max(0.0, min(1.0, q))

    work = dist.detach().clone()
    work.fill_diagonal_(float("inf"))
    nn = torch.topk(work, k=k, largest=False, dim=1).values.reshape(-1)
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return dist.new_tensor(0.0)
    return torch.quantile(nn, q).to(dist.device)


def _build_pseudo_edge_target(dist, x_flat, hparams, device):
    """
    Build a weak edge target Q_ij without using ground-truth edges.

    Q combines geometric proximity and temporal similarity, then keeps only the
    strongest candidates per source node. The DGM distribution P is trained with
    cross-entropy against Q, so the loss decreases when P assigns more mass to
    these observable proxy edges.
    """
    n = dist.shape[0]
    eye = torch.eye(n, device=device, dtype=torch.bool)

    mode = str(getattr(hparams, "pseudo_score_mode", "geo_temporal"))
    if bool(getattr(hparams, "pseudo_use_static_target", False)) or mode.startswith("observable_"):
        return _build_static_pseudo_edge_target(
            dist=dist,
            hparams=hparams,
            device=device,
        )

    geo_tau = max(_get_float_attr(hparams, "pseudo_geo_tau", 1.0), 1e-6)
    geo_affinity = torch.exp(-dist / geo_tau).masked_fill(eye, 0.0)

    mode = str(getattr(hparams, "pseudo_score_mode", "geo_temporal"))
    temporal_mix = max(0.0, min(1.0, _get_float_attr(hparams, "pseudo_temporal_weight", 0.5)))
    temporal_affinity = _build_temporal_affinity(
        x_flat=x_flat,
        hparams=hparams,
        n=n,
        device=device,
        mode=mode,
    )

    if mode == "geo_partial_bridge" and temporal_affinity is not None:
        return _build_local_bridge_target(
            geo_affinity=geo_affinity,
            temporal_affinity=temporal_affinity,
            dist=dist,
            hparams=hparams,
            device=device,
        )

    if mode == "geo_only" or temporal_affinity is None:
        score = geo_affinity
    elif mode == "geo_temporal":
        score = geo_affinity * ((1.0 - temporal_mix) + temporal_mix * temporal_affinity)
    elif mode == "temporal_only":
        score = temporal_affinity
    else:
        score = geo_affinity * temporal_affinity

    candidate_k = int(getattr(hparams, "pseudo_candidate_k", 0))
    if candidate_k > 0:
        candidate_k = min(candidate_k, n - 1)
        candidate_idx = torch.topk(dist.masked_fill(eye, float("inf")), k=candidate_k, largest=False, dim=1).indices
        candidate_mask = torch.zeros((n, n), device=device, dtype=torch.bool)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(candidate_idx)
        candidate_mask[rows, candidate_idx] = True
        candidate_mask = candidate_mask | candidate_mask.t()
        score = score.masked_fill(~candidate_mask, 0.0)

    topk = min(max(1, int(getattr(hparams, "pseudo_topk", 2))), n - 1)
    top_idx = torch.topk(score, k=topk, largest=True, dim=1).indices
    target = torch.zeros_like(score)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(top_idx)
    target[rows, top_idx] = score[rows, top_idx]

    include_mst = bool(getattr(hparams, "pseudo_include_mst", True))
    if include_mst:
        mst_edges = _geometric_mst_edges(dist)
        if mst_edges.numel() > 0:
            u = mst_edges[:, 0]
            v = mst_edges[:, 1]
            mst_bonus = float(getattr(hparams, "pseudo_mst_bonus", 1.0))
            target[u, v] = torch.maximum(target[u, v], geo_affinity[u, v] * mst_bonus)
            target[v, u] = torch.maximum(target[v, u], geo_affinity[v, u] * mst_bonus)

    target = target.masked_fill(eye, 0.0)
    row_sum = target.sum(dim=1, keepdim=True)
    fallback = geo_affinity / (geo_affinity.sum(dim=1, keepdim=True) + 1e-12)
    target = torch.where(row_sum > 0.0, target / (row_sum + 1e-12), fallback)
    return target


def _build_local_bridge_target(geo_affinity, temporal_affinity, dist, hparams, device):
    """
    Two-channel pseudo target:
      local  = short geometric edges with strong pressure evidence
      bridge = longer edges only when pressure evidence is strong

    The bridge channel receives a small fixed row mass, so long edges can be
    selected, but only sparingly.
    """
    n = dist.shape[0]
    eye = torch.eye(n, device=device, dtype=torch.bool)
    local_topk = min(max(1, int(getattr(hparams, "pseudo_topk", 2))), n - 1)
    bridge_topk = min(max(1, int(getattr(hparams, "pseudo_bridge_topk", 1))), n - 1)
    bridge_mass = max(0.0, min(0.5, _get_float_attr(hparams, "pseudo_bridge_mass", 0.15)))

    local_score = geo_affinity * temporal_affinity
    local_candidate_k = int(getattr(hparams, "pseudo_candidate_k", 0))
    if local_candidate_k > 0:
        local_candidate_k = min(local_candidate_k, n - 1)
        idx = torch.topk(dist.masked_fill(eye, float("inf")), k=local_candidate_k, largest=False, dim=1).indices
        local_mask = torch.zeros((n, n), device=device, dtype=torch.bool)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(idx)
        local_mask[rows, idx] = True
        local_mask = local_mask | local_mask.t()
        local_score = local_score.masked_fill(~local_mask, 0.0)

    bridge_score = _build_bridge_score(
        dist=dist,
        temporal_affinity=temporal_affinity,
        hparams=hparams,
        device=device,
    )

    local_target = _row_topk_target(local_score, local_topk)
    bridge_target = _row_topk_target(bridge_score, bridge_topk)
    fallback = geo_affinity / (geo_affinity.sum(dim=1, keepdim=True) + 1e-12)

    target = (1.0 - bridge_mass) * local_target + bridge_mass * bridge_target
    row_sum = target.sum(dim=1, keepdim=True)
    target = torch.where(row_sum > 0.0, target / (row_sum + 1e-12), fallback)
    return target.masked_fill(eye, 0.0)


def _build_bridge_score(dist, temporal_affinity, hparams, device):
    n = dist.shape[0]
    eye = torch.eye(n, device=device, dtype=torch.bool)
    q = max(0.0, min(1.0, _get_float_attr(hparams, "pseudo_bridge_quantile", 0.65)))
    temp = max(_get_float_attr(hparams, "pseudo_bridge_temperature", 0.08), 1e-6)

    work = dist.detach().clone().masked_fill(eye, float("nan"))
    finite = work[torch.isfinite(work)]
    if finite.numel() == 0:
        return torch.zeros_like(dist)
    threshold = torch.quantile(finite, q)
    long_gate = torch.sigmoid((dist - threshold) / temp).masked_fill(eye, 0.0)

    min_temporal = _get_float_attr(hparams, "pseudo_bridge_min_temporal", 0.55)
    evidence_gate = torch.relu(temporal_affinity - min_temporal) / max(1e-6, 1.0 - min_temporal)
    bridge_score = temporal_affinity * evidence_gate * long_gate

    bridge_candidate_k = int(getattr(hparams, "pseudo_bridge_candidate_k", 0))
    if bridge_candidate_k > 0:
        bridge_candidate_k = min(bridge_candidate_k, n - 1)
        idx = torch.topk(bridge_score.masked_fill(eye, -float("inf")), k=bridge_candidate_k, largest=True, dim=1).indices
        mask = torch.zeros((n, n), device=device, dtype=torch.bool)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(idx)
        mask[rows, idx] = True
        mask = mask | mask.t()
        bridge_score = bridge_score.masked_fill(~mask, 0.0)

    return bridge_score.masked_fill(eye, 0.0)


def _row_topk_target(score, topk):
    n = score.shape[0]
    eye = torch.eye(n, device=score.device, dtype=torch.bool)
    topk = min(max(1, int(topk)), n - 1)
    work = score.masked_fill(eye, 0.0)
    top_idx = torch.topk(work, k=topk, largest=True, dim=1).indices
    target = torch.zeros_like(work)
    rows = torch.arange(n, device=score.device).unsqueeze(1).expand_as(top_idx)
    target[rows, top_idx] = work[rows, top_idx]
    target = target.masked_fill(eye, 0.0)
    row_sum = target.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0.0, target / (row_sum + 1e-12), torch.zeros_like(target))


def _build_static_pseudo_edge_target(dist, hparams, device):
    """Static Q from all observed node series in data_npz, never from GT edges."""
    cache_key = (
        "static_pseudo_target",
        str(getattr(hparams, "data_npz", "")),
        str(getattr(hparams, "pseudo_score_mode", "geo_temporal")),
        int(getattr(hparams, "pseudo_candidate_k", 0)),
        int(getattr(hparams, "pseudo_topk", 2)),
        float(getattr(hparams, "pseudo_partial_ridge", 0.05)),
        int(getattr(hparams, "pseudo_max_lag", 6)),
        int(getattr(hparams, "pseudo_bridge_topk", 1)),
        float(getattr(hparams, "pseudo_bridge_mass", 0.15)),
        float(getattr(hparams, "pseudo_bridge_quantile", 0.65)),
        float(getattr(hparams, "pseudo_bridge_temperature", 0.08)),
        float(getattr(hparams, "pseudo_bridge_min_temporal", 0.55)),
        int(getattr(hparams, "pseudo_bridge_candidate_k", 0)),
    )
    cached = getattr(hparams, "_pseudo_target_cache", None)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached["target"].to(device)

    data_npz = getattr(hparams, "data_npz", None)
    if not data_npz:
        raise ValueError("pseudo_use_static_target=True requires hparams.data_npz")
    data = np.load(data_npz)
    pressure_np = data["pressure"].astype(np.float32)
    demand_np = data["demand"].astype(np.float32) if "demand" in data.files else None

    n = dist.shape[0]
    pressure = torch.as_tensor(pressure_np[:, :n], dtype=dist.dtype, device=device)
    demand = (
        torch.as_tensor(demand_np[:, :n], dtype=dist.dtype, device=device)
        if demand_np is not None
        else None
    )

    mode = str(getattr(hparams, "pseudo_score_mode", "geo_temporal"))
    geo_tau = max(_get_float_attr(hparams, "pseudo_geo_tau", 1.0), 1e-6)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    geo_affinity = torch.exp(-dist / geo_tau).masked_fill(eye, 0.0)

    prebuilt_target = None

    if mode in {"observable_geometry", "observable_geometry_plus", "observable_expert_pressure", "observable_expert_dynamic", "observable_expert_full", "observable_electric_voltage", "observable_electric_dynamic", "observable_electric_grid", "observable_electric_inverse", "observable_electric_full"}:
        geo_rank = _geometry_rank_affinity(dist)
        geometry = geo_rank
        if mode not in {"observable_geometry"}:
            geometry = _offdiag_minmax(geo_rank * _nonmediated_geometry_affinity(dist))
        if mode in {"observable_geometry", "observable_geometry_plus"}:
            score = geometry
        else:
            pressure_smooth = _pressure_smooth_affinity(pressure)
            if demand is None:
                demand_global = _global_var_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
                pressure_from_demand = demand_global
            else:
                demand_global = _global_var_affinity(demand, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
                pressure_from_demand = _global_cross_var_affinity(
                    source=demand,
                    target=pressure,
                    ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)),
                )
            dynamics = _offdiag_minmax(torch.maximum(demand_global, pressure_from_demand))
            if mode in {"observable_expert_pressure", "observable_electric_voltage"}:
                score = _offdiag_minmax(geometry * pressure_smooth)
            elif mode == "observable_expert_dynamic":
                score = _offdiag_minmax(geometry * dynamics)
            elif mode == "observable_electric_dynamic":
                score = _offdiag_minmax(geometry * pressure_smooth * dynamics)
            elif mode == "observable_electric_grid":
                backbone = _geometric_backbone_affinity(dist)
                score = _offdiag_minmax(geometry * pressure_smooth * dynamics * (0.65 + 0.35 * backbone))
            elif mode in {"observable_electric_inverse", "observable_electric_full"}:
                inverse_response = _inverse_source_target_affinity(demand if demand is not None else pressure, pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
                if mode == "observable_electric_inverse":
                    score = _offdiag_minmax(geometry * inverse_response)
                else:
                    score = _offdiag_minmax(geometry * pressure_smooth * inverse_response)
            else:
                score = _offdiag_minmax(geometry * pressure_smooth * dynamics)
    elif mode == "geo_only":
        score = geo_affinity
    elif mode == "demand_corr" and demand is not None:
        score = _positive_corr_affinity(demand)
    elif mode == "pressure_corr":
        score = _positive_corr_affinity(pressure)
    elif mode in {"pressure_smooth", "geo_pressure_smooth"}:
        temporal = _pressure_smooth_affinity(pressure)
        score = temporal if mode == "pressure_smooth" else geo_affinity * temporal
    elif mode == "pressure_partial":
        score = _partial_corr_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
    elif mode == "geo_partial_smooth":
        partial = _partial_corr_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
        smooth = _pressure_smooth_affinity(pressure)
        score = geo_affinity * _offdiag_minmax(partial * smooth)
    elif mode in {"global_dynamic_proxy", "geo_global_dynamic_proxy", "geo_global_dynamic_bridge_proxy"}:
        if demand is None:
            demand_global = _global_var_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
            pressure_from_demand = demand_global
        else:
            demand_global = _global_var_affinity(demand, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
            pressure_from_demand = _global_cross_var_affinity(
                source=demand,
                target=pressure,
                ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)),
            )
        smooth = _pressure_smooth_affinity(pressure)
        dynamic = _offdiag_minmax(smooth * torch.maximum(demand_global, pressure_from_demand))
        if mode == "global_dynamic_proxy":
            score = dynamic
        elif mode == "geo_global_dynamic_proxy":
            score = geo_affinity * dynamic
        else:
            prebuilt_target = _build_static_global_dynamic_bridge_target(
                geo_affinity=geo_affinity,
                dynamic_affinity=dynamic,
                dist=dist,
                hparams=hparams,
                device=device,
            )
            score = geo_affinity * dynamic
    elif mode == "geo_pressure_evidence":
        corr = _positive_corr_affinity(pressure)
        smooth = _pressure_smooth_affinity(pressure)
        lag = _static_lag_affinity(pressure, max_lag=int(getattr(hparams, "pseudo_max_lag", 6)))
        score = geo_affinity * _offdiag_minmax(corr * smooth * lag)
    else:
        corr = _positive_corr_affinity(pressure)
        score = geo_affinity * corr

    if prebuilt_target is not None:
        hparams._pseudo_target_cache = {"key": cache_key, "target": prebuilt_target.detach().cpu()}
        return prebuilt_target

    candidate_k = int(getattr(hparams, "pseudo_candidate_k", 0))
    if candidate_k > 0:
        candidate_k = min(candidate_k, n - 1)
        candidate_idx = torch.topk(dist.masked_fill(eye, float("inf")), k=candidate_k, largest=False, dim=1).indices
        candidate_mask = torch.zeros((n, n), device=device, dtype=torch.bool)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(candidate_idx)
        candidate_mask[rows, candidate_idx] = True
        candidate_mask = candidate_mask | candidate_mask.t()
        score = score.masked_fill(~candidate_mask, 0.0)

    topk = min(max(1, int(getattr(hparams, "pseudo_topk", 2))), n - 1)
    top_idx = torch.topk(score, k=topk, largest=True, dim=1).indices
    target = torch.zeros_like(score)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(top_idx)
    target[rows, top_idx] = score[rows, top_idx]
    target = target.masked_fill(eye, 0.0)
    row_sum = target.sum(dim=1, keepdim=True)
    fallback = geo_affinity / (geo_affinity.sum(dim=1, keepdim=True) + 1e-12)
    target = torch.where(row_sum > 0.0, target / (row_sum + 1e-12), fallback)

    hparams._pseudo_target_cache = {"key": cache_key, "target": target.detach().cpu()}
    return target


def _build_static_global_dynamic_bridge_target(geo_affinity, dynamic_affinity, dist, hparams, device):
    """Static two-channel Q for sparse bridge edges, using only observable signals."""
    n = dist.shape[0]
    eye = torch.eye(n, device=device, dtype=torch.bool)
    local_topk = min(max(1, int(getattr(hparams, "pseudo_topk", 2))), n - 1)
    bridge_topk = min(max(1, int(getattr(hparams, "pseudo_bridge_topk", 1))), n - 1)
    bridge_mass = max(0.0, min(0.5, _get_float_attr(hparams, "pseudo_bridge_mass", 0.15)))

    local_score = geo_affinity * dynamic_affinity
    local_candidate_k = int(getattr(hparams, "pseudo_candidate_k", 0))
    if local_candidate_k > 0:
        local_candidate_k = min(local_candidate_k, n - 1)
        idx = torch.topk(dist.masked_fill(eye, float("inf")), k=local_candidate_k, largest=False, dim=1).indices
        local_mask = torch.zeros((n, n), device=device, dtype=torch.bool)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(idx)
        local_mask[rows, idx] = True
        local_mask = local_mask | local_mask.t()
        local_score = local_score.masked_fill(~local_mask, 0.0)

    bridge_score = _build_bridge_score(
        dist=dist,
        temporal_affinity=dynamic_affinity,
        hparams=hparams,
        device=device,
    )

    local_target = _row_topk_target(local_score, local_topk)
    bridge_target = _row_topk_target(bridge_score, bridge_topk)
    fallback = geo_affinity / (geo_affinity.sum(dim=1, keepdim=True) + 1e-12)

    target = (1.0 - bridge_mass) * local_target + bridge_mass * bridge_target

    include_mst = bool(getattr(hparams, "pseudo_include_mst", True))
    if include_mst:
        mst_edges = _geometric_mst_edges(dist)
        if mst_edges.numel() > 0:
            u = mst_edges[:, 0]
            v = mst_edges[:, 1]
            mst_bonus = float(getattr(hparams, "pseudo_mst_bonus", 1.0))
            target[u, v] = torch.maximum(target[u, v], geo_affinity[u, v] * mst_bonus)
            target[v, u] = torch.maximum(target[v, u], geo_affinity[v, u] * mst_bonus)

    target = target.masked_fill(eye, 0.0)
    row_sum = target.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0.0, target / (row_sum + 1e-12), fallback).masked_fill(eye, 0.0)


def _build_temporal_affinity(x_flat, hparams, n, device, mode):
    """Pressure/demand affinity from observed windows only, no GT edges."""
    if x_flat is None or x_flat.shape[0] % n != 0:
        return None

    input_features = int(getattr(hparams, "input_features", 2))
    if input_features <= 0 or x_flat.shape[-1] % input_features != 0:
        z = x_flat.detach().reshape(-1, n, x_flat.shape[-1])
        z = F.normalize(z, p=2, dim=-1)
        sim = torch.matmul(z, z.transpose(1, 2)).mean(dim=0)
        return _offdiag_minmax(0.5 * (sim + 1.0))

    lags = x_flat.shape[-1] // input_features
    seq = x_flat.detach().reshape(-1, n, lags, input_features)
    pressure = seq[..., 0].permute(0, 2, 1).reshape(-1, n)
    demand = seq[..., 1].permute(0, 2, 1).reshape(-1, n) if input_features > 1 else None

    if mode == "demand_corr":
        return _positive_corr_affinity(demand) if demand is not None else None
    if mode == "pressure_corr":
        return _positive_corr_affinity(pressure)
    if mode == "pressure_smooth":
        return _pressure_smooth_affinity(pressure)
    if mode == "pressure_lag":
        return _pressure_lag_affinity(seq[..., 0], max_lag=int(getattr(hparams, "pseudo_max_lag", 6)))
    if mode == "pressure_partial":
        return _partial_corr_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
    if mode == "geo_pressure_smooth":
        return _pressure_smooth_affinity(pressure)
    if mode == "geo_pressure_evidence":
        corr = _positive_corr_affinity(pressure)
        smooth = _pressure_smooth_affinity(pressure)
        lag = _pressure_lag_affinity(seq[..., 0], max_lag=int(getattr(hparams, "pseudo_max_lag", 6)))
        return _offdiag_minmax(corr * smooth * lag)
    if mode in {"geo_partial_smooth", "geo_partial_bridge"}:
        partial = _partial_corr_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
        smooth = _pressure_smooth_affinity(pressure)
        return _offdiag_minmax(partial * smooth)
    if mode in {"global_dynamic_proxy", "geo_global_dynamic_proxy"}:
        if demand is None:
            demand_global = _global_var_affinity(pressure, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
            pressure_from_demand = demand_global
        else:
            demand_global = _global_var_affinity(demand, ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)))
            pressure_from_demand = _global_cross_var_affinity(
                source=demand,
                target=pressure,
                ridge=float(getattr(hparams, "pseudo_partial_ridge", 0.05)),
            )
        smooth = _pressure_smooth_affinity(pressure)
        return _offdiag_minmax(smooth * torch.maximum(demand_global, pressure_from_demand))

    z = F.normalize(x_flat.detach().reshape(-1, n, x_flat.shape[-1]), p=2, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2)).mean(dim=0)
    return _offdiag_minmax(0.5 * (sim + 1.0))


def _offdiag_minmax(score):
    n = score.shape[0]
    eye = torch.eye(n, device=score.device, dtype=torch.bool)
    score = score.masked_fill(eye, 0.0)
    vals = score[~eye]
    finite = torch.isfinite(vals)
    if not finite.any():
        return torch.zeros_like(score)
    lo = vals[finite].min()
    hi = vals[finite].max()
    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    score = (score - lo) / (hi - lo + 1e-12)
    score = score.clamp(0.0, 1.0).masked_fill(eye, 0.0)
    return score


def _geometry_rank_affinity(dist):
    n = dist.shape[0]
    eye = torch.eye(n, device=dist.device, dtype=torch.bool)
    work = dist.detach().clone().masked_fill(eye, float("inf"))
    order = torch.argsort(work, dim=1)
    ranks = torch.empty_like(order, dtype=dist.dtype)
    rank_values = torch.arange(n, device=dist.device, dtype=dist.dtype).view(1, n).expand(n, n)
    ranks.scatter_(1, order, rank_values)
    score = 1.0 / (1.0 + ranks)
    score = 0.5 * (score + score.t())
    return _offdiag_minmax(score.masked_fill(eye, 0.0))


def _geometric_backbone_affinity(dist):
    """Observable hub/backbone plausibility from coordinates only."""
    n = dist.shape[0]
    eye = torch.eye(n, device=dist.device, dtype=torch.bool)
    work = dist.detach().clone().masked_fill(eye, float("inf"))
    finite = work[torch.isfinite(work)]
    if finite.numel() == 0:
        return torch.zeros_like(dist)
    closeness = 1.0 / (work.mean(dim=1) + 1e-12)
    closeness = (closeness - closeness.min()) / (closeness.max() - closeness.min() + 1e-12)
    moderate = torch.exp(-torch.abs(work - torch.quantile(finite, 0.35)) / (torch.std(finite) + 1e-12))
    hub_pair = torch.maximum(closeness[:, None], closeness[None, :])
    return _offdiag_minmax(hub_pair * moderate)


def _nonmediated_geometry_affinity(dist):
    n = dist.shape[0]
    out = torch.zeros_like(dist)
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                witness = torch.maximum(dist[i], dist[j]) < (dist[i, j] - 1e-12)
                witness[i] = False
                witness[j] = False
                if not bool(witness.any()):
                    out[i, j] = 1.0
                    out[j, i] = 1.0
    return 0.25 + 0.75 * out


def _standardize_time(x):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-12)


def _positive_corr_affinity(x):
    if x is None:
        return None
    z = _standardize_time(x)
    corr = z.t().matmul(z) / max(1, z.shape[0] - 1)
    return _offdiag_minmax(torch.relu(corr))


def _pressure_smooth_affinity(pressure):
    """Pressure smoothness score computed without materializing [T, N, N].

    Analytically this is unchanged:
        S_ij = exp(- mean_t |p_i(t) - p_j(t)| / median_positive_diff)

    The chunked implementation is needed for C-Town-size graphs, where the
    naive [T, N, N] tensor can require several GB of memory.
    """
    n = pressure.shape[1]
    chunk_size = 32
    diff = pressure.new_empty((n, n))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = torch.mean(
            torch.abs(pressure[:, start:stop, None] - pressure[:, None, :]),
            dim=0,
        )
        diff[start:stop] = block
    positive = diff[diff > 0]
    scale = torch.median(positive) if positive.numel() > 0 else diff.new_tensor(1.0)
    return _offdiag_minmax(torch.exp(-diff / (scale + 1e-12)))


def _pressure_lag_affinity(pressure_seq, max_lag):
    # pressure_seq: [B, N, L]
    b, n, lags = pressure_seq.shape
    if lags < 2:
        return torch.zeros((n, n), device=pressure_seq.device, dtype=pressure_seq.dtype)
    max_lag = min(max(1, int(max_lag)), lags - 1)
    best = torch.zeros((n, n), device=pressure_seq.device, dtype=pressure_seq.dtype)
    p = pressure_seq.permute(0, 2, 1)
    for lag in range(1, max_lag + 1):
        a = p[:, lag:, :].reshape(-1, n)
        c = _positive_corr_affinity(a)
        b_lag = p[:, :-lag, :].reshape(-1, n)
        a_lag = p[:, lag:, :].reshape(-1, n)
        za = _standardize_time(a_lag)
        zb = _standardize_time(b_lag)
        lag_corr = za.t().matmul(zb) / max(1, za.shape[0] - 1)
        lag_corr = torch.maximum(lag_corr, lag_corr.t())
        best = torch.maximum(best, torch.maximum(c, _offdiag_minmax(torch.relu(lag_corr))))
    return _offdiag_minmax(best)


def _static_lag_affinity(pressure, max_lag):
    if pressure.shape[0] < 2:
        return torch.zeros((pressure.shape[1], pressure.shape[1]), device=pressure.device, dtype=pressure.dtype)
    max_lag = min(max(1, int(max_lag)), pressure.shape[0] - 1)
    n = pressure.shape[1]
    best = torch.zeros((n, n), device=pressure.device, dtype=pressure.dtype)
    for lag in range(1, max_lag + 1):
        a = pressure[lag:]
        b = pressure[:-lag]
        za = _standardize_time(a)
        zb = _standardize_time(b)
        lag_corr = za.t().matmul(zb) / max(1, za.shape[0] - 1)
        lag_corr = torch.maximum(lag_corr, lag_corr.t())
        best = torch.maximum(best, _offdiag_minmax(torch.relu(lag_corr)))
    return _offdiag_minmax(best)


def _partial_corr_affinity(x, ridge):
    z = _standardize_time(x)
    cov = z.t().matmul(z) / max(1, z.shape[0] - 1)
    cov = cov + float(ridge) * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    precision = torch.linalg.pinv(cov)
    d = torch.sqrt(torch.clamp(torch.diag(precision), min=1e-12))
    pcorr = -precision / (d[:, None] * d[None, :] + 1e-12)
    return _offdiag_minmax(torch.relu(pcorr))


def _inverse_source_target_affinity(source, target, ridge):
    """Inverse response score from observable source changes to target changes."""
    if source is None or target is None or source.shape[0] < 3:
        n = target.shape[1]
        return torch.zeros((n, n), device=target.device, dtype=target.dtype)
    source_diff = _standardize_time(source[1:] - source[:-1])
    target_diff = _standardize_time(target[1:] - target[:-1])
    n = target.shape[1]
    eye = torch.eye(n, device=target.device, dtype=target.dtype)
    gram = source_diff.t().matmul(source_diff) + float(ridge) * eye
    response = torch.linalg.pinv(gram).matmul(source_diff.t().matmul(target_diff))
    inverse = torch.linalg.pinv(response)
    score = torch.maximum(torch.abs(inverse), torch.abs(inverse).t())
    score = score.masked_fill(torch.eye(n, device=target.device, dtype=torch.bool), 0.0)
    return _offdiag_minmax(score)


def _global_var_affinity(x, ridge):
    """Off-diagonal coefficients of X[t+1] = X[t] B, no GT edges."""
    if x is None or x.shape[0] < 2:
        n = 0 if x is None else x.shape[1]
        return torch.zeros((n, n), device=x.device if x is not None else "cpu")
    z = _standardize_time(x)
    source = z[:-1]
    target = z[1:]
    n = source.shape[1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    gram = source.t().matmul(source) + float(ridge) * eye
    coef = torch.linalg.pinv(gram).matmul(source.t().matmul(target))
    score = torch.maximum(torch.abs(coef), torch.abs(coef).t())
    score = score.masked_fill(eye.bool(), 0.0)
    return _offdiag_minmax(score)


def _global_cross_var_affinity(source, target, ridge):
    """All source nodes at t predict all target nodes at t+1, symmetrized."""
    if source is None or target is None or source.shape[0] < 2:
        n = target.shape[1]
        return torch.zeros((n, n), device=target.device, dtype=target.dtype)
    source_z = _standardize_time(source)
    target_z = _standardize_time(target)
    n = target_z.shape[1]
    design = torch.cat([target_z[:-1], source_z[:-1]], dim=1)
    y = target_z[1:]
    eye = torch.eye(2 * n, device=target.device, dtype=target.dtype)
    gram = design.t().matmul(design) + float(ridge) * eye
    coef = torch.linalg.pinv(gram).matmul(design.t().matmul(y))
    source_to_target = torch.abs(coef[n:, :])
    score = torch.maximum(source_to_target, source_to_target.t())
    score = score.masked_fill(torch.eye(n, device=target.device, dtype=torch.bool), 0.0)
    return _offdiag_minmax(score)


def _geometric_mst_edges(dist: torch.Tensor) -> torch.Tensor:
    """Prim MST over fixed geometric distances; returns [N-1, 2] long edges."""
    with torch.no_grad():
        n = dist.shape[0]
        if n <= 1:
            return torch.empty((0, 2), device=dist.device, dtype=torch.long)

        visited = torch.zeros(n, device=dist.device, dtype=torch.bool)
        visited[0] = True
        edges = []

        work_dist = dist.detach().clone()
        work_dist.fill_diagonal_(float("inf"))

        for _ in range(n - 1):
            masked = work_dist.masked_fill(~visited[:, None], float("inf"))
            masked = masked.masked_fill(visited[None, :], float("inf"))
            flat_idx = torch.argmin(masked)
            u = torch.div(flat_idx, n, rounding_mode="floor")
            v = flat_idx % n
            if not torch.isfinite(masked[u, v]):
                break
            edges.append(torch.stack([u, v]))
            visited[v] = True

        if not edges:
            return torch.empty((0, 2), device=dist.device, dtype=torch.long)
        return torch.stack(edges, dim=0).long()




def _mean_logprob_per_node(logprobs: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Supports both:
      [B, N, K]
      [B, N, K, L]

    and also the flattened-DGM case used here:
      [1, B*N, K]
      [1, B*N, K, L]

    Returns:
      node_score: [N]
    """
    if logprobs is None:
        raise ValueError("logprobs is None")

    if logprobs.dim() == 3:
        # [B0, Ntot, K]
        B0, Ntot, K = logprobs.shape

        if Ntot == num_nodes:
            return logprobs.mean(dim=(0, 2))   # [N]

        if Ntot % num_nodes != 0:
            raise ValueError(
                f"Unexpected logprobs shape {tuple(logprobs.shape)}: "
                f"second dim {Ntot} is neither num_nodes={num_nodes} "
                f"nor a multiple of it"
            )

        B = Ntot // num_nodes
        logprobs = logprobs.reshape(B0, B, num_nodes, K)
        return logprobs.mean(dim=(0, 1, 3))   # [N]

    elif logprobs.dim() == 4:
        # [B0, Ntot, K, L]
        B0, Ntot, K, L = logprobs.shape

        if Ntot == num_nodes:
            return logprobs.mean(dim=(0, 2, 3))   # [N]

        if Ntot % num_nodes != 0:
            raise ValueError(
                f"Unexpected logprobs shape {tuple(logprobs.shape)}: "
                f"second dim {Ntot} is neither num_nodes={num_nodes} "
                f"nor a multiple of it"
            )

        B = Ntot // num_nodes
        logprobs = logprobs.reshape(B0, B, num_nodes, K, L)
        return logprobs.mean(dim=(0, 1, 3, 4))   # [N]

    else:
        raise ValueError(f"Unexpected logprobs shape: {tuple(logprobs.shape)}")


def _nodewise_forecast_error(y_predicted, y_true, hparams, num_nodes):
    """
    Scalar forecasting error per node, averaged over batch, horizon and features.
    Returns:
      err_node: [N]
    """
    H = int(getattr(hparams, "prediction_window", 1))
    Fout = int(getattr(hparams, "output_features", 1))

    y_pred = y_predicted.view(-1, H, Fout)
    y_true = y_true.view(-1, H, Fout)

    sq = (y_pred - y_true).pow(2)  # [B*N, H, F]

    if _is_water_like_dataset(hparams) and Fout >= 2:
        wp = float(getattr(hparams, "pressure_weight", 1.0))
        wd = float(getattr(hparams, "demand_weight", 1.0))

        feat_w = torch.ones(Fout, device=sq.device, dtype=sq.dtype)
        feat_w[0] = wp
        feat_w[1] = wd
        sq = sq * feat_w.view(1, 1, -1)

    err_bn = sq.mean(dim=(1, 2))                    # [B*N]
    err_node = err_bn.view(-1, num_nodes).mean(0)  # [N]
    return err_node


def _task_driven_graph_loss(model, y_predicted, y_true, logprobs, hparams):
    """
    Task-driven graph loss for water-network forecasting.
    """
    device = y_predicted.device

    if logprobs is None:
        return torch.tensor(0.0, device=device), {}

    err_node = _nodewise_forecast_error(
        y_predicted=y_predicted,
        y_true=y_true,
        hparams=hparams,
        num_nodes=model.num_nodes,
    ).detach()   # [N]

    if model.avg_accuracy is None or model.avg_accuracy.shape != err_node.shape:
        model.avg_accuracy = err_node.clone()

    # positive delta => current node is worse than its EMA baseline
    delta = err_node - model.avg_accuracy

    tau = float(getattr(hparams, "graph_delta_tau", 0.01))
    beta = float(getattr(hparams, "graph_ema_beta", 0.95))

    point_w = F.softplus(delta / tau)               # [N]
    point_w = point_w / (point_w.mean() + 1e-12)   # normalize for stability

    node_graph_cost = -_mean_logprob_per_node(
        logprobs=logprobs,
        num_nodes=model.num_nodes,
    )  # [N]

    if node_graph_cost.shape != point_w.shape:
        raise ValueError(
            f"Shape mismatch in task-driven graph loss: "
            f"point_w={tuple(point_w.shape)}, node_graph_cost={tuple(node_graph_cost.shape)}"
        )

    graph_loss = (point_w * node_graph_cost).mean()

    model.avg_accuracy = (
        beta * model.avg_accuracy.to(err_node.device)
        + (1.0 - beta) * err_node
    )

    stats = {
        "err_node_mean": float(err_node.mean().item()),
        "delta_mean": float(delta.mean().item()),
        "point_w_mean": float(point_w.mean().item()),
        "node_graph_cost_mean": float(node_graph_cost.mean().item()),
    }
    return graph_loss, stats


def _reshape_ctown_epanet_batch(batch, hparams, device):
    """
    PyG collate for ctown_epanet currently returns:
        x       : [B*L, N, F]
        y       : [B,   N, F]   when prediction_window = 1
        node_xy : [B*N, 2]

    This function reconstructs:
        x       : [B, L, N, F]
        y       : [B, H, N, F]
        node_xy : [B, N, 2]
    """
    x = batch.x.to(device)
    y = batch.y.to(device)

    L = int(hparams.lags)
    H = int(hparams.prediction_window)
    N = int(hparams.num_nodes)
    Fdim = int(getattr(hparams, "input_features", 2))

    if x.dim() != 3:
        raise ValueError(f"ctown_epanet: expected x with dim=3, got shape {tuple(x.shape)}")

    if y.dim() not in [3, 4]:
        raise ValueError(f"ctown_epanet: expected y with dim=3 or 4, got shape {tuple(y.shape)}")

    # x: [B*L, N, F] -> [B, L, N, F]
    if x.shape[1] != N or x.shape[2] != Fdim:
        raise ValueError(
            f"ctown_epanet: unexpected x shape {tuple(x.shape)}, expected [B*L, {N}, {Fdim}]"
        )
    if x.shape[0] % L != 0:
        raise ValueError(
            f"ctown_epanet: first x dim {x.shape[0]} is not divisible by lags={L}"
        )
    B = x.shape[0] // L
    x = x.reshape(B, L, N, Fdim)

    # y can come as:
    #   [B, N, F] when H=1
    #   [B*H, N, F] or [B, H, N, F] in other cases
    if y.dim() == 3:
        if y.shape[0] == B and y.shape[1] == N and y.shape[2] == Fdim:
            y = y.unsqueeze(1)  # [B, 1, N, F]
        elif y.shape[0] == B * H and y.shape[1] == N and y.shape[2] == Fdim:
            y = y.reshape(B, H, N, Fdim)
        else:
            raise ValueError(
                f"ctown_epanet: unexpected y shape {tuple(y.shape)} for B={B}, H={H}, N={N}, F={Fdim}"
            )
    elif y.dim() == 4:
        # already [B, H, N, F]
        pass

    # reconstruct node_xy if present
    if hasattr(batch, "node_xy") and batch.node_xy is not None:
        node_xy = batch.node_xy.to(device)
        if node_xy.dim() == 2 and node_xy.shape[0] == B * N and node_xy.shape[1] == 2:
            batch.node_xy = node_xy.reshape(B, N, 2)
        elif node_xy.dim() == 2 and node_xy.shape[0] == N and node_xy.shape[1] == 2:
            batch.node_xy = node_xy.unsqueeze(0).repeat(B, 1, 1)
        elif node_xy.dim() == 3:
            pass
        else:
            raise ValueError(
                f"ctown_epanet: unexpected node_xy shape {tuple(node_xy.shape)} for B={B}, N={N}"
            )

    return x, y, batch


def _flatten_for_dgm(x, y):
    """
    [B, L, N, F] -> [(B*N), (L*F)]
    [B, H, N, F] -> [(B*N), (H*F)]
    """
    if rearrange is None:
        raise RuntimeError("einops.rearrange is required.")
    x = rearrange(x, "b l n f -> (b n) (l f)")
    y = rearrange(y, "b h n f -> (b n) (h f)")
    return x, y


def _canonical_undirected(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _component_count(num_nodes: int, edge_set: set[tuple[int, int]]) -> int:
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edge_set:
        union(int(u), int(v))
    return len({find(i) for i in range(num_nodes)})


def _edge_length_stats(edge_set, node_xy):
    if not edge_set:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    lengths = []
    for u, v in edge_set:
        d = node_xy[int(u)] - node_xy[int(v)]
        lengths.append(float(np.sqrt((d * d).sum())))
    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _candidate_length_target(node_xy, hparams):
    n = int(hparams.num_nodes)
    k = int(getattr(hparams, "proxy_length_candidate_k", getattr(hparams, "pseudo_candidate_k", 8)))
    q = float(getattr(hparams, "proxy_length_quantile", 0.65))
    q = max(0.0, min(1.0, q))

    xy = torch.as_tensor(node_xy[:n], dtype=torch.float32)
    dist = torch.cdist(xy, xy, p=2).numpy()
    np.fill_diagonal(dist, np.inf)

    k = min(max(1, k), n - 1)
    nn_idx = np.argsort(dist, axis=1)[:, :k]
    lengths = []
    for u in range(n):
        for v in nn_idx[u]:
            lengths.append(float(dist[u, v]))

    if not lengths:
        return 0.0
    return float(np.quantile(np.asarray(lengths, dtype=np.float32), q))


@torch.no_grad()
def _compute_topology_proxy_score(model, val_loader, device, hparams):
    """
    GT-free static topology proxy for model selection.

    Lower is better. It discourages the failure mode observed in the synthetic
    benchmarks: good forecasting but consensus edges collapsed onto only the
    very shortest geometric links.
    """
    if val_loader is None:
        return None

    model.eval()
    n = int(hparams.num_nodes)
    top_edges = int(getattr(hparams, "static_top_edges", 0))
    edge_counts: dict[tuple[int, int], int] = {}
    n_graphs = 0
    node_xy_ref = None

    for batch in val_loader:
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None
        x, y, batch = _reshape_ctown_epanet_batch(batch, hparams, device)
        if node_xy_ref is None and hasattr(batch, "node_xy") and batch.node_xy is not None:
            xy = batch.node_xy.detach().cpu().numpy()
            node_xy_ref = xy[0] if xy.ndim == 3 else xy[:n]

        x, y = _flatten_for_dgm(x, y)
        _ = model(x, edge_index, edge_weight)

        if not hasattr(model, "edges") or model.edges is None:
            continue
        edges_np = model.edges.detach().cpu().numpy().astype(np.int64)
        src = edges_np[0]
        dst = edges_np[1]
        if src.size == 0:
            continue
        batch_ids = src // n
        for batch_id in range(int(batch_ids.max()) + 1):
            mask = batch_ids == batch_id
            local_edges = set()
            for u, v in zip((src[mask] - batch_id * n).tolist(), (dst[mask] - batch_id * n).tolist()):
                if u == v:
                    continue
                if 0 <= u < n and 0 <= v < n:
                    local_edges.add(_canonical_undirected(int(u), int(v)))
            if local_edges:
                n_graphs += 1
                for edge in local_edges:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

    if not edge_counts or node_xy_ref is None:
        return None

    if top_edges <= 0:
        top_edges = len(edge_counts)
    consensus = [
        edge
        for edge, _ in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)[:top_edges]
    ]
    consensus_set = set(consensus)

    length_stats = _edge_length_stats(consensus_set, node_xy_ref)
    target_length = _candidate_length_target(node_xy_ref, hparams)
    mean_length = length_stats["mean"]
    shortfall = max(0.0, target_length - mean_length) / max(target_length, 1e-12)

    components = _component_count(n, consensus_set)
    component_penalty = max(0, components - 1) / max(1, n - 1)

    counts = np.asarray([edge_counts[e] for e in consensus], dtype=np.float32)
    frequency_cv = float(counts.std() / (counts.mean() + 1e-12)) if counts.size else 0.0

    w_len = float(getattr(hparams, "proxy_length_weight", 1.0))
    w_conn = float(getattr(hparams, "proxy_connectivity_weight", 1.0))
    w_freq = float(getattr(hparams, "proxy_frequency_weight", 0.05))
    score = w_len * shortfall + w_conn * component_penalty - w_freq * frequency_cv

    return {
        "score": float(score),
        "components": int(components),
        "edge_length_mean": float(mean_length),
        "edge_length_target": float(target_length),
        "edge_length_shortfall": float(shortfall),
        "frequency_cv": float(frequency_cv),
        "n_graphs": int(n_graphs),
        "n_consensus_edges": int(len(consensus_set)),
    }


def _pointwise_forecast_loss(pred, true, loss_type: str, huber_delta: float):
    if loss_type == "mae":
        return torch.abs(pred - true).mean()
    if loss_type == "huber":
        return F.huber_loss(pred, true, delta=float(huber_delta))
    return F.mse_loss(pred, true)


def _forecasting_loss_ctown(y_predicted, y_true, hparams):
    """Forecast loss with optional per-feature scale normalization.

    The reported pressure/demand losses stay on the original normalized data
    scale. The optimized task loss can additionally divide each feature by an
    observed target scale so that pressure and demand do not dominate only
    because their numeric variance differs.
    """
    H = int(hparams.prediction_window)
    Fout = int(getattr(hparams, "output_features", 2))

    y_pred = y_predicted.view(-1, H, Fout)
    y_true = y_true.view(-1, H, Fout)

    p_pred = y_pred[:, :, 0]
    p_true = y_true[:, :, 0]

    d_pred = y_pred[:, :, 1]
    d_true = y_true[:, :, 1]

    loss_type = str(getattr(hparams, "task_loss_type", "mse"))
    huber_delta = _get_float_attr(hparams, "task_huber_delta", 1.0)
    task_target = str(getattr(hparams, "task_target", "all"))
    combine = str(getattr(hparams, "task_feature_combine", "weighted_sum"))
    normalization = str(getattr(hparams, "task_feature_normalization", "none"))

    wp = _get_float_attr(hparams, "pressure_weight", 1.0)
    wd = _get_float_attr(hparams, "demand_weight", 1.0)

    raw_loss_p = _pointwise_forecast_loss(p_pred, p_true, loss_type, huber_delta)
    raw_loss_d = _pointwise_forecast_loss(d_pred, d_true, loss_type, huber_delta)

    if normalization == "target_std":
        p_scale = p_true.detach().std().clamp_min(1e-6)
        d_scale = d_true.detach().std().clamp_min(1e-6)
        p_pred_loss = p_pred / p_scale
        p_true_loss = p_true / p_scale
        d_pred_loss = d_pred / d_scale
        d_true_loss = d_true / d_scale
    elif normalization == "target_mad":
        p_center = p_true.detach().median()
        d_center = d_true.detach().median()
        p_scale = (p_true.detach() - p_center).abs().median().clamp_min(1e-6)
        d_scale = (d_true.detach() - d_center).abs().median().clamp_min(1e-6)
        p_pred_loss = p_pred / p_scale
        p_true_loss = p_true / p_scale
        d_pred_loss = d_pred / d_scale
        d_true_loss = d_true / d_scale
    else:
        p_pred_loss, p_true_loss = p_pred, p_true
        d_pred_loss, d_true_loss = d_pred, d_true

    loss_p = _pointwise_forecast_loss(p_pred_loss, p_true_loss, loss_type, huber_delta)
    loss_d = _pointwise_forecast_loss(d_pred_loss, d_true_loss, loss_type, huber_delta)

    if task_target == "pressure":
        loss = loss_p
    elif task_target == "demand":
        loss = loss_d
    elif combine == "weighted_mean":
        loss = (wp * loss_p + wd * loss_d) / max(1e-12, wp + wd)
    else:
        loss = wp * loss_p + wd * loss_d

    return loss, float(raw_loss_p.detach().item()), float(raw_loss_d.detach().item())




def _build_knn_candidate_mask(node_xy, k):
    """
    node_xy: [N, 2]
    returns:
        candidate_mask: [N, N] bool
    Undirected/symmetrized kNN graph.
    """
    N = node_xy.shape[0]

    # pairwise Euclidean distances
    dist = torch.cdist(node_xy, node_xy, p=2)   # [N, N]

    # exclude self from nearest neighbors
    eye = torch.eye(N, device=node_xy.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    k = min(int(k), N - 1)
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices   # [N, k]

    candidate_mask = torch.zeros((N, N), device=node_xy.device, dtype=torch.bool)
    row_idx = torch.arange(N, device=node_xy.device).unsqueeze(1).expand_as(nn_idx)
    candidate_mask[row_idx, nn_idx] = True

    # symmetrize
    candidate_mask = candidate_mask | candidate_mask.t()

    # no self-loops
    candidate_mask.fill_diagonal_(False)
    return candidate_mask


def _compute_knn_candidate_loss(model, batch, device, hparams):
    """
    Penalizes learned edges that fall outside the geometric kNN candidate graph.

    Returns:
        loss_knn: scalar tensor
        inside_ratio: float in [0, 1]
    """
    lam_knn = _get_float_attr(hparams, "lambda_knn", 0.0)
    if lam_knn <= 0.0:
        return torch.tensor(0.0, device=device), 0.0

    if not hasattr(model, "edges") or model.edges is None:
        return torch.tensor(0.0, device=device), 0.0

    if not hasattr(batch, "node_xy") or batch.node_xy is None:
        return torch.tensor(0.0, device=device), 0.0

    edges = model.edges
    if edges.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0

    node_xy = batch.node_xy.to(device)

    # batch.node_xy can be [B, N, 2] or [N, 2]
    if node_xy.dim() == 3:
        node_xy_ref = node_xy[0]       # [N, 2]
    elif node_xy.dim() == 2:
        N = int(hparams.num_nodes)
        if node_xy.shape[0] == N:
            node_xy_ref = node_xy
        else:
            node_xy_ref = node_xy[:N]
    else:
        raise ValueError(f"Unexpected node_xy shape: {tuple(node_xy.shape)}")

    N = int(hparams.num_nodes)
    k = int(getattr(hparams, "k_geo_candidates", 6))

    candidate_mask = _build_knn_candidate_mask(node_xy_ref, k=k)   # [N, N]

    src = edges[0] % N
    dst = edges[1] % N

    valid = src != dst
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device), 0.0

    src = src[valid]
    dst = dst[valid]

    inside = candidate_mask[src, dst]          # [E]
    inside_ratio = float(inside.float().mean().detach().item())

    # loss = fraction of learned edges outside candidate set
    loss_knn = (~inside).float().mean()

    return loss_knn, inside_ratio





def _compute_degree_penalty(model, hparams, device):
    """
    Penalizes nodes whose degree exceeds a target maximum.
    """
    lam_deg = _get_float_attr(hparams, "lambda_degree", 0.0)
    if lam_deg <= 0.0:
        return torch.tensor(0.0, device=device), 0.0

    if not hasattr(model, "edges") or model.edges is None:
        return torch.tensor(0.0, device=device), 0.0

    edges = model.edges
    if edges.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0

    N = int(hparams.num_nodes)
    dmax = float(getattr(hparams, "max_degree_target", 4))

    src = edges[0] % N
    dst = edges[1] % N

    deg = torch.bincount(src, minlength=N).float() + torch.bincount(dst, minlength=N).float()
    excess = torch.clamp(deg - dmax, min=0.0)

    loss_deg = (excess ** 2).mean()
    deg_max = float(deg.max().detach().item())

    return loss_deg, deg_max







def train_one_epoch(model, dataloader, optimizer, device, hparams):
    model.train()

    all_y_true = []
    all_y_pred = []

    running_mse = 0.0
    running_graph_loss = 0.0
    running_stability = 0.0
    running_topology = 0.0
    running_soft_pseudo_edge = 0.0
    running_total_loss = 0.0
    running_weighted_forecast = 0.0
    running_weighted_graph = 0.0
    running_weighted_topology = 0.0
    running_weighted_stability = 0.0
    running_pressure_mse = 0.0
    running_demand_mse = 0.0
    n_batches = 0

    lam_graph = _get_float_attr(hparams, "lambda_graph", 0.0)
    lam_stability = _get_float_attr(hparams, "lambda_stability", 0.0)
    alpha_forecast = _get_float_attr(
        hparams,
        "current_alpha_forecast",
        _get_float_attr(hparams, "alpha_forecast", 1.0),
    )
    use_complementary_loss = bool(getattr(hparams, "use_complementary_loss", False))

    for batch in dataloader:
        optimizer.zero_grad()

        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

        if _is_water_like_dataset(hparams):
            x, y, batch = _reshape_ctown_epanet_batch(batch, hparams, device)
            x, y = _flatten_for_dgm(x, y)

            y_predicted, logprobs = model(x, edge_index, edge_weight)

            loss_forecasting, loss_pressure, loss_demand = _forecasting_loss_ctown(
                y_predicted, y, hparams
            )
        else:
            x = batch.x.to(device)
            y = batch.y[:, : hparams.prediction_window].to(device)

            if hasattr(hparams, "dataset_name") and hparams.dataset_name in ["METR-LA", "solar", "electricity"]:
                if len(x.shape) == 4 and rearrange is not None:
                    x = rearrange(x, "b t n f -> (b n) (t f)")
                    y = rearrange(y, "b t n f -> (b n) (t f)")

            y_predicted, logprobs = model(x, edge_index, edge_weight)
            loss_forecasting = F.mse_loss(y_predicted, y)
            loss_pressure = float(loss_forecasting.detach().item())
            loss_demand = 0.0

        loss_stability = _compute_dgm_stability_loss(model, device)
        loss_soft_topology, topology_debug = _compute_soft_topology_loss(
            model, batch, device, hparams, x_flat=x
        )

        graph_loss = torch.tensor(0.0, device=device)
        graph_debug = {}

        if lam_graph > 0.0 and logprobs is not None:
            graph_loss, graph_debug = _task_driven_graph_loss(
                model=model,
                y_predicted=y_predicted,
                y_true=y,
                logprobs=logprobs,
                hparams=hparams,
            )

        topology_loss = loss_soft_topology
        stability_term = lam_stability * loss_stability

        if use_complementary_loss:
            alpha = max(0.0, min(1.0, alpha_forecast))
            weighted_forecast = alpha * loss_forecasting
            weighted_graph = alpha * lam_graph * graph_loss
            weighted_topology = (1.0 - alpha) * topology_loss
        else:
            weighted_forecast = loss_forecasting
            weighted_graph = lam_graph * graph_loss
            weighted_topology = topology_loss
        loss_total = weighted_forecast + weighted_graph + weighted_topology + stability_term

        loss_total.backward()
        optimizer.step()

        graph_loss_value = float(graph_loss.detach().item())

        running_mse += float(loss_forecasting.detach().item())
        running_pressure_mse += loss_pressure
        running_demand_mse += loss_demand
        running_graph_loss += graph_loss_value
        running_stability += float(loss_stability.detach().item())
        running_topology += float(topology_loss.detach().item())
        running_soft_pseudo_edge += topology_debug["soft_pseudo_edge"]
        running_total_loss += float(loss_total.detach().item())
        running_weighted_forecast += float(weighted_forecast.detach().item())
        running_weighted_graph += float(weighted_graph.detach().item())
        running_weighted_topology += float(weighted_topology.detach().item())
        running_weighted_stability += float(stability_term.detach().item())
        n_batches += 1

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_predicted.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    train_mae = MAE(all_y_pred, all_y_true)
    train_rmse = root_mean_squared_error(all_y_pred, all_y_true)
    train_mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    train_mse = running_mse / max(1, n_batches)
    avg_graph_loss = running_graph_loss / max(1, n_batches)
    avg_stability = running_stability / max(1, n_batches)
    avg_topology = running_topology / max(1, n_batches)
    avg_soft_pseudo_edge = running_soft_pseudo_edge / max(1, n_batches)
    avg_pressure_mse = running_pressure_mse / max(1, n_batches)
    avg_demand_mse = running_demand_mse / max(1, n_batches)

    stats = {
        "mse": train_mse,
        "rmse": train_rmse,
        "mae": train_mae,
        "mape": train_mape,
        "graph_loss": avg_graph_loss,
        "stability_loss": avg_stability,
        "topology_loss": avg_topology,
        "soft_pseudo_edge_loss": avg_soft_pseudo_edge,
        "pressure_mse": avg_pressure_mse,
        "demand_mse": avg_demand_mse,
        "alpha_forecast": alpha_forecast,
        "total_loss": running_total_loss / max(1, n_batches),
        "weighted_forecast_term": running_weighted_forecast / max(1, n_batches),
        "weighted_graph_term": running_weighted_graph / max(1, n_batches),
        "weighted_topology_term": running_weighted_topology / max(1, n_batches),
        "weighted_stability_term": running_weighted_stability / max(1, n_batches),
    }
    return stats


@torch.no_grad()
def evaluate(model, dataloader, device, hparams, split="val"):
    model.eval()

    all_y_true = []
    all_y_pred = []

    running_mse = 0.0
    running_pressure_mse = 0.0
    running_demand_mse = 0.0
    running_stability = 0.0
    running_topology = 0.0
    running_soft_pseudo_edge = 0.0
    n_batches = 0
    lam_stability = _get_float_attr(hparams, "lambda_stability", 0.0)

    for batch in dataloader:
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

        if _is_water_like_dataset(hparams):
            x, y, batch = _reshape_ctown_epanet_batch(batch, hparams, device)
            x, y = _flatten_for_dgm(x, y)

            y_predicted, _ = model(x, edge_index, edge_weight)

            loss_forecasting, loss_pressure, loss_demand = _forecasting_loss_ctown(
                y_predicted, y, hparams
            )
        else:
            x = batch.x.to(device)
            y = batch.y[:, : hparams.prediction_window].to(device)

            if hasattr(hparams, "dataset_name") and hparams.dataset_name in ["METR-LA", "solar", "electricity"]:
                if len(x.shape) == 4 and rearrange is not None:
                    x = rearrange(x, "b t n f -> (b n) (t f)")
                    y = rearrange(y, "b t n f -> (b n) (t f)")

            y_predicted, _ = model(x, edge_index, edge_weight)
            loss_forecasting = F.mse_loss(y_predicted, y)
            loss_pressure = float(loss_forecasting.detach().item())
            loss_demand = 0.0

        loss_stability = _compute_dgm_stability_loss(model, device)
        loss_soft_topology, topology_debug = _compute_soft_topology_loss(
            model, batch, device, hparams, x_flat=x
        )
        topology_loss = loss_soft_topology

        running_mse += float(loss_forecasting.detach().item())
        running_pressure_mse += loss_pressure
        running_demand_mse += loss_demand
        running_stability += float(loss_stability.detach().item())
        running_topology += float(topology_loss.detach().item())
        running_soft_pseudo_edge += topology_debug["soft_pseudo_edge"]
        n_batches += 1

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_predicted.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    mae = MAE(all_y_pred, all_y_true)
    rmse = root_mean_squared_error(all_y_pred, all_y_true)
    mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    mse = running_mse / max(1, n_batches)
    pressure_mse = running_pressure_mse / max(1, n_batches)
    demand_mse = running_demand_mse / max(1, n_batches)
    stability = running_stability / max(1, n_batches)
    topology = running_topology / max(1, n_batches)
    soft_pseudo_edge = running_soft_pseudo_edge / max(1, n_batches)

    stats = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "pressure_mse": pressure_mse,
        "demand_mse": demand_mse,
        "stability_loss": stability,
        "topology_loss": topology,
        "soft_pseudo_edge_loss": soft_pseudo_edge,
    }
    return stats


def _export_learned_graph_artifacts(learned_edges_pt: str, meta_json: str, data_npz: str, out_dir: str = "."):
    """
    Exports:
      - learned_edges.csv
      - learned_graph_stats.json
    Requires:
      - learned_edges_pt: torch saved edges [2, E_global]
      - meta_json: contains node_ids
      - data_npz: contains node_xy and optionally edge_index (GT)
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = torch.load(learned_edges_pt, map_location="cpu").numpy()
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    npz = np.load(data_npz)

    node_ids = meta["node_ids"]
    N = len(node_ids)

    node_xy = npz["node_xy"]
    if node_xy.ndim == 3:
        node_xy0 = node_xy[0]
    else:
        node_xy0 = node_xy

    src_g = edges[0].astype(int)
    dst_g = edges[1].astype(int)

    src = src_g % N
    dst = dst_g % N

    E = len(src)
    deg_out = np.bincount(src, minlength=N)
    deg_in = np.bincount(dst, minlength=N)
    density = float(E / (N * (N - 1) + 1e-12))

    S_dir = set(zip(src.tolist(), dst.tolist()))
    bidir = sum((v, u) in S_dir for (u, v) in S_dir)
    bidir_ratio = float(bidir / (E + 1e-12))

    mask = np.isfinite(node_xy0[:, 0]) & np.isfinite(node_xy0[:, 1])
    ok = mask[src] & mask[dst]
    if np.any(ok):
        dxy = node_xy0[src[ok]] - node_xy0[dst[ok]]
        dist = np.sqrt((dxy ** 2).sum(1))
        dist_mean = float(dist.mean())
        dist_median = float(np.median(dist))
        dist_max = float(dist.max())
    else:
        dist_mean = None
        dist_median = None
        dist_max = None

    stats = {
        "N": int(N),
        "E": int(E),
        "density_directed": density,
        "mean_out_degree": float(deg_out.mean()),
        "min_out_degree": int(deg_out.min()),
        "max_out_degree": int(deg_out.max()),
        "mean_in_degree": float(deg_in.mean()),
        "min_in_degree": int(deg_in.min()),
        "max_in_degree": int(deg_in.max()),
        "edge_length_mean": dist_mean,
        "edge_length_median": dist_median,
        "edge_length_max": dist_max,
        "bidirectional_ratio": bidir_ratio,
    }

    # Optional GT comparison if present
    if "edge_index" in npz.files:
        gt = npz["edge_index"]
        gt_set = set(zip(gt[0].astype(int).tolist(), gt[1].astype(int).tolist()))
        lr_set = set(zip(src.tolist(), dst.tolist()))
        tp = len(lr_set & gt_set)
        fp = len(lr_set - gt_set)
        fn = len(gt_set - lr_set)
        prec = float(tp / (tp + fp + 1e-12))
        rec = float(tp / (tp + fn + 1e-12))
        f1 = float(2 * prec * rec / (prec + rec + 1e-12))
        stats["edge_recovery_directed"] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": prec, "recall": rec, "f1": f1}

    # CSV
    csv_path = out_dir / "learned_edges.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("src_id,dst_id,src_idx,dst_idx,dist\n")
        for u, v in zip(src, dst):
            d = ""
            if np.isfinite(node_xy0[u]).all() and np.isfinite(node_xy0[v]).all():
                d = float(np.linalg.norm(node_xy0[u] - node_xy0[v]))
            f.write(f"{node_ids[u]},{node_ids[v]},{u},{v},{d}\n")

    stats_path = out_dir / "learned_graph_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[INFO] Exported: {csv_path}")
    print(f"[INFO] Exported: {stats_path}")


def _diagnostic_current_graph_f1(model, hparams):
    """Post-hoc current-batch graph F1; never used by optimization or selection."""
    if not bool(getattr(hparams, "diagnostic_graph_metrics", False)):
        return None
    if not hasattr(model, "edges") or model.edges is None:
        return None
    data = np.load(str(hparams.data_npz))
    n = int(hparams.num_nodes)
    reference = {(min(int(u), int(v)), max(int(u), int(v)))
                 for u, v in data["edge_index"].T if int(u) != int(v)}
    value = model.edges.detach().cpu().numpy()
    predicted = {(min(int(u) % n, int(v) % n), max(int(u) % n, int(v) % n))
                 for u, v in value.T if int(u) % n != int(v) % n}
    tp = len(reference & predicted); fp = len(predicted - reference); fn = len(reference - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def train_and_test_dgm(model, train_loader, val_loader, test_loader, hparams):
    """
    Train + val (scheduler) + test.

    Saves:
      - dgm_learned_edges.pt                -> BEST validation edges (main file used downstream)
      - dgm_learned_edges_best.pt           -> copy of best validation edges
      - dgm_learned_edges_last.pt           -> last epoch edges
      - trained model state_dict            -> last epoch model
      - trained best model state_dict       -> best validation model
      - optional learned_edges.csv + learned_graph_stats.json
      - optional loss curve
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.8,
    )

    n_epochs = hparams.max_epochs
    selection_metric = getattr(hparams, "select_best_by", "val_mse")
    train_stats = None
    val_stats = None

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "alpha_forecast": [],
    }

    alpha_start = float(getattr(hparams, "alpha_forecast_start", hparams.alpha_forecast))
    alpha_end = float(getattr(hparams, "alpha_forecast_end", hparams.alpha_forecast))
    alpha_schedule_epochs = int(getattr(hparams, "alpha_schedule_epochs", 0))
    alpha_hold_epochs = int(getattr(hparams, "alpha_hold_epochs", 0))

    def scheduled_alpha(epoch):
        if epoch <= alpha_hold_epochs:
            return alpha_start
        if alpha_schedule_epochs <= 1:
            return alpha_end
        ramp_epoch = epoch - alpha_hold_epochs
        progress = min(1.0, max(0.0, (ramp_epoch - 1) / float(alpha_schedule_epochs - 1)))
        return alpha_start + progress * (alpha_end - alpha_start)

    def observable_validation_score(stats):
        if stats is None:
            return float("inf")
        alpha = alpha_end
        mse = float(stats.get("mse", 0.0))
        topology = float(
            stats.get(
                "topology_loss",
                stats.get("soft_pseudo_edge_loss", 0.0),
            )
        )
        stability = float(stats.get("stability_loss", 0.0))
        lam_stability = _get_float_attr(hparams, "lambda_stability", 0.0)
        return alpha * mse + (1.0 - alpha) * topology + lam_stability * stability

    # --- keep best validation snapshots in memory ---
    best_state_dict = None
    best_edges = None
    best_epoch = None
    best_val_stats = None
    best_observable_state_dict = None
    best_observable_edges = None
    best_observable_epoch = None
    best_observable_stats = None
    best_observable_score = float("inf")
    best_proxy_state_dict = None
    best_proxy_edges = None
    best_proxy_epoch = None
    best_proxy_stats = None
    best_proxy_score = float("inf")

    for epoch in range(1, n_epochs + 1):
        hparams.current_alpha_forecast = scheduled_alpha(epoch)
        print(
            f"\nEpoch {epoch}/{n_epochs} | "
            f"alpha_forecast={hparams.current_alpha_forecast:.4f} | "
            f"topology_weight={1.0 - hparams.current_alpha_forecast:.4f}"
        )

        train_stats = train_one_epoch(model, train_loader, optimizer, device, hparams)
        diagnostic_f1 = _diagnostic_current_graph_f1(model, hparams)
        print(
            f"Train - MSE: {train_stats['mse']:.4f} | RMSE: {train_stats['rmse']:.4f} "
            f"| MAE: {train_stats['mae']:.4f} | MAPE: {train_stats['mape']:.4f} "
            f"| P-MSE: {train_stats.get('pressure_mse', float('nan')):.4f} "
            f"| D-MSE: {train_stats.get('demand_mse', float('nan')):.4f} "
            f"| GraphLoss: {train_stats['graph_loss']:.4f} "
            f"| Stability: {train_stats.get('stability_loss', 0.0):.4f} "
            f"| L_Q: {train_stats.get('soft_pseudo_edge_loss', 0.0):.4f} "
            f"| L_topology: {train_stats.get('topology_loss', 0.0):.4f} "
            f"| Total: {train_stats.get('total_loss', 0.0):.8f} "
            f"| WTask: {train_stats.get('weighted_forecast_term', 0.0):.8f} "
            f"| WGraph: {train_stats.get('weighted_graph_term', 0.0):.8f} "
            f"| WTopo: {train_stats.get('weighted_topology_term', 0.0):.8f} "
            f"| WStab: {train_stats.get('weighted_stability_term', 0.0):.8f}"
        )

        if diagnostic_f1 is not None:
            diagnostic_task_score = 1.0 / (1.0 + max(0.0, float(train_stats["mse"])))
            diagnostic_eval0p5 = 0.5 * diagnostic_task_score + 0.5 * diagnostic_f1
            print(
                f"DiagnosticGraph - F1: {diagnostic_f1:.8f} | "
                f"TaskScore: {diagnostic_task_score:.8f} | Eval0.5: {diagnostic_eval0p5:.8f}"
            )

        history["epoch"].append(epoch)
        history["train_mse"].append(train_stats["mse"])
        history["alpha_forecast"].append(hparams.current_alpha_forecast)

        if epoch % hparams.check_val_every_n_epoch == 0 and val_loader is not None:
            val_stats = evaluate(model, val_loader, device, hparams, split="val")
            print(
                f"Val   - MSE: {val_stats['mse']:.4f} | RMSE: {val_stats['rmse']:.4f} "
                f"| MAE: {val_stats['mae']:.4f} | MAPE: {val_stats['mape']:.4f} "
                f"| P-MSE: {val_stats.get('pressure_mse', float('nan')):.4f} "
                f"| D-MSE: {val_stats.get('demand_mse', float('nan')):.4f} "
                f"| L_Q: {val_stats.get('soft_pseudo_edge_loss', 0.0):.4f} "
                f"| L_topology: {val_stats.get('topology_loss', 0.0):.4f}"
            )

            observable_score = observable_validation_score(val_stats)
            print(f"ValObjective - J_obs: {observable_score:.4f}")
            if observable_score < best_observable_score:
                best_observable_score = observable_score
                best_observable_epoch = epoch
                best_observable_stats = dict(val_stats)
                best_observable_stats["observable_val_score"] = observable_score
                best_observable_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                if hasattr(model, "edges") and model.edges is not None:
                    best_observable_edges = model.edges.detach().cpu().clone()
                print(
                    f"[INFO] New best observable-validation epoch: {best_observable_epoch} | "
                    f"J_obs={best_observable_score:.6f}"
                )

            proxy_stats = None
            if selection_metric == "topology_proxy":
                proxy_stats = _compute_topology_proxy_score(model, val_loader, device, hparams)
            if proxy_stats is not None:
                print(
                    "TopoProxy - "
                    f"Score: {proxy_stats['score']:.4f} | "
                    f"Comp: {proxy_stats['components']} | "
                    f"LenMean: {proxy_stats['edge_length_mean']:.4f} | "
                    f"LenTarget: {proxy_stats['edge_length_target']:.4f} | "
                    f"Shortfall: {proxy_stats['edge_length_shortfall']:.4f} | "
                    f"FreqCV: {proxy_stats['frequency_cv']:.4f}"
                )
                if proxy_stats["score"] < best_proxy_score:
                    best_proxy_score = proxy_stats["score"]
                    best_proxy_epoch = epoch
                    best_proxy_stats = dict(proxy_stats)
                    best_proxy_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    if hasattr(model, "edges") and model.edges is not None:
                        best_proxy_edges = model.edges.detach().cpu().clone()
                    print(
                        f"[INFO] New best topology-proxy epoch: {best_proxy_epoch} | "
                        f"proxy_score={best_proxy_score:.6f}"
                    )

            # Update best validation snapshot
            if val_stats["mse"] < model.best_mse:
                model.best_mse = val_stats["mse"]
                model.best_rmse = val_stats["rmse"]
                model.best_mae = val_stats["mae"]
                model.best_mape = val_stats["mape"]

                best_epoch = epoch
                best_val_stats = dict(val_stats)

                # Save best model weights in memory
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                # Save best learned edges in memory
                if hasattr(model, "edges") and model.edges is not None:
                    best_edges = model.edges.detach().cpu().clone()

                print(
                    f"[INFO] New best validation epoch: {best_epoch} | "
                    f"val_mse={model.best_mse:.6f}"
                )

            scheduler.step(val_stats["mse"])
            history["val_mse"].append(val_stats["mse"])
        else:
            history["val_mse"].append(np.nan)

    export_dir = getattr(hparams, "export_dir", ".")
    os.makedirs(export_dir, exist_ok=True)

    learned_edges_path = os.path.join(export_dir, "dgm_learned_edges.pt")
    learned_edges_last_path = os.path.join(export_dir, "dgm_learned_edges_last.pt")
    learned_edges_best_path = os.path.join(export_dir, "dgm_learned_edges_best.pt")
    learned_edges_observable_path = os.path.join(export_dir, "dgm_learned_edges_best_observable_val.pt")
    learned_edges_proxy_path = os.path.join(export_dir, "dgm_learned_edges_best_topology_proxy.pt")

    # Save last-epoch edges separately
    learned_edges_last = None
    if hasattr(model, "edges") and model.edges is not None:
        learned_edges_last = model.edges.detach().cpu().clone()
        torch.save(learned_edges_last, learned_edges_last_path)
        print(f"\n[INFO] Saved last-epoch learned graph edges to: {learned_edges_last_path}")
    else:
        print("\n[WARN] model.edges non presente; no last-epoch graph saved.")

    # Save BEST-validation edges as the main graph file used downstream
    if best_edges is not None:
        torch.save(best_edges, learned_edges_path)
        torch.save(best_edges, learned_edges_best_path)
        print(f"[INFO] Saved BEST-validation learned graph edges to: {learned_edges_path}")
        print(f"[INFO] Saved BEST-validation learned graph edges copy to: {learned_edges_best_path}")
    elif learned_edges_last is not None:
        torch.save(learned_edges_last, learned_edges_path)
        print(f"[WARN] No best validation edges found; using last epoch edges for: {learned_edges_path}")
    else:
        print("\n[WARN] No learned graph edges available to save.")

    if best_proxy_edges is not None:
        torch.save(best_proxy_edges, learned_edges_proxy_path)
        print(f"[INFO] Saved BEST-topology-proxy learned graph edges to: {learned_edges_proxy_path}")

    selected_state_dict = best_state_dict
    selected_edges = best_edges
    selected_epoch = best_epoch
    selected_name = "BEST-validation"
    if selection_metric == "observable_val" and best_observable_state_dict is not None:
        selected_state_dict = best_observable_state_dict
        selected_edges = best_observable_edges
        selected_epoch = best_observable_epoch
        selected_name = "BEST-observable-validation"
    elif selection_metric == "topology_proxy" and best_proxy_state_dict is not None:
        selected_state_dict = best_proxy_state_dict
        selected_edges = best_proxy_edges
        selected_epoch = best_proxy_epoch
        selected_name = "BEST-topology-proxy"

    if best_observable_edges is not None:
        torch.save(best_observable_edges, learned_edges_observable_path)
        print(f"[INFO] Saved BEST-observable-validation learned graph edges to: {learned_edges_observable_path}")

    if selected_edges is not None:
        torch.save(selected_edges, learned_edges_path)
        print(f"[INFO] Saved selected learned graph edges to main file: {learned_edges_path}")

    # Restore selected model before test/export, if available
    if selected_state_dict is not None:
        model.load_state_dict(selected_state_dict, strict=True)
        model.to(device)
        print(f"[INFO] Restored {selected_name} model from epoch {selected_epoch} before test.")
    else:
        print("[WARN] No selected model snapshot found; test will use last epoch model.")

    # Test
    test_stats = None
    if test_loader is not None:
        test_stats = evaluate(model, test_loader, device, hparams, split="test")
        print(
            f"\nTEST - MSE: {test_stats['mse']:.4f} | RMSE: {test_stats['rmse']:.4f} "
            f"| MAE: {test_stats['mae']:.4f} | MAPE: {test_stats['mape']:.4f} "
            f"| P-MSE: {test_stats.get('pressure_mse', float('nan')):.4f} "
            f"| D-MSE: {test_stats.get('demand_mse', float('nan')):.4f}"
        )

    # Save model checkpoints
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = getattr(hparams, "dataset_name", "unknown")
    run_name = getattr(hparams, "run_name", None)

    if run_name is None:
        ckpt_path = os.path.join(save_dir, f"dgm_{dataset_name}.pt")
    else:
        ckpt_path = os.path.join(save_dir, f"dgm_{dataset_name}_{run_name}.pt")

    # Save current in-memory model after optional best restore
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[INFO] Saved model checkpoint to: {ckpt_path}")

    if best_state_dict is not None:
        best_ckpt_path = ckpt_path.replace(".pt", "_best.pt")
        torch.save(best_state_dict, best_ckpt_path)
        print(f"[INFO] Saved BEST-validation model to: {best_ckpt_path}")

    if best_observable_state_dict is not None:
        observable_ckpt_path = ckpt_path.replace(".pt", "_best_observable_val.pt")
        torch.save(best_observable_state_dict, observable_ckpt_path)
        print(f"[INFO] Saved BEST-observable-validation model to: {observable_ckpt_path}")

    if best_proxy_state_dict is not None:
        proxy_ckpt_path = ckpt_path.replace(".pt", "_best_topology_proxy.pt")
        torch.save(best_proxy_state_dict, proxy_ckpt_path)
        print(f"[INFO] Saved BEST-topology-proxy model to: {proxy_ckpt_path}")

    # Optional export of printable graph + stats using the main learned_edges_path
    meta_json = getattr(hparams, "meta_json", None)
    data_npz = getattr(hparams, "data_npz", None)

    if meta_json is not None and data_npz is not None and Path(learned_edges_path).exists():
        try:
            _export_learned_graph_artifacts(
                learned_edges_pt=str(learned_edges_path),
                meta_json=str(meta_json),
                data_npz=str(data_npz),
                out_dir=str(export_dir),
            )
        except Exception as e:
            print(f"[WARN] Export learned graph artifacts failed: {e}")

    # Optional loss plot
    loss_plot_path = getattr(hparams, "loss_plot_path", None)
    if loss_plot_path is not None:
        try:
            _save_loss_curve(history, loss_plot_path)
            print(f"[INFO] Saved loss curve to: {loss_plot_path}")
        except Exception as e:
            print(f"[WARN] Could not save loss curve: {e}")

    if best_epoch is not None:
        print(f"[INFO] Best validation epoch: {best_epoch}")
        print(f"[INFO] Best validation stats: {best_val_stats}")
    if best_observable_epoch is not None:
        print(f"[INFO] Best observable-validation epoch: {best_observable_epoch}")
        print(f"[INFO] Best observable-validation stats: {best_observable_stats}")
    if best_proxy_epoch is not None:
        print(f"[INFO] Best topology-proxy epoch: {best_proxy_epoch}")
        print(f"[INFO] Best topology-proxy stats: {best_proxy_stats}")

    forecast_metrics_path = os.path.join(export_dir, "forecast_metrics.json")
    forecast_metrics = {
        "selected_model": selected_name,
        "selected_epoch": selected_epoch,
        "best_validation_epoch": best_epoch,
        "best_validation": best_val_stats,
        "best_observable_validation_epoch": best_observable_epoch,
        "best_observable_validation": best_observable_stats,
        "best_topology_proxy_epoch": best_proxy_epoch,
        "best_topology_proxy": best_proxy_stats,
        "last_train": train_stats,
        "last_validation": val_stats,
        "test": test_stats,
    }
    try:
        with open(forecast_metrics_path, "w", encoding="utf-8") as f:
            json.dump(forecast_metrics, f, indent=2)
        print(f"[INFO] Saved forecast metrics to: {forecast_metrics_path}")
    except Exception as e:
        print(f"[WARN] Could not save forecast metrics: {e}")

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_stats": best_val_stats,
        "best_observable_epoch": best_observable_epoch,
        "best_observable_stats": best_observable_stats,
    }




















def _is_spatiotemporal_batch(x):
    return x.dim() == 4


def _flatten_batch_xy_for_dgm(x, y, hparams):
    """
    Converts:
        x: [B, L, N, F] -> [(B*N), (L*F)]
        y: [B, H, N, F] -> [(B*N), (H*F)]
    """
    if rearrange is None:
        raise RuntimeError("einops.rearrange is required for spatiotemporal datasets.")

    x = rearrange(x, "b t n f -> (b n) (t f)")
    y = rearrange(y, "b t n f -> (b n) (t f)")
    return x, y


def _forecasting_loss_multifeature(y_predicted, y_true, hparams):
    """
    Weighted MSE for multi-feature forecasting.

    Assumes flattened node-wise outputs:
        y_predicted: [(B*N), (H*F)]
        y_true:      [(B*N), (H*F)]

    For ctown_epanet:
        F = 2
        feature 0 = pressure
        feature 1 = demand
    """
    dataset_name = getattr(hparams, "dataset_name", "")

    if dataset_name not in WATER_LIKE_DATASETS:
        loss = F.mse_loss(y_predicted, y_true)
        return loss, {
            "loss_total": float(loss.detach().item()),
            "loss_pressure": float(loss.detach().item()),
            "loss_demand": 0.0,
        }

    out_features = int(getattr(hparams, "output_features", 2))
    horizon = int(getattr(hparams, "prediction_window", 1))

    if y_predicted.shape[-1] != horizon * out_features:
        raise ValueError(
            f"Output dim mismatch: got {y_predicted.shape[-1]}, "
            f"expected {horizon * out_features} = prediction_window * output_features"
        )

    y_pred_4d = y_predicted.view(-1, horizon, out_features)
    y_true_4d = y_true.view(-1, horizon, out_features)

    y_pred_pressure = y_pred_4d[:, :, 0]
    y_true_pressure = y_true_4d[:, :, 0]

    y_pred_demand = y_pred_4d[:, :, 1]
    y_true_demand = y_true_4d[:, :, 1]

    pressure_weight = _get_float_attr(hparams, "pressure_weight", 1.0)
    demand_weight = _get_float_attr(hparams, "demand_weight", 1.0)

    loss_pressure = F.mse_loss(y_pred_pressure, y_true_pressure)
    loss_demand = F.mse_loss(y_pred_demand, y_true_demand)

    loss = pressure_weight * loss_pressure + demand_weight * loss_demand

    return loss, {
        "loss_total": float(loss.detach().item()),
        "loss_pressure": float(loss_pressure.detach().item()),
        "loss_demand": float(loss_demand.detach().item()),
    }
