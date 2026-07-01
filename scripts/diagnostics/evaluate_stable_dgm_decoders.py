"""Evaluate DGM-output-only stable consensus decoders.

The score uses only statistics produced downstream by the DGM probabilities:
mean(P_ij) and std(P_ij) across validation batches/layers. No geometry, no T_obs,
no pressure/demand features enter the consensus score. GT is loaded only for
post-hoc metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_train_synthetic import build_q_dynamic_candidate_mask, configure_run  # noqa: E402
from src.train_dgm import _flatten_for_dgm, _reshape_ctown_epanet_batch  # noqa: E402
from src.utils.utils import get_datamodule, get_model, set_params_wrt_dataset  # noqa: E402

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


def parse_module(run_name: str) -> tuple[str, str]:
    match = re.search(r"_huber_([^_]+)_e\d+$", run_name)
    if not match:
        return "gcn", "gcn"
    parts = match.group(1).split("-")
    return parts[0], parts[1]


def parse_epochs(run_name: str) -> int:
    match = re.search(r"_e(\d+)$", run_name)
    return int(match.group(1)) if match else 16


def make_args(run_dir: Path) -> SimpleNamespace:
    dataset = infer_dataset(run_dir.name)
    ffun, gfun = parse_module(run_dir.name)
    epochs = parse_epochs(run_dir.name)
    return SimpleNamespace(
        verbose=False,
        data_dir=DATASETS[dataset],
        out_dir=run_dir,
        run_name=run_dir.name,
        num_nodes=50,
        lags=6,
        prediction_window=12,
        stride=1,
        batch_size=16,
        max_epochs=epochs,
        check_val_every_n_epoch=4,
        lr=1e-3,
        k=2,
        ffun=ffun,
        gfun=gfun,
        lambda_graph=0.0,
        lambda_stability=0.0,
        use_complementary_loss=True,
        alpha_forecast=0.8,
        alpha_forecast_start=None,
        alpha_forecast_end=None,
        alpha_schedule_epochs=0,
        alpha_hold_epochs=0,
        task_loss_type="huber",
        task_huber_delta=1.0,
        task_target="all",
        task_feature_combine="weighted_mean",
        task_feature_normalization="target_std",
        pressure_weight=1.0,
        demand_weight=1.0,
        coordinate_embedding_dim=0,
        coordinate_embedding_trainable=True,
        lambda_soft_pseudo_edge=0.5,
        pseudo_geo_tau=1.0,
        pseudo_temporal_weight=0.5,
        pseudo_score_mode="observable_expert_full",
        pseudo_max_lag=6,
        pseudo_partial_ridge=0.05,
        pseudo_use_static_target=False,
        pseudo_bridge_topk=1,
        pseudo_bridge_mass=0.15,
        pseudo_bridge_quantile=0.65,
        pseudo_bridge_temperature=0.08,
        pseudo_bridge_min_temporal=0.55,
        pseudo_bridge_candidate_k=4,
        pseudo_candidate_k=6,
        pseudo_topk=2,
        pseudo_include_mst=True,
        pseudo_mst_bonus=1.0,
        select_best_by="observable_val",
        proxy_length_candidate_k=12,
        proxy_length_quantile=0.65,
        proxy_length_weight=1.0,
        proxy_connectivity_weight=1.0,
        proxy_frequency_weight=0.05,
        soft_degree_target=1.0,
        static_top_edges=0,
        static_row_k=3,
        static_symmetry="mutual",
        static_include_mst=True,
        ensure_connected_consensus=True,
        export_soft_graph=True,
        save_soft_score=True,
        assign_candidate_mask=True,
        candidate_mode="q_dynamic",
        candidate_k=3,
        candidate_signal_k=4,
        seed=42,
    )


def local_probability_blocks(logits: torch.Tensor, num_nodes: int) -> list[torch.Tensor]:
    if logits is None or logits.dim() != 2:
        return []
    n_total = logits.shape[0]
    if n_total % num_nodes != 0:
        return []
    blocks = []
    for batch_id in range(n_total // num_nodes):
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
        blocks.append(prob.detach().cpu().double())
    return blocks


def collect_probability_stats(run_dir: Path, force: bool) -> tuple[np.ndarray, np.ndarray]:
    mean_path = run_dir / "dgm_prob_mean.npy"
    std_path = run_dir / "dgm_prob_std.npy"
    if mean_path.exists() and std_path.exists() and not force:
        return np.load(mean_path), np.load(std_path)

    args = make_args(run_dir)
    params = configure_run(args)
    train_l, val_l, _test_l, params = get_datamodule(params)
    params = set_params_wrt_dataset(params, train_l)
    model = get_model(params)

    mask = build_q_dynamic_candidate_mask(
        data_npz=params.data_npz,
        pseudo_score_mode=params.pseudo_score_mode,
        q_k=args.candidate_k,
        partial_ridge=params.pseudo_partial_ridge,
        geo_tau=params.pseudo_geo_tau,
        include_mst=params.pseudo_include_mst,
    )
    for gf in model.graph_f:
        if hasattr(gf, "candidate_mask"):
            gf.candidate_mask = mask

    ckpt = PROJECT_ROOT / "models" / f"dgm_synthetic_water_{run_dir.name}_best_observable_val.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "models" / f"dgm_synthetic_water_{run_dir.name}.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n = int(params.num_nodes)
    sum_prob = torch.zeros((n, n), dtype=torch.float64)
    sum_sq = torch.zeros((n, n), dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for batch in val_l:
            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None
            x, y, batch = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            _ = model(x, edge_index, edge_weight)
            for gf in getattr(model, "graph_f", []):
                for prob in local_probability_blocks(getattr(gf, "last_logits", None), n):
                    sum_prob += prob
                    sum_sq += prob * prob
                    count += 1
    if count == 0:
        raise RuntimeError(f"No probability blocks collected for {run_dir}")
    mean = sum_prob / count
    var = torch.clamp(sum_sq / count - mean * mean, min=0.0)
    std = torch.sqrt(var)
    mean = 0.5 * (mean + mean.T)
    std = 0.5 * (std + std.T)
    mean.fill_diagonal_(0.0)
    std.fill_diagonal_(0.0)
    np.save(mean_path, mean.numpy())
    np.save(std_path, std.numpy())
    return mean.numpy(), std.numpy()


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


def recovery(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    jaccard = tp / max(1, tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard}


def edge_lengths(edges: set[tuple[int, int]], xy: np.ndarray) -> tuple[float, float, float]:
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    vals = [float(np.linalg.norm(xy[u] - xy[v])) for u, v in edges]
    if not vals:
        return 0.0, 0.0, 0.0
    arr = np.asarray(vals)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


def read_task_error(run_dir: Path) -> float:
    data = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    return float((data.get("test") or {}).get("mse", 0.0))


def evaluate_run(run_dir: Path, betas: list[float], gamma: float, force: bool) -> list[dict]:
    dataset = infer_dataset(run_dir.name)
    data = np.load(DATASETS[dataset] / "data.npz")
    xy = data["node_xy"]
    if xy.ndim == 3:
        xy = xy[0]
    num_nodes = int(xy.shape[0])
    gt = undirected_from_edge_index(data["edge_index"], num_nodes)
    mean, std = collect_probability_stats(run_dir, force=force)
    task_error = read_task_error(run_dir)
    task_score = 1.0 / (1.0 + max(0.0, task_error))
    rows = []
    for beta in betas:
        score = mean - beta * std
        score = 0.5 * (score + score.T)
        np.fill_diagonal(score, 0.0)
        for row_k in [2, 3]:
            for symmetry in ["mutual", "union"]:
                for include_mst in [False, True]:
                    name = f"stable_mean_minus_{beta:g}std_row{row_k}_{symmetry}_{'mst' if include_mst else 'nomst'}"
                    edges = decode(score, row_k=row_k, symmetry=symmetry, include_mst=include_mst, xy=xy)
                    rec = recovery(gt, edges)
                    eval_gamma = gamma * task_score + (1.0 - gamma) * float(rec["f1"])
                    length_mean, length_median, length_max = edge_lengths(edges, xy)
                    vals_mean = np.asarray([mean[u, v] for u, v in edges]) if edges else np.asarray([0.0])
                    vals_std = np.asarray([std[u, v] for u, v in edges]) if edges else np.asarray([0.0])
                    rows.append({
                        "run": run_dir.name,
                        "dataset": dataset,
                        "decoder": name,
                        "beta_std_penalty": beta,
                        "row_k": row_k,
                        "symmetry": symmetry,
                        "include_mst": int(include_mst),
                        "task_error": task_error,
                        "task_score": task_score,
                        "eval_gamma": eval_gamma,
                        "num_edges": len(edges),
                        "components": components(num_nodes, edges),
                        "selected_mean_prob": float(vals_mean.mean()),
                        "selected_std_prob": float(vals_std.mean()),
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
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    rows: list[dict] = []
    for run_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        if (run_dir / "forecast_metrics.json").exists():
            rows.extend(evaluate_run(run_dir, betas=args.betas, gamma=args.gamma, force=args.force_recompute))
    write_csv(args.out_csv, rows)
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.out_csv}")
    print("\nBest stable DGM-only decoder rows by Eval_gamma")
    print("| rank | dataset | beta | row | sym | mst | task | F1 | P | R | edges | comp | Eval |")
    print("| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(sorted(rows, key=lambda r: r["eval_gamma"], reverse=True)[: args.limit], start=1):
        print(
            f"| {i} | {row['dataset']} | {row['beta_std_penalty']:.2f} | {row['row_k']} | {row['symmetry']} | {row['include_mst']} | "
            f"{row['task_error']:.4f} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['num_edges']} | {row['components']} | {row['eval_gamma']:.3f} |"
        )


if __name__ == "__main__":
    main()
