"""Run the definitive Section VI water experiments on fixed topologies.

The three reference topologies are never regenerated. A separate tuning signal
seed selects DGM hyperparameters with the GT-free observable validation
objective; five held-out signal/model seeds are then evaluated post-hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostics.train_consensus_learner import (  # noqa: E402
    build_pair_dataset,
    decode,
    train_consensus,
)
from scripts.synthetic.generate_synthetic_water import (  # noqa: E402
    build_knn_edges,
    make_splits,
    simulate_timeseries,
    to_edge_index,
)


SCENARIOS = ("base", "bridges", "loopy")
ALPHAS = (0.7, 0.8)
LAMBDA_QS = (0.5, 1.0)
FINAL_SEEDS = (11, 22, 33, 44, 55)
TUNING_SEED = 7

DGM_CONFIG = {
    "lags": 6,
    "prediction_window": 12,
    "epochs": 16,
    "check_every": 4,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "dgm_k": 2,
    "candidate_k": 3,
    "ffun": "gcn",
    "gfun": "gcn",
    "prior": "observable_expert_full",
    "lambda_stability": 0.1,
    "task_loss": "huber",
    "task_huber_delta": 1.0,
    "task_feature_combine": "weighted_mean",
    "task_feature_normalization": "target_std",
    "pressure_weight": 1.0,
    "demand_weight": 1.0,
    "coordinate_embedding_dim": 4,
}

DECODER_CONFIG = {
    "epochs": 800,
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "hidden_dim": 16,
    "lambda_stability": 2.0,
    "lambda_degree": 0.02,
    "lambda_entropy": 0.01,
    "target_degree": 2.8,
    "row_k": 2,
    "symmetry": "mutual",
    "include_mst": True,
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical(int(u), int(v))
        for u, v in np.asarray(edge_index).T
        if int(u) != int(v)
    }


def prepare_dataset(topology_path: Path, out_dir: Path, seed: int) -> None:
    data_path = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if data_path.exists() and meta_path.exists():
        return

    topology = np.load(topology_path)
    xy = topology["node_xy"].astype(np.float32)
    gt = undirected_edges(topology["edge_index"])
    initial = build_knn_edges(xy, k=3)
    pressure, demand = simulate_timeseries(
        xy=xy,
        edges=gt,
        timesteps=720,
        rng=np.random.default_rng(seed),
        pressure_mode="smooth_laplacian",
        pressure_smooth_tau=1.8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if "block_labels" in topology.files:
        save_kwargs["block_labels"] = topology["block_labels"]
    np.savez_compressed(
        data_path,
        pressure=pressure,
        demand=demand,
        node_xy=xy,
        edge_index=to_edge_index(gt, bidirectional=True),
        initial_edge_index=to_edge_index(initial, bidirectional=True),
        time_s=np.arange(720, dtype=np.int64) * 3600,
        **save_kwargs,
    )
    meta = {
        "name": "section_vi_fixed_water",
        "topology_source": str(topology_path.relative_to(PROJECT_ROOT)),
        "topology_regenerated": False,
        "signal_seed": seed,
        "node_ids": [f"N{i:03d}" for i in range(xy.shape[0])],
        "num_nodes": int(xy.shape[0]),
        "num_gt_undirected_edges": len(gt),
        "num_initial_undirected_edges": len(initial),
        "splits": make_splits(720),
        "signal_generator": {
            "pressure_mode": "smooth_laplacian",
            "pressure_smooth_tau": 1.8,
            "timesteps": 720,
            "initial_knn": 3,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def run_name(scenario: str, seed: int, alpha: float, lambda_q: float) -> str:
    return f"water_section_vi_{scenario}_s{seed}_a{fmt(alpha)}_lq{fmt(lambda_q)}"


def observable_score(run_dir: Path) -> float:
    metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    val = metrics.get("best_observable_validation") or {}
    if "observable_val_score" not in val:
        raise RuntimeError(f"Missing observable validation score in {run_dir}")
    return float(val["observable_val_score"])


def run_dgm(
    data_dir: Path,
    out_dir: Path,
    scenario: str,
    seed: int,
    alpha: float,
    lambda_q: float,
) -> None:
    if (out_dir / "forecast_metrics.json").exists() and (out_dir / "dgm_prob_mean.npy").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DGM_CONFIG
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--run_name", run_name(scenario, seed, alpha, lambda_q),
        "--num_nodes", "50",
        "--lags", str(cfg["lags"]),
        "--prediction_window", str(cfg["prediction_window"]),
        "--max_epochs", str(cfg["epochs"]),
        "--check_val_every_n_epoch", str(cfg["check_every"]),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["learning_rate"]),
        "--k", str(cfg["dgm_k"]),
        "--ffun", str(cfg["ffun"]),
        "--gfun", str(cfg["gfun"]),
        "--lambda_graph", "0.0",
        "--use_complementary_loss",
        "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", str(lambda_q),
        "--lambda_stability", str(cfg["lambda_stability"]),
        "--pseudo_score_mode", str(cfg["prior"]),
        "--pseudo_topk", str(cfg["dgm_k"]),
        "--pseudo_include_mst",
        "--select_best_by", "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode", "q_dynamic",
        "--candidate_k", str(cfg["candidate_k"]),
        "--static_top_edges", "0",
        "--static_row_k", "2",
        "--static_symmetry", "mutual",
        "--static_include_mst",
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
        "--task_loss_type", str(cfg["task_loss"]),
        "--task_huber_delta", str(cfg["task_huber_delta"]),
        "--task_feature_combine", str(cfg["task_feature_combine"]),
        "--task_feature_normalization", str(cfg["task_feature_normalization"]),
        "--pressure_weight", str(cfg["pressure_weight"]),
        "--demand_weight", str(cfg["demand_weight"]),
        "--coordinate_embedding_dim", str(cfg["coordinate_embedding_dim"]),
        "--seed", str(seed),
    ]
    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        tail = (out_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(f"DGM run failed: {out_dir}\n" + "\n".join(tail))


def save_edge_index(edges: set[tuple[int, int]], path: Path) -> None:
    directed = [(u, v) for u, v in sorted(edges)] + [(v, u) for u, v in sorted(edges)]
    edge_index = (
        torch.tensor(directed, dtype=torch.long).T
        if directed
        else torch.empty((2, 0), dtype=torch.long)
    )
    torch.save(edge_index, path)


def graph_metrics(
    gt: set[tuple[int, int]],
    pred: set[tuple[int, int]],
    num_nodes: int,
) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    total = num_nodes * (num_nodes - 1) // 2
    tn = total - tp - fp - fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, total)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
    }


def evaluate_neural_decoder(
    run_dir: Path,
    data_dir: Path,
    scenario: str,
    seed: int,
    alpha: float,
    lambda_q: float,
) -> dict[str, float | int | str]:
    out_dir = run_dir / "neural_consensus"
    result_path = out_dir / "metrics.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_dir / "data.npz")
    xy = data["node_xy"].astype(np.float32)
    gt = undirected_edges(data["edge_index"])
    mean_p = np.load(run_dir / "dgm_prob_mean.npy")
    std_p = np.load(run_dir / "dgm_prob_std.npy")
    features, target, std_norm, pairs, _ = build_pair_dataset(mean_p, std_p)
    args = SimpleNamespace(seed=seed, **{k: v for k, v in DECODER_CONFIG.items() if k not in {"row_k", "symmetry", "include_mst"}})
    learned = train_consensus(features, target, std_norm, pairs, xy.shape[0], args)
    pred = decode(
        learned.score,
        row_k=int(DECODER_CONFIG["row_k"]),
        symmetry=str(DECODER_CONFIG["symmetry"]),
        include_mst=bool(DECODER_CONFIG["include_mst"]),
        xy=xy,
    )
    np.save(out_dir / "neural_consensus_score.npy", learned.score)
    edge_path = out_dir / "neural_consensus_edges.pt"
    save_edge_index(pred, edge_path)

    forecast = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    test = forecast.get("test") or {}
    mse = float(test["mse"])
    mae = float(test["mae"])
    recovery = graph_metrics(gt, pred, int(xy.shape[0]))
    task_score = 1.0 / (1.0 + max(0.0, mse))
    joint = 0.5 * task_score + 0.5 * float(recovery["f1"])
    result = {
        "scenario": scenario,
        "seed": seed,
        "alpha": alpha,
        "lambda_Q": lambda_q,
        "observable_validation_score": observable_score(run_dir),
        "mse": mse,
        "mae": mae,
        "task_score": task_score,
        "joint_eval_0p5": joint,
        "predicted_edges": len(pred),
        "gt_edges": len(gt),
        "decoder_train_loss": learned.loss,
        "decoder_recon_loss": learned.recon_loss,
        "decoder_stability_loss": learned.stability_loss,
        "decoder_degree_loss": learned.degree_loss,
        "decoder_entropy_loss": learned.entropy_loss,
        **recovery,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    comparison_dir = out_dir / "graph_comparison"
    cmd = [
        sys.executable,
        "scripts/water/compare_graphs_epanet.py",
        "--learned", str(edge_path),
        "--npz", str(data_dir / "data.npz"),
        "--meta", str(data_dir / "meta.json"),
        "--outdir", str(comparison_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = ("precision", "accuracy", "recall", "f1", "mse", "mae", "joint_eval_0p5")
    out = []
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        record: dict[str, float | int | str] = {
            "scenario": scenario,
            "n_seeds": len(selected),
            "alpha": selected[0]["alpha"],
            "lambda_Q": selected[0]["lambda_Q"],
        }
        for key in metrics:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1))
        out.append(record)
    return out


def copy_representative_figures(output_root: Path, selected: dict[str, tuple[float, float]]) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    representative_seed = FINAL_SEEDS[0]
    for scenario, (alpha, lambda_q) in selected.items():
        source = (
            output_root
            / "final_runs"
            / scenario
            / f"seed_{representative_seed}"
            / f"a{fmt(alpha)}_lq{fmt(lambda_q)}"
            / "neural_consensus"
            / "graph_comparison"
            / "graph_compare_undirected.png"
        )
        shutil.copy2(source, figure_dir / f"water_{scenario}_gt_vs_neural_seed_{representative_seed}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "water_topologies",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "water_final_experiments",
    )
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()
    topology_root = args.topology_root.resolve()
    output_root = args.output_root.resolve()
    data_root = output_root / "datasets"
    output_root.mkdir(parents=True, exist_ok=True)

    all_seeds = (TUNING_SEED,) + FINAL_SEEDS
    for scenario in SCENARIOS:
        topology_path = topology_root / f"water_topology_{scenario}.npz"
        for seed in all_seeds:
            prepare_dataset(topology_path, data_root / scenario / f"seed_{seed}", seed)

    selected: dict[str, tuple[float, float]] = {}
    grid_rows: list[dict] = []
    for scenario in SCENARIOS:
        for alpha in ALPHAS:
            for lambda_q in LAMBDA_QS:
                run_dir = output_root / "tuning_runs" / scenario / f"a{fmt(alpha)}_lq{fmt(lambda_q)}"
                if not args.collect_only:
                    run_dgm(
                        data_root / scenario / f"seed_{TUNING_SEED}",
                        run_dir,
                        scenario,
                        TUNING_SEED,
                        alpha,
                        lambda_q,
                    )
                grid_rows.append({
                    "scenario": scenario,
                    "tuning_seed": TUNING_SEED,
                    "alpha": alpha,
                    "lambda_Q": lambda_q,
                    "observable_validation_score": observable_score(run_dir),
                })
        best = min(
            (row for row in grid_rows if row["scenario"] == scenario),
            key=lambda row: float(row["observable_validation_score"]),
        )
        selected[scenario] = (float(best["alpha"]), float(best["lambda_Q"]))

    write_csv(output_root / "tuning_grid.csv", grid_rows)

    rows: list[dict] = []
    for scenario in SCENARIOS:
        alpha, lambda_q = selected[scenario]
        for seed in FINAL_SEEDS:
            run_dir = (
                output_root
                / "final_runs"
                / scenario
                / f"seed_{seed}"
                / f"a{fmt(alpha)}_lq{fmt(lambda_q)}"
            )
            if not args.collect_only:
                run_dgm(
                    data_root / scenario / f"seed_{seed}",
                    run_dir,
                    scenario,
                    seed,
                    alpha,
                    lambda_q,
                )
            rows.append(evaluate_neural_decoder(
                run_dir,
                data_root / scenario / f"seed_{seed}",
                scenario,
                seed,
                alpha,
                lambda_q,
            ))

    aggregate_rows = aggregate(rows)
    write_csv(output_root / "results_per_seed.csv", rows)
    write_csv(output_root / "results_aggregate.csv", aggregate_rows)
    copy_representative_figures(output_root, selected)
    protocol = {
        "fixed_topologies": {
            scenario: str((topology_root / f"water_topology_{scenario}.npz").relative_to(PROJECT_ROOT))
            for scenario in SCENARIOS
        },
        "tuning_seed": TUNING_SEED,
        "final_seeds": list(FINAL_SEEDS),
        "selection_rule": "minimum best-checkpoint observable_validation_score on tuning seed; no GT metrics",
        "grid": {"alpha": list(ALPHAS), "lambda_Q": list(LAMBDA_QS)},
        "dgm": DGM_CONFIG,
        "neural_decoder": DECODER_CONFIG,
        "joint_evaluation": "0.5 * (1 / (1 + test_MSE)) + 0.5 * topology_F1",
        "accuracy_definition": "(TP + TN) / C(N,2), undirected node pairs without self-pairs",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "aggregate": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
