"""Extended GT-free study for the four Section VI water base candidates."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostics.search_edge_scores import build_scores, top_edges  # noqa: E402
from scripts.experiments.run_section_vi_water_final import (  # noqa: E402
    DECODER_CONFIG,
    evaluate_neural_decoder,
    observable_score,
    undirected_edges,
)
from scripts.experiments.run_train_synthetic import (  # noqa: E402
    build_knn_candidate_mask,
    build_q_dynamic_candidate_mask,
)
from scripts.synthetic.generate_synthetic_water import (  # noqa: E402
    build_knn_edges,
    make_splits,
    simulate_timeseries,
    to_edge_index,
)


SCENARIOS = {
    "option_a_organic": "base_option_a_organic.npz",
    "option_b_street_mesh": "base_option_b_street_mesh.npz",
    "option_c_dual_main": "base_option_c_dual_main.npz",
    "option_d_neighborhoods": "base_option_d_neighborhoods.npz",
}
TUNING_SEED = 7
FINAL_SEEDS = (11, 22, 33, 44, 55)
ALPHAS = tuple(round(value / 10, 1) for value in range(1, 10))
LAMBDA_QS = (0.1, 0.25, 0.5, 1.0, 2.0)
CANDIDATE_KS = (3, 5, 8)
DGM_KS = (2, 3)
LEARNING_RATES = (5e-4, 1e-3)
COORDINATE_DIMS = (0, 4)

COMMON = {
    "lags": 6,
    "prediction_window": 12,
    "batch_size": 32,
    "check_every": 4,
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
    "pressure_smooth_tau": 2.2,
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def prepare_dataset(topology_path: Path, out_dir: Path, seed: int) -> None:
    data_path = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tau = float((meta.get("signal_generator") or {}).get("pressure_smooth_tau", -1))
        if abs(tau - float(COMMON["pressure_smooth_tau"])) < 1e-12:
            return
        raise RuntimeError(f"Existing dataset has incompatible pressure_smooth_tau: {out_dir}")

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
        pressure_smooth_tau=float(COMMON["pressure_smooth_tau"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        pressure=pressure,
        demand=demand,
        node_xy=xy,
        edge_index=to_edge_index(gt, bidirectional=True),
        initial_edge_index=to_edge_index(initial, bidirectional=True),
        time_s=np.arange(720, dtype=np.int64) * 3600,
        block_labels=topology["block_labels"],
    )
    meta = {
        "name": "section_vi_water_base_candidate",
        "topology_source": str(topology_path.relative_to(PROJECT_ROOT)),
        "topology_regenerated": False,
        "signal_seed": seed,
        "node_ids": [f"N{i:03d}" for i in range(len(xy))],
        "num_nodes": len(xy),
        "num_gt_undirected_edges": len(gt),
        "num_initial_undirected_edges": len(initial),
        "splits": make_splits(720),
        "signal_generator": {
            "timesteps": 720,
            "pressure_mode": "smooth_laplacian",
            "pressure_smooth_tau": COMMON["pressure_smooth_tau"],
            "initial_knn": 3,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def tag(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def run_dgm(
    data_dir: Path,
    out_dir: Path,
    scenario: str,
    seed: int,
    *,
    alpha: float,
    lambda_q: float,
    candidate_k: int,
    dgm_k: int,
    learning_rate: float,
    coordinate_dim: int,
    epochs: int,
) -> None:
    if (out_dir / "forecast_metrics.json").exists() and (out_dir / "dgm_prob_mean.npy").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{scenario}_s{seed}_a{tag(alpha)}_lq{tag(lambda_q)}_ck{candidate_k}_"
        f"k{dgm_k}_lr{tag(learning_rate)}_ce{coordinate_dim}_e{epochs}"
    )
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--run_name", run_name,
        "--num_nodes", "50",
        "--lags", str(COMMON["lags"]),
        "--prediction_window", str(COMMON["prediction_window"]),
        "--max_epochs", str(epochs),
        "--check_val_every_n_epoch", str(COMMON["check_every"]),
        "--batch_size", str(COMMON["batch_size"]),
        "--lr", str(learning_rate),
        "--k", str(dgm_k),
        "--ffun", str(COMMON["ffun"]),
        "--gfun", str(COMMON["gfun"]),
        "--lambda_graph", "0.0",
        "--use_complementary_loss",
        "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", str(lambda_q),
        "--lambda_stability", str(COMMON["lambda_stability"]),
        "--pseudo_score_mode", str(COMMON["prior"]),
        "--pseudo_topk", str(dgm_k),
        "--pseudo_include_mst",
        "--select_best_by", "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode", "q_dynamic",
        "--candidate_k", str(candidate_k),
        "--static_top_edges", "0",
        "--static_row_k", "2",
        "--static_symmetry", "mutual",
        "--static_include_mst",
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
        "--task_loss_type", str(COMMON["task_loss"]),
        "--task_huber_delta", str(COMMON["task_huber_delta"]),
        "--task_feature_combine", str(COMMON["task_feature_combine"]),
        "--task_feature_normalization", str(COMMON["task_feature_normalization"]),
        "--pressure_weight", str(COMMON["pressure_weight"]),
        "--demand_weight", str(COMMON["demand_weight"]),
        "--coordinate_embedding_dim", str(coordinate_dim),
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
        tail = (out_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        raise RuntimeError(f"Run failed: {out_dir}\n" + "\n".join(tail))


def validation_parts(run_dir: Path) -> tuple[float, float, float]:
    metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    val = metrics.get("best_observable_validation") or {}
    return (
        float(val["mse"]),
        float(val["soft_pseudo_edge_loss"]),
        float(val["stability_loss"]),
    )


def rank01(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def add_fixed_selection_score(rows: list[dict]) -> None:
    for scenario in SCENARIOS:
        indices = [i for i, row in enumerate(rows) if row["scenario"] == scenario]
        task_rank = rank01([float(rows[i]["val_mse"]) for i in indices])
        q_rank = rank01([float(rows[i]["val_LQ_raw"]) for i in indices])
        stability_rank = rank01([float(rows[i]["val_stability"]) for i in indices])
        for local, index in enumerate(indices):
            rows[index]["task_rank"] = float(task_rank[local])
            rows[index]["q_rank"] = float(q_rank[local])
            rows[index]["stability_rank"] = float(stability_rank[local])
            rows[index]["selection_score"] = float(
                0.475 * task_rank[local]
                + 0.475 * q_rank[local]
                + 0.05 * stability_rank[local]
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def proxy_diagnostics(data_root: Path, output_root: Path) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        data_path = data_root / scenario / f"seed_{TUNING_SEED}" / "data.npz"
        data = np.load(data_path)
        xy = data["node_xy"]
        gt = undirected_edges(data["edge_index"])
        scores = build_scores(xy, data["pressure"], data["demand"], ridge=0.05, lag=4)
        proxy = set(top_edges(scores["geo_global_dynamic_proxy"], len(gt)))
        tp = len(proxy & gt)
        proxy_f1 = 2.0 * tp / max(1, len(proxy) + len(gt))
        for candidate_k in CANDIDATE_KS:
            mask = build_q_dynamic_candidate_mask(
                str(data_path),
                "observable_expert_full",
                candidate_k,
                0.05,
                1.0,
                True,
            ).numpy()
            candidates = {
                (u, v)
                for u in range(len(xy))
                for v in range(u + 1, len(xy))
                if mask[u, v]
            }
            rows.append({
                "scenario": scenario,
                "gt_edges": len(gt),
                "proxy_top_budget_f1": proxy_f1,
                "candidate_k": candidate_k,
                "candidate_edges": len(candidates),
                "candidate_recall": len(candidates & gt) / len(gt),
                "geometry_knn_recall": (
                    len(
                        {
                            (u, v)
                            for u in range(len(xy))
                            for v in range(u + 1, len(xy))
                            if build_knn_candidate_mask(xy, candidate_k).numpy()[u, v]
                        }
                        & gt
                    )
                    / len(gt)
                ),
            })
    write_csv(output_root / "proxy_candidate_diagnostics.csv", rows)
    return rows


def stage1(data_root: Path, output_root: Path, collect_only: bool) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for scenario in SCENARIOS:
        for alpha in ALPHAS:
            for lambda_q in LAMBDA_QS:
                run_dir = output_root / "stage1_weights" / scenario / f"a{tag(alpha)}_lq{tag(lambda_q)}"
                if not collect_only:
                    run_dgm(
                        data_root / scenario / f"seed_{TUNING_SEED}",
                        run_dir,
                        scenario,
                        TUNING_SEED,
                        alpha=alpha,
                        lambda_q=lambda_q,
                        candidate_k=5,
                        dgm_k=2,
                        learning_rate=1e-3,
                        coordinate_dim=4,
                        epochs=20,
                    )
                val_mse, val_lq, val_stability = validation_parts(run_dir)
                rows.append({
                    "scenario": scenario,
                    "alpha": alpha,
                    "lambda_Q": lambda_q,
                    "candidate_k": 5,
                    "dgm_k": 2,
                    "learning_rate": 1e-3,
                    "coordinate_dim": 4,
                    "epochs": 20,
                    "val_mse": val_mse,
                    "val_LQ_raw": val_lq,
                    "val_stability": val_stability,
                    "checkpoint_J_obs": observable_score(run_dir),
                    "run_dir": str(run_dir),
                })
    add_fixed_selection_score(rows)
    write_csv(output_root / "stage1_weights.csv", rows)
    selected = {
        scenario: min(
            (row for row in rows if row["scenario"] == scenario),
            key=lambda row: (float(row["selection_score"]), float(row["val_stability"])),
        )
        for scenario in SCENARIOS
    }
    return rows, selected


def stage2(
    data_root: Path,
    output_root: Path,
    weights: dict[str, dict],
    collect_only: bool,
) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for scenario, weight_row in weights.items():
        alpha = float(weight_row["alpha"])
        lambda_q = float(weight_row["lambda_Q"])
        for candidate_k in CANDIDATE_KS:
            for dgm_k in DGM_KS:
                for learning_rate in LEARNING_RATES:
                    for coordinate_dim in COORDINATE_DIMS:
                        name = (
                            f"ck{candidate_k}_k{dgm_k}_lr{tag(learning_rate)}_ce{coordinate_dim}"
                        )
                        run_dir = output_root / "stage2_structure" / scenario / name
                        if not collect_only:
                            run_dgm(
                                data_root / scenario / f"seed_{TUNING_SEED}",
                                run_dir,
                                scenario,
                                TUNING_SEED,
                                alpha=alpha,
                                lambda_q=lambda_q,
                                candidate_k=candidate_k,
                                dgm_k=dgm_k,
                                learning_rate=learning_rate,
                                coordinate_dim=coordinate_dim,
                                epochs=40,
                            )
                        val_mse, val_lq, val_stability = validation_parts(run_dir)
                        rows.append({
                            "scenario": scenario,
                            "alpha": alpha,
                            "lambda_Q": lambda_q,
                            "candidate_k": candidate_k,
                            "dgm_k": dgm_k,
                            "learning_rate": learning_rate,
                            "coordinate_dim": coordinate_dim,
                            "epochs": 40,
                            "val_mse": val_mse,
                            "val_LQ_raw": val_lq,
                            "val_stability": val_stability,
                            "checkpoint_J_obs": observable_score(run_dir),
                            "run_dir": str(run_dir),
                        })
    add_fixed_selection_score(rows)
    write_csv(output_root / "stage2_structure.csv", rows)
    selected = {
        scenario: min(
            (row for row in rows if row["scenario"] == scenario),
            key=lambda row: (float(row["selection_score"]), float(row["val_stability"])),
        )
        for scenario in SCENARIOS
    }
    return rows, selected


def final_runs(
    data_root: Path,
    output_root: Path,
    selected: dict[str, dict],
    collect_only: bool,
) -> list[dict]:
    rows = []
    for scenario, config in selected.items():
        for seed in FINAL_SEEDS:
            run_dir = output_root / "final_runs" / scenario / f"seed_{seed}"
            if not collect_only:
                run_dgm(
                    data_root / scenario / f"seed_{seed}",
                    run_dir,
                    scenario,
                    seed,
                    alpha=float(config["alpha"]),
                    lambda_q=float(config["lambda_Q"]),
                    candidate_k=int(config["candidate_k"]),
                    dgm_k=int(config["dgm_k"]),
                    learning_rate=float(config["learning_rate"]),
                    coordinate_dim=int(config["coordinate_dim"]),
                    epochs=60,
                )
            rows.append(evaluate_neural_decoder(
                run_dir,
                data_root / scenario / f"seed_{seed}",
                scenario,
                seed,
                float(config["alpha"]),
                float(config["lambda_Q"]),
            ))
    write_csv(output_root / "results_per_seed.csv", rows)
    return rows


def aggregate(rows: list[dict], selected: dict[str, dict], output_root: Path) -> list[dict]:
    metrics = ("precision", "accuracy", "recall", "f1", "mse", "mae", "joint_eval_0p5")
    output = []
    for scenario in SCENARIOS:
        values = [row for row in rows if row["scenario"] == scenario]
        config = selected[scenario]
        record = {
            "scenario": scenario,
            "n_seeds": len(values),
            "alpha": config["alpha"],
            "lambda_Q": config["lambda_Q"],
            "candidate_k": config["candidate_k"],
            "dgm_k": config["dgm_k"],
            "learning_rate": config["learning_rate"],
            "coordinate_dim": config["coordinate_dim"],
            "epochs": 60,
        }
        for metric in metrics:
            array = np.asarray([float(row[metric]) for row in values])
            record[f"{metric}_mean"] = float(array.mean())
            record[f"{metric}_std"] = float(array.std(ddof=1))
        output.append(record)
    write_csv(output_root / "results_aggregate.csv", output)
    return output


def copy_figures(output_root: Path) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        source = (
            output_root
            / "final_runs"
            / scenario
            / f"seed_{FINAL_SEEDS[0]}"
            / "neural_consensus"
            / "graph_comparison"
            / "graph_compare_undirected.png"
        )
        shutil.copy2(source, figure_dir / f"{scenario}_gt_vs_neural_seed_{FINAL_SEEDS[0]}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "water_topologies" / "base_candidates",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "water_base_candidate_experiments",
    )
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()
    topology_root = args.topology_root.resolve()
    output_root = args.output_root.resolve()
    data_root = output_root / "datasets"
    output_root.mkdir(parents=True, exist_ok=True)

    for scenario, filename in SCENARIOS.items():
        for seed in (TUNING_SEED,) + FINAL_SEEDS:
            prepare_dataset(topology_root / filename, data_root / scenario / f"seed_{seed}", seed)

    proxy_rows = proxy_diagnostics(data_root, output_root)
    _, selected_weights = stage1(data_root, output_root, args.collect_only)
    _, selected_structure = stage2(
        data_root,
        output_root,
        selected_weights,
        args.collect_only,
    )
    final_rows = final_runs(data_root, output_root, selected_structure, args.collect_only)
    aggregate_rows = aggregate(final_rows, selected_structure, output_root)
    copy_figures(output_root)
    protocol = {
        "topologies": {
            scenario: str((topology_root / filename).relative_to(PROJECT_ROOT))
            for scenario, filename in SCENARIOS.items()
        },
        "topology_regenerated": False,
        "tuning_seed": TUNING_SEED,
        "final_seeds": list(FINAL_SEEDS),
        "signal": {"timesteps": 720, "pressure_smooth_tau": COMMON["pressure_smooth_tau"]},
        "stage1_grid": {"alpha": list(ALPHAS), "lambda_Q": list(LAMBDA_QS), "epochs": 20},
        "stage2_grid": {
            "candidate_k": list(CANDIDATE_KS),
            "dgm_k": list(DGM_KS),
            "learning_rate": list(LEARNING_RATES),
            "coordinate_dim": list(COORDINATE_DIMS),
            "epochs": 40,
        },
        "final_epochs": 60,
        "selection": (
            "GT-free within-scenario percentile score: "
            "0.475*rank(val_MSE) + 0.475*rank(raw val_LQ) + 0.05*rank(val_stability)"
        ),
        "common_architecture": COMMON,
        "neural_decoder": DECODER_CONFIG,
        "joint_evaluation": "0.5*(1/(1+test_MSE)) + 0.5*F1; post-hoc only",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({
        "proxy_diagnostics": proxy_rows,
        "selected_weights": selected_weights,
        "selected_structure": selected_structure,
        "aggregate": aggregate_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
