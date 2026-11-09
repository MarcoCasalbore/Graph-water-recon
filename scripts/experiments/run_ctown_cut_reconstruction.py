"""Extended GT-free reconstruction study on the frozen C-Town cut A.

The subgraph is never regenerated. Hyperparameters are selected from
forecasting, observable-prior agreement, and DGM stability only. Ground-truth
edges are loaded by the evaluation functions only after each stage selection
has been written to disk.
"""

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

from scripts.experiments.run_section_vi_water_final import (  # noqa: E402
    DECODER_CONFIG,
    evaluate_neural_decoder,
)
from scripts.synthetic.generate_synthetic_water import make_splits  # noqa: E402


TUNING_SEED = 7
FINAL_SEEDS = (11, 22, 33)
WINDOW_STARTS = {7: 0, 11: 1980, 22: 3960, 33: 5940}
WINDOW_LENGTH = 720

PRIORS = (
    "observable_geometry",
    "observable_geometry_plus",
    "observable_expert_pressure",
    "observable_expert_dynamic",
    "observable_expert_full",
    "pressure_corr",
    "pressure_smooth",
    "pressure_partial",
    "geo_pressure_evidence",
    "global_dynamic_proxy",
    "geo_global_dynamic_proxy",
    "geo_global_dynamic_bridge_proxy",
)
ALPHAS = tuple(round(value / 10, 1) for value in range(1, 10))
LAMBDA_QS = (0.1, 0.25, 0.5, 1.0, 2.0)
CANDIDATE_MODES = ("geo_knn", "observable_union", "local_bridge", "q_dynamic")
CANDIDATE_KS = (3, 5, 8)
FFUNS = ("gcn", "gat", "mlp")
GFUNS = ("gcn", "gat")
DGM_KS = (2, 3)
LEARNING_RATES = (5e-4, 1e-3)
COORDINATE_DIMS = (0, 4)
TASK_LOSSES = ("mse", "mae", "huber")

BASE = {
    "lags": 6,
    "prediction_window": 12,
    "batch_size": 64,
    "check_every": 4,
    "alpha": 0.5,
    "lambda_q": 1.0,
    "candidate_mode": "observable_union",
    "candidate_k": 5,
    "candidate_signal_k": 4,
    "dgm_k": 2,
    "ffun": "gcn",
    "gfun": "gcn",
    "learning_rate": 1e-3,
    "coordinate_dim": 4,
    "task_loss": "huber",
    "lambda_stability": 0.1,
}


def tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_window(source_dir: Path, out_dir: Path, seed: int) -> None:
    data_path = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if data_path.exists() and meta_path.exists():
        return

    source = np.load(source_dir / "data.npz")
    source_meta = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
    start = WINDOW_STARTS[seed]
    stop = start + WINDOW_LENGTH
    if stop > source["pressure"].shape[0]:
        raise ValueError(f"Temporal window {start}:{stop} exceeds source data")

    out_dir.mkdir(parents=True, exist_ok=True)
    extras = {}
    for key in ("global_node_index", "source_nodes", "block_labels"):
        if key in source.files:
            extras[key] = source[key]
    time_s = (
        source["time_s"][start:stop]
        if "time_s" in source.files
        else np.arange(WINDOW_LENGTH, dtype=np.int64)
    )
    np.savez_compressed(
        data_path,
        pressure=source["pressure"][start:stop],
        demand=source["demand"][start:stop],
        node_xy=source["node_xy"],
        edge_index=source["edge_index"],
        initial_edge_index=source["initial_edge_index"],
        time_s=time_s,
        **extras,
    )
    meta = {
        **source_meta,
        "name": "ctown_frozen_cut_a_temporal_window",
        "topology_regenerated": False,
        "source_dataset": str(source_dir.relative_to(PROJECT_ROOT)),
        "signal_seed": seed,
        "temporal_window": {"start": start, "stop": stop},
        "splits": make_splits(WINDOW_LENGTH),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def validation_parts(run_dir: Path) -> tuple[float, float, float, float]:
    metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    val = metrics.get("best_observable_validation") or {}
    return (
        float(val["mse"]),
        float(val["soft_pseudo_edge_loss"]),
        float(val["stability_loss"]),
        float(val["observable_val_score"]),
    )


def rank01(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def add_selection_score(rows: list[dict]) -> None:
    task = rank01([float(row["val_mse"]) for row in rows])
    prior = rank01([float(row["val_LQ"]) for row in rows])
    stability = rank01([float(row["val_stability"]) for row in rows])
    for i, row in enumerate(rows):
        row["task_rank"] = float(task[i])
        row["prior_rank"] = float(prior[i])
        row["stability_rank"] = float(stability[i])
        row["selection_score"] = float(
            0.475 * task[i] + 0.475 * prior[i] + 0.05 * stability[i]
        )


def run_dgm(
    data_dir: Path,
    out_dir: Path,
    seed: int,
    *,
    prior: str,
    alpha: float,
    lambda_q: float,
    candidate_mode: str,
    candidate_k: int,
    dgm_k: int,
    ffun: str,
    gfun: str,
    learning_rate: float,
    coordinate_dim: int,
    task_loss: str,
    epochs: int,
) -> None:
    if (out_dir / "forecast_metrics.json").exists() and (out_dir / "dgm_prob_mean.npy").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = int(np.load(data_dir / "data.npz", mmap_mode="r")["pressure"].shape[1])
    run_name = "_".join(
        (
            "ctown_cut_a",
            f"s{seed}",
            prior,
            f"a{tag(alpha)}",
            f"lq{tag(lambda_q)}",
            f"cm{candidate_mode}",
            f"ck{candidate_k}",
            f"k{dgm_k}",
            f"{ffun}-{gfun}",
            f"lr{tag(learning_rate)}",
            f"ce{coordinate_dim}",
            task_loss,
            f"e{epochs}",
        )
    )
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--run_name", run_name,
        "--num_nodes", str(num_nodes),
        "--lags", str(BASE["lags"]),
        "--prediction_window", str(BASE["prediction_window"]),
        "--max_epochs", str(epochs),
        "--check_val_every_n_epoch", str(BASE["check_every"]),
        "--batch_size", str(BASE["batch_size"]),
        "--lr", str(learning_rate),
        "--k", str(dgm_k),
        "--ffun", ffun,
        "--gfun", gfun,
        "--lambda_graph", "0.0",
        "--use_complementary_loss",
        "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", str(lambda_q),
        "--topology_loss_normalization", "uniform_baseline",
        "--lambda_stability", str(BASE["lambda_stability"]),
        "--pseudo_score_mode", prior,
        "--pseudo_use_static_target",
        "--pseudo_topk", str(dgm_k),
        "--pseudo_include_mst",
        "--select_best_by", "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode", candidate_mode,
        "--candidate_k", str(candidate_k),
        "--candidate_signal_k", str(BASE["candidate_signal_k"]),
        "--static_top_edges", "0",
        "--static_row_k", "2",
        "--static_symmetry", "mutual",
        "--static_include_mst",
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
        "--task_loss_type", task_loss,
        "--task_huber_delta", "1.0",
        "--task_feature_combine", "weighted_mean",
        "--task_feature_normalization", "target_std",
        "--pressure_weight", "1.0",
        "--demand_weight", "1.0",
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
        tail = (out_dir / "run.log").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-50:]
        raise RuntimeError(f"DGM run failed: {out_dir}\n" + "\n".join(tail))


def execute_stage(
    name: str,
    configs: list[dict],
    data_dir: Path,
    output_root: Path,
    epochs: int,
    collect_only: bool,
) -> tuple[list[dict], dict]:
    rows = []
    stage_dir = output_root / name
    for index, config in enumerate(configs, start=1):
        run_dir = stage_dir / f"run_{index:03d}"
        print(f"[{name}] {index}/{len(configs)} {config}", flush=True)
        if not collect_only:
            run_dgm(
                data_dir,
                run_dir,
                TUNING_SEED,
                epochs=epochs,
                **config,
            )
        mse, lq, stability, checkpoint_score = validation_parts(run_dir)
        rows.append(
            {
                **config,
                "epochs": epochs,
                "val_mse": mse,
                "val_LQ": lq,
                "val_stability": stability,
                "checkpoint_J_obs": checkpoint_score,
                "run_dir": str(run_dir),
            }
        )
    add_selection_score(rows)
    write_csv(output_root / f"{name}.csv", rows)
    selected = min(
        rows,
        key=lambda row: (float(row["selection_score"]), float(row["val_stability"])),
    )
    (output_root / f"{name}_selected_gt_free.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(f"[{name}] selected: {selected}", flush=True)
    return rows, selected


def config_from(row: dict) -> dict:
    keys = (
        "prior",
        "alpha",
        "lambda_q",
        "candidate_mode",
        "candidate_k",
        "dgm_k",
        "ffun",
        "gfun",
        "learning_rate",
        "coordinate_dim",
        "task_loss",
    )
    return {key: row[key] for key in keys}


def prior_configs(smoke: bool) -> list[dict]:
    priors = PRIORS[:2] if smoke else PRIORS
    return [config_from({**BASE, "prior": prior}) for prior in priors]


def weight_configs(base: dict, smoke: bool) -> list[dict]:
    alphas = (0.3, 0.7) if smoke else ALPHAS
    lambdas = (0.5, 1.0) if smoke else LAMBDA_QS
    return [
        {**base, "alpha": alpha, "lambda_q": lambda_q}
        for alpha in alphas
        for lambda_q in lambdas
    ]


def candidate_configs(base: dict, smoke: bool) -> list[dict]:
    modes = CANDIDATE_MODES[:2] if smoke else CANDIDATE_MODES
    ks = (3,) if smoke else CANDIDATE_KS
    return [
        {**base, "candidate_mode": mode, "candidate_k": candidate_k}
        for mode in modes
        for candidate_k in ks
    ]


def module_configs(base: dict, smoke: bool) -> list[dict]:
    ffuns = FFUNS[:1] if smoke else FFUNS
    gfuns = GFUNS[:1] if smoke else GFUNS
    return [{**base, "ffun": ffun, "gfun": gfun} for ffun in ffuns for gfun in gfuns]


def optimization_configs(base: dict, smoke: bool) -> list[dict]:
    dgm_ks = DGM_KS[:1] if smoke else DGM_KS
    lrs = LEARNING_RATES[:1] if smoke else LEARNING_RATES
    dims = COORDINATE_DIMS[:1] if smoke else COORDINATE_DIMS
    losses = TASK_LOSSES[:1] if smoke else TASK_LOSSES
    return [
        {
            **base,
            "dgm_k": dgm_k,
            "learning_rate": learning_rate,
            "coordinate_dim": coordinate_dim,
            "task_loss": task_loss,
        }
        for dgm_k in dgm_ks
        for learning_rate in lrs
        for coordinate_dim in dims
        for task_loss in losses
    ]


def evaluate_prior_ablation(
    rows: list[dict],
    data_dir: Path,
    output_root: Path,
) -> None:
    posthoc = []
    for row in rows:
        result = evaluate_neural_decoder(
            Path(row["run_dir"]),
            data_dir,
            str(row["prior"]),
            TUNING_SEED,
            float(row["alpha"]),
            float(row["lambda_q"]),
        )
        posthoc.append(
            {
                "prior": row["prior"],
                "selection_score_gt_free": row["selection_score"],
                "mse": result["mse"],
                "mae": result["mae"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "joint_eval_0p5": result["joint_eval_0p5"],
            }
        )
    write_csv(output_root / "prior_ablation_posthoc.csv", posthoc)


def final_runs(
    selected: dict,
    data_root: Path,
    output_root: Path,
    collect_only: bool,
    smoke: bool,
) -> list[dict]:
    rows = []
    seeds = FINAL_SEEDS[:1] if smoke else FINAL_SEEDS
    epochs = 8 if smoke else 80
    config = config_from(selected)
    for seed in seeds:
        run_dir = output_root / "final_runs" / f"seed_{seed}"
        print(f"[final] seed={seed} config={config}", flush=True)
        if not collect_only:
            run_dgm(
                data_root / f"seed_{seed}",
                run_dir,
                seed,
                epochs=epochs,
                **config,
            )
        result = evaluate_neural_decoder(
            run_dir,
            data_root / f"seed_{seed}",
            "ctown_cut_a",
            seed,
            float(config["alpha"]),
            float(config["lambda_q"]),
        )
        rows.append({**config, "epochs": epochs, **result})
    write_csv(output_root / "results_per_seed.csv", rows)
    return rows


def aggregate(rows: list[dict], output_root: Path) -> dict:
    metrics = ("precision", "accuracy", "recall", "f1", "mse", "mae", "joint_eval_0p5")
    result: dict[str, float | int] = {"n_seeds": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    write_csv(output_root / "results_aggregate.csv", [result])
    return result


def copy_representative_figure(rows: list[dict], output_root: Path) -> None:
    representative = max(rows, key=lambda row: float(row["joint_eval_0p5"]))
    seed = int(representative["seed"])
    source = (
        output_root
        / "final_runs"
        / f"seed_{seed}"
        / "neural_consensus"
        / "graph_comparison"
        / "graph_compare_undirected.png"
    )
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, figure_dir / f"ctown_cut_a_gt_vs_reconstructed_seed_{seed}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=PROJECT_ROOT
        / "results_section_vi"
        / "ctown_subgraph_candidates"
        / "ctown_cut_a_n50",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "ctown_cut_a_reconstruction",
    )
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    data_root = output_root / "datasets"
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = (TUNING_SEED,) + (FINAL_SEEDS[:1] if args.smoke else FINAL_SEEDS)
    for seed in seeds:
        prepare_window(source_dir, data_root / f"seed_{seed}", seed)

    epochs = {
        "stage0_prior": 4 if args.smoke else 12,
        "stage1_weights": 4 if args.smoke else 16,
        "stage2_candidates": 4 if args.smoke else 20,
        "stage3_modules": 4 if args.smoke else 24,
        "stage4_optimization": 4 if args.smoke else 24,
    }

    prior_rows, selected = execute_stage(
        "stage0_prior",
        prior_configs(args.smoke),
        data_root / f"seed_{TUNING_SEED}",
        output_root,
        epochs["stage0_prior"],
        args.collect_only,
    )
    evaluate_prior_ablation(prior_rows, data_root / f"seed_{TUNING_SEED}", output_root)

    _, selected = execute_stage(
        "stage1_weights",
        weight_configs(config_from(selected), args.smoke),
        data_root / f"seed_{TUNING_SEED}",
        output_root,
        epochs["stage1_weights"],
        args.collect_only,
    )
    _, selected = execute_stage(
        "stage2_candidates",
        candidate_configs(config_from(selected), args.smoke),
        data_root / f"seed_{TUNING_SEED}",
        output_root,
        epochs["stage2_candidates"],
        args.collect_only,
    )
    _, selected = execute_stage(
        "stage3_modules",
        module_configs(config_from(selected), args.smoke),
        data_root / f"seed_{TUNING_SEED}",
        output_root,
        epochs["stage3_modules"],
        args.collect_only,
    )
    _, selected = execute_stage(
        "stage4_optimization",
        optimization_configs(config_from(selected), args.smoke),
        data_root / f"seed_{TUNING_SEED}",
        output_root,
        epochs["stage4_optimization"],
        args.collect_only,
    )

    final = final_runs(
        selected,
        data_root,
        output_root,
        args.collect_only,
        args.smoke,
    )
    summary = aggregate(final, output_root)
    copy_representative_figure(final, output_root)

    protocol = {
        "source_topology": str(source_dir.relative_to(PROJECT_ROOT)),
        "topology_regenerated": False,
        "tuning_seed": TUNING_SEED,
        "final_seeds": list(FINAL_SEEDS[:1] if args.smoke else FINAL_SEEDS),
        "temporal_windows": {
            str(seed): [WINDOW_STARTS[seed], WINDOW_STARTS[seed] + WINDOW_LENGTH]
            for seed in seeds
        },
        "selection": (
            "GT-free fixed score: 0.475*rank(val_MSE) + "
            "0.475*rank(val_LQ) + 0.05*rank(val_stability)"
        ),
        "stages": {
            "prior": list(PRIORS),
            "alpha": list(ALPHAS),
            "lambda_Q": list(LAMBDA_QS),
            "candidate_modes": list(CANDIDATE_MODES),
            "candidate_k": list(CANDIDATE_KS),
            "ffun": list(FFUNS),
            "gfun": list(GFUNS),
            "dgm_k": list(DGM_KS),
            "learning_rate": list(LEARNING_RATES),
            "coordinate_dim": list(COORDINATE_DIMS),
            "task_loss": list(TASK_LOSSES),
            "epochs": epochs,
        },
        "base_architecture": BASE,
        "neural_consensus": DECODER_CONFIG,
        "joint_evaluation": "0.5*(1/(1+test_MSE)) + 0.5*topology_F1; post-hoc only",
        "selected_configuration": config_from(selected),
        "aggregate": summary,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": config_from(selected), "aggregate": summary}, indent=2))


if __name__ == "__main__":
    main()
