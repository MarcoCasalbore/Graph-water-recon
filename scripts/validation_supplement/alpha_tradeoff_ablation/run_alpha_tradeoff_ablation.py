"""Run and analyse the predeclared alpha structural-constraint ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
OUT = FINAL / "alpha_tradeoff_ablation"
RUN_ROOT = OUT / "runs"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
BASELINE = FINAL / "alpha_sensitivity_dense_numeric/dense_alpha_per_seed.csv"
ALPHAS = [round(value / 10, 1) for value in range(11)]
SEEDS = [42, 43, 44]

sys.path.insert(0, str(PROJECT / "scripts/validation"))
import run_campaign_v2 as campaign  # noqa: E402
import run_corrected_metric_sweeps as corrected  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts/experiments"))
from run_train_synthetic import decode_observable_score  # noqa: E402


NEW_VARIANTS = {
    "lq1_mask": (1.0, True),
    "lq05_nomask": (0.5, False),
    "lq1_nomask": (1.0, False),
}


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.1f}".replace(".", "p")


def replace_arg(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def build_command(out: Path, run_name: str, alpha: float, seed: int,
                  lambda_q: float, use_mask: bool) -> list[str]:
    command = campaign.joint_command(out, run_name, 6, 12, alpha, 2, 3, seed, 120)
    replace_arg(command, "--data_dir", str(DATA))
    replace_arg(command, "--lambda_soft_pseudo_edge", str(lambda_q))
    if not use_mask:
        command.remove("--assign_candidate_mask")
    return command


def run_one(spec: tuple[str, float, bool, float, int]) -> tuple[str, float]:
    variant, lambda_q, use_mask, alpha, seed = spec
    run_name = (
        f"{variant}_g211_s1211_t{seed}_l6_h12_a{alpha_slug(alpha)}_"
        "kd2_kc3_kq2_e120"
    )
    out = RUN_ROOT / variant / run_name
    out.mkdir(parents=True, exist_ok=True)
    command = build_command(out, run_name, alpha, seed, lambda_q, use_mask)
    config = {
        "variant": variant,
        "lambda_Q": lambda_q,
        "q_candidate_mask": use_mask,
        "alpha": alpha,
        "graph_seed": 211,
        "signal_seed": 1211,
        "training_seed": seed,
        "lags": 6,
        "horizon": 12,
        "epochs": 120,
        "k_dgm": 2,
        "candidate_k": 3 if use_mask else None,
        "q_topk": 2,
        "decoder_training_export": "row_k3_mutual_mst",
        "command": command,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    required = [out / "test_predictions.npz", out / "dgm_soft_score.npy",
                out / "forecast_metrics.json"]
    if all(path.exists() for path in required):
        return run_name, 0.0

    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    with (out / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    (out / "elapsed_seconds.txt").write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    return run_name, elapsed


def canonical_reference() -> set[tuple[int, int]]:
    edge_index = np.load(DATA / "data.npz")["edge_index"]
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.T if int(u) != int(v)
    }


def graph_metrics(edges: list[tuple[int, int]], reference: set[tuple[int, int]]) -> dict:
    learned = set(edges)
    tp = len(learned & reference)
    fp = len(learned - reference)
    fn = len(reference - learned)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    graph.add_edges_from(learned)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "edge_count": len(learned),
        "mean_degree": 2 * len(learned) / 50,
        "components": nx.number_connected_components(graph),
    }


def decoded_metrics(out: Path, include_mst: bool,
                    reference: set[tuple[int, int]]) -> dict:
    score = torch.as_tensor(np.load(out / "dgm_soft_score.npy"), dtype=torch.float32)
    edges = decode_observable_score(score, row_k=3, symmetry="mutual",
                                    include_mst=include_mst,
                                    data_npz=str(DATA / "data.npz"))
    return graph_metrics(edges, reference)


def baseline_runs() -> list[dict]:
    source = pd.read_csv(BASELINE)
    rows = []
    for _, row in source.iterrows():
        rows.append({
            "training_variant": "lq05_mask",
            "lambda_Q": 0.5,
            "q_candidate_mask": True,
            "alpha": float(row.alpha),
            "training_seed": int(row.training_seed),
            "output_dir": str(row.output_dir),
            "runtime_seconds": float(row.runtime_seconds),
        })
    return rows


def new_runs() -> list[dict]:
    rows = []
    for variant, (lambda_q, use_mask) in NEW_VARIANTS.items():
        for alpha in ALPHAS:
            for seed in SEEDS:
                run_name = (
                    f"{variant}_g211_s1211_t{seed}_l6_h12_a{alpha_slug(alpha)}_"
                    "kd2_kc3_kq2_e120"
                )
                out = RUN_ROOT / variant / run_name
                elapsed_file = out / "elapsed_seconds.txt"
                rows.append({
                    "training_variant": variant,
                    "lambda_Q": lambda_q,
                    "q_candidate_mask": use_mask,
                    "alpha": alpha,
                    "training_seed": seed,
                    "output_dir": str(out.relative_to(PROJECT)),
                    "runtime_seconds": float(elapsed_file.read_text())
                    if elapsed_file.exists() else 0.0,
                })
    return rows


def analyse_training_run(run: dict, reference: set[tuple[int, int]]) -> list[dict]:
    out = PROJECT / run["output_dir"]
    summary = corrected.summarize(
        out, run["training_variant"], 6, 12, run["alpha"], 2,
        3 if run["q_candidate_mask"] else 0,
        run["training_seed"], run["runtime_seconds"],
    )
    common = {
        **run,
        "selected_epoch": int(summary["selected_epoch"]),
        "test_mse": float(summary["mse"]),
        "test_mae": float(summary["mae"]),
        "forecast_loss": float(summary["task_loss"]),
        "structural_loss": float(summary["topology_loss"]),
    }
    rows = []
    for decoder, include_mst in [("row_k3_mutual_mst", True),
                                 ("row_k3_mutual_no_mst", False)]:
        topology = decoded_metrics(out, include_mst, reference)
        task_score = 1.0 / (1.0 + common["test_mse"])
        rows.append({
            **common,
            "decoder": decoder,
            **topology,
            "task_score": task_score,
            "joint_S_0.5": 0.5 * (task_score + topology["f1"]),
        })
    return rows


def add_conditional_variants(frame: pd.DataFrame) -> pd.DataFrame:
    additions = []
    for lambda_label in ["lq05", "lq1"]:
        masked = f"{lambda_label}_mask"
        unmasked = f"{lambda_label}_nomask"
        selected = frame[
            ((frame.training_variant == masked) & (frame.alpha < 1.0))
            | ((frame.training_variant == unmasked) & np.isclose(frame.alpha, 1.0))
        ].copy()
        selected["training_variant"] = f"{lambda_label}_conditional"
        selected["q_candidate_mask"] = selected.alpha < 1.0
        additions.append(selected)
    return pd.concat([frame, *additions], ignore_index=True)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_mse", "test_mae", "precision", "recall", "f1", "jaccard",
        "edge_count", "mean_degree", "components", "task_score", "joint_S_0.5",
        "forecast_loss", "structural_loss", "selected_epoch",
    ]
    grouped = frame.groupby(["training_variant", "decoder", "alpha"], sort=True)
    pieces = []
    for metric in metrics:
        pieces.append(grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    result = pd.concat(pieces, axis=1).reset_index()
    result["n_seeds"] = grouped.size().values
    return result


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_variant(aggregate_frame: pd.DataFrame, variant: str, decoder: str,
                 stem: str) -> None:
    data = aggregate_frame[
        (aggregate_frame.training_variant == variant)
        & (aggregate_frame.decoder == decoder)
    ].sort_values("alpha")
    if len(data) != len(ALPHAS):
        raise ValueError(f"Incomplete plot data for {variant}/{decoder}: {len(data)} rows")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.25, 3.43))
    colors = {"mse": "#1f77b4", "mae": "#ff7f0e",
              "precision": "#2ca02c", "recall": "#d62728"}
    axes[0].plot(data.alpha, data.test_mse_mean, "o-", color=colors["mse"],
                 lw=2, ms=4.2, label="MSE")
    axes[0].plot(data.alpha, data.test_mae_mean, "s-", color=colors["mae"],
                 lw=2, ms=4.0, label="MAE")
    axes[1].plot(data.alpha, data.precision_mean, "o-", color=colors["precision"],
                 lw=2, ms=4.2, label="Precision")
    axes[1].plot(data.alpha, data.recall_mean, "s-", color=colors["recall"],
                 lw=2, ms=4.0, label="Recall")

    axes[0].set_ylabel("Forecasting error")
    axes[1].set_ylabel("Topology-recovery score")
    for ax in axes:
        ax.set_xlabel(r"Weight $\alpha$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(ALPHAS)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
        ax.legend(frameon=False, loc="best")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.15)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.97, wspace=0.28)
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 400})]:
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_primary_table(aggregate_frame: pd.DataFrame, variant: str,
                        decoder: str) -> None:
    data = aggregate_frame[
        (aggregate_frame.training_variant == variant)
        & (aggregate_frame.decoder == decoder)
    ].sort_values("alpha")
    columns = {
        "alpha": "alpha",
        "test_mse_mean": "Test MSE",
        "test_mae_mean": "Test MAE",
        "precision_mean": "Precision",
        "recall_mean": "Recall",
        "f1_mean": "F1",
        "jaccard_mean": "Jaccard",
        "edge_count_mean": "Edges",
        "task_score_mean": "Task score",
        "joint_S_0.5_mean": "Joint S_0.5",
    }
    table = data[list(columns)].rename(columns=columns)
    table.to_csv(OUT / "primary_large_table.csv", index=False)
    (OUT / "primary_large_table.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    specs = [
        (variant, lambda_q, use_mask, alpha, seed)
        for variant, (lambda_q, use_mask) in NEW_VARIANTS.items()
        for alpha in ALPHAS for seed in SEEDS
    ]
    if not args.analyse_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, spec): spec for spec in specs}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                spec = futures[future]
                name, elapsed = future.result()
                print(f"[OK {index:03d}/{len(specs)}] {name} ({elapsed:.1f}s)", flush=True)

    reference = canonical_reference()
    run_rows = baseline_runs() + new_runs()
    analysed = []
    for index, run in enumerate(run_rows, 1):
        analysed.extend(analyse_training_run(run, reference))
        print(f"[ANALYSE {index:03d}/{len(run_rows)}] {run['training_variant']} "
              f"alpha={run['alpha']:.1f} seed={run['training_seed']}", flush=True)
    per_seed = add_conditional_variants(pd.DataFrame(analysed))
    aggregated = aggregate(per_seed)
    per_seed.to_csv(OUT / "alpha_tradeoff_per_seed.csv", index=False)
    aggregated.to_csv(OUT / "alpha_tradeoff_aggregated.csv", index=False)

    plot_specs = [
        ("lq05_mask", "row_k3_mutual_mst"),
        ("lq1_mask", "row_k3_mutual_mst"),
        ("lq05_conditional", "row_k3_mutual_mst"),
        ("lq1_conditional", "row_k3_mutual_mst"),
        ("lq05_nomask", "row_k3_mutual_no_mst"),
        ("lq1_nomask", "row_k3_mutual_no_mst"),
    ]
    for variant, decoder in plot_specs:
        plot_variant(aggregated, variant, decoder, f"alpha_tradeoff_{variant}_{decoder}")

    primary_variant = "lq1_nomask"
    primary_decoder = "row_k3_mutual_no_mst"
    plot_variant(aggregated, primary_variant, primary_decoder,
                 "alpha_tradeoff_primary_two_panels")
    write_primary_table(aggregated, primary_variant, primary_decoder)

    manifest = {
        "design": "predeclared 2x2 lambda_Q x Q-candidate-mask ablation",
        "alphas": ALPHAS,
        "training_seeds": SEEDS,
        "reused_baseline_runs": 33,
        "new_runs": len(specs),
        "lambda_Q_values": [0.5, 1.0],
        "mask_modes": ["always", "never", "conditional_at_alpha_1_postcomposition"],
        "decoders": ["row_k3_mutual_mst", "row_k3_mutual_no_mst"],
        "primary_analysis": {
            "variant": primary_variant,
            "decoder": primary_decoder,
            "rationale": "constant architecture across alpha; alpha controls only the two loss terms",
        },
        "dataset": str(DATA),
    }
    (OUT / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] {OUT}")


if __name__ == "__main__":
    main()
