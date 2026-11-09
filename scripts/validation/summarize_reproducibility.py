"""Aggregate the completed three-seed Section V runs and plot saved histories."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
sys.path.insert(0, str(RESULTS_ROOT))
from scripts.plot_style import COLORS, apply_style, save_figure


RUNS = {
    42: PROJECT_ROOT / "outputs/final_observable_hyperparam_grid/water_base_geo_global_dynamic_proxy_a0p70_q1_ck2_stab1p000_e25",
    43: RESULTS_ROOT / "raw/section_v_main_seed43_e25",
    44: RESULTS_ROOT / "raw/section_v_main_seed44_e25",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def edges(path: Path) -> set[tuple[int, int]]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in value.numpy().T if int(u) != int(v)}


def markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def write_table(number: int, slug: str, frame: pd.DataFrame) -> None:
    stem = RESULTS_ROOT / "tables" / f"table_{number}_{slug}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown(frame), encoding="utf-8")
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")


def collect() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, set[tuple[int, int]]]]:
    result_rows, decoder_rows = [], []
    soft_edges = {}
    for seed, directory in RUNS.items():
        forecast = read_json(directory / "forecast_metrics.json")
        selected = forecast["best_observable_validation"]
        comparison = read_json(directory / "graph_comparison_softprob/comparison_summary.json")
        metrics = comparison["edge_recovery_undirected"]
        graph = comparison["learned_undirected"]
        result_rows.append({
            "seed": seed, "selected_epoch": forecast["selected_epoch"],
            "J_obs": selected["observable_val_score"], "val_mse": selected["mse"],
            "val_LQ": selected["soft_pseudo_edge_loss"], "test_mse": forecast["test"]["mse"],
            "edges": graph["num_edges"], "components": graph["num_connected_components_undirected"],
            **{key: metrics[key] for key in ("precision", "recall", "f1", "jaccard", "tp", "fp", "fn")},
        })
        for label, filename, folder in [
            ("sample-frequency consensus", "dgm_learned_edges_consensus.pt", "graph_comparison"),
            ("mean-probability row-k mutual + MST", "dgm_learned_edges_softprob.pt", "graph_comparison_softprob"),
        ]:
            summary = read_json(directory / folder / "comparison_summary.json")
            metric = summary["edge_recovery_undirected"]
            stats = summary["learned_undirected"]
            decoder_rows.append({"seed": seed, "decoder_or_consensus": label,
                                 "edges": stats["num_edges"], "components": stats["num_connected_components_undirected"],
                                 **{key: metric[key] for key in ("precision", "recall", "f1", "jaccard")}})
        soft_edges[seed] = edges(directory / "dgm_learned_edges_softprob.pt")
    return pd.DataFrame(result_rows), pd.DataFrame(decoder_rows), soft_edges


def plot_histories() -> None:
    histories = []
    for seed in (43, 44):
        frame = pd.read_csv(RUNS[seed] / "training_history.csv")
        frame["seed"] = seed
        histories.append(frame)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(RESULTS_ROOT / "figures/data/loss_components.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    panels = [("train_mse", "Forecast loss"), ("train_lq", r"Observable loss $L_Q$"),
              ("train_stability", "Stability loss")]
    for ax, (column, ylabel) in zip(axes, panels):
        for seed, group in history.groupby("seed"):
            ax.plot(group["epoch"], group[column], label=f"Seed {seed}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    save_figure(fig, RESULTS_ROOT / "figures", "loss_components")
    plt.close(fig)


def plot_edge_stability(edge_sets: dict[int, set[tuple[int, int]]]) -> pd.DataFrame:
    pair_rows = []
    for first, second in combinations(sorted(edge_sets), 2):
        a, b = edge_sets[first], edge_sets[second]
        pair_rows.append({"seed_a": first, "seed_b": second, "edge_jaccard": len(a & b) / len(a | b)})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(RESULTS_ROOT / "figures/data/edge_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    labels = [f"{row.seed_a}-{row.seed_b}" for row in pairs.itertuples()]
    ax.bar(labels, pairs["edge_jaccard"], color=COLORS["learned"])
    ax.set_xlabel("Training-seed pair"); ax.set_ylabel("Edge-set Jaccard")
    ax.set_ylim(0, 1)
    save_figure(fig, RESULTS_ROOT / "figures", "edge_stability")
    plt.close(fig)
    return pairs


def main() -> None:
    apply_style()
    results, decoders, edge_sets = collect()
    mean_row = {"seed": "mean ± std", "selected_epoch": "—"}
    for column in results.columns[2:]:
        mean_row[column] = f"{results[column].mean():.4f} ± {results[column].std(ddof=1):.4f}"
    table4 = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
    write_table(4, "main_quantitative_results", table4)

    aggregate = decoders.groupby("decoder_or_consensus", as_index=False).agg(
        edges_mean=("edges", "mean"), edges_std=("edges", "std"),
        components_mean=("components", "mean"), precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        jaccard_mean=("jaccard", "mean"))
    write_table(6, "decoder_comparison", aggregate)
    write_table(7, "consensus_comparison", decoders)
    plot_histories()
    pairs = plot_edge_stability(edge_sets)

    manifest_path = RESULTS_ROOT / "experiment_manifest.csv"
    manifest = pd.DataFrame([
        {"experiment_id": RUNS[seed].name, "status": "completed", "scenario": "water_base",
         "dataset_seed": 31, "training_seed": seed, "selection": "observable_val",
         "source": str(RUNS[seed].relative_to(PROJECT_ROOT)), "oracle": False}
        for seed in sorted(RUNS)
    ] + [{"experiment_id": "section_v_smoke_seed42", "status": "completed_smoke_only",
          "scenario": "water_base", "dataset_seed": 31, "training_seed": 42,
          "selection": "observable_val", "source": "results_section_v/raw/section_v_smoke_seed42", "oracle": False}])
    manifest.to_csv(manifest_path, index=False)
    print("[OK] three-seed summary written")
    print(results.to_string(index=False))
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
