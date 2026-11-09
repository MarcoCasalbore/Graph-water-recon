"""Create paper-ready temporal explainability tables and figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
sys.path.insert(0, str(RESULTS_ROOT))
from scripts.plot_style import COLORS, apply_style, save_figure


RAW = RESULTS_ROOT / "raw/temporal_explainability"
BALANCED = RESULTS_ROOT / "raw/temporal_balanced_joint/balanced_joint_results.csv"


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


def enrich() -> pd.DataFrame:
    base = pd.read_csv(RAW / "temporal_explainability_results.csv")
    balanced = pd.read_csv(BALANCED).rename(columns={
        "test_mse_feature_mean": "balanced_joint_mse",
        "test_mse": "balanced_reported_task_mse",
        "f1": "balanced_f1", "task_score": "balanced_task_score",
        "eval_0p5": "balanced_eval_0p5", "J_obs": "balanced_J_obs",
    })
    keep = ["lags", "horizon", "balanced_joint_mse", "balanced_reported_task_mse",
            "balanced_f1", "balanced_task_score", "balanced_eval_0p5", "balanced_J_obs"]
    frame = base.merge(balanced[keep], on=["lags", "horizon"], validate="one_to_one")

    no_edge, gt_sum = [], []
    for row in frame.itertuples():
        fixed = pd.read_csv(RAW / f"l{row.lags}_h{row.horizon}_seed42/fixed_graphs/oracle_graph_loss_results.csv").set_index("graph_name")
        no_edge.append(float(fixed.loc["no_edges", "test_mse"]) / 2.0)
        gt_sum.append(float(fixed.loc["ground_truth", "test_mse"]) / 2.0)
    frame["neighbor_only_no_edge_mse"] = no_edge
    frame["neighbor_only_gt_mse"] = gt_sum
    frame["neighbor_dependence_gain"] = (
        frame["neighbor_only_no_edge_mse"] - frame["neighbor_only_gt_mse"]
    ) / frame["neighbor_only_no_edge_mse"]
    frame["local_history_sufficiency"] = 1.0 - frame["node_mlp_mse"] / frame["persistence_mse"]
    frame["balanced_joint_gap_vs_node"] = (
        frame["balanced_joint_mse"] - frame["node_mlp_mse"]
    ) / frame["node_mlp_mse"]
    return frame.sort_values(["lags", "horizon"])


def heatmap(frame: pd.DataFrame, value: str, stem: str, cbar_label: str, cmap: str, vmin=None, vmax=None) -> None:
    pivot = frame.pivot(index="lags", columns="horizon", values=value).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_xlabel("Forecast horizon H")
    ax.set_ylabel("History length L")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value_ij = pivot.iloc[i, j]
            ax.text(j, i, f"{value_ij:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value_ij) > 0.65 * np.nanmax(np.abs(pivot.to_numpy())) else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.grid(False)
    save_figure(fig, RESULTS_ROOT / "figures", stem)
    plt.close(fig)
    frame[["lags", "horizon", value]].to_csv(RESULTS_ROOT / f"figures/data/{stem}.csv", index=False)


def plot_balanced(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=False)
    panels = [
        ("balanced_task_score", "TaskScore", COLORS["forecast"]),
        ("balanced_f1", "Graph F1 (post-hoc)", COLORS["topology"]),
        ("balanced_eval_0p5", r"$Eval_{0.5}$ (post-hoc)", COLORS["objective"]),
    ]
    for ax, (column, ylabel, color) in zip(axes, panels):
        for lags, group in frame.groupby("lags"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[column], marker="o", color=color,
                    alpha=0.45 + 0.5 * lags / frame["lags"].max(), label=f"L={lags}")
        ax.set_xlabel("Forecast horizon H"); ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(frame["horizon"].unique()))
    axes[0].legend(frameon=False)
    save_figure(fig, RESULTS_ROOT / "figures", "temporal_balanced_evaluation")
    plt.close(fig)
    frame[["lags", "horizon", "balanced_task_score", "balanced_f1", "balanced_eval_0p5"]].to_csv(
        RESULTS_ROOT / "figures/data/temporal_balanced_evaluation.csv", index=False)


def plot_model_comparison(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values("horizon_lag_ratio")
    labels = [f"L{int(row.lags)}/H{int(row.horizon)}" for row in ordered.itertuples()]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(x, ordered["node_mlp_mse"], marker="o", label="Node-only MLP", color="#222222")
    ax.plot(x, ordered["balanced_joint_mse"], marker="s", label="Balanced joint DGM", color=COLORS["learned"])
    ax.plot(x, ordered["gt_graph_mse"], marker="^", label="Fixed GT graph (oracle)", color=COLORS["topology"])
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("Temporal setting, ordered by H/L")
    ax.set_ylabel("Feature-mean normalized test MSE")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, RESULTS_ROOT / "figures", "temporal_model_comparison")
    plt.close(fig)
    ordered[["lags", "horizon", "horizon_lag_ratio", "node_mlp_mse", "balanced_joint_mse", "gt_graph_mse"]].to_csv(
        RESULTS_ROOT / "figures/data/temporal_model_comparison.csv", index=False)


def main() -> None:
    apply_style()
    frame = enrich()
    frame.to_csv(RESULTS_ROOT / "tables/table_9_temporal_explainability.csv", index=False)
    summary_columns = [
        "lags", "horizon", "persistence_mse", "node_mlp_mse", "local_history_sufficiency",
        "gt_graph_mse", "random_graph_mse", "topology_selectivity_gt_vs_random",
        "neighbor_only_no_edge_mse", "neighbor_only_gt_mse", "neighbor_dependence_gain",
        "balanced_joint_mse", "balanced_f1", "balanced_task_score", "balanced_eval_0p5",
    ]
    write_table(9, "temporal_explainability", frame[summary_columns])
    write_table(10, "balanced_temporal_evaluation", frame[[
        "lags", "horizon", "balanced_J_obs", "balanced_reported_task_mse",
        "balanced_task_score", "balanced_f1", "balanced_eval_0p5",
    ]])
    heatmap(frame, "local_history_sufficiency", "temporal_local_history_sufficiency",
            "Gain over persistence", "Blues", vmin=0, vmax=1)
    heatmap(frame, "topology_selectivity_gt_vs_random", "temporal_topology_selectivity",
            "GT gain over random graph", "YlOrRd", vmin=0, vmax=0.5)
    heatmap(frame, "neighbor_dependence_gain", "temporal_neighbor_dependence",
            "GT gain when self-history is removed", "Purples", vmin=0, vmax=1)
    plot_balanced(frame)
    plot_model_comparison(frame)
    print(frame[summary_columns].to_string(index=False))
    print("[OK] temporal explainability tables and figures written")


if __name__ == "__main__":
    main()
