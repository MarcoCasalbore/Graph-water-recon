"""Aggregate three-seed easy/hard temporal corner results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
sys.path.insert(0, str(RESULTS_ROOT))
from scripts.plot_style import COLORS, apply_style, save_figure


def markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def main() -> None:
    grid = pd.read_csv(RESULTS_ROOT / "tables/table_9_temporal_explainability.csv")
    seed42 = []
    for regime, lags, horizon in [("easy", 24, 1), ("hard", 4, 12)]:
        row = grid[(grid.lags == lags) & (grid.horizon == horizon)].iloc[0]
        seed42.append({
            "regime": regime, "lags": lags, "horizon": horizon, "seed": 42,
            "node_mlp_mse": row.node_mlp_mse, "balanced_joint_mse": row.balanced_joint_mse,
            "joint_gain_vs_node": (row.node_mlp_mse-row.balanced_joint_mse)/row.node_mlp_mse,
            "f1": row.balanced_f1, "task_score": row.balanced_task_score,
            "eval_0p5": row.balanced_eval_0p5,
        })
    extra = pd.read_csv(RESULTS_ROOT / "raw/temporal_corner_seeds/corner_seed_results.csv")
    rows = pd.concat([pd.DataFrame(seed42), extra.drop(columns=["node_seconds", "joint_seconds"])], ignore_index=True)
    summary = rows.groupby(["regime", "lags", "horizon"], as_index=False).agg(
        node_mlp_mse_mean=("node_mlp_mse", "mean"), node_mlp_mse_std=("node_mlp_mse", "std"),
        balanced_joint_mse_mean=("balanced_joint_mse", "mean"), balanced_joint_mse_std=("balanced_joint_mse", "std"),
        joint_gain_mean=("joint_gain_vs_node", "mean"), joint_gain_std=("joint_gain_vs_node", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        eval_0p5_mean=("eval_0p5", "mean"), eval_0p5_std=("eval_0p5", "std"))

    stem = RESULTS_ROOT / "tables/table_11_temporal_corner_multiseed"
    rows.to_csv(RESULTS_ROOT / "figures/data/temporal_corner_seed_comparison.csv", index=False)
    summary.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    stem.with_suffix(".tex").write_text(summary.to_latex(index=False, escape=True), encoding="utf-8")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, regime in zip(axes, ["easy", "hard"]):
        group = rows[rows.regime == regime].sort_values("seed")
        x = range(len(group))
        ax.plot(x, group.node_mlp_mse, marker="o", color="#222222", label="Node-only")
        ax.plot(x, group.balanced_joint_mse, marker="s", color=COLORS["learned"], label="Balanced joint")
        ax.set_xticks(list(x), [str(seed) for seed in group.seed])
        ax.set_xlabel("Training seed"); ax.set_ylabel("Feature-mean normalized MSE")
        ax.text(0.03, 0.95, f"{regime}: L={int(group.lags.iloc[0])}, H={int(group.horizon.iloc[0])}",
                transform=ax.transAxes, va="top")
    axes[0].legend(frameon=False)
    save_figure(fig, RESULTS_ROOT / "figures", "temporal_corner_seed_comparison")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
