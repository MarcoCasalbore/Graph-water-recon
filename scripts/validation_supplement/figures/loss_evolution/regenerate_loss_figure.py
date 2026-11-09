"""Regenerate the Section V loss figures from original per-epoch CSV files.

The script aggregates the five 300-epoch main-experiment training seeds.  It
never reads a previously exported figure and never plots the stability loss.
"""
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
HERE = PROJECT / "results_section_v_final/figures/loss_evolution"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT = (
    PROJECT
    / "results_section_v/campaign_v2/relational/main_experiment"
)
PATTERN = "main_experiment_g211_s1211_t*_l6_h12_a0p80_kd2_kc3_kq2_e300/training_history.csv"
SEED_RE = re.compile(r"_t(?P<seed>\d+)_")
DATA_CSV = PROJECT / "results_section_v_final/figures/loss_evolution" / "loss_evolution_aggregated.csv"

LOSS_COLUMNS = {
    "forecasting": "train_task",
    "structural": "train_topology",
    "total": "train_total",
}
COLORS = {
    "forecasting": "#1f77b4",
    "structural": "#d97706",
    "total": "#2a7f62",
}


def load_original_histories():
    paths = sorted(SOURCE_ROOT.glob(PATTERN))
    if not paths:
        raise FileNotFoundError(f"No training histories matched {SOURCE_ROOT / PATTERN}")
    frames = []
    source_rows = []
    for path in paths:
        match = SEED_RE.search(path.parent.name)
        if not match:
            raise ValueError(f"Cannot parse training seed from {path}")
        seed = int(match.group("seed"))
        frame = pd.read_csv(path)
        required = {"epoch", "alpha", *LOSS_COLUMNS.values(), "train_w_task",
                    "train_w_graph", "train_w_topology", "train_w_stability"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        if frame.epoch.tolist() != list(range(1, 301)):
            raise ValueError(f"Expected exactly epochs 1..300 in {path}")
        if frame[list(LOSS_COLUMNS.values())].isna().any().any():
            raise ValueError(f"NaN in plotted losses: {path}")
        frame = frame[["epoch", "alpha", *LOSS_COLUMNS.values(), "train_w_task",
                       "train_w_graph", "train_w_topology", "train_w_stability"]].copy()
        frame["training_seed"] = seed
        frames.append(frame)
        source_rows.append({"training_seed": seed, "source_path": str(path)})
    all_runs = pd.concat(frames, ignore_index=True)
    if all_runs.groupby("epoch").training_seed.nunique().nunique() != 1:
        raise ValueError("Seed coverage changes across epochs")
    return all_runs, pd.DataFrame(source_rows)


def aggregate(all_runs):
    grouped = all_runs.groupby("epoch", sort=True)
    result = pd.DataFrame({"epoch": sorted(all_runs.epoch.unique())})
    for name, column in LOSS_COLUMNS.items():
        result[f"{name}_mean"] = grouped[column].mean().to_numpy()
        result[f"{name}_std"] = grouped[column].std(ddof=1).to_numpy()
    result["seed_count"] = grouped.training_seed.nunique().to_numpy()
    return result


def configure_style(font_size):
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [selected],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return selected


def draw(aggregated, names, stem, width, height, font_size, wspace):
    configure_style(font_size)
    fig, axes = plt.subplots(1, len(names), figsize=(width, height), sharex=True)
    axes = np.atleast_1d(axes)
    labels = {
        "forecasting": "Forecasting loss",
        "structural": "Structural loss",
        "total": "Total loss",
    }
    for ax, name in zip(axes, names):
        epoch = aggregated.epoch.to_numpy(dtype=float)
        mean = aggregated[f"{name}_mean"].to_numpy(dtype=float)
        std = aggregated[f"{name}_std"].to_numpy(dtype=float)
        color = COLORS[name]
        ax.plot(epoch, mean, color=color, linewidth=2.0)
        ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.13,
                        linewidth=0)
        ax.set_xlabel("Epoch", labelpad=2.5)
        ax.set_ylabel(labels[name], labelpad=3.0)
        ax.set_xlim(1, 300)
        ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
        ax.grid(True, color="#b8b8b8", alpha=0.28, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    fig.subplots_adjust(left=0.12 if width < 4 else 0.075,
                        right=0.985, bottom=0.25 if width < 4 else 0.19,
                        top=0.97, wspace=wspace)
    for suffix in ("pdf", "svg", "png"):
        dpi = 400 if suffix == "png" else None
        fig.savefig(HERE / f"{stem}.{suffix}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    all_runs, sources = load_original_histories()
    aggregated = aggregate(all_runs)
    aggregated.to_csv(DATA_CSV, index=False, float_format="%.10g")
    sources.to_csv(PROJECT / "results_section_v_final/figures/loss_evolution" / "loss_evolution_sources.csv", index=False)

    # IEEE-like widths: 3.5 in single column and 7.16 in double column.
    draw(aggregated, ["forecasting", "structural"],
         "loss_evolution_two_panel_single_column", 3.50, 1.72, 7.5, 0.62)
    draw(aggregated, ["forecasting", "structural"],
         "loss_evolution_two_panel_double_column", 7.16, 2.35, 10.0, 0.34)
    draw(aggregated, ["forecasting", "structural", "total"],
         "loss_evolution_three_panel_single_column", 3.50, 1.72, 6.6, 0.72)
    draw(aggregated, ["forecasting", "structural", "total"],
         "loss_evolution_three_panel_double_column", 7.16, 2.35, 9.0, 0.40)

    # Audit the exact logged total against its logged weighted components.
    component_sum = (all_runs.train_w_task + all_runs.train_w_graph
                     + all_runs.train_w_topology + all_runs.train_w_stability)
    residual = float((all_runs.train_total - component_sum).abs().max())
    print(f"sources={len(sources)} epochs={len(aggregated)}")
    print(f"total_vs_logged_components_max_abs_residual={residual:.3e}")
    print(f"output={HERE}")


if __name__ == "__main__":
    main()
