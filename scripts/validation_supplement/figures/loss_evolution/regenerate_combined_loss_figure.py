"""Regenerate the single-axis three-loss Section V figure from raw histories."""
from pathlib import Path
import re
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
HERE = PROJECT / "results_section_v_final/figures/loss_evolution"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE = PROJECT / "results_section_v/campaign_v2/relational/main_experiment"
PATTERN = "main_experiment_g211_s1211_t*_l6_h12_a0p80_kd2_kc3_kq2_e300/training_history.csv"
SEED_RE = re.compile(r"_t(\d+)_")
COLORS = {"forecasting": "#1f77b4", "structural": "#d97706", "total": "#2a7f62"}
LABELS = {
    "forecasting": r"$\overline{\mathcal{L}}_{\mathrm{forecast}}$",
    "structural": r"$\overline{\mathcal{L}}_{\mathrm{struct}}$",
    "total": r"$\mathcal{L}_{\mathrm{train}}$",
}


def load_and_aggregate():
    frames = []
    paths = sorted(SOURCE.glob(PATTERN))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five original histories, found {len(paths)}")
    for path in paths:
        match = SEED_RE.search(path.parent.name)
        if not match:
            raise RuntimeError(f"Cannot parse seed from {path}")
        frame = pd.read_csv(path)
        required = ["epoch", "train_task", "train_topology", "train_total"]
        if any(column not in frame for column in required):
            raise RuntimeError(f"Missing loss column in {path}")
        if frame.epoch.tolist() != list(range(1, 301)):
            raise RuntimeError(f"Incomplete epoch range in {path}")
        frame = frame[required].copy()
        frame["training_seed"] = int(match.group(1))
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    grouped = runs.groupby("epoch", sort=True)
    out = pd.DataFrame({"epoch": range(1, 301)})
    for name, column in {"forecasting": "train_task", "structural": "train_topology",
                         "total": "train_total"}.items():
        out[f"{name}_mean"] = grouped[column].mean().to_numpy()
        out[f"{name}_std"] = grouped[column].std(ddof=1).to_numpy()
    out["seed_count"] = grouped.training_seed.nunique().to_numpy()
    return out


def style(font_size):
    available = {font.name for font in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [serif], "font.size": font_size,
        "axes.labelsize": font_size, "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def draw(data, stem, size, font_size, legend_size):
    style(font_size)
    fig, ax = plt.subplots(figsize=size)
    for name in ("forecasting", "structural", "total"):
        ax.plot(data.epoch, data[f"{name}_mean"], color=COLORS[name],
                linewidth=2.1, label=LABELS[name])
    ax.set_xlabel("Epoch", labelpad=4)
    ax.set_ylabel("Loss", labelpad=5)
    ax.set_xlim(1, 300)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
    ax.grid(True, color="#b8b8b8", alpha=0.32, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.05)
    ax.tick_params(direction="out", length=3.8, width=0.85, pad=3)
    ax.legend(loc="upper right", frameon=False, fontsize=legend_size,
              handlelength=2.4, handletextpad=0.55, labelspacing=0.38)
    fig.subplots_adjust(left=0.15 if size[0] < 4 else 0.105,
                        right=0.975, bottom=0.20 if size[0] < 4 else 0.18, top=0.97)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"{stem}.{extension}", bbox_inches="tight",
                    dpi=400 if extension == "png" else None)
    plt.close(fig)


def main():
    data = load_and_aggregate()
    data.to_csv(PROJECT / "results_section_v_final/figures/loss_evolution" / "loss_evolution_combined_aggregated.csv", index=False,
                float_format="%.10g")
    draw(data, "loss_evolution_combined_single_column", (3.50, 2.55), 10.5, 8.8)
    draw(data, "loss_evolution_combined_double_column", (7.16, 3.20), 12.5, 11.5)
    for extension in ("pdf", "svg", "png"):
        destination = PROJECT / f"results_section_v_final/figures/main/{extension}/loss_evolution.{extension}"
        shutil.copy2(HERE / f"loss_evolution_combined_double_column.{extension}", destination)
    shutil.copy2(PROJECT / "results_section_v_final/figures/loss_evolution" / "loss_evolution_combined_aggregated.csv",
                 PROJECT / "results_section_v_final/figures/main/data/loss_evolution.csv")
    print(f"seeds={int(data.seed_count.iloc[0])} epochs={len(data)} output={HERE}")


if __name__ == "__main__":
    main()
