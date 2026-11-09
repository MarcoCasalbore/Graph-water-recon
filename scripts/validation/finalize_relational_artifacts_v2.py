"""Create final comparison figures and one manifest from completed runs."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
ROOT = RESULTS / "campaign_v2/relational"
RAW, FIG = ROOT / "raw_metrics", ROOT / "figures"
DATA = FIG / "data"
sys.path.insert(0, str(RESULTS))
from scripts.plot_style import COLORS, apply_style, save_figure  # noqa: E402


def bar_plot(frame, label_col, stem, xlabel):
    frame = frame.sort_values("f1_mean")
    frame.to_csv(DATA / f"{stem}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(frame))
    ax.bar(x, frame.f1_mean, yerr=frame.f1_std, capsize=3,
           color=COLORS["topology"], alpha=.88)
    ax.set_xticks(x, frame[label_col], rotation=22, ha="right")
    ax.set(xlabel=xlabel, ylabel="Graph F1", ylim=(.65, .93))
    save_figure(fig, FIG, stem)
    plt.close(fig)


def main():
    apply_style()
    DATA.mkdir(parents=True, exist_ok=True)
    archive = np.load(RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211/data.npz")
    pairs = {tuple(sorted((int(u), int(v)))) for u, v in archive["edge_index"].T if u != v}
    degree = np.zeros(len(archive["node_xy"]), dtype=int)
    for u, v in pairs:
        degree[u] += 1
        degree[v] += 1
    values, counts = np.unique(degree, return_counts=True)
    pd.DataFrame({"degree": values, "node_count": counts}).to_csv(
        DATA / "degree_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.bar(values, counts, color=COLORS["learned"], width=.72)
    ax.set(xlabel="Node degree", ylabel="Number of nodes", xticks=values)
    save_figure(fig, FIG, "degree_distribution")
    plt.close(fig)

    bar_plot(pd.read_csv(ROOT / "tables/table_9_decoder_comparison.csv"),
             "decoder", "decoder_comparison", "Decoder")
    bar_plot(pd.read_csv(ROOT / "tables/table_10_consensus_comparison.csv"),
             "consensus", "consensus_comparison", "Consensus method")

    joint = pd.read_csv(RAW / "main_runs.csv")
    joint["forecast_mse"] = .5 * (joint.test_pressure_mse + joint.test_demand_mse)
    node = pd.read_csv(RAW / "main_node_runs.csv")
    paired = joint[["training_seed", "forecast_mse"]].merge(
        node[["training_seed", "test_mse"]], on="training_seed")
    paired = paired.rename(columns={"test_mse": "node_only_mse"})
    paired.to_csv(DATA / "main_seed_forecasting_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(paired.training_seed, paired.node_only_mse, "o-", label="Node-only",
            color=COLORS["candidate"])
    ax.plot(paired.training_seed, paired.forecast_mse, "s-", label="Joint",
            color=COLORS["forecast"])
    ax.set(xlabel="Training seed", ylabel="Test forecast MSE", xticks=paired.training_seed)
    ax.legend(frameon=False)
    save_figure(fig, FIG, "main_seed_forecasting_comparison")
    plt.close(fig)

    parts = []
    for path in sorted(RAW.glob("*_runs.csv")):
        frame = pd.read_csv(path)
        if {"run_id", "status"}.issubset(frame.columns):
            frame.insert(0, "source_file", path.name)
            parts.append(frame)
    manifest = pd.concat(parts, ignore_index=True, sort=False)
    manifest.to_csv(ROOT / "experiment_manifest.csv", index=False)
    print(f"[OK] figures=4 manifest_rows={len(manifest)}")


if __name__ == "__main__":
    main()
