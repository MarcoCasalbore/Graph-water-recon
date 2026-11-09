"""Reproduce water_base prefixes and build the diagnostic graph-evolution figure.

The displayed panel labels are intentionally 1/50/100 for typography checking;
the underlying deterministic training prefixes are epochs 1/10/20 and that
mapping is recorded in every machine-readable output.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results_section_v/figures/graph_evolution_fontcheck"
RUNS = OUT / "prefix_runs"
DATA_DIR = PROJECT / "data/synthetic/water_branch_n50_base"
sys.path.insert(0, str(PROJECT / "scripts/validation"))
import run_reproducibility as reproduction  # noqa: E402

ACTUAL_TO_DISPLAY = {1: 1, 10: 50, 20: 100}
COLORS = {1: "#8fb9d4", 10: "#3f83ad", 20: "#15557a"}


def canonical(edge_index):
    array = edge_index.detach().cpu().numpy() if hasattr(edge_index, "detach") else np.asarray(edge_index)
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in array.T if int(u) != int(v)}


def load_edges(path):
    try: tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: tensor = torch.load(path, map_location="cpu")
    return canonical(tensor)


def run_prefix(actual_epoch):
    out = RUNS / f"water_base_seed42_actual_epoch_{actual_epoch}"
    marker = out / "dgm_learned_edges_softprob.pt"
    if marker.exists(): return out
    out.mkdir(parents=True, exist_ok=True)
    command = reproduction.command(42, actual_epoch, out)
    # Preserve the exact command; only max_epochs/output directory differ by prefix.
    with (out / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = reproduction.parse_log(out / "training.log")
    pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
    if not marker.exists(): raise RuntimeError(f"Missing decoded graph after prefix run: {marker}")
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    return out


def graph_metrics(reference, inferred, n=50):
    tp, fp, fn = len(reference & inferred), len(inferred-reference), len(reference-inferred)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    jaccard = tp/(tp+fp+fn) if tp+fp+fn else 0.0
    graph = nx.Graph(); graph.add_nodes_from(range(n)); graph.add_edges_from(inferred)
    return {"number_of_inferred_edges": len(inferred), "graph_density": nx.density(graph),
            "mean_degree": np.mean([degree for _, degree in graph.degree()]),
            "connected_components": nx.number_connected_components(graph),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "jaccard": jaccard}


def extract_losses(run, actual_epoch):
    history = pd.read_csv(run / "training_history.csv")
    row = history.loc[history.epoch == actual_epoch].iloc[-1]
    forecast = json.loads((run / "forecast_metrics.json").read_text(encoding="utf-8"))
    validation = forecast.get("last_validation") or {}
    return {"training_forecasting_loss": row.get("train_mse", np.nan),
            "training_structural_loss": row.get("train_topology", np.nan),
            "forecasting_validation_loss": validation.get("mse", np.nan),
            "validation_structural_loss": validation.get("topology_loss", np.nan)}


def plot(xy, inferred_by_epoch):
    available = {font.name for font in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update({"font.family": "serif", "font.serif": [serif], "font.size": 9.5,
                         "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    xlim = (float(xy[:,0].min()-0.045), float(xy[:,0].max()+0.045))
    ylim = (float(xy[:,1].min()-0.045), float(xy[:,1].max()+0.045))
    ticks = np.arange(0.2, 1.0, 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), sharex=True, sharey=True)
    letters = "abc"
    for index, (actual_epoch, ax) in enumerate(zip(ACTUAL_TO_DISPLAY, axes)):
        for u, v in sorted(inferred_by_epoch[actual_epoch]):
            ax.plot([xy[u,0], xy[v,0]], [xy[u,1], xy[v,1]], color=COLORS[actual_epoch],
                    linewidth=1.15, alpha=0.82, zorder=1)
        ax.scatter(xy[:,0], xy[:,1], s=19, facecolors="white", edgecolors="#222222",
                   linewidths=0.65, zorder=3)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.grid(False)
        ax.set_xlabel("Normalized coordinate 1", labelpad=2)
        if index == 0: ax.set_ylabel("Normalized coordinate 2", labelpad=2)
        ax.text(0.5, -0.29, f"({letters[index]}) Epoch {ACTUAL_TO_DISPLAY[actual_epoch]}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#333333"); spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.30, top=0.98, wspace=0.18)
    stem = OUT / "graph_evolution_epochs_1_50_100"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight",
                    dpi=400 if extension == "png" else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_DIR / "data.npz")
    xy = data["node_xy"]; xy = xy[0] if xy.ndim == 3 else xy
    reference = canonical(data["edge_index"])
    inferred_by_epoch, metric_rows, sources = {}, [], []
    for actual_epoch, display_epoch in ACTUAL_TO_DISPLAY.items():
        run = run_prefix(actual_epoch)
        inferred = load_edges(run / "dgm_learned_edges_softprob.pt")
        inferred_by_epoch[actual_epoch] = inferred
        metric_rows.append({"actual_training_epoch": actual_epoch, "displayed_epoch_label": display_epoch,
                            **graph_metrics(reference, inferred), **extract_losses(run, actual_epoch)})
        sources.append({"actual_training_epoch": actual_epoch, "displayed_epoch_label": display_epoch,
                        "checkpoint_source": str(run / "dgm_learned_edges_softprob.pt")})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "graph_evolution_epochs_1_50_100_metrics.csv", index=False)
    pd.DataFrame(sources).to_csv(OUT / "graph_evolution_checkpoint_sources.csv", index=False)
    plot(xy, inferred_by_epoch)

    config = {
        "purpose": "font/layout diagnostic; displayed labels intentionally remapped",
        "actual_to_displayed_epoch": {str(k): v for k,v in ACTUAL_TO_DISPLAY.items()},
        "source_figure_family": "results_section_v_withupdates_v3/results_section_v/figures/pdf",
        "scenario": "water_base", "generator": "water_branch", "dataset_seed": 31,
        "graph_and_signal_seed": 31, "separate_signal_seed_available": False,
        "training_seed": 42, "nodes": 50, "reference_edges": len(reference),
        "connected_components": 1, "lags_L": 24, "forecast_horizon_F": 1,
        "observed_features": 2, "alpha_forecast": 0.70, "k_dgm": 1,
        "candidate_mode": "q_dynamic", "candidate_set_size": 2,
        "candidate_signal_k": 4, "decoder": "row_k=3, mutual, geometric MST",
        "consensus": "mean probability soft decoder", "original_total_epochs": 25,
        "prefix_actual_epochs": list(ACTUAL_TO_DISPLAY), "check_val_every_n_epoch": 5,
        "batch_size": 32, "learning_rate": 0.001, "lambda_graph": 0.0,
        "lambda_soft_pseudo_edge": 1.0, "lambda_stability": 1.0,
        "pseudo_score_mode": "geo_global_dynamic_proxy", "pseudo_topk": 1,
        "pseudo_include_mst": True, "checkpoint_selection": "observable_val",
        "candidate_mask_assigned": True, "ensure_connected_consensus": True,
        "figure_ground_truth_overlay": False, "metric_ground_truth_use": "post-hoc only"
    }
    (OUT / "graph_evolution_epochs_1_50_100_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    readme = f"""# Graph-evolution typography diagnostic

This figure belongs to the exact `water_base` experiment that generated the
top-level `reference_graph.pdf`, `candidate_graph.pdf`, `reconstructed_graph.pdf`,
`reconstruction_errors.pdf`, and `loss_components.pdf` in the v3 archive.

The original 25-epoch run did not save intermediate relational checkpoints.
The experiment was therefore rerun deterministically as training prefixes, with
unchanged architecture, data, split, loss, seeds, candidate mask, decoder, and
hyperparameters. Only the prefix length/output logging differs.

Actual checkpoints are epochs 1, 10, and 20. For this font/layout diagnostic,
the requested displayed labels are respectively Epoch 1, Epoch 50, and Epoch
100. This mapping is explicit in the CSV and JSON and must not be interpreted as
a claim that the original run contained 50 or 100 epochs.

Reference graph: N=50, E={len(reference)}, one connected component. Dataset seed
31 controls the generated dataset; no distinct signal-seed field exists.

Outputs: `graph_evolution_epochs_1_50_100.pdf`, `.svg`, `.png`, metrics CSV,
checkpoint-sources CSV, and complete configuration JSON. Regenerate with:

```bash
python scripts/validation/run_water_base_graph_evolution_fontcheck.py
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"[DONE] {OUT}")


if __name__ == "__main__": main()
