"""Build Section V tables and figures from saved, non-oracle synthetic runs.

This script never trains a model. It reads immutable datasets and existing run
artifacts, then exports plot data separately from PDF/SVG/PNG figures.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
sys.path.insert(0, str(RESULTS_ROOT))

from scripts.plot_style import COLORS, apply_style, save_figure


DATASET_DIRS = {
    "simple": PROJECT_ROOT / "data/synthetic/simple_water_n50",
    "water_base": PROJECT_ROOT / "data/synthetic/water_branch_n50_base",
    "water_bridges": PROJECT_ROOT / "data/synthetic/water_branch_n50_bridges",
    "water_loopy": PROJECT_ROOT / "data/synthetic/water_branch_n50_loopy",
}
GRID_CSV = PROJECT_ROOT / "outputs/final_observable_hyperparam_grid/grid_results.csv"
MAIN_RUN = "water_base_geo_global_dynamic_proxy_a0p70_q1_ck2_stab1p000_e25"
MAIN_RUN_DIR = PROJECT_ROOT / "outputs/final_observable_hyperparam_grid" / MAIN_RUN


def canonical_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    if edge_index.size == 0:
        return set()
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.T
        if int(u) != int(v)
    }


def load_torch_edges(path: Path) -> set[tuple[int, int]]:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    return canonical_edges(tensor.detach().cpu().numpy())


def graph_stats(n: int, edges: set[tuple[int, int]]) -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    return {
        "nodes": n,
        "edges": len(edges),
        "density": nx.density(graph),
        "mean_degree": degrees.mean(),
        "min_degree": degrees.min(),
        "max_degree": degrees.max(),
        "components": nx.number_connected_components(graph),
    }


def edge_metrics(reference: set[tuple[int, int]], learned: set[tuple[int, int]]) -> dict:
    tp = len(reference & learned)
    fp = len(learned - reference)
    fn = len(reference - learned)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "jaccard": jaccard}


def write_table(number: int, slug: str, frame: pd.DataFrame) -> None:
    stem = f"table_{number}_{slug}"
    table_dir = RESULTS_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_dir / f"{stem}.csv", index=False)
    (table_dir / f"{stem}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (table_dir / f"{stem}.tex").write_text(
        frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )


def export_plot_data(stem: str, frame: pd.DataFrame) -> None:
    out = RESULTS_ROOT / "figures/data"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}.csv", index=False)


def draw_edges(ax, xy, edges, color, width=1.2, alpha=0.8, linestyle="-"):
    for u, v in sorted(edges):
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], color=color,
                lw=width, alpha=alpha, ls=linestyle, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=18, c="white", edgecolors="#222222",
               linewidths=0.6, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("Normalized coordinate 1")
    ax.set_ylabel("Normalized coordinate 2")
    ax.grid(False)


def dataset_table() -> pd.DataFrame:
    rows = []
    for name, directory in DATASET_DIRS.items():
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        data = np.load(directory / "data.npz")
        edges = canonical_edges(data["edge_index"])
        stats = graph_stats(int(meta["num_nodes"]), edges)
        generator = meta.get("generator", {})
        rows.append({
            "scenario": name,
            "generator": generator.get("type", "unknown"),
            "seed": meta.get("seed"),
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "density": stats["density"],
            "mean_degree": stats["mean_degree"],
            "components": stats["components"],
            "timesteps": int(data["pressure"].shape[0]),
            "variables_per_node": 2,
            "train_val_test": "503/109/108",
        })
    return pd.DataFrame(rows)


def modules_table() -> pd.DataFrame:
    rows = [
        ("Graph/data generator", "scripts/synthetic/generate_synthetic_water.py", "generate", "No", "Generation only", "Writes edge_index GT; model does not receive it"),
        ("Windowing/scaler", "src/dataset/dataset_epanet.py", "EpanetDataModule", "No", "Train/val/test", "Scaler fitted on train split only"),
        ("Candidate builder", "scripts/experiments/run_train_synthetic.py", "build_q_dynamic_candidate_mask", "No", "Before training", "Coordinates and observed series only"),
        ("DGM pairwise scorer/sampler", "src/model/dgm.py", "DGM_d", "Yes", "Training and inference", "No GT"),
        ("Forecasting model", "src/model/dgm.py", "DGMModel", "Yes", "Training and inference", "Forecast targets only"),
        ("Observable prior", "src/train_dgm.py", "_build_pseudo_edge_target", "No", "Loss construction", "No GT"),
        ("Loss/training", "src/train_dgm.py", "train_one_epoch", "Yes", "Training", "No GT in selected workflow"),
        ("Frequency consensus", "scripts/experiments/run_train_synthetic.py", "export_consensus_graph", "No", "Post-processing", "No GT"),
        ("Soft decoder", "scripts/experiments/run_train_synthetic.py", "decode_observable_score", "No", "Post-processing", "No GT; row-k, mutual/union, optional geometric MST"),
        ("Evaluation", "scripts/water/compare_graphs_epanet.py", "main", "No", "Post-hoc only", "Uses GT for metrics/figures"),
        ("Section V plotting", "scripts/validation/build_section_v.py", "main", "No", "Post-hoc only", "GT used only in diagnostic plots/metrics"),
    ]
    return pd.DataFrame(rows, columns=["module", "file", "class_or_function", "trainable", "stage", "ground_truth_use"])


def config_table() -> pd.DataFrame:
    values = [
        ("scenario", "water_base"), ("dataset seed", 31), ("training seed", 42),
        ("nodes", 50), ("timesteps", 720), ("lags L", 24), ("horizon H", 1),
        ("split", "503/109/108"), ("batch size", 32), ("epochs", 25),
        ("optimizer", "Adam"), ("learning rate", 0.001), ("ffun/gfun", "gcn/gcn"),
        ("embedding dimension", 32), ("DGM k", 1), ("candidate mode", "q_dynamic"),
        ("candidate k", 2), ("observable prior", "geo_global_dynamic_proxy"),
        ("alpha forecast", 0.70), ("lambda Q", 1.0), ("lambda stability", 1.0),
        ("checkpoint criterion", "observable validation objective"),
        ("soft decoder", "row_k=3, mutual, geometric MST"),
    ]
    return pd.DataFrame(values, columns=["parameter", "value"])


def results_tables(grid: pd.DataFrame, reference: set[tuple[int, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = grid.loc[grid["run"] == MAIN_RUN].copy()
    selected.insert(0, "selection_role", "main non-oracle (best J_obs within water_base)")
    main_columns = ["selection_role", "run", "j_val", "val_mse", "val_lq", "test_mse",
                    "f1", "precision", "recall", "tp", "fp", "fn", "components"]
    main = selected[main_columns]

    base = grid[grid["dataset"] == "water_base"].copy()
    sensitivity = (base.groupby(["alpha", "topk", "candidate_k", "lambda_stability"], as_index=False)
                   .agg(j_val_mean=("j_val", "mean"), j_val_std=("j_val", "std"),
                        f1_mean=("f1", "mean"), f1_std=("f1", "std"), runs=("run", "count"))
                   .sort_values("j_val_mean"))

    decoder_rows = []
    for label, filename in [
        ("sample-frequency consensus", "dgm_learned_edges_consensus.pt"),
        ("mean-probability row-k mutual + MST", "dgm_learned_edges_softprob.pt"),
    ]:
        edges = load_torch_edges(MAIN_RUN_DIR / filename)
        stats = graph_stats(50, edges)
        metrics = edge_metrics(reference, edges)
        decoder_rows.append({"decoder_or_consensus": label, "edges": len(edges),
                             "density": stats["density"], "mean_degree": stats["mean_degree"],
                             "components": stats["components"], **metrics})
    return main, sensitivity, pd.DataFrame(decoder_rows)


def complexity_table() -> pd.DataFrame:
    rows = [
        ("Window encoder", "O(B N L d h)", "O(B N L d + B N h)", "implemented dense node-wise embedding"),
        ("Full pairwise DGM score", "O(B N^2 h)", "O(B N^2)", "pairwise squared distances in DGM_d"),
        ("Candidate mask construction", "O(N^2 T + N^3)", "O(N^2 + T N)", "pairwise observables plus dense VAR solve"),
        ("Masked top-k selection", "O(N C + N C log k)", "O(N C)", "C candidate neighbors per row; current logits remain dense"),
        ("Sparse message passing", "O(B N k h)", "O(B N k + B N h)", "sampled directed graph"),
        ("Frequency consensus", "O(S N k)", "O(E_c)", "S validation windows/graphs"),
        ("Mean-probability aggregation", "O(S N^2)", "O(N^2)", "dense probability moments"),
        ("Row-k decoder", "O(N^2)", "O(N^2)", "argpartition plus symmetry"),
        ("Geometric MST", "O(N^3)", "O(N^2)", "project Prim implementation scans dense distances"),
    ]
    return pd.DataFrame(rows, columns=["module", "time_complexity", "memory_complexity", "code_specific_note"])


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.axis("off")
    labels = ["Coordinates +\ntime series", "Observable Q +\ncandidate mask", "DGM +\nforecasting", "Dynamic edge\nprobabilities", "GT-free static\ndecoder", "Post-hoc GT\nevaluation"]
    x = np.linspace(0.07, 0.93, len(labels))
    for idx, (xi, label) in enumerate(zip(x, labels)):
        face = "#EAF2F8" if idx < 5 else "#FCE8EE"
        ax.text(xi, 0.5, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.45", "fc": face, "ec": "#444444", "lw": 0.9})
        if idx < len(labels) - 1:
            ax.annotate("", xy=(x[idx + 1] - 0.065, 0.5), xytext=(xi + 0.065, 0.5),
                        xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"})
    save_figure(fig, RESULTS_ROOT / "figures", "pipeline")
    plt.close(fig)


def figure_graphs(data: np.lib.npyio.NpzFile, reference, candidate, learned) -> None:
    xy = data["node_xy"]
    if xy.ndim == 3:
        xy = xy[0]
    for stem, edges, color in [
        ("reference_graph", reference, COLORS["reference"]),
        ("candidate_graph", candidate, COLORS["candidate"]),
        ("reconstructed_graph", learned, COLORS["learned"]),
    ]:
        fig, ax = plt.subplots(figsize=(3.45, 3.0))
        draw_edges(ax, xy, edges, color)
        export_plot_data(stem, pd.DataFrame(sorted(edges), columns=["u", "v"]))
        save_figure(fig, RESULTS_ROOT / "figures", stem)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    draw_edges(ax, xy, reference & learned, COLORS["tp"], width=1.8)
    draw_edges(ax, xy, learned - reference, COLORS["fp"], width=1.6)
    draw_edges(ax, xy, reference - learned, COLORS["fn"], width=1.2, linestyle="--")
    handles = [plt.Line2D([0], [0], color=COLORS[key], lw=2, ls="--" if key == "fn" else "-")
               for key in ("tp", "fp", "fn")]
    ax.legend(handles, ["TP", "FP", "FN"], frameon=False, loc="best")
    save_figure(fig, RESULTS_ROOT / "figures", "reconstruction_errors")
    plt.close(fig)


def figure_signals(data: np.lib.npyio.NpzFile) -> None:
    pressure = data["pressure"]
    demand = data["demand"]
    time = np.arange(min(168, pressure.shape[0]))
    nodes = [0, 10, 20, 30]
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
    for node in nodes:
        axes[0].plot(time, pressure[:len(time), node], label=f"Node {node}")
        axes[1].plot(time, demand[:len(time), node], label=f"Node {node}")
        rows.extend({"hour": int(t), "node": node, "pressure": float(pressure[t, node]),
                     "demand": float(demand[t, node])} for t in time)
    axes[0].set_ylabel("Pressure-like signal")
    axes[1].set_ylabel("Demand-like signal")
    axes[1].set_xlabel("Time [h]")
    axes[0].legend(ncol=4, frameon=False, loc="upper center")
    export_plot_data("signal_examples", pd.DataFrame(rows))
    save_figure(fig, RESULTS_ROOT / "figures", "signal_examples")
    plt.close(fig)

    corr = np.corrcoef(pressure, rowvar=False)
    lag = np.corrcoef(pressure[1:].T, pressure[:-1].T)[:pressure.shape[1], pressure.shape[1]:]
    for stem, matrix in [("pressure_correlation", corr), ("lagged_pressure_correlation", lag)]:
        fig, ax = plt.subplots(figsize=(3.45, 3.0))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xlabel("Node index")
        ax.set_ylabel("Node index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        np.savez_compressed(RESULTS_ROOT / "figures/data" / f"{stem}.npz", matrix=matrix)
        save_figure(fig, RESULTS_ROOT / "figures", stem)
        plt.close(fig)


def figure_degree(reference, candidate, learned) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    for label, edges, color in [("Reference", reference, COLORS["reference"]),
                                ("Candidate", candidate, COLORS["candidate"]),
                                ("Reconstructed", learned, COLORS["learned"])]:
        graph = nx.Graph(); graph.add_nodes_from(range(50)); graph.add_edges_from(edges)
        degrees = np.array([degree for _, degree in graph.degree()])
        bins = np.arange(degrees.max() + 2) - 0.5
        ax.hist(degrees, bins=bins, density=True, histtype="step", lw=1.8, label=label, color=color)
        rows.extend({"graph": label, "node": node, "degree": int(degree)} for node, degree in graph.degree())
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Relative frequency")
    ax.legend(frameon=False)
    export_plot_data("degree_distribution", pd.DataFrame(rows))
    save_figure(fig, RESULTS_ROOT / "figures", "degree_distribution")
    plt.close(fig)


def figure_sensitivity(grid: pd.DataFrame) -> None:
    base = grid[(grid["dataset"] == "water_base") & (grid["topk"] == 1) & (grid["candidate_k"] == 2)]
    summary = base.groupby(["alpha", "lambda_stability"], as_index=False).agg(j_val=("j_val", "mean"), f1=("f1", "mean"))
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    for stability, group in summary.groupby("lambda_stability"):
        group = group.sort_values("alpha")
        ax.plot(group["alpha"], group["j_val"], marker="o", label=rf"$\lambda_s={stability:g}$")
    ax.set_xlabel(r"Forecast weight $\alpha$")
    ax.set_ylabel(r"Observable validation objective $J_{obs}$")
    ax.legend(frameon=False)
    export_plot_data("alpha_sensitivity", summary)
    save_figure(fig, RESULTS_ROOT / "figures", "alpha_sensitivity")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.scatter(grid["test_mse"], grid["f1"], s=22, alpha=0.75, c=grid["alpha"], cmap="viridis")
    ax.set_xlabel("Test forecasting MSE")
    ax.set_ylabel("Post-hoc graph F1")
    export_plot_data("forecast_vs_graph_f1", grid[["run", "dataset", "alpha", "test_mse", "f1"]])
    save_figure(fig, RESULTS_ROOT / "figures", "forecast_vs_graph_f1")
    plt.close(fig)


def figure_score_distribution(data, reference) -> None:
    score = np.load(MAIN_RUN_DIR / "dgm_soft_score.npy")
    if score.ndim > 2:
        score = np.asarray(score).squeeze()
    rows = []
    true_values, false_values = [], []
    for u in range(score.shape[0]):
        for v in range(u + 1, score.shape[1]):
            value = float(0.5 * (score[u, v] + score[v, u]))
            is_edge = (u, v) in reference
            (true_values if is_edge else false_values).append(value)
            rows.append({"u": u, "v": v, "score": value, "reference_edge": int(is_edge)})
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.hist(false_values, bins=30, density=True, histtype="step", lw=1.8, label="Non-edge", color=COLORS["candidate"])
    ax.hist(true_values, bins=30, density=True, histtype="step", lw=1.8, label="Reference edge", color=COLORS["learned"])
    ax.set_xlabel("Symmetrized mean DGM score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    export_plot_data("score_distribution", pd.DataFrame(rows))
    save_figure(fig, RESULTS_ROOT / "figures", "score_distribution")
    plt.close(fig)


def write_manifest() -> None:
    rows = [
        {"experiment_id": MAIN_RUN, "status": "historical_completed", "scenario": "water_base",
         "dataset_seed": 31, "training_seed": 42, "selection": "observable_val",
         "source": str(MAIN_RUN_DIR.relative_to(PROJECT_ROOT)), "oracle": False},
        {"experiment_id": "section_v_smoke_seed42", "status": "planned_until_smoke_is_run", "scenario": "water_base",
         "dataset_seed": 31, "training_seed": 42, "selection": "observable_val",
         "source": "results_section_v/raw/section_v_smoke_seed42", "oracle": False},
    ]
    pd.DataFrame(rows).to_csv(RESULTS_ROOT / "experiment_manifest.csv", index=False)


def main() -> None:
    apply_style()
    for path in [RESULTS_ROOT / "raw", RESULTS_ROOT / "configs", RESULTS_ROOT / "tables",
                 RESULTS_ROOT / "figures/data", RESULTS_ROOT / "figures/pdf",
                 RESULTS_ROOT / "figures/svg", RESULTS_ROOT / "figures/png"]:
        path.mkdir(parents=True, exist_ok=True)

    data = np.load(DATASET_DIRS["water_base"] / "data.npz")
    reference = canonical_edges(data["edge_index"])
    candidate = canonical_edges(data["initial_edge_index"])
    learned = load_torch_edges(MAIN_RUN_DIR / "dgm_learned_edges_softprob.pt")
    grid = pd.read_csv(GRID_CSV)

    table1 = dataset_table()
    table2 = modules_table()
    table3 = config_table()
    table4, table5, table67 = results_tables(grid, reference)
    table8 = complexity_table()
    write_table(1, "synthetic_dataset_properties", table1)
    write_table(2, "model_modules_and_roles", table2)
    write_table(3, "main_configuration", table3)
    write_table(4, "main_quantitative_results", table4)
    write_table(5, "hyperparameter_sensitivity", table5)
    write_table(6, "decoder_comparison", table67)
    write_table(7, "consensus_comparison", table67)
    write_table(8, "complexity", table8)

    figure_pipeline()
    figure_graphs(data, reference, candidate, learned)
    figure_signals(data)
    figure_degree(reference, candidate, learned)
    figure_sensitivity(grid)
    figure_score_distribution(data, reference)
    write_manifest()
    print(f"[OK] Section V artifacts written to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
