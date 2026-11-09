"""Assemble the corrected, paper-ready Section V package.

Forecast MSE/MAE for joint models are always derived from exported aligned
predictions.  Older fields named pressure_mse/demand_mse contain Huber loss and
are deliberately never consumed here.
"""
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
TABLES = FINAL / "tables"
FIG = FINAL / "figures"
SIMPLE = SRC / "campaign_v2/relational/simple_topological_baselines"
RAW = SRC / "campaign_v2/relational/raw_metrics"

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})


def save_table(df, stem):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    (TABLES / f"{stem}.tex").write_text(
        df.to_latex(index=False, escape=True, float_format="%.5f"), encoding="utf-8")


def save_figure(fig, stem, data=None):
    for ext in ["pdf", "svg", "png"]:
        out = FIG / "main" / ext
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    if data is not None:
        out = FIG / "main/data"; out.mkdir(parents=True, exist_ok=True)
        data.to_csv(out / f"{stem}.csv", index=False)
    plt.close(fig)


def mean_std(df, keys, metrics):
    g = df.groupby(keys, dropna=False)
    pieces = []
    for metric in metrics:
        pieces.append(g[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    out = pd.concat(pieces, axis=1).reset_index()
    out["training_seeds"] = g.size().values
    return out


def corrected_methods():
    methods = pd.read_csv(SIMPLE / "all_method_seed_metrics.csv")
    err = pd.read_csv(FINAL / "error_analysis/figures/data/error_statistics_by_seed.csv")
    aliases = {
        "node_only": "node_only_empty_graph",
        "proposed_joint": "proposed_joint",
        "euclidean_mst_frozen": "euclidean_mst",
        "geometric_mutual_knn_frozen": "geometric_knn3_mutual",
        "geometric_union_knn_frozen": "geometric_knn3_union",
        "correlation_frozen": "correlation_abs_knn3_union",
        "hybrid_frozen": "distance_correlation_hybrid_knn3_union",
        "proposed_learned_graph_frozen": "proposed_learned_graph_frozen",
        "reference_graph_frozen": "oracle_reference_graph",
        "random_graph_frozen": "random_learned_edge_count_control",
    }
    for old, new in aliases.items():
        replacement = err[err.method == old]
        for _, row in replacement.iterrows():
            mask = (methods.method == new) & (methods.training_seed == row.training_seed)
            methods.loc[mask, ["test_mse", "test_mae", "test_rmse"]] = \
                [row.mse, row.mae, row.rmse]
    methods["task_score"] = 1 / (1 + methods.test_mse)
    return methods


def build_tables(sweeps, methods):
    # 1--2: descriptive settings retained from the corrected relational package.
    for number, source, stem in [
        (1, "table_1_synthetic_dataset_properties.csv", "table_1_synthetic_settings"),
        (2, "table_2_model_modules_and_roles.csv", "table_2_model_configuration")]:
        save_table(pd.read_csv(SRC / "tables" / source), stem)

    main = pd.read_csv(RAW / "main_runs.csv")
    actual = pd.read_csv(FINAL / "error_analysis/figures/data/error_statistics_by_seed.csv")
    actual = actual[actual.method == "proposed_joint"][["training_seed", "mse", "mae", "rmse"]]
    main = main.merge(actual, on="training_seed", how="left")
    table3 = mean_std(main, ["group", "lags", "horizon", "epochs"],
                      ["mse", "mae", "precision", "recall", "f1", "edge_count", "runtime_seconds"])
    save_table(table3, "table_3_main_topology_reconstruction")

    topology = mean_std(methods, ["method"],
                        ["precision", "recall", "f1", "jaccard", "edge_count", "mean_degree", "components"])
    save_table(topology.sort_values("f1_mean", ascending=False), "table_4_simple_topological_baselines")
    forecast = mean_std(methods, ["method"], ["test_mse", "test_mae", "test_rmse", "f1", "runtime_seconds"])
    node_mse = forecast.loc[forecast.method == "node_only_empty_graph", "test_mse_mean"].iloc[0]
    forecast["mse_improvement_vs_node_pct"] = 100 * (node_mse - forecast.test_mse_mean) / node_mse
    save_table(forecast.sort_values("test_mse_mean"), "table_5_fixed_graph_forecasting")

    node = pd.read_csv(RAW / "lh_node_runs.csv").rename(
        columns={"test_mse": "node_mse", "test_mae": "node_mae"})
    joint = sweeps[sweeps.group == "lh"].rename(columns={"mse": "joint_mse", "mae": "joint_mae"})
    lh = joint.merge(node[["lags", "horizon", "training_seed", "node_mse", "node_mae"]],
                     on=["lags", "horizon", "training_seed"], how="left")
    lh["mse_gain_pct"] = 100 * (lh.node_mse - lh.joint_mse) / lh.node_mse
    lh["mae_gain_pct"] = 100 * (lh.node_mae - lh.joint_mae) / lh.node_mae
    table6 = mean_std(lh, ["lags", "horizon"],
                      ["node_mse", "joint_mse", "node_mae", "joint_mae", "mse_gain_pct", "mae_gain_pct", "f1"])
    save_table(table6, "table_6_temporal_lag_horizon_explainability")

    save_table(pd.read_csv(FINAL / "error_analysis/tables/table_7_error_distribution_statistics.csv"),
               "table_7_error_distribution_statistics")
    alpha = mean_std(sweeps[sweeps.group == "alpha"], ["alpha"],
                     ["mse", "mae", "total_loss", "task_loss", "topology_loss", "eval0p5",
                      "precision", "recall", "f1", "edge_count", "mean_degree"])
    save_table(alpha, "table_8_alpha_balanced_loss_sensitivity")
    kdgm = mean_std(sweeps[sweeps.group == "k_dgm"], ["k_dgm"],
                    ["mse", "mae", "precision", "recall", "f1", "edge_count", "runtime_seconds"])
    save_table(kdgm, "table_9_k_dgm_sensitivity")
    multi = pd.read_csv(FINAL / "multiple_graph_realizations/per_graph_summary.csv")
    save_table(multi, "table_10_multiple_graph_realizations")

    summary = mean_std(methods, ["method"], ["test_mse", "f1"])
    summary["task_score"] = 1 / (1 + summary.test_mse_mean)
    betas = np.linspace(0, 1, 101)
    curves = []
    oracle = "oracle_reference_graph"
    for beta in betas:
        for _, row in summary.iterrows():
            curves.append({"beta": beta, "method": row.method,
                           "joint_score": (1-beta)*row.f1_mean + beta*row.task_score,
                           "includes_oracle": row.method == oracle,
                           "includes_node": row.method == "node_only_empty_graph"})
    curves = pd.DataFrame(curves)
    rows = []
    for variant, q in {
        "all_including_oracle": curves,
        "deployable_with_node": curves[~curves.includes_oracle],
        "deployable_graph_methods_only": curves[(~curves.includes_oracle) & (~curves.includes_node)],
    }.items():
        for beta, block in q.groupby("beta"):
            best = block.loc[block.joint_score.idxmax()]
            rows.append({"variant": variant, "beta": beta, "best_method": best.method,
                         "best_joint_score": best.joint_score})
    beta_table = pd.DataFrame(rows)
    save_table(beta_table, "table_11_corrected_balanced_joint_score_beta")
    curves.to_csv(TABLES / "source_table_11_all_method_beta_curves.csv", index=False)

    dec = pd.read_csv(SRC / "tables/table_6_decoder_comparison.csv"); dec.insert(0, "analysis", "decoder")
    con = pd.read_csv(SRC / "tables/table_7_consensus_comparison.csv"); con.insert(0, "analysis", "consensus")
    table12 = pd.concat([dec, con], ignore_index=True, sort=False)
    save_table(table12, "table_12_decoder_consensus_supplement")
    return table3, topology, forecast, table6, alpha, kdgm, multi, curves, beta_table


def plot_outputs(tables):
    table3, topology, forecast, lh, alpha, kdgm, multi, curves, beta_table = tables
    # L/H interaction: corrected MSE gain and paired errors.
    pivot = lh.pivot(index="lags", columns="horizon", values="mse_gain_pct_mean")
    fig, ax = plt.subplots(figsize=(5.2, 3.6)); im = ax.imshow(pivot, cmap="RdBu", aspect="auto", vmin=-max(abs(pivot.min().min()), abs(pivot.max().max())), vmax=max(abs(pivot.min().min()), abs(pivot.max().max())))
    ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticks=range(len(pivot.index)), yticklabels=pivot.index, xlabel="Forecast horizon H", ylabel="Input lags L")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)): ax.text(j, i, f"{pivot.iloc[i,j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Joint improvement vs node-only (%)"); save_figure(fig, "corrected_lh_mse_gain_heatmap", lh)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharex=True)
    for h, block in lh.groupby("horizon"):
        axes[0].plot(block.lags, block.node_mse_mean, "--o", alpha=.75, label=f"node H={h}")
        axes[1].plot(block.lags, block.joint_mse_mean, "-o", alpha=.75, label=f"joint H={h}")
    axes[0].set(xlabel="Lags L", ylabel="Test MSE"); axes[1].set(xlabel="Lags L", ylabel="Test MSE")
    axes[0].legend(fontsize=7, ncol=2); axes[1].legend(fontsize=7, ncol=2); save_figure(fig, "corrected_lh_node_joint_mse", lh)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].errorbar(alpha.alpha, alpha.mse_mean, yerr=alpha.mse_std, marker="o", label="MSE")
    axes[0].errorbar(alpha.alpha, alpha.mae_mean, yerr=alpha.mae_std, marker="s", label="MAE")
    axes[0].set(xlabel="Task weight alpha", ylabel="Forecast error"); axes[0].legend()
    axes[1].errorbar(alpha.alpha, alpha.f1_mean, yerr=alpha.f1_std, marker="o", label="F1")
    axes[1].plot(alpha.alpha, alpha.eval0p5_mean, marker="s", label="Eval(0.5)")
    axes[1].set(xlabel="Task weight alpha", ylabel="Score"); axes[1].legend(); save_figure(fig, "corrected_alpha_tradeoff", alpha)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(alpha.alpha, alpha.total_loss_mean, "-o", label="total")
    ax.plot(alpha.alpha, alpha.task_loss_mean, "-s", label="task (Huber)")
    ax.plot(alpha.alpha, alpha.topology_loss_mean, "-^", label="topology")
    ax.set(xlabel="Task weight alpha", ylabel="Selected-epoch training loss"); ax.legend(); save_figure(fig, "balanced_loss_components_vs_alpha", alpha)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].errorbar(kdgm.k_dgm, kdgm.mse_mean, yerr=kdgm.mse_std, marker="o")
    axes[0].set(xlabel="k_DGM", ylabel="Test MSE")
    axes[1].errorbar(kdgm.k_dgm, kdgm.f1_mean, yerr=kdgm.f1_std, marker="o", label="F1")
    axes[1].plot(kdgm.k_dgm, kdgm.edge_count_mean / kdgm.edge_count_mean.max(), "--s", label="edge count (scaled)")
    axes[1].set(xlabel="k_DGM", ylabel="Topology score"); axes[1].legend(); save_figure(fig, "corrected_k_dgm_sensitivity", kdgm)

    selected = multi[multi.method.isin(["node_only", "proposed_joint", "proposed_learned_graph_frozen", "geometric_knn3_mutual"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = np.arange(selected.graph_seed.nunique()); width=.18
    for i, (method, block) in enumerate(selected.groupby("method")):
        block = block.sort_values("graph_seed"); ax.bar(x+(i-1.5)*width, block.mse_mean, width, yerr=block.mse_std, label=method)
    ax.set(xticks=x, xticklabels=sorted(selected.graph_seed.unique()), xlabel="Graph realization seed", ylabel="Test MSE"); ax.legend(fontsize=7); save_figure(fig, "multiple_graph_forecast_robustness", selected)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for method, block in multi[multi.method != "node_only"].groupby("method"):
        ax.scatter(block.mse_mean, block.f1_mean, label=method, s=28)
    ax.set(xlabel="Test MSE (lower is better)", ylabel="Topology F1 (higher is better)"); ax.legend(fontsize=6, ncol=2); save_figure(fig, "multiple_graph_accuracy_topology_tradeoff", multi)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for _, row in forecast.iterrows():
        ax.scatter(row.test_mse_mean, row.f1_mean, s=35); ax.annotate(row.method.replace("_", " "), (row.test_mse_mean, row.f1_mean), fontsize=6, xytext=(3,2), textcoords="offset points")
    ax.set(xlabel="True element-wise test MSE (lower is better)", ylabel="Topology F1 (higher is better)"); save_figure(fig, "corrected_pareto_forecast_topology", forecast)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    show = curves[curves.method.isin(["node_only_empty_graph", "proposed_joint", "proposed_learned_graph_frozen", "geometric_knn3_mutual", "oracle_reference_graph"])]
    for method, block in show.groupby("method"): ax.plot(block.beta, block.joint_score, label=method)
    ax.set(xlabel="beta (forecast weight)", ylabel="Balanced JointScore"); ax.legend(fontsize=7); save_figure(fig, "corrected_joint_score_beta", show)


def copy_selected_figures():
    selected = ["signal_forecast_examples", "reference_graph", "candidate_graph", "final_inferred_graph",
                "reconstruction_errors", "graph_evolution", "graph_evolution_metrics", "loss_total",
                "score_distribution", "seed_stability", "decoder_comparison", "consensus_comparison"]
    source = SRC / "campaign_v2/relational/figures"
    for stem in selected:
        for ext in ["pdf", "svg", "png"]:
            src = source / ext / f"{stem}.{ext}"
            if src.exists():
                dst = FIG / "main" / ext / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        data = source / "data" / f"{stem}.csv"
        if data.exists():
            dst = FIG / "main/data" / data.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(data, dst)
    err = FINAL / "error_analysis/figures"
    for ext in ["pdf", "svg", "png"]:
        for src in (err / ext).glob(f"*.{ext}"):
            dst = FIG / "main" / ext / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    for src in (err / "data").glob("*.csv"):
        dst = FIG / "main/data" / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def validate(sweeps):
    checks = {
        "corrected_sweep_rows_84": len(sweeps) == 84,
        "corrected_sweep_predictions_all_finite": bool(np.isfinite(sweeps[["mse", "mae", "f1"]]).all().all()),
        "three_graph_realizations": pd.read_csv(FINAL / "multiple_graph_realizations/per_graph_summary.csv").graph_seed.nunique() == 3,
        "multi_graph_method_runs_54": len(pd.read_csv(FINAL / "multiple_graph_realizations/multiple_graph_runs.csv")) == 54,
        "twelve_primary_tables": len(list(TABLES.glob("table_[0-9]*.csv"))) >= 12,
        "all_tables_have_markdown_latex": all(p.with_suffix(".md").exists() and p.with_suffix(".tex").exists() for p in TABLES.glob("table_[0-9]*.csv")),
        "main_figures_all_three_formats": all(len(list((FIG/"main"/e).glob(f"*.{e}"))) >= 15 for e in ["pdf", "svg", "png"]),
    }
    (FINAL / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    pd.DataFrame([{"check": k, "passed": v} for k, v in checks.items()]).to_csv(FINAL / "validation_checks.csv", index=False)
    if not all(checks.values()): raise RuntimeError(f"Validation failed: {checks}")


def main():
    sweeps = pd.read_csv(FINAL / "metric_corrected_runs/corrected_joint_sweeps.csv")
    if len(sweeps) != 84: raise RuntimeError(f"Expected 84 corrected sweeps, found {len(sweeps)}")
    methods = corrected_methods()
    tables = build_tables(sweeps, methods)
    plot_outputs(tables); copy_selected_figures(); validate(sweeps)
    print(f"[DONE] tables={len(list(TABLES.glob('table_[0-9]*.csv')))} figures={len(list((FIG/'main/pdf').glob('*.pdf')))}")


if __name__ == "__main__":
    main()
