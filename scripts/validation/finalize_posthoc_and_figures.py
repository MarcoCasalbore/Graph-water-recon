"""Finalize beta rankings, Pareto frontier, and main/supplementary figures."""
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
R = P / "results_section_v_final"
T = R / "tables"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_section_v_final import corrected_methods, save_figure, save_table  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False})


def beta_outputs(methods):
    oracle, node = "oracle_reference_graph", "node_only_empty_graph"
    requested = [0, .1, .25, .5, .75, .9, 1]
    variants = {
        "all_including_oracle": lambda d: pd.Series(True, index=d.index),
        "deployable_with_node": lambda d: d.method != oracle,
        "deployable_graph_methods_only": lambda d: (d.method != oracle) & (d.method != node),
    }
    rows = []
    for beta in requested:
        d = methods.copy(); d["joint_score"] = beta/(1+d.test_mse) + (1-beta)*d.f1
        for variant, selector in variants.items():
            agg = d[selector(d)].groupby("method").joint_score.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
            for rank, (method, row) in enumerate(agg.iterrows(), 1):
                rows.append({"variant": variant, "beta": beta, "rank": rank, "method": method,
                             "joint_score_mean": row["mean"], "joint_score_std": row["std"], "seed_count": int(row["count"])})
    save_table(pd.DataFrame(rows), "table_11_corrected_balanced_joint_score_beta")

    summary = methods.groupby("method").agg(mse=("test_mse", "mean"), f1=("f1", "mean")).reset_index()
    summary["task"] = 1/(1+summary.mse)
    dense = []
    for beta in np.linspace(0, 1, 1001):
        for _, row in summary.iterrows():
            dense.append({"beta": beta, "method": row.method,
                          "joint_score": beta*row.task+(1-beta)*row.f1})
    dense = pd.DataFrame(dense); dense.to_csv(T / "source_table_11_all_method_beta_curves.csv", index=False)
    intervals = []
    for variant, selector in variants.items():
        q = dense.merge(summary[["method"]], on="method")
        keep_methods = methods[selector(methods)].method.unique(); q = q[q.method.isin(keep_methods)]
        win = q.loc[q.groupby("beta").joint_score.idxmax()].sort_values("beta").reset_index(drop=True)
        start = 0
        for i in range(1, len(win)+1):
            if i == len(win) or win.loc[i, "method"] != win.loc[start, "method"]:
                intervals.append({"variant": variant, "best_method": win.loc[start, "method"],
                                  "beta_start": win.loc[start, "beta"], "beta_end": win.loc[i-1, "beta"]})
                start = i
    save_table(pd.DataFrame(intervals), "joint_score_dominance_intervals")
    intersections = []
    for i, a in summary.iterrows():
        for _, b in summary.iloc[i+1:].iterrows():
            den = (a.task-a.f1)-(b.task-b.f1)
            if abs(den) > 1e-12:
                beta = (b.f1-a.f1)/den
                if 0 <= beta <= 1: intersections.append({"method_a": a.method, "method_b": b.method, "beta": beta})
    save_table(pd.DataFrame(intersections), "joint_score_pairwise_intersections")
    return dense


def pareto_front(df):
    deploy = df[df.method != "oracle_reference_graph"].sort_values("test_mse_mean")
    best = -np.inf; rows = []
    for _, row in deploy.iterrows():
        if row.f1_mean > best:
            rows.append(row); best = row.f1_mean
    return pd.DataFrame(rows)


def figures(methods, dense):
    fore = methods.groupby("method").agg(test_mse_mean=("test_mse", "mean"), test_mse_std=("test_mse", "std"),
        test_mae_mean=("test_mae", "mean"), test_mae_std=("test_mae", "std"), f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index()
    labels = fore.method.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(7.4, 3.7)); order = np.argsort(fore.f1_mean.values)
    ax.barh(np.arange(len(fore)), fore.f1_mean.iloc[order], xerr=fore.f1_std.iloc[order], color="#4c78a8")
    ax.set(yticks=np.arange(len(fore)), yticklabels=labels.iloc[order], xlabel="Topology F1"); save_figure(fig, "topology_baseline_comparison", fore)

    fig, ax = plt.subplots(figsize=(7.4, 3.7)); order = np.argsort(fore.test_mse_mean.values)[::-1]
    ax.barh(np.arange(len(fore)), fore.test_mse_mean.iloc[order], xerr=fore.test_mse_std.iloc[order], color="#f58518")
    ax.set(yticks=np.arange(len(fore)), yticklabels=labels.iloc[order], xlabel="True element-wise test MSE"); save_figure(fig, "fixed_graph_forecasting_comparison", fore)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for _, row in fore.iterrows():
        oracle = row.method == "oracle_reference_graph"
        ax.errorbar(row.test_mse_mean, row.f1_mean, xerr=row.test_mse_std, yerr=row.f1_std,
                    fmt="*" if oracle else "o", ms=9 if oracle else 5, capsize=2, color="#d62728" if oracle else "#4c78a8")
        ax.annotate(row.method.replace("_", " "), (row.test_mse_mean, row.f1_mean), fontsize=6, xytext=(3, 2), textcoords="offset points")
    front = pareto_front(fore).sort_values("test_mse_mean")
    ax.plot(front.test_mse_mean, front.f1_mean, "--", color="black", lw=1, label="deployable Pareto front")
    ax.set(xlabel="True element-wise test MSE (lower is better)", ylabel="Topology F1 (higher is better)"); ax.legend(fontsize=7)
    save_figure(fig, "corrected_pareto_forecast_topology", fore)
    front.to_csv(R / "figures/main/data/corrected_pareto_frontier.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    chosen = ["node_only_empty_graph", "proposed_joint", "proposed_learned_graph_frozen", "geometric_knn3_mutual", "oracle_reference_graph"]
    for method, block in dense[dense.method.isin(chosen)].groupby("method"):
        ax.plot(block.beta, block.joint_score, label=method.replace("_", " "))
    ax.set(xlabel="beta (forecast weight)", ylabel="Post-hoc joint task–topology score"); ax.legend(fontsize=7)
    save_figure(fig, "corrected_joint_score_beta", dense[dense.method.isin(chosen)])


def supplementary():
    stems = ["decoder_comparison", "consensus_comparison", "degree_distribution", "score_distribution",
             "error_by_selected_node", "error_by_variable", "node_only_vs_joint_node_error", "absolute_error_boxplot"]
    for ext in ["pdf", "svg", "png"]:
        out = R / "figures/supplementary" / ext; out.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            candidates = [R/"figures/main"/ext/f"{stem}.{ext}", R/"error_analysis/figures"/ext/f"{stem}.{ext}"]
            for src in candidates:
                if src.exists(): shutil.copy2(src, out/src.name); break
    out = R / "figures/supplementary/data"; out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for src in [R/"figures/main/data"/f"{stem}.csv", R/"error_analysis/figures/data"/f"{stem}.csv"]:
            if src.exists(): shutil.copy2(src, out/src.name); break


def main():
    methods = corrected_methods(); dense = beta_outputs(methods); figures(methods, dense); supplementary()
    print("[DONE] post-hoc rankings, intervals, intersections, Pareto and comparison figures")


if __name__ == "__main__": main()
