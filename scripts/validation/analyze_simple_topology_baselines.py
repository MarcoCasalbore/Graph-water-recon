"""Analyze fixed-graph baselines, joint scores, Pareto trade-offs and figures."""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
CAMPAIGN = RESULTS / "campaign_v2/relational"
ROOT = CAMPAIGN / "simple_topological_baselines"
FIG, TABLE = ROOT / "figures", ROOT / "tables"
DATASET = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
sys.path.insert(0, str(RESULTS))
from scripts.plot_style import COLORS, apply_style, save_figure  # noqa: E402

BETAS = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
META = {
    "node_only_empty_graph": ("No edges; independent shared node MLP", False, False, False, False),
    "euclidean_mst": ("Euclidean coordinate MST", False, True, False, False),
    "geometric_knn3_mutual": ("Mutual geometric 3-NN", False, True, False, False),
    "geometric_knn3_union": ("Union geometric 3-NN", False, True, False, False),
    "correlation_abs_knn3_union": ("Union top-3 mean absolute Pearson correlation; train split only", False, False, True, False),
    "distance_correlation_hybrid_knn3_union": ("Union top-3 of 0.5 geometry-rank + 0.5 train-correlation", False, True, True, False),
    "proposed_learned_graph_frozen": ("DGM graph, then frozen in exact forecasting path", True, True, True, False),
    "proposed_joint": ("Joint DGM topology and forecasting", True, True, True, False),
    "random_learned_edge_count_control": ("Random control matched to learned-graph edge count", False, False, False, False),
    "oracle_reference_graph": ("Oracle reference-graph forecaster", False, False, False, True),
}
LABEL = {
    "node_only_empty_graph": "Node-only", "euclidean_mst": "Euclidean MST",
    "geometric_knn3_mutual": "Geo 3-NN mutual", "geometric_knn3_union": "Geo 3-NN union",
    "correlation_abs_knn3_union": "Correlation 3-NN", "distance_correlation_hybrid_knn3_union": "Hybrid 3-NN",
    "proposed_learned_graph_frozen": "DGM graph (frozen)", "proposed_joint": "Proposed joint",
    "random_learned_edge_count_control": "Random control", "oracle_reference_graph": "Oracle reference",
}


def save_table(frame, stem):
    TABLE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE / f"{stem}.csv", index=False)
    (TABLE / f"{stem}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (TABLE / f"{stem}.tex").write_text(frame.to_latex(index=False, float_format="%.5f"), encoding="utf-8")


def edge_set(path):
    frame = pd.read_csv(path)
    return {tuple(sorted((int(r.u), int(r.v)))) for r in frame.itertuples()}


def aggregate(frame, keys):
    metrics = ["test_mse", "test_mae", "test_rmse", "f1", "precision", "recall", "jaccard",
               "edge_count", "components", "mean_degree", "min_degree", "max_degree",
               "isolated_nodes", "bridge_count", "tp", "fp", "fn"]
    specs = {m: ["mean", "std"] for m in metrics if m in frame.columns}
    out = frame.groupby(keys).agg(specs).reset_index()
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out


def prepare_rows():
    fixed = pd.read_csv(ROOT / "fixed_graph_forecasting_runs.csv")
    fixed.loc[fixed.method == "proposed_final_decoded_graph", "method"] = "proposed_learned_graph_frozen"
    main = pd.read_csv(CAMPAIGN / "tables/table_4_main_multiseed_results.csv")
    fixed.loc[fixed.method == "random_final_decoded_edge_count_control", "method"] = "random_learned_edge_count_control"
    proposed = pd.DataFrame({
        "method": "proposed_joint", "training_seed": main.training_seed,
        "test_mse": main.forecast_mse, "test_mae": main.test_mae, "test_rmse": np.sqrt(main.forecast_mse),
        "f1": main.f1, "precision": main.precision, "recall": main.recall,
        "jaccard": main.jaccard, "edge_count": main.edge_count, "components": main.components,
    })
    nodes = pd.read_csv(CAMPAIGN / "raw_metrics/main_node_runs.csv")
    node = pd.DataFrame({
        "method": "node_only_empty_graph", "training_seed": nodes.training_seed,
        "test_mse": nodes.test_mse, "test_mae": nodes.test_mae, "test_rmse": nodes.test_rmse,
        "f1": 0.0, "precision": 0.0, "recall": 0.0, "jaccard": 0.0,
        "edge_count": 0, "components": 50, "mean_degree": 0.0, "min_degree": 0,
        "max_degree": 0, "isolated_nodes": 50, "bridge_count": 0, "tp": 0, "fp": 0, "fn": 79,
    })
    rows = pd.concat([fixed, proposed, node], ignore_index=True, sort=False)
    rows["task_score"] = 1.0 / (1.0 + rows.test_mse)
    for beta in BETAS:
        rows[f"joint_beta_{beta:g}"] = beta * rows.task_score + (1 - beta) * rows.f1
    return rows


def pareto_methods(summary, allowed):
    selected = summary[summary.method.isin(allowed)]
    frontier = []
    for row in selected.itertuples():
        dominated = any((other.test_mse_mean <= row.test_mse_mean and other.f1_mean >= row.f1_mean)
                        and (other.test_mse_mean < row.test_mse_mean or other.f1_mean > row.f1_mean)
                        for other in selected.itertuples())
        if not dominated:
            frontier.append(row.method)
    return frontier


def beta_intervals(curves, methods):
    intersections = {0.0, 1.0}
    params = {}
    for method in methods:
        row = curves[curves.method == method].iloc[0]
        params[method] = (row.f1_mean, row.task_score_mean - row.f1_mean)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            ia, sa = params[a]; ib, sb = params[b]
            if abs(sa - sb) > 1e-12:
                x = (ib - ia) / (sa - sb)
                if 0 < x < 1: intersections.add(float(x))
    points = sorted(intersections)
    intervals = []
    for lo, hi in zip(points[:-1], points[1:]):
        mid = .5 * (lo + hi)
        winner = max(methods, key=lambda m: params[m][0] + mid * params[m][1])
        if intervals and intervals[-1]["winner"] == winner:
            intervals[-1]["beta_max"] = hi
        else:
            intervals.append({"winner": winner, "beta_min": lo, "beta_max": hi})
    return pd.DataFrame(intervals), pd.DataFrame(
        [{"method_a": a, "method_b": b, "beta_intersection": (params[b][0]-params[a][0])/(params[a][1]-params[b][1])}
         for i, a in enumerate(methods) for b in methods[i+1:]
         if abs(params[a][1]-params[b][1]) > 1e-12
         and 0 <= (params[b][0]-params[a][0])/(params[a][1]-params[b][1]) <= 1])


def graph_visual():
    archive = np.load(DATASET / "data.npz")
    xy = archive["node_xy"]
    seed_dir = ROOT / "runs"
    methods = ["oracle_reference_graph", "euclidean_mst", "geometric_knn3_union",
               "correlation_abs_knn3_union", "proposed_final_decoded_graph"]
    titles = ["Reference (oracle)", "Euclidean MST", "Geometric 3-NN", "Correlation 3-NN", "Proposed graph"]
    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.2), sharex=True, sharey=True)
    rows = []
    for ax, method, title in zip(axes, methods, titles):
        edges = edge_set(seed_dir / f"{method}_seed42_l6_h12_e300/edges.csv")
        for u, v in edges:
            ax.plot(xy[[u, v], 0], xy[[u, v], 1], color=COLORS["learned"], lw=.7, alpha=.72)
            rows.append({"method": method, "u": u, "v": v, "x_u": xy[u,0], "y_u": xy[u,1],
                         "x_v": xy[v,0], "y_v": xy[v,1]})
        ax.scatter(xy[:, 0], xy[:, 1], s=7, color="black", zorder=3)
        ax.set_xlabel(title); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    pd.DataFrame(rows).to_csv(FIG / "data/graph_visual_comparison.csv", index=False)
    save_figure(fig, FIG, "graph_visual_comparison"); plt.close(fig)


def main():
    apply_style(); (FIG / "data").mkdir(parents=True, exist_ok=True)
    rows = prepare_rows()
    rows.to_csv(ROOT / "all_method_seed_metrics.csv", index=False)
    summary = aggregate(rows, ["method"])
    node_mse = summary.loc[summary.method == "node_only_empty_graph", "test_mse_mean"].iloc[0]
    oracle_mse = summary.loc[summary.method == "oracle_reference_graph", "test_mse_mean"].iloc[0]
    summary["improvement_vs_node_pct"] = 100 * (node_mse - summary.test_mse_mean) / node_mse
    summary["gap_from_oracle_mse"] = summary.test_mse_mean - oracle_mse
    summary.to_csv(FIG / "data/method_summary.csv", index=False)

    topology = summary.copy()
    topology["graph_construction_rule"] = topology.method.map(lambda m: META[m][0])
    topology["trainable"] = topology.method.map(lambda m: META[m][1])
    topology["uses_coordinates"] = topology.method.map(lambda m: META[m][2])
    topology["uses_signals"] = topology.method.map(lambda m: META[m][3])
    topology["uses_reference_topology"] = topology.method.map(lambda m: META[m][4])
    table_a = topology[["method", "graph_construction_rule", "trainable", "uses_coordinates", "uses_signals",
                        "uses_reference_topology", "edge_count_mean", "components_mean", "precision_mean",
                        "recall_mean", "f1_mean", "jaccard_mean"]]
    save_table(table_a, "table_A_topology_only_baselines")

    table_b = summary[["method", "test_mse_mean", "test_mse_std", "test_mae_mean", "test_mae_std",
                       "test_rmse_mean", "improvement_vs_node_pct", "gap_from_oracle_mse"]].copy()
    table_b.insert(1, "lags", 6); table_b.insert(2, "horizon", 12); table_b["seeds"] = 5
    save_table(table_b, "table_B_fixed_graph_forecasting")

    curve_rows = []
    for method, frame in rows.groupby("method"):
        for beta in BETAS:
            values = frame[f"joint_beta_{beta:g}"]
            curve_rows.append({"method": method, "beta": beta, "joint_score_mean": values.mean(),
                               "joint_score_std": values.std(), "task_score_mean": frame.task_score.mean(),
                               "task_score_std": frame.task_score.std(), "f1_mean": frame.f1.mean(),
                               "f1_std": frame.f1.std()})
    curves = pd.DataFrame(curve_rows)
    curves["rank_all"] = curves.groupby("beta").joint_score_mean.rank(ascending=False, method="min").astype(int)
    deployable = [m for m in META if m not in ["oracle_reference_graph", "random_learned_edge_count_control",
                                                "proposed_learned_graph_frozen"]]
    curves["rank_deployable"] = np.nan
    mask = curves.method.isin(deployable)
    curves.loc[mask, "rank_deployable"] = curves[mask].groupby("beta").joint_score_mean.rank(
        ascending=False, method="min")
    curves.to_csv(FIG / "data/joint_score_beta.csv", index=False)
    save_table(curves, "table_joint_score_all_beta")

    wide = curves.pivot(index="method", columns="beta", values="joint_score_mean").reset_index()
    table_c = summary[["method"]].merge(wide, on="method")
    table_c = table_c.merge(curves[curves.beta == .5][["method", "task_score_mean", "f1_mean"]], on="method")
    for beta in [.25, .5, .75]:
        col = beta
        table_c[f"rank_beta_{beta:g}"] = table_c[col].rank(ascending=False, method="min").astype(int)
    table_c = table_c.rename(columns={.25: "joint_beta_0.25", .5: "joint_beta_0.5", .75: "joint_beta_0.75",
                                      "f1_mean": "graph_score", "task_score_mean": "task_score"})
    keep = ["method", "task_score", "graph_score", "joint_beta_0.25", "joint_beta_0.5", "joint_beta_0.75",
            "rank_beta_0.25", "rank_beta_0.5", "rank_beta_0.75"]
    save_table(table_c[keep], "table_C_joint_task_topology")

    interval_base = curves[curves.beta == 0][["method", "f1_mean", "task_score_mean"]]
    intervals, intersections = beta_intervals(interval_base, deployable)
    save_table(intervals, "best_method_beta_intervals_deployable")
    save_table(intersections, "curve_intersections_deployable")
    all_intervals, all_intersections = beta_intervals(interval_base, list(META))
    save_table(all_intervals, "best_method_beta_intervals_all_including_oracle")
    save_table(all_intersections, "curve_intersections_all_including_oracle")

    graph_visual()
    ordered = summary.sort_values("f1_mean")
    for metric, stem, ylabel in [("f1", "graph_f1_comparison", "Graph F1"),
                                 ("test_mse", "forecasting_error_comparison", "Test forecast MSE")]:
        fig, ax = plt.subplots(figsize=(7.0, 3.1)); x = np.arange(len(ordered))
        ax.bar(x, ordered[f"{metric}_mean"], yerr=ordered[f"{metric}_std"], capsize=3,
               color=COLORS["learned"] if metric == "test_mse" else COLORS["topology"])
        ax.set_xticks(x, [LABEL[m] for m in ordered.method], rotation=25, ha="right")
        ax.set_ylabel(ylabel); save_figure(fig, FIG, stem); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 3.1)); x = np.arange(len(summary))
    ax.bar(x, summary.improvement_vs_node_pct, color=np.where(summary.improvement_vs_node_pct >= 0,
           COLORS["objective"], COLORS["topology"]))
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, [LABEL[m] for m in summary.method], rotation=25, ha="right")
    ax.set_ylabel("Improvement over node-only (%)"); save_figure(fig, FIG, "improvement_over_node_only"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    primary = ["node_only_empty_graph", "euclidean_mst", "geometric_knn3_union",
               "correlation_abs_knn3_union", "distance_correlation_hybrid_knn3_union",
               "proposed_joint", "oracle_reference_graph"]
    for method in primary:
        frame = curves[curves.method == method]
        ax.plot(frame.beta, frame.joint_score_mean, marker="o", label=LABEL[method])
        ax.fill_between(frame.beta, frame.joint_score_mean-frame.joint_score_std.fillna(0),
                        frame.joint_score_mean+frame.joint_score_std.fillna(0), alpha=.08)
    ax.set(xlabel=r"Forecast weight $\beta$", ylabel=r"Post-hoc JointScore$_\beta$", xticks=BETAS)
    ax.legend(frameon=False, ncol=2); save_figure(fig, FIG, "joint_score_beta"); plt.close(fig)

    frontier = pareto_methods(summary, deployable)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for row in summary[summary.method.isin(primary)].itertuples():
        ax.errorbar(row.test_mse_mean, row.f1_mean, xerr=row.test_mse_std, yerr=row.f1_std,
                    fmt="o", capsize=3, color=COLORS["objective"] if row.method in frontier else COLORS["learned"])
        ax.annotate(LABEL[row.method], (row.test_mse_mean, row.f1_mean), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Test forecast MSE (better left)", ylabel="Graph F1 (better up)")
    save_figure(fig, FIG, "forecast_topology_pareto"); plt.close(fig)
    pd.DataFrame({"pareto_deployable_method": frontier}).to_csv(FIG / "data/pareto_frontier.csv", index=False)

    degree_rows = []
    for method in ["euclidean_mst", "geometric_knn3_union", "correlation_abs_knn3_union",
                   "proposed_final_decoded_graph", "oracle_reference_graph"]:
        edges = edge_set(ROOT / "runs" / f"{method}_seed42_l6_h12_e300/edges.csv")
        degree = np.zeros(50, dtype=int)
        for u, v in edges: degree[u] += 1; degree[v] += 1
        degree_rows.extend({"method": method, "node": i, "degree": d} for i, d in enumerate(degree))
    degree_frame = pd.DataFrame(degree_rows); degree_frame.to_csv(FIG / "data/degree_distribution_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for method, frame in degree_frame.groupby("method"):
        values, counts = np.unique(frame.degree, return_counts=True)
        ax.plot(values, counts, marker="o", label=LABEL.get(method, method))
    ax.set(xlabel="Node degree", ylabel="Number of nodes"); ax.legend(frameon=False, ncol=2)
    save_figure(fig, FIG, "degree_distribution_comparison"); plt.close(fig)
    print(f"[OK] methods={summary.method.nunique()} tables={len(list(TABLE.glob('*.csv')))} figure_sets={len(list((FIG/'png').glob('*.png')))}")


if __name__ == "__main__":
    main()
