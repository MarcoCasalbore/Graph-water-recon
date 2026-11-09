"""Build the paper-ready alpha sensitivity analysis for the homogeneous v3 scenario.

Forecast errors are checked from aligned test predictions, topology metrics are
checked against the soft mean-probability decoder outputs, and all joint scores
are post-hoc only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 1.0]
BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = [42, 43, 44]
METRICS = [
    "test_mse", "test_rmse", "test_mae", "precision", "recall", "f1",
    "jaccard", "true_positives", "false_positives", "false_negatives",
    "inferred_edge_count", "graph_density", "mean_degree",
    "connected_components", "TaskScore", "TopologyScore", "JointScore_0.5",
]


def parse_args():
    here = Path(__file__).resolve()
    project = here.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--source", type=Path,
                        default=project / "results_section_v_final/metric_corrected_runs/corrected_joint_sweeps.csv")
    parser.add_argument("--out", type=Path,
                        default=project / "results_section_v_final/alpha_sensitivity_posthoc")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def close(a, b):
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=1e-9))


def load_and_verify(project, source):
    raw = pd.read_csv(source)
    q = raw.loc[raw.group.eq("alpha")].sort_values(
        ["alpha", "training_seed"]).reset_index(drop=True)
    expected = {(a, s) for a in ALPHAS for s in SEEDS}
    actual = {(round(float(a), 8), int(s))
              for a, s in zip(q.alpha, q.training_seed)}
    if len(q) != 21 or actual != expected:
        raise RuntimeError(f"Incomplete 7x3 alpha sweep: rows={len(q)}, pairs={actual}")
    invariants = {"lags": 6, "horizon": 12, "k_dgm": 2,
                  "k_candidate": 3, "k_decoder": 3, "k_q": 2, "epochs": 120}
    for key, value in invariants.items():
        if sorted(q[key].dropna().unique().tolist()) != [value]:
            raise RuntimeError(f"Configuration mismatch for {key}")

    dataset = project / "results_section_v/campaign_v2/datasets/homogeneous_relational_g211_s1211"
    meta = json.loads((dataset / "meta.json").read_text(encoding="utf-8"))
    if not (meta["graph_seed"] == 211 and meta["signal_seed"] == 1211 and
            meta["num_nodes"] == 50 and meta["num_gt_undirected_edges"] == 79):
        raise RuntimeError(f"Unexpected dataset metadata: {meta}")

    rows = []
    for row in q.itertuples(index=False):
        run = project / row.output_dir
        if "results_section_v_final/metric_corrected_runs/alpha/" not in run.as_posix():
            raise RuntimeError(f"Rejected non-v3 source: {run}")
        pred_path = run / "test_predictions.npz"
        topo_path = run / "graph_comparison_softprob/comparison_summary.json"
        pred = np.load(pred_path)
        error = pred["y_pred"].astype(float) - pred["y_true"].astype(float)
        mse = float(np.mean(error * error))
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(mse))
        if not (close(mse, row.mse) and close(mae, row.mae) and close(rmse, row.rmse)):
            raise RuntimeError(f"Forecast metric mismatch: {run}")
        comp = json.loads(topo_path.read_text(encoding="utf-8"))
        gt = comp["ground_truth_undirected"]
        graph = comp["learned_undirected"]
        rec = comp["edge_recovery_undirected"]
        if int(gt["num_edges"]) != 79 or int(gt["num_connected_components_undirected"]) != 1:
            raise RuntimeError(f"Wrong reference graph: {run}")
        for key in ["precision", "recall", "f1", "jaccard"]:
            if not close(rec[key], getattr(row, key)):
                raise RuntimeError(f"Topology metric mismatch for {key}: {run}")
        task = 1.0 / (1.0 + mse)
        topo = float(rec["f1"])
        counts = comp["counts_undirected"]
        rows.append({
            "alpha": float(row.alpha), "training_seed": int(row.training_seed),
            "selected_epoch": int(row.selected_epoch), "test_mse": mse,
            "test_rmse": rmse, "test_mae": mae,
            "precision": float(rec["precision"]), "recall": float(rec["recall"]),
            "f1": topo, "jaccard": float(rec["jaccard"]),
            "true_positives": int(counts["correct_edges"]),
            "false_positives": int(counts["extra_edges"]),
            "false_negatives": int(counts["missing_edges"]),
            "inferred_edge_count": int(graph["num_edges"]),
            "graph_density": float(graph["density"]),
            "mean_degree": float(graph["degree_mean"]),
            "connected_components": int(graph["num_connected_components_undirected"]),
            "TaskScore": task, "TopologyScore": topo,
            "JointScore_0.5": 0.5 * task + 0.5 * topo,
            "run_directory": str(run), "prediction_source": str(pred_path),
            "topology_source": str(topo_path),
        })
    config = {
        "analysis_scope": "homogeneous connected relational scenario only",
        "dataset_directory": str(dataset), "source_sweep_csv": str(source),
        "graph_seed": 211, "signal_seed": 1211, "training_seeds": SEEDS,
        "alphas": ALPHAS, "betas": BETAS, "num_nodes": 50,
        "reference_undirected_edges": 79, "reference_connected_components": 1,
        "communities_imposed": False, "lags_L": 6, "forecast_horizon_F": 12,
        "splits": meta["splits"],
        "preprocessing": "sliding windows with stride 1; train/validation/test windows remain entirely inside their split",
        "scaler": "per-feature z-score using mean and standard deviation fitted on the training interval only",
        "k_dgm": 2, "candidate_set_size": 3, "k_q": 2,
        "decoder": "row_k=3, mutual symmetrization, geometric MST inclusion",
        "consensus": "mean probability; soft-probability exported graph",
        "epochs": 120, "validation_cadence_epochs": 5, "batch_size": 32,
        "optimizer": "Adam (training implementation default)",
        "learning_rate": 0.001,
        "architecture": {"feature_function": "MLP", "graph_function": "GCN"},
        "checkpoint_selection": "observable_val", "candidate_mode": "q_dynamic",
        "static_top_edges": 0,
        "training_loss": "alpha*normalized_forecasting + (1-alpha)*normalized_structural",
        "task_loss_internal": "Huber(delta=1.0), target-std normalization, weighted-mean features",
        "lambda_graph": 0.0, "lambda_soft_pseudo_edge": 0.5,
        "lambda_stability": 0.0, "pseudo_score_mode": "observable_expert_full",
        "pseudo_static_target": True, "pseudo_topk": 2, "pseudo_include_mst": True,
        "forecast_metric": "true element-wise test MSE/MAE recomputed from aligned test_predictions.npz",
        "topology_metric_timing": "post-hoc only against the 79-edge reference graph",
        "joint_score_role": "post-hoc descriptive only; unused by training, checkpointing, decoding, consensus, or alpha selection",
        "joint_score_formula": "S_beta = beta/(1+test_MSE) + (1-beta)*F1",
        "main_beta": 0.5,
    }
    return pd.DataFrame(rows), config


def aggregate(per_seed):
    grouped = per_seed.groupby("alpha", sort=True)
    parts = []
    for metric in METRICS:
        parts.append(grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    out = pd.concat(parts, axis=1).reset_index()
    out["n_seeds"] = grouped.size().values
    return out


def beta_outputs(per_seed, agg):
    rows = []
    for beta in BETAS:
        for row in per_seed.itertuples(index=False):
            rows.append({"alpha": row.alpha, "beta": beta,
                         "training_seed": row.training_seed,
                         "TaskScore": row.TaskScore,
                         "TopologyScore": row.TopologyScore,
                         "S_beta": beta * row.TaskScore + (1-beta) * row.TopologyScore})
    seed_grid = pd.DataFrame(rows)
    grid = seed_grid.groupby(["alpha", "beta"], as_index=False).agg(
        S_beta_mean=("S_beta", "mean"), S_beta_std=("S_beta", "std"),
        n_seeds=("S_beta", "size"))
    grid["rank"] = grid.groupby("beta").S_beta_mean.rank(
        method="min", ascending=False).astype(int)
    ranking = grid.sort_values(["beta", "rank", "alpha"]).reset_index(drop=True)
    matrix = grid.pivot(index="alpha", columns="beta", values="S_beta_mean").reset_index()
    matrix.columns = ["alpha"] + [f"beta_{float(c):g}" for c in matrix.columns[1:]]

    means = agg[["alpha", "TaskScore_mean", "TopologyScore_mean"]].copy()
    cuts = {0.0, 1.0}
    values = means.to_dict("records")
    for i, left in enumerate(values):
        for right in values[i+1:]:
            dl = left["TaskScore_mean"] - left["TopologyScore_mean"]
            dr = right["TaskScore_mean"] - right["TopologyScore_mean"]
            if abs(dl-dr) < 1e-14:
                continue
            beta = ((right["TopologyScore_mean"] - left["TopologyScore_mean"])
                    / (dl-dr))
            if 0 < beta < 1:
                cuts.add(float(beta))
    cuts = sorted(cuts)
    segments = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo+hi)/2
        score = (means.TopologyScore_mean +
                 mid*(means.TaskScore_mean-means.TopologyScore_mean))
        winner = float(means.loc[score.idxmax(), "alpha"])
        if segments and segments[-1]["best_alpha"] == winner:
            segments[-1]["beta_end"] = hi
        else:
            segments.append({"beta_start": lo, "beta_end": hi,
                             "best_alpha": winner})
    return seed_grid, grid, ranking, matrix, pd.DataFrame(segments)


def configure_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update({"font.family": "serif", "font.serif": [serif],
        "font.size": 11, "axes.labelsize": 12.5, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "legend.fontsize": 10.5, "axes.linewidth": .9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none"})
    return serif


def plot(agg, out):
    serif = configure_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.65))
    specs = [("TaskScore", "Task score", "#5B3A78", "o"),
             ("TopologyScore", "Topology score", "#A64B3C", "s"),
             ("JointScore_0.5", r"Joint score ($\beta=0.5$)", "#146B73", "^")]
    for key, label, color, marker in specs:
        ax.errorbar(agg.alpha, agg[f"{key}_mean"], yerr=agg[f"{key}_std"],
                    label=label, color=color, marker=marker, linewidth=2.2,
                    markersize=6.5, markerfacecolor="white", markeredgewidth=1.4,
                    capsize=2.8, elinewidth=.9, capthick=.9, zorder=3)
    ax.set(xlabel=r"Forecast weight $\alpha$",
           ylabel="Normalized performance score", xlim=(-.025, 1.025), ylim=(0, 1))
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels(["0", "0.1", "0.25", "0.5", "0.7", "0.85", "1"])
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(True, color="#C8C8C8", linewidth=.55, alpha=.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#333333"); spine.set_linewidth(.9)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor="#666666", framealpha=.95, borderpad=.55, handlelength=2)
    fig.subplots_adjust(left=.13, right=.985, bottom=.19, top=.98)
    stem = out / "alpha_task_topology_tradeoff"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)
    return serif


def write_readme(out, agg, envelope, serif):
    best = lambda metric: agg.loc[agg[f"{metric}_mean"].idxmax()]
    task, topo, joint = best("TaskScore"), best("TopologyScore"), best("JointScore_0.5")
    stable = agg.loc[agg["JointScore_0.5_mean"] >=
                     joint["JointScore_0.5_mean"]-joint["JointScore_0.5_std"], "alpha"].tolist()
    env = "; ".join(f"beta in [{r.beta_start:.4f}, {r.beta_end:.4f}]: alpha={r.best_alpha:g}"
                    for r in envelope.itertuples(index=False))
    text = f"""# Alpha sensitivity: task--topology trade-off

Only the homogeneous connected relational scenario is used: graph seed 211,
signal seed 1211, N=50, 79 reference edges, and no imposed communities. Three
training seeds (42, 43, 44) are aggregated for every alpha in {ALPHAS}. No
legacy `water_base` result is consumed.

`alpha` is the training balance parameter in
`L_train = alpha*L_forecast_bar + (1-alpha)*L_struct_bar`. `beta` is strictly
post-hoc: `S_beta = beta*S_task + (1-beta)*S_topo`, with
`S_task=1/(1+test_MSE)` and `S_topo=F1`. The plotted joint score uses beta=0.5
and was not used for training, checkpoint selection, early stopping, decoder
selection, consensus selection, or alpha selection.

MSE/RMSE/MAE are true element-wise test errors independently recomputed from
aligned `test_predictions.npz`; legacy Huber-labelled MSE fields are excluded.
Topology metrics are post-hoc measurements from the soft mean-probability
decoder against the 79-edge reference graph.

Fixed configuration: L=6, F=12, k_DGM=2, candidate-set size=3, k_Q=2,
row-k=3 mutual decoder with geometric MST inclusion, mean-probability
consensus, 120 epochs, batch size 32, learning rate 0.001, MLP feature
function, GCN graph function, and observable-validation checkpoint selection.

## Findings

- Highest mean Task score: alpha={task.alpha:g} ({task.TaskScore_mean:.6f}).
- Highest mean Topology score: alpha={topo.alpha:g} ({topo.TopologyScore_mean:.6f}).
- Highest mean post-hoc Joint score at beta=0.5: alpha={joint.alpha:g} ({joint["JointScore_0.5_mean"]:.6f}).
- Values within one standard deviation of the maximum Joint score: alpha={stable}.
- Mean-score upper envelope: {env}.
- Alpha=0 is forecasting-imbalanced. Alpha=1 increases inferred edge count and
  reduces precision relative to the best balanced region. Alpha=0.7--1.0 is
  comparatively stable in Joint score, but precision/recall and graph size vary.

The figure uses {serif}; Times New Roman is selected automatically when available.
Run this script normally to rebuild all outputs, or add `--plot-only` to
regenerate the figure only from `alpha_sensitivity_aggregated.csv`.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    project, source, out = args.project.resolve(), args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    agg_path = out / "alpha_sensitivity_aggregated.csv"
    if args.plot_only:
        print(f"[DONE] Figure regenerated using {plot(pd.read_csv(agg_path), out)}")
        return
    per_seed, config = load_and_verify(project, source)
    agg = aggregate(per_seed)
    seed_grid, beta_grid, ranking, matrix, envelope = beta_outputs(per_seed, agg)
    per_seed.to_csv(out / "alpha_sensitivity_per_seed.csv", index=False)
    agg.to_csv(agg_path, index=False)
    seed_grid.to_csv(out / "alpha_beta_scores_per_seed.csv", index=False)
    beta_grid.to_csv(out / "alpha_beta_grid.csv", index=False)
    ranking.to_csv(out / "alpha_rankings_by_beta.csv", index=False)
    matrix.to_csv(out / "alpha_beta_score_matrix.csv", index=False)
    envelope.to_csv(out / "alpha_beta_best_envelope.csv", index=False)
    serif = plot(agg, out)
    config.update({"font_used": serif, "run_count": len(per_seed),
                   "seeds_per_alpha": 3})
    (out / "alpha_sensitivity_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    write_readme(out, agg, envelope, serif)
    print(agg[["alpha", "test_mse_mean", "test_mae_mean", "precision_mean",
        "recall_mean", "f1_mean", "jaccard_mean", "inferred_edge_count_mean",
        "TaskScore_mean", "TopologyScore_mean", "JointScore_0.5_mean"]].to_string(index=False))
    print("\nBeta ranking winners:")
    print(ranking.loc[ranking["rank"].eq(1),
        ["beta", "alpha", "S_beta_mean"]].to_string(index=False))
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
