"""Recalculate the nominal candidate-set-3 k_DGM sensitivity tables."""
from pathlib import Path
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
OUT = FINAL / "k_dgm_sensitivity_nominal_kc3"
NEW_ROOT = FINAL / "metric_corrected_runs/k_dgm_nominal_kc3"
MASTER = FINAL / "metric_corrected_runs/corrected_joint_sweeps.csv"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
SEEDS = [42, 43, 44]
K_VALUES = [1, 2, 3]
sys.path.insert(0, str(PROJECT / "scripts/validation"))
import run_campaign_v2 as campaign  # noqa: E402


def run_sources():
    master = pd.read_csv(MASTER)
    nominal = master[(master.group.eq("lh")) & (master.lags.eq(6)) &
                     (master.horizon.eq(12)) & (master.alpha.eq(0.8)) &
                     (master.k_dgm.eq(2)) & (master.k_candidate.eq(3))]
    sources = {}
    for row in nominal.itertuples(index=False):
        sources[(2, int(row.training_seed))] = (
            PROJECT / row.output_dir, float(row.runtime_seconds), "reused")
    for k in [1, 3]:
        for seed in SEEDS:
            name = f"k_dgm_nominal_g211_s1211_t{seed}_l6_h12_a0p80_kd{k}_kc3_e120"
            run = NEW_ROOT / name
            elapsed = float((run / "elapsed_seconds.txt").read_text())
            sources[(k, seed)] = (run, elapsed, "new")
    if set(sources) != {(k, s) for k in K_VALUES for s in SEEDS}:
        raise RuntimeError("Incomplete source map")
    return sources


def canonical_edges(array):
    a = np.asarray(array)
    if a.shape[0] == 2:
        a = a.T
    return {(min(int(u), int(v)), max(int(u), int(v)))
            for u, v in a if int(u) != int(v)}


def row_for(k, seed, run, runtime, source_status):
    prediction = np.load(run / "test_predictions.npz")
    error = (prediction["y_pred"].astype(float) -
             prediction["y_true"].astype(float))
    mse = float(np.mean(error * error))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    comparison = json.loads((run / "graph_comparison_softprob/comparison_summary.json").read_text())
    rec = comparison["edge_recovery_undirected"]
    graph_stats = comparison["learned_undirected"]
    gt = comparison["ground_truth_undirected"]
    if int(gt["num_edges"]) != 79 or int(gt["num_connected_components_undirected"]) != 1:
        raise RuntimeError(f"Wrong reference graph in {run}")
    learned = np.load(run / "graph_comparison_softprob/learned_edge_index_undirected.npy")
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    graph.add_edges_from(canonical_edges(learned))
    degrees = np.array([degree for _, degree in graph.degree()], dtype=int)
    forecast = json.loads((run / "forecast_metrics.json").read_text())
    selected_epoch = int(forecast["selected_epoch"])
    validation = (forecast.get("best_observable_validation") or
                  forecast.get("best_validation") or {})
    history_path = run / "training_history.csv"
    if not history_path.exists():
        campaign.parse_history(run / "training.log", history_path)
    history = pd.read_csv(history_path)
    final = history.iloc[-1]
    prob = np.load(run / "dgm_prob_mean.npy")
    candidate_counts = (prob > 0).sum(axis=1)
    positive_directed = int((prob > 0).sum())
    positive_undirected = len(canonical_edges(np.argwhere(prob > 0)))
    counts = comparison["counts_undirected"]
    return {
        "k_dgm": k, "training_seed": seed, "graph_seed": 211,
        "signal_seed": 1211, "L": 6, "F": 12, "alpha": 0.8,
        "candidate_set_size": 3, "minimum_effective_candidates": int(candidate_counts.min()),
        "maximum_effective_candidates": int(candidate_counts.max()), "k_q": 2,
        "decoder": "row_k=3, mutual, geometric MST",
        "consensus": "mean-probability soft consensus", "test_mse": mse,
        "test_mae": mae, "test_rmse": rmse,
        "validation_forecasting_error_internal": float(validation.get("mse", np.nan)),
        "precision": float(rec["precision"]), "recall": float(rec["recall"]),
        "f1": float(rec["f1"]), "jaccard": float(rec["jaccard"]),
        "tp": int(counts["correct_edges"]), "fp": int(counts["extra_edges"]),
        "fn": int(counts["missing_edges"]),
        "inferred_edges": int(graph_stats["num_edges"]),
        "graph_density": float(graph_stats["density"]),
        "mean_degree": float(degrees.mean()), "minimum_degree": int(degrees.min()),
        "maximum_degree": int(degrees.max()),
        "connected_components": int(nx.number_connected_components(graph)),
        "isolated_nodes": int(np.count_nonzero(degrees == 0)),
        "dynamic_directed_relation_budget_per_window": int(50 * k),
        "positive_mean_probability_directed_pairs": positive_directed,
        "positive_mean_probability_undirected_pairs": positive_undirected,
        "final_forecasting_loss": float(final.get("train_task", np.nan)),
        "final_structural_loss": float(final.get("train_topology", np.nan)),
        "training_time_seconds": runtime,
        "mean_epoch_time_seconds": runtime / 120.0,
        "peak_memory_mb": np.nan, "best_epoch": selected_epoch,
        "source_status": source_status, "run_directory": str(run),
    }


def aggregate(per_seed):
    metrics = [
        "test_mse", "test_mae", "test_rmse",
        "validation_forecasting_error_internal", "precision", "recall", "f1",
        "jaccard", "tp", "fp", "fn", "inferred_edges", "graph_density",
        "mean_degree", "minimum_degree", "maximum_degree",
        "connected_components", "isolated_nodes",
        "dynamic_directed_relation_budget_per_window",
        "positive_mean_probability_directed_pairs",
        "positive_mean_probability_undirected_pairs", "final_forecasting_loss",
        "final_structural_loss", "training_time_seconds",
        "mean_epoch_time_seconds", "best_epoch",
    ]
    grouped = per_seed.groupby("k_dgm", sort=True)
    parts = []
    for metric in metrics:
        parts.append(grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    result = pd.concat(parts, axis=1).reset_index()
    result["number_of_seeds"] = grouped.size().values
    return result


def compact(agg):
    metrics = ["test_mse", "test_mae", "precision", "recall", "f1",
               "jaccard", "inferred_edges"]
    cols = ["k_dgm"]
    for metric in metrics:
        cols += [f"{metric}_mean", f"{metric}_std"]
    return agg[cols].copy()


def formatted_tables(agg):
    spec = [
        ("Test MSE", "test_mse", "min"), ("Test MAE", "test_mae", "min"),
        ("Precision", "precision", "max"), ("Recall", "recall", "max"),
        ("F1", "f1", "max"), ("Jaccard", "jaccard", "max"),
        ("Inferred edges", "inferred_edges", None),
    ]
    winners = {}
    for _, key, direction in spec:
        if direction == "min":
            winners[key] = agg[f"{key}_mean"].min()
        elif direction == "max":
            winners[key] = agg[f"{key}_mean"].max()
    rows = []
    for r in agg.itertuples(index=False):
        row = {"k_DGM": int(r.k_dgm)}
        for label, key, direction in spec:
            mean = getattr(r, f"{key}_mean")
            std = getattr(r, f"{key}_std")
            digits = 2 if key == "inferred_edges" else 4
            value = f"{mean:.{digits}f} +/- {std:.{digits}f}"
            if direction and np.isclose(mean, winners[key]):
                value = f"**{value}**"
            row[label] = value
        rows.append(row)
    display = pd.DataFrame(rows)
    full = agg[["k_dgm", "test_mse_mean", "test_mse_std", "test_mae_mean",
        "test_mae_std", "precision_mean", "precision_std", "recall_mean",
        "recall_std", "f1_mean", "f1_std", "jaccard_mean", "jaccard_std",
        "inferred_edges_mean", "inferred_edges_std", "mean_degree_mean",
        "mean_degree_std", "connected_components_mean",
        "connected_components_std", "training_time_seconds_mean",
        "training_time_seconds_std", "number_of_seeds"]]
    return display, full


def latex_table(display):
    columns = list(display.columns)
    lines = ["\\begin{tabular}{r" + "c" * (len(columns)-1) + "}",
             "\\toprule", " & ".join(columns).replace("_", "\\_") + " " + chr(92) * 2,
             "\\midrule"]
    for row in display.itertuples(index=False, name=None):
        cells = []
        for value in row:
            text = str(value).replace("+/-", "$\\pm$")
            if text.startswith("**") and text.endswith("**"):
                text = "\\textbf{" + text[2:-2] + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + " " + chr(92) * 2)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def classify(agg):
    # Conservative: mean differences are small relative to three-seed variability.
    return {
        "A": "not supported", "B": "insufficient evidence",
        "C": "partially supported", "D": "partially supported",
        "E": "not supported",
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / "meta.json").read_text())
    sources = run_sources()
    rows = [row_for(k, seed, *sources[(k, seed)])
            for k in K_VALUES for seed in SEEDS]
    per_seed = pd.DataFrame(rows).sort_values(["k_dgm", "training_seed"])
    if not per_seed.minimum_effective_candidates.ge(per_seed.k_dgm).all():
        raise RuntimeError("k_DGM exceeds effective candidate count")
    agg = aggregate(per_seed)
    compact_df = compact(agg)
    display, full = formatted_tables(agg)
    hypotheses = classify(agg)
    per_seed.to_csv(OUT / "k_dgm_results_per_seed.csv", index=False)
    agg.to_csv(OUT / "k_dgm_results_aggregated.csv", index=False)
    compact_df.to_csv(OUT / "k_dgm_results_compact.csv", index=False)
    md = "# k_DGM sensitivity\n\n" + full.to_markdown(index=False) + "\n\n## Compact\n\n" + display.to_markdown(index=False) + "\n"
    (OUT / "k_dgm_results_table.md").write_text(md, encoding="utf-8")
    (OUT / "k_dgm_results_table.tex").write_text(latex_table(display), encoding="utf-8")
    config = {
        "scenario": "homogeneous_relational_g211_s1211", "graph_seed": 211,
        "signal_seed": 1211, "training_seeds": SEEDS, "nodes": 50,
        "reference_edges": 79, "reference_connected_components": 1,
        "communities_imposed": False, "splits": meta["splits"],
        "preprocessing": "stride-1 windows contained within each split",
        "scaler": "per-feature z-score fitted on training interval only",
        "L": 6, "F": 12, "alpha": 0.8, "candidate_set_size": 3,
        "minimum_effective_candidates_observed": int(per_seed.minimum_effective_candidates.min()),
        "maximum_effective_candidates_observed": int(per_seed.maximum_effective_candidates.max()),
        "admissible_k_dgm": K_VALUES, "tested_k_dgm": K_VALUES, "k_q": 2,
        "candidate_mode": "q_dynamic", "decoder": "row_k=3, mutual, geometric MST",
        "consensus": "mean-probability soft consensus", "epochs": 120,
        "batch_size": 32, "learning_rate": 0.001,
        "optimizer": "Adam (training implementation default)",
        "architecture": {"feature_function": "MLP", "graph_function": "GCN"},
        "checkpoint_selection": "observable_val", "task_loss": "Huber delta=1.0",
        "candidate_mask_assigned": True, "self_loops": False,
        "reference_graph_usage": "post-hoc topology evaluation only",
        "stability_term_enabled": False,
        "reused_runs": 3, "new_runs": 6,
        "excluded_noncomparable_sweep": "candidate-set-6 k_DGM=1..5 sweep",
        "peak_memory": "not logged; unavailable",
        "hypotheses": hypotheses,
    }
    (OUT / "k_dgm_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    a = agg.set_index("k_dgm")
    readme = f"""# k_DGM sensitivity with nominal candidate set

Only the homogeneous connected g211/s1211 scenario is used. The nominal
candidate-set size is 3; positive mean-probability rows contain at least
{int(per_seed.minimum_effective_candidates.min())} candidates, so the admissible
values are k_DGM=1,2,3. The older candidate-set-6 sweep is excluded.

Fixed settings: L=6, F=12, alpha=0.8, k_Q=2, 120 epochs, batch size 32,
learning rate 0.001, MLP/GCN architecture, observable-validation checkpoint
selection, mean-probability consensus, and row-k=3 mutual decoder with MST.
Training seeds are 42, 43, and 44. Three nominal k_DGM=2 runs were reused and
six runs (k_DGM=1,3) were executed.

Test MSE/MAE/RMSE are recomputed from aligned test predictions. Topology
metrics use the reference graph only post-hoc. No joint score is calculated.
The stored validation forecasting field is the internal normalized validation
task metric, not the independently recomputed element-wise test MSE. Peak
memory was not logged and remains unavailable.

Final decoded edge counts are {a.loc[1,'inferred_edges_mean']:.2f},
{a.loc[2,'inferred_edges_mean']:.2f}, and {a.loc[3,'inferred_edges_mean']:.2f}
for k_DGM=1,2,3. Connected components remain
{a.loc[1,'connected_components_mean']:.0f},
{a.loc[2,'connected_components_mean']:.0f}, and
{a.loc[3,'connected_components_mean']:.0f}. Mean training times are
{a.loc[1,'training_time_seconds_mean']:.2f},
{a.loc[2,'training_time_seconds_mean']:.2f}, and
{a.loc[3,'training_time_seconds_mean']:.2f} seconds.

Hypotheses: A {hypotheses['A']}; B {hypotheses['B']}; C {hypotheses['C']};
D {hypotheses['D']}; E {hypotheses['E']}.

Regenerate metrics and tables from saved runs with:

```bash
python scripts/validation_supplement/k_dgm_sensitivity_nominal_kc3/analyze_k_dgm_sensitivity.py
```
"""
    (OUT / "README_K_DGM_SENSITIVITY.md").write_text(readme, encoding="utf-8")
    print(display.to_string(index=False))
    print("Hypotheses", hypotheses)
    print(f"[DONE] {OUT}")


if __name__ == "__main__":
    main()
