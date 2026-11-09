"""Aggregate the internally comparable candidate-set-6 k_DGM=1..6 sweep."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
FINAL = PROJECT / "results_section_v_final"
RESULTS = PROJECT / "results_section_v"
MASTER = FINAL / "metric_corrected_runs/corrected_joint_sweeps.csv"
NEW = FINAL / "metric_corrected_runs/k_dgm_candidate6_extended"
OUT = FINAL / "k_dgm_sensitivity_candidate6_extended"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
K_VALUES = list(range(1, 7))
SEEDS = [42, 43, 44]


def source_map():
    master = pd.read_csv(MASTER)
    old = master[(master.group.eq("k_dgm")) &
                 (master.k_candidate.eq(6))].copy()
    sources = {}
    for row in old.itertuples(index=False):
        sources[(int(row.k_dgm), int(row.training_seed))] = (
            PROJECT / row.output_dir, float(row.runtime_seconds), "reused")
    for seed in SEEDS:
        name = f"k_dgm_extended_g211_s1211_t{seed}_l6_h12_a0p80_kd6_kc6_e120"
        run = NEW / name
        elapsed = float((run / "elapsed_seconds.txt").read_text())
        sources[(6, seed)] = (run, elapsed, "new")
    expected = {(k, seed) for k in K_VALUES for seed in SEEDS}
    if set(sources) != expected:
        raise RuntimeError(f"Incomplete sweep: missing {expected-set(sources)}")
    return sources


def metric_row(k, seed, run, runtime, status):
    prediction = np.load(run / "test_predictions.npz")
    error = prediction["y_pred"].astype(float) - prediction["y_true"].astype(float)
    mse = float(np.mean(error * error))
    comparison = json.loads((run / "graph_comparison_softprob/comparison_summary.json").read_text())
    rec = comparison["edge_recovery_undirected"]
    graph = comparison["learned_undirected"]
    gt = comparison["ground_truth_undirected"]
    if int(gt["num_edges"]) != 79:
        raise RuntimeError(f"Wrong graph in {run}")
    prob = np.load(run / "dgm_prob_mean.npy")
    candidate_counts = (prob > 0).sum(axis=1)
    counts = comparison["counts_undirected"]
    return {
        "k_dgm": k, "training_seed": seed, "graph_seed": 211,
        "signal_seed": 1211, "L": 6, "F": 12, "alpha": 0.8,
        "candidate_set_size": 6,
        "minimum_effective_candidates": int(candidate_counts.min()),
        "maximum_effective_candidates": int(candidate_counts.max()), "k_q": 2,
        "test_mse": mse, "test_rmse": float(np.sqrt(mse)),
        "test_mae": float(np.mean(np.abs(error))),
        "precision": float(rec["precision"]), "recall": float(rec["recall"]),
        "f1": float(rec["f1"]), "jaccard": float(rec["jaccard"]),
        "tp": int(counts["correct_edges"]), "fp": int(counts["extra_edges"]),
        "fn": int(counts["missing_edges"]),
        "inferred_edges": int(graph["num_edges"]),
        "graph_density": float(graph["density"]),
        "mean_degree": float(graph["degree_mean"]),
        "connected_components": int(graph["num_connected_components_undirected"]),
        "training_time_seconds": runtime, "mean_epoch_time_seconds": runtime/120,
        "source_status": status, "run_directory": str(run),
    }


def aggregate(per_seed):
    metrics = ["test_mse", "test_rmse", "test_mae", "precision", "recall",
               "f1", "jaccard", "tp", "fp", "fn", "inferred_edges",
               "graph_density", "mean_degree", "connected_components",
               "training_time_seconds", "mean_epoch_time_seconds"]
    grouped = per_seed.groupby("k_dgm", sort=True)
    parts = []
    for metric in metrics:
        parts.append(grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    out = pd.concat(parts, axis=1).reset_index()
    out["number_of_seeds"] = grouped.size().values
    return out


def display_table(agg):
    spec = [("Test MSE", "test_mse", "min"),
            ("Test MAE", "test_mae", "min"),
            ("Precision", "precision", "max"),
            ("Recall", "recall", "max"), ("F1", "f1", "max"),
            ("Jaccard", "jaccard", "max"),
            ("Inferred edges", "inferred_edges", None)]
    winners = {}
    for _, metric, direction in spec:
        if direction == "min": winners[metric] = agg[f"{metric}_mean"].min()
        if direction == "max": winners[metric] = agg[f"{metric}_mean"].max()
    rows = []
    for row in agg.itertuples(index=False):
        values = {"k_DGM": int(row.k_dgm)}
        for label, metric, direction in spec:
            mean, std = getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_std")
            digits = 2 if metric == "inferred_edges" else 4
            text = f"{mean:.{digits}f} +/- {std:.{digits}f}"
            if direction and np.isclose(mean, winners[metric]):
                text = f"**{text}**"
            values[label] = text
        rows.append(values)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = source_map()
    per_seed = pd.DataFrame([
        metric_row(k, seed, *sources[(k, seed)])
        for k in K_VALUES for seed in SEEDS
    ]).sort_values(["k_dgm", "training_seed"])
    if not per_seed.minimum_effective_candidates.ge(per_seed.k_dgm).all():
        raise RuntimeError("Inadmissible k_DGM detected")
    agg = aggregate(per_seed)
    display = display_table(agg)
    per_seed.to_csv(OUT / "k_dgm_1_to_6_per_seed.csv", index=False)
    agg.to_csv(OUT / "k_dgm_1_to_6_aggregated.csv", index=False)
    display.to_csv(OUT / "k_dgm_1_to_6_compact.csv", index=False)
    headers = [str(column) for column in display.columns]
    md_lines = ["# k_DGM=1..6, candidate-set size 6", "",
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in display.itertuples(index=False, name=None):
        md_lines.append("| " + " | ".join(map(str, row)) + " |")
    (OUT / "k_dgm_1_to_6_table.md").write_text(
        chr(10).join(md_lines) + chr(10), encoding="utf-8")
    config = {
        "scenario": "homogeneous_relational_g211_s1211",
        "graph_seed": 211, "signal_seed": 1211, "training_seeds": SEEDS,
        "nodes": 50, "reference_edges": 79, "L": 6, "F": 12,
        "alpha": 0.8, "candidate_set_size": 6,
        "minimum_effective_candidates_observed": int(per_seed.minimum_effective_candidates.min()),
        "maximum_effective_candidates_observed": int(per_seed.maximum_effective_candidates.max()),
        "tested_k_dgm": K_VALUES, "k_q": 2,
        "decoder": "row_k=3, mutual, geometric MST",
        "consensus": "mean-probability soft consensus", "epochs": 120,
        "batch_size": 32, "learning_rate": 0.001,
        "checkpoint_selection": "observable_val",
        "reused_runs_k_1_to_5": 15, "new_runs_k_6": 3,
        "comparability_note": "Internally comparable candidate-set-6 sweep; not directly interchangeable with the nominal candidate-set-3 table.",
    }
    (OUT / "k_dgm_1_to_6_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(display.to_string(index=False))
    print(f"[DONE] {OUT}")


if __name__ == "__main__":
    main()
