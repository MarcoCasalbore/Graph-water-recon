"""Complete the dense alpha grid and print corrected aggregate numbers."""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
FINAL = PROJECT / "results_section_v_final"
RESULTS = PROJECT / "results_section_v"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
MASTER = FINAL / "metric_corrected_runs/corrected_joint_sweeps.csv"
RUN_ROOT = FINAL / "metric_corrected_runs/alpha_dense"
OUT = FINAL / "alpha_sensitivity_dense_numeric"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 43, 44]

sys.path.insert(0, str(PROJECT / "scripts/validation"))
import run_campaign_v2 as campaign  # noqa: E402
import run_corrected_metric_sweeps as corrected  # noqa: E402


def slug(value):
    return f"{value:.2f}".replace(".", "p")


def existing_rows():
    source = pd.read_csv(MASTER)
    return source[source.group.eq("alpha")].copy()


def run_missing(alpha, seed):
    name = f"alpha_dense_g211_s1211_t{seed}_l6_h12_a{slug(alpha)}_kd2_kc3_e120"
    out = RUN_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "test_predictions.npz"
    topology_marker = out / "graph_comparison_softprob/comparison_summary.json"
    elapsed_file = out / "elapsed_seconds.txt"
    command = campaign.joint_command(out, name, 6, 12, alpha, 2, 3, seed, 120)
    command[command.index("--data_dir") + 1] = str(DATA)
    config = {
        "run_id": name, "graph_seed": 211, "signal_seed": 1211,
        "training_seed": seed, "alpha": alpha, "lags": 6, "horizon": 12,
        "k_dgm": 2, "candidate_set_size": 3, "k_q": 2,
        "decoder": "row_k3_mutual_mst", "consensus": "mean_probability_soft",
        "epochs": 120, "checkpoint_selection": "observable_val",
        "dataset": str(DATA), "command": command,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not (marker.exists() and topology_marker.exists()):
        started = time.perf_counter()
        with (out / "training.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - started
        elapsed_file.write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    else:
        elapsed = float(elapsed_file.read_text()) if elapsed_file.exists() else 0.0
    return corrected.summarize(out, "alpha_dense", 6, 12, alpha, 2, 3,
                               seed, elapsed)


def main():
    existing = existing_rows()
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            match = existing[np.isclose(existing.alpha, alpha) &
                             existing.training_seed.eq(seed)]
            if len(match):
                row = match.iloc[0].to_dict()
                row["source_status"] = "reused_existing_corrected_run"
            else:
                row = run_missing(alpha, seed)
                row["source_status"] = "new_dense_alpha_run"
            rows.append(row)
            print(f"[OK] alpha={alpha:.1f} seed={seed} mse={row['mse']:.6f} "
                  f"f1={row['f1']:.6f}", flush=True)

    per_seed = pd.DataFrame(rows).sort_values(["alpha", "training_seed"])
    per_seed["TaskScore"] = 1.0 / (1.0 + per_seed.mse)
    per_seed["TopologyScore"] = per_seed.f1
    per_seed["JointScore_0.5"] = 0.5 * (per_seed.TaskScore + per_seed.TopologyScore)
    metrics = ["mse", "mae", "precision", "recall", "f1", "jaccard",
               "edge_count", "TaskScore", "TopologyScore", "JointScore_0.5"]
    grouped = per_seed.groupby("alpha", sort=True)
    pieces = []
    for metric in metrics:
        pieces.append(grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    aggregate = pd.concat(pieces, axis=1).reset_index()
    aggregate["n_seeds"] = grouped.size().values

    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "dense_alpha_per_seed.csv", index=False)
    aggregate.to_csv(OUT / "dense_alpha_aggregated.csv", index=False)
    print("\nAGGREGATED")
    print(aggregate[["alpha", "mse_mean", "mae_mean", "precision_mean",
                     "recall_mean", "f1_mean", "jaccard_mean",
                     "edge_count_mean", "TaskScore_mean",
                     "JointScore_0.5_mean"]].to_string(index=False))
    print(f"\n[DONE] {OUT}")


if __name__ == "__main__":
    main()
