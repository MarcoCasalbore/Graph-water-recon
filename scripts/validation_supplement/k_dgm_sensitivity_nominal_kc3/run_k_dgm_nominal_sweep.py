"""Run only missing nominal candidate-set-3 k_DGM experiments."""
from pathlib import Path
import json
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT = FINAL / "metric_corrected_runs/k_dgm_nominal_kc3"
sys.path.insert(0, str(PROJECT / "scripts/validation"))
import run_campaign_v2 as campaign  # noqa: E402


def main():
    for k_dgm in [1, 3]:
        for seed in [42, 43, 44]:
            name = f"k_dgm_nominal_g211_s1211_t{seed}_l6_h12_a0p80_kd{k_dgm}_kc3_e120"
            out = ROOT / name
            out.mkdir(parents=True, exist_ok=True)
            prediction = out / "test_predictions.npz"
            topology = out / "graph_comparison_softprob/comparison_summary.json"
            command = campaign.joint_command(out, name, 6, 12, 0.8,
                                             k_dgm, 3, seed, 120)
            command[command.index("--data_dir") + 1] = str(DATA)
            config = {
                "run_id": name, "graph_seed": 211, "signal_seed": 1211,
                "training_seed": seed, "L": 6, "F": 12, "alpha": 0.8,
                "k_dgm": k_dgm, "candidate_set_size": 3, "k_q": 2,
                "decoder": "row_k3_mutual_mst",
                "consensus": "mean_probability_soft", "epochs": 120,
                "checkpoint_selection": "observable_val",
                "dataset": str(DATA), "command": command,
            }
            (out / "config.json").write_text(json.dumps(config, indent=2),
                                              encoding="utf-8")
            if not (prediction.exists() and topology.exists()):
                started = time.perf_counter()
                with (out / "training.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=PROJECT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                elapsed = time.perf_counter() - started
                (out / "elapsed_seconds.txt").write_text(
                    f"{elapsed:.9f}\n", encoding="utf-8")
            campaign.parse_history(out / "training.log",
                                   out / "training_history.csv")
            print(f"[OK] k_dgm={k_dgm} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
