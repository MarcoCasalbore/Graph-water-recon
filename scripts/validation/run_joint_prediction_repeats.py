"""Reproduce only the five main joint runs with element-wise prediction export."""
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign_v2 as campaign  # noqa: E402

DATA = PROJECT / "results_section_v/campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT = PROJECT / "results_section_v_final/error_analysis/joint_retrained"

for seed in [42, 43, 44, 45, 46]:
    out = ROOT / f"joint_g211_s1211_t{seed}_l6_h12_e300"
    prediction = out / "test_predictions.npz"
    if prediction.exists():
        print(f"[SKIP] {out.name}", flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)
    command = campaign.joint_command(out, out.name, 6, 12, .8, 2, 3, seed, 300)
    data_index = command.index("--data_dir") + 1
    command[data_index] = str(DATA)
    with (out / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"[OK] {out.name}", flush=True)
