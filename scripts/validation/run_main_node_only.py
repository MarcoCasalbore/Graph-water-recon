"""Run the five-seed 300-epoch node-only baseline for the relational main setting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign_v2 as campaign

campaign.ROOT = campaign.RESULTS_ROOT / "campaign_v2/relational"
campaign.DATA_DIR = campaign.RESULTS_ROOT / "campaign_v2/datasets/homogeneous_relational_g211_s1211"

if __name__ == "__main__":
    rows = []
    for seed in campaign.MAIN_SEEDS:
        print(f"[MAIN NODE] seed={seed}", flush=True)
        rows.append(campaign.run_node(6, 12, seed, 300, False))
    campaign.write_rows(campaign.ROOT / "raw_metrics/main_node_runs.csv", rows)
