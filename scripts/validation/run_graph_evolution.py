"""Reproduce deterministic training prefixes for graph-evolution diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_campaign_v2 as campaign

campaign.ROOT = campaign.RESULTS_ROOT / "campaign_v2/relational"
campaign.DATA_DIR = campaign.RESULTS_ROOT / "campaign_v2/datasets/homogeneous_relational_g211_s1211"


def main():
    rows = []
    for epochs in (1, 25, 150):
        print(f"[EVOLUTION] prefix={epochs}", flush=True)
        rows.append(campaign.run_joint(
            "graph_evolution", 6, 12, 0.8, 2, 3, 42, epochs, False
        ))
    main_rows = campaign.ROOT / "raw_metrics/main_runs.csv"
    print(f"[OK] prefixes={len(rows)}; final source={main_rows}")


if __name__ == "__main__":
    main()
