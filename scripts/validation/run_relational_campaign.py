"""Run campaign v2 unchanged on the selected neighbor-dominant signal variant."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_campaign_v2 as campaign


campaign.ROOT = campaign.RESULTS_ROOT / "campaign_v2/relational"
campaign.DATA_DIR = campaign.RESULTS_ROOT / "campaign_v2/datasets/homogeneous_relational_g211_s1211"


if __name__ == "__main__":
    campaign.main()
