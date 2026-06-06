"""Quick inspection utility for one node's EPANET pressure time series."""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_NPZ = PROJECT_ROOT / "data" / "ctown_ts" / "data.npz"
META_JSON = PROJECT_ROOT / "data" / "ctown_ts" / "meta.json"


def main() -> None:
    npz = np.load(DATA_NPZ)
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))

    pressure = npz["pressure"]
    node_ids = meta["node_ids"]

    node_idx = 0
    print(node_ids[node_idx], pressure[:, node_idx])


if __name__ == "__main__":
    main()
