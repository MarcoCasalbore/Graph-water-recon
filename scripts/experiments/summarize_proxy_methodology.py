"""Summarize observable-to-topology alignment in the proxy methodology study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DATASETS = ["simple", "water_base", "water_bridges", "water_loopy"]
NUMERIC = [
    "observable_score", "score_consistency", "edge_consistency", "forecast_bic",
    "f1", "average_precision", "num_edges",
]


def rank_correlation(x: list[float], y: list[float]) -> float:
    rank_x = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    rank_y = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    for dataset in DATASETS:
        path = args.study_dir / f"{dataset}_posthoc_audit.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for row in rows:
            for name in NUMERIC:
                row[name] = float(row[name])
        print(f"\n{dataset}")
        for name in [
            "observable_score", "score_consistency", "edge_consistency",
            "forecast_bic", "average_precision", "num_edges",
        ]:
            value = rank_correlation([row[name] for row in rows], [row["f1"] for row in rows])
            print(f"  corr_rank({name}, F1) = {value:+.3f}")
        for label, key, reverse in [
            ("observable", "observable_score", True),
            ("posthoc_oracle", "f1", True),
            ("forecast", "forecast_bic", False),
        ]:
            row = sorted(rows, key=lambda item: item[key], reverse=reverse)[0]
            config = ", ".join(
                f"{name}={row[name]}" for name in [
                    "dynamic", "geometry", "fusion", "row_k", "symmetry", "include_mst",
                    "num_edges", "observable_score", "f1", "average_precision",
                ]
            )
            print(f"  {label}: {config}")


if __name__ == "__main__":
    main()
