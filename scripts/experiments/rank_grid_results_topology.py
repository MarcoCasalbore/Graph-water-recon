#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-rank water DGM grid-search results with topology-first logic.

Main idea:
- prioritize overlap with the real topology
- reward high precision (few extra edges)
- only mildly reward good forecasting as a secondary criterion

Inputs:
  - grid_results_water_repaired.csv (or any compatible CSV)
Outputs:
  - topology_ranked CSV
  - compact top-k CSV for inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def minmax_inverse(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return 1.0 - (s - mn) / (mx - mn)


def build_topology_ranking(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Robust numeric conversion
    numeric_cols = [
        "tp_undirected", "fp_undirected", "fn_undirected",
        "precision_undirected", "recall_undirected",
        "f1_undirected", "jaccard_undirected", "test_mse",
        "gt_num_edges_undirected",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # If GT edge count is missing, reconstruct it from TP+FN
    if "gt_num_edges_undirected" not in out.columns or out["gt_num_edges_undirected"].isna().all():
        if {"tp_undirected", "fn_undirected"}.issubset(out.columns):
            out["gt_num_edges_undirected"] = out["tp_undirected"] + out["fn_undirected"]

    gt_ref = out["gt_num_edges_undirected"].replace(0, np.nan)

    # Interpretable topology rates
    out["correct_edge_rate"] = out["tp_undirected"] / gt_ref
    out["extra_edge_rate"] = out["fp_undirected"] / gt_ref
    out["missing_edge_rate"] = out["fn_undirected"] / gt_ref

    # Secondary forecasting score, normalized so larger is better
    out["forecast_score"] = minmax_inverse(out["test_mse"])

    # Topology-first selection score
    #  - jaccard: strict overlap metric, penalizes both FP and FN
    #  - precision: explicit reward for fewer extra edges
    #  - forecast_score: mild tie-breaker, not dominant
    out["selection_score"] = (
        0.80 * out["jaccard_undirected"].fillna(0.0)
        + 0.15 * out["precision_undirected"].fillna(0.0)
        + 0.05 * out["forecast_score"].fillna(0.0)
    )

    # Final sort: topology first, prediction last
    sort_cols = [
        "selection_score",
        "jaccard_undirected",
        "precision_undirected",
        "recall_undirected",
        "test_mse",
    ]
    ascending = [False, False, False, False, True]
    out = out.sort_values(by=sort_cols, ascending=ascending, na_position="last")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input grid CSV")
    ap.add_argument("--out", default=None, help="Output ranked CSV")
    ap.add_argument("--topk", type=int, default=15, help="Top-k rows to export separately")
    args = ap.parse_args()

    in_csv = Path(args.csv).resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    out_csv = Path(args.out).resolve() if args.out else in_csv.with_name(in_csv.stem + "_topology_ranked.csv")
    topk_csv = out_csv.with_name(out_csv.stem + f"_top{args.topk}.csv")

    df = pd.read_csv(in_csv)
    ranked = build_topology_ranking(df)

    ranked.to_csv(out_csv, index=False)
    ranked.head(args.topk).to_csv(topk_csv, index=False)

    show_cols = [
        "run_name",
        "selection_score",
        "jaccard_undirected",
        "precision_undirected",
        "recall_undirected",
        "f1_undirected",
        "correct_edge_rate",
        "extra_edge_rate",
        "missing_edge_rate",
        "test_mse",
        "test_p_mse",
        "test_d_mse",
    ]
    show_cols = [c for c in show_cols if c in ranked.columns]

    print("\nTop runs with topology-first ranking:\n")
    print(ranked[show_cols].head(args.topk).to_string(index=False))
    print(f"\nSaved ranked CSV to: {out_csv}")
    print(f"Saved top-{args.topk} CSV to: {topk_csv}")


if __name__ == "__main__":
    main()
