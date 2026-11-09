"""Validate temporal explainability runs, formulas, tables, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "results_section_v"


def main() -> None:
    checks = []
    base = pd.read_csv(ROOT / "raw/temporal_explainability/temporal_explainability_results.csv")
    balanced = pd.read_csv(ROOT / "raw/temporal_balanced_joint/balanced_joint_results.csv")
    corners = pd.read_csv(ROOT / "figures/data/temporal_corner_seed_comparison.csv")
    checks.append(("base_grid_3x3", len(base) == 9 and set(zip(base.lags, base.horizon)) == {(l,h) for l in (4,12,24) for h in (1,6,12)}))
    checks.append(("balanced_grid_3x3", len(balanced) == 9 and set(zip(balanced.lags, balanced.horizon)) == {(l,h) for l in (4,12,24) for h in (1,6,12)}))
    checks.append(("corner_three_seeds", len(corners) == 6 and set(corners.seed) == {42,43,44}))
    numeric_base = base.select_dtypes(include=[np.number]).to_numpy()
    numeric_balanced = balanced.select_dtypes(include=[np.number]).to_numpy()
    checks.append(("finite_metrics", np.isfinite(numeric_base).all() and np.isfinite(numeric_balanced).all()))
    formula = 0.5 * balanced.task_score + 0.5 * balanced.f1
    checks.append(("eval_0p5_formula", np.allclose(formula, balanced.eval_0p5, atol=1e-12)))
    hard = corners[corners.regime == "hard"]
    easy = corners[corners.regime == "easy"]
    checks.append(("hard_joint_wins_all_seeds", (hard.joint_gain_vs_node > 0).all()))
    checks.append(("easy_node_wins_all_seeds", (easy.joint_gain_vs_node < 0).all()))
    for number, slug in [(9,"temporal_explainability"),(10,"balanced_temporal_evaluation"),(11,"temporal_corner_multiseed")]:
        checks.append((f"table_{number}_formats", all((ROOT / f"tables/table_{number}_{slug}.{ext}").stat().st_size > 0 for ext in ("csv","md","tex"))))
    stems = ["temporal_local_history_sufficiency", "temporal_topology_selectivity",
             "temporal_neighbor_dependence", "temporal_model_comparison",
             "temporal_balanced_evaluation", "temporal_corner_seed_comparison"]
    checks.append(("figure_formats", all((ROOT / f"figures/{ext}/{stem}.{ext}").stat().st_size > 0 for stem in stems for ext in ("png","pdf","svg"))))
    image_ok = True
    for stem in stems:
        array = np.asarray(Image.open(ROOT / f"figures/png/{stem}.png").convert("RGB"))
        image_ok &= array.shape[0] >= 500 and array.shape[1] >= 800 and np.mean(np.any(array < 250, axis=2)) > 0.01
    checks.append(("png_content", bool(image_ok)))
    report = pd.DataFrame([{"check": name, "status": "PASS" if ok else "FAIL"} for name, ok in checks])
    report.to_csv(ROOT / "temporal_validation_checks.csv", index=False)
    lines = ["# Temporal explainability validation", "", "| check | status |", "| --- | --- |"]
    lines += [f"| {row.check} | {row.status} |" for row in report.itertuples()]
    (ROOT / "temporal_validation_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))
    if not (report.status == "PASS").all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
