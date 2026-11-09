"""Integrity and scope audit for the final Section V package."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

P = Path(__file__).resolve().parents[2]
R = P / "results_section_v_final"
checks = {}

sweeps = pd.read_csv(R / "metric_corrected_runs/corrected_joint_sweeps.csv")
checks["84_corrected_runs"] = len(sweeps) == 84 and sweeps.groupby("group").size().to_dict() == {"alpha": 21, "k_dgm": 15, "lh": 48}
checks["corrected_metrics_finite"] = bool(np.isfinite(sweeps[["mse", "mae", "rmse", "f1"]]).all().all())
checks["corrected_prediction_archives_84"] = len(list((R / "metric_corrected_runs").rglob("test_predictions.npz"))) == 84

multi = pd.read_csv(R / "multiple_graph_realizations/multiple_graph_runs.csv")
checks["multi_graph_3x3x6"] = len(multi) == 54 and multi.graph_seed.nunique() == 3 and multi.training_seed.nunique() == 3 and multi.method.nunique() == 6
checks["graph_signal_seed_distinct"] = multi[["graph_seed", "signal_seed"]].drop_duplicates().shape[0] == 3 and not (multi.graph_seed == multi.signal_seed).any()
stats = pd.read_csv(R / "multiple_graph_realizations/graph_realization_statistics.csv")
checks["graphs_connected"] = bool((stats.components == 1).all())

primary = sorted((R / "tables").glob("table_[0-9]*.csv"))
checks["12_primary_tables"] = len(primary) == 12
checks["tables_three_formats"] = all(p.with_suffix(".md").exists() and p.with_suffix(".tex").exists() for p in primary)
checks["tables_parse_and_nonempty"] = all(len(pd.read_csv(p)) > 0 for p in primary)
manifest = pd.read_csv(R / "new_scenario_manifest.csv", low_memory=False)
included = manifest[manifest.included_in_final_analysis_yes_no == "yes"]
checks["no_legacy_included"] = len(included[included.legacy_yes_no == "yes"]) == 0
checks["readme_24_sections"] = sum(line.startswith("## ") for line in (R / "README_SECTION_V.md").read_text(encoding="utf-8").splitlines()) == 24
checks["hypotheses_A_to_H"] = pd.read_csv(R / "hypothesis_matrix.csv").hypothesis.tolist() == list("ABCDEFGH")

format_ok = True
for p in (R / "figures").rglob("*.png"):
    try:
        with Image.open(p) as im: im.verify()
    except Exception: format_ok = False
for p in (R / "figures").rglob("*.pdf"):
    if not p.read_bytes().startswith(b"%PDF-"): format_ok = False
for p in (R / "figures").rglob("*.svg"):
    try: ET.parse(p)
    except Exception: format_ok = False
checks["figure_files_parse"] = format_ok
checks["main_figure_triplets"] = all(len(list((R/"figures/main"/e).glob(f"*.{e}"))) >= 15 for e in ["pdf", "svg", "png"])
checks["supplementary_figure_triplets"] = all(len(list((R/"figures/supplementary"/e).glob(f"*.{e}"))) >= 8 for e in ["pdf", "svg", "png"])
checks["no_zero_size_files"] = not any(p.stat().st_size == 0 for p in R.rglob("*") if p.is_file())
checks["no_patch_backups"] = not any(P.rglob("*.orig")) and not any(P.rglob("*.rej")) and not any(P.rglob("*.patch"))

pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()]).to_csv(R / "final_audit.csv", index=False)
(R / "final_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()]).to_string(index=False))
if not all(checks.values()): raise SystemExit(1)
print("[DONE] final package audit passed")
