"""Validate Section V datasets, masks, saved scores, metrics, and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_train_synthetic import _global_dynamic_score, build_q_dynamic_candidate_mask
from src.config import Parameters
from src.dataset.dataset_epanet import EpanetDataModule


DATA_DIR = PROJECT_ROOT / "data/synthetic/water_branch_n50_base"
RUNS = {
    42: PROJECT_ROOT / "outputs/final_observable_hyperparam_grid/water_base_geo_global_dynamic_proxy_a0p70_q1_ck2_stab1p000_e25",
    43: RESULTS_ROOT / "raw/section_v_main_seed43_e25",
    44: RESULTS_ROOT / "raw/section_v_main_seed44_e25",
}


def edges_from_array(value: np.ndarray) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in value.T if int(u) != int(v)}


def load_edges(path: Path) -> set[tuple[int, int]]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True).numpy()
    except TypeError:
        value = torch.load(path, map_location="cpu").numpy()
    return edges_from_array(value)


def metric(reference, learned):
    tp, fp, fn = len(reference & learned), len(learned - reference), len(reference - learned)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def check(report, name, condition, detail):
    report.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def main() -> None:
    report = []
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    data = np.load(DATA_DIR / "data.npz")
    arrays = [data[key] for key in ("pressure", "demand", "node_xy", "edge_index", "initial_edge_index")]
    check(report, "dataset_finite", all(np.isfinite(array).all() for array in arrays), "all saved arrays finite")
    splits = meta["splits"]
    split_ok = (splits["train"] == {"start": 0, "end": 503} and
                splits["val"] == {"start": 503, "end": 612} and
                splits["test"] == {"start": 612, "end": 720})
    check(report, "temporal_splits", split_ok, json.dumps(splits, sort_keys=True))

    params = type("Params", (), {"data_npz": str(DATA_DIR / "data.npz"),
                                  "meta_json": str(DATA_DIR / "meta.json"),
                                  "lags": 24, "prediction_window": 1, "stride": 1,
                                  "batch_size": 32})()
    module = EpanetDataModule(params)
    raw = np.stack([data["pressure"], data["demand"]], axis=-1)
    expected_mean = raw[:503].mean(axis=(0, 1), keepdims=True)
    scaler_ok = np.allclose(module.train_dataset.mean, expected_mean)
    check(report, "train_only_scaler", scaler_ok, f"mean={module.train_dataset.mean.reshape(-1).tolist()}")
    windows_ok = (len(module.train_dataset), len(module.val_dataset), len(module.test_dataset)) == (479, 85, 84)
    check(report, "window_counts", windows_ok, "train/val/test=479/85/84 for L=24,H=1")

    mask = build_q_dynamic_candidate_mask(str(DATA_DIR / "data.npz"), "geo_global_dynamic_proxy", 2, 0.05, 1.0, True).numpy()
    mask_ok = mask.shape == (50, 50) and np.array_equal(mask, mask.T) and not np.diag(mask).any() and mask.sum(axis=1).min() >= 1
    check(report, "candidate_mask", mask_ok, f"undirected_edges={int(np.triu(mask, 1).sum())}, min_degree={int(mask.sum(axis=1).min())}")
    q_score = _global_dynamic_score(str(DATA_DIR / "data.npz"), "geo_global_dynamic_proxy", 0.05, 1.0)
    q_ok = np.isfinite(q_score).all() and q_score.min() >= 0 and q_score.max() <= 1 and np.allclose(np.diag(q_score), 0)
    check(report, "observable_Q_score", q_ok, f"range=[{q_score.min():.6g},{q_score.max():.6g}]")

    reference = edges_from_array(data["edge_index"])
    for seed, directory in RUNS.items():
        score = np.load(directory / "dgm_soft_score.npy")
        score_ok = np.isfinite(score).all() and score.min() >= 0
        check(report, f"P_score_seed_{seed}", score_ok, f"shape={score.shape}, range=[{score.min():.6g},{score.max():.6g}]")
        learned = load_edges(directory / "dgm_learned_edges_softprob.pt")
        graph_ok = all(u != v for u, v in learned)
        check(report, f"final_graph_seed_{seed}", graph_ok, f"symmetric serialization after canonicalization; no self-loops; edges={len(learned)}")
        saved = json.loads((directory / "graph_comparison_softprob/comparison_summary.json").read_text(encoding="utf-8"))["edge_recovery_undirected"]
        recomputed = metric(reference, learned)
        metrics_ok = all(abs(float(saved[key]) - float(recomputed[key])) < 1e-10 for key in recomputed)
        check(report, f"metrics_seed_{seed}", metrics_ok, json.dumps(recomputed, sort_keys=True))

    for seed in (43, 44):
        history = pd.read_csv(RUNS[seed] / "training_history.csv")
        train_columns = [column for column in history if column.startswith("train_")]
        history_ok = len(history) == 25 and np.isfinite(history[train_columns].to_numpy()).all()
        check(report, f"history_seed_{seed}", history_ok, f"epochs={len(history)}, columns={len(history.columns)}")

    figure_stems = {path.stem for path in (RESULTS_ROOT / "figures/png").glob("*.png")}
    formats_ok = all((RESULTS_ROOT / f"figures/{ext}/{stem}.{ext}").stat().st_size > 0
                     for stem in figure_stems for ext in ("png", "pdf", "svg"))
    check(report, "figure_formats", formats_ok, f"{len(figure_stems)} figures in PNG/PDF/SVG")
    table_ok = all((RESULTS_ROOT / f"tables/table_{number}_{slug}.{ext}").stat().st_size > 0
                   for number, slug in [(1,"synthetic_dataset_properties"),(2,"model_modules_and_roles"),
                                        (3,"main_configuration"),(4,"main_quantitative_results"),
                                        (5,"hyperparameter_sensitivity"),(6,"decoder_comparison"),
                                        (7,"consensus_comparison"),(8,"complexity")]
                   for ext in ("csv", "md", "tex"))
    check(report, "table_formats", table_ok, "8 tables in CSV/Markdown/LaTeX")
    config_ok = (RESULTS_ROOT / "configs/main_water_base.json").exists() and len(pd.read_csv(RESULTS_ROOT / "experiment_manifest.csv")) == 4
    check(report, "config_and_seeds_saved", config_ok, "main config plus manifest for three full runs and smoke")

    frame = pd.DataFrame(report)
    frame.to_csv(RESULTS_ROOT / "validation_checks.csv", index=False)
    lines = ["# Validation checks", "", frame.to_markdown(index=False), "",
             f"Overall: {'PASS' if (frame.status == 'PASS').all() else 'FAIL'} ({(frame.status == 'PASS').sum()}/{len(frame)} checks).", ""]
    (RESULTS_ROOT / "validation_checks.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_ROOT / "validation_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    if not (frame.status == "PASS").all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
