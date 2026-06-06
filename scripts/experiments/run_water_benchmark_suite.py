"""Run the final water-like benchmark suite.

The suite trains DGM variants, exports soft scores, runs the diagnostic static
graph decoder sweep, and trains non-graph deep forecasting baselines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "simple": ("data/synthetic/simple_water_n50", 61),
    "water_base": ("data/synthetic/water_branch_n50_base", 63),
    "water_bridges": ("data/synthetic/water_branch_n50_bridges", 62),
    "water_loopy": ("data/synthetic/water_branch_n50_loopy", 73),
}


VARIANTS = [
    {"name": "a005_q2_mst", "alpha": 0.05, "q": 2, "candidate": 2, "mst": True},
    {"name": "a002_q2_mst", "alpha": 0.02, "q": 2, "candidate": 2, "mst": True},
    {"name": "a010_q2_mst", "alpha": 0.10, "q": 2, "candidate": 2, "mst": True},
    {"name": "a005_q3_mst", "alpha": 0.05, "q": 3, "candidate": 3, "mst": True},
]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final water benchmark experiments.")
    parser.add_argument("--datasets", default="simple,water_base,water_bridges,water_loopy")
    parser.add_argument("--variants", default="a005_q2_mst,a002_q2_mst,a010_q2_mst,a005_q3_mst")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--out_root", type=Path, default=PROJECT_ROOT / "outputs" / "final_suite")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_dgm", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    selected_variants = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants = [variant for variant in VARIANTS if variant["name"] in selected_variants]

    manifest = {
        "datasets": selected_datasets,
        "variants": variants,
        "epochs": args.epochs,
    }
    (args.out_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for dataset_name in selected_datasets:
        data_dir, budget = DATASETS[dataset_name]
        for variant in variants:
            run_name = f"{dataset_name}_{variant['name']}_e{args.epochs}"
            out_dir = args.out_root / run_name
            if not args.skip_dgm:
                cmd = [
                    sys.executable,
                    "scripts/experiments/run_train_synthetic.py",
                    "--data_dir",
                    data_dir,
                    "--out_dir",
                    str(out_dir),
                    "--run_name",
                    run_name,
                    "--max_epochs",
                    str(args.epochs),
                    "--check_val_every_n_epoch",
                    "5",
                    "--batch_size",
                    "32",
                    "--k",
                    "2",
                    "--static_top_edges",
                    str(budget),
                    "--use_complementary_loss",
                    "--alpha_forecast",
                    str(variant["alpha"]),
                    "--lambda_graph",
                    "0.0",
                    "--lambda_soft_geo",
                    "0.0",
                    "--lambda_soft_pseudo_edge",
                    "1.0",
                    "--pseudo_score_mode",
                    "geo_global_dynamic_proxy",
                    "--pseudo_use_static_target",
                    "--pseudo_candidate_k",
                    "0",
                    "--pseudo_topk",
                    str(variant["q"]),
                    "--assign_candidate_mask",
                    "--candidate_mode",
                    "q_dynamic",
                    "--candidate_k",
                    str(variant["candidate"]),
                    "--consensus_threshold",
                    "0.5",
                    "--ensure_connected_consensus",
                    "--export_soft_graph",
                    "--save_soft_score",
                ]
                cmd.append("--pseudo_include_mst" if variant["mst"] else "--no-pseudo_include_mst")
                run(cmd, args.dry_run)

            score_path = out_dir / "dgm_soft_score.npy"
            if score_path.exists() or args.dry_run:
                sweep_dir = args.out_root / f"{run_name}_soft_sweep"
                run(
                    [
                        sys.executable,
                        "scripts/diagnostics/sweep_soft_postprocess.py",
                        "--score",
                        str(score_path),
                        "--npz",
                        str(PROJECT_ROOT / data_dir / "data.npz"),
                        "--meta",
                        str(PROJECT_ROOT / data_dir / "meta.json"),
                        "--out_dir",
                        str(sweep_dir),
                        "--budget",
                        str(budget),
                        "--gammas",
                        "0,0.05,0.1,0.25,0.5,0.75,1,1.5,2",
                        "--taus",
                        "0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3",
                        "--max_degrees",
                        "0,4,5",
                        "--include_mst_options",
                        "0,1",
                        "--connect_options",
                        "0,1",
                    ],
                    args.dry_run,
                )

        if not args.skip_baselines:
            for baseline in ["global_gru", "node_mlp"]:
                run(
                    [
                        sys.executable,
                        "scripts/experiments/train_forecast_baseline.py",
                        "--data_dir",
                        data_dir,
                        "--out_dir",
                        str(args.out_root / f"{dataset_name}_forecast_{baseline}"),
                        "--model",
                        baseline,
                        "--epochs",
                        "100",
                        "--hidden",
                        "128",
                    ],
                    args.dry_run,
                )


if __name__ == "__main__":
    main()
