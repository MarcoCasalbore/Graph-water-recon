"""Run a non-oracle synthetic grid for the minimal DGM topology loss.

Selection objective uses only observable validation quantities:

    J_val = alpha * MSE_val + (1 - alpha) * L_topology_val

Ground-truth edges are used only by the comparison script after each run to
report reconstruction metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("\n[CMD]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_one(
    dataset_name: str,
    data_dir: Path,
    budget: int,
    mode: str,
    alpha: float,
    topk: int,
    candidate_k: int,
    lambda_stability: float,
    epochs: int,
    batch_size: int,
    out_root: Path,
    dry_run: bool,
) -> dict:
    run_name = (
        f"{dataset_name}_{mode}_a{alpha:.2f}_q{topk}_ck{candidate_k}"
        f"_stab{lambda_stability:.3f}_e{epochs}"
    ).replace(".", "p")
    out_dir = out_root / run_name
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--run_name",
        run_name,
        "--num_nodes",
        "50",
        "--max_epochs",
        str(epochs),
        "--check_val_every_n_epoch",
        "5",
        "--batch_size",
        str(batch_size),
        "--k",
        str(topk),
        "--static_top_edges",
        str(budget),
        "--use_complementary_loss",
        "--alpha_forecast",
        str(alpha),
        "--lambda_graph",
        "0.0",
        "--lambda_soft_pseudo_edge",
        "1.0",
        "--lambda_stability",
        str(lambda_stability),
        "--pseudo_score_mode",
        mode,
        "--pseudo_use_static_target",
        "--pseudo_topk",
        str(topk),
        "--pseudo_include_mst",
        "--select_best_by",
        "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode",
        "q_dynamic",
        "--candidate_k",
        str(candidate_k),
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
    ]
    run_cmd(cmd, dry_run=dry_run)

    forecast = read_json(out_dir / "forecast_metrics.json")
    comparison = read_json(out_dir / "graph_comparison_softprob" / "comparison_summary.json")
    val = forecast.get("best_observable_validation") or forecast.get("best_validation") or forecast.get("last_validation") or {}
    test = forecast.get("test") or {}
    undir = comparison.get("edge_recovery_undirected", {})
    learned_stats = comparison.get("learned_undirected", {})

    val_mse = float(val.get("mse", 0.0))
    val_topology = float(val.get("topology_loss", val.get("soft_pseudo_edge_loss", 0.0)))
    j_val = float(alpha) * val_mse + (1.0 - float(alpha)) * val_topology

    return {
        "run": run_name,
        "dataset": dataset_name,
        "mode": mode,
        "loss_formula": "alpha*L_forecast+(1-alpha)*(lambda_Q*L_Q+lambda_stability*L_stability)",
        "selection": "observable_val",
        "lambda_Q": 1.0,
        "alpha": alpha,
        "topk": topk,
        "candidate_k": candidate_k,
        "lambda_stability": lambda_stability,
        "epochs": epochs,
        "val_mse": val_mse,
        "val_lq": float(val.get("soft_pseudo_edge_loss", 0.0)),
        "val_topology": val_topology,
        "j_val": j_val,
        "test_mse": float(test.get("mse", 0.0)),
        "f1": float(undir.get("f1", 0.0)),
        "precision": float(undir.get("precision", 0.0)),
        "recall": float(undir.get("recall", 0.0)),
        "tp": int(undir.get("tp", 0)),
        "fp": int(undir.get("fp", 0)),
        "fn": int(undir.get("fn", 0)),
        "components": int(learned_stats.get("num_connected_components_undirected", 0)),
        "out_dir": str(out_dir),
    }


def collect_existing_run(run_dir: Path) -> dict:
    run_name = run_dir.name
    dataset = next(
        (name for name in sorted(DATASETS, key=len, reverse=True) if run_name.startswith(f"{name}_")),
        run_name.split("_")[0] if run_name else run_name,
    )

    forecast = read_json(run_dir / "forecast_metrics.json")
    comparison = read_json(run_dir / "graph_comparison_softprob" / "comparison_summary.json")
    val = forecast.get("best_observable_validation") or forecast.get("best_validation") or forecast.get("last_validation") or {}
    test = forecast.get("test") or {}
    undir = comparison.get("edge_recovery_undirected", {})
    learned_stats = comparison.get("learned_undirected", {})

    alpha = 0.0
    alpha_match = re.search(r"_a([0-9]+p[0-9]+)_", run_name)
    if alpha_match:
        alpha = float(alpha_match.group(1).replace("p", "."))

    topk = 2
    topk_match = re.search(r"_q([0-9]+)_", run_name)
    if topk_match:
        topk = int(topk_match.group(1))

    candidate_k = 2
    candidate_match = re.search(r"_ck([0-9]+)_", run_name)
    if candidate_match:
        candidate_k = int(candidate_match.group(1))

    lambda_stability = 0.0
    stability_match = re.search(r"_stab([0-9]+p[0-9]+)_", run_name)
    if stability_match:
        lambda_stability = float(stability_match.group(1).replace("p", "."))

    epochs = 25
    epoch_match = re.search(r"_e([0-9]+)$", run_name)
    if epoch_match:
        epochs = int(epoch_match.group(1))

    mode = ""
    mode_match = re.match(rf"^{re.escape(dataset)}_(?P<mode>.+?)_a[0-9]+p[0-9]+_", run_name)
    if mode_match:
        mode = mode_match.group("mode")

    val_mse = float(val.get("mse", 0.0))
    val_topology = float(val.get("topology_loss", val.get("soft_pseudo_edge_loss", 0.0)))
    j_val = float(alpha) * val_mse + (1.0 - float(alpha)) * val_topology

    return {
        "run": run_dir.name,
        "dataset": dataset,
        "mode": mode,
        "loss_formula": "alpha*L_forecast+(1-alpha)*(lambda_Q*L_Q+lambda_stability*L_stability)",
        "selection": "observable_val",
        "lambda_Q": 1.0,
        "alpha": alpha,
        "topk": topk,
        "candidate_k": candidate_k,
        "lambda_stability": lambda_stability,
        "epochs": epochs,
        "val_mse": val_mse,
        "val_lq": float(val.get("soft_pseudo_edge_loss", 0.0)),
        "val_topology": val_topology,
        "j_val": j_val,
        "test_mse": float(test.get("mse", 0.0)),
        "f1": float(undir.get("f1", 0.0)),
        "precision": float(undir.get("precision", 0.0)),
        "recall": float(undir.get("recall", 0.0)),
        "tp": int(undir.get("tp", 0)),
        "fp": int(undir.get("fp", 0)),
        "fn": int(undir.get("fn", 0)),
        "components": int(learned_stats.get("num_connected_components_undirected", 0)),
        "out_dir": str(run_dir),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    by_j = sorted(rows, key=lambda row: row["j_val"])
    by_f1 = sorted(rows, key=lambda row: row["f1"], reverse=True)
    lines = [
        "# Minimal Loss Synthetic Grid",
        "",
        "Selection is non-oracle: rows are ranked by observable validation objective",
        "`J_val = alpha * MSE_val + (1-alpha) * L_topology_val`.",
        "Ground-truth edges are used only for the reported F1/precision/recall metrics.",
        "",
        "## Loss And Selection Settings",
        "",
        "```text",
        "L = alpha * L_forecast + (1-alpha) * (lambda_Q * L_Q + lambda_stability * L_stability)",
        "L_Q = mean_i -sum_j Q_ij log(P_ij + eps)",
        "lambda_Q = 1.0",
        "Q is built only from coordinates and observed time series.",
        "Checkpoint selection: observable_val = alpha*MSE_val + (1-alpha)*L_topology_val",
        "Ground truth is used only after each run is fixed, for reported metrics.",
        "```",
        "",
        "## Best By Observable Validation Objective",
        "",
        "| rank | run | J_val | val_mse | val_LQ | F1 | components |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(by_j[:15], start=1):
        lines.append(
            f"| {idx} | `{row['run']}` | {row['j_val']:.4f} | {row['val_mse']:.4f} | "
            f"{row['val_lq']:.4f} | {row['f1']:.4f} | {row['components']} |"
        )
    lines.extend(
        [
            "",
            "## Best By F1 Evaluation Only",
            "",
            "| rank | run | F1 | precision | recall | J_val | components |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(by_f1[:15], start=1):
        lines.append(
            f"| {idx} | `{row['run']}` | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['j_val']:.4f} | {row['components']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=Path, default=PROJECT_ROOT / "outputs" / "minimal_loss_grid")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", default=["pressure_corr", "pressure_smooth", "geo_global_dynamic_proxy"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.20, 0.40])
    parser.add_argument("--topks", nargs="+", type=int, default=[2])
    parser.add_argument("--candidate_ks", nargs="+", type=int, default=[2])
    parser.add_argument("--candidate_policy", choices=["grid", "same_as_topk"], default="grid")
    parser.add_argument("--candidate_min", type=int, default=2)
    parser.add_argument("--lambda_stabilities", nargs="+", type=float, default=[0.0])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.collect_only:
        for run_dir in sorted(p for p in args.out_root.iterdir() if p.is_dir()):
            if (run_dir / "forecast_metrics.json").exists():
                rows.append(collect_existing_run(run_dir))
        write_csv(args.out_root / "grid_results.csv", rows)
        write_markdown(args.out_root / "RESULTS.md", rows)
        print(f"[OK] collected {len(rows)} existing runs")
        return

    for dataset_name in args.datasets:
        data_dir_str, budget = DATASETS[dataset_name]
        for mode in args.modes:
            for alpha in args.alphas:
                for topk in args.topks:
                    candidate_ks = (
                        [max(int(args.candidate_min), int(topk))]
                        if args.candidate_policy == "same_as_topk"
                        else args.candidate_ks
                    )
                    for candidate_k in candidate_ks:
                        for lambda_stability in args.lambda_stabilities:
                            row = run_one(
                                dataset_name=dataset_name,
                                data_dir=PROJECT_ROOT / data_dir_str,
                                budget=budget,
                                mode=mode,
                                alpha=alpha,
                                topk=topk,
                                candidate_k=candidate_k,
                                lambda_stability=lambda_stability,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                out_root=args.out_root,
                                dry_run=args.dry_run,
                            )
                            rows.append(row)
                            write_csv(args.out_root / "grid_results.csv", rows)
                            write_markdown(args.out_root / "RESULTS.md", rows)

    print(f"[OK] wrote {args.out_root / 'grid_results.csv'}")
    print(f"[OK] wrote {args.out_root / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
