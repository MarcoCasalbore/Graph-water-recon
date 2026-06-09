"""Summarize which hyperparameters most influence observable loss and F1.

Ground-truth F1 is used only for post-hoc evaluation after the grid is complete.
The selected runs remain those chosen by observable validation quantities.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUMERIC_KEYS = [
    "alpha",
    "topk",
    "candidate_k",
    "lambda_stability",
    "val_mse",
    "val_lq",
    "val_topology",
    "j_val",
    "test_mse",
    "f1",
    "precision",
    "recall",
]


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for key in NUMERIC_KEYS:
            row[key] = as_float(row, key)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def group_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(sorted(out.items(), key=lambda item: item[0]))


def summarize_group(label: str, rows: list[dict]) -> dict:
    return {
        "group": label,
        "n": len(rows),
        "mean_f1": mean([row["f1"] for row in rows]),
        "best_f1": max(row["f1"] for row in rows),
        "mean_j_val": mean([row["j_val"] for row in rows]),
        "best_j_val": min(row["j_val"] for row in rows),
        "mean_val_mse": mean([row["val_mse"] for row in rows]),
        "mean_val_lq": mean([row["val_lq"] for row in rows]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_table(rows: list[dict], key: str, reverse: bool) -> list[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)


def write_markdown(path: Path, rows: list[dict], influence_rows: list[dict]) -> None:
    by_dataset = group_rows(rows, "dataset")
    lines = [
        "# Final Hyperparameter Influence Summary",
        "",
        "Ground-truth F1 is used only after completed runs, for evaluation.",
        "",
        "## Loss Used In Every Run",
        "",
        "```text",
        "L = alpha * L_forecast + (1-alpha) * (lambda_Q * L_Q + lambda_stability * L_stability)",
        "L_Q = mean_i -sum_j Q_ij log(P_ij + eps)",
        "lambda_Q = 1.0",
        "Q = observable pseudo-target from coordinates + node time series only",
        "selection = observable_val",
        "J_obs = alpha*MSE_val + (1-alpha)*L_topology_val",
        "```",
        "",
        "## Best Per Dataset By Observable Selection",
        "",
        "| dataset | selected run | J_obs | F1 | val_mse | val_LQ | test_mse |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, dataset_rows in by_dataset.items():
        row = min(dataset_rows, key=lambda item: item["j_val"])
        lines.append(
            f"| {dataset} | `{row['run']}` | {row['j_val']:.4f} | {row['f1']:.4f} | "
            f"{row['val_mse']:.4f} | {row['val_lq']:.4f} | {row['test_mse']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Best Per Dataset By F1 Evaluation Only",
            "",
            "| dataset | best evaluated run | F1 | J_obs | val_mse | val_LQ |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset, dataset_rows in by_dataset.items():
        row = max(dataset_rows, key=lambda item: item["f1"])
        lines.append(
            f"| {dataset} | `{row['run']}` | {row['f1']:.4f} | {row['j_val']:.4f} | "
            f"{row['val_mse']:.4f} | {row['val_lq']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Hyperparameter Influence",
            "",
            "| factor | value | n | mean F1 | best F1 | mean J_obs | best J_obs | mean val_mse | mean val_LQ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in influence_rows:
        lines.append(
            f"| {row['factor']} | `{row['group']}` | {row['n']} | {row['mean_f1']:.4f} | "
            f"{row['best_f1']:.4f} | {row['mean_j_val']:.4f} | {row['best_j_val']:.4f} | "
            f"{row['mean_val_mse']:.4f} | {row['mean_val_lq']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top Runs By Observable Objective",
            "",
            "| rank | run | dataset | J_obs | F1 | alpha | k | candidate_k | stability |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(best_table(rows, "j_val", reverse=False)[:20], start=1):
        lines.append(
            f"| {idx} | `{row['run']}` | {row['dataset']} | {row['j_val']:.4f} | {row['f1']:.4f} | "
            f"{row['alpha']:.2f} | {row['topk']:.0f} | {row['candidate_k']:.0f} | {row['lambda_stability']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Top Runs By F1 Evaluation Only",
            "",
            "| rank | run | dataset | F1 | J_obs | alpha | k | candidate_k | stability |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(best_table(rows, "f1", reverse=True)[:20], start=1):
        lines.append(
            f"| {idx} | `{row['run']}` | {row['dataset']} | {row['f1']:.4f} | {row['j_val']:.4f} | "
            f"{row['alpha']:.2f} | {row['topk']:.0f} | {row['candidate_k']:.0f} | {row['lambda_stability']:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.grid_csv)
    influence_rows: list[dict] = []
    for factor in ["dataset", "mode", "alpha", "topk", "candidate_k", "lambda_stability"]:
        for label, group in group_rows(rows, factor).items():
            row = summarize_group(label, group)
            row["factor"] = factor
            influence_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "hyperparameter_influence.csv", influence_rows)
    write_markdown(args.out_dir / "HYPERPARAMETER_INFLUENCE.md", rows, influence_rows)

    print("========== GRID INFLUENCE SUMMARY ==========")
    print(f"rows: {len(rows)}")
    print(f"[OK] wrote {args.out_dir / 'hyperparameter_influence.csv'}")
    print(f"[OK] wrote {args.out_dir / 'HYPERPARAMETER_INFLUENCE.md'}")


if __name__ == "__main__":
    main()
