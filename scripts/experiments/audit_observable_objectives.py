"""Audit observable selection objectives against final topology metrics.

Ground-truth F1 is used only after runs are complete, to evaluate how well
observable objectives select good topology. Do not use this script to choose
hyperparameters for a reported result; use it to design the next GT-free
objective before running a fresh non-oracle grid.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["source"] = str(path.parent)
                rows.append(row)
    return rows


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman_for_min_objective(objective: list[float], f1: list[float]) -> float:
    # Negate objective because lower observable objective should mean higher F1.
    return pearson(rank_values([-v for v in objective]), rank_values(f1))


def normalized_metrics(rows: list[dict], keys: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key in keys:
        vals = [as_float(row, key) for row in rows]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 0.0
        scale = hi - lo
        out[key] = [(v - lo) / scale if scale > 1e-12 else 0.0 for v in vals]
    return out


def objective_values(rows: list[dict]) -> dict[str, list[float]]:
    norm = normalized_metrics(rows, ["val_mse", "val_lq", "val_topology", "test_mse"])
    alpha = [as_float(row, "alpha") for row in rows]
    raw_j = [as_float(row, "j_val") for row in rows]
    val_mse = [as_float(row, "val_mse") for row in rows]
    val_lq = [as_float(row, "val_lq") for row in rows]
    val_topology = [as_float(row, "val_topology") for row in rows]

    return {
        "j_val_current": raw_j,
        "val_mse": val_mse,
        "val_lq": val_lq,
        "val_topology": val_topology,
        "balanced_norm_mse_lq_50_50": [
            0.5 * norm["val_mse"][idx] + 0.5 * norm["val_lq"][idx]
            for idx in range(len(rows))
        ],
        "balanced_norm_mse_lq_alpha": [
            alpha[idx] * norm["val_mse"][idx] + (1.0 - alpha[idx]) * norm["val_lq"][idx]
            for idx in range(len(rows))
        ],
        "forecast_guard_then_lq": [
            norm["val_lq"][idx] + 0.25 * max(0.0, norm["val_mse"][idx] - 0.5)
            for idx in range(len(rows))
        ],
        "topology_guard_then_mse": [
            norm["val_mse"][idx] + 0.25 * max(0.0, norm["val_lq"][idx] - 0.5)
            for idx in range(len(rows))
        ],
    }


def audit_group(label: str, rows: list[dict]) -> list[dict]:
    f1 = [as_float(row, "f1") for row in rows]
    best_f1 = max(f1) if f1 else 0.0
    objectives = objective_values(rows)
    report_rows = []
    for name, values in objectives.items():
        best_idx = min(range(len(rows)), key=lambda idx: values[idx])
        selected = rows[best_idx]
        selected_f1 = as_float(selected, "f1")
        report_rows.append(
            {
                "group": label,
                "objective": name,
                "selected_run": selected.get("run", ""),
                "selected_f1": selected_f1,
                "best_f1": best_f1,
                "f1_regret": best_f1 - selected_f1,
                "spearman": spearman_for_min_objective(values, f1),
                "n": len(rows),
            }
        )
    return sorted(report_rows, key=lambda row: (row["f1_regret"], -row["spearman"], row["objective"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Observable Objective Audit",
        "",
        "Ground-truth F1 is used only after completed runs, to audit alignment.",
        "Lower `f1_regret` means the observable objective selected a run closer to",
        "the best evaluated topology in that group.",
        "",
        "| group | objective | selected F1 | best F1 | regret | Spearman | selected run |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | `{row['objective']}` | {row['selected_f1']:.4f} | "
            f"{row['best_f1']:.4f} | {row['f1_regret']:.4f} | {row['spearman']:.3f} | "
            f"`{row['selected_run']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/observable_objective_audit"))
    parser.add_argument("--group_by", choices=["dataset", "source_dataset"], default="source_dataset")
    args = parser.parse_args()

    rows = read_rows(args.grid_csvs)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if args.group_by == "dataset":
            key = row.get("dataset", "")
        else:
            key = f"{row.get('source', '')}:{row.get('dataset', '')}"
        groups[key].append(row)

    report_rows = []
    for label, group_rows in sorted(groups.items()):
        report_rows.extend(audit_group(label, group_rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "objective_audit.csv", report_rows)
    write_markdown(args.out_dir / "OBJECTIVE_AUDIT.md", report_rows)

    print("========== OBSERVABLE OBJECTIVE AUDIT ==========")
    print(f"groups: {len(groups)}")
    print(f"rows  : {len(rows)}")
    print(f"[OK] wrote {args.out_dir / 'objective_audit.csv'}")
    print(f"[OK] wrote {args.out_dir / 'OBJECTIVE_AUDIT.md'}")
    for row in report_rows[:12]:
        print(
            f"{row['group']} | {row['objective']} | "
            f"F1 {row['selected_f1']:.4f}/{row['best_f1']:.4f} | "
            f"regret {row['f1_regret']:.4f} | rho {row['spearman']:.3f}"
        )


if __name__ == "__main__":
    main()
