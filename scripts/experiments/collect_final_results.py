"""Collect final benchmark outputs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def topology_rows(out_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sweep_json in sorted(out_root.glob("*_soft_sweep/best_soft_postprocess.json")):
        run = sweep_json.parent.name.removesuffix("_soft_sweep")
        data = read_json(sweep_json)
        if not data:
            continue
        rows.append(
            {
                "run": run,
                "decoder": "soft_postprocess_oracle_diagnostic",
                "f1": data["f1"],
                "precision": data["precision"],
                "recall": data["recall"],
                "tp": data["tp"],
                "fp": data["fp"],
                "fn": data["fn"],
                "edges": data["edges"],
                "components": data["components"],
                "length_mean": data["length_mean"],
                "length_max": data["length_max"],
                "gamma": data["gamma"],
                "tau": data["tau"],
                "include_mst": data["include_mst"],
                "ensure_connected": data["ensure_connected"],
                "image": str(sweep_json.parent / "best_graph_comparison" / "graph_compare_undirected.png"),
            }
        )

    for summary_json in sorted(out_root.glob("*/graph_comparison_softprob/comparison_summary.json")):
        run = summary_json.parents[1].name
        data = read_json(summary_json)
        if not data:
            continue
        undirected = data.get("undirected", data)
        recovery = undirected.get("recovery", undirected.get("edge_recovery_undirected", {}))
        learned = undirected.get("learned_stats", undirected.get("learned_undirected", {}))
        rows.append(
            {
                "run": run,
                "decoder": "soft_top_edges",
                "f1": recovery.get("f1"),
                "precision": recovery.get("precision"),
                "recall": recovery.get("recall"),
                "tp": recovery.get("tp"),
                "fp": recovery.get("fp"),
                "fn": recovery.get("fn"),
                "edges": learned.get("num_edges"),
                "components": learned.get("num_connected_components_undirected"),
                "length_mean": learned.get("edge_length_mean"),
                "length_max": learned.get("edge_length_max"),
                "gamma": "",
                "tau": "",
                "include_mst": "",
                "ensure_connected": "",
                "image": str(summary_json.parent / "graph_compare_undirected.png"),
            }
        )
    return rows


def forecast_rows(out_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_json in sorted(out_root.glob("*/forecast_metrics.json")):
        run = metrics_json.parent.name
        data = read_json(metrics_json)
        if not data or not data.get("test"):
            continue
        test = data["test"]
        rows.append(
            {
                "run": run,
                "model": "dgm",
                "mse": test["mse"],
                "rmse": test["rmse"],
                "mae": test["mae"],
                "mape": test["mape"],
                "pressure_mse": test["pressure_mse"],
                "demand_mse": test["demand_mse"],
            }
        )

    for metrics_json in sorted(out_root.glob("*_forecast_*/forecast_baseline_metrics.json")):
        run = metrics_json.parent.name
        data = read_json(metrics_json)
        if not data:
            continue
        test = data["test"].get("normalized", data["test"])
        rows.append(
            {
                "run": run,
                "model": data["model"],
                "mse": test["mse"],
                "rmse": test["rmse"],
                "mae": test["mae"],
                "mape": test["mape"],
                "pressure_mse": test["pressure_mse"],
                "demand_mse": test["demand_mse"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], cols: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final benchmark result tables.")
    parser.add_argument("--out_root", type=Path, default=Path("outputs/final_suite"))
    args = parser.parse_args()

    topo = sorted(topology_rows(args.out_root), key=lambda r: (str(r["run"]), str(r["decoder"])))
    forecast = sorted(forecast_rows(args.out_root), key=lambda r: (str(r["run"]), str(r["model"])))

    write_csv(args.out_root / "topology_results.csv", topo)
    write_csv(args.out_root / "forecast_results.csv", forecast)

    best_topo = sorted(
        [r for r in topo if r["decoder"] == "soft_postprocess_oracle_diagnostic"],
        key=lambda r: float(r["f1"]),
        reverse=True,
    )
    best_forecast = sorted(forecast, key=lambda r: float(r["mse"]))
    report = [
        "# Final Benchmark Results",
        "",
        "Ground-truth edges are used only for evaluation and for the diagnostic postprocess sweep.",
        "",
        "## Best Topology Results",
        markdown_table(best_topo, ["run", "f1", "precision", "recall", "components", "gamma", "tau", "include_mst"], limit=20),
        "",
        "## Forecast Results",
        markdown_table(best_forecast, ["run", "model", "mse", "rmse", "mae", "pressure_mse", "demand_mse"], limit=40),
        "",
    ]
    (args.out_root / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] Wrote result tables to: {args.out_root}")


if __name__ == "__main__":
    main()
