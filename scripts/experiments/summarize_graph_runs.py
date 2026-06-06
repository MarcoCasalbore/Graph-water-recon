"""Summarize graph comparison outputs for experiment folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_summary(run_dir: Path) -> dict | None:
    path = run_dir / "graph_comparison" / "comparison_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize graph reconstruction runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    print("run,f1,precision,recall,tp,fp,fn,learned_edges,learned_len_mean,components")
    for run_dir in args.runs:
        summary = read_summary(run_dir)
        if summary is None:
            continue

        if "undirected" in summary:
            undirected = summary["undirected"]
            recovery = undirected["recovery"]
            learned = undirected["learned_stats"]
        else:
            recovery = summary["edge_recovery_undirected"]
            learned = summary["learned_undirected"]
        print(
            f"{run_dir.name},"
            f"{recovery['f1']:.4f},"
            f"{recovery['precision']:.4f},"
            f"{recovery['recall']:.4f},"
            f"{recovery['tp']},"
            f"{recovery['fp']},"
            f"{recovery['fn']},"
            f"{learned['num_edges']},"
            f"{learned['edge_length_mean']:.4f},"
            f"{learned['num_connected_components_undirected']}"
        )


if __name__ == "__main__":
    main()
