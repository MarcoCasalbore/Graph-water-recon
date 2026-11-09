#!/usr/bin/env python3
"""Render final graph/error/loss artifacts for a completed selected run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_protocol import load_learned_edges


def segments(edges: set[tuple[int, int]], xy: np.ndarray, color: str, label: str, linewidth: float = 1.2) -> None:
    first = True
    for u, v in sorted(edges):
        plt.plot(xy[[u, v], 0], xy[[u, v], 1], color=color, linewidth=linewidth, alpha=0.85, label=label if first else None)
        first = False


def draw_graph(ax: plt.Axes, xy: np.ndarray, edges: set[tuple[int, int]], title: str, color: str) -> None:
    plt.sca(ax)
    segments(edges, xy, color, "edge")
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c="#18212b", zorder=3)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def render(observation_dir: Path, evaluation_dir: Path, run_dir: Path) -> None:
    output = run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    observation = np.load(observation_dir / "data.npz")
    xy = observation["node_xy"].astype(np.float64)
    reference = np.load(evaluation_dir / "reference_topology.npz")
    gt = {tuple(sorted((int(u), int(v)))) for u, v in reference["edge_index"].T if int(u) != int(v)}
    learned = load_learned_edges(run_dir / "learned_edges.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    draw_graph(axes[0], xy, gt, f"Reference topology (E={len(gt)})", "#4c78a8")
    draw_graph(axes[1], xy, learned, f"Reconstructed topology (E={len(learned)})", "#7b61a8")
    figure.savefig(output / "gt_vs_reconstructed.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plt.sca(ax)
    segments(gt & learned, xy, "#2ca02c", "TP", 1.6)
    segments(learned - gt, xy, "#d62728", "FP", 1.8)
    segments(gt - learned, xy, "#ff9f1c", "FN", 1.8)
    ax.scatter(xy[:, 0], xy[:, 1], s=15, c="#18212b", zorder=3)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Topology reconstruction errors")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    figure.savefig(output / "error_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    epoch = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epoch, [row["train_forecast"] for row in history], label="train forecast")
    axes[0].plot(epoch, [row["val_forecast"] for row in history], label="validation forecast")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="Forecasting loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epoch, [row["train_structural"] for row in history], label="train structural")
    axes[1].plot(epoch, [row["val_structural"] for row in history], label="validation structural")
    axes[1].plot(epoch, [row["val_joint"] for row in history], label="validation joint", linewidth=2)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Structural and joint objectives")
    axes[1].legend(frameon=False)
    figure.savefig(output / "loss_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        run_dir / "learned_graph.npz",
        edge_index=np.asarray(sorted(learned), dtype=np.int64).T,
        node_xy=xy.astype(np.float32),
        node_ids=reference["node_ids"],
    )
    print(json.dumps({"run": str(run_dir), "figures": sorted(path.name for path in output.glob("*.png"))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.observation_dir, args.evaluation_dir, args.run_dir)


if __name__ == "__main__":
    main()
