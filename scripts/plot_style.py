"""Shared IEEE-oriented Matplotlib style for Section V figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "reference": "#222222",
    "candidate": "#8C8C8C",
    "learned": "#0072B2",
    "tp": "#009E73",
    "fp": "#D55E00",
    "fn": "#CC79A7",
    "forecast": "#0072B2",
    "topology": "#D55E00",
    "objective": "#009E73",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    for extension in ("pdf", "svg", "png"):
        target = root / extension / f"{stem}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300 if extension == "png" else None)
