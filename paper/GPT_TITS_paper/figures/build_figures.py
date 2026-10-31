#!/usr/bin/env python3
"""Build the main-text figures from the accompanying source-data exports.

Run from any working directory:
    python3 figures/build_figures.py

Requires Python 3, matplotlib and numpy.  Figure dimensions are final IEEE
column widths; PDFs retain vector text, shapes and confidence-interval marks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
INK = "#192D41"
BLUE = "#244C70"
LINE = "#748391"
LIGHT = "#EFF4F8"
GREEN = "#EAF2EE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.facecolor": "white",
})


def save(fig: plt.Figure, stem: str) -> None:
    """Preserve the authored physical page size in both formats."""
    fig.savefig(HERE / f"{stem}.pdf", metadata={"Creator": "Matplotlib", "Title": stem})
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    plt.close(fig)


def box(ax, x, y, w, h, text, *, fill=LIGHT, fontsize=8.5):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=0.75, edgecolor=LINE, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.25)


def arrow(ax, points, *, color=LINE):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color=color, shrinkA=0, shrinkB=0,
    ))


def method_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    ax.text(0.008, 0.96, "PREDEPLOYMENT ADAPTATION", weight="bold", fontsize=9)
    box(ax, 0.01, 0.61, 0.195, 0.245,
        "Matched SUMO states\nSource + target data\nAll phase outcomes", fontsize=8.2)
    ax.text(0.1075, 0.55, "Two target adaptation\nseeds supply action labels", ha="center", va="center",
            fontsize=7.6, color=BLUE, linespacing=1.2)
    box(ax, 0.26, 0.75, 0.195, 0.12,
        "Per-source components\nReference + all-pair", fontsize=7.8)
    box(ax, 0.26, 0.56, 0.195, 0.12, "Target anchor", fontsize=8.0)
    box(ax, 0.515, 0.61, 0.215, 0.245,
        "8 target development seeds\nSelect source weights\nand execution guard", fontsize=7.8)
    box(ax, 0.79, 0.61, 0.195, 0.245,
        "Freeze selected\npredictor and guard\nbefore evaluation", fontsize=8.2)

    arrow(ax, [(0.21, 0.733), (0.235, 0.733), (0.235, 0.81), (0.254, 0.81)])
    arrow(ax, [(0.235, 0.733), (0.235, 0.62), (0.254, 0.62)])
    arrow(ax, [(0.46, 0.81), (0.485, 0.81), (0.485, 0.733), (0.509, 0.733)])
    arrow(ax, [(0.46, 0.62), (0.485, 0.62), (0.485, 0.733)])
    arrow(ax, [(0.735, 0.733), (0.784, 0.733)])

    ax.text(0.008, 0.405, "ONLINE CONTROL", weight="bold", fontsize=9)
    box(ax, 0.01, 0.18, 0.195, 0.18,
        "Observed target state\n+ feasible phases", fill=GREEN, fontsize=8.0)
    box(ax, 0.26, 0.18, 0.195, 0.18,
        "Frozen predictor\nranks phase costs", fill=GREEN, fontsize=8.0)
    box(ax, 0.515, 0.18, 0.215, 0.18,
        "Guard and coordination\nLearned phase or\npressure reference", fill=GREEN, fontsize=7.9)
    box(ax, 0.79, 0.18, 0.195, 0.18,
        "Shared signal executor\nMinimum green,\nyellow and all-red", fill=GREEN, fontsize=7.8)
    arrow(ax, [(0.21, 0.27), (0.254, 0.27)])
    arrow(ax, [(0.46, 0.27), (0.509, 0.27)])
    arrow(ax, [(0.735, 0.27), (0.784, 0.27)])
    arrow(ax, [(0.8875, 0.604), (0.8875, 0.45), (0.3575, 0.45), (0.3575, 0.366)])
    ax.text(0.53, 0.48, "frozen predictor and guard", fontsize=7.8, color=BLUE)

    ax.plot([0.01, 0.985], [0.11, 0.11], color="#CFD7DE", linewidth=0.65)
    ax.text(0.5, 0.045,
        "The 56 evaluation seeds are withheld from model and guard selection.",
        ha="center", va="center", fontsize=8.2)
    save(fig, "method_workflow")



if __name__ == "__main__":
    method_workflow()
