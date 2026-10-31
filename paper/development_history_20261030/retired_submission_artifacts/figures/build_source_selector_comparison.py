#!/usr/bin/env python3
"""Plot the frozen source-selector confirmation without recomputing statistics.

Run from any working directory. The accompanying CSV files preserve the
registered equal-city, paired-seed effects and their original bootstrap CIs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "source_data"
INK = "#192D41"
BLUE = "#244C70"
LINE = "#748391"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
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


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    comparisons = {
        row["comparator"]: row
        for row in read_rows("source_selector_confirmation_comparisons.csv")
        if row["scope"] == "macro"
    }
    methods = ["target_only", "equal_all_sources", "h2oplus", "phase_pressure"]
    if set(comparisons) != set(methods):
        raise ValueError("The figure requires all four registered macro comparisons.")
    seed_rows = sorted(
        (row for row in read_rows("source_selector_confirmation_seed.csv")
         if row["scope"] == "macro" and row["comparator"] == "phase_pressure"),
        key=lambda row: int(row["seed_order_1based"]),
    )
    if [int(row["seed_order_1based"]) for row in seed_rows] != list(range(1, 57)):
        raise ValueError("The figure requires all 56 seeds in their frozen order.")

    means = np.array([float(comparisons[key]["registered_mean_relative_delta_pct"])
                      for key in methods])
    lows = np.array([float(comparisons[key]["registered_bootstrap_ci95_low_pct"])
                     for key in methods])
    highs = np.array([float(comparisons[key]["registered_bootstrap_ci95_high_pct"])
                      for key in methods])
    effects = np.array([float(row["registered_relative_delta_pct"]) for row in seed_rows])
    pp = comparisons["phase_pressure"]

    fig = plt.figure(figsize=(7.1, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.143,
                           right=0.985, bottom=0.27, top=0.84, wspace=0.43)
    left, right = [fig.add_subplot(cell) for cell in grid]

    positions = np.arange(3, -1, -1)
    left.axvline(0, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    left.errorbar(means, positions, xerr=np.array([means - lows, highs - means]),
                  fmt="o", color=BLUE, ecolor=BLUE, markersize=4.2,
                  elinewidth=1.1, capsize=3, capthick=0.9, zorder=3)
    left.set(yticks=positions, yticklabels=["Target anchor", "Equal sources", "Dense linear", "PP"],
             ylim=(-0.55, 3.55), xlim=(-11.2, 1.5), xticks=[-10, -5, 0],
             xlabel="Waiting-time change (%)")
    left.tick_params(axis="y", length=0, pad=6)
    left.text(-0.39, 1.16, "a", transform=left.transAxes, weight="bold", fontsize=9)
    left.text(0, 1.16, "Against each comparator", transform=left.transAxes,
              weight="bold", fontsize=8.5)

    order = np.arange(1, len(effects) + 1)
    right.axhspan(lows[-1], highs[-1], color=BLUE, alpha=0.10, linewidth=0)
    right.axhline(0, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    right.axhline(means[-1], color=BLUE, lw=0.9, zorder=2)
    right.vlines(order, 0, effects, color=LINE, linewidth=0.55, alpha=0.6, zorder=2)
    right.scatter(order, effects, s=12, c=BLUE, edgecolors="none", zorder=3)
    right.set(xlim=(0, 57), xticks=[1, 14, 28, 42, 56], ylim=(-44, 13),
              yticks=[-40, -20, 0, 10], xlabel="Frozen confirmation seed order",
              ylabel="Waiting-time change (%)")
    right.text(-0.19, 1.16, "b", transform=right.transAxes, weight="bold", fontsize=9)
    right.text(0, 1.16, "All paired effects against PP", transform=right.transAxes,
               weight="bold", fontsize=8.5)
    fig.text(0.5, 0.092,
             f"Negative values favor CFCMT.  Against PP: {means[-1]:.2f}% "
             f"(95% CI [{lows[-1]:.2f}, {highs[-1]:.2f}]%); "
             f"{int(pp['improved_seed_count'])}/56 seeds lower.",
             ha="center", fontsize=7.4)
    fig.text(0.5, 0.025,
             "56 new paired seeds  ·  Equal city weight  ·  Registered 95% bootstrap confidence intervals",
             ha="center", fontsize=7.4, color=LINE)

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"source_selector_comparison.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
