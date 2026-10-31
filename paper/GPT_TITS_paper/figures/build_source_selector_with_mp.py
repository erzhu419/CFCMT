#!/usr/bin/env python3
"""Draw the frozen comparison, city effects, and all paired-effect distributions.

Intervals and mean estimates are read unchanged from the accompanying exports.
The empirical cumulative curves count all 56 recorded effects for each pressure
comparator. No bootstrap or statistical test is calculated by this script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "source_data"
INK = "#192D41"
BLUE = "#244C70"
LINE = "#748391"
TEAL = "#5C7D78"
PRESSURE = (("phase_pressure", "PP", TEAL), ("max_pressure", "MP", BLUE))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.8,
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


def interval(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(float(row[key]) for key in (
        "registered_mean_relative_delta_pct",
        "registered_bootstrap_ci95_low_pct",
        "registered_bootstrap_ci95_high_pct",
    ))


def point_interval(ax, row, position, color):
    mean, low, high = interval(row)
    ax.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="o",
                color=color, ecolor=color, markersize=4.1, elinewidth=1.1,
                capsize=2.7, capthick=0.9, zorder=3)


def panel_title(ax, letter: str, title: str, *, left: float = 0.0):
    ax.text(left, 1.10, letter, transform=ax.transAxes,
            weight="bold", fontsize=9)
    ax.text(0, 1.10, title, transform=ax.transAxes,
            weight="bold", fontsize=8.3)


def main() -> None:
    methods = ["target_only", "equal_all_sources", "h2oplus", "phase_pressure", "max_pressure"]
    comparisons = {
        (row["scope"], row["comparator"]): row
        for row in read_rows("source_selector_with_mp_comparisons.csv")
    }
    if {key for scope, key in comparisons if scope == "macro"} != set(methods):
        raise ValueError("The figure requires all five recorded macro comparisons.")
    seed_rows = read_rows("source_selector_with_mp_seed.csv")
    pressure_rows = {
        method: sorted(
            (row for row in seed_rows
             if row["scope"] == "macro" and row["comparator"] == method),
            key=lambda row: int(row["seed_order_1based"]),
        )
        for method, _, _ in PRESSURE
    }
    for method, rows in pressure_rows.items():
        if [int(row["seed_order_1based"]) for row in rows] != list(range(1, 57)):
            raise ValueError(f"All 56 paired effects are required for {method}.")
    if [row["seed"] for row in pressure_rows["phase_pressure"]] != [
        row["seed"] for row in pressure_rows["max_pressure"]
    ]:
        raise ValueError("PP and MP must retain the same complete paired seeds.")
    effects = {
        method: np.array([float(row["registered_relative_delta_pct"]) for row in rows])
        for method, rows in pressure_rows.items()
    }

    fig = plt.figure(figsize=(7.1, 4.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.10], left=0.143,
                           right=0.985, bottom=0.205, top=0.885,
                           wspace=0.58, hspace=0.80)
    forest = fig.add_subplot(grid[:, 0])
    city_ax = fig.add_subplot(grid[0, 1])
    ecdf_ax = fig.add_subplot(grid[1, 1])

    positions = np.arange(4, -1, -1)
    forest.axvline(0, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for method, position in zip(methods, positions):
        color = TEAL if method == "phase_pressure" else BLUE
        point_interval(forest, comparisons["macro", method], position, color)
    bounds = np.array([interval(comparisons["macro", method])[1:] for method in methods])
    forest.set(yticks=positions,
               yticklabels=["Target anchor", "Equal sources", "Dense linear", "PP", "MP"],
               ylim=(-0.6, 4.6), xlim=(min(bounds.min(), 0) - 0.8, max(bounds.max(), 0) + 0.8),
               xlabel="Waiting-time change (%)")
    forest.xaxis.set_major_locator(MaxNLocator(nbins=4))
    forest.tick_params(axis="y", length=0, pad=6)
    panel_title(forest, "a", "Overall comparison", left=-0.44)

    city_rows = [(city, label, method, short, color)
                 for city, label in (("los_angeles", "Los Angeles"), ("jinan", "Jinan"))
                 for method, short, color in PRESSURE]
    city_ax.axvline(0, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for position, (city, _, method, _, color) in zip(range(3, -1, -1), city_rows):
        point_interval(city_ax, comparisons[city, method], position, color)
    city_ax.set(yticks=[3, 2, 1, 0],
                yticklabels=[f"{label} / {short}" for _, label, _, short, _ in city_rows],
                ylim=(-0.55, 3.55), xlim=(-9.5, 1.5), xticks=[-8, -4, 0],
                xlabel="Waiting-time change (%)")
    city_ax.tick_params(axis="y", length=0, pad=5, labelsize=7.2)
    city_ax.tick_params(axis="x", labelsize=7.2)
    panel_title(city_ax, "b", "City-specific effects", left=-0.48)

    combined = np.concatenate(list(effects.values()))
    lower = 5 * np.floor(combined.min() / 5)
    upper = 5 * np.ceil(combined.max() / 5)
    ecdf_ax.axvline(0, color=LINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for method, short, color in PRESSURE:
        observed = np.sort(effects[method])
        cumulative = np.arange(1, observed.size + 1) / observed.size
        ecdf_ax.step(np.r_[lower, observed, upper], np.r_[0, cumulative, 1],
                     where="post", color=color, lw=1.15, label=short)
    ecdf_ax.set(xlim=(lower, upper), ylim=(0, 1.025), yticks=[0, 0.5, 1],
                xticks=[-40, -20, 0, 10], xlabel="Paired waiting-time change (%)",
                ylabel="Cumulative proportion")
    ecdf_ax.tick_params(labelsize=7.2)
    ecdf_ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.3,
                   borderaxespad=0.2, handlelength=1.4, columnspacing=0.9)
    panel_title(ecdf_ax, "c", "Distribution of all 56 paired effects", left=-0.48)

    summaries = []
    for method, label, _ in PRESSURE:
        mean, low, high = interval(comparisons["macro", method])
        summaries.append(f"{label}: {mean:.2f}% [{low:.2f}, {high:.2f}]%")
    fig.text(0.5, 0.075, "   ·   ".join(summaries), ha="center", fontsize=7.6)
    fig.text(0.5, 0.025,
             "Negative changes favor CFCMT  ·  56 paired seeds  ·  95% paired-seed bootstrap intervals",
             ha="center", fontsize=7.2, color=LINE)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"source_selector_with_mp.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
