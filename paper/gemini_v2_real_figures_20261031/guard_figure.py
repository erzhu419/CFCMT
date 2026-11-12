"""V93 guard-profile development results from the recorded paired runs."""

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


mpl.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
ROOT = Path(__file__).resolve().parent
PROFILES = ["raw", "mean_only", "uncertainty_0p25", "uncertainty_0p5"]
BLUE, RED = "#2980B9", "#C0392B"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def draw_guard_comparison():
    """Return paired guard effects, showing every seed and its full range."""
    with (ROOT / "source_data/source_guard.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), sharey=True)
        figure.subplots_adjust(left=0.083, right=0.99, bottom=0.265,
                               top=0.805, wspace=0.14)
        panels = [
            ("los_angeles", "(a) Los Angeles", ["target_only"], 0),
            ("jinan", "(b) Jinan", ["target_only", "selected_source"], 3),
        ]
        for axis, (city, title, candidates, chosen) in zip(axes, panels):
            axis.axvspan(chosen - 0.33, chosen + 0.33,
                         facecolor="#F0F2F3", zorder=0)
            axis.axhline(0.0, color="#6A737B", linewidth=0.7, zorder=1)
            axis.axhline(1.0, color="#58616A", linewidth=0.85,
                         linestyle=(0, (4, 3)), zorder=1)
            for candidate in candidates:
                color = BLUE if candidate == "target_only" else RED
                marker = "s" if candidate == "target_only" else "D"
                offset = 0.0 if len(candidates) == 1 else (
                    -0.14 if candidate == "target_only" else 0.14)
                for index, profile in enumerate(PROFILES):
                    matched = sorted(
                        [row for row in rows if row["city"] == city
                         and row["candidate_key"] == candidate
                         and row["guard_profile"] == profile],
                        key=lambda row: int(row["seed"]),
                    )
                    assert len(matched) == 8
                    values = np.array([float(row["delta_vs_raw_same_candidate_pct"])
                                       for row in matched])
                    center = index + offset
                    axis.scatter(center + np.linspace(-0.055, 0.055, 8), values,
                                 s=12, marker="o", color=color, alpha=0.48,
                                 linewidths=0, zorder=3)
                    mean = values.mean()
                    axis.errorbar(center, mean,
                                  yerr=[[mean - values.min()],
                                        [values.max() - mean]],
                                  color=color, ecolor=color, linewidth=1.1,
                                  capsize=3, capthick=0.9, fmt=marker,
                                  markersize=5.0, markeredgecolor="white",
                                  markeredgewidth=0.6, zorder=4)
            axis.text(chosen, 8.85, "Selected", ha="center", va="center",
                      fontsize=7.0, fontweight="bold", color="#39464F")
            axis.set(title=title, xlim=(-0.5, 3.5), ylim=(-8.3, 10.0))
            axis.set_xticks(range(4), ["No score\nguard", "Risk = 0",
                                      "Risk = 0.25", "Risk = 0.5"])
            axis.set_yticks([-8, -4, 0, 4, 8])
            axis.grid(axis="y", linestyle=":", alpha=0.32, linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_xlabel("Score-guard profile", labelpad=6)

        axes[0].set_ylabel("Paired waiting change (%)")
        handles = [
            Line2D([], [], color=BLUE, marker="s", linestyle="none",
                   label="Target anchor", markersize=5),
            Line2D([], [], color=RED, marker="D", linestyle="none",
                   label="Atlanta source (weight 1)", markersize=5),
            Line2D([], [], color="#58616A", linestyle=(0, (4, 3)),
                   linewidth=0.85, label="Worst-seed limit (+1%)"),
        ]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.98),
                      ncol=3, frameon=False, fontsize=7.1,
                      columnspacing=1.1, handletextpad=0.45, handlelength=1.6)
        figure.text(0.54, 0.105,
                    "Eight development seeds; small points = seeds; large markers = means;",
                    ha="center", va="center", fontsize=7.0, color="#4B5660")
        figure.text(0.54, 0.055,
                    "whiskers = full seed range. Reference: same candidate without the score guard.",
                    ha="center", va="center", fontsize=7.0, color="#4B5660")
        return figure


if __name__ == "__main__":
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    figure = draw_guard_comparison()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(destination / f"fig6_guard_comparison.{extension}",
                       bbox_inches="tight", dpi=300)
    plt.close(figure)
