"""Render the archived case replays for the Partial-CV revision."""

import json
from pathlib import Path

import numpy as np

from build_figures import C_CFCMT, C_CONS, plt, save_both
from paired_comparisons import EXP_DIR


COLORS = {
    "conservation_gp_10": C_CONS,
    "zero_shot_cfcmt_b0": "#BA639B",
    "cfcmt_b6": C_CFCMT,
}
LABELS = {
    "conservation_gp_10": "Cons-GP",
    "zero_shot_cfcmt_b0": "CFCMT ($B=0$)",
    "cfcmt_b6": "CFCMT ($B=6$)",
}


def trailing_green_percent(values):
    ending = np.arange(1, len(values) + 1)
    starting = np.maximum(ending - 30, 0)
    totals = np.r_[0, np.cumsum(values)]
    return 100 * (totals[ending] - totals[starting]) / (ending - starting)


def draw_case_figure():
    data = json.loads((EXP_DIR / "phase4_case_trace_summary.json").read_text())
    time = np.asarray(data["time_sec"]) / 60
    profiles = data["profiles"]
    fig, axes = plt.subplots(3, len(profiles), sharex="col", figsize=(7.17, 3.68))
    handles = {}
    titles = ("LA, 920104", "LA, 920105", "LA, 920108", "Salt Lake, 930108")
    for j, profile in enumerate(profiles):
        axes[0, j].set_title(titles[j])
        for policy, series in profile["policies"].items():
            color = COLORS[policy]
            line = axes[0, j].plot(time[:-1], series["entry_pending"], color=color)
            handles[policy] = line[0]
            axes[1, j].plot(time, series["completed"], color=color)
            requested = trailing_green_percent(series["entry_requested_green"])
            sampled = trailing_green_percent(series["entry_sampled_green"])
            axes[2, j].plot(
                time[:-1], sampled, color=color, linewidth=1.1,
            )
            axes[2, j].plot(
                time[:-1], np.where(np.isclose(requested, sampled), np.nan, requested),
                color=color, linestyle="--", alpha=0.8, linewidth=0.9, zorder=5,
            )
        axes[2, j].set_ylim(-4, 104)
        axes[2, j].set_yticks((0, 50, 100))
        axes[2, j].set_xticks((0, 20, 40, 60))
        axes[2, j].set_xlabel("Time (min)")
        for ax in axes[:, j]:
            ax.set_xlim(0, 60)
            ax.grid(axis="y", color="0.87", linewidth=0.5)

    for i, label in enumerate(("Entry pending", "Completed", "Entry green (%)")):
        axes[i, 0].set_ylabel(label)

    fig.legend(
        [handles[policy] for policy in LABELS], list(LABELS.values()),
        loc="lower center", bbox_to_anchor=(0.54, 0.0), ncol=3, frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.2, wspace=0.4, hspace=0.24)
    save_both(fig, Path(__file__).resolve().parents[1] / "figures", "fig5_case_diagnostics")


if __name__ == "__main__":
    draw_case_figure()
