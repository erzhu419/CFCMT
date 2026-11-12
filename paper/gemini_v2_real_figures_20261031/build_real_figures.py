"""Draw the six V93 figures using archived experimental observations only."""

import os
from pathlib import Path
import zipfile

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from schematic_figures import draw_framework, draw_timeline
from guard_figure import draw_guard_comparison


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
FIGURES = ROOT / "figures"
WAITING = "mean_tripinfo_waiting_time"
COLORS = {
    "phase_pressure": "#7F8C8D", "max_pressure": "#E67E22",
    "h2oplus": "#56708B", "target_only": "#2980B9",
    "equal_all_sources": "#8E44AD", "frozen_selector": "#C0392B",
}
METHODS = ["phase_pressure", "max_pressure", "h2oplus", "target_only",
           "equal_all_sources", "frozen_selector"]
METHOD_LABELS = ["Phase\nPressure", "Max\nPressure", "Dense", "Target\nanchor",
                 "Equal\nsources", "CFCMT"]
GROUPS = ["atlanta", "cologne", "hangzhou", "ingolstadt", "new_york",
          "resco_synthetic", "salt_lake_city"]
GROUP_LABELS = ["Atlanta", "Cologne", "Hangzhou", "Ingolstadt", "New York",
                "Synthetic", "Salt Lake City"]
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.8,
    "axes.titlesize": 8.2, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 300, "axes.linewidth": 0.7,
})


def clean(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D5D9DD", linewidth=0.5, linestyle=":")
    ax.set_axisbelow(True)


def note(fig, text):
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=6.6,
             color="#46545E")


def draw_departure_waiting():
    data = pd.read_csv(DATA / "runtime_departure_bins.csv")
    totals = pd.read_csv(DATA / "runtime_seed_case_metrics.csv").set_index("method")
    methods = ["phase_pressure", "max_pressure", "equal_all_sources", "frozen_selector"]
    names = ["PhasePressure", "MaxPressure", "Equal source weights", "CFCMT"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.84,
                        wspace=0.18, hspace=0.42)
    maximum = data["p90_waiting_sec"].max()
    for letter, ax, method, name in zip("abcd", axes.flat, methods, names):
        rows = data.loc[data.method == method].sort_values("bin_start_sec")
        assert len(rows) == 60 and rows.seed.nunique() == 1
        x = rows.departure_minute_midpoint.to_numpy()
        ax.plot(x, rows.mean_waiting_sec, color=COLORS[method], linewidth=1.35)
        ax.plot(x, rows.p90_waiting_sec, color=COLORS[method], linewidth=0.9,
                linestyle="--", alpha=0.85)
        total = totals.loc[method]
        ax.set_title(f"({letter}) {name}", loc="left", fontweight="bold")
        ax.text(0.98, 0.96, f"{int(total.n_departed):,} departed trips",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_xlim(0, 60)
        ax.set_ylim(0, maximum * 1.13)
        ax.set_xticks([0, 15, 30, 45, 60])
        clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("Departure time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accrued trip waiting (s)")
    fig.suptitle("Los Angeles: observed waiting by departure cohort", y=0.98,
                 fontsize=9, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color="#34495E", lw=1.35, label="Mean"),
                        Line2D([0], [0], color="#34495E", lw=0.9, ls="--", label="90th percentile")],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    note(fig, "Fixed seed 1020485 · 60-s departure bins · Waiting observed to 3,600 s, including unfinished trips")
    return fig


def draw_source_selection():
    identity = pd.read_csv(DATA / "source_identity.csv")
    weight = pd.read_csv(DATA / "source_weight.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.80, wspace=0.43)
    for ax, city, title in zip(axes[:2], ["los_angeles", "jinan"],
                                ["(a) Los Angeles", "(b) Jinan"]):
        for row, group in enumerate(GROUPS):
            values = identity.loc[(identity.city == city) & (identity.source_group == group),
                                  "delta_vs_target_pct"].to_numpy()
            assert len(values) == 8
            selected = city == "jinan" and group == "atlanta"
            color = COLORS["frozen_selector"] if selected else "#56708B"
            ax.scatter(values, row + np.linspace(-0.14, 0.14, len(values)),
                       color="#AAB4BB", s=5, alpha=0.8, zorder=2, linewidths=0)
            ax.errorbar(values.mean(), row, xerr=values.std(ddof=1), fmt="o",
                        color=color, markersize=4.0, elinewidth=1.0, capsize=2, zorder=3)
        ax.axvline(0, color="#6F7880", ls="--", lw=0.8)
        ax.set_yticks(range(len(GROUPS)), GROUP_LABELS if city == "los_angeles" else [])
        ax.invert_yaxis()
        ax.set_ylim(6.5, -0.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Waiting change vs.\ntarget anchor (%)")
        clean(ax, "x")
    ax = axes[2]
    rows = weight.loc[weight.city == "jinan"]
    table = rows.pivot(index="source_mass", columns="seed", values="mean_waiting_sec").sort_index()
    assert table.shape == (5, 8)
    for values in table.to_numpy().T:
        ax.plot(table.index, values, color="#CBD2D7", lw=0.55, zorder=1)
    means = table.mean(axis=1)
    std = table.std(axis=1, ddof=1)
    ax.errorbar(table.index, means, yerr=std, fmt="o-", color=COLORS["frozen_selector"],
                lw=1.4, capsize=2.2, markersize=4.2, zorder=3)
    ax.set_xlabel("Atlanta source weight")
    ax.set_ylabel("Mean waiting (s)", labelpad=2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("(c) Jinan: weight scan", loc="left", fontweight="bold")
    clean(ax)
    fig.suptitle("Source selection on eight paired development seeds", y=0.97,
                 fontsize=9, fontweight="bold")
    note(fig, "Source screen: weight 0.75; LA selects anchor; Jinan curve precedes guard selection · Means ± SD")
    return fig


def draw_confirmation_distributions():
    data = pd.read_csv(DATA / "confirmation_rollouts.csv")
    assert len(data) == 1344 and not data.duplicated(["scenario", "method", "seed"]).any()
    scenarios = ["la_1x4", "jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"]
    titles = ["(a) Los Angeles (log scale)", "(b) Jinan: original demand",
              "(c) Jinan: demand file 2000", "(d) Jinan: demand file 2500"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.88,
                        wspace=0.27, hspace=0.55)
    for ax, scenario, title in zip(axes.flat, scenarios, titles):
        samples = []
        for method in METHODS:
            values = data.loc[(data.scenario == scenario) & (data.method == method), WAITING]
            assert len(values) == 56
            samples.append(values.to_numpy())
        x = np.arange(1, 7)
        violins = ax.violinplot(samples, positions=x, widths=0.68,
                               showmeans=False, showmedians=False, showextrema=False)
        for body, method in zip(violins["bodies"], METHODS):
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_linewidth(0.65)
            body.set_alpha(0.15)
        boxes = ax.boxplot(samples, positions=x, widths=0.23, showfliers=False,
                           patch_artist=True, manage_ticks=False,
                           medianprops=dict(color="#17242C", lw=1.0),
                           whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        for box, method in zip(boxes["boxes"], METHODS):
            box.set_facecolor(COLORS[method]); box.set_alpha(0.6)
            box.set_edgecolor("#34434D"); box.set_linewidth(0.65)
        for position, values, method in zip(x, samples, METHODS):
            # Deterministic display offsets; every y value is an observed run mean.
            order = np.argsort(values, kind="stable")
            offsets = ((np.arange(56) % 7) - 3) * 0.024
            ax.scatter(position + offsets, values[order], s=4.2,
                       color=COLORS[method], alpha=0.6, edgecolors="none", zorder=4)
        if scenario == "la_1x4":
            ax.set_yscale("log")
            ax.set_yticks([180, 250, 400, 600, 1000])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        lowest = min(v.min() for v in samples)
        highest = max(v.max() for v in samples)
        if scenario == "la_1x4":
            ax.set_ylim(lowest / 1.10, highest * 1.10)
        else:
            padding = (highest - lowest) * 0.08
            ax.set_ylim(max(0.1, lowest - padding), highest + padding)
        ax.set_xticks(x, METHOD_LABELS, fontsize=6.6)
        ax.set_xlim(0.4, 6.6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Run-mean waiting (s)")
        clean(ax)
    fig.suptitle("Frozen confirmation: all 56 runs per method and scenario", y=0.97,
                 fontsize=9, fontweight="bold")
    note(fig, "1,344 recorded runs · All observations and tails retained · Boxes: median and quartiles; density estimated from observations")
    return fig


def main():
    os.chdir(ROOT)
    FIGURES.mkdir(exist_ok=True)
    items = [
        ("fig1_mechanism_timeline", draw_timeline()),
        ("fig2_framework_overview", draw_framework()),
        ("fig3_departure_waiting", draw_departure_waiting()),
        ("fig4_source_selection", draw_source_selection()),
        ("fig5_confirmation_distributions", draw_confirmation_distributions()),
        ("fig6_guard_comparison", draw_guard_comparison()),
    ]
    for name, figure in items:
        for extension in ("pdf", "png", "svg"):
            figure.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.06)
        print("Saved", name)
    for name, selected in [("CFCMT_V93_six_figures.pdf", items),
                           ("CFCMT_V93_five_main_figures.pdf", items[1:])]:
        with PdfPages(ROOT / name) as document:
            for _, figure in selected:
                document.savefig(figure, bbox_inches="tight", pad_inches=0.06)
    members = [*ROOT.glob("*.py"), ROOT / "README.md",
               *sorted(DATA.glob("*")), *sorted(FIGURES.glob("*.png")),
               *sorted(FIGURES.glob("*.pdf")), *sorted(FIGURES.glob("*.svg")),
               ROOT / "CFCMT_V93_six_figures.pdf", ROOT / "CFCMT_V93_five_main_figures.pdf"]
    with zipfile.ZipFile(ROOT / "CFCMT_V93_real_figures.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    plt.close("all")
    print("Saved all six real-data figures and their source bundle.")


if __name__ == "__main__":
    main()
