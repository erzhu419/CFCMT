"""Draw the main figures and supplementary figure/table from recorded data."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import textwrap

from schematic_figures import draw_framework
from guard_figure import draw_guard_comparison


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
FIGURES = ROOT / "figures"
WAITING = "mean_tripinfo_waiting_time"
COLORS = {
    "phase_pressure": "#7F8C8D", "max_pressure": "#E67E22",
    "exact_generalized_pressure": "#4F6F68",
    "h2oplus": "#56708B", "target_only": "#2980B9",
    "equal_all_sources": "#8E44AD", "frozen_selector": "#C0392B",
}
METHODS = ["phase_pressure", "max_pressure", "exact_generalized_pressure",
           "h2oplus", "target_only", "equal_all_sources", "frozen_selector"]
METHOD_LABELS = ["PP", "MP", "Exact\nGP", "Dense",
                 "Target\nanchor", "Equal\nsources", "CFCMT"]
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
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.285, top=0.82,
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
    fig.suptitle("Supplementary Fig. S1. Waiting by departure cohort", y=0.98,
                 fontsize=10, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color="#34495E", lw=1.35, label="Mean"),
                        Line2D([0], [0], color="#34495E", lw=0.9, ls="--", label="90th percentile")],
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.text(0.5, 0.185,
             "Los Angeles · Seed 1020485 · 60-s departure bins",
             ha="center", va="bottom", fontsize=7.0, color="#46545E")
    caption = (
        "Vehicles are grouped by actual departure time. Curves show the mean and "
        "90th-percentile waiting accrued by the 3,600-s endpoint, including "
        "unfinished trips. Later departures have shorter observation windows. "
        "Seed 1020485 is the smallest seed in the evaluation panel; it was not "
        "selected by controller performance. Panel totals show actual departed "
        "populations, which can differ across controllers. CFCMT denotes "
        "Causal-Factored Cross-City Model Transfer."
    )
    fig.text(0.03, 0.035, textwrap.fill(caption, width=139),
             ha="left", va="bottom", fontsize=7.0, linespacing=1.25)
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
    note(fig, "Weight 0.75 source screen; Los Angeles uses anchor; Jinan curve before guard selection · Means ± SD")
    return fig


def draw_benchmark_distributions():
    data = pd.read_csv(DATA / "seven_controller_rollouts.csv")
    scenarios = ["la_1x4", "jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"]
    assert len(data) == 1568 and not data.duplicated(["scenario", "method", "seed"]).any()
    assert set(data.method) == set(METHODS) and set(data.scenario) == set(scenarios)
    original = pd.read_csv(DATA / "confirmation_rollouts.csv")
    for scenario in scenarios:
        panel = set(original.loc[original.scenario == scenario, "seed"])
        assert len(panel) == 56
        for method in METHODS:
            assert set(data.loc[(data.scenario == scenario) & (data.method == method), "seed"]) == panel
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
        x = np.arange(1, len(METHODS) + 1)
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
        ax.set_xlim(0.4, len(METHODS) + 0.6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Run-mean waiting (s)")
        clean(ax)
    fig.suptitle("Frozen policy evaluation on the benchmark seed panel", y=0.97,
                 fontsize=9, fontweight="bold")
    note(fig, "1,568 recorded runs · Boxes: medians and quartiles · Densities from recorded observations")
    return fig


def draw_execution_summary():
    """Tabulate complete logged proposal counts; no physical-change inference."""
    data = pd.read_csv(DATA / "execution_summary.csv")
    order = ["frozen_selector", "target_only", "equal_all_sources", "h2oplus"]
    names = ["CFCMT", "Target anchor", "Equal source weights", "Dense linear residual"]
    count_fields = ["model_reference_agreements", "guard_rejections_total",
                    "cooldown_rejections", "spatial_rejections",
                    "executor_rejections", "effective_overrides"]
    for row in data.itertuples(index=False):
        assert sum(int(getattr(row, key)) for key in count_fields) == int(row.post_warmup_intersection_decisions)
        assert int(row.effective_switch_overrides) + int(row.effective_stay_overrides) == int(row.effective_overrides)
        assert np.isclose(100 * row.effective_overrides / row.post_warmup_intersection_decisions,
                          row.effective_overrides_pct_of_decisions, atol=1e-12)
    figure = plt.figure(figsize=(7.16, 6.4))
    figure.text(0.5, 0.965, "Supplementary Table S1. Logged learned-policy decisions",
                ha="center", va="top", fontsize=10, fontweight="bold")
    panels = [("los_angeles", "(a) Los Angeles", [0.035, 0.58, 0.93, 0.26]),
              ("jinan", "(b) Jinan", [0.035, 0.26, 0.93, 0.26])]
    columns = ["Controller", "Reference\nagreement", "Guard\nrejections",
               "Cooldown\nrejections", "Spatial\nrejections", "Executor\nrejections",
               "Effective\noverrides"]
    widths = [0.22, 0.14, 0.14, 0.13, 0.12, 0.12, 0.13]
    for city, title, bounds in panels:
        rows = data.loc[data.city == city].set_index("method").loc[order]
        assert rows.post_warmup_intersection_decisions.nunique() == 1
        assert rows.rollout_count.nunique() == 1
        denominator = int(rows.post_warmup_intersection_decisions.iloc[0])
        runs = int(rows.rollout_count.iloc[0])
        axis = figure.add_axes(bounds)
        axis.axis("off")
        axis.text(0, 1.10, f"{title}: {runs} runs; D = {denominator:,} decisions per controller",
                  transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=8.2, fontweight="bold")
        values = []
        for name, (_, row) in zip(names, rows.iterrows()):
            values.append([name, *[f"{int(row[key]):,}\n({100 * row[key] / denominator:.6f}%)"
                                   for key in count_fields]])
        table = axis.table(cellText=values, colLabels=columns, colWidths=widths,
                           cellLoc="right", colLoc="right", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(7.4)
        for (rr, cc), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor("#CBD2D7")
            cell.PAD = 0.045
            if cc == 0:
                cell.get_text().set_ha("left")
            if rr == 0:
                cell.set_facecolor("#E9EEF1")
                cell.get_text().set_fontweight("bold")
            elif rr == 1:
                cell.set_facecolor("#F4F7F9")
    notes = (
        "D counts post-warm-up intersection-by-epoch decisions, excluding the 60-s warm-up, "
        "pooled across 56 benchmark seeds and all demands for each city. Each cell gives "
        "a count and its percentage of D. Reference agreement means that the phase selected "
        "by the model matches the generalized pressure reference. Effective overrides are executor-accepted non-reference "
        "switch requests plus effective holds of the current green; switch requests "
        "can still be in yellow or all-red clearance. Guard rejections are separate "
        "from cooldown, spatial-coordination, and executor rejections. The six count "
        "columns sum to D in every row. These logs cover the four learned policies; "
        "the three pressure controllers use no learned-override audit. "
        "Per-run counts are in source_data/execution_rollouts.csv; "
        "recompute them with source_data/recompute_execution_summaries.py."
    )
    figure.text(0.035, 0.035, textwrap.fill(notes, width=132),
                ha="left", va="bottom", fontsize=7.0, linespacing=1.3)
    return figure


def main():
    FIGURES.mkdir(exist_ok=True)
    items = [
        ("fig1_framework", draw_framework()),
        ("fig2_source_selection", draw_source_selection()),
        ("fig3_benchmark_distributions", draw_benchmark_distributions()),
        ("fig4_guard_comparison", draw_guard_comparison()),
    ]
    for name, figure in items:
        for extension in ("pdf", "png", "svg"):
            figure.savefig(FIGURES / f"{name}.{extension}", dpi=300,
                           bbox_inches="tight", pad_inches=0.06)
        print("Saved", name)
    supplementary = draw_departure_waiting()
    execution = draw_execution_summary()
    with PdfPages(ROOT / "Historical_supplement.pdf") as document:
        document.savefig(supplementary, bbox_inches="tight", pad_inches=0.06)
        document.savefig(execution, bbox_inches="tight", pad_inches=0.06)
    execution.savefig(FIGURES / "tableS1_execution_summary.png", dpi=300,
                      bbox_inches="tight", pad_inches=0.06)
    for extension in ("png", "svg"):
        supplementary.savefig(FIGURES / f"figS1_departure_waiting.{extension}",
                               dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close("all")
    print("Saved four figure assets and Historical_supplement.pdf (Fig. S1 and Table S1).")


if __name__ == "__main__":
    main()
