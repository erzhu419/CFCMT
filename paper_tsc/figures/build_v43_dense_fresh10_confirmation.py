"""Reproduce the frozen-model fresh-seed confirmation figure and source tables.

The primary comparison is fixed-city seed replication: every bootstrap draw
retains both policies, both cities and all four scenarios in a seed block.
Only the registered overall interval is plotted; city values are descriptive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "paper_tsc/source_data"
TABLES = REPO / "paper_tsc/tables"
CAMPAIGN = REPO / "experiments/tsc_v43_dense_confirmation_20261022"
METHOD = "cfcmt_selected_mpc"
DENSE = "h2oplus_style_dense_residual_mpc"
WAIT = "mean_tripinfo_waiting_time"
METRICS = (WAIT, "mean_completed_tripinfo_waiting_time", "system_vehicle_hours", "completion_ratio")
BLUE, TEAL, INK, GREY = "#345E87", "#2E7D57", "#262C33", "#A9B1BB"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "legend.frameon": False,
})


def write_csv(name, rows):
    with (SOURCE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    report = json.loads((CAMPAIGN / "reporting.json").read_text())
    protocol = json.loads((CAMPAIGN / "protocol.json").read_text())
    if not report["complete"] or report["ready"] != 80 or report["failures"]:
        raise ValueError("The complete, unfiltered 80-rollout confirmation is required")
    SOURCE.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    macro = report["macro"]
    groups = {"equal_city_macro": report["secondary"], **report["by_city"], **report["by_scenario"]}
    summary = []
    for scope, group in groups.items():
        for metric in METRICS:
            c, d = (group["means"][policy][metric] for policy in (METHOD, DENSE))
            primary = scope == "equal_city_macro" and metric == WAIT
            summary.append(dict(
                scope=scope, metric=metric, cfcmt_mean=c, dense_mean=d, difference=c-d,
                relative_difference_percent=100*(c/d-1), n_seed_blocks=10,
                improved_waiting_seed_count=group["better_seed_count"],
                ci95_difference_lower=macro["ci95_difference_sec"][0] if primary else "",
                ci95_difference_upper=macro["ci95_difference_sec"][1] if primary else "",
                ci95_relative_lower=macro["ci95_relative_difference_percent"][0] if primary else "",
                ci95_relative_upper=macro["ci95_relative_difference_percent"][1] if primary else "",
            ))
    write_csv("v43_dense_fresh10_summary.csv", summary)
    records = {(r["city"], r["scenario"], r["seed"], r["policy"]): r for r in report["records"]}
    seed_rows = []
    for row in report["by_seed"]:
        seed_rows.append(dict(seed=row["seed"], scope="equal_city_macro",
            cfcmt_wait_sec=row["cfcmt_wait_sec"], dense_wait_sec=row["dense_wait_sec"],
            difference_sec=row["difference_sec"], relative_difference_percent=row["relative_difference_percent"]))
        for city, scenarios in protocol["city_scenarios"].items():
            c, d = [float(np.mean([records[(city, scenario, row["seed"], policy)]["metrics"][WAIT]
                for scenario in scenarios])) for policy in (METHOD, DENSE)]
            seed_rows.append(dict(seed=row["seed"], scope=city, cfcmt_wait_sec=c, dense_wait_sec=d,
                difference_sec=c-d, relative_difference_percent=100*(c/d-1)))
    write_csv("v43_dense_fresh10_seed.csv", seed_rows)
    write_csv("v43_dense_fresh10_rollouts.csv", [dict(city=r["city"], scenario=r["scenario"],
        seed=r["seed"], policy=r["policy"], **{key:r["metrics"][key] for key in (*METRICS,
        "tripinfo_count", "tripinfo_completed_count", "tripinfo_unfinished_count",
        "starting_teleports", "ending_teleports", "collision_incidents")}) for r in report["records"]])

    table = [r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Aggregation & \shortstack{CFCMT\\wait (s)} & \shortstack{Dense\\wait (s)} & \shortstack{Difference\\(s)} & \shortstack{Reduction\\(\%)} & \shortstack{Improved\\seeds} \\",
        r"\midrule"]
    for scope, label in (("equal_city_macro", "Equal-city mean"), ("los_angeles", "Los Angeles"), ("jinan", "Jinan")):
        group = groups[scope]
        c, d = (group["means"][p][WAIT] for p in (METHOD, DENSE))
        table.append(f"{label} & {c:.2f} & {d:.2f} & {c-d:+.2f} & {100*(1-c/d):.2f} & {group['better_seed_count']}/10 " + r"\\")
    table += [r"\midrule", r"\multicolumn{6}{l}{Secondary outcomes, equal-city means} \\",
        r"Metric & CFCMT & Dense & \multicolumn{3}{l}{Difference} \\"]
    for metric, label, scale, unit in (
        ("mean_completed_tripinfo_waiting_time", "Completed-only wait (s)", 1, ""),
        ("system_vehicle_hours", "System vehicle-hours", 1, ""),
        ("completion_ratio", r"Completion (\%)", 100, " pp"),
    ):
        c, d = (groups["equal_city_macro"]["means"][p][metric]*scale for p in (METHOD, DENSE))
        table.append(f"{label} & {c:.2f} & {d:.2f} & " + r"\multicolumn{3}{l}{" + f"{c-d:+.2f}{unit}" + r"} \\")
    (TABLES / "v43_dense_fresh10_confirmation.tex").write_text("\n".join(table+[r"\bottomrule", r"\end{tabular}", ""]))

    fig, (ax, city_ax) = plt.subplots(1, 2, figsize=(7.2, 3.7),
        gridspec_kw={"width_ratios": [1, 1.22], "wspace": .48})
    seed_diffs = np.array([r["difference_sec"] for r in report["by_seed"]])
    ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.scatter(seed_diffs, np.linspace(.50, .92, 10), color=GREY, s=23, zorder=3,
        label="Individual seed means")
    ci = macro["ci95_difference_sec"]
    ax.errorbar(macro["difference_sec"], .12, xerr=[[macro["difference_sec"]-ci[0]],
        [ci[1]-macro["difference_sec"]]], fmt="D", color=INK, markersize=6,
        capsize=4, linewidth=1.7, zorder=4)
    ax.text(-23, 1.68, f"{abs(macro['relative_difference_percent']):.2f}% lower waiting",
        fontsize=11, color=INK, fontweight="bold")
    relative_ci = macro["ci95_relative_difference_percent"]
    ax.text(-23, 1.44, f"95% interval: {abs(relative_ci[1]):.2f}–{abs(relative_ci[0]):.2f}%", color=INK)
    ax.text(-23, 1.21, f"{report['better_seed_count']}/10 seed means improved", color=INK)
    ax.text(-23, -.13, f"Mean: {macro['difference_sec']:.2f} s\n95% interval: [{ci[0]:.2f}, {ci[1]:.2f}] s",
        fontsize=7.4, va="top", linespacing=1.5)
    ax.set(xlim=(-24, 3), ylim=(-.55, 1.9), yticks=[], xticks=[-20,-10,0],
        xlabel="CFCMT − dense waiting (s)")
    ax.spines["left"].set_visible(False)
    ax.set_title("a   Overall confirmation", loc="left", fontweight="bold", pad=10)

    city_ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    y = np.arange(10)
    for i, row in enumerate(report["by_seed"]):
        values = row["city_differences_sec"]
        city_ax.plot([values["los_angeles"], values["jinan"]], [i,i], color="#D8DDE3", lw=.8, zorder=1)
    for city, label, color, marker in (("los_angeles", "Los Angeles (6/10)", BLUE, "o"),
            ("jinan", "Jinan (10/10)", TEAL, "s")):
        city_ax.scatter([r["city_differences_sec"][city] for r in report["by_seed"]], y,
            s=22, marker=marker, color=color, zorder=3, label=label)
    city_ax.set(yticks=y, yticklabels=[str(r["seed"]) for r in report["by_seed"]],
        xlabel="CFCMT − dense waiting (s)", ylabel="Simulator seed", xlim=(-35,16),
        xticks=[-30,-20,-10,0,10], ylim=(9.6,-1.45))
    city_ax.tick_params(axis="y", labelsize=7.3)
    city_ax.legend(loc="upper left", bbox_to_anchor=(-.015,1.01), fontsize=7,
        borderaxespad=0, handletextpad=.3, labelspacing=.4)
    city_ax.set_title("b   City variation in every seed", loc="left", fontweight="bold", pad=10)
    fig.subplots_adjust(top=.87, bottom=.17, left=.055, right=.99)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"tsc_v43_dense_fresh10_confirmation.{extension}",
            dpi=300 if extension == "png" else 600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote independent fresh10 figure, table and three source-data CSVs.")


if __name__ == "__main__":
    main()
