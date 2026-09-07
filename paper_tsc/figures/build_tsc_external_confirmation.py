from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper_tsc"
TABLES = PAPER / "tables"
SOURCE_DATA = PAPER / "source_data"

DEFAULT_DEVELOPMENT = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v41r37_seed_heldout_b100_ensemble_20260809"
    / "audit"
    / "audit.json"
)
DEFAULT_EXTERNAL_ROOT = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v43r39_external_full_budget_20260809"
)
DEFAULT_HELDOUT = DEFAULT_EXTERNAL_ROOT / "heldout_seed8171_evaluation_v1" / "result.json"
DEFAULT_CLOSED_LOOP = DEFAULT_EXTERNAL_ROOT / "closed_loop_v4_aggregate.json"
DEFAULT_JOINT_AUDIT = DEFAULT_EXTERNAL_ROOT / "joint_freeze_audit_v1.json"
DEFAULT_SOURCE_ABLATION = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v90_external_source_contribution_20260830"
    / "source_contribution_audit_v1.json"
)
DEFAULT_SOURCE_CONFIRMATION = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v91_external_source_contribution_20260830"
    / "source_contribution_audit_v1.json"
)
DEFAULT_SOURCE_MANIFEST = REPO / "cf_h2o" / "config" / "traffic_signal_cross_city_v2_saltlake18.json"
DEFAULT_EXTERNAL_CONVERSION = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v42r38_external_la_jinan_20260809"
    / "conversion"
    / "full_networks_v8"
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "cfcmt-v43-paper",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


COLORS = {
    "ink": "#262C33",
    "muted": "#697582",
    "grid": "#D9DEE5",
    "fixed": "#A9B1BB",
    "transfer": "#6F94B8",
    "transfer_dark": "#345E87",
    "cfcmt": "#2E7D57",
    "cfcmt_soft": "#DDEFE5",
    "pressure": "#C08A3E",
    "pressure_dark": "#8D6128",
    "harm": "#B54E49",
    "source_road": "#C5CED8",
    "source_tls": "#496E9A",
    "target_road": "#B9CBC3",
    "target_tls": "#2E7D57",
}


POLICY_LABELS = {
    "fixed_time": "Fixed time",
    "h2oplus_style_dense_residual_mpc": "Dense residual",
    "simulator_only_mpc": "Simulator only",
    "rigid_anchor_mpc": "Rigid anchor",
    "cfcmt_selected_mpc": "Anchored CFCMT",
    "phase_pressure": "Phase pressure",
    "max_pressure": "MaxPressure",
}


POLICY_ORDER = (
    "fixed_time",
    "h2oplus_style_dense_residual_mpc",
    "simulator_only_mpc",
    "rigid_anchor_mpc",
    "cfcmt_selected_mpc",
    "phase_pressure",
    "max_pressure",
)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"cannot write empty source-data table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def export_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def validate_results(development: dict, heldout: dict, closed_loop: dict) -> None:
    require(development.get("status") == "PASS", "v41 development audit is not PASS")
    require(
        development["development_gate"]["decision"]
        == "freeze_cross_fitted_anchored_selector_for_external_confirmation",
        "v41 selector was not frozen for external confirmation",
    )
    require(
        heldout.get("protocol") == "tsc-v43r39-external-seed8171-heldout-evaluation-v1",
        "unexpected external held-out protocol",
    )
    require(heldout["offline_confirmation_gate"]["passed"], "external offline gate failed")
    require(set(heldout["city_results"]) == {"los_angeles", "jinan"}, "external city set changed")
    require(
        closed_loop.get("protocol") == "tsc-v43r39-external-closed-loop-aggregate-v4",
        "unexpected external closed-loop protocol",
    )
    require(closed_loop["validity_gate"]["passed"], "closed-loop validity gate failed")
    require(closed_loop["matrix"]["rollout_count"] == 56, "closed-loop rollout count changed")
    require(closed_loop["validity_gate"]["teleport_count"] == 0, "closed-loop teleports are nonzero")
    require(set(closed_loop["macro_summary"]) == set(POLICY_ORDER), "closed-loop policy set changed")


def validate_source_contribution(v90: dict, v91: dict) -> None:
    require(
        v90.get("protocol") == "tsc-v90r86-source-contribution-ablation-audit-v1"
        and v90.get("status") == "PASS"
        and v90.get("target_only_model_frozen_before_original_confirmation") is True
        and v90.get("original_confirmation_unchanged") is True,
        "v90 source-contribution ablation is not valid",
    )
    require(
        v91.get("protocol")
        == "tsc-v91r87-source-contribution-fresh-confirmation-audit-v1"
        and v91.get("status") == "PASS"
        and v91.get("integrity_gate", {}).get("passed") is True
        and v91.get("confirmation_gate", {}).get("passed") is False
        and v91.get("decision") == "reject_fresh_source_contribution_confirmation"
        and v91.get("confirmation_summary", {}).get("seed_count") == 64,
        "v91 fresh source-contribution decision changed",
    )


def method_freeze_rows(external_root: Path) -> dict[str, dict]:
    rows = {}
    for city in ("los_angeles", "jinan"):
        path = external_root / "full_freeze_v3" / city / "method_freeze.json"
        row = load_json(path)
        require(row["freeze_gate"]["passed"], f"{city} method freeze gate failed")
        require(row["evaluation_seed_not_loaded"], f"{city} loaded held-out seed before freeze")
        require(row["diagnostic_seed_not_loaded"], f"{city} loaded diagnostic seed during freeze")
        rows[city] = row
    return rows


def development_rows(development: dict) -> list[dict]:
    loco = development["development_gate"]["loco"]
    rows = []
    for row in loco["rows"]:
        rows.append(
            {
                "city": row["heldout_city"],
                "anchor_regret": float(row["anchor_regret"]),
                "selected_regret": float(row["selected_regret"]),
                "absolute_improvement": float(row["improvement"]),
                "relative_improvement": (
                    float(row["improvement"]) / float(row["anchor_regret"])
                    if float(row["anchor_regret"]) > 0.0
                    else 0.0
                ),
                "selected_selector": row["selected_selector"],
                "selected_non_anchor_targets": int(row["selected_non_anchor_target_count"]),
                "evaluation_seed": 4047,
            }
        )
    return rows


def offline_rows(heldout: dict, freezes: dict[str, dict]) -> list[dict]:
    rows = []
    for city in ("los_angeles", "jinan"):
        city_result = heldout["city_results"][city]
        anchor = city_result["equal_scenario_summary"][city_result["anchor_policy"]]
        selected = city_result["equal_scenario_summary"][city_result["selected_policy"]]
        inference = city_result["inference"]["selected_vs_anchor"]
        target_only = city_result["equal_scenario_summary"][
            "baseline/causal_target_only"
        ]
        target_only_inference = city_result["inference"]["selected_vs_baselines"][
            "causal_target_only"
        ]
        freeze = freezes[city]
        heldout_groups = int(sum(inference["group_counts"]))
        rows.append(
            {
                "city": city,
                "adaptation_groups": int(freeze["target_group_count"]),
                "heldout_groups": heldout_groups,
                "scenario_count": len(freeze["scenarios"]),
                "selected_candidate": city_result["selected_candidate"],
                "anchor_regret": float(anchor["mean_normalized_action_regret"]),
                "target_only_regret": float(
                    target_only["mean_normalized_action_regret"]
                ),
                "selected_regret": float(selected["mean_normalized_action_regret"]),
                "absolute_improvement": float(inference["observed_improvement"]),
                "relative_improvement": (
                    float(anchor["mean_normalized_action_regret"])
                    - float(selected["mean_normalized_action_regret"])
                )
                / float(anchor["mean_normalized_action_regret"]),
                "anchor_optimal_action_rate": float(anchor["optimal_action_rate"]),
                "target_only_optimal_action_rate": float(
                    target_only["optimal_action_rate"]
                ),
                "selected_optimal_action_rate": float(selected["optimal_action_rate"]),
                "ci95_low": float(inference["ci95"][0]),
                "ci95_high": float(inference["ci95"][1]),
                "bootstrap_p_two_sided": float(inference["two_sided_bootstrap_p"]),
                "source_absolute_improvement": float(
                    target_only_inference["observed_improvement"]
                ),
                "source_relative_improvement": float(
                    target_only["mean_normalized_action_regret"]
                    - selected["mean_normalized_action_regret"]
                )
                / float(target_only["mean_normalized_action_regret"]),
                "source_ci95_low": float(target_only_inference["ci95"][0]),
                "source_ci95_high": float(target_only_inference["ci95"][1]),
                "source_bootstrap_p_two_sided": float(
                    target_only_inference["two_sided_bootstrap_p"]
                ),
                "bootstrap_replicates": int(inference["replicates"]),
                "evaluation_seed": int(heldout["evaluation_seed"]),
            }
        )
    return rows


def source_contribution_rows(v90: dict, v91: dict) -> list[dict]:
    v90_method = float(v90["macro_summary"]["cfcmt_selected_mpc"])
    v90_target = float(v90["macro_summary"]["causal_target_only_mpc"])
    rows = [
        {
            "stage": "v90 post-freeze",
            "evidence_class": "post-hoc same-seed ablation",
            "seed_count": len(v90["matrix"]["seeds"]),
            "paired_cell_count": int(v90["matrix"]["paired_cells"]),
            "target_only_wait_sec": v90_target,
            "cfcmt_wait_sec": v90_method,
            "mean_relative_delta": (v90_method - v90_target) / v90_target,
            "los_angeles_relative_delta": (
                float(v90["city_summary"]["los_angeles"]["cfcmt_selected_mpc"])
                - float(v90["city_summary"]["los_angeles"]["causal_target_only_mpc"])
            )
            / float(v90["city_summary"]["los_angeles"]["causal_target_only_mpc"]),
            "jinan_relative_delta": (
                float(v90["city_summary"]["jinan"]["cfcmt_selected_mpc"])
                - float(v90["city_summary"]["jinan"]["causal_target_only_mpc"])
            )
            / float(v90["city_summary"]["jinan"]["causal_target_only_mpc"]),
            "ci95_low": "",
            "ci95_high": "",
            "improved_count": int(v90["improved_cell_count"]),
            "improved_denominator": int(v90["matrix"]["paired_cells"]),
            "cfcmt_collision_incidents": int(
                v90["safety"]["cfcmt_selected_mpc"]["collision_incidents"]
            ),
            "target_only_collision_incidents": int(
                v90["safety"]["causal_target_only_mpc"]["collision_incidents"]
            ),
            "decision": "descriptive only",
        }
    ]
    summary = v91["confirmation_summary"]
    rows.append(
        {
            "stage": "v91 fresh",
            "evidence_class": "preregistered fresh-seed confirmation",
            "seed_count": int(summary["seed_count"]),
            "paired_cell_count": int(summary["seed_count"]) * 4,
            "target_only_wait_sec": float(summary["target_only_mean_macro_waiting"]),
            "cfcmt_wait_sec": float(summary["method_mean_macro_waiting"]),
            "mean_relative_delta": float(summary["mean_relative_delta"]),
            "los_angeles_relative_delta": float(
                summary["city_mean_relative_deltas"]["los_angeles"]
            ),
            "jinan_relative_delta": float(
                summary["city_mean_relative_deltas"]["jinan"]
            ),
            "ci95_low": float(summary["bootstrap"]["ci95"][0]),
            "ci95_high": float(summary["bootstrap"]["ci95"][1]),
            "improved_count": int(summary["improved_seed_count"]),
            "improved_denominator": int(summary["seed_count"]),
            "cfcmt_collision_incidents": int(
                summary["collision_incidents"]["cfcmt_selected_mpc"]
            ),
            "target_only_collision_incidents": int(
                summary["collision_incidents"]["causal_target_only_mpc"]
            ),
            "decision": "confirmation rejected",
        }
    )
    return rows


def closed_loop_rows(closed_loop: dict) -> tuple[list[dict], list[dict], list[dict]]:
    macro_rows = []
    city_rows = []
    comparison_rows = []
    for policy in POLICY_ORDER:
        row = closed_loop["macro_summary"][policy]
        macro_rows.append(
            {
                "policy": policy,
                "label": POLICY_LABELS[policy],
                "all_departed_wait_sec": float(row["mean_tripinfo_waiting_time"]),
                "completed_wait_sec": float(row["mean_completed_tripinfo_waiting_time"]),
                "system_vehicle_hours": float(row["system_vehicle_hours"]),
                "unfinished_fraction": float(row["tripinfo_unfinished_fraction"]),
                "completion_ratio": float(row["completion_ratio"]),
                "throughput_ratio": float(row["throughput_ratio"]),
                "collision_incidents_city_macro": float(row["collision_incidents"]),
            }
        )
        for city in ("los_angeles", "jinan"):
            city_row = closed_loop["city_summary"][city][policy]
            city_rows.append(
                {
                    "city": city,
                    "policy": policy,
                    "label": POLICY_LABELS[policy],
                    "all_departed_wait_sec": float(city_row["mean_tripinfo_waiting_time"]),
                    "completed_wait_sec": float(city_row["mean_completed_tripinfo_waiting_time"]),
                    "system_vehicle_hours": float(city_row["system_vehicle_hours"]),
                    "unfinished_fraction": float(city_row["tripinfo_unfinished_fraction"]),
                    "completion_ratio": float(city_row["completion_ratio"]),
                    "collision_incidents": float(city_row["collision_incidents"]),
                }
            )
    for baseline in POLICY_ORDER:
        if baseline == "cfcmt_selected_mpc":
            continue
        row = closed_loop["method_comparisons"][baseline]["mean_tripinfo_waiting_time"]
        bootstrap = row["descriptive_bootstrap"]
        collision = closed_loop["collision_audit"]["method_vs_baselines"][baseline]
        comparison_rows.append(
            {
                "baseline": baseline,
                "label": POLICY_LABELS[baseline],
                "cfcmt_gain_sec": float(row["oriented_improvement"]),
                "relative_gain": float(row["relative_improvement"]),
                "ci95_low": float(bootstrap["improvement_95pct_ci"][0]),
                "ci95_high": float(bootstrap["improvement_95pct_ci"][1]),
                "bootstrap_probability_positive": float(bootstrap["bootstrap_probability_positive"]),
                "los_angeles_gain_sec": float(row["city_oriented_improvements"]["los_angeles"]),
                "jinan_gain_sec": float(row["city_oriented_improvements"]["jinan"]),
                "collision_all_paired_noninferior": bool(collision["all_paired_rollouts_noninferior"]),
                "independent_target_city_count": int(bootstrap["independent_target_city_count"]),
                "bootstrap_replicates": int(bootstrap["replicates"]),
            }
        )
    return macro_rows, city_rows, comparison_rows


def build_result_tables(
    development: dict,
    offline: list[dict],
    macro: list[dict],
    city_rows: list[dict],
    comparisons: list[dict],
    source_contribution: list[dict],
) -> None:
    loco = development["development_gate"]["loco"]
    target_only_macro = development_value(offline, "target_only_regret")
    selected_macro = development_value(offline, "selected_regret")
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"City & Adapt. groups & Held-out groups & Anchor & Target only & CFCMT & CFCMT vs target only & Selected blend \\",
        r"\midrule",
    ]
    for row in offline:
        city = "Los Angeles" if row["city"] == "los_angeles" else "Jinan"
        lines.append(
            f"{city} & {row['adaptation_groups']} & {row['heldout_groups']} & "
            f"{row['anchor_regret']:.3f} & {row['target_only_regret']:.3f} & "
            f"{row['selected_regret']:.3f} & "
            f"{100.0 * row['source_relative_improvement']:.2f}\\% & "
            f"{latex_escape(row['selected_candidate'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"External macro & -- & 2 cities & "
                f"{development_value(offline, 'anchor_regret'):.3f} & "
                f"{target_only_macro:.3f} & {selected_macro:.3f} & "
                f"{100.0 * (target_only_macro - selected_macro) / target_only_macro:.2f}\\% & "
                r"city-specific \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_text(TABLES / "external_offline_confirmation.tex", "\n".join(lines))

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & All-departed wait & Completed wait & System veh-h & Unfinished & Completion & Incidents \\",
        r"\midrule",
    ]
    for row in macro:
        label = r"\textbf{Anchored CFCMT}" if row["policy"] == "cfcmt_selected_mpc" else row["label"]
        lines.append(
            f"{label} & {row['all_departed_wait_sec']:.2f} & {row['completed_wait_sec']:.2f} & "
            f"{row['system_vehicle_hours']:.2f} & {row['unfinished_fraction']:.3f} & "
            f"{row['completion_ratio']:.3f} & {row['collision_incidents_city_macro']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLES / "external_closed_loop_macro.tex", "\n".join(lines))

    city_lookup = {
        (row["city"], row["policy"]): row
        for row in city_rows
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & LA wait & Jinan wait & LA completion & Jinan completion & LA incidents & Jinan incidents \\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        la = city_lookup[("los_angeles", policy)]
        jinan = city_lookup[("jinan", policy)]
        label = r"\textbf{Anchored CFCMT}" if policy == "cfcmt_selected_mpc" else POLICY_LABELS[policy]
        lines.append(
            f"{label} & {la['all_departed_wait_sec']:.2f} & {jinan['all_departed_wait_sec']:.2f} & "
            f"{la['completion_ratio']:.3f} & {jinan['completion_ratio']:.3f} & "
            f"{la['collision_incidents']:.2f} & {jinan['collision_incidents']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLES / "external_closed_loop_city.tex", "\n".join(lines))

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Baseline & CFCMT gain (s) & Relative gain & Descriptive 95\% interval & LA gain & Jinan gain \\",
        r"\midrule",
    ]
    for row in comparisons:
        lines.append(
            f"{row['label']} & {row['cfcmt_gain_sec']:.2f} & {100.0 * row['relative_gain']:.2f}\\% & "
            f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] & "
            f"{row['los_angeles_gain_sec']:.2f} & {row['jinan_gain_sec']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLES / "external_closed_loop_comparisons.tex", "\n".join(lines))

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Evidence & Seeds & Paired cells & Target-only wait & CFCMT wait & $\Delta$ wait & LA $\Delta$ & Jinan $\Delta$ & Improved & Incidents C/T \\",
        r"\midrule",
    ]
    for row in source_contribution:
        label = (
            "V91 fresh (rejected)"
            if row["stage"] == "v91 fresh"
            else "V90 post-hoc"
        )
        lines.append(
            f"{label} & {row['seed_count']} & {row['paired_cell_count']} & "
            f"{row['target_only_wait_sec']:.2f} & {row['cfcmt_wait_sec']:.2f} & "
            f"{100.0 * row['mean_relative_delta']:+.3f}\\% & "
            f"{100.0 * row['los_angeles_relative_delta']:+.3f}\\% & "
            f"{100.0 * row['jinan_relative_delta']:+.3f}\\% & "
            f"{row['improved_count']}/{row['improved_denominator']} & "
            f"{row['cfcmt_collision_incidents']}/{row['target_only_collision_incidents']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(
        TABLES / "external_source_contribution_closed_loop.tex",
        "\n".join(lines),
    )

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Development result & Anchor regret & Selected regret & Relative gain \\",
        r"\midrule",
        (
            r"Seven-city leave-one-city-out & "
            f"{loco['anchor_macro_regret']:.3f} & {loco['selected_macro_regret']:.3f} & "
            f"{100.0 * loco['macro_relative_improvement']:.2f}\\% \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_text(TABLES / "development_seed_heldout_confirmation.tex", "\n".join(lines))


def development_value(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def macro_relative(rows: list[dict]) -> float:
    anchor = development_value(rows, "anchor_regret")
    selected = development_value(rows, "selected_regret")
    return (anchor - selected) / anchor


def result_figure(
    development: dict,
    offline: list[dict],
    macro: list[dict],
    city: list[dict],
    comparisons: list[dict],
) -> None:
    fig = plt.figure(figsize=(7.25, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.35],
        height_ratios=[1.05, 0.95],
        hspace=0.72,
        wspace=0.62,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    loco = development["development_gate"]["loco"]
    pairs = [
        ("Development\n7 cities", loco["anchor_macro_regret"], loco["selected_macro_regret"]),
        ("Los Angeles", offline[0]["anchor_regret"], offline[0]["selected_regret"]),
        ("Jinan", offline[1]["anchor_regret"], offline[1]["selected_regret"]),
        (
            "External macro\n2 cities",
            development_value(offline, "anchor_regret"),
            development_value(offline, "selected_regret"),
        ),
    ]
    y = np.arange(len(pairs))[::-1]
    for position, (label, anchor, selected) in zip(y, pairs, strict=True):
        ax_a.plot([anchor, selected], [position, position], color=COLORS["grid"], lw=2.0, zorder=1)
        ax_a.scatter(anchor, position, s=25, color=COLORS["fixed"], edgecolor="white", linewidth=0.5, zorder=3)
        ax_a.scatter(selected, position, s=30, color=COLORS["cfcmt"], edgecolor="white", linewidth=0.5, zorder=3)
        gain = 100.0 * (anchor - selected) / anchor
        ax_a.text(max(anchor, selected) + 0.012, position, f"{gain:.1f}%", va="center", fontsize=6.2, color=COLORS["ink"])
    ax_a.set_yticks(y, [row[0] for row in pairs])
    ax_a.set_xlabel("Normalized action regret\n(lower is better)")
    ax_a.set_xlim(0.0, 0.52)
    ax_a.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax_a.tick_params(axis="y", length=0)
    ax_a.text(
        0.0,
        1.04,
        "rigid anchor",
        transform=ax_a.transAxes,
        color=COLORS["fixed"],
        fontsize=6.2,
        va="bottom",
    )
    ax_a.text(
        0.38,
        1.04,
        r"$\rightarrow$ selected CFCMT",
        transform=ax_a.transAxes,
        color=COLORS["cfcmt"],
        fontsize=6.2,
        va="bottom",
    )
    ax_a.set_title("a  Seed-held-out action regret", loc="left", fontweight="bold", fontsize=8.1, pad=30)

    macro_map = {row["policy"]: row for row in macro}
    city_map = {(row["city"], row["policy"]): row for row in city}
    policies = list(POLICY_ORDER)
    positions = np.arange(len(policies))[::-1]
    bar_colors = []
    for policy in policies:
        if policy == "cfcmt_selected_mpc":
            bar_colors.append(COLORS["cfcmt"])
        elif policy in {"phase_pressure", "max_pressure"}:
            bar_colors.append(COLORS["pressure"])
        elif policy == "fixed_time":
            bar_colors.append(COLORS["fixed"])
        else:
            bar_colors.append(COLORS["transfer"])
    values = [macro_map[policy]["all_departed_wait_sec"] for policy in policies]
    ax_b.barh(positions, values, height=0.62, color=bar_colors, alpha=0.92, zorder=2)
    for position, policy in zip(positions, policies, strict=True):
        la = city_map[("los_angeles", policy)]["all_departed_wait_sec"]
        jinan = city_map[("jinan", policy)]["all_departed_wait_sec"]
        ax_b.scatter(la, position, marker="o", s=17, facecolor="white", edgecolor=COLORS["ink"], linewidth=0.7, zorder=4)
        ax_b.scatter(jinan, position, marker="D", s=15, facecolor=COLORS["ink"], edgecolor="white", linewidth=0.4, zorder=4)
    figure_labels = {
        **POLICY_LABELS,
        "cfcmt_selected_mpc": "CFCMT",
        "h2oplus_style_dense_residual_mpc": "Dense residual",
    }
    ax_b.set_yticks(positions, [figure_labels[policy] for policy in policies])
    ax_b.set_xlabel("All-departed waiting time (s)\n3,600-s horizon; lower is better")
    ax_b.set_xlim(0.0, 555.0)
    ax_b.grid(axis="x", color=COLORS["grid"], lw=0.6, zorder=0)
    ax_b.tick_params(axis="y", length=0, labelsize=6.3)
    ax_b.text(
        0.99,
        1.04,
        "○ Los Angeles     ◆ Jinan",
        transform=ax_b.transAxes,
        color=COLORS["ink"],
        fontsize=6.2,
        ha="right",
        va="bottom",
    )
    ax_b.set_title("b  External closed-loop waiting", loc="left", fontweight="bold", fontsize=8.1, pad=30)

    comparison_order = (
        "h2oplus_style_dense_residual_mpc",
        "simulator_only_mpc",
        "rigid_anchor_mpc",
        "phase_pressure",
        "max_pressure",
    )
    comparison_map = {row["baseline"]: row for row in comparisons}
    positions = np.arange(len(comparison_order))[::-1]
    for position, baseline in zip(positions, comparison_order, strict=True):
        row = comparison_map[baseline]
        color = COLORS["pressure_dark"] if baseline in {"phase_pressure", "max_pressure"} else COLORS["cfcmt"]
        ax_c.plot([row["ci95_low"], row["ci95_high"]], [position, position], color=color, lw=1.5, solid_capstyle="round")
        ax_c.scatter(row["cfcmt_gain_sec"], position, s=25, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.axvline(0.0, color=COLORS["ink"], lw=0.8, ls=(0, (3, 2)))
    ax_c.set_yticks(positions, [figure_labels[key] for key in comparison_order])
    ax_c.set_xlabel("Paired CFCMT waiting-time gain (s)\npositive favors CFCMT")
    ax_c.set_xlim(-20.0, 30.0)
    ax_c.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax_c.tick_params(axis="y", length=0, labelsize=6.3)
    ax_c.set_title("c  Paired CFCMT policy contrasts", loc="left", fontweight="bold", fontsize=8.1, pad=10)
    ax_c.text(
        0.0,
        -0.31,
        "Fixed time is shown in panel b; intervals are descriptive for two target cities.",
        transform=ax_c.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        va="top",
    )

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])
    fig.subplots_adjust(left=0.15, right=0.985, top=0.93, bottom=0.12)
    export_figure(fig, HERE / "tsc_external_confirmation")
    plt.close(fig)


def resolve_sumocfg(manifest_path: Path, value: str) -> Path:
    if "${" in value:
        raise ValueError(f"unresolved environment placeholder in {value}")
    return (manifest_path.parent / value).resolve()


def net_path_from_sumocfg(sumocfg: Path) -> Path:
    root = ET.parse(sumocfg).getroot()
    item = root.find("./input/net-file")
    require(item is not None and bool(item.get("value")), f"SUMO config has no net-file: {sumocfg}")
    return (sumocfg.parent / str(item.get("value"))).resolve()


def parse_network(scenario: str, city: str, sumocfg: Path, role: str) -> dict:
    net_path = net_path_from_sumocfg(sumocfg)
    root = ET.parse(net_path).getroot()
    shapes: list[list[tuple[float, float]]] = []
    lane_count = 0
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or str(edge.get("id", "")).startswith(":"):
            continue
        lanes = edge.findall("lane")
        if not lanes:
            continue
        lane_count += len(lanes)
        shape = lanes[0].get("shape")
        if shape:
            points = []
            for token in shape.split():
                x, y = token.split(",")[:2]
                points.append((float(x), float(y)))
            if points:
                shapes.append(points)
    signals = []
    for junction in root.findall("junction"):
        if "traffic_light" in str(junction.get("type", "")):
            signals.append((float(junction.get("x", "0")), float(junction.get("y", "0"))))
    require(bool(shapes) and bool(signals), f"network has no plottable lanes or signals: {scenario}")
    return {
        "scenario": scenario,
        "city": city,
        "role": role,
        "sumocfg": str(sumocfg),
        "net": str(net_path),
        "lane_count": lane_count,
        "signal_count": len(signals),
        "shapes": shapes,
        "signals": signals,
    }


def network_rows(source_manifest: Path, external_conversion: Path) -> list[dict]:
    manifest = load_json(source_manifest)
    rows = [
        parse_network(
            str(entry["scenario"]),
            str(entry["city_group"]),
            resolve_sumocfg(source_manifest, str(entry["sumocfg"])),
            "source",
        )
        for entry in manifest["scenarios"]
    ]
    rows.append(
        parse_network(
            "la_1x4",
            "los_angeles",
            external_conversion / "la_1x4" / "la_1x4.sumocfg",
            "external target",
        )
    )
    rows.append(
        parse_network(
            "jinan_3x4 (3 demand files)",
            "jinan",
            external_conversion / "jinan_3x4_real" / "jinan_3x4_real.sumocfg",
            "external target",
        )
    )
    require(len(rows) == 20, "publication network plate requires 18 source and 2 external networks")
    return rows


def short_scenario(name: str) -> str:
    replacements = {
        "saltlake_400s_200w_q1_weekday_peak": "Salt Lake 400S-200W",
        "saltlake_state_university_q1_weekday_peak": "Salt Lake State Univ.",
        "hangzhou_4x4_hetero": "Hangzhou 4x4 hetero",
        "hangzhou_bc_tyc": "Hangzhou bc-tyc",
        "hangzhou_kn_hz": "Hangzhou kn-hz",
        "hangzhou_qc_yn": "Hangzhou qc-yn",
        "hangzhou_sb_sx": "Hangzhou sb-sx",
        "hangzhou_4x4": "Hangzhou 4x4",
        "manhattan_28x7": "Manhattan 28x7",
        "atlanta_1x5": "Atlanta 1x5",
        "la_1x4": "Los Angeles 1x4",
        "jinan_3x4 (3 demand files)": "Jinan 3x4 (3 demands)",
    }
    return replacements.get(name, name)


def network_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(7.25, 8.25), facecolor="white")
    letters = "abcdefghijklmnopqrst"
    for index, (ax, row) in enumerate(zip(axes.flat, rows, strict=True)):
        target = row["role"] == "external target"
        road = COLORS["target_road"] if target else COLORS["source_road"]
        signal = COLORS["target_tls"] if target else COLORS["source_tls"]
        for shape in row["shapes"]:
            values = np.asarray(shape, dtype=float)
            ax.plot(values[:, 0], values[:, 1], color=road, lw=0.55, solid_capstyle="round", zorder=1)
        signals = np.asarray(row["signals"], dtype=float)
        marker_size = 9 if len(signals) < 40 else 4
        ax.scatter(signals[:, 0], signals[:, 1], s=marker_size, color=signal, edgecolor="white", linewidth=0.25, zorder=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.08)
        ax.set_axis_off()
        title_color = COLORS["cfcmt"] if target else COLORS["transfer_dark"]
        ax.text(0.0, 1.10, f"{letters[index]}  {short_scenario(row['scenario'])}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color=title_color, fontweight="bold")
        role = "target" if target else row["city"].replace("_", " ")
        ax.text(0.0, 1.01, f"{role} | {row['signal_count']} TLS | {row['lane_count']} lanes", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=COLORS["muted"])
    fig.suptitle(
        "Cross-network evidence spans 18 source networks and two external target cities",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.965,
        "Blue signals denote source networks; green signals denote the Los Angeles and Jinan confirmation targets.",
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.925, bottom=0.025, hspace=0.64, wspace=0.24)
    export_figure(fig, HERE / "traffic_signal_transfer_networks")
    plt.close(fig)


def build_network_table(rows: list[dict]) -> None:
    source_rows = [row for row in rows if row["role"] == "source"]
    city_groups: dict[str, list[dict]] = {}
    for row in source_rows:
        city_groups.setdefault(row["city"], []).append(row)
    external_rows = [row for row in rows if row["role"] == "external target"]
    table_rows = []
    for city_name in sorted(city_groups):
        entries = city_groups[city_name]
        table_rows.append(
            {
                "role": "Source",
                "city": city_name,
                "networks": len(entries),
                "signals": sum(int(row["signal_count"]) for row in entries),
                "lanes": sum(int(row["lane_count"]) for row in entries),
                "scenarios": ", ".join(str(row["scenario"]) for row in entries),
            }
        )
    for row in external_rows:
        table_rows.append(
            {
                "role": "External target",
                "city": row["city"],
                "networks": 1,
                "signals": int(row["signal_count"]),
                "lanes": int(row["lane_count"]),
                "scenarios": str(row["scenario"]),
            }
        )
    write_csv(SOURCE_DATA / "traffic_signal_transfer_networks.csv", table_rows)
    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Role & City group & Networks & TLS & Lanes & Scenarios \\",
        r"\midrule",
    ]
    for row in table_rows:
        lines.append(
            f"{row['role']} & {latex_escape(row['city'])} & {row['networks']} & "
            f"{row['signals']} & {row['lanes']} & {latex_escape(row['scenarios'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLES / "traffic_signal_transfer_networks.tex", "\n".join(lines))


def qa_manifest(
    development: dict,
    heldout: dict,
    closed_loop: dict,
    joint_audit: dict,
    source_ablation: dict,
    source_confirmation: dict,
    freezes: dict[str, dict],
    *,
    development_path: Path,
    heldout_path: Path,
    closed_loop_path: Path,
    joint_audit_path: Path,
    source_ablation_path: Path,
    source_confirmation_path: Path,
) -> None:
    provenance = {
        "development audit": sha256(development_path),
        "source cache": joint_audit["source_cache_sha256"],
        "target adaptation cache": joint_audit["adaptation_cache_sha256"],
        "joint method freeze": sha256(joint_audit_path),
        "held-out seed-8171 result": sha256(heldout_path),
        "closed-loop aggregate": sha256(closed_loop_path),
        "post-freeze target-only ablation": sha256(source_ablation_path),
        "fresh target-only confirmation": sha256(source_confirmation_path),
        "Los Angeles deployment model": freezes["los_angeles"]["model_artifact"]["sha256"],
        "Jinan deployment model": freezes["jinan"]["model_artifact"]["sha256"],
        "closed-loop source tree": closed_loop["provenance"]["source_tree_sha256"],
    }
    payload = {
        "protocol": "paper-tsc-external-confirmation-artifacts-v1",
        "inputs": {
            "development_protocol": development.get("audit"),
            "heldout_protocol": heldout.get("protocol"),
            "closed_loop_protocol": closed_loop.get("protocol"),
            "source_ablation_protocol": source_ablation.get("protocol"),
            "source_confirmation_protocol": source_confirmation.get("protocol"),
            "los_angeles_method_protocol": freezes["los_angeles"].get("protocol"),
            "jinan_method_protocol": freezes["jinan"].get("protocol"),
        },
        "provenance": {
            key.replace(" ", "_"): value
            for key, value in provenance.items()
        },
        "hard_gates": {
            "development_passed": development.get("status") == "PASS",
            "external_offline_passed": heldout["offline_confirmation_gate"]["passed"],
            "closed_loop_passed": closed_loop["validity_gate"]["passed"],
            "rollouts": closed_loop["matrix"]["rollout_count"],
            "teleports": closed_loop["validity_gate"]["teleport_count"],
            "external_target_cities": len(heldout["city_results"]),
            "source_ablation_integrity_passed": source_ablation.get("status") == "PASS",
            "source_confirmation_integrity_passed": source_confirmation.get("integrity_gate", {}).get("passed") is True,
            "source_confirmation_decision": source_confirmation.get("decision"),
        },
        "statistical_boundary": closed_loop["estimands"]["bootstrap_scope"],
        "metric_population": closed_loop["estimands"]["implemented_primary_population"],
        "outputs": {
            "figure": "figures/tsc_external_confirmation.{pdf,svg,png,tiff}",
            "network_figure": "figures/traffic_signal_transfer_networks.{pdf,svg,png,tiff}",
            "source_data": [
                "source_data/development_seed_heldout_action_regret.csv",
                "source_data/external_offline_confirmation.csv",
                "source_data/external_closed_loop_macro.csv",
                "source_data/external_closed_loop_city.csv",
                "source_data/external_closed_loop_comparisons.csv",
                "source_data/external_source_contribution_closed_loop.csv",
                "source_data/traffic_signal_transfer_networks.csv",
            ],
            "tables": [
                "tables/development_seed_heldout_confirmation.tex",
                "tables/external_offline_confirmation.tex",
                "tables/external_closed_loop_macro.tex",
                "tables/external_closed_loop_city.tex",
                "tables/external_closed_loop_comparisons.tex",
                "tables/external_source_contribution_closed_loop.tex",
                "tables/traffic_signal_transfer_networks.tex",
                "tables/external_confirmation_provenance.tex",
            ],
        },
    }
    lines = [
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Artifact & SHA-256 \\",
        r"\midrule",
    ]
    for label, digest in provenance.items():
        lines.append(
            f"{latex_escape(label)} & \\texttt{{\\detokenize{{{digest}}}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLES / "external_confirmation_provenance.tex", "\n".join(lines))
    write_text(SOURCE_DATA / "external_confirmation_artifact_manifest.json", json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--closed-loop", type=Path, default=DEFAULT_CLOSED_LOOP)
    parser.add_argument("--joint-audit", type=Path, default=DEFAULT_JOINT_AUDIT)
    parser.add_argument(
        "--source-ablation", type=Path, default=DEFAULT_SOURCE_ABLATION
    )
    parser.add_argument(
        "--source-confirmation", type=Path, default=DEFAULT_SOURCE_CONFIRMATION
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--external-conversion", type=Path, default=DEFAULT_EXTERNAL_CONVERSION)
    args = parser.parse_args()

    development = load_json(args.development)
    heldout = load_json(args.heldout)
    closed_loop = load_json(args.closed_loop)
    joint_audit = load_json(args.joint_audit)
    source_ablation = load_json(args.source_ablation)
    source_confirmation = load_json(args.source_confirmation)
    validate_results(development, heldout, closed_loop)
    validate_source_contribution(source_ablation, source_confirmation)
    require(
        joint_audit.get("protocol")
        == "tsc-v43r39-external-full-budget-joint-freeze-audit-v1"
        and joint_audit.get("status") == "PASS"
        and joint_audit["gate"]["passed"],
        "joint external method freeze is not valid",
    )
    freezes = method_freeze_rows(args.external_root)
    development_source = development_rows(development)
    offline_source = offline_rows(heldout, freezes)
    macro_source, city_source, comparison_source = closed_loop_rows(closed_loop)
    source_contribution = source_contribution_rows(
        source_ablation, source_confirmation
    )
    networks = network_rows(args.source_manifest, args.external_conversion)

    write_csv(SOURCE_DATA / "development_seed_heldout_action_regret.csv", development_source)
    write_csv(SOURCE_DATA / "external_offline_confirmation.csv", offline_source)
    write_csv(SOURCE_DATA / "external_closed_loop_macro.csv", macro_source)
    write_csv(SOURCE_DATA / "external_closed_loop_city.csv", city_source)
    write_csv(SOURCE_DATA / "external_closed_loop_comparisons.csv", comparison_source)
    write_csv(
        SOURCE_DATA / "external_source_contribution_closed_loop.csv",
        source_contribution,
    )
    build_result_tables(
        development,
        offline_source,
        macro_source,
        city_source,
        comparison_source,
        source_contribution,
    )
    build_network_table(networks)
    result_figure(development, offline_source, macro_source, city_source, comparison_source)
    network_figure(networks)
    qa_manifest(
        development,
        heldout,
        closed_loop,
        joint_audit,
        source_ablation,
        source_confirmation,
        freezes,
        development_path=args.development,
        heldout_path=args.heldout,
        closed_loop_path=args.closed_loop,
        joint_audit_path=args.joint_audit,
        source_ablation_path=args.source_ablation,
        source_confirmation_path=args.source_confirmation,
    )
    print("publication artifacts generated and protocol gates verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
