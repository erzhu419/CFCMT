"""Generate paper tables and figures from validation result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _q(summary: dict[str, Any], block: str, key: str) -> float | None:
    value = summary.get(block, {}).get(key)
    return float(value) if value is not None else None


def _maybe_round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return round(value, digits)
    return value


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda value: _maybe_round(value))
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    display = _display_frame(df)
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _latex_table(df: pd.DataFrame) -> str:
    display = _display_frame(df)
    cols = "l" * len(display.columns)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(col) for col in display.columns) + r" \\")
    lines.append(r"\midrule")
    for row in display.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    display = _display_frame(df)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    md_path = out_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(_latex_table(display), encoding="utf-8")
    md_path.write_text(_markdown_table(display), encoding="utf-8")
    return {"csv": str(csv_path), "tex": str(tex_path), "md": str(md_path)}


DATA_EVIDENCE = {
    "singapore_lta_all": {
        "demand_evidence": "observed stop OD/PV monthly",
        "traffic_evidence": "observed speed bands",
    },
    "austin_capmetro_all": {
        "demand_evidence": "schedule proxy",
        "traffic_evidence": "schedule derived",
    },
    "halifax_transit_all": {
        "demand_evidence": "route APC apportioned",
        "traffic_evidence": "schedule derived",
    },
    "mbta_all": {
        "demand_evidence": "stop board/alight apportioned",
        "traffic_evidence": "schedule derived",
    },
}


def build_dataset_table(suite: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    sanity = suite["experiments"]["data_sanity"]["cities"]
    transitions = suite["experiments"]["efficiency"]["city_transitions"]
    rows = []
    for key, spec in config["generated_envs"].items():
        city = sanity[key]
        rows.append(
            {
                "city": spec.get("city", key),
                "env_key": key,
                "demand_evidence": DATA_EVIDENCE.get(key, {}).get("demand_evidence", "unknown"),
                "traffic_evidence": DATA_EVIDENCE.get(key, {}).get("traffic_evidence", "unknown"),
                "lines": int(city.get("line_count", spec.get("line_count", 0))),
                "stops": int(city.get("stop_count", spec.get("stations", 0))),
                "segments": int(city.get("segment_count", spec.get("segments", 0))),
                "timetables": int(city.get("timetable_count", spec.get("timetables", 0))),
                "transitions": int(transitions.get(key, 0)),
                "median_route_km": (_q(city, "route_distance_m_quantiles", "p50") or 0.0) / 1000.0,
                "median_speed_mps": _q(city, "speed_mps_quantiles", "p50"),
                "median_headway_s": _q(city, "headway_s_quantiles", "p50"),
                "peak_demand_hour": city.get("peak_demand_hour"),
                "demand_source": spec.get("demand_source", ""),
                "traffic_source": spec.get("traffic_source", ""),
            }
        )
    return pd.DataFrame(rows)


def build_strict_leave_one_out_table(suite: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in suite["experiments"]["strict_leave_one_city_out"]["splits"]:
        h2o = split["metrics"]["h2oplus_dense"]["total_mse"]
        cfcmt = split["metrics"]["cfcmt_mechanism"]["total_mse"]
        weighted = split["metrics"]["cfcmt_similarity_weighted"]["total_mse"]
        rows.append(
            {
                "target_city": split["target_city"],
                "source_cities": ", ".join(split["source_envs"]),
                "target_transitions": int(split["target_transitions"]),
                "h2oplus_total_mse": h2o,
                "cfcmt_total_mse": cfcmt,
                "weighted_cfcmt_total_mse": weighted,
                "cfcmt_vs_h2oplus_ratio": cfcmt / h2o if h2o else None,
                "weighted_cfcmt_vs_h2oplus_ratio": split["comparisons"][
                    "cfcmt_similarity_weighted_vs_h2oplus_ratio"
                ],
                "weighted_cfcmt_vs_unweighted_ratio": split["comparisons"][
                    "cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_cross_city_dynamics_table(performance: dict[str, Any], suite: dict[str, Any]) -> pd.DataFrame:
    weighted = {
        row["name"]: row
        for row in suite["experiments"]["source_similarity_weighting"]["splits"]
    }
    rows = []
    for split in performance["splits"]:
        comparisons = split["comparisons"]
        weighted_row = weighted.get(split["name"])
        rows.append(
            {
                "split": split["name"],
                "target_city": split["metrics"]["target_city"],
                "train_transitions": int(split["train_transitions"]),
                "target_transitions": int(split["target_transitions"]),
                "h2oplus_total_mse": split["metrics"]["h2oplus_dense"]["total_mse"],
                "cfcmt_total_mse": split["metrics"]["cfcmt_mechanism"]["total_mse"],
                "cfcmt_vs_h2oplus_ratio": comparisons["cfcmt_vs_h2oplus_total_mse_ratio"],
                "weighted_cfcmt_vs_h2oplus_ratio": (
                    weighted_row["comparisons"]["cfcmt_similarity_weighted_vs_h2oplus_ratio"]
                    if weighted_row
                    else None
                ),
                "weighted_cfcmt_vs_unweighted_ratio": (
                    weighted_row["comparisons"]["cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"]
                    if weighted_row
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cross_city_policy_table(policy: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in [item for item in policy["splits"] if item["name"].startswith("leave_one_city_out_all::")]:
        h2o = split["methods"]["h2oplus_dense_policy"]
        cfcmt = split["methods"]["cfcmt_mechanism_policy"]
        comparisons = split["comparisons"]
        rows.append(
            {
                "split": split["name"],
                "target_city": split["target_city"],
                "h2oplus_reward": h2o["mean_reward"],
                "cfcmt_reward": cfcmt["mean_reward"],
                "cfcmt_reward_gain": comparisons["cfcmt_reward_gain_vs_h2oplus"],
                "cfcmt_regret_ratio": comparisons["cfcmt_regret_ratio_vs_h2oplus"],
                "cfcmt_headway_error_ratio": comparisons["cfcmt_headway_error_ratio_vs_h2oplus"],
                "h2oplus_bunching_rate": h2o.get("bunching_rate"),
                "cfcmt_bunching_rate": cfcmt.get("bunching_rate"),
                "cfcmt_bunching_rate_ratio": comparisons.get("cfcmt_bunching_rate_ratio_vs_h2oplus"),
                "h2oplus_large_gap_rate": h2o.get("large_gap_rate"),
                "cfcmt_large_gap_rate": cfcmt.get("large_gap_rate"),
                "cfcmt_large_gap_rate_ratio": comparisons.get("cfcmt_large_gap_rate_ratio_vs_h2oplus"),
                "cfcmt_hold_seconds_delta": comparisons.get("cfcmt_mean_hold_seconds_delta_vs_h2oplus"),
            }
        )
    return pd.DataFrame(rows)


def build_source_subset_robustness_table(suite: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for subset in suite["experiments"]["source_subset_robustness"]["subsets"]:
        summary = subset["summary"]
        rows.append(
            {
                "subset": subset["name"],
                "cities": ", ".join(subset["cities"]),
                "splits": summary["splits"],
                "weighted_wins_vs_h2oplus": summary["cfcmt_similarity_weighted_wins_vs_h2oplus"],
                "unweighted_wins_vs_h2oplus": summary["cfcmt_wins_vs_h2oplus"],
                "mean_unweighted_ratio_vs_h2oplus": summary["mean_cfcmt_vs_h2oplus_ratio"],
                "mean_weighted_ratio_vs_h2oplus": summary["mean_cfcmt_similarity_weighted_vs_h2oplus_ratio"],
                "mean_weighted_ratio_vs_unweighted": summary[
                    "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_generator_robustness_table(suite: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for scenario, summary in suite["experiments"]["generator_robustness"]["summary_by_scenario"].items():
        rows.append(
            {
                "scenario": scenario,
                "splits": summary["splits"],
                "weighted_wins_vs_h2oplus": summary["cfcmt_similarity_weighted_wins_vs_h2oplus"],
                "unweighted_wins_vs_h2oplus": summary["cfcmt_wins_vs_h2oplus"],
                "weighted_wins_vs_unweighted": summary["cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt"],
                "mean_unweighted_ratio_vs_h2oplus": summary["mean_cfcmt_vs_h2oplus_ratio"],
                "mean_weighted_ratio_vs_h2oplus": summary["mean_cfcmt_similarity_weighted_vs_h2oplus_ratio"],
                "mean_weighted_ratio_vs_unweighted": summary[
                    "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_experiment_scope_table(suite: dict[str, Any]) -> pd.DataFrame:
    rollout = suite["experiments"].get("sampled_rollout", {})
    rollout_episodes = sum(
        int(value.get("episodes") or 0)
        for value in rollout.get("summary_by_policy", {}).values()
    )
    rows = [
        {
            "experiment": "strict_leave_one_city_out",
            "paper_role": "primary cross-city dynamics evidence",
            "unit": "city target",
            "scope_note": "four unique targets; no duplicated target counted as independent",
        },
        {
            "experiment": "generator_robustness",
            "paper_role": "bias/noise defense",
            "unit": "scenario x city target",
            "scope_note": "static-derived target perturbation; not real AVL/APC trajectory ground truth",
        },
        {
            "experiment": "source_subset_robustness",
            "paper_role": "data-source confound defense",
            "unit": "city subset",
            "scope_note": "reports excluding Singapore and excluding Austin proxy-demand subsets",
        },
        {
            "experiment": "cross_city_policy_validation",
            "paper_role": "primary one-step policy evidence",
            "unit": "city target",
            "scope_note": "one-step lookahead, not live SUMO rollout",
        },
        {
            "experiment": "sampled_rollout",
            "paper_role": "auxiliary sanity check only",
            "unit": "BusSimEnv episode",
            "scope_note": f"{rollout_episodes} policy episodes across runnable line envs; do not use as primary claim",
        },
    ]
    return pd.DataFrame(rows)


def build_source_weighting_sensitivity_table(suite: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in suite["experiments"]["source_weighting_sensitivity"]["grid"]:
        rows.append(
            {
                "temperature": row["temperature"],
                "floor": row["floor"],
                "wins_vs_h2oplus": row["cfcmt_similarity_weighted_wins_vs_h2oplus"],
                "wins_vs_unweighted_cfcmt": row["cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt"],
                "mean_ratio_vs_h2oplus": row["mean_cfcmt_similarity_weighted_vs_h2oplus_ratio"],
                "mean_ratio_vs_unweighted_cfcmt": row[
                    "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_sampled_rollout_table(suite: dict[str, Any]) -> pd.DataFrame:
    rollout = suite["experiments"].get("sampled_rollout", {})
    rows = []
    for policy, summary in rollout.get("summary_by_policy", {}).items():
        rows.append(
            {
                "policy": policy,
                "episodes": summary["episodes"],
                "mean_reward": summary["mean_reward"],
                "mean_headway_abs_error": summary["mean_headway_abs_error"],
                "mean_hold_seconds": summary["mean_hold_seconds"],
            }
        )
    return pd.DataFrame(rows)


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def plot_cross_city_dynamics(df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.target_city}\n{idx + 1}" for idx, row in enumerate(df.itertuples())]
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - width / 2, df["cfcmt_vs_h2oplus_ratio"], width, label="CFCMT", color="#2f6f9f")
    ax.bar(
        x + width / 2,
        df["weighted_cfcmt_vs_h2oplus_ratio"],
        width,
        label="CFCMT + source weighting",
        color="#c05a2b",
    )
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", label="H2O+ baseline")
    ax.set_ylabel("Total MSE ratio vs H2O+")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.08, float(np.nanmax(df["cfcmt_vs_h2oplus_ratio"])) + 0.08))
    ax.set_title("Cross-city dynamics validation")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "cross_city_total_mse_ratio.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_strict_leave_one_out(df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(df["target_city"])
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - width / 2, df["cfcmt_vs_h2oplus_ratio"], width, label="CFCMT", color="#2f6f9f")
    ax.bar(
        x + width / 2,
        df["weighted_cfcmt_vs_h2oplus_ratio"],
        width,
        label="CFCMT + source weighting",
        color="#c05a2b",
    )
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", label="H2O+ baseline")
    ax.set_ylabel("Total MSE ratio vs H2O+")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.08, float(np.nanmax(df["cfcmt_vs_h2oplus_ratio"])) + 0.08))
    ax.set_title("Strict leave-one-city-out dynamics")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "strict_leave_one_out_total_mse_ratio.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_source_subset_robustness(df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [value.replace("_", "\n") for value in df["subset"]]
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - width / 2, df["mean_unweighted_ratio_vs_h2oplus"], width, label="CFCMT", color="#2f6f9f")
    ax.bar(x + width / 2, df["mean_weighted_ratio_vs_h2oplus"], width, label="Weighted CFCMT", color="#c05a2b")
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", label="H2O+ baseline")
    ax.set_ylabel("Mean MSE ratio vs H2O+")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.08, float(np.nanmax(df["mean_unweighted_ratio_vs_h2oplus"])) + 0.08))
    ax.set_title("Source subset robustness")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "source_subset_robustness.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_generator_robustness(df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [value.replace("_", "\n") for value in df["scenario"]]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.plot(x, df["mean_unweighted_ratio_vs_h2oplus"], marker="o", linewidth=1.8, label="CFCMT", color="#2f6f9f")
    ax.plot(x, df["mean_weighted_ratio_vs_h2oplus"], marker="o", linewidth=1.8, label="Weighted CFCMT", color="#c05a2b")
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", label="H2O+ baseline")
    ax.set_ylabel("Mean MSE ratio vs H2O+")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.08, float(np.nanmax(df["mean_unweighted_ratio_vs_h2oplus"])) + 0.08))
    ax.set_title("Generator bias and noise robustness")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "generator_robustness.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_source_weighting_heatmap(df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="floor", columns="temperature", values="mean_ratio_vs_h2oplus").sort_index()
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(values, cmap="YlGnBu_r", vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Floor")
    ax.set_title("Source weighting sensitivity")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", color="#111111", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean MSE ratio vs H2O+")
    fig.tight_layout()
    path = out_dir / "source_weighting_sensitivity_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_policy_regret(policy_df: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.target_city}\n{idx + 1}" for idx, row in enumerate(policy_df.itertuples())]
    x = np.arange(len(policy_df))
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar(x, policy_df["cfcmt_regret_ratio"], color="#52796f", label="Regret ratio")
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--", label="H2O+ baseline")
    ax.set_ylabel("CFCMT / H2O+ regret")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.08, float(np.nanmax(policy_df["cfcmt_regret_ratio"])) + 0.08))
    ax.set_title("Cross-city policy validation")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "cross_city_policy_regret_ratio.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_sampled_rollout(rollout_df: pd.DataFrame, out_dir: Path) -> str | None:
    if rollout_df.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [value.replace("_", "\n") for value in rollout_df["policy"]]
    x = np.arange(len(rollout_df))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x, rollout_df["mean_reward"], color="#9a6b22")
    ax.set_ylabel("Mean reward")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Sampled executable rollout")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = out_dir / "sampled_rollout_mean_reward.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    suite = _read_json(args.suite)
    performance = _read_json(args.performance)
    policy = _read_json(args.policy)
    config = _read_json(Path(suite["config"]))

    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    _figure_style()

    dataset_df = build_dataset_table(suite, config)
    strict_loo_df = build_strict_leave_one_out_table(suite)
    dynamics_df = build_cross_city_dynamics_table(performance, suite)
    policy_df = build_cross_city_policy_table(policy)
    sensitivity_df = build_source_weighting_sensitivity_table(suite)
    subset_df = build_source_subset_robustness_table(suite)
    generator_df = build_generator_robustness_table(suite)
    rollout_df = build_sampled_rollout_table(suite)
    scope_df = build_experiment_scope_table(suite)

    tables = {
        "dataset": _write_table(dataset_df, tables_dir, "dataset_table"),
        "strict_leave_one_out": _write_table(strict_loo_df, tables_dir, "strict_leave_one_out_table"),
        "cross_city_dynamics": _write_table(dynamics_df, tables_dir, "cross_city_dynamics_table"),
        "cross_city_policy": _write_table(policy_df, tables_dir, "cross_city_policy_table"),
        "source_weighting_sensitivity": _write_table(
            sensitivity_df,
            tables_dir,
            "source_weighting_sensitivity_table",
        ),
        "source_subset_robustness": _write_table(subset_df, tables_dir, "source_subset_robustness_table"),
        "generator_robustness": _write_table(generator_df, tables_dir, "generator_robustness_table"),
        "sampled_rollout": _write_table(rollout_df, tables_dir, "sampled_rollout_table"),
        "experiment_scope": _write_table(scope_df, tables_dir, "experiment_scope_table"),
    }

    figures = {
        "strict_leave_one_out_total_mse_ratio": plot_strict_leave_one_out(strict_loo_df, figures_dir),
        "cross_city_total_mse_ratio": plot_cross_city_dynamics(dynamics_df, figures_dir),
        "source_weighting_sensitivity_heatmap": plot_source_weighting_heatmap(sensitivity_df, figures_dir),
        "source_subset_robustness": plot_source_subset_robustness(subset_df, figures_dir),
        "generator_robustness": plot_generator_robustness(generator_df, figures_dir),
        "cross_city_policy_regret_ratio": plot_policy_regret(policy_df, figures_dir),
    }
    sampled_path = plot_sampled_rollout(rollout_df, figures_dir)
    if sampled_path:
        figures["sampled_rollout_mean_reward"] = sampled_path

    manifest = {
        "ok": True,
        "sources": {
            "suite": str(args.suite),
            "performance": str(args.performance),
            "policy": str(args.policy),
            "config": suite["config"],
        },
        "tables": tables,
        "figures": figures,
        "summaries": {
            "cross_city_performance": performance["summary"],
            "cross_city_policy": policy["summary"],
            "strict_cross_city_policy": policy.get("strict_leave_one_city_out_summary"),
            "strict_leave_one_out": suite["experiments"]["strict_leave_one_city_out"]["summary"],
            "source_similarity_weighting": suite["experiments"]["source_similarity_weighting"]["summary"],
            "source_weighting_sensitivity": suite["experiments"]["source_weighting_sensitivity"]["summary"],
            "source_subset_robustness": suite["experiments"]["source_subset_robustness"]["summary"],
            "generator_robustness": suite["experiments"]["generator_robustness"]["summary"],
            "sampled_rollout": suite["experiments"].get("sampled_rollout", {}).get("summary_by_policy"),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("cf_h2o/results/paper_experiment_suite.json"))
    parser.add_argument("--performance", type=Path, default=Path("cf_h2o/results/cross_city_performance_validation.json"))
    parser.add_argument("--policy", type=Path, default=Path("cf_h2o/results/cross_city_policy_validation.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("cf_h2o/results/paper_artifacts"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = generate(parse_args(argv))
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
