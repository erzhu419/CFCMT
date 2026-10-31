from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / "cf_h2o/results/traffic_signal_resco_cfcmt_extended_multiseed_validation.json"
OUT_FIG = HERE / "traffic_signal_resco_extended_summary.png"
OUT_CSV = ROOT / "cf_h2o/results/paper_artifacts/tables/traffic_signal_resco_extended_summary.csv"
OUT_MD = ROOT / "cf_h2o/results/paper_artifacts/tables/traffic_signal_resco_extended_summary.md"
OUT_TEX = ROOT / "cf_h2o/results/paper_artifacts/tables/traffic_signal_resco_extended_summary.tex"

POLICY_LABELS = {
    "cfcmt_phase_pressure_guard_mpc": "CFCMT + pressure guard",
    "max_pressure": "MaxPressure",
    "phase_pressure": "Phase pressure",
    "actuated_program": "Native actuated SUMO",
    "h2oplus_dense_phase_mpc": "H2O+-style dense",
    "sim_target_static_phase_mpc": "Target-static simulator",
    "fixed_program": "Fixed RESCO program",
}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _finite_or_fallback(primary: object, fallback: object = float("nan")) -> float:
    primary_value = float(primary)
    if np.isfinite(primary_value):
        return primary_value
    return float(fallback)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _latex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Policy & Mean queue & Seed std & P90 queue & Trip time & Trip wait & Time loss & Throughput \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['policy'])} & "
            f"{_fmt(float(row['mean_queue']))} & "
            f"{_fmt(float(row['seed_std']))} & "
            f"{_fmt(float(row['p90_queue']))} & "
            f"{_fmt(float(row['trip_time']))} & "
            f"{_fmt(float(row['trip_wait']))} & "
            f"{_fmt(float(row['time_loss']))} & "
            f"{_fmt(float(row['throughput']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_summary(
    *,
    result_path: Path,
    out_fig: Path,
    out_csv: Path,
    out_md: Path,
    out_tex: Path,
    title: str,
) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for policy, label in POLICY_LABELS.items():
        if policy not in result["aggregate"]:
            continue
        metrics = result["aggregate"][policy]
        completed_trip_time = float(metrics.get("mean_completed_travel_time", float("nan")))
        completed_trip_wait = float(metrics.get("mean_completed_waiting_time", float("nan")))
        rows.append(
            {
                "policy": label,
                "mean_queue": float(metrics["mean_queue"]),
                "seed_std": float(metrics.get("seed_std_mean_queue", 0.0)),
                "p90_queue": float(metrics["p90_queue"]),
                "active_wait": float(metrics.get("mean_vehicle_waiting_time", float("nan"))),
                "trip_time": _finite_or_fallback(metrics.get("mean_tripinfo_duration", float("nan")), completed_trip_time),
                "trip_wait": _finite_or_fallback(metrics.get("mean_tripinfo_waiting_time", float("nan")), completed_trip_wait),
                "time_loss": float(metrics.get("mean_tripinfo_time_loss", float("nan"))),
                "depart_delay": float(metrics.get("mean_tripinfo_depart_delay", float("nan"))),
                "completed_trips": float(metrics.get("completed_trips", float("nan"))),
                "tripinfo_count": float(metrics.get("tripinfo_count", float("nan"))),
                "throughput": float(metrics["throughput_ratio"]),
            }
        )
    rows.sort(key=lambda item: float(item["mean_queue"]))

    labels = [str(row["policy"]) for row in rows]
    means = np.asarray([float(row["mean_queue"]) for row in rows], dtype=float)
    stds = np.asarray([float(row["seed_std"]) for row in rows], dtype=float)
    colors = ["#2f6f9f" if "CFCMT" in label else "#6f7f8f" for label in labels]
    colors = ["#2f6f9f" if label.startswith("CFCMT") else "#c05a2b" if label == "MaxPressure" else "#7f8a96" for label in labels]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="#26313a", linewidth=0.6, capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean queue across extended RESCO scenarios")
    ax.set_title(title)
    ax.grid(axis="x", color="#d7dde2", linewidth=0.8)
    ax.set_axisbelow(True)
    for idx, value in enumerate(means):
        ax.text(value + stds[idx] + 0.35, idx, f"{value:.2f}", va="center", ha="left", fontsize=8.5, color="#26313a")
    ax.margins(x=0.14)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "mean_queue",
                "seed_std",
                "p90_queue",
                "active_wait",
                "trip_time",
                "trip_wait",
                "time_loss",
                "depart_delay",
                "completed_trips",
                "tripinfo_count",
                "throughput",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_rows = [
        {
            "policy": row["policy"],
            "mean_queue": _fmt(float(row["mean_queue"])),
            "seed_std": _fmt(float(row["seed_std"])),
            "p90_queue": _fmt(float(row["p90_queue"])),
            "active_wait": _fmt(float(row["active_wait"])),
            "trip_time": _fmt(float(row["trip_time"])),
            "trip_wait": _fmt(float(row["trip_wait"])),
            "time_loss": _fmt(float(row["time_loss"])),
            "depart_delay": _fmt(float(row["depart_delay"])),
            "completed_trips": _fmt(float(row["completed_trips"])),
            "tripinfo_count": _fmt(float(row["tripinfo_count"])),
            "throughput": _fmt(float(row["throughput"])),
        }
        for row in rows
    ]
    out_md.write_text(
        "# Extended RESCO Summary\n\n"
        + _markdown_table(
            md_rows,
            [
                "policy",
                "mean_queue",
                "seed_std",
                "p90_queue",
                "active_wait",
                "trip_time",
                "trip_wait",
                "time_loss",
                "depart_delay",
                "completed_trips",
                "tripinfo_count",
                "throughput",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    out_tex.write_text(_latex_table(rows), encoding="utf-8")
    print(f"wrote {out_fig}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"wrote {out_tex}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--fig-out", type=Path, default=OUT_FIG)
    parser.add_argument("--csv-out", type=Path, default=OUT_CSV)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    parser.add_argument("--tex-out", type=Path, default=OUT_TEX)
    parser.add_argument("--title", default="Extended RESCO multi-seed phase-transfer validation")
    args = parser.parse_args()
    build_summary(
        result_path=args.result,
        out_fig=args.fig_out,
        out_csv=args.csv_out,
        out_md=args.md_out,
        out_tex=args.tex_out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
