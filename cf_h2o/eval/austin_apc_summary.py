"""Summarize Austin real APC validation runs for paper artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = [
    Path("cf_h2o/results/austin_real_apc_validation.json"),
    Path("cf_h2o/results/austin_real_apc_validation_2016_aug.json"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(metrics: dict[str, Any], method: str, denominator: str = "passive_no_correction") -> float | None:
    value = metrics.get(method)
    base = metrics.get(denominator)
    if not value or not base:
        return None
    base_total = base.get("total_mse")
    if not base_total:
        return None
    return float(value["total_mse"] / base_total)


def build_summary(paths: list[Path]) -> dict[str, Any]:
    rows = []
    few_shot_rows = []
    for path in paths:
        data = _read_json(path)
        dataset = data["dataset"]
        real = data["real_apc_summary"]
        comparisons = data["validation"]["comparisons"]
        few = data.get("few_shot_validation", {})
        row = {
            "result_path": str(path),
            "dataset": dataset["label"],
            "source": dataset["source"],
            "rows_available": dataset["download"]["rows_available"],
            "transitions": real["transitions"],
            "routes": real["route_count"],
            "stops": real["stop_count"],
            "trips": real["trip_count"],
            "date_min": real["date_min"],
            "date_max": real["date_max"],
            "h2oplus_vs_passive_ratio": comparisons["h2oplus_vs_passive_total_mse_ratio"],
            "cfcmt_vs_passive_ratio": comparisons["cfcmt_vs_passive_total_mse_ratio"],
            "weighted_cfcmt_vs_passive_ratio": comparisons["cfcmt_similarity_weighted_vs_passive_total_mse_ratio"],
            "h2oplus_vs_uncalibrated_ratio": comparisons["h2oplus_vs_uncalibrated_total_mse_ratio"],
            "cfcmt_vs_uncalibrated_ratio": comparisons["cfcmt_vs_uncalibrated_total_mse_ratio"],
            "weighted_cfcmt_vs_uncalibrated_ratio": comparisons[
                "cfcmt_similarity_weighted_vs_uncalibrated_total_mse_ratio"
            ],
            "cfcmt_vs_h2oplus_ratio": comparisons["cfcmt_vs_h2oplus_total_mse_ratio"],
            "weighted_cfcmt_vs_h2oplus_ratio": comparisons["cfcmt_similarity_weighted_vs_h2oplus_total_mse_ratio"],
            "few_shot_summary": few.get("summary"),
            "notes": real["notes"],
        }
        rows.append(row)

        for shot in few.get("rows", []):
            metrics = shot["metrics"]
            few_shot_rows.append(
                {
                    "dataset": dataset["label"],
                    "target_line_budget_fraction": shot["target_line_budget_fraction"],
                    "calibration_lines": shot["calibration_lines"],
                    "evaluation_lines": shot["evaluation_lines"],
                    "calibration_transitions": shot["calibration_transitions"],
                    "evaluation_transitions": shot["evaluation_transitions"],
                    "weighted_source_only_vs_passive": _ratio(metrics, "cfcmt_weighted_source_only"),
                    "weighted_source_plus_target_vs_passive": _ratio(
                        metrics,
                        "cfcmt_weighted_source_plus_target_budget",
                    ),
                    "target_only_vs_passive": _ratio(metrics, "cfcmt_target_only_budget"),
                    "residual_gate_vs_passive": _ratio(metrics, "cfcmt_weighted_source_residual_gate"),
                    "bias_adapter_vs_passive": _ratio(metrics, "cfcmt_weighted_source_bias_adapter"),
                }
            )

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / max(1, len(rows))

    summary = {
        "datasets": len(rows),
        "total_transitions": sum(int(row["transitions"]) for row in rows),
        "all_cfcmt_beats_h2oplus": all(row["cfcmt_vs_h2oplus_ratio"] < 1.0 for row in rows),
        "all_weighted_cfcmt_beats_h2oplus": all(row["weighted_cfcmt_vs_h2oplus_ratio"] < 1.0 for row in rows),
        "all_weighted_cfcmt_beats_passive": all(row["weighted_cfcmt_vs_passive_ratio"] < 1.0 for row in rows),
        "all_weighted_cfcmt_beats_uncalibrated": all(row["weighted_cfcmt_vs_uncalibrated_ratio"] < 1.0 for row in rows),
        "mean_h2oplus_vs_passive_ratio": mean("h2oplus_vs_passive_ratio"),
        "mean_cfcmt_vs_passive_ratio": mean("cfcmt_vs_passive_ratio"),
        "mean_weighted_cfcmt_vs_passive_ratio": mean("weighted_cfcmt_vs_passive_ratio"),
        "mean_h2oplus_vs_uncalibrated_ratio": mean("h2oplus_vs_uncalibrated_ratio"),
        "mean_cfcmt_vs_uncalibrated_ratio": mean("cfcmt_vs_uncalibrated_ratio"),
        "mean_weighted_cfcmt_vs_uncalibrated_ratio": mean("weighted_cfcmt_vs_uncalibrated_ratio"),
        "mean_cfcmt_vs_h2oplus_ratio": mean("cfcmt_vs_h2oplus_ratio"),
        "mean_weighted_cfcmt_vs_h2oplus_ratio": mean("weighted_cfcmt_vs_h2oplus_ratio"),
    }
    if few_shot_rows:
        gate_rows = [row for row in few_shot_rows if row["target_line_budget_fraction"] > 0.0]
        summary["few_shot"] = {
            "rows": len(few_shot_rows),
            "budgets": sorted({row["target_line_budget_fraction"] for row in few_shot_rows}),
            "min_residual_gate_vs_passive": min(row["residual_gate_vs_passive"] for row in gate_rows),
            "min_target_only_vs_passive": min(
                row["target_only_vs_passive"] for row in gate_rows if row["target_only_vs_passive"] is not None
            ),
            "all_positive_budget_gate_beats_passive": all(row["residual_gate_vs_passive"] < 1.0 for row in gate_rows),
        }
    return {
        "ok": True,
        "validation_level": "austin_real_apc_external_validation_summary",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": rows,
        "few_shot_rows": few_shot_rows,
        "summary": summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/austin_real_apc_validation_summary.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
