"""Aggregate the seven frozen V143 complete-city target results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    RESULT_PROTOCOL as TARGET_RESULT_PROTOCOL,
    TARGET_BUDGET,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v143-multicity-uniform-source-ensemble-aggregate-v1"
MINIMUM_MEAN_IMPROVEMENT = 0.0005
MINIMUM_IMPROVING_CITIES = 5
MAXIMUM_CITY_REGRESSION = 0.01
COMPARATORS = (
    "target_only_causal",
    "pooled_h2oplus",
    "matched_source_placebo",
)


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V143 result is not a JSON object: {path}")
    return value


def _city_effect_summary(values: Mapping[str, float]) -> dict[str, Any]:
    if set(values) != set(EXPECTED_CITY_GROUPS):
        raise ValueError("V143 city effect inventory changed")
    ordered = np.asarray(
        [float(values[city]) for city in EXPECTED_CITY_GROUPS], dtype=float
    )
    if not np.all(np.isfinite(ordered)):
        raise ValueError("V143 city effect contains a non-finite value")
    critical = float(student_t.ppf(0.95, df=ordered.size - 1))
    return {
        **_paired_bootstrap(ordered.tolist()),
        "upper_95_one_sided": float(
            np.mean(ordered)
            + critical * np.std(ordered, ddof=1) / np.sqrt(ordered.size)
        ),
        "improving_city_count": int(np.count_nonzero(ordered < 0.0)),
        "maximum_city_regression": float(max(np.max(ordered), 0.0)),
        "city_values": {
            city: float(values[city]) for city in EXPECTED_CITY_GROUPS
        },
    }


def aggregate_multicity_uniform_source_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    if len(rows) != len(EXPECTED_CITY_GROUPS):
        raise ValueError("V143 requires exactly seven target results")
    by_city = {str(row.get("target_city")): row for row in rows}
    if set(by_city) != set(EXPECTED_CITY_GROUPS) or len(by_city) != len(rows):
        raise ValueError("V143 target city results are missing or duplicated")
    input_identities = None
    city_rows = {}
    effects = {comparator: {} for comparator in COMPARATORS}
    for city in EXPECTED_CITY_GROUPS:
        row = by_city[city]
        if (
            row.get("protocol") != TARGET_RESULT_PROTOCOL
            or int(row.get("target_budget", -1)) != TARGET_BUDGET
            or set(row.get("heldout_target_city_scenarios", ()))
            & {
                scenario
                for scenarios in row.get("source_scenarios_by_group", {}).values()
                for scenario in scenarios
            }
            or city in set(row.get("source_city_groups", ()))
            or len(row.get("source_city_groups", ())) != 6
            or row.get("information_budget", {}).get(
                "evaluation_groups_used_for_fit_or_selection"
            )
            is not False
        ):
            raise ValueError(f"{city}: V143 information boundary changed")
        identities = dict(row.get("inputs", {}))
        if input_identities is None:
            input_identities = identities
        elif identities != input_identities:
            raise ValueError("V143 target results do not share input identities")
        arms = row.get("arm_summaries", {})
        expected_arms = {
            "phase_pressure",
            "target_only_causal",
            "pooled_h2oplus",
            "uniform_cfcmt",
            "matched_source_placebo",
        }
        if set(arms) != expected_arms:
            raise ValueError(f"{city}: V143 arm inventory changed")
        city_rows[city] = {
            arm: float(arms[arm]["mean"]) for arm in sorted(expected_arms)
        }
        for comparator in COMPARATORS:
            effects[comparator][city] = float(
                arms["uniform_cfcmt"]["mean"] - arms[comparator]["mean"]
            )
    summaries = {
        comparator: _city_effect_summary(values)
        for comparator, values in effects.items()
    }
    comparator_pass = {}
    for comparator, summary in summaries.items():
        passed = bool(
            float(summary["mean"]) <= -MINIMUM_MEAN_IMPROVEMENT
            and float(summary["upper_95_one_sided"]) < 0.0
            and int(summary["improving_city_count"])
            >= MINIMUM_IMPROVING_CITIES
        )
        if comparator != "matched_source_placebo":
            passed = passed and bool(
                float(summary["maximum_city_regression"])
                <= MAXIMUM_CITY_REGRESSION
            )
        comparator_pass[comparator] = passed
    passed = all(comparator_pass.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "seven-city-complete-holdout-development-aggregate",
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "target_budget": TARGET_BUDGET,
        "city_arm_means": city_rows,
        "paired_city_effects": {
            f"uniform_cfcmt_minus_{comparator}": summary
            for comparator, summary in summaries.items()
        },
        "development_gate": {
            "comparator_pass": comparator_pass,
            "passed": passed,
            "decision": (
                "authorize_separately_frozen_untouched_city_protocol"
                if passed
                else "reject_fixed_uniform_source_ensemble"
            ),
            "minimum_mean_improvement": MINIMUM_MEAN_IMPROVEMENT,
            "minimum_improving_cities": MINIMUM_IMPROVING_CITIES,
            "maximum_city_regression": MAXIMUM_CITY_REGRESSION,
        },
        "inputs": input_identities,
        "claim_boundary": (
            "V143 is seven-city development. A pass may authorize a new frozen "
            "confirmation protocol but is not itself untouched-city evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V143 aggregate: {args.out}")
    result = aggregate_multicity_uniform_source_results(
        [_read_result(path) for path in args.results]
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["development_gate"]["passed"] else "REJECT",
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "paired_city_effects": result["paired_city_effects"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
