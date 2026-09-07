"""Aggregate the frozen V140 budget matrix with simultaneous intervals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    TARGET_BUDGETS,
)
from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting import (
    PRIMARY_BUDGET,
    RESULT_PROTOCOL as BUDGET_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v140-total-budget-causal-source-weighting-aggregate-v1"
EFFECT_KEYS = (
    "source_weighted_minus_target_only",
    "source_weighted_minus_matched_placebo",
)


def simultaneous_max_t_intervals(
    seed_effects: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = np.asarray(seed_effects, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("V140 max-t matrix must be seeds by budgets")
    if not np.all(np.isfinite(values)) or int(replicates) <= 0:
        raise ValueError("V140 max-t inputs are invalid")
    means = np.mean(values, axis=0)
    standard_errors = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    centered = values - means
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.shape[0], size=(int(replicates), values.shape[0]))
    bootstrap_means = np.mean(centered[indices], axis=1)
    t_values = np.zeros_like(bootstrap_means)
    active = standard_errors > 0.0
    t_values[:, active] = bootstrap_means[:, active] / standard_errors[active]
    critical = float(np.quantile(np.max(np.abs(t_values), axis=1), 0.95))
    half_width = critical * standard_errors
    return {
        "means": means.tolist(),
        "standard_errors": standard_errors.tolist(),
        "lower_95_simultaneous": (means - half_width).tolist(),
        "upper_95_simultaneous": (means + half_width).tolist(),
        "critical_value": critical,
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": "shared_evaluation_seed_across_all_budgets",
        "two_sided_familywise_level": 0.95,
    }


def _read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V140 result is not a JSON object: {path}")
    return payload


def aggregate_budget_results(paths: Sequence[Path]) -> dict[str, Any]:
    rows = [_read_result(Path(path)) for path in paths]
    by_budget = {int(row.get("budget", -1)): row for row in rows}
    if tuple(sorted(by_budget)) != TARGET_BUDGETS or len(rows) != len(by_budget):
        raise ValueError("V140 aggregate requires exactly one result per frozen budget")
    ordered = [by_budget[budget] for budget in TARGET_BUDGETS]
    first = ordered[0]
    evaluation_seeds = tuple(int(value) for value in first["evaluation_seeds"])
    adaptation_seeds = tuple(int(value) for value in first["adaptation_seeds"])
    source_order = tuple(str(value) for value in first["source_group_order"])
    input_identity = dict(first["inputs"])
    selected_groups = {}
    for budget, row in zip(TARGET_BUDGETS, ordered):
        if (
            row.get("protocol") != BUDGET_RESULT_PROTOCOL
            or int(row.get("primary_budget", -1)) != PRIMARY_BUDGET
            or tuple(int(value) for value in row["evaluation_seeds"]) != evaluation_seeds
            or tuple(int(value) for value in row["adaptation_seeds"]) != adaptation_seeds
            or tuple(str(value) for value in row["source_group_order"]) != source_order
            or dict(row["inputs"]) != input_identity
        ):
            raise ValueError(f"V140 B{budget} protocol or input identity changed")
        information = dict(row["information_budget"])
        groups = tuple(str(value) for value in information["adaptation_group_ids"])
        if (
            len(groups) != budget
            or int(information["union_target_group_count"]) != budget
            or information["evaluation_seed_labels_used_for_fit_weighting_or_selection"]
            is not False
        ):
            raise ValueError(f"V140 B{budget} target-information budget changed")
        selected_groups[budget] = groups
    for lower, upper in zip(TARGET_BUDGETS, TARGET_BUDGETS[1:]):
        if not set(selected_groups[lower]) < set(selected_groups[upper]):
            raise ValueError(f"V140 aggregate budgets are not nested: {lower}, {upper}")

    effects = {}
    for effect_key in EFFECT_KEYS:
        matrix = np.column_stack(
            [
                np.asarray(row["paired_effects"][effect_key]["seed_values"], dtype=float)
                for row in ordered
            ]
        )
        if matrix.shape != (len(evaluation_seeds), len(TARGET_BUDGETS)):
            raise ValueError(f"V140 {effect_key} seed matrix is misaligned")
        effects[effect_key] = {
            "seed_effect_matrix": matrix.tolist(),
            "simultaneous_interval": simultaneous_max_t_intervals(matrix),
        }

    curve = []
    for index, (budget, row) in enumerate(zip(TARGET_BUDGETS, ordered)):
        source_target_interval = effects["source_weighted_minus_target_only"][
            "simultaneous_interval"
        ]
        source_placebo_interval = effects[
            "source_weighted_minus_matched_placebo"
        ]["simultaneous_interval"]
        curve.append(
            {
                "budget": budget,
                "adaptation_source_authorized": bool(
                    row["development_gate"]["adaptation_source_authorized"]
                ),
                "selector_relative_passed": bool(
                    row["development_gate"]["selector_relative_passed"]
                ),
                "effective_source_null_weight": float(
                    row["effective_source_null_weight"]
                ),
                "source_weighted_minus_target_only": {
                    **dict(row["paired_effects"]["source_weighted_minus_target_only"]),
                    "lower_95_simultaneous": source_target_interval[
                        "lower_95_simultaneous"
                    ][index],
                    "upper_95_simultaneous": source_target_interval[
                        "upper_95_simultaneous"
                    ][index],
                },
                "source_weighted_minus_matched_placebo": {
                    **dict(
                        row["paired_effects"][
                            "source_weighted_minus_matched_placebo"
                        ]
                    ),
                    "lower_95_simultaneous": source_placebo_interval[
                        "lower_95_simultaneous"
                    ][index],
                    "upper_95_simultaneous": source_placebo_interval[
                        "upper_95_simultaneous"
                    ][index],
                },
                "target_only_minus_phase_pressure": float(
                    row["arm_summaries"]["target_only"]["mean"]
                ),
                "source_weighted_minus_phase_pressure": float(
                    row["arm_summaries"]["causal_source_weighted"]["mean"]
                ),
                "effective_source_weights": dict(row["effective_source_weights"]),
            }
        )

    primary = by_budget[PRIMARY_BUDGET]
    primary_passed = bool(primary["development_gate"]["primary_budget_passed"])
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "total-target-label-budget-relative-transfer-development",
        "target_city": "jinan",
        "primary_budget": PRIMARY_BUDGET,
        "target_budgets": list(TARGET_BUDGETS),
        "adaptation_seeds": list(adaptation_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "source_group_order": list(source_order),
        "inputs": input_identity,
        "curve": curve,
        "familywise_effects": effects,
        "development_decision": {
            "primary_budget_passed": primary_passed,
            "decision": (
                "authorize_frozen_untouched_city_confirmation"
                if primary_passed
                else "do_not_authorize_untouched_city_confirmation"
            ),
            "budget_selection_after_evaluation_permitted": False,
        },
        "claim_boundary": (
            "B25 is the sole predeclared primary relative-transfer test. "
            "Other budgets and max-t intervals describe the adaptation curve; "
            "absolute PhasePressure comparisons remain separate columns."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V140 aggregate: {args.out}")
    result = aggregate_budget_results(args.results)
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "development_decision": result["development_decision"],
                "curve": result["curve"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
