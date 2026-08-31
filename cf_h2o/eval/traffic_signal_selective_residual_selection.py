"""Frozen six-city selector for source-only selective residual arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    MACRO_IMPROVEMENT_THRESHOLD_PCT,
    MAX_ABSOLUTE_REGRESSION,
    MIN_IMPROVED_TARGETS,
    NUMERICAL_TOLERANCE,
    RIGID_FAMILY,
    TARGETS,
)


@dataclass(frozen=True)
class SelectiveResidualCandidate:
    family: str
    label: str
    mechanism_count: int
    error_quantile: float


CANDIDATES = tuple(
    SelectiveResidualCandidate(
        family=f"cfcmt_one_step_selective_residual_{base}_q{int(quantile * 100):02d}",
        label=(
            f"Selective {base.replace('_', ' + ')} residual, "
            f"q={quantile:.2f}"
        ),
        mechanism_count=1 if base == "mobility" else 2,
        error_quantile=quantile,
    )
    for base in ("mobility", "mobility_queue")
    for quantile in (0.50, 0.75, 0.90)
)

EVALUATION_FAMILIES = tuple(candidate.family for candidate in CANDIDATES)
SELECTION_FAMILIES = (RIGID_FAMILY,) + EVALUATION_FAMILIES
MACRO_TIE_TOLERANCE = 0.005


def _validated_regrets(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(regrets) != set(SELECTION_FAMILIES):
        missing = sorted(set(SELECTION_FAMILIES) - set(regrets))
        extra = sorted(set(regrets) - set(SELECTION_FAMILIES))
        raise ValueError(
            f"selective-residual family mismatch: missing={missing}, extra={extra}"
        )
    result: dict[str, dict[str, float]] = {}
    for family in SELECTION_FAMILIES:
        if set(regrets[family]) != set(TARGETS):
            raise ValueError(f"selective-residual target mismatch for {family}")
        result[family] = {}
        for target in TARGETS:
            value = float(regrets[family][target])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid regret for {family}/{target}: {value}")
            result[family][target] = value
    return result


def select_selective_residual_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    values = _validated_regrets(regrets)
    rigid = values[RIGID_FAMILY]
    rigid_macro = sum(rigid.values()) / len(TARGETS)
    if rigid_macro <= NUMERICAL_TOLERANCE:
        raise ValueError("rigid macro regret must be positive")

    summaries: list[dict[str, Any]] = []
    for fixed_order, candidate in enumerate(CANDIDATES):
        target_values = values[candidate.family]
        macro = sum(target_values.values()) / len(TARGETS)
        deltas = {
            target: rigid[target] - target_values[target] for target in TARGETS
        }
        improvement_pct = 100.0 * (rigid_macro - macro) / rigid_macro
        improved_count = sum(
            delta > NUMERICAL_TOLERANCE for delta in deltas.values()
        )
        max_regression = max(
            max(target_values[target] - rigid[target], 0.0) for target in TARGETS
        )
        gate = {
            "macro_improvement_at_least_10_pct": (
                improvement_pct + NUMERICAL_TOLERANCE
                >= MACRO_IMPROVEMENT_THRESHOLD_PCT
            ),
            "at_least_four_cities_improved": improved_count >= MIN_IMPROVED_TARGETS,
            "max_regression_at_most_0_05": (
                max_regression
                <= MAX_ABSOLUTE_REGRESSION + NUMERICAL_TOLERANCE
            ),
        }
        gate["passed"] = all(gate.values())
        summaries.append(
            {
                "family": candidate.family,
                "label": candidate.label,
                "mechanism_count": candidate.mechanism_count,
                "error_quantile": candidate.error_quantile,
                "fixed_order": fixed_order,
                "city_macro_mean_normalized_action_regret": macro,
                "relative_improvement_vs_rigid_pct": improvement_pct,
                "cities_improved": improved_count,
                "max_absolute_regression": max_regression,
                "target_improvements_vs_rigid": deltas,
                "targets": target_values,
                "stage_gate": gate,
            }
        )

    passing = [row for row in summaries if row["stage_gate"]["passed"]]
    selected = None
    tie_pool: list[dict[str, Any]] = []
    if passing:
        best_macro = min(
            row["city_macro_mean_normalized_action_regret"] for row in passing
        )
        tie_pool = [
            row
            for row in passing
            if row["city_macro_mean_normalized_action_regret"]
            <= best_macro + MACRO_TIE_TOLERANCE + NUMERICAL_TOLERANCE
        ]
        selected = min(
            tie_pool,
            key=lambda row: (
                -row["error_quantile"],
                row["mechanism_count"],
                row["fixed_order"],
            ),
        )
    diagnostic_best = min(
        summaries,
        key=lambda row: (
            row["city_macro_mean_normalized_action_regret"],
            -row["error_quantile"],
            row["mechanism_count"],
            row["fixed_order"],
        ),
    )
    return {
        "protocol": "frozen-six-city-source-only-selective-residual-selector-v1",
        "passed": selected is not None,
        "decision": (
            "promote_selected_family"
            if selected
            else "selective_residual_stage_failed_no_promotion"
        ),
        "rigid": {
            "family": RIGID_FAMILY,
            "city_macro_mean_normalized_action_regret": rigid_macro,
            "targets": rigid,
        },
        "thresholds": {
            "macro_improvement_pct": MACRO_IMPROVEMENT_THRESHOLD_PCT,
            "minimum_improved_targets": MIN_IMPROVED_TARGETS,
            "maximum_absolute_regression": MAX_ABSOLUTE_REGRESSION,
            "macro_tie_tolerance": MACRO_TIE_TOLERANCE,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
        "candidate_order": [candidate.family for candidate in CANDIDATES],
        "family_summaries": summaries,
        "passing_families": [row["family"] for row in passing],
        "tie_pool": [row["family"] for row in tie_pool],
        "selected_family": selected["family"] if selected else None,
        "selected_summary": selected,
        "diagnostic_best_family": diagnostic_best["family"],
        "diagnostic_best_summary": diagnostic_best,
    }
