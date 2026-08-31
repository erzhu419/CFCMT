"""Frozen six-city selector for the rigid-anchored mechanism residual screen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


TARGETS = (
    "grid4x4",
    "cologne1",
    "ingolstadt1",
    "atlanta_1x5",
    "hangzhou_4x4",
    "manhattan_28x7",
)

RIGID_FAMILY = "causal_rigid_advantage"


@dataclass(frozen=True)
class RigidResidualCandidate:
    family: str
    label: str
    mechanism_count: int


CANDIDATES = (
    RigidResidualCandidate(
        "cfcmt_one_step_rigid_residual_queue_only",
        "Rigid + queue residual",
        1,
    ),
    RigidResidualCandidate(
        "cfcmt_one_step_rigid_residual_mobility_only",
        "Rigid + mobility residual",
        1,
    ),
    RigidResidualCandidate(
        "cfcmt_one_step_rigid_residual_mobility_queue",
        "Rigid + mobility and queue residuals",
        2,
    ),
    RigidResidualCandidate(
        "cfcmt_one_step_rigid_residual_mechanism",
        "Rigid + full five-mechanism residual",
        5,
    ),
)

EVALUATION_FAMILIES = (RIGID_FAMILY,) + tuple(
    candidate.family for candidate in CANDIDATES
)

MACRO_IMPROVEMENT_THRESHOLD_PCT = 10.0
MIN_IMPROVED_TARGETS = 4
MAX_ABSOLUTE_REGRESSION = 0.05
MACRO_TIE_TOLERANCE = 0.005
NUMERICAL_TOLERANCE = 1e-12


def _validated_regrets(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(regrets) != set(EVALUATION_FAMILIES):
        missing = sorted(set(EVALUATION_FAMILIES) - set(regrets))
        extra = sorted(set(regrets) - set(EVALUATION_FAMILIES))
        raise ValueError(
            f"rigid-residual family mismatch: missing={missing}, extra={extra}"
        )
    validated: dict[str, dict[str, float]] = {}
    for family in EVALUATION_FAMILIES:
        target_values = regrets[family]
        if set(target_values) != set(TARGETS):
            missing = sorted(set(TARGETS) - set(target_values))
            extra = sorted(set(target_values) - set(TARGETS))
            raise ValueError(
                f"rigid-residual target mismatch for {family}: "
                f"missing={missing}, extra={extra}"
            )
        validated[family] = {}
        for target in TARGETS:
            value = float(target_values[target])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid regret for {family}/{target}: {value}")
            validated[family][target] = value
    return validated


def select_rigid_residual_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the frozen efficacy, breadth, harm, and parsimony rules."""

    values = _validated_regrets(regrets)
    rigid = values[RIGID_FAMILY]
    rigid_macro = sum(rigid.values()) / len(TARGETS)
    if rigid_macro <= NUMERICAL_TOLERANCE:
        raise ValueError("rigid macro regret must be positive")

    summaries: list[dict[str, Any]] = []
    for order, candidate in enumerate(CANDIDATES):
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
            "at_least_four_cities_improved": (
                improved_count >= MIN_IMPROVED_TARGETS
            ),
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
                "fixed_order": order,
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
    tie_pool: list[dict[str, Any]] = []
    selected = None
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
            key=lambda row: (row["mechanism_count"], row["fixed_order"]),
        )

    diagnostic_best = min(
        summaries,
        key=lambda row: (
            row["city_macro_mean_normalized_action_regret"],
            row["mechanism_count"],
            row["fixed_order"],
        ),
    )
    return {
        "protocol": "frozen-six-city-rigid-anchored-residual-selector-v1",
        "passed": selected is not None,
        "decision": (
            "promote_selected_family"
            if selected
            else "rigid_residual_stage_failed_no_promotion"
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
