"""Frozen six-city selector for the policy-conditioned rollout-value residual."""

from __future__ import annotations

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


ROLLOUT_VALUE_FAMILY = "cfcmt_rollout_value_rigid_residual"
EVALUATION_FAMILIES = (ROLLOUT_VALUE_FAMILY,)


def select_rollout_value_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    expected = {RIGID_FAMILY, ROLLOUT_VALUE_FAMILY}
    if set(regrets) != expected:
        raise ValueError("rollout-value family matrix changed")
    values: dict[str, dict[str, float]] = {}
    for family in (RIGID_FAMILY, ROLLOUT_VALUE_FAMILY):
        if set(regrets[family]) != set(TARGETS):
            raise ValueError(f"rollout-value target mismatch for {family}")
        values[family] = {}
        for target in TARGETS:
            value = float(regrets[family][target])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid regret for {family}/{target}: {value}")
            values[family][target] = value

    rigid = values[RIGID_FAMILY]
    candidate = values[ROLLOUT_VALUE_FAMILY]
    rigid_macro = sum(rigid.values()) / len(TARGETS)
    candidate_macro = sum(candidate.values()) / len(TARGETS)
    improvement_pct = 100.0 * (rigid_macro - candidate_macro) / rigid_macro
    improvements = {
        target: rigid[target] - candidate[target] for target in TARGETS
    }
    improved_count = sum(
        value > NUMERICAL_TOLERANCE for value in improvements.values()
    )
    max_regression = max(
        max(candidate[target] - rigid[target], 0.0) for target in TARGETS
    )
    gate = {
        "macro_improvement_at_least_10_pct": (
            improvement_pct + NUMERICAL_TOLERANCE
            >= MACRO_IMPROVEMENT_THRESHOLD_PCT
        ),
        "at_least_four_cities_improved": improved_count >= MIN_IMPROVED_TARGETS,
        "max_regression_at_most_0_05": (
            max_regression <= MAX_ABSOLUTE_REGRESSION + NUMERICAL_TOLERANCE
        ),
    }
    gate["passed"] = all(gate.values())
    summary = {
        "family": ROLLOUT_VALUE_FAMILY,
        "label": "Rigid + policy-conditioned rollout-value residual",
        "city_macro_mean_normalized_action_regret": candidate_macro,
        "relative_improvement_vs_rigid_pct": improvement_pct,
        "cities_improved": improved_count,
        "max_absolute_regression": max_regression,
        "target_improvements_vs_rigid": improvements,
        "targets": candidate,
        "stage_gate": gate,
    }
    return {
        "protocol": "frozen-six-city-rollout-value-residual-selector-v1",
        "passed": gate["passed"],
        "decision": (
            "promote_rollout_value_residual"
            if gate["passed"]
            else "rollout_value_residual_failed_no_promotion"
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
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
        "selected_family": ROLLOUT_VALUE_FAMILY if gate["passed"] else None,
        "candidate_summary": summary,
    }
