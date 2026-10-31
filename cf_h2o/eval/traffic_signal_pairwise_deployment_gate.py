"""Calibration-only deployment gate for target-adapted pairwise control."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import TARGETS


TARGET_GROUP_BUDGET = 60
MIN_CALIBRATION_GROUPS = 8
MIN_MEAN_GAIN = 0.01
LOWER_CONFIDENCE_Z = 1.645
MIN_GAIN_LOWER_BOUND = -0.01
MIN_NONWORSE_FRACTION = 0.50
TEMPORAL_BLOCK_COUNT = 4
MIN_WORST_TEMPORAL_BLOCK_GAIN = -0.05
MACRO_IMPROVEMENT_THRESHOLD_PCT = 5.0
MIN_SELECTED_IMPROVED_TARGETS = 4
MAX_ABSOLUTE_REGRESSION = 0.02
NUMERICAL_TOLERANCE = 1e-12


def select_target_pairwise(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select pairwise or exact fallback from disjoint calibration groups."""

    if len(calibration_rows) < MIN_CALIBRATION_GROUPS:
        gains = np.asarray([], dtype=float)
        times = np.asarray([], dtype=float)
    else:
        group_ids = [str(row["group_id"]) for row in calibration_rows]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("calibration groups must be unique")
        gains = np.asarray([float(row["paired_gain"]) for row in calibration_rows])
        times = np.asarray([float(row["snapshot_time_sec"]) for row in calibration_rows])
        if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(times)):
            raise ValueError("calibration rows must be finite")

    count = int(gains.size)
    mean_gain = float(np.mean(gains)) if count else float("-inf")
    sample_sd = float(np.std(gains, ddof=1)) if count >= 2 else float("inf")
    lower_bound = (
        mean_gain - LOWER_CONFIDENCE_Z * sample_sd / math.sqrt(count)
        if count >= 2
        else float("-inf")
    )
    nonworse_fraction = (
        float(np.mean(gains >= -NUMERICAL_TOLERANCE)) if count else 0.0
    )
    temporal_block_gains: list[float] = []
    if count:
        order = np.lexsort((np.arange(count), times))
        for rows in np.array_split(order, min(TEMPORAL_BLOCK_COUNT, count)):
            temporal_block_gains.append(float(np.mean(gains[rows])))
    worst_temporal_block_gain = (
        min(temporal_block_gains) if temporal_block_gains else float("-inf")
    )
    gate = {
        "at_least_eight_calibration_groups": count >= MIN_CALIBRATION_GROUPS,
        "mean_gain_at_least_0_01": (
            mean_gain + NUMERICAL_TOLERANCE >= MIN_MEAN_GAIN
        ),
        "one_sided_lower_bound_at_least_minus_0_01": (
            lower_bound + NUMERICAL_TOLERANCE >= MIN_GAIN_LOWER_BOUND
        ),
        "nonworse_fraction_at_least_0_50": (
            nonworse_fraction + NUMERICAL_TOLERANCE >= MIN_NONWORSE_FRACTION
        ),
        "worst_temporal_block_gain_at_least_minus_0_05": (
            worst_temporal_block_gain + NUMERICAL_TOLERANCE
            >= MIN_WORST_TEMPORAL_BLOCK_GAIN
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "calibration-only-paired-regret-lcb-temporal-gate-v1",
        "selected_family": (
            PAIRWISE_PREFERENCE_FAMILY
            if gate["passed"]
            else GROUP_NORMALIZED_RIGID_FAMILY
        ),
        "calibration_group_count": count,
        "mean_paired_gain": mean_gain,
        "sample_sd_paired_gain": sample_sd,
        "one_sided_95_lower_bound": lower_bound,
        "nonworse_fraction": nonworse_fraction,
        "temporal_block_mean_gains": temporal_block_gains,
        "worst_temporal_block_gain": worst_temporal_block_gain,
        "gate": gate,
    }


def summarize_deployment_gate(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    evaluation_regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the frozen six-city deployment efficacy gate."""

    if set(decisions) != set(TARGETS):
        raise ValueError("deployment decisions do not cover the frozen targets")
    expected_families = {
        GROUP_NORMALIZED_RIGID_FAMILY,
        PAIRWISE_PREFERENCE_FAMILY,
    }
    if set(evaluation_regrets) != expected_families:
        raise ValueError("deployment evaluation family set changed")
    for family in expected_families:
        if set(evaluation_regrets[family]) != set(TARGETS):
            raise ValueError(f"deployment evaluation targets changed for {family}")

    selected_regrets: dict[str, float] = {}
    fallback_regrets: dict[str, float] = {}
    selected_pairwise_targets = []
    for target in TARGETS:
        decision = decisions[target]
        selected = str(decision["selected_family"])
        if selected not in expected_families:
            raise ValueError(f"invalid selected family for {target}: {selected}")
        if selected == PAIRWISE_PREFERENCE_FAMILY:
            if decision.get("gate", {}).get("passed") is not True:
                raise ValueError(f"{target}: pairwise selected without passing gate")
            selected_pairwise_targets.append(target)
        elif decision.get("gate", {}).get("passed") is not False:
            raise ValueError(f"{target}: fallback selected despite passing gate")
        selected_regrets[target] = float(evaluation_regrets[selected][target])
        fallback_regrets[target] = float(
            evaluation_regrets[GROUP_NORMALIZED_RIGID_FAMILY][target]
        )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (*selected_regrets.values(), *fallback_regrets.values())
    ):
        raise ValueError("deployment evaluation regrets must be finite and nonnegative")

    selected_macro = float(np.mean(list(selected_regrets.values())))
    fallback_macro = float(np.mean(list(fallback_regrets.values())))
    improvements = {
        target: fallback_regrets[target] - selected_regrets[target]
        for target in TARGETS
    }
    improvement_pct = 100.0 * (fallback_macro - selected_macro) / fallback_macro
    selected_improved = sum(
        target in selected_pairwise_targets
        and improvements[target] > NUMERICAL_TOLERANCE
        for target in TARGETS
    )
    max_regression = max(max(-value, 0.0) for value in improvements.values())
    stage_gate = {
        "macro_improvement_at_least_5_pct": (
            improvement_pct + NUMERICAL_TOLERANCE
            >= MACRO_IMPROVEMENT_THRESHOLD_PCT
        ),
        "at_least_four_selected_cities_improved": (
            selected_improved >= MIN_SELECTED_IMPROVED_TARGETS
        ),
        "max_regression_at_most_0_02": (
            max_regression <= MAX_ABSOLUTE_REGRESSION + NUMERICAL_TOLERANCE
        ),
    }
    stage_gate["passed"] = all(stage_gate.values())
    return {
        "protocol": "frozen-six-city-calibration-gated-pairwise-selector-v1",
        "passed": stage_gate["passed"],
        "decision": (
            "freeze_selector_for_salt_lake"
            if stage_gate["passed"]
            else "reject_calibration_only_pairwise_deployment"
        ),
        "selected_pairwise_targets": selected_pairwise_targets,
        "selected_pairwise_target_count": len(selected_pairwise_targets),
        "selected_policy_macro_regret": selected_macro,
        "fallback_macro_regret": fallback_macro,
        "relative_improvement_vs_fallback_pct": improvement_pct,
        "selected_improved_target_count": selected_improved,
        "max_absolute_regression": max_regression,
        "target_improvements_vs_fallback": improvements,
        "selected_regrets": selected_regrets,
        "fallback_regrets": fallback_regrets,
        "stage_gate": stage_gate,
    }
