"""Leave-one-target-group-out deployment selector for pairwise control."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    NUMERICAL_TOLERANCE,
    summarize_deployment_gate,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)


LOGO_GROUP_COUNT = 60
LOGO_TRAIN_GROUP_COUNT = 59
MIN_MEAN_LOGO_GAIN = 0.01
MIN_LOGO_NONWORSE_FRACTION = 0.50
MIN_WORST_SEED_GAIN = -0.02


def select_leave_one_group_pairwise(
    logo_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(logo_rows) != LOGO_GROUP_COUNT:
        raise ValueError("LOGO gate requires exactly 60 rows")
    group_ids = [str(row["group_id"]) for row in logo_rows]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("every target group must have exactly one LOGO row")
    training_counts = {
        int(row.get("training_group_count", -1)) for row in logo_rows
    }
    if training_counts != {LOGO_TRAIN_GROUP_COUNT}:
        raise ValueError("every LOGO prediction must use exactly 59 target groups")
    gains = np.asarray([float(row["paired_gain"]) for row in logo_rows], dtype=float)
    seeds = np.asarray(
        [int(row["simulator_seed"]) for row in logo_rows],
        dtype=int,
    )
    if not np.all(np.isfinite(gains)):
        raise ValueError("LOGO paired gains must be finite")
    if len(set(seeds.tolist())) < 2:
        raise ValueError("LOGO rows must cover multiple simulator seeds")
    mean_gain = float(np.mean(gains))
    nonworse_fraction = float(np.mean(gains >= -NUMERICAL_TOLERANCE))
    seed_mean_gains = {
        int(seed): float(np.mean(gains[seeds == seed])) for seed in sorted(set(seeds))
    }
    worst_seed_gain = min(seed_mean_gains.values())
    gate = {
        "exactly_sixty_unique_logo_groups": len(set(group_ids)) == 60,
        "every_logo_fit_uses_fifty_nine_groups": training_counts == {59},
        "mean_logo_gain_at_least_0_01": (
            mean_gain + NUMERICAL_TOLERANCE >= MIN_MEAN_LOGO_GAIN
        ),
        "logo_nonworse_fraction_at_least_0_50": (
            nonworse_fraction + NUMERICAL_TOLERANCE
            >= MIN_LOGO_NONWORSE_FRACTION
        ),
        "worst_seed_gain_at_least_minus_0_02": (
            worst_seed_gain + NUMERICAL_TOLERANCE >= MIN_WORST_SEED_GAIN
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "leave-one-target-group-out-seed-robust-pairwise-gate-v1",
        "selected_family": (
            PAIRWISE_PREFERENCE_FAMILY
            if gate["passed"]
            else GROUP_NORMALIZED_RIGID_FAMILY
        ),
        "logo_group_count": len(logo_rows),
        "logo_training_group_count": LOGO_TRAIN_GROUP_COUNT,
        "mean_logo_paired_gain": mean_gain,
        "logo_nonworse_fraction": nonworse_fraction,
        "seed_mean_gains": seed_mean_gains,
        "worst_seed_gain": worst_seed_gain,
        "gate": gate,
    }


def summarize_logo_deployment(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    evaluation_regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    summary = summarize_deployment_gate(
        decisions=decisions,
        evaluation_regrets=evaluation_regrets,
    )
    summary["protocol"] = "frozen-six-city-logo-pairwise-selector-v1"
    summary["decision"] = (
        "freeze_logo_selector_for_salt_lake"
        if summary["passed"]
        else "reject_logo_pairwise_deployment"
    )
    return summary
