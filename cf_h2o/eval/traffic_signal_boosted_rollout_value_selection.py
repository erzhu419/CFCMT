"""Frozen selector for the parent-restricted nonlinear rollout value."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_direct_rollout_value_selection import (
    DIRECT_ROLLOUT_VALUE_FAMILY,
    select_direct_rollout_value_family,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY


BOOSTED_ROLLOUT_VALUE_FAMILY = (
    "cfcmt_boosted_direct_rollout_value_rigid_residual"
)
EVALUATION_FAMILIES = (BOOSTED_ROLLOUT_VALUE_FAMILY,)


def select_boosted_rollout_value_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != {RIGID_FAMILY, BOOSTED_ROLLOUT_VALUE_FAMILY}:
        raise ValueError("boosted rollout-value family matrix changed")
    mapped = {
        RIGID_FAMILY: regrets[RIGID_FAMILY],
        DIRECT_ROLLOUT_VALUE_FAMILY: regrets[BOOSTED_ROLLOUT_VALUE_FAMILY],
    }
    result = deepcopy(select_direct_rollout_value_family(mapped))
    result["protocol"] = "frozen-six-city-boosted-rollout-value-selector-v1"
    result["decision"] = (
        "promote_boosted_rollout_value_residual"
        if result["passed"]
        else "boosted_rollout_value_residual_failed_no_promotion"
    )
    result["selected_family"] = (
        BOOSTED_ROLLOUT_VALUE_FAMILY if result["passed"] else None
    )
    result["candidate_summary"]["family"] = BOOSTED_ROLLOUT_VALUE_FAMILY
    result["candidate_summary"]["label"] = (
        "Rigid + parent-restricted nonlinear rollout value"
    )
    return result
