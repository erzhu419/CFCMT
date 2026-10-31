"""Frozen selector for the zero-prior invariant rollout-value residual."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY
from cf_h2o.eval.traffic_signal_rollout_value_selection import (
    ROLLOUT_VALUE_FAMILY,
    select_rollout_value_family,
)


DIRECT_ROLLOUT_VALUE_FAMILY = "cfcmt_direct_rollout_value_rigid_residual"
EVALUATION_FAMILIES = (DIRECT_ROLLOUT_VALUE_FAMILY,)


def select_direct_rollout_value_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != {RIGID_FAMILY, DIRECT_ROLLOUT_VALUE_FAMILY}:
        raise ValueError("direct rollout-value family matrix changed")
    mapped = {
        RIGID_FAMILY: regrets[RIGID_FAMILY],
        ROLLOUT_VALUE_FAMILY: regrets[DIRECT_ROLLOUT_VALUE_FAMILY],
    }
    result = deepcopy(select_rollout_value_family(mapped))
    result["protocol"] = "frozen-six-city-direct-rollout-value-selector-v1"
    result["decision"] = (
        "promote_direct_rollout_value_residual"
        if result["passed"]
        else "direct_rollout_value_residual_failed_no_promotion"
    )
    result["selected_family"] = (
        DIRECT_ROLLOUT_VALUE_FAMILY if result["passed"] else None
    )
    result["candidate_summary"]["family"] = DIRECT_ROLLOUT_VALUE_FAMILY
    result["candidate_summary"]["label"] = (
        "Rigid + direct invariant policy-conditioned rollout value"
    )
    return result
