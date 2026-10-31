"""Frozen selector for the group-regret-aligned rigid action model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY
from cf_h2o.eval.traffic_signal_rollout_value_selection import (
    ROLLOUT_VALUE_FAMILY,
    select_rollout_value_family,
)


GROUP_NORMALIZED_RIGID_FAMILY = "causal_group_normalized_rigid_advantage"
EVALUATION_FAMILIES = (GROUP_NORMALIZED_RIGID_FAMILY,)


def select_group_normalized_rigid_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != {RIGID_FAMILY, GROUP_NORMALIZED_RIGID_FAMILY}:
        raise ValueError("group-normalized rigid family matrix changed")
    mapped = {
        RIGID_FAMILY: regrets[RIGID_FAMILY],
        ROLLOUT_VALUE_FAMILY: regrets[GROUP_NORMALIZED_RIGID_FAMILY],
    }
    result = deepcopy(select_rollout_value_family(mapped))
    result["protocol"] = "frozen-six-city-group-normalized-rigid-selector-v1"
    result["decision"] = (
        "promote_group_normalized_rigid_core"
        if result["passed"]
        else "group_normalized_rigid_core_failed_no_promotion"
    )
    result["selected_family"] = (
        GROUP_NORMALIZED_RIGID_FAMILY if result["passed"] else None
    )
    result["candidate_summary"]["family"] = GROUP_NORMALIZED_RIGID_FAMILY
    result["candidate_summary"]["label"] = (
        "Causal rigid core with matched-group regret normalization"
    )
    return result
