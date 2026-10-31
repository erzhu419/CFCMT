"""Frozen selector for the all-action antisymmetric causal preference core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY
from cf_h2o.eval.traffic_signal_rollout_value_selection import (
    ROLLOUT_VALUE_FAMILY,
    select_rollout_value_family,
)


PAIRWISE_PREFERENCE_FAMILY = "causal_antisymmetric_pairwise_advantage"
EVALUATION_FAMILIES = (PAIRWISE_PREFERENCE_FAMILY,)


def select_pairwise_preference_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != {RIGID_FAMILY, PAIRWISE_PREFERENCE_FAMILY}:
        raise ValueError("pairwise preference family matrix changed")
    mapped = {
        RIGID_FAMILY: regrets[RIGID_FAMILY],
        ROLLOUT_VALUE_FAMILY: regrets[PAIRWISE_PREFERENCE_FAMILY],
    }
    result = deepcopy(select_rollout_value_family(mapped))
    result["protocol"] = "frozen-six-city-antisymmetric-pairwise-selector-v1"
    result["decision"] = (
        "promote_antisymmetric_pairwise_core"
        if result["passed"]
        else "antisymmetric_pairwise_core_failed_no_promotion"
    )
    result["selected_family"] = (
        PAIRWISE_PREFERENCE_FAMILY if result["passed"] else None
    )
    result["candidate_summary"]["family"] = PAIRWISE_PREFERENCE_FAMILY
    result["candidate_summary"]["label"] = (
        "All-action antisymmetric causal preference core"
    )
    return result
