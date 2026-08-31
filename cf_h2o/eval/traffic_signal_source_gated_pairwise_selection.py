"""Frozen selector for source-city validated pairwise expert gating."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY
from cf_h2o.eval.traffic_signal_rollout_value_selection import (
    ROLLOUT_VALUE_FAMILY,
    select_rollout_value_family,
)


SOURCE_GATED_PAIRWISE_FAMILY = "cfcmt_source_gated_pairwise_advantage"
EVALUATION_FAMILIES = (SOURCE_GATED_PAIRWISE_FAMILY,)


def select_source_gated_pairwise_family(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != {RIGID_FAMILY, SOURCE_GATED_PAIRWISE_FAMILY}:
        raise ValueError("source-gated pairwise family matrix changed")
    mapped = {
        RIGID_FAMILY: regrets[RIGID_FAMILY],
        ROLLOUT_VALUE_FAMILY: regrets[SOURCE_GATED_PAIRWISE_FAMILY],
    }
    result = deepcopy(select_rollout_value_family(mapped))
    result["protocol"] = "frozen-six-city-source-gated-pairwise-selector-v1"
    result["decision"] = (
        "promote_source_gated_pairwise_core"
        if result["passed"]
        else "source_gated_pairwise_core_failed_no_promotion"
    )
    result["selected_family"] = (
        SOURCE_GATED_PAIRWISE_FAMILY if result["passed"] else None
    )
    result["candidate_summary"]["family"] = SOURCE_GATED_PAIRWISE_FAMILY
    result["candidate_summary"]["label"] = (
        "Source-city OOF gated rigid/pairwise causal core"
    )
    return result
