from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS
from cf_h2o.eval.traffic_signal_rollout_value_selection import (
    ROLLOUT_VALUE_FAMILY,
    select_rollout_value_family,
)


def test_rollout_value_selector_requires_all_frozen_gates() -> None:
    values = {
        RIGID_FAMILY: {target: 0.40 for target in TARGETS},
        ROLLOUT_VALUE_FAMILY: {target: 0.34 for target in TARGETS},
    }
    result = select_rollout_value_family(values)
    assert result["passed"] is True
    assert result["selected_family"] == ROLLOUT_VALUE_FAMILY

    values[ROLLOUT_VALUE_FAMILY][TARGETS[-1]] = 0.46
    result = select_rollout_value_family(values)
    assert result["passed"] is False
    assert (
        result["candidate_summary"]["stage_gate"][
            "max_regression_at_most_0_05"
        ]
        is False
    )


def test_rollout_value_selector_rejects_changed_target_matrix() -> None:
    values = {
        RIGID_FAMILY: {target: 0.40 for target in TARGETS},
        ROLLOUT_VALUE_FAMILY: {target: 0.34 for target in TARGETS},
    }
    del values[ROLLOUT_VALUE_FAMILY][TARGETS[0]]
    with pytest.raises(ValueError, match="target mismatch"):
        select_rollout_value_family(values)
