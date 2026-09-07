from cf_h2o.eval.traffic_signal_direct_rollout_value_selection import (
    DIRECT_ROLLOUT_VALUE_FAMILY,
    select_direct_rollout_value_family,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS


def test_direct_rollout_value_selector_preserves_frozen_single_family_gate():
    result = select_direct_rollout_value_family(
        {
            RIGID_FAMILY: {target: 0.40 for target in TARGETS},
            DIRECT_ROLLOUT_VALUE_FAMILY: {target: 0.34 for target in TARGETS},
        }
    )
    assert result["passed"] is True
    assert result["selected_family"] == DIRECT_ROLLOUT_VALUE_FAMILY
    assert result["candidate_summary"]["family"] == DIRECT_ROLLOUT_VALUE_FAMILY
