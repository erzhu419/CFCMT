from cf_h2o.eval.traffic_signal_boosted_rollout_value_selection import (
    BOOSTED_ROLLOUT_VALUE_FAMILY,
    select_boosted_rollout_value_family,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS


def test_boosted_rollout_value_selector_preserves_frozen_single_family_gate():
    result = select_boosted_rollout_value_family(
        {
            RIGID_FAMILY: {target: 0.40 for target in TARGETS},
            BOOSTED_ROLLOUT_VALUE_FAMILY: {
                target: 0.34 for target in TARGETS
            },
        }
    )
    assert result["passed"] is True
    assert result["selected_family"] == BOOSTED_ROLLOUT_VALUE_FAMILY
