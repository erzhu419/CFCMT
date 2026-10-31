from __future__ import annotations

import numpy as np

from cf_h2o.traffic_signal.action_scaling import (
    action_group_range,
    action_scale_floor,
    domain_action_scale,
)


def test_action_group_range_preserves_small_nonzero_effects() -> None:
    values = np.asarray([0.5, 0.50001, 0.50002])

    assert np.isclose(action_scale_floor(values), 5.0002e-7)
    assert np.isclose(action_group_range(values), 2.0e-5)


def test_domain_action_scale_uses_median_positive_group_range() -> None:
    target = np.asarray([0.5, 0.50001, 0.4, 0.40003, 0.3, 0.3])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])

    assert np.isclose(domain_action_scale(target, groups), 2.0e-5)
