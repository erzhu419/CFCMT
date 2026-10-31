import pytest

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
    select_group_normalized_rigid_family,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS


def _matrix(candidate: float) -> dict[str, dict[str, float]]:
    return {
        RIGID_FAMILY: {target: 0.30 for target in TARGETS},
        GROUP_NORMALIZED_RIGID_FAMILY: {target: candidate for target in TARGETS},
    }


def test_group_normalized_selector_promotes_only_on_frozen_gate():
    result = select_group_normalized_rigid_family(_matrix(0.25))
    assert result["passed"] is True
    assert result["selected_family"] == GROUP_NORMALIZED_RIGID_FAMILY

    result = select_group_normalized_rigid_family(_matrix(0.28))
    assert result["passed"] is False
    assert result["selected_family"] is None


def test_group_normalized_selector_rejects_family_drift():
    matrix = _matrix(0.25)
    matrix["extra"] = {target: 0.0 for target in TARGETS}
    with pytest.raises(ValueError, match="matrix changed"):
        select_group_normalized_rigid_family(matrix)
