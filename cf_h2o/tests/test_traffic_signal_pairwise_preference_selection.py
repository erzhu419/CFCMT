import pytest

from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
    select_pairwise_preference_family,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS


def _matrix(candidate: float) -> dict[str, dict[str, float]]:
    return {
        RIGID_FAMILY: {target: 0.30 for target in TARGETS},
        PAIRWISE_PREFERENCE_FAMILY: {target: candidate for target in TARGETS},
    }


def test_pairwise_selector_uses_frozen_promotion_gate():
    result = select_pairwise_preference_family(_matrix(0.25))
    assert result["passed"] is True
    assert result["selected_family"] == PAIRWISE_PREFERENCE_FAMILY

    result = select_pairwise_preference_family(_matrix(0.28))
    assert result["passed"] is False
    assert result["selected_family"] is None


def test_pairwise_selector_rejects_family_drift():
    matrix = _matrix(0.25)
    matrix["extra"] = {target: 0.0 for target in TARGETS}
    with pytest.raises(ValueError, match="matrix changed"):
        select_pairwise_preference_family(matrix)
