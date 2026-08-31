import pytest

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS
from cf_h2o.eval.traffic_signal_source_gated_pairwise_selection import (
    SOURCE_GATED_PAIRWISE_FAMILY,
    select_source_gated_pairwise_family,
)


def _matrix(candidate: float) -> dict[str, dict[str, float]]:
    return {
        RIGID_FAMILY: {target: 0.30 for target in TARGETS},
        SOURCE_GATED_PAIRWISE_FAMILY: {target: candidate for target in TARGETS},
    }


def test_source_gated_pairwise_selector_uses_frozen_gate():
    assert select_source_gated_pairwise_family(_matrix(0.25))["passed"] is True
    assert select_source_gated_pairwise_family(_matrix(0.28))["passed"] is False


def test_source_gated_pairwise_selector_rejects_family_drift():
    matrix = _matrix(0.25)
    matrix["extra"] = {target: 0.0 for target in TARGETS}
    with pytest.raises(ValueError, match="matrix changed"):
        select_source_gated_pairwise_family(matrix)
