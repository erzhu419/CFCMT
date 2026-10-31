import pytest

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    TARGETS,
    summarize_saltlake_confirmation,
)


def _regrets(candidate: tuple[float, float]) -> dict[str, dict[str, float]]:
    return {
        target: {
            GROUP_NORMALIZED_RIGID_FAMILY: 0.20,
            PAIRWISE_PREFERENCE_FAMILY: value,
        }
        for target, value in zip(TARGETS, candidate, strict=True)
    }


def test_saltlake_confirmation_passes_only_global_two_network_gate() -> None:
    result = summarize_saltlake_confirmation(_regrets((0.15, 0.16)))

    assert result["passed"] is True
    assert result["improved_network_count"] == 2
    assert result["macro_relative_improvement_pct"] == pytest.approx(22.5)


def test_saltlake_confirmation_rejects_single_network_improvement() -> None:
    result = summarize_saltlake_confirmation(_regrets((0.10, 0.21)))

    assert result["passed"] is False
    assert result["gate"]["both_networks_strictly_improve"] is False


def test_saltlake_confirmation_rejects_changed_target_set() -> None:
    with pytest.raises(ValueError, match="target set changed"):
        summarize_saltlake_confirmation({TARGETS[0]: _regrets((0.1, 0.1))[TARGETS[0]]})
