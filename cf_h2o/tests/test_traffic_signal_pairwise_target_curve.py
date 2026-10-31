import pytest

from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_target_curve import (
    CURVE_FAMILIES,
    TARGET_GROUP_BUDGETS,
    summarize_pairwise_target_curve,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    RIGID_FAMILY,
    TARGETS,
)


def _curve(pairwise_macros: list[float]) -> dict:
    values = {}
    for budget, pairwise in zip(
        TARGET_GROUP_BUDGETS,
        pairwise_macros,
        strict=True,
    ):
        values[budget] = {
            family: {target: 0.30 for target in TARGETS}
            for family in CURVE_FAMILIES
        }
        values[budget][PAIRWISE_PREFERENCE_FAMILY] = {
            target: pairwise for target in TARGETS
        }
    return values


def test_pairwise_target_curve_passes_frozen_gate() -> None:
    result = summarize_pairwise_target_curve(
        _curve([0.40, 0.34, 0.30, 0.27, 0.25])
    )
    assert result["passed"] is True
    assert result["pairwise_budget_regret_spearman"] == pytest.approx(-1.0)


def test_pairwise_target_curve_rejects_nonmonotone_endpoint() -> None:
    result = summarize_pairwise_target_curve(
        _curve([0.28, 0.25, 0.24, 0.27, 0.26])
    )
    assert result["passed"] is False
    assert result["stage_gate"][
        "pairwise_budget_regret_spearman_at_most_minus_0_8"
    ] is False


def test_pairwise_target_curve_requires_exact_budget_set() -> None:
    values = _curve([0.40, 0.34, 0.30, 0.27, 0.25])
    values.pop(8)
    with pytest.raises(ValueError, match="target-budget set"):
        summarize_pairwise_target_curve(values)


def test_pairwise_target_curve_rejects_single_city_harm() -> None:
    values = _curve([0.40, 0.34, 0.30, 0.27, 0.25])
    values[60][PAIRWISE_PREFERENCE_FAMILY][TARGETS[0]] = 0.37
    result = summarize_pairwise_target_curve(values)
    assert result["passed"] is False
    assert result["budget_rows"][60]["pairwise_max_absolute_regression"] > 0.05


def test_pairwise_target_curve_rejects_family_drift() -> None:
    values = _curve([0.40, 0.34, 0.30, 0.27, 0.25])
    values[16].pop(RIGID_FAMILY)
    with pytest.raises(ValueError, match="family set changed"):
        summarize_pairwise_target_curve(values)
