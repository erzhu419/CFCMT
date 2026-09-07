from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_rigid_residual_selection import RIGID_FAMILY, TARGETS
from cf_h2o.eval.traffic_signal_selective_residual_selection import (
    CANDIDATES,
    EVALUATION_FAMILIES,
    SELECTION_FAMILIES,
    select_selective_residual_family,
)


def _regrets(default: float = 0.40) -> dict[str, dict[str, float]]:
    return {
        family: {target: default for target in TARGETS}
        for family in SELECTION_FAMILIES
    }


def test_selective_residual_selector_requires_efficacy_breadth_and_harm_gates():
    values = _regrets()
    passing = CANDIDATES[0].family
    values[passing] = {target: 0.34 for target in TARGETS}
    insufficient_breadth = CANDIDATES[1].family
    values[insufficient_breadth] = {
        target: 0.20 if index < 3 else 0.40
        for index, target in enumerate(TARGETS)
    }
    unsafe = CANDIDATES[2].family
    values[unsafe] = {
        target: 0.30 if index < 5 else 0.46
        for index, target in enumerate(TARGETS)
    }

    result = select_selective_residual_family(values)

    assert result["selected_family"] == passing
    rows = {row["family"]: row for row in result["family_summaries"]}
    assert rows[passing]["stage_gate"]["passed"] is True
    assert (
        rows[insufficient_breadth]["stage_gate"][
            "at_least_four_cities_improved"
        ]
        is False
    )
    assert rows[unsafe]["stage_gate"]["max_regression_at_most_0_05"] is False


def test_selective_residual_selector_prefers_higher_quantile_then_parsimony():
    values = _regrets()
    mobility_q50 = CANDIDATES[0].family
    mobility_q90 = CANDIDATES[2].family
    pair_q90 = CANDIDATES[5].family
    values[mobility_q50] = {target: 0.340 for target in TARGETS}
    values[mobility_q90] = {target: 0.342 for target in TARGETS}
    values[pair_q90] = {target: 0.338 for target in TARGETS}

    result = select_selective_residual_family(values)

    assert set(result["tie_pool"]) == {mobility_q50, mobility_q90, pair_q90}
    assert result["selected_family"] == mobility_q90
    assert tuple(result["candidate_order"]) == EVALUATION_FAMILIES


def test_selective_residual_selector_rejects_changed_matrix():
    values = _regrets()
    del values[RIGID_FAMILY][TARGETS[0]]
    with pytest.raises(ValueError, match="target mismatch"):
        select_selective_residual_family(values)
