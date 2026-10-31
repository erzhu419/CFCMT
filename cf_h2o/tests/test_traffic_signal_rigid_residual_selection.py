from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    CANDIDATES,
    EVALUATION_FAMILIES,
    RIGID_FAMILY,
    TARGETS,
    select_rigid_residual_family,
)


def _regrets(default: float = 0.40) -> dict[str, dict[str, float]]:
    return {
        family: {target: default for target in TARGETS}
        for family in EVALUATION_FAMILIES
    }


def test_rigid_residual_selector_requires_all_frozen_gates() -> None:
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

    result = select_rigid_residual_family(values)

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


def test_rigid_residual_selector_uses_parsimony_and_fails_closed() -> None:
    values = _regrets()
    first_single = CANDIDATES[0].family
    second_single = CANDIDATES[1].family
    pair = CANDIDATES[2].family
    values[first_single] = {target: 0.342 for target in TARGETS}
    values[second_single] = {target: 0.341 for target in TARGETS}
    values[pair] = {target: 0.338 for target in TARGETS}

    result = select_rigid_residual_family(values)

    assert set(result["tie_pool"]) == {first_single, second_single, pair}
    assert result["selected_family"] == first_single

    del values[RIGID_FAMILY][TARGETS[0]]
    with pytest.raises(ValueError, match="target mismatch"):
        select_rigid_residual_family(values)
