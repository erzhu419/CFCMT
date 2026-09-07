from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_stage_a_selection import (
    CANDIDATES,
    EVALUATION_FAMILIES,
    RIGID_FAMILY,
    TARGETS,
    select_stage_a_family,
)


def _regrets(default: float = 0.40) -> dict[str, dict[str, float]]:
    return {
        family: {target: default for target in TARGETS}
        for family in EVALUATION_FAMILIES
    }


def test_stage_a_requires_all_three_frozen_gates() -> None:
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

    result = select_stage_a_family(values)

    assert result["passed"] is True
    assert result["selected_family"] == passing
    rows = {row["family"]: row for row in result["family_summaries"]}
    assert rows[passing]["stage_a_gate"]["passed"] is True
    assert rows[insufficient_breadth]["stage_a_gate"]["at_least_four_cities_improved"] is False
    assert rows[unsafe]["stage_a_gate"]["max_regression_at_most_0_05"] is False


def test_stage_a_tie_prefers_fewer_mechanisms_then_frozen_order() -> None:
    values = _regrets()
    first_single = CANDIDATES[0].family
    later_single = CANDIDATES[1].family
    pair = next(candidate.family for candidate in CANDIDATES if candidate.mechanism_count == 2)
    values[first_single] = {target: 0.342 for target in TARGETS}
    values[later_single] = {target: 0.341 for target in TARGETS}
    values[pair] = {target: 0.338 for target in TARGETS}

    result = select_stage_a_family(values)

    assert set(result["tie_pool"]) == {first_single, later_single, pair}
    assert result["selected_family"] == first_single


def test_stage_a_failure_never_promotes_diagnostic_best() -> None:
    values = _regrets()
    best = CANDIDATES[4].family
    values[best] = {target: 0.38 for target in TARGETS}

    result = select_stage_a_family(values)

    assert result["passed"] is False
    assert result["selected_family"] is None
    assert result["diagnostic_best_family"] == best


def test_stage_a_fails_closed_on_missing_or_nonfinite_values() -> None:
    values = _regrets()
    del values[RIGID_FAMILY][TARGETS[0]]
    with pytest.raises(ValueError, match="target mismatch"):
        select_stage_a_family(values)

    values = _regrets()
    values[CANDIDATES[0].family][TARGETS[0]] = float("nan")
    with pytest.raises(ValueError, match="invalid regret"):
        select_stage_a_family(values)
