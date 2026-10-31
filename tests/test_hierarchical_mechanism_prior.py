from copy import deepcopy

import numpy as np

from cf_h2o.eval.traffic_signal_hierarchical_mechanism_prior import (
    PRIOR_PROTOCOL,
    RESULT_PROTOCOL,
    _MetaFold,
    _RobustPrior,
    _candidate_key,
    _fit_from_statistics,
    _robust_prior,
    _select_meta_candidate,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    RIDGE_L2,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    _SourceStatistics,
)


def _statistics(beta: np.ndarray) -> _SourceStatistics:
    width = int(beta.size)
    normal = np.eye(width)
    return _SourceStatistics(
        second_moment=np.ones(width),
        normal=normal,
        right=np.asarray(beta, dtype=float),
        placebo_right=-np.asarray(beta, dtype=float),
    )


def test_robust_prior_uses_equal_city_median_and_bounded_precision() -> None:
    statistics = {
        "city_a": _statistics(np.asarray([0.0, 1.0, 2.0, 4.0])),
        "city_b": _statistics(np.asarray([1.0, 2.0, 4.0, 8.0])),
        "city_c": _statistics(np.asarray([20.0, 3.0, 6.0, 12.0])),
    }
    prior = _robust_prior(
        statistics,
        np.ones(4),
        {
            "queue_service": np.asarray([0, 1]),
            "spillback": np.asarray([2, 3]),
        },
        placebo=False,
    )
    expected = np.median(
        np.vstack(
            [
                row.right / (1.0 + RIDGE_L2)
                for row in statistics.values()
            ]
        ),
        axis=0,
    )

    assert np.allclose(prior.center, expected)
    assert np.all(prior.precision >= 0.25)
    assert np.all(prior.precision <= 2.0)
    assert prior.diagnostics["source_city_count"] == 3


def test_zero_strength_is_exact_target_ridge_independent_of_prior() -> None:
    normal = np.asarray([[2.0, 0.3], [0.3, 1.5]])
    right = np.asarray([0.5, -0.25])
    prior = _RobustPrior(
        center=np.asarray([100.0, -100.0]),
        precision=np.asarray([2.0, 0.25]),
        diagnostics={},
    )
    actual = _fit_from_statistics(
        normal,
        right,
        prior=prior,
        block_indices=(0, 1),
        strength=0.0,
    )
    expected_normal = normal.copy()
    expected_normal.flat[::3] += RIDGE_L2
    expected = np.linalg.solve(expected_normal, right)

    assert np.array_equal(actual, expected)


def _fold(city: str, *, passing: bool) -> _MetaFold:
    key = _candidate_key("queue_service", 0.005)
    rejected = _candidate_key("spillback", 0.005)
    rigid = np.asarray([1.0, 1.0])
    source = np.asarray([0.99, 0.99]) if passing else rigid.copy()
    return _MetaFold(
        city=city,
        rigid_values=rigid,
        source_values={key: source, rejected: np.asarray([1.1, 1.1])},
        placebo_values={
            key: np.asarray([1.0, 1.0]),
            rejected: np.asarray([1.0, 1.0]),
        },
        zero_values={
            key: np.asarray([1.005, 1.005]),
            rejected: np.asarray([1.0, 1.0]),
        },
        source_interventions={key: 1, rejected: 1},
        source_cities=("source_a", "source_b"),
        prior_diagnostics={},
    )


def test_meta_selector_admits_only_source_specific_candidate() -> None:
    selector = _select_meta_candidate(
        [_fold(city, passing=True) for city in EXPECTED_CITY_GROUPS[:6]]
    )

    assert selector["admitted"] is True
    assert selector["candidate"] == _candidate_key("queue_service", 0.005)
    assert selector["selected_diagnostics"]["checks"][
        "mean_gain_vs_placebo"
    ] is True


def test_meta_selector_rejects_candidate_without_minimum_gain() -> None:
    selector = _select_meta_candidate(
        [_fold(city, passing=False) for city in EXPECTED_CITY_GROUPS[:6]]
    )

    assert selector["admitted"] is False
    assert selector["eligible_candidate_count"] == 0


def _result(city: str, *, admitted: bool) -> dict:
    rigid = {str(seed): 1.0 for seed in SOURCE_SEEDS}
    source = {
        str(seed): value - (0.02 if admitted else 0.0)
        for seed, value in rigid.items()
    }
    placebo = {
        str(seed): value + (0.01 if admitted else 0.0)
        for seed, value in source.items()
    }
    zero = {
        str(seed): value + (0.015 if admitted else 0.0)
        for seed, value in source.items()
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "prior_protocol": PRIOR_PROTOCOL,
            "selector": {"admitted": admitted},
        },
        "information_budget": {"target_meta_selection_label_count": 0},
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_hierarchical_source_prior": {"seed_values": source},
            "selected_matched_placebo_prior": {"seed_values": placebo},
            "selected_zero_centred_prior": {"seed_values": zero},
            "forced_hierarchical_source_prior": {"seed_values": source},
        },
    }


def test_aggregate_passes_two_source_specific_admissions_and_exact_fallback() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v152a_hierarchical_prior_closed_loop_development"
    )


def test_aggregate_rejects_zero_centred_tie() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]
    for index in range(2):
        rows[index]["arm_summaries"]["selected_zero_centred_prior"] = deepcopy(
            rows[index]["arm_summaries"]["selected_hierarchical_source_prior"]
        )

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["checks"][
        "mean_selected_effect_beats_zero_centred_prior"
    ] is False
