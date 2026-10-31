import numpy as np

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    RESULT_PROTOCOL,
    SOURCE_SEEDS,
    _fold_groups,
    _policy_values,
    _select_candidate,
    aggregate_results,
)


def test_fold_groups_partition_b25_without_overlap() -> None:
    groups = tuple(f"g{index:02d}" for index in range(25))
    folds = _fold_groups(groups)
    heldout = [group for fold in folds for group in fold["heldout_groups"]]
    assert len(folds) == 5
    assert all(len(fold["heldout_groups"]) ==  5 for fold in folds)
    assert sorted(heldout) == list(groups)
    assert len(set(heldout)) == len(groups)


def test_selector_uses_target_only_fallback_below_minimum_gain() -> None:
    selected = _select_candidate(
        {"source|block": [-0.01, 0.01]},
        target_values=[0.0, 0.0],
        minimum_gain=0.001,
    )
    assert selected["candidate"] == "source|block"
    assert selected["gain_vs_target_only"] == 0.0
    assert selected["admitted"] is False


def test_policy_values_choose_lowest_score_within_each_group() -> None:
    actual = np.asarray([0.0, -0.4, 0.0, 0.2])
    score = np.asarray([0.0, 0.3, 0.0, -0.1])
    groups = np.asarray(["a", "a", "b", "b"])
    references = np.asarray([True, False, True, False])
    assert np.array_equal(
        _policy_values(actual, score, groups, references),
        np.asarray([0.0, 0.2]),
    )


def _result(city: str, effect: float, placebo_effect: float) -> dict:
    target = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    selected = {str(seed): -0.2 + effect for seed in SOURCE_SEEDS}
    placebo = {
        str(seed): selected[str(seed)] - placebo_effect for seed in SOURCE_SEEDS
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "method": {
            "selector": {
                "candidate": "source|queue_service",
                "admitted": True,
            }
        },
        "arm_summaries": {
            "target_only_mechanism": {"seed_values": target},
            "selected_source_mechanism_prior": {"seed_values": selected},
            "selected_matched_placebo_prior": {"seed_values": placebo},
            "forced_source_mechanism_prior": {"seed_values": selected},
        },
    }


def test_aggregate_passes_consistent_source_improvement() -> None:
    result = aggregate_results(
        [_result(city, -0.02, -0.01) for city in EXPECTED_CITY_GROUPS]
    )
    assert result["development_gate"]["passed"] is True
    assert result["selected_effect_vs_target_only"]["mean"] < 0.0
    assert result["selected_effect_vs_matched_placebo"]["mean"] < 0.0
