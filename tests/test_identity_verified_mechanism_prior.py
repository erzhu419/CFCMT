import numpy as np

from cf_h2o.eval.traffic_signal_identity_verified_mechanism_prior import (
    RESULT_PROTOCOL,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    select_crossfitted_identity_verified_candidate,
    select_identity_verified_candidate,
)


def test_identity_selector_rejects_target_gain_reproduced_by_placebo() -> None:
    selector = select_identity_verified_candidate(
        {"source|queue_service": [-0.2, -0.2, -0.2]},
        {"source|queue_service": [-0.2, -0.2, -0.2]},
        target_values=[0.0, 0.0, 0.0],
        minimum_target_gain=0.01,
        minimum_identity_gain=0.01,
    )
    assert np.isclose(selector["gain_vs_target_only"], 0.2)
    assert selector["gain_vs_matched_placebo"] == 0.0
    assert selector["admitted"] is False


def test_identity_selector_admits_source_specific_gain() -> None:
    selector = select_identity_verified_candidate(
        {
            "a|spillback": [-0.3, -0.2, -0.1],
            "b|spillback": [-0.1, -0.1, -0.1],
        },
        {
            "a|spillback": [0.0, 0.0, 0.0],
            "b|spillback": [-0.1, -0.1, -0.1],
        },
        target_values=[0.0, 0.0, 0.0],
        minimum_target_gain=0.01,
        minimum_identity_gain=0.01,
    )
    assert selector["candidate"] == "a|spillback"
    assert selector["admitted"] is True


def test_crossfitted_selector_rejects_full_sample_winner_with_unstable_folds() -> None:
    selector = select_crossfitted_identity_verified_candidate(
        {"source|queue_service": [[-2.0], [-2.0], [-2.0], [2.0], [2.0]]},
        {"source|queue_service": [[0.0], [0.0], [0.0], [0.0], [0.0]]},
        target_fold_values=[[0.0], [0.0], [0.0], [0.0], [0.0]],
        minimum_target_gain=0.01,
        minimum_identity_gain=0.01,
    )

    assert selector["full_data_candidate_admitted"] is True
    assert selector["crossfitted_selection_audit"]["passed"] is False
    assert selector["admitted"] is False


def test_crossfitted_selector_admits_stable_source_gain() -> None:
    selector = select_crossfitted_identity_verified_candidate(
        {"source|queue_service": [[-0.2], [-0.2], [-0.2], [-0.2], [-0.2]]},
        {"source|queue_service": [[0.0], [0.0], [0.0], [0.0], [0.0]]},
        target_fold_values=[[0.0], [0.0], [0.0], [0.0], [0.0]],
        minimum_target_gain=0.01,
        minimum_identity_gain=0.01,
    )

    assert selector["crossfitted_selection_audit"]["passed"] is True
    assert selector["admitted"] is True


def _result(city: str, *, admitted: bool, target_effect: float, placebo_effect: float) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): rigid[str(seed)] + target_effect for seed in SOURCE_SEEDS
    }
    placebo = {
        str(seed): source[str(seed)] - placebo_effect for seed in SOURCE_SEEDS
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "method": {"selector": {"candidate": "source|spillback", "admitted": admitted}},
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {"seed_values": placebo},
        },
    }


def test_identity_aggregate_passes_selective_exact_fallback() -> None:
    rows = []
    for index, city in enumerate(EXPECTED_CITY_GROUPS):
        rows.append(
            _result(
                city,
                admitted=index < 2,
                target_effect=-0.02 if index < 2 else 0.0,
                placebo_effect=-0.01 if index < 2 else 0.0,
            )
        )
    result = aggregate_results(rows)
    assert result["development_gate"]["passed"] is True
    assert np.isclose(
        result["selected_effect_vs_rigid_target_only"]["city_unit"]["mean"],
        -0.04 / 7.0,
    )


def test_identity_aggregate_rejects_admitted_city_that_loses_to_placebo() -> None:
    rows = []
    for index, city in enumerate(EXPECTED_CITY_GROUPS):
        rows.append(
            _result(
                city,
                admitted=index < 2,
                target_effect=-0.02 if index < 2 else 0.0,
                placebo_effect=(0.01 if index == 0 else -0.01) if index < 2 else 0.0,
            )
        )
    result = aggregate_results(rows)
    assert result["development_gate"]["passed"] is False
    assert (
        result["development_gate"]["checks"][
            "every_admitted_city_beats_matched_placebo"
        ]
        is False
    )
