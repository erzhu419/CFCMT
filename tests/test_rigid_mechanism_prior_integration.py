import numpy as np

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    RESULT_PROTOCOL,
    aggregate_results,
    residual_corrected_rigid_score,
)


def test_residual_correction_recovers_rigid_exactly_for_source_null() -> None:
    rigid = np.asarray([0.0, -0.3, 0.0, 0.2])
    mechanism = np.asarray([0.0, -0.1, 0.0, 0.4])
    references = np.asarray([True, False, True, False])
    corrected = residual_corrected_rigid_score(
        rigid, mechanism.copy(), mechanism, references
    )
    assert np.array_equal(corrected, rigid)


def test_residual_correction_adds_only_source_induced_delta() -> None:
    corrected = residual_corrected_rigid_score(
        np.asarray([0.0, -0.2]),
        np.asarray([0.0, -0.5]),
        np.asarray([0.0, -0.1]),
        np.asarray([True, False]),
    )
    assert np.allclose(corrected, np.asarray([0.0, -0.6]))


def _result(
    city: str,
    *,
    target_effect: float,
    selected_placebo_effect: float,
    matched_placebo_effect: float,
) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): rigid[str(seed)] + target_effect for seed in SOURCE_SEEDS
    }
    selected_placebo = {
        str(seed): source[str(seed)] - selected_placebo_effect
        for seed in SOURCE_SEEDS
    }
    matched_placebo = {
        str(seed): source[str(seed)] - matched_placebo_effect
        for seed in SOURCE_SEEDS
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "method": {"selector": {"candidate": "source|spillback", "admitted": True}},
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "forced_source_corrected_rigid": {"seed_values": source},
            "selected_placebo_corrected_rigid": {
                "seed_values": selected_placebo
            },
            "same_candidate_placebo_corrected_rigid": {
                "seed_values": matched_placebo
            },
        },
    }


def test_aggregate_passes_consistent_source_specific_improvement() -> None:
    result = aggregate_results(
        [
            _result(
                city,
                target_effect=-0.02,
                selected_placebo_effect=-0.01,
                matched_placebo_effect=-0.01,
            )
            for city in EXPECTED_CITY_GROUPS
        ]
    )
    assert result["development_gate"]["passed"] is True
    assert result["development_gate"]["inference_unit"] == "city"


def test_aggregate_rejects_effect_not_separated_from_matched_placebo() -> None:
    result = aggregate_results(
        [
            _result(
                city,
                target_effect=-0.02,
                selected_placebo_effect=0.0,
                matched_placebo_effect=0.0,
            )
            for city in EXPECTED_CITY_GROUPS
        ]
    )
    assert result["development_gate"]["passed"] is False
    assert (
        result["development_gate"]["checks"][
            "city_bootstrap_upper_vs_same_candidate_placebo_below_zero"
        ]
        is False
    )
