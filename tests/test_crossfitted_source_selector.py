import numpy as np

from cf_h2o.eval.traffic_signal_crossfitted_source_selector import (
    RESULT_PROTOCOL,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)


def _result(
    city: str, *, admitted: bool, target_effect: float, placebo_effect: float
) -> dict:
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
        "method": {
            "selector": {
                "candidate": "source|spillback",
                "admitted": admitted,
                "crossfitted_selection_audit": {"passed": admitted},
            }
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {"seed_values": placebo},
        },
    }


def test_crossfitted_aggregate_passes_two_source_cities_and_exact_fallback():
    rows = [
        _result(
            city,
            admitted=index < 2,
            target_effect=-0.02 if index < 2 else 0.0,
            placebo_effect=-0.01 if index < 2 else 0.0,
        )
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert np.isclose(
        result["selected_effect_vs_rigid_target_only"]["city_unit"]["mean"],
        -0.04 / 7.0,
    )


def test_crossfitted_aggregate_rejects_any_admitted_regression():
    rows = [
        _result(
            city,
            admitted=index < 2,
            target_effect=(0.01 if index == 0 else -0.02 if index == 1 else 0.0),
            placebo_effect=-0.01 if index < 2 else 0.0,
        )
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["checks"]["no_city_regresses_rigid"] is False
