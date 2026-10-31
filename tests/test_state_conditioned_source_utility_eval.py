import numpy as np

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    RESULT_PROTOCOL,
    SOURCE_PRIOR_STRENGTH,
    UTILITY_GATE_PROTOCOL,
    _ridge_from_unscaled_statistics,
    _ridge_statistics,
    aggregate_results,
)
from cf_h2o.traffic_signal.mechanism_parameter_prior import ridge_coefficients


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
        "target_budget": 100,
        "method": {
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "selector": {
                "candidate": "per_state_source_mechanism_pool",
                "admitted": admitted,
            },
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {"seed_values": placebo},
        },
    }


def test_unscaled_source_statistics_match_direct_scaled_ridge() -> None:
    generator = np.random.default_rng(7)
    x = generator.normal(size=(60, 9))
    y = generator.normal(size=60)
    groups = np.repeat(np.arange(20), 3).astype(str)
    scale = np.exp(generator.normal(size=9))
    normal, right = _ridge_statistics(x, y, groups)

    observed = _ridge_from_unscaled_statistics(normal, right, scale)
    expected = ridge_coefficients(x / scale, y, groups, ridge_l2=0.05)

    assert np.allclose(observed, expected, rtol=1e-11, atol=1e-11)


def test_state_utility_aggregate_passes_two_identity_verified_cities() -> None:
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
    assert result["source_admission_count"] == 2
