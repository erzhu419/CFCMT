from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_pressure_pairwise_source_utility import (
    METHOD_PROTOCOL,
    RESULT_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    SOURCE_PRIOR_STRENGTH,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
)
from scripts.cluster.launch_tsc_pressure_pairwise_source_utility import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    _validate_config,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import _read_json


def _result(city: str, *, admitted: bool) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): value - (0.02 if admitted else 0.0)
        for seed, value in rigid.items()
    }
    placebo = {
        str(seed): value + (0.01 if admitted else 0.0)
        for seed, value in source.items()
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "selector": {
                "candidate": "per_state_source_mechanism_pool",
                "admitted": admitted,
                "candidate_score_constraint": (
                    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL
                ),
            },
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {"seed_values": placebo},
        },
    }


def test_pressure_pairwise_config_and_aggregate_contract() -> None:
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v150n_pressure_pairwise_closed_loop_development"
    )
