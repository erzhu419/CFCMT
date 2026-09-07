import copy

import pytest

import cf_h2o.eval.traffic_signal_multicity_uniform_source_aggregate as aggregate
import cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble as v143


def _result(city: str, *, uniform: float = -0.03) -> dict:
    sources = [value for value in v143.EXPECTED_CITY_GROUPS if value != city]
    scenario = f"{city}_scenario"
    return {
        "protocol": v143.RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": v143.TARGET_BUDGET,
        "heldout_target_city_scenarios": [scenario],
        "source_city_groups": sources,
        "source_scenarios_by_group": {
            source: [f"{source}_scenario"] for source in sources
        },
        "information_budget": {
            "evaluation_groups_used_for_fit_or_selection": False
        },
        "arm_summaries": {
            "phase_pressure": {"mean": 0.0},
            "target_only_causal": {"mean": 0.0},
            "pooled_h2oplus": {"mean": 0.0},
            "uniform_cfcmt": {"mean": uniform},
            "matched_source_placebo": {"mean": 0.0},
        },
        "inputs": {"fit": "same", "source": "same"},
    }


def test_v143_aggregate_passes_consistent_multicity_gain() -> None:
    result = aggregate.aggregate_multicity_uniform_source_results(
        [_result(city) for city in v143.EXPECTED_CITY_GROUPS]
    )
    assert result["development_gate"]["passed"] is True
    assert all(result["development_gate"]["comparator_pass"].values())


def test_v143_aggregate_rejects_large_city_regression() -> None:
    rows = [_result(city) for city in v143.EXPECTED_CITY_GROUPS]
    rows[0] = _result(v143.EXPECTED_CITY_GROUPS[0], uniform=0.02)
    result = aggregate.aggregate_multicity_uniform_source_results(rows)
    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["comparator_pass"]["target_only_causal"] is False


def test_v143_aggregate_rejects_target_scenario_leakage() -> None:
    rows = [_result(city) for city in v143.EXPECTED_CITY_GROUPS]
    bad = copy.deepcopy(rows[0])
    source = bad["source_city_groups"][0]
    bad["source_scenarios_by_group"][source].append(
        bad["heldout_target_city_scenarios"][0]
    )
    rows[0] = bad
    with pytest.raises(ValueError, match="information boundary"):
        aggregate.aggregate_multicity_uniform_source_results(rows)
