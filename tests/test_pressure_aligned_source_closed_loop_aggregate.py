from cf_h2o.eval.traffic_signal_pressure_aligned_source_closed_loop import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_pressure_aligned_source_closed_loop_aggregate import (
    OUTCOME_FIELDS,
    POLICY_ARMS,
    aggregate_results,
    expected_identities,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
)


def _config() -> dict:
    return {
        "protocol": "tsc-v150m-pressure-aligned-source-closed-loop-development-v1",
        "city_scenarios": {"admitted": ["s1"], "rejected": ["s2"]},
        "offline_admitted_source_cities": ["admitted"],
        "smoke": {
            "seeds": [74417],
            "city_scenarios": {"admitted": ["s1"], "rejected": ["s2"]},
            "matrix_size": 8,
        },
        "full": {"seeds": [74417, 85523, 96631], "matrix_size": 24},
        "primary_metric": "mean_tripinfo_waiting_time",
        "development_gate": {
            "smoke_maximum_primary_relative_regression": 0.10,
            "smoke_maximum_p90_relative_regression": 0.20,
            "smoke_maximum_completion_ratio_drop": 0.05,
            "minimum_admitted_cities_nondegrading": 1,
            "maximum_primary_relative_regression": 0.01,
            "maximum_p90_relative_regression": 0.05,
            "maximum_completion_ratio_drop": 0.01,
        },
    }


def _result(city: str, scenario: str, seed: int, arm: str) -> dict:
    admitted = city == "admitted"
    costs = {
        "phase_pressure": 80.0,
        "rigid_target_only": 100.0,
        "selected_source_corrected_rigid": 90.0 if admitted else 100.0,
        "same_capacity_placebo_corrected_rigid": 95.0 if admitted else 100.0,
    }
    metrics = {name: 0 for name in OUTCOME_FIELDS}
    metrics.update(
        {
            "mean_tripinfo_waiting_time": costs[arm],
            "p90_tripinfo_waiting_time": costs[arm] * 2.0,
            "completion_ratio": 0.90,
            "departed": 100,
            "arrived": 90,
        }
    )
    changed = int(admitted and arm == "selected_source_corrected_rigid") * 2
    placebo_changed = int(admitted and arm == "same_capacity_placebo_corrected_rigid")
    if arm == "same_capacity_placebo_corrected_rigid":
        changed = placebo_changed
    return {
        "protocol": RESULT_PROTOCOL,
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "arm": arm,
        "originator_diagnostics": (
            None
            if arm == "phase_pressure"
            else {
                "decision_count": 10,
                "candidate_score_constraint": PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
                "source_changed_rigid_action_count": changed,
                "source_changed_to_reference_action_count": changed,
                "source_changed_off_reference_action_count": 0,
            }
        ),
        "metrics": metrics,
    }


def test_smoke_uses_relative_effects_and_requires_real_intervention() -> None:
    config = _config()
    rows = [
        _result(city, scenario, seed, arm)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]

    result = aggregate_results(rows, config=config, stage="smoke")

    assert len(rows) == len(POLICY_ARMS) * 2
    assert result["development_gate"]["passed"] is True
    assert result["admitted_city_macro_primary_relative_change_vs_rigid"] == -0.1
    assert result["source_changed_rigid_action_count"] == 2
    assert result["exact_fallback_failures"] == []


def test_smoke_rejects_any_applied_off_reference_source_action() -> None:
    config = _config()
    rows = [
        _result(city, scenario, seed, arm)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]
    source = next(
        row
        for row in rows
        if row["city"] == "admitted"
        and row["arm"] == "selected_source_corrected_rigid"
    )
    source["originator_diagnostics"]["source_changed_to_reference_action_count"] = 1
    source["originator_diagnostics"]["source_changed_off_reference_action_count"] = 1

    result = aggregate_results(rows, config=config, stage="smoke")

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["operational_checks"][
        "pressure_alignment_enforced_for_every_applied_change"
    ] is False
