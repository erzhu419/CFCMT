from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    POLICY_ARMS,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop_aggregate import (
    OUTCOME_FIELDS,
    aggregate_results,
    expected_identities,
)
from scripts.cluster.launch_tsc_state_conditioned_source_closed_loop import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    _read_json,
)


def _result(city: str, scenario: str, seed: int, arm: str, *, admitted: bool) -> dict:
    costs = {
        "phase_pressure": 101.0,
        "rigid_target_only": 100.0,
        "selected_source_corrected_rigid": 99.0 if admitted else 100.0,
        "same_capacity_placebo_corrected_rigid": 99.5 if admitted else 100.0,
    }
    metrics = {name: 0 for name in OUTCOME_FIELDS}
    metrics.update(
        {
            "mean_tripinfo_waiting_time": costs[arm],
            "p90_tripinfo_waiting_time": costs[arm] * 2,
            "mean_queue": costs[arm] / 2,
            "p90_queue": costs[arm],
            "halted_vehicle_seconds": costs[arm] * 10,
            "system_vehicle_seconds": costs[arm] * 20,
            "departed": 100,
            "arrived": 90,
            "completion_ratio": 0.9,
        }
    )
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
                "source_changed_rigid_action_count": (
                    2
                    if admitted and arm == "selected_source_corrected_rigid"
                    else 0
                ),
            }
        ),
        "metrics": metrics,
    }


def test_smoke_aggregate_requires_runtime_and_performance_for_expansion() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    admitted = set(config["offline_admitted_source_cities"])
    rows = [
        _result(city, scenario, seed, arm, admitted=city in admitted)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]

    result = aggregate_results(rows, config=config, stage="smoke")

    assert result["rollout_count"] == 16
    assert result["development_gate"]["passed"] is True
    assert result["development_gate"]["operational_passed"] is True
    assert all(result["development_gate"]["performance_checks"].values())
    assert result["development_gate"]["assessment_basis"].startswith("post_smoke")
    assert result["exact_fallback_failures"] == []


def test_smoke_operational_pass_does_not_authorize_harmful_full_matrix() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    admitted = set(config["offline_admitted_source_cities"])
    rows = [
        _result(city, scenario, seed, arm, admitted=city in admitted)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]
    source = next(
        row for row in rows
        if row["city"] == "cologne"
        and row["arm"] == "selected_source_corrected_rigid"
    )
    source["metrics"]["mean_tripinfo_waiting_time"] = 137.9

    result = aggregate_results(rows, config=config, stage="smoke")

    gate = result["development_gate"]
    assert gate["operational_passed"] is True
    assert gate["passed"] is False
    assert gate["decision"] == "reject_v150l_full_matrix_after_smoke"
    assert gate["performance_checks"]["no_admitted_city_regresses_over_one_percent"] is False


def test_full_aggregate_requires_source_and_identity_gain() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    admitted = set(config["offline_admitted_source_cities"])
    rows = [
        _result(city, scenario, seed, arm, admitted=city in admitted)
        for city, scenario, seed, arm in expected_identities(config, stage="full")
    ]

    result = aggregate_results(rows, config=config, stage="full")

    assert len(rows) == 216
    assert result["development_gate"]["passed"] is True
    assert result["admitted_city_nondegrading_count"] == 3
    assert result["admitted_city_source_minus_rigid_mean"] == -1.0
    assert result["admitted_city_source_minus_matched_placebo_mean"] == -0.5
    assert set(result["city_summaries"]) == set(config["city_scenarios"])
    assert tuple(config["arms"]) == POLICY_ARMS


def test_smoke_aggregate_accepts_baseline_collision_when_source_is_noninferior() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    admitted = set(config["offline_admitted_source_cities"])
    rows = [
        _result(city, scenario, seed, arm, admitted=city in admitted)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]
    phase = next(row for row in rows if row["arm"] == "phase_pressure")
    phase["metrics"]["collision_events"] = 2
    phase["metrics"]["collision_incidents"] = 1

    result = aggregate_results(rows, config=config, stage="smoke")

    assert result["development_gate"]["passed"] is True
    assert len(result["observed_collision_rows"]) == 1
    assert result["paired_collision_failures"] == []


def test_smoke_aggregate_rejects_source_collision_inferiority() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    admitted = set(config["offline_admitted_source_cities"])
    rows = [
        _result(city, scenario, seed, arm, admitted=city in admitted)
        for city, scenario, seed, arm in expected_identities(config, stage="smoke")
    ]
    source = next(
        row
        for row in rows
        if row["arm"] == "selected_source_corrected_rigid"
    )
    source["metrics"]["collision_events"] = 1
    source["metrics"]["collision_incidents"] = 1

    result = aggregate_results(rows, config=config, stage="smoke")

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["operational_checks"][
        "selected_source_collision_incidents_noninferior_to_rigid_and_phase"
    ] is False
    assert len(result["paired_collision_failures"]) == 1
