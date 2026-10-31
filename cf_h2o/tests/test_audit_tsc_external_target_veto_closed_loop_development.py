from scripts.cluster.audit_tsc_external_target_veto_closed_loop_development import (
    select_common_candidate,
)


def _candidate(key: str, cooldown: int) -> dict:
    return {
        "key": key,
        "runtime_policy": "causal_group_normalized_rigid_advantage_contrast_hierarchical_guard",
        "coordination_mode": "direct",
        "cooldown_intervals": cooldown,
        "max_simultaneous_overrides": 1,
        "execution_trust_region": {"global_cooldown_intervals": cooldown},
    }


def _rows() -> list[dict]:
    rows = []
    safety = {
        "starting_teleports": 0,
        "ending_teleports": 0,
        "collision_events": 0,
        "collision_incidents": 0,
    }
    for seed in range(8):
        for key, active_wait in (("good", 90.0), ("bad", 102.0)):
            rows.append(
                {
                    "city": "jinan",
                    "scenario": "j",
                    "seed": seed,
                    "candidate": key,
                    "target_veto_active": True,
                    "method_mean_waiting_time": active_wait,
                    "phase_mean_waiting_time": 100.0,
                    "relative_delta": (active_wait - 100.0) / 100.0,
                    "method_mean_travel_time": 200.0,
                    "phase_mean_travel_time": 200.0,
                    "executed_overrides": 2,
                    "rejected_long_horizon_veto": 1,
                    "method_safety": safety,
                    "phase_safety": safety,
                }
            )
            fallback_wait = 100.0 if key == "good" else 100.001
            rows.append(
                {
                    "city": "los_angeles",
                    "scenario": "la",
                    "seed": seed,
                    "candidate": key,
                    "target_veto_active": False,
                    "method_mean_waiting_time": fallback_wait,
                    "phase_mean_waiting_time": 100.0,
                    "relative_delta": (fallback_wait - 100.0) / 100.0,
                    "method_mean_travel_time": 200.0,
                    "phase_mean_travel_time": 200.0,
                    "executed_overrides": 0,
                    "rejected_long_horizon_veto": 1,
                    "method_safety": safety,
                    "phase_safety": safety,
                }
            )
    return rows


def test_selector_requires_active_improvement_and_fallback_exact_abstention() -> None:
    result = select_common_candidate(
        candidates=[_candidate("good", 29), _candidate("bad", 11)],
        paired_rows=_rows(),
        city_scenarios={"jinan": ["j"], "los_angeles": ["la"]},
        active_cities=["jinan"],
        fallback_cities=["los_angeles"],
        seeds=list(range(8)),
        selection_rule={
            "active_city": {
                "minimum_city_mean_relative_improvement": 0.005,
                "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.01,
                "maximum_seed_level_city_mean_relative_regression": 0.01,
                "minimum_improved_seed_fraction": 0.5,
                "minimum_executed_interventions_per_scenario_seed": 1,
            },
            "fallback_city": {
                "executed_interventions_must_equal": 0,
                "mean_waiting_time_must_equal_phase_atol": 1e-12,
                "mean_travel_time_must_equal_phase_atol": 1e-12,
                "safety_counts_must_equal_phase": True,
            },
        },
        bootstrap={"replicates": 1000, "seed": 19},
    )

    assert result["decision"] == "freeze_common_target_veto_candidate"
    assert result["selected_config"]["key"] == "good"
    assert result["feasible_common_candidate_count"] == 1
    assert result["selected"]["city_summaries"]["los_angeles"]["gates"][
        "waiting_time_exact_safe_abstention"
    ]
