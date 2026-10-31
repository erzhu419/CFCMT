from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (
    bootstrap_mean_interval,
    select_common_candidate,
    summarize_candidate_city,
)


RULE = {
    "minimum_city_mean_relative_improvement": 0.005,
    "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.01,
    "maximum_seed_level_city_mean_relative_regression": 0.01,
    "minimum_improved_seed_fraction": 0.5,
    "minimum_executed_interventions_per_scenario_seed": 1,
}
BOOTSTRAP = {"replicates": 1000, "seed": 123}


def _candidate(key: str, cooldown: int = 5) -> dict:
    return {
        "key": key,
        "runtime_policy": "hierarchical",
        "coordination_mode": "direct",
        "cooldown_intervals": cooldown,
        "execution_trust_region": {"key": key},
    }


def _row(city: str, scenario: str, seed: int, candidate: str, delta: float) -> dict:
    return {
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "candidate": candidate,
        "method_mean_waiting_time": 100.0 * (1.0 + delta),
        "phase_mean_waiting_time": 100.0,
        "relative_delta": delta,
        "executed_overrides": 2,
        "method_safety": {
            "starting_teleports": 0,
            "ending_teleports": 0,
            "collision_incidents": 0,
        },
        "phase_safety": {
            "starting_teleports": 0,
            "ending_teleports": 0,
            "collision_incidents": 0,
        },
    }


def test_candidate_summary_uses_equal_scenario_relative_deltas() -> None:
    rows = []
    for seed in (1, 2, 3, 4):
        rows.append(_row("jinan", "a", seed, "good", -0.04))
        rows.append(_row("jinan", "b", seed, "good", -0.02))

    summary = summarize_candidate_city(
        city="jinan",
        candidate=_candidate("good"),
        paired_rows=rows,
        scenarios=("a", "b"),
        seeds=(1, 2, 3, 4),
        selection_rule=RULE,
        bootstrap=BOOTSTRAP,
    )

    assert summary["city_mean_relative_delta"] == -0.03
    assert summary["feasible"] is True


def test_common_selector_requires_every_city_to_pass() -> None:
    candidates = [_candidate("common", 8), _candidate("jinan_only", 5)]
    rows = []
    for seed in (1, 2, 3, 4):
        rows.append(_row("jinan", "j", seed, "common", -0.03))
        rows.append(_row("los_angeles", "la", seed, "common", -0.02))
        rows.append(_row("jinan", "j", seed, "jinan_only", -0.08))
        rows.append(_row("los_angeles", "la", seed, "jinan_only", 0.02))

    selection = select_common_candidate(
        candidates=candidates,
        paired_rows=rows,
        city_scenarios={"jinan": ["j"], "los_angeles": ["la"]},
        seeds=(1, 2, 3, 4),
        selection_rule=RULE,
        bootstrap=BOOTSTRAP,
    )

    assert selection["decision"] == "freeze_common_active_hierarchical_candidate"
    assert selection["selected_config"]["key"] == "common"
    assert selection["feasible_common_candidate_count"] == 1


def test_common_selector_falls_back_without_feasible_candidate() -> None:
    rows = [
        _row(city, scenario, seed, "bad", 0.02)
        for city, scenario in (("jinan", "j"), ("los_angeles", "la"))
        for seed in (1, 2, 3, 4)
    ]
    selection = select_common_candidate(
        candidates=[_candidate("bad")],
        paired_rows=rows,
        city_scenarios={"jinan": ["j"], "los_angeles": ["la"]},
        seeds=(1, 2, 3, 4),
        selection_rule=RULE,
        bootstrap=BOOTSTRAP,
    )

    assert selection["decision"] == "retain_phase_pressure_fallback"
    assert selection["selected_config"]["key"] == "phase_pressure"


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_interval([-0.1, -0.02, 0.01, -0.04], replicates=500, seed=7)
    second = bootstrap_mean_interval([-0.1, -0.02, 0.01, -0.04], replicates=500, seed=7)

    assert first == second
