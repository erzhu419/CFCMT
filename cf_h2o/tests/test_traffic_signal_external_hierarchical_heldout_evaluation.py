import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _city_gate,
    _equal_seed_scenario_bootstrap,
    parse_action_group_seed,
)


def _rows(values):
    rows = []
    for simulator_seed, scenario, deltas in values:
        for index, delta in enumerate(deltas):
            rows.append(
                {
                    "simulator_seed": simulator_seed,
                    "scenario": scenario,
                    "group_id": f"{scenario}:seed{simulator_seed}:{index}:tls",
                    "selected": delta != 0.0,
                    "actual_delta": delta,
                }
            )
    return rows


def test_parse_action_group_seed_is_exact() -> None:
    assert parse_action_group_seed("la_1x4:seed80314:12:tls") == 80314
    with pytest.raises(ValueError):
        parse_action_group_seed("la_1x4:80314:12:tls")


def test_equal_seed_scenario_summary_does_not_weight_group_count() -> None:
    rows = _rows(
        [
            (11, "a", [-1.0]),
            (11, "b", [-1.0, -1.0, -1.0]),
            (22, "a", [1.0]),
            (22, "b", [-1.0]),
        ]
    )
    summary = _equal_seed_scenario_bootstrap(
        rows,
        seeds=(11, 22),
        scenarios=("a", "b"),
        replicates=100,
        seed=7,
    )
    assert np.isclose(summary["observed_mean_delta_vs_phase_pressure"], -0.5)
    assert np.isclose(summary["seed_mean_delta"]["11"], -1.0)
    assert np.isclose(summary["seed_mean_delta"]["22"], 0.0)
    assert summary["nonpositive_seed_fraction"] == 1.0


def test_equal_seed_scenario_summary_rejects_missing_stratum() -> None:
    with pytest.raises(ValueError, match="lacks stratum"):
        _equal_seed_scenario_bootstrap(
            _rows([(11, "a", [-1.0])]),
            seeds=(11,),
            scenarios=("a", "b"),
            replicates=10,
            seed=7,
        )


def test_city_gate_executes_all_frozen_thresholds() -> None:
    rows = [
        {"selected": index < 2, "actual_delta": -0.1 if index < 2 else 0.0}
        for index in range(100)
    ]
    bootstrap = {
        "observed_mean_delta_vs_phase_pressure": -0.002,
        "ci95": [-0.01, 0.005],
        "worst_seed_scenario_mean_delta": 0.005,
        "nonpositive_seed_fraction": 2.0 / 3.0,
    }
    gate = _city_gate(
        rows=rows,
        bootstrap=bootstrap,
        gate_spec={
            "minimum_equal_seed_scenario_mean_improvement": 0.001,
            "maximum_bootstrap_95pct_upper_delta": 0.01,
            "maximum_seed_scenario_mean_delta": 0.01,
            "minimum_nonpositive_seed_fraction": 2.0 / 3.0,
            "minimum_override_fraction": 0.005,
            "maximum_override_fraction": 0.5,
        },
    )
    assert gate["passed"] is True
    bootstrap["worst_seed_scenario_mean_delta"] = 0.011
    assert (
        _city_gate(rows=rows, bootstrap=bootstrap, gate_spec={
            "minimum_equal_seed_scenario_mean_improvement": 0.001,
            "maximum_bootstrap_95pct_upper_delta": 0.01,
            "maximum_seed_scenario_mean_delta": 0.01,
            "minimum_nonpositive_seed_fraction": 2.0 / 3.0,
            "minimum_override_fraction": 0.005,
            "maximum_override_fraction": 0.5,
        })["passed"]
        is False
    )
