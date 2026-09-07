import numpy as np

from cf_h2o.eval.traffic_signal_external_pressure_heldout_evaluation import (
    _city_gate,
    _equal_scenario_bootstrap,
)


def test_equal_scenario_bootstrap_does_not_overweight_large_scenario() -> None:
    rows = [
        {"scenario": "small", "actual_delta": -0.2},
        *(
            {"scenario": "large", "actual_delta": 0.0}
            for _ in range(20)
        ),
    ]

    result = _equal_scenario_bootstrap(rows, seed=7)

    assert np.isclose(result["observed_mean_delta_vs_phase_pressure"], -0.1)
    assert result["group_counts"] == {"large": 20, "small": 1}


def test_city_gate_requires_mean_improvement_and_interventions() -> None:
    summary = {
        "accepted_overrides": 4,
        "minimum_required_overrides": 2,
        "worst_stratum_mean_delta": -0.002,
    }
    passed = _city_gate(
        summary=summary,
        bootstrap={
            "observed_mean_delta_vs_phase_pressure": -0.02,
            "ci95": [-0.04, -0.001],
        },
    )
    failed = _city_gate(
        summary={**summary, "accepted_overrides": 0},
        bootstrap={
            "observed_mean_delta_vs_phase_pressure": 0.0,
            "ci95": [0.0, 0.0],
        },
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
