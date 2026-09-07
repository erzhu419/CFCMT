from __future__ import annotations

import copy

import pytest

from cf_h2o.eval.traffic_signal_target_regime_trust_diagnostic import (
    run_cross_fitted_diagnostic,
)
from cf_h2o.traffic_signal.target_regime_trust import (
    FEATURE_NAME,
    TargetRegimeTrustModel,
    fit_target_regime_trust,
)


SEEDS = (11, 22, 33, 44, 55, 66)
SCENARIOS = ("low", "middle", "high")


def _rows():
    base_queue = {"low": 0.7, "middle": 0.9, "high": 1.2}
    base_delta = {"low": 0.01, "middle": 0.003, "high": -0.012}
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "phase_mean_queue_per_lane": base_queue[scenario] + seed * 1e-5,
            "relative_delta": base_delta[scenario] + ((seed % 3) - 1) * 0.001,
        }
        for seed in SEEDS
        for scenario in SCENARIOS
    ]


def _run(rows):
    return run_cross_fitted_diagnostic(
        rows,
        seeds=SEEDS,
        scenarios=SCENARIOS,
        l2=1.0,
        bootstrap={"replicates": 1000, "seed": 7},
        advancement_rule={
            "maximum_mean_relative_delta": 0.0,
            "maximum_bootstrap_95pct_upper_relative_delta": 0.01,
            "maximum_seed_mean_relative_regression": 0.01,
            "minimum_improved_seed_fraction": 0.5,
            "minimum_mean_correction_vs_always_cfcmt": 0.002,
            "minimum_selection_rate": 0.1,
            "maximum_selection_rate": 0.9,
        },
    )


def test_target_regime_model_round_trip_and_declared_feature():
    model = fit_target_regime_trust(
        [0.7, 0.9, 1.2, 0.72, 0.92, 1.22],
        [0.01, 0.003, -0.012, 0.011, 0.002, -0.013],
        l2=1.0,
    )
    restored = TargetRegimeTrustModel.from_dict(model.to_dict())
    assert restored.to_dict()["feature_names"] == [FEATURE_NAME]
    assert restored.coefficient < 0.0
    assert restored.trusts_local_controller(1.2)
    assert not restored.trusts_local_controller(0.7)


def test_heldout_day_feature_cannot_change_its_own_fold_prediction():
    original_rows = _rows()
    original = _run(original_rows)
    perturbed_rows = copy.deepcopy(original_rows)
    for row in perturbed_rows:
        if row["seed"] == SEEDS[0]:
            row["phase_mean_queue_per_lane"] += 1000.0
    perturbed = _run(perturbed_rows)
    original_fold = [row for row in original["rows"] if row["seed"] == SEEDS[0]]
    perturbed_fold = [row for row in perturbed["rows"] if row["seed"] == SEEDS[0]]
    assert [row["predicted_relative_delta"] for row in perturbed_fold] == pytest.approx(
        [row["predicted_relative_delta"] for row in original_fold]
    )
    assert [row["trust_local_controller"] for row in perturbed_fold] == [
        row["trust_local_controller"] for row in original_fold
    ]


def test_cross_fit_uses_nontrivial_abstention_and_passes_synthetic_gate():
    result = _run(_rows())
    assert result["feature_names"] == [FEATURE_NAME]
    assert result["selection_rate"] == pytest.approx(1.0 / 3.0)
    assert all(
        row["seed"] not in row["history_source_seeds"] for row in result["rows"]
    )
    assert result["cross_fitted_regime_gate"]["mean_relative_delta"] < 0.0
    assert result["advancement_gate"]["passed"]
