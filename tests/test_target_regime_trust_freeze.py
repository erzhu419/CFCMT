from __future__ import annotations

from cf_h2o.eval.traffic_signal_target_regime_trust_freeze import (
    ARTIFACT_PROTOCOL,
    build_full_fit_artifact,
)


def test_full_fit_artifact_is_selective_and_does_not_use_scenario_id_as_feature():
    seeds = (1, 2, 3, 4, 5, 6)
    scenarios = ("low", "middle", "high")
    queue = {"low": 0.7, "middle": 0.9, "high": 1.2}
    delta = {"low": 0.01, "middle": 0.003, "high": -0.012}
    diagnostic = {
        "decision": "authorize_target_regime_trust_full_fit_freeze_only",
        "advance_gate": {"passed": True},
        "source_rows": [
            {
                "seed": seed,
                "scenario": scenario,
                "phase_mean_queue_per_lane": queue[scenario] + seed * 1e-4,
                "relative_delta": delta[scenario],
            }
            for seed in seeds
            for scenario in scenarios
        ],
    }
    artifact = build_full_fit_artifact(
        diagnostic,
        city="test",
        scenarios=scenarios,
        seeds=seeds,
        l2=1.0,
    )
    assert artifact["protocol"] == ARTIFACT_PROTOCOL
    assert artifact["model"]["feature_names"] == [
        "historical_phase_mean_queue_per_lane"
    ]
    assert artifact["scenario_id_is_model_feature"] is False
    assert artifact["heldout_or_validation_metrics_used"] is False
    assert artifact["scenario_history"]["high"]["trust_local_controller"] is True
    assert artifact["scenario_history"]["low"]["trust_local_controller"] is False
