from __future__ import annotations

from cf_h2o.eval.traffic_signal_external_target_regime_trust_validation import (
    all_rollout_identities,
    frozen_regime_decision,
)
from cf_h2o.traffic_signal.target_regime_trust import fit_target_regime_trust


def _protocol():
    return {
        "development_evidence": {"seeds": [1, 2, 3, 4, 5, 6]},
        "validation": {
            "seeds": [101, 102],
            "city_scenarios": {"jinan": ["high", "low"], "la": ["la"]},
            "scenario_gate": {"high": True, "low": False, "la": False},
            "candidate": {
                "key": "cfcmt_target_regime_trust_cd450",
                "runtime_policy": "causal_group_normalized_rigid_advantage_contrast_hierarchical_guard",
                "coordination_mode": "direct",
                "cooldown_intervals": 44,
                "max_simultaneous_overrides": 1,
            },
        },
    }


def _artifact_model():
    model = fit_target_regime_trust(
        [0.7, 0.9, 1.2, 0.72, 0.92, 1.22],
        [0.01, 0.003, -0.012, 0.011, 0.002, -0.013],
        l2=1.0,
    )
    history = {}
    for scenario, feature in (("high", 1.21), ("low", 0.71)):
        prediction = model.predict_relative_delta(feature)
        history[scenario] = {
            "historical_phase_mean_queue_per_lane": feature,
            "predicted_relative_delta": prediction,
            "trust_local_controller": prediction < 0.0,
            "history_source_seeds": [1, 2, 3, 4, 5, 6],
        }
    return {"city": "jinan", "scenario_history": history}, model


def test_regime_decision_is_resolved_without_validation_seed_or_metric():
    artifact, model = _artifact_model()
    active, evidence = frozen_regime_decision(
        protocol=_protocol(),
        artifact=artifact,
        model=model,
        city="jinan",
        scenario="high",
    )
    assert active is True
    assert evidence["heldout_or_validation_metrics_used"] is False
    assert set(evidence["history_source_seeds"]) == {1, 2, 3, 4, 5, 6}


def test_fallback_city_is_inactive_and_validation_matrix_is_paired():
    artifact, model = _artifact_model()
    active, evidence = frozen_regime_decision(
        protocol=_protocol(), artifact=artifact, model=model, city="la", scenario="la"
    )
    assert active is False
    assert evidence["source"] == "frozen_city_fallback"
    identities = all_rollout_identities(_protocol())
    assert len(identities) == 2 * 2 * 3
    assert {identity[3] for identity in identities} == {
        "phase_pressure",
        "cfcmt_target_regime_trust_cd450",
    }
