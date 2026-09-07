from __future__ import annotations

import json
from pathlib import Path
import shlex

import numpy as np

from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    GuardProfile,
    Prediction,
    _blend_predictions,
    _paired_dominance,
    _profile_acceptance,
    _select_guard,
)
from scripts.cluster.launch_tsc_conservative_source_intervention_gate import (
    SEEDS,
    SIGNATURE,
    build_spec,
)


def test_blend_uses_conservative_source_trust() -> None:
    target = Prediction(
        score=np.asarray([0.0, 2.0]),
        uncertainty=np.asarray([0.1, 0.3]),
        trust=np.asarray([0.9, 0.4]),
    )
    source = Prediction(
        score=np.asarray([2.0, 0.0]),
        uncertainty=np.asarray([0.5, 0.1]),
        trust=np.asarray([0.3, 0.8]),
    )
    blended = _blend_predictions(target, source, source_weight=0.25)
    np.testing.assert_allclose(blended.score, [0.5, 1.5])
    np.testing.assert_allclose(blended.uncertainty, [0.2, 0.25])
    np.testing.assert_allclose(blended.trust, [0.3, 0.4])


def test_profile_uses_training_thresholds_and_target_agreement() -> None:
    features = {
        "proposed": np.ones(5, dtype=bool),
        "advantage": np.asarray([1.0, 0.8, 0.6, 0.4, 9.0]),
        "uncertainty": np.zeros(5),
        "trust": np.ones(5),
        "pressure_loss": np.zeros(5),
        "target_agreement": np.asarray([True, False, True, True, True]),
    }
    profile = GuardProfile(
        risk_multiplier=0.0,
        retention_fraction=0.5,
        trust_quantile=0.0,
        pressure_loss_quantile=1.0,
        require_target_agreement=True,
    )
    accepted, thresholds = _profile_acceptance(
        features,
        profile=profile,
        training_mask=np.asarray([True, True, True, True, False]),
    )
    assert thresholds["robust_advantage_threshold"] == 0.8
    np.testing.assert_array_equal(
        accepted, np.asarray([True, False, False, False, True])
    )


def _synthetic_guard_inputs(*, learned_delta: float) -> dict[str, np.ndarray]:
    seeds = np.repeat(np.arange(6), 4)
    policy_rows = np.arange(seeds.size * 2).reshape(-1, 2)
    references = np.zeros(seeds.size * 2, dtype=bool)
    references[policy_rows[:, 0]] = True
    actual = np.zeros(seeds.size * 2)
    actual[policy_rows[:, 1]] = learned_delta
    score = np.zeros(seeds.size * 2)
    score[policy_rows[:, 1]] = -1.0
    prediction = Prediction(
        score=score,
        uncertainty=np.zeros_like(score),
        trust=np.ones_like(score),
    )
    return {
        "prediction": prediction,
        "actual": actual,
        "policy_rows": policy_rows,
        "references": references,
        "pressure_delta": np.zeros_like(score),
        "group_seeds": seeds,
    }


def test_guard_authorizes_consistent_benefit_and_rejects_harm() -> None:
    beneficial = _synthetic_guard_inputs(learned_delta=-0.2)
    accepted = _select_guard(
        **beneficial,
        training_seeds={0, 1, 2, 3, 4},
        target_prediction=None,
    )
    assert accepted["enabled"] is True
    assert accepted["training_intervention_count"] == 20
    assert accepted["training_upper_95_one_sided"] < 0.0
    assert type(accepted["training_active_seed_count"]) is int
    json.dumps(
        {key: value for key, value in accepted.items() if key != "accepted"},
        allow_nan=False,
    )

    harmful = _synthetic_guard_inputs(learned_delta=0.2)
    rejected = _select_guard(
        **harmful,
        training_seeds={0, 1, 2, 3, 4},
        target_prediction=None,
    )
    assert rejected["enabled"] is False
    assert not np.any(rejected["accepted"])


def test_source_authorization_requires_paired_target_gain() -> None:
    training = {0, 1, 2, 3, 4}
    target = {
        "enabled": True,
        "training_seed_values": {seed: -0.01 for seed in training},
    }
    source = {
        "enabled": True,
        "training_seed_values": {seed: -0.02 for seed in training},
    }
    assert _paired_dominance(
        source, target, training_seeds=training
    )["source_authorized"] is True
    source["training_seed_values"] = {seed: -0.009 for seed in training}
    assert _paired_dominance(
        source, target, training_seeds=training
    )["source_authorized"] is False


def test_v124_launcher_freezes_v123_artifact_and_selector() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        prediction_artifact_remote=Path("/v123/predictions.pkl"),
        prediction_artifact_sha256="a" * 64,
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/audit/selector.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/results/v124"),
        node="node005",
        cache_workers=20,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20
    assert spec["ram_mb"] == 65536
    assert "cf_h2o.eval.traffic_signal_conservative_source_intervention_gate" in tokens
    assert tokens[tokens.index("--prediction-artifact-sha256") + 1] == "a" * 64
    seed_start = tokens.index("--selector-seeds") + 1
    seed_stop = tokens.index("--selector-collection-shards")
    assert [int(value) for value in tokens[seed_start:seed_stop]] == list(SEEDS)
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")
