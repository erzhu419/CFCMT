from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import ANCHOR_FAMILY
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_source_only_guard import (
    PROTOCOL,
    anchored_prediction_bundle,
    familywise_guard_options,
    load_source_only_guard,
    resolve_frozen_prior_policy,
    validate_source_only_guard_payload,
)


def _payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "scientific_status": "source-only-frozen-before-unseen-target-closed-loop",
        "family": ANCHOR_FAMILY,
        "prior_policy": "phase_pressure",
        "source_domains": ["a", "b"],
        "target_data_consumed": False,
        "target_closed_loop_consumed": False,
        "guard_config": {
            "enabled": True,
            "risk_multiplier": 0.5,
            "min_context_trust": 0.2,
            "margin": 0.0,
            "max_relative_rule_gap": 0.25,
        },
        "uncertainty_scale": 1.75,
    }


def test_familywise_guard_options_controls_complete_grid() -> None:
    options, audit = familywise_guard_options(
        thresholds=(0.0, 0.5),
        max_relative_rule_gaps=(0.1, 0.5, 1.0),
        familywise_alpha=0.05,
    )

    assert audit["candidate_count"] == 6
    assert options["selection_mode"] == "mean_ucb"
    assert options["confidence_z"] == pytest.approx(
        NormalDist().inv_cdf(1.0 - 0.05 / 6.0)
    )


def test_familywise_guard_options_rejects_duplicate_grid_values() -> None:
    with pytest.raises(ValueError, match="family-wise guard grid"):
        familywise_guard_options(thresholds=(0.0, 0.0))


def test_frozen_generalized_pressure_key_resolves_without_string_parsing() -> None:
    prior = resolve_frozen_prior_policy("gp_d0_o0p02_s0")

    assert prior.key == "gp_d0_o0p02_s0"
    assert prior.downstream_occupancy_weight == pytest.approx(0.02)


def test_anchored_prediction_bundle_matches_online_constant_blend(
    monkeypatch,
) -> None:
    dataset = SimpleNamespace(
        size=2,
        feature_names=("x",),
        features=np.asarray([[1.0], [2.0]]),
        domains=np.asarray(["d", "d"]),
    )

    def adjusted(unused_dataset, prediction, *, objective_mode):
        assert unused_dataset is dataset
        assert objective_mode == "control_only"
        return (
            np.asarray(prediction["score"]),
            np.asarray(prediction["uncertainty"]),
            np.asarray(prediction["trust"]),
            np.asarray([0, 0]),
        )

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_source_only_guard._group_adjusted_scores",
        adjusted,
    )
    bundle = anchored_prediction_bundle(
        {
            "d": (
                dataset,
                {"score": [0.0, 2.0], "uncertainty": [0.0, 1.0], "trust": [1.0, 0.8]},
            )
        },
        {
            "d": (
                dataset,
                {"score": [0.0, -1.0], "uncertainty": [0.0, 3.0], "trust": [0.9, 0.7]},
            )
        },
    )

    prediction = bundle["d"][1]["control_cost"]
    assert np.allclose(prediction["mean"], [0.0, -0.7])
    assert np.allclose(prediction["uncertainty"], [0.0, 2.8])
    assert np.allclose(prediction["context_trust"], [0.9, 0.7])


def test_source_only_guard_payload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "guard.json"
    path.write_text(json.dumps(_payload(), sort_keys=True), encoding="utf-8")

    guard, scale, loaded = load_source_only_guard(
        path,
        expected_sha256=_sha256(path),
        expected_prior_policy="phase_pressure",
    )

    assert guard.enabled
    assert guard.max_relative_rule_gap == pytest.approx(0.25)
    assert scale == pytest.approx(1.75)
    assert loaded["target_closed_loop_consumed"] is False


def test_source_only_guard_rejects_target_data() -> None:
    payload = _payload()
    payload["target_data_consumed"] = True

    with pytest.raises(ValueError, match="contract changed"):
        validate_source_only_guard_payload(
            payload,
            expected_prior_policy="phase_pressure",
        )
