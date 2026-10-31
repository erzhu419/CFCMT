from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (
    _prediction_digest,
    fit_artifact_models,
    load_multihorizon_state_originator,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_audit import (
    online_contrast_feature_names,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_ACTION_FEATURES,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (
    ARTIFACT_KEYS,
    PRIMARY_ARTIFACT_KEY,
    PRIMARY_SELECTION_KEY,
    SECONDARY_ARTIFACT_KEY,
    SECONDARY_SELECTION_KEY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v79_external_v9_"
    "multihorizon_state_latent_artifacts.json"
)


def _contrast() -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    candidate_states = []
    row_tls = []
    row_times = []
    for seed_index, seed in enumerate((11, 22, 33)):
        for group_index, queue in enumerate((2.0, 10.0)):
            group = f"test:seed{seed}:{group_index}:tls0"
            for action in range(3):
                row = np.zeros(len(STATE_ACTION_FEATURES), dtype=float)
                row[STATE_ACTION_FEATURES.index("total_q")] = queue
                row[STATE_ACTION_FEATURES.index("total_veh")] = queue + 3.0
                row[STATE_ACTION_FEATURES.index("mean_speed")] = 12.0 - queue / 2.0
                row[STATE_ACTION_FEATURES.index("delta_green_q")] = action * queue
                row[STATE_ACTION_FEATURES.index("delta_service_pressure")] = (
                    action * (queue - 4.0)
                )
                features.append(row)
                targets.append((0.0, -1.0 if queue > 5.0 else 0.5, 0.25)[action])
                groups.append(group)
                references.append(action == 0)
                candidate_states.append(f"phase{action}")
                row_tls.append("tls0")
                row_times.append(300.0 + 10.0 * seed_index)
    size = len(features)
    target_name = "prefix_mean_cost_60s"
    return MechanismDataset(
        feature_names=STATE_ACTION_FEATURES,
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.zeros((size, 1), dtype=float),
        priors={target_name: np.zeros(size, dtype=float)},
        targets={target_name: np.asarray(targets, dtype=float)},
        domains=np.asarray(["target"] * size),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": candidate_states,
            "row_tls": row_tls,
            "row_times": row_times,
        },
    )


def _spec(
    *, artifact_key: str, feature_set: str, role: str, eligible: bool
) -> dict[str, object]:
    return {
        "artifact_key": artifact_key,
        "role": role,
        "selection_eligible_closed_loop": eligible,
        "model_key": f"{feature_set}_k10_s2::h60s",
        "model_family": "target_state_conditioned_spatiotemporal_latent",
        "model_config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 2.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": feature_set,
            "neighbor_count": 2,
            "local_shrinkage": 1.0,
            "distance_bandwidth": 1.0,
            "uncertainty_quantile": 0.9,
        },
        "target_name": "prefix_mean_cost_60s",
        "prediction_horizon_sec": 60,
        "risk_multiplier": 0.0,
        "minimum_context_trust": 0.25,
    }


def test_v79_protocol_keeps_primary_and_stability_diagnostic_distinct() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    artifacts = {row["artifact_key"]: row for row in protocol["artifacts"]}

    assert tuple(protocol["artifact_keys"]) == ARTIFACT_KEYS
    assert protocol["primary_artifact_key"] == PRIMARY_ARTIFACT_KEY
    assert protocol["secondary_artifact_key"] == SECONDARY_ARTIFACT_KEY
    assert artifacts[PRIMARY_ARTIFACT_KEY]["selection_key"] == PRIMARY_SELECTION_KEY
    assert artifacts[SECONDARY_ARTIFACT_KEY]["selection_key"] == SECONDARY_SELECTION_KEY
    assert artifacts[PRIMARY_ARTIFACT_KEY]["selection_eligible_closed_loop"] is True
    assert artifacts[SECONDARY_ARTIFACT_KEY]["selection_eligible_closed_loop"] is False
    assert {row["prediction_horizon_sec"] for row in artifacts.values()} == {60}
    assert len(protocol["training"]["seeds"]) == 22
    assert len(protocol["training"]["sealed_confirmatory_seeds"]) == 32
    assert len(protocol["training"]["sealed_prospective_seeds"]) == 8


def test_artifact_roundtrip_preserves_each_model_and_gate(tmp_path: Path) -> None:
    contrast = _contrast()
    specs = (
        _spec(
            artifact_key=PRIMARY_ARTIFACT_KEY,
            feature_set="compact_state",
            role="preregistered_primary",
            eligible=True,
        ),
        _spec(
            artifact_key=SECONDARY_ARTIFACT_KEY,
            feature_set="state_action",
            role="predeclared_stability_diagnostic",
            eligible=False,
        ),
    )
    records = fit_artifact_models(
        contrast=contrast,
        artifact_specs=specs,
        city="test",
        scenario="test_scenario",
        training_seeds=(11, 22, 33),
        reference_policy="phase_pressure",
        output_root=tmp_path / "artifacts",
    )

    assert tuple(records) == ARTIFACT_KEYS
    for key, expected_feature_set in (
        (PRIMARY_ARTIFACT_KEY, "compact_state"),
        (SECONDARY_ARTIFACT_KEY, "state_action"),
    ):
        originator, certificate, artifact = load_multihorizon_state_originator(
            tmp_path / "artifacts", key
        )
        assert artifact["model"].config.feature_set == expected_feature_set
        assert originator.config.minimum_context_trust == 0.25
        assert originator.config.rollout_value_horizon_sec == 60
        assert certificate["prediction_sha256"] == _prediction_digest(
            originator.model, contrast
        )
        assert records[key]["prediction_sha256"] == certificate["prediction_sha256"]


def test_artifact_loader_rejects_unregistered_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown multihorizon artifact key"):
        load_multihorizon_state_originator(tmp_path, "posthoc_model")


def test_online_contrast_contract_contains_dynamic_state_action_features() -> None:
    assert set(STATE_ACTION_FEATURES).issubset(online_contrast_feature_names())
