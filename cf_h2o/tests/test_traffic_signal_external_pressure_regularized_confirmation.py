from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    FALSIFICATION_RESULT_PROTOCOL,
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    METHOD,
    PROTOCOL,
    run_pressure_regularized_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import ANCHOR_POLICY
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import PriorRegularizationConfig
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pressure_regularized_rollout_installs_oof_regularizer_and_support(
    tmp_path, monkeypatch
):
    parent_path = tmp_path / "parent.json"
    parent_sha = _write_json(parent_path, {"protocol": "parent"})
    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(
        joint_path,
        {
            "protocol": JOINT_AUDIT_PROTOCOL,
            "status": "PASS",
            "decision": JOINT_AUDIT_DECISION,
        },
    )
    protocol_path = tmp_path / "protocol.json"
    _write_json(
        protocol_path,
        {
            "protocol": PROTOCOL,
            "parent_protocol": {"path": str(parent_path), "sha256": parent_sha},
            "pressure_regularized_joint_audit": {"sha256": joint_sha},
            "city_artifacts": {
                "los_angeles": {
                    "certificate": {"sha256": "certificate"},
                    "model": {"sha256": "model"},
                }
            },
            "contaminated_falsification": {
                "closed_loop_seeds": [8081],
                "policies": [METHOD, "phase_pressure"],
            },
            "untouched_robustness_development": {
                "closed_loop_seeds": [52282],
                "policies": [METHOD, "phase_pressure"],
            },
            "prospective_confirmation_reservation": {
                "closed_loop_seeds": [10091],
                "city_scenarios": {"los_angeles": ["la_1x4"]},
            },
        },
    )
    prior = type("Prior", (), {"key": ANCHOR_POLICY})()
    offline = OfflineScreeningModels(
        family_models={ANCHOR_FAMILY: object(), CORRECTION_FAMILY: object()},
        prior_spec=prior,
        objective_modes={ANCHOR_FAMILY: "control_only", CORRECTION_FAMILY: "control_only"},
        diagnostics={},
    )
    regularizer = PriorRegularizationConfig(
        enabled=True,
        blend_weight=6.0,
        risk_multiplier=0.0,
        min_context_trust=0.25,
    )
    support = TargetActionSupport(
        feature_names=("total_q",),
        center=np.asarray([0.0]),
        scale=np.asarray([1.0]),
        prototypes=np.asarray([[0.0]]),
        label_free=True,
    )
    certificate = {
        "selected_candidate": "constant_alpha_0p5",
        "regularizer_selection": {"selected": {"feasible": True}},
        "deployment": {
            "regularizer": {
                "enabled": True,
                "blend_weight": 6.0,
                "risk_multiplier": 0.0,
                "min_context_trust": 0.25,
            }
        },
    }
    model = {
        "selected_candidate": "constant_alpha_0p5",
        "models": offline,
        "regularizer": regularizer,
        "target_support": support,
    }
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation._load_artifacts",
        lambda **_: (certificate, model),
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation._prediction_horizon_sec",
        lambda _: 60,
    )
    captured = {}

    def fake_rollout(**kwargs):
        captured.update(kwargs)
        return {"metrics": {"mean_tripinfo_waiting_time": 1.0}}

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation.run_external_closed_loop_rollout",
        fake_rollout,
    )
    result = run_pressure_regularized_rollout(
        protocol_path=protocol_path,
        joint_audit_path=joint_path,
        expected_joint_audit_sha256=joint_sha,
        pressure_freeze_root=tmp_path / "freeze",
        analysis_stage="falsification",
        policy=METHOD,
        scenario="la_1x4",
        seed=8081,
        rollout_kwargs={"protocol_spec_path": parent_path},
    )

    override = captured["method_runtime_override"]
    diagnostic = captured["diagnostic_spec"]
    assert override.runtime_policy.endswith("_contrast_pressure_gate")
    assert override.models.regularizers[ANCHOR_FAMILY] == regularizer
    assert override.models.target_support is support
    assert diagnostic.coordination_mode == "direct"
    assert diagnostic.result_protocol == FALSIFICATION_RESULT_PROTOCOL
    assert result["exact_phase_fallback"] is False


def test_pressure_regularized_rollout_rejects_unlicensed_stages(tmp_path):
    for stage in ("robustness", "confirmation"):
        try:
            run_pressure_regularized_rollout(
                protocol_path=tmp_path / "unused.json",
                joint_audit_path=tmp_path / "unused-audit.json",
                expected_joint_audit_sha256="unused",
                pressure_freeze_root=tmp_path,
                analysis_stage=stage,
                policy=METHOD,
                scenario="la_1x4",
                seed=52282,
                rollout_kwargs={},
            )
        except ValueError as exc:
            assert "permanently unauthorized" in str(exc)
        else:
            raise AssertionError(f"v47 stage unexpectedly authorized: {stage}")
