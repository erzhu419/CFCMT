from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    ALIGNED_AUDIT_DECISION,
    ALIGNED_AUDIT_PROTOCOL,
    ALIGNED_PROTOCOL,
    DEVELOPMENT_RESULT_PROTOCOL,
    EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL,
    METHOD,
    run_estimand_aligned_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (
    ANCHOR_POLICY,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(parent_sha: str, joint_sha: str) -> dict:
    return {
        "protocol": ALIGNED_PROTOCOL,
        "parent_protocol": {"sha256": parent_sha},
        "estimand_aligned_joint_audit": {"sha256": joint_sha},
        "city_artifacts": {
            "los_angeles": {
                "certificate": {"sha256": "certificate"},
                "model": {"sha256": "model"},
            }
        },
        "development": {
            "closed_loop_seeds": [8081, 9091],
            "policies": [METHOD, "phase_pressure"],
        },
        "prospective_confirmation_reservation": {
            "closed_loop_seeds": [10091, 11003, 12007],
            "city_scenarios": {"los_angeles": ["la_1x4"]},
        },
        "deployment": {
            "coordination_mode": "sparse",
            "old_v43_offline_authorization_role": "environment only",
        },
    }


def _model_payload(*, enabled: bool) -> tuple[dict, dict]:
    guard = ContrastGuardConfig(
        enabled=enabled,
        risk_multiplier=0.0,
        min_context_trust=0.5,
        margin=0.2,
        max_relative_rule_gap=0.0,
    )
    support = TargetActionSupport(
        feature_names=("total_q",),
        center=np.asarray([0.0]),
        scale=np.asarray([1.0]),
        prototypes=np.asarray([[0.0]]),
    )
    models = OfflineScreeningModels(
        family_models={ANCHOR_FAMILY: object(), CORRECTION_FAMILY: object()},
        prior_spec=type("Prior", (), {"key": ANCHOR_POLICY})(),
        objective_modes={ANCHOR_FAMILY: "control_only", CORRECTION_FAMILY: "control_only"},
        diagnostics={},
    )
    certificate = {
        "selected_candidate": "constant_alpha_0p5",
        "estimand_alignment": {"passed": True},
        "guard_selection": {
            "decision": (
                "deploy_oof_conformal_trust_region"
                if enabled
                else "deploy_pressure_prior_only"
            )
        },
        "deployment": {
            "guard": {
                "enabled": enabled,
                "risk_multiplier": 0.0,
                "min_context_trust": 0.5,
                "margin": 0.2,
                "max_relative_rule_gap": 0.0,
            }
        },
    }
    return certificate, {
        "selected_candidate": "constant_alpha_0p5",
        "models": models,
        "target_support": support,
    }


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    enabled: bool,
    execution_variant: str | None = None,
):
    parent_path = tmp_path / "parent.json"
    parent_sha = _write_json(parent_path, {"protocol": "parent"})
    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(
        joint_path,
        {
            "protocol": ALIGNED_AUDIT_PROTOCOL,
            "status": "PASS",
            "decision": ALIGNED_AUDIT_DECISION,
        },
    )
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, _protocol(parent_sha, joint_sha))
    certificate, model_payload = _model_payload(enabled=enabled)
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation._load_aligned_artifacts",
        lambda **_: (certificate, model_payload),
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation._prediction_horizon_sec",
        lambda _: 60,
    )
    captured = {}

    def fake_rollout(**kwargs):
        captured.update(kwargs)
        return {"metrics": {"mean_tripinfo_waiting_time": 1.0}}

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation.run_external_closed_loop_rollout",
        fake_rollout,
    )
    result = run_estimand_aligned_rollout(
        aligned_protocol_path=protocol_path,
        aligned_joint_audit_path=joint_path,
        expected_aligned_joint_audit_sha256=joint_sha,
        aligned_freeze_root=tmp_path / "freeze",
        analysis_stage="development",
        policy=METHOD,
        scenario="la_1x4",
        seed=8081,
        rollout_kwargs={"protocol_spec_path": parent_path},
        execution_variant=execution_variant,
    )
    return result, captured


def test_enabled_aligned_method_installs_frozen_guard_and_target_support(
    tmp_path, monkeypatch
):
    result, captured = _run(tmp_path, monkeypatch, enabled=True)

    override = captured["method_runtime_override"]
    assert captured["diagnostic_spec"].result_protocol == DEVELOPMENT_RESULT_PROTOCOL
    assert captured["diagnostic_spec"].source_policy == "cfcmt_selected_mpc"
    assert override.models.prior_policy == ANCHOR_POLICY
    assert override.models.target_support is not None
    assert override.models.guards[ANCHOR_FAMILY].enabled
    assert result["exact_phase_fallback"] is False


def test_disabled_aligned_method_is_exact_phase_pressure_fallback(tmp_path, monkeypatch):
    result, captured = _run(tmp_path, monkeypatch, enabled=False)

    assert captured["diagnostic_spec"].source_policy == ANCHOR_POLICY
    assert captured["method_runtime_override"] is None
    assert result["exact_phase_fallback"] is True


def test_raw_direct_execution_diagnostic_reuses_model_without_guard(
    tmp_path, monkeypatch
):
    result, captured = _run(
        tmp_path,
        monkeypatch,
        enabled=True,
        execution_variant="direct_raw",
    )

    diagnostic = captured["diagnostic_spec"]
    override = captured["method_runtime_override"]
    assert captured["policy"] == f"{METHOD}_direct_raw"
    assert diagnostic.result_protocol == EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL
    assert diagnostic.coordination_mode == "direct"
    assert diagnostic.cooldown_intervals_override == 0
    assert diagnostic.guard_config is None
    assert override.runtime_policy.endswith("_contrast_raw")
    assert override.models.target_support is None
    assert result["execution_diagnostic"]["prospective_evidence"] is False


def test_guarded_spatial_execution_diagnostic_installs_frozen_support(
    tmp_path, monkeypatch
):
    _, captured = _run(
        tmp_path,
        monkeypatch,
        enabled=True,
        execution_variant="spatial_guarded",
    )

    diagnostic = captured["diagnostic_spec"]
    override = captured["method_runtime_override"]
    assert diagnostic.coordination_mode == "spatial_only"
    assert diagnostic.guard_config is not None
    assert override.runtime_policy.endswith("_contrast_guard")
    assert override.models.target_support is not None


def test_execution_diagnostic_rejects_city_without_enabled_guard(tmp_path, monkeypatch):
    with np.testing.assert_raises_regex(
        ValueError, "require a city with an enabled frozen guard"
    ):
        _run(
            tmp_path,
            monkeypatch,
            enabled=False,
            execution_variant="direct_raw",
        )
