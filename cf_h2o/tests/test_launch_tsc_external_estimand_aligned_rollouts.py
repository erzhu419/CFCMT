from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    AUTHORIZATION_DECISION,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    ALIGNED_AUDIT_DECISION,
    ALIGNED_PROTOCOL,
    METHOD,
    POLICIES,
)
from scripts.cluster import launch_tsc_external_estimand_aligned_rollouts as launcher


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_aligned_launcher_separates_development_and_prospective_matrices(
    tmp_path, monkeypatch
):
    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        {
            "decision": AUTHORIZATION_DECISION,
            "offline_confirmation_gate": {"passed": True},
        },
    )
    parent_joint = tmp_path / "parent_joint.json"
    _write_json(parent_joint, {"status": "PASS"})
    aligned_joint = tmp_path / "aligned_joint.json"
    aligned_sha = _write_json(
        aligned_joint,
        {"status": "PASS", "decision": ALIGNED_AUDIT_DECISION},
    )
    protocol_path = tmp_path / launcher.ALIGNED_PROTOCOL_RELATIVE
    _write_json(
        protocol_path,
        {
            "protocol": ALIGNED_PROTOCOL,
            "estimand_aligned_joint_audit": {"sha256": aligned_sha},
            "city_artifacts": {
                "los_angeles": {
                    "deployment_decision": "deploy_oof_conformal_trust_region"
                },
                "jinan": {"deployment_decision": "deploy_pressure_prior_only"},
            },
            "development": {
                "closed_loop_seeds": [8081, 9091],
                "policies": [METHOD, "phase_pressure"],
            },
            "prospective_confirmation_reservation": {
                "closed_loop_seeds": [10091, 11003, 12007],
                "city_scenarios": {
                    "los_angeles": ["la_1x4"],
                    "jinan": [
                        "jinan_3x4_real",
                        "jinan_3x4_real_2000",
                        "jinan_3x4_real_2500",
                    ],
                },
            },
        },
    )
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    common = {
        "snapshot_root": Path("/remote/snapshot"),
        "offline_authorization_local": authorization,
        "offline_authorization_remote": Path("/remote/evidence/authorization.json"),
        "parent_joint_audit_local": parent_joint,
        "parent_joint_audit_remote": Path("/remote/evidence/parent.json"),
        "aligned_joint_audit_local": aligned_joint,
        "aligned_joint_audit_remote": Path("/remote/evidence/aligned.json"),
        "aligned_freeze_root": Path("/remote/aligned-freeze"),
        "freeze_root": Path("/remote/old-freeze"),
        "conversion_root": Path("/remote/conversion"),
        "conversion_manifest": Path("/remote/conversion/manifest.json"),
        "external_manifest_sha256": "external",
        "conversion_manifest_sha256": "conversion",
        "conversion_tree_sha256": "tree",
        "remote_results_root": Path("/remote/results"),
        "cpu_cores": 1,
        "ram_mb": 4096,
    }

    development, _ = launcher.build_specs(
        analysis_stage="development",
        local_results_root=tmp_path / "development",
        **common,
    )
    confirmation, _ = launcher.build_specs(
        analysis_stage="confirmation",
        local_results_root=tmp_path / "confirmation",
        **common,
    )
    diagnostic, _ = launcher.build_specs(
        analysis_stage="development",
        local_results_root=tmp_path / "diagnostic",
        scenario_filter=("la_1x4",),
        policy_filter=(METHOD,),
        run_tag="trace-v2",
        **common,
    )
    execution_diagnostic, execution_hashes = launcher.build_specs(
        analysis_stage="development",
        local_results_root=tmp_path / "execution-diagnostic",
        execution_variant_filter=("direct_raw", "spatial_guarded"),
        run_tag="execution-v1",
        **common,
    )

    assert len(development) == 4 * 2 * 2 == 16
    assert len(confirmation) == 4 * 3 * len(POLICIES) == 96
    assert len(diagnostic) == 2
    assert len(execution_diagnostic) == 2 * 2
    assert (
        execution_hashes["result_protocol"]
        == launcher.EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL
    )
    assert all("/la_1x4/" in spec["signature"] for spec in execution_diagnostic)
    assert all("jinan" not in spec["signature"] for spec in execution_diagnostic)
    assert all("--execution-variant" in spec["cmd"] for spec in execution_diagnostic)
    assert all(spec["signature"].endswith("/trace-v2") for spec in diagnostic)
    assert {spec["require_node"] for spec in development} == set(launcher.NODES)
    assert all(
        "traffic_signal_external_estimand_aligned_confirmation" in spec["cmd"]
        for spec in development + confirmation
    )
    assert all(
        "seed-10091" not in spec["signature"]
        and "seed-11003" not in spec["signature"]
        and "seed-12007" not in spec["signature"]
        for spec in development
    )
