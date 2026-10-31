from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_closed_loop_diagnostic import POLICIES
from scripts.cluster.launch_tsc_external_closed_loop_diagnostic import (
    NODES,
    build_specs,
)


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_launcher_builds_complete_six_node_matrix(tmp_path):
    joint_path = tmp_path / "joint.json"
    joint_sha = _write_json(
        joint_path,
        {
            "status": "PASS",
            "decision": "authorize_external_seed8171_collection",
        },
    )
    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        {
            "protocol": "tsc-v43r39-external-seed8171-heldout-evaluation-v1",
            "decision": "authorize_closed_loop_external_confirmation",
            "joint_freeze_audit_sha256": joint_sha,
            "offline_confirmation_gate": {"passed": True},
        },
    )

    specs, authorization_sha, observed_joint_sha = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        offline_authorization_local=authorization_path,
        offline_authorization_remote=Path("/remote/evidence/offline.json"),
        joint_audit_local=joint_path,
        joint_audit_remote=Path("/remote/evidence/joint.json"),
        freeze_root=Path("/remote/freeze"),
        conversion_root=Path("/remote/conversion"),
        conversion_manifest=Path("/remote/conversion/conversion_manifest.json"),
        external_manifest_sha256="external-manifest",
        conversion_manifest_sha256="conversion-manifest",
        conversion_tree_sha256="conversion-tree",
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "local-results",
        cpu_cores=1,
        ram_mb=4096,
    )

    assert len(specs) == 48
    assert len({spec["signature"] for spec in specs}) == 48
    assert authorization_sha == hashlib.sha256(
        authorization_path.read_bytes()
    ).hexdigest()
    assert observed_joint_sha == joint_sha
    assert {spec["require_node"] for spec in specs} == set(NODES)
    assert {
        spec["signature"].rsplit("/", 1)[-1] for spec in specs
    } == set(POLICIES)
    assert all(
        "--seed 8081" in spec["cmd"] or "--seed 9091" in spec["cmd"]
        for spec in specs
    )
    assert all(
        "traffic_signal_external_closed_loop_diagnostic" in spec["cmd"]
        for spec in specs
    )
