import hashlib
import json
from pathlib import Path

from scripts.cluster.launch_tsc_external_evaluation_counterfactual_cache import (
    ASSIGNMENTS,
    AUTHORIZATION_DECISION,
    build_specs,
)


def test_external_evaluation_cache_requires_joint_freeze(tmp_path: Path) -> None:
    authorization = tmp_path / "joint-freeze.json"
    authorization.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate": {"passed": True},
                "decision": AUTHORIZATION_DECISION,
                "evaluation_cache_file_count_at_freeze": 0,
            }
        ),
        encoding="utf-8",
    )
    specs, digest = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        authorization_audit_local=authorization,
        authorization_audit_remote=Path("/remote/joint-freeze.json"),
        conversion_root=Path("/remote/conversion"),
        conversion_manifest_sha256="a" * 64,
        conversion_tree_sha256="b" * 64,
        remote_cache_root=Path("/remote/evaluation-cache"),
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        collection_shards=16,
        workers=16,
        cpu_cores=16,
        ram_mb=16384,
    )

    assert len(specs) == 4
    assert digest == hashlib.sha256(authorization.read_bytes()).hexdigest()
    assert [spec["require_node"] for spec in specs] == [
        node for node, _, _ in ASSIGNMENTS
    ]
    for spec in specs:
        assert "--seeds 7079" in spec["cmd"]
        assert "--authorization-audit /remote/joint-freeze.json" in spec["cmd"]
        assert f"--required-authorization-decision {AUTHORIZATION_DECISION}" in spec[
            "cmd"
        ]
