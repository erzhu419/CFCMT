import hashlib
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    EVALUATION_SEED,
    V9_EVALUATION_SEED,
)
from scripts.cluster.launch_tsc_external_full_evaluation_cache import (
    ASSIGNMENTS,
    AUTHORIZATION_DECISION,
    V9_ASSIGNMENTS,
    V9_AUTHORIZATION_DECISION,
    build_specs,
)


def test_full_evaluation_cache_requires_seed8171_joint_freeze(
    tmp_path: Path,
) -> None:
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
        assert f"--seeds {EVALUATION_SEED}" in spec["cmd"]
        assert "--authorization-audit /remote/joint-freeze.json" in spec["cmd"]
        assert f"--required-authorization-decision {AUTHORIZATION_DECISION}" in spec[
            "cmd"
        ]


def test_v9_evaluation_cache_uses_fresh_seed_and_repaired_manifest(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "joint-freeze-v9.json"
    authorization.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate": {"passed": True},
                "decision": V9_AUTHORIZATION_DECISION,
                "evaluation_cache_file_count_at_freeze": 0,
            }
        ),
        encoding="utf-8",
    )
    specs, _ = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        authorization_audit_local=authorization,
        authorization_audit_remote=Path("/remote/joint-freeze-v9.json"),
        conversion_root=Path("/remote/conversion-v9"),
        conversion_manifest_sha256="c" * 64,
        conversion_tree_sha256="d" * 64,
        remote_cache_root=Path("/remote/evaluation-cache-v9"),
        remote_results_root=Path("/remote/results-v9"),
        local_results_root=tmp_path / "results-v9",
        collection_shards=16,
        workers=16,
        cpu_cores=16,
        ram_mb=16384,
        generation="v9",
    )

    assert [spec["require_node"] for spec in specs] == [
        node for node, _, _ in V9_ASSIGNMENTS
    ]
    assert all(f"--seeds {V9_EVALUATION_SEED}" in spec["cmd"] for spec in specs)
    assert all(
        "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        in spec["cmd"]
        for spec in specs
    )
    assert all(
        f"--required-authorization-decision {V9_AUTHORIZATION_DECISION}"
        in spec["cmd"]
        for spec in specs
    )
