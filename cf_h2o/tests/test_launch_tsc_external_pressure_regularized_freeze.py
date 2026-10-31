import hashlib
import json
from pathlib import Path

import scripts.cluster.launch_tsc_external_pressure_regularized_freeze as launcher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v9_pressure_specs_bind_aligned_audit_and_repaired_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    joint = tmp_path / "joint.json"
    joint.write_text(
        json.dumps(
            {
                "status": "PASS",
                "decision": "authorize_v9_estimand_aligned_frozen_development_replay",
            }
        ),
        encoding="utf-8",
    )
    protocol_path = tmp_path / launcher.V9_ALIGNED_PROTOCOL_RELATIVE
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(
        json.dumps(
            {
                "protocol": "tsc-v55r51-external-v9-oof-validated-support-development-v1",
                "estimand_aligned_joint_audit": {"sha256": _sha(joint)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)

    specs, hashes = launcher.build_specs(
        snapshot_root=Path("/remote/snapshot"),
        aligned_joint_audit_local=joint,
        aligned_joint_audit_remote=Path("/remote/joint.json"),
        aligned_freeze_root=Path("/remote/aligned-v2"),
        external_cache_root=Path("/remote/cache-v9"),
        conversion_root=Path("/remote/conversion-v9"),
        expected_external_cache_sha256="a" * 64,
        remote_results_root=Path("/remote/pressure-v9"),
        local_results_root=tmp_path / "pressure-v9",
        workers=24,
        cpu_cores=32,
        ram_mb=65536,
        generation="v9",
    )

    assert hashes == {"aligned_joint_audit": _sha(joint)}
    assert len(specs) == 2
    assert all("v56r52" in spec["signature"] for spec in specs)
    assert all(
        "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        in spec["cmd"]
        for spec in specs
    )
    assert all("--aligned-freeze-root /remote/aligned-v2" in spec["cmd"] for spec in specs)
    assert {spec["require_node"] for spec in specs} == {"node003", "node004"}
