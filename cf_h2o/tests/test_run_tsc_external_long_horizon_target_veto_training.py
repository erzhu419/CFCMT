import json
from pathlib import Path

from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (
    AUDIT_DECISION,
    AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_training import (
    PROTOCOL,
)
import scripts.cluster.run_tsc_external_long_horizon_target_veto_training as runmod


def test_executable_training_freeze_rejects_source_hash_drift(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runmod, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("before\n", encoding="utf-8")
    from scripts.cluster.launch_tsc_external_network_admission_v6 import _sha256

    cache_audit = tmp_path / "cache_audit.json"
    cache_audit.write_text(
        json.dumps(
            {
                "protocol": AUDIT_PROTOCOL,
                "status": "PASS",
                "decision": AUDIT_DECISION,
                "integrity_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    relative_source = source.resolve().relative_to(tmp_path.resolve())
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "executable_sources": {str(relative_source): _sha256(source)},
                "required_cache_audit": {
                    "future_path": str(cache_audit.resolve().relative_to(tmp_path.resolve()))
                },
            }
        ),
        encoding="utf-8",
    )

    assert runmod.validate_training_freeze(
        training_protocol_path=training, cache_audit_path=cache_audit
    )["gate"]["passed"] is True
    source.write_text("after\n", encoding="utf-8")
    assert runmod.validate_training_freeze(
        training_protocol_path=training, cache_audit_path=cache_audit
    )["gate"]["passed"] is False
