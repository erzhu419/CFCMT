import hashlib
import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_counterfactual_cache import (
    _validated_authorization_provenance,
    precompute_counterfactual_cache,
)


def _kwargs(tmp_path: Path):
    return {
        "manifest_path": tmp_path / "unused-manifest.json",
        "scenarios": ("unused",),
        "seeds": (7079,),
        "duration_sec": 600.0,
        "control_interval_sec": 10,
        "warmup_sec": 60.0,
        "max_focal_tls": 4,
        "counterfactual_horizon_intervals": 6,
        "collection_shards": 16,
        "workers": 16,
        "cache_root": tmp_path / "cache",
        "behavior_policy": "phase_pressure",
        "min_tls_coverage": 1.0,
    }


def test_counterfactual_cache_rejects_authorization_hash_mismatch(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate": {"passed": True},
                "decision": "authorize_external_seed7079_collection",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authorization audit SHA-256"):
        precompute_counterfactual_cache(
            **_kwargs(tmp_path),
            authorization_audit=audit,
            expected_authorization_sha256="0" * 64,
            required_authorization_decision=(
                "authorize_external_seed7079_collection"
            ),
        )


def test_counterfactual_cache_rejects_wrong_authorization_decision(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate": {"passed": True},
                "decision": "prohibit_external_seed7079_collection",
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(audit.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not permit collection"):
        precompute_counterfactual_cache(
            **_kwargs(tmp_path),
            authorization_audit=audit,
            expected_authorization_sha256=digest,
            required_authorization_decision=(
                "authorize_external_seed7079_collection"
            ),
        )


def test_counterfactual_cache_accepts_integrity_gate_authorization(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "integrity_gate": {"passed": True},
                "decision": "authorize_hierarchical_collection",
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(audit.read_bytes()).hexdigest()

    result = _validated_authorization_provenance(
        audit,
        expected_sha256=digest,
        required_decision="authorize_hierarchical_collection",
    )

    assert result["sha256"] == digest
    assert result["decision"] == "authorize_hierarchical_collection"
