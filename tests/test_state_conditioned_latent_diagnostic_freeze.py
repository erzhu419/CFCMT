from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (
    RESULT_PROTOCOL as V70_RESULT_PROTOCOL,
)
from scripts.cluster.audit_tsc_external_waiting_aligned_latent_closed_loop import (
    AUDIT_PROTOCOL as V72_AUDIT_PROTOCOL,
    REJECTION_DECISION as V72_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (
    MANIFEST_PROTOCOL,
    PROTOCOL,
    freeze_protocol,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PARTITION_PROTOCOL,
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_diagnostic import (
    PROTOCOL as V70_PROTOCOL,
)


V72_PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-latent-closed-loop-v1"


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence(tmp_path: Path) -> dict[str, Path]:
    v70_protocol = _write(
        tmp_path / "v70_protocol.json",
        {
            "protocol": V70_PROTOCOL,
            "training": {
                "seeds": [11, 22],
                "model_candidates": [],
            },
        },
    )
    v70_result = _write(
        tmp_path / "v70_result.json",
        {
            "protocol": V70_RESULT_PROTOCOL,
            "status": "PASS",
            "decision": "authorize_waiting_aligned_spatiotemporal_latent_freeze",
            "diagnostic_protocol_sha256": _sha256(v70_protocol),
        },
    )
    v72_protocol = _write(
        tmp_path / "v72_protocol.json",
        {"protocol": V72_PROTOCOL},
    )
    v72_audit = _write(
        tmp_path / "v72_audit.json",
        {
            "protocol": V72_AUDIT_PROTOCOL,
            "status": "PASS",
            "decision": V72_REJECTION_DECISION,
            "protocol_sha256": _sha256(v72_protocol),
            "integrity_gate": {"passed": True},
            "candidate_summaries": [
                {
                    "policy": "waiting_latent_cd2",
                    "mean_relative_delta": 0.01,
                    "bootstrap": {"ci95": [0.0, 0.02]},
                }
            ],
        },
    )
    cache_protocol = _write(
        tmp_path / "cache_protocol.json",
        {"protocol": CACHE_PROTOCOL},
    )
    cache_audit = _write(
        tmp_path / "cache_audit.json",
        {
            "protocol": CACHE_AUDIT_PROTOCOL,
            "status": "PASS",
            "gate": {"passed": True},
        },
    )
    partition = _write(
        tmp_path / "partition.json",
        {
            "protocol": PARTITION_PROTOCOL,
            "redevelopment": {"seeds": [11, 22]},
            "confirmatory": {"root": str(tmp_path / "sealed_confirmatory")},
            "prospective": {"root": str(tmp_path / "sealed_prospective")},
        },
    )
    manifest = _write(
        tmp_path / "manifest.json",
        {"protocol": MANIFEST_PROTOCOL},
    )
    return {
        "v70_protocol_path": v70_protocol,
        "v70_result_path": v70_result,
        "v72_protocol_path": v72_protocol,
        "v72_audit_path": v72_audit,
        "cache_protocol_path": cache_protocol,
        "cache_audit_path": cache_audit,
        "partition_path": partition,
        "manifest_path": manifest,
    }


def test_freeze_protocol_binds_parent_evidence(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    output = tmp_path / "v73.json"

    payload = freeze_protocol(**evidence, output_path=output)

    assert payload["protocol"] == PROTOCOL
    assert len(payload["training"]["model_candidates"]) == 8
    assert len(payload["selection"]["gate_candidates"]) == 8
    assert payload["frozen_inputs"]["v72_protocol_sha256"] == _sha256(
        evidence["v72_protocol_path"]
    )


@pytest.mark.parametrize("parent", ["cache_protocol_path", "manifest_path"])
def test_freeze_protocol_rejects_wrong_parent_protocol(
    tmp_path: Path,
    parent: str,
) -> None:
    evidence = _evidence(tmp_path)
    _write(evidence[parent], {"protocol": "wrong-parent"})

    with pytest.raises(ValueError, match="authorization changed"):
        freeze_protocol(**evidence, output_path=tmp_path / "v73.json")


def test_freeze_protocol_rejects_opened_confirmatory_root(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    (tmp_path / "sealed_confirmatory").mkdir()

    with pytest.raises(ValueError, match="no longer sealed"):
        freeze_protocol(**evidence, output_path=tmp_path / "v73.json")
