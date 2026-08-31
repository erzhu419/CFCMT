#!/usr/bin/env python3
"""Freeze the selected waiting-aligned latent artifact protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (  # noqa: E402
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as DIAGNOSTIC_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_diagnostic import (  # noqa: E402
    PROTOCOL as DIAGNOSTIC_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-latent-artifact-v1"
ARTIFACT_ROLE = "waiting_aligned_target_spatiotemporal_action_originator"
ADAPTATION_CLASSIFICATION = "target_simulator_labeled_few_shot_adaptation"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def freeze_protocol(
    *,
    diagnostic_protocol_path: Path,
    diagnostic_result_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    diagnostic_protocol = _read_json(diagnostic_protocol_path)
    diagnostic_result = _read_json(diagnostic_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite latent artifact protocol: {output_path}")
    if (
        diagnostic_protocol.get("protocol") != DIAGNOSTIC_PROTOCOL
        or diagnostic_result.get("protocol") != DIAGNOSTIC_RESULT_PROTOCOL
        or diagnostic_result.get("status") != "PASS"
        or diagnostic_result.get("decision") != AUTHORIZATION_DECISION
        or diagnostic_result.get("integrity_gate", {}).get("passed") is not True
        or diagnostic_result.get("advance_gate", {}).get("passed") is not True
        or diagnostic_result.get("diagnostic_protocol_sha256")
        != _sha256(diagnostic_protocol_path)
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("latent artifact evidence is not authorized")
    if int(diagnostic_result["gate_selection"]["feasible_candidate_count"]) != 1:
        raise ValueError("latent artifact requires a unique feasible candidate")
    selected = dict(diagnostic_result["advance_gate"]["selected_candidate"])
    training = dict(diagnostic_protocol["training"])
    model_specs = {
        str(spec["key"]): dict(spec) for spec in training["model_candidates"]
    }
    model_key = str(selected["model_key"])
    if model_key not in model_specs:
        raise ValueError("selected latent model is absent from the frozen grid")
    frozen = dict(diagnostic_protocol["frozen_inputs"])
    expected = {
        "cache_protocol_sha256": _sha256(cache_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "partition_sha256": _sha256(partition_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    if any(str(frozen.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("latent artifact input identity changed")
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": {
            "diagnostic_protocol_path": str(diagnostic_protocol_path.resolve()),
            "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
            "diagnostic_result_path": str(diagnostic_result_path.resolve()),
            "diagnostic_result_sha256": _sha256(diagnostic_result_path),
            "cache_protocol_path": str(cache_protocol_path.resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(cache_audit_path.resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "partition_path": str(partition_path.resolve()),
            "partition_sha256": _sha256(partition_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "training": training,
        "selected": {
            "candidate_key": str(selected["key"]),
            "model_key": model_key,
            "model_spec": model_specs[model_key],
            "risk_multiplier": float(selected["risk_multiplier"]),
            "minimum_context_trust": float(selected["minimum_context_trust"]),
            "retained_group_count_oof": int(selected["retained_group_count"]),
            "oof_equal_seed_mean_delta": float(selected["mean_delta"]),
            "oof_bootstrap_ci95": list(selected["bootstrap"]["ci95"]),
        },
        "artifact_contract": {
            "artifact_role": ARTIFACT_ROLE,
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
            "proposal_origin_changed": True,
            "reference_policy": "phase_pressure",
            "counterfactual_cost_mode": "halted_queue",
            "rollout_value_horizon_sec": 450,
            "online_inputs": ["tls_id", "simulation_time", "candidate_phase_state"],
            "online_target_outcomes": False,
        },
        "claim_boundary": (
            "The artifact is fitted only on the 22 disclosed redevelopment seeds. "
            "It is target-simulator-labeled few-shot adaptation, not zero-shot transfer. "
            "Closed-loop efficacy and untouched-seed generalization remain untested."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        diagnostic_protocol_path=args.diagnostic_protocol,
        diagnostic_result_path=args.diagnostic_result,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "selected": payload["selected"]["candidate_key"],
                "classification": payload["artifact_contract"][
                    "adaptation_classification"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
