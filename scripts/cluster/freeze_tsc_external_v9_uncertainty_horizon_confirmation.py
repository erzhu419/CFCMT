#!/usr/bin/env python3
"""Freeze the untouched v84 uncertainty-horizon confirmation."""

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
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_audit import (  # noqa: E402
    RESULT_PROTOCOL as ARTIFACT_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze import (  # noqa: E402
    RESULT_PROTOCOL as ARTIFACT_FREEZE_PROTOCOL,
    load_uncertainty_horizon_originator,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_uncertainty_horizon_closed_loop import (  # noqa: E402
    AUDIT_PROTOCOL as V83_AUDIT_PROTOCOL,
    AUTHORIZATION_DECISION as V83_AUTHORIZATION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (  # noqa: E402
    PROTOCOL as ARTIFACT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_closed_loop_development import (  # noqa: E402
    PHASE_POLICY,
    PROTOCOL as V83_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
)


PROTOCOL = "tsc-v84r80-external-v9-uncertainty-horizon-confirmation-v1"
METHOD_KEY = "uh_state_action_h120"
METHOD_KEYS = (METHOD_KEY,)
ARTIFACT_KEY = "state_action_h120_r025_t010"
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def freeze_protocol(
    *,
    v83_protocol_path: Path,
    v83_audit_path: Path,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
    artifact_audit_path: Path,
    artifact_root: Path,
    partition_path: Path,
    parent_runtime_protocol_path: Path,
    hierarchical_parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_manifest_path: Path,
    conversion_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v84 protocol: {output_path}")
    v83 = _read_json(v83_protocol_path)
    audit = _read_json(v83_audit_path)
    artifact_protocol = _read_json(artifact_protocol_path)
    artifact_result = _read_json(artifact_result_path)
    artifact_audit = _read_json(artifact_audit_path)
    partition = _read_json(partition_path)
    if not (
        v83.get("protocol") == V83_PROTOCOL
        and audit.get("protocol") == V83_AUDIT_PROTOCOL
        and audit.get("status") == "PASS"
        and audit.get("decision") == V83_AUTHORIZATION_DECISION
        and audit.get("integrity_gate", {}).get("passed") is True
        and audit.get("advance_gate", {}).get("passed") is True
        and audit.get("advance_gate", {}).get("selected_policy") == METHOD_KEY
        and audit.get("protocol_sha256") == _sha256(v83_protocol_path)
        and artifact_protocol.get("protocol") == ARTIFACT_PROTOCOL
        and artifact_result.get("protocol") == ARTIFACT_FREEZE_PROTOCOL
        and artifact_result.get("status") == "PASS"
        and artifact_audit.get("protocol") == ARTIFACT_AUDIT_PROTOCOL
        and artifact_audit.get("status") == "PASS"
        and partition.get("protocol") == PARTITION_PROTOCOL
        and _sha256(artifact_protocol_path)
        == v83["artifact"]["protocol"]["sha256"]
        and _sha256(artifact_result_path)
        == v83["artifact"]["freeze_result"]["sha256"]
        and _sha256(artifact_audit_path)
        == v83["artifact"]["independent_audit"]["sha256"]
        and _sha256(partition_path) == v83["partition"]["sha256"]
    ):
        raise ValueError("v84 authorization chain changed")

    candidates = {
        str(row["key"]): dict(row)
        for row in v83["development"]["method_candidates"]
    }
    candidate = candidates[METHOD_KEY]
    selected = dict(audit["selected_candidate"])
    artifact_record = dict(v83["artifact"]["artifacts"][ARTIFACT_KEY])
    originator, certificate, artifact = load_uncertainty_horizon_originator(
        artifact_root, ARTIFACT_KEY
    )
    model_path = Path(artifact_root) / ARTIFACT_KEY / "model.pkl"
    certificate_path = Path(artifact_root) / ARTIFACT_KEY / "freeze.json"
    if not (
        candidate["artifact_key"] == ARTIFACT_KEY
        and candidate["prediction_horizon_sec"] == 120
        and candidate["selection_eligible_closed_loop"] is True
        and selected["policy"] == METHOD_KEY
        and selected["artifact_key"] == ARTIFACT_KEY
        and selected["prediction_horizon_sec"] == 120
        and selected["feasible"] is True
        and artifact_record["model"]["sha256"] == _sha256(model_path)
        and artifact_record["certificate"]["sha256"]
        == _sha256(certificate_path)
        and certificate["prediction_sha256"]
        == artifact_record["prediction_sha256"]
        and artifact["artifact_key"] == ARTIFACT_KEY
        and int(originator.config.rollout_value_horizon_sec) == 120
        and float(originator.config.risk_multiplier) == 0.25
        and float(originator.config.minimum_context_trust) == 0.10
    ):
        raise ValueError("v84 selected candidate or artifact changed")

    inherited_paths = {
        "parent_runtime_protocol": parent_runtime_protocol_path,
        "hierarchical_parent": hierarchical_parent_path,
        "offline_result": offline_result_path,
        "support_audit": support_audit_path,
        "hierarchical_freeze_audit": hierarchical_freeze_audit_path,
        "environment_protocol": environment_protocol_path,
        "environment_parent": environment_parent_path,
        "external_manifest": external_manifest_path,
        "conversion_manifest": conversion_manifest_path,
    }
    inherited_hashes = {
        "parent_runtime_protocol": v83["runtime_parent"]["protocol"]["sha256"],
        "hierarchical_parent": v83["runtime_parent"]["hierarchical_parent"][
            "sha256"
        ],
        "offline_result": v83["runtime_parent"]["offline_result"]["sha256"],
        "support_audit": v83["runtime_parent"]["support_audit"]["sha256"],
        "hierarchical_freeze_audit": v83["runtime_parent"][
            "hierarchical_freeze_audit"
        ]["sha256"],
        "environment_protocol": v83["environment"]["protocol"]["sha256"],
        "environment_parent": v83["environment"]["network_parent_protocol"][
            "sha256"
        ],
        "external_manifest": v83["environment"]["external_manifest"]["sha256"],
        "conversion_manifest": v83["environment"]["conversion_manifest"][
            "sha256"
        ],
    }
    if any(
        _sha256(path) != str(inherited_hashes[key])
        for key, path in inherited_paths.items()
    ):
        raise ValueError("v84 inherited runtime evidence changed")
    if str(Path(hierarchical_freeze_root).resolve()) != str(
        Path(v83["runtime_parent"]["hierarchical_freeze_root"]).resolve()
    ):
        raise ValueError("v84 hierarchical freeze root changed")

    confirmation = dict(partition["confirmatory"])
    prospective = dict(partition["prospective"])
    confirmation_root = Path(confirmation["root"])
    prospective_root = Path(prospective["root"])
    seeds = tuple(int(value) for value in confirmation["seeds"])
    if not (
        len(seeds) == len(set(seeds)) == int(confirmation["seed_count"]) == 32
        and str(Path(local_results_root).resolve())
        == str(confirmation_root.resolve())
        and not confirmation_root.exists()
        and not prospective_root.exists()
        and not set(seeds).intersection(v83["development"]["seeds"])
        and not set(seeds).intersection(prospective["seeds"])
    ):
        raise ValueError("v84 confirmatory partition is not untouched")

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v83_selection_pre_v84_outcomes",
        "confirmation_parent": {
            "v83_protocol": _path_record(v83_protocol_path),
            "v83_independent_audit": _path_record(v83_audit_path),
            "v83_decision": V83_AUTHORIZATION_DECISION,
        },
        "artifact": {
            "protocol": _path_record(artifact_protocol_path),
            "freeze_result": _path_record(artifact_result_path),
            "independent_audit": _path_record(artifact_audit_path),
            "root": str(Path(artifact_root).resolve()),
            "artifact_key": ARTIFACT_KEY,
            "model": _path_record(model_path),
            "certificate": _path_record(certificate_path),
            "prediction_sha256": certificate["prediction_sha256"],
            "risk_multiplier": 0.25,
            "minimum_context_trust": 0.10,
            "prediction_horizon_sec": 120,
        },
        "partition": _path_record(partition_path),
        "runtime_parent": dict(v83["runtime_parent"]),
        "environment": dict(v83["environment"]),
        "confirmation": {
            "classification": "untouched_confirmatory",
            "city_scenarios": {"jinan": ["jinan_3x4_real"]},
            "seeds": list(seeds),
            "policies": [PHASE_POLICY, METHOD_KEY],
            "phase_policy": PHASE_POLICY,
            "method_key": METHOD_KEY,
            "method_candidate": candidate,
            "seed_count": len(seeds),
            "policy_count": 2,
            "matrix_size": len(seeds) * 2,
            "pairing_unit": "same_scenario_same_simulator_seed",
            "primary_metric": "mean_tripinfo_waiting_time",
            "confirmation_rule": dict(v83["development"]["selection_rule"]),
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260831,
                "unit": "paired_simulator_seed",
            },
        },
        "prospective_partition": {
            "root": str(prospective_root),
            "seed_count": int(prospective["seed_count"]),
            "remains_sealed": True,
        },
        "output_roots": {
            "remote_results_root": str(remote_results_root),
            "local_results_root": str(Path(local_results_root).resolve()),
        },
        "scientific_status": {
            "candidate_selection_complete_before_v84": True,
            "candidate_count": 1,
            "threshold_or_architecture_tuning_in_v84": False,
            "v84_outcomes_inspected_before_freeze": False,
            "v85_outcomes_inspected_before_freeze": False,
        },
        "claim_boundary": (
            "This 64-rollout experiment is the first untouched confirmation of "
            "the single v83-selected uh_state_action_h120 policy. It compares "
            "against paired PhasePressure under the same full-day SUMO protocol. "
            "No v84 outcome was available when the policy and gate were frozen; "
            "the eight-seed v85 prospective root remains sealed."
        ),
    }
    if payload["confirmation"]["matrix_size"] != 64:
        raise RuntimeError("v84 confirmation matrix size changed")
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v83-protocol", type=Path, required=True)
    parser.add_argument("--v83-audit", type=Path, required=True)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--parent-runtime-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v83_protocol_path=args.v83_protocol,
        v83_audit_path=args.v83_audit,
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
        artifact_audit_path=args.artifact_audit,
        artifact_root=args.artifact_root,
        partition_path=args.partition,
        parent_runtime_protocol_path=args.parent_runtime_protocol,
        hierarchical_parent_path=args.hierarchical_parent,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_manifest_path=args.conversion_manifest,
        conversion_root=args.conversion_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "matrix_size": payload["confirmation"]["matrix_size"],
                "method": payload["confirmation"]["method_key"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
