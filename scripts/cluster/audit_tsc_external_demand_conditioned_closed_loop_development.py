#!/usr/bin/env python3
"""Audit v71 paired development without opening sealed validation seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_demand_conditioned_closed_loop_development import (  # noqa: E402
    ANALYSIS_STAGE,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_method_rollout_identities,
    candidate_payload,
    development_candidates,
)
from scripts.cluster import (  # noqa: E402
    audit_tsc_external_target_veto_closed_loop_development as base,
)
from scripts.cluster.launch_tsc_external_demand_conditioned_closed_loop_development import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_demand_conditioned_closed_loop_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = (
    "tsc-v71r67-external-v9-demand-conditioned-closed-loop-development-audit-v1"
)
ADVANCE_DECISION = "authorize_demand_conditioned_candidate_for_untouched_validation"
REJECTION_DECISION = (
    "reject_demand_conditioned_development_and_preserve_sealed_seed_sets"
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit(
    *,
    results_root: Path,
    launch_manifest_path: Path,
    protocol_path: Path,
    training_protocol_path: Path,
    training_result_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    veto_joint_audit_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
) -> dict[str, Any]:
    base.ANALYSIS_STAGE = ANALYSIS_STAGE
    base.PROTOCOL = PROTOCOL
    base.RESULT_PROTOCOL = RESULT_PROTOCOL
    base.all_method_rollout_identities = all_method_rollout_identities
    base.candidate_payload = candidate_payload
    base.development_candidates = development_candidates
    base.LAUNCH_PROTOCOL = LAUNCH_PROTOCOL
    base.NODES = NODES
    base.SHARD_PROTOCOL = SHARD_PROTOCOL
    base.AUDIT_PROTOCOL = AUDIT_PROTOCOL
    base.ADVANCE_DECISION = ADVANCE_DECISION
    base.REJECTION_DECISION = REJECTION_DECISION
    payload = base.audit_target_veto_development(
        results_root=results_root,
        launch_manifest_path=launch_manifest_path,
        protocol_path=protocol_path,
        parent_protocol_path=training_protocol_path,
        collection_protocol_path=training_result_path,
        proposal_parent_protocol_path=proposal_parent_protocol_path,
        freeze_audit_path=freeze_audit_path,
        veto_joint_audit_path=veto_joint_audit_path,
        external_manifest_path=external_manifest_path,
        conversion_root=conversion_root,
        conversion_manifest_path=conversion_manifest_path,
    )
    protocol = _read(protocol_path)
    launch = _read(launch_manifest_path)
    remote_sync = dict(launch.get("remote_input_sync", {}))
    demand_profiles = protocol["demand_schedule_context"]["profiles"]
    row_demand_gates = []
    for row in payload.get("rows", ()):
        result = _read(Path(row["result_path"]))
        scenario = str(row["scenario"])
        profile = dict(result.get("demand_profile", {}))
        expected_profile = dict(demand_profiles[scenario])
        config = dict(result.get("target_veto_config", {}))
        gate = {
            "demand_profile_input_hash_matches": profile.get("input_sha256")
            == expected_profile["input_sha256"],
            "demand_profile_features_match": profile.get("feature_names")
            == expected_profile["feature_names"]
            and profile.get("features") == expected_profile["features"],
            "estimand_is_450_seconds": int(
                config.get("rollout_value_horizon_sec", -1)
            )
            == 450,
            "few_shot_classification_explicit": result.get(
                "target_veto_classification"
            )
            == "target_simulator_labeled_few_shot_adaptation",
        }
        gate["passed"] = all(gate.values())
        row["demand_conditioning_gate"] = gate
        row["gate"]["demand_conditioning_bound"] = bool(gate["passed"])
        row["gate"]["passed"] = all(
            value for key, value in row["gate"].items() if key != "passed"
        )
        row_demand_gates.append(gate)
    artifact_sync = dict(remote_sync.get("veto_artifact_root", {}))
    extra_gate = {
        "explicit_training_protocol_bound": protocol["training_protocol"][
            "sha256"
        ]
        == _sha256(training_protocol_path),
        "explicit_training_result_bound": protocol[
            "target_veto_training_result"
        ]["sha256"]
        == _sha256(training_result_path),
        "remote_training_result_sync_bound": remote_sync.get(
            "training_result", {}
        ).get("sha256")
        == _sha256(training_result_path),
        "remote_joint_audit_sync_bound": remote_sync.get(
            "veto_joint_audit", {}
        ).get("sha256")
        == _sha256(veto_joint_audit_path),
        "remote_artifact_file_count_exact": int(
            artifact_sync.get("file_count", -1)
        )
        == 2 * len(protocol["city_target_veto_artifacts"]),
        "cross_node_input_visibility_passed": launch.get(
            "cross_node_input_visibility", {}
        ).get("passed")
        is True
        and int(
            launch.get("cross_node_input_visibility", {}).get(
                "expected_file_count", -1
            )
        )
        == 2 + 2 * len(protocol["city_target_veto_artifacts"]),
        "all_rollout_demand_profiles_and_estimands_bound": len(row_demand_gates)
        == len(all_method_rollout_identities(protocol))
        and all(bool(row["passed"]) for row in row_demand_gates),
    }
    extra_gate["passed"] = all(extra_gate.values())
    payload["v71_extra_gate"] = extra_gate
    integrity = payload["integrity_gate"]
    integrity["v71_extra_gate_passed"] = bool(extra_gate["passed"])
    integrity["all_method_row_gates_passed"] = bool(payload.get("rows")) and all(
        bool(row["gate"]["passed"]) for row in payload["rows"]
    )
    integrity["passed"] = all(
        value for key, value in integrity.items() if key != "passed"
    )
    advance = payload["advance_gate"]
    advance["integrity_passed"] = bool(integrity["passed"])
    advance["passed"] = all(
        value for key, value in advance.items() if key != "passed"
    )
    payload.update(
        {
            "protocol": AUDIT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if integrity["passed"] else "FAIL",
            "decision": (
                ADVANCE_DECISION if advance["passed"] else REJECTION_DECISION
            ),
            "claim_boundary": (
                "Disclosed eight-seed, target-simulator-labeled few-shot "
                "development only. The causal proposal is frozen; the new component "
                "is a demand-conditioned execution veto. Validation and prospective "
                "seeds remain sealed."
            ),
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v71 audit: {args.out}")
    payload = audit(
        results_root=args.results_root,
        launch_manifest_path=args.launch_manifest,
        protocol_path=args.protocol,
        training_protocol_path=args.training_protocol,
        training_result_path=args.training_result,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
        veto_joint_audit_path=args.veto_joint_audit,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "selected": payload["selector"]["selected_config"],
                "integrity": payload["integrity_gate"]["passed"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
