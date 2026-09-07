"""Run one frozen v89 fresh-seed confirmation rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import _read_json
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (
    InterferenceExecutionCandidate,
    candidate_payload,
    redevelopment_candidates,
    run_interference_aware_redevelopment,
)
from scripts.cluster.audit_tsc_external_bounded_stay_redevelopment import (
    AUDIT_PROTOCOL as V88_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (
    PROTOCOL as V88_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_fresh_confirmation import (
    METHOD_POLICY,
    PHASE_POLICY,
    PROTOCOL,
    V88_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (
    PROTOCOL as V86_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v89r85-fresh-independent-confirmation-rollout-v1"
ANALYSIS_STAGE = "fresh_independent_seed_confirmation"
ALIGNMENT_PROTOCOL = "v89-frozen-global-switch-horizon-alignment-v1"


def selected_candidate(
    protocol: Mapping[str, Any], v86_protocol: Mapping[str, Any]
) -> InterferenceExecutionCandidate:
    expected = {
        candidate.key: candidate
        for candidate in redevelopment_candidates(v86_protocol)
    }[METHOD_POLICY]
    row = dict(protocol["confirmation"]["method_candidate"])
    if candidate_payload(expected) != row:
        raise ValueError("v89 selected candidate changed")
    return expected


def all_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    confirmation = dict(protocol["confirmation"])
    identities = tuple(
        ("jinan", "jinan_3x4_real", int(seed), str(policy))
        for seed in confirmation["seeds"]
        for policy in confirmation["policies"]
    )
    if not (
        confirmation["policies"] == [PHASE_POLICY, METHOD_POLICY]
        and len(identities) == len(set(identities)) == 128
        and int(confirmation["matrix_size"]) == 128
    ):
        raise ValueError("v89 rollout identity set changed")
    return identities


def global_switch_alignment(
    metrics: Mapping[str, Any], *, expected_horizon_sec: int
) -> dict[str, Any]:
    effective = tuple(
        dict(row)
        for row in metrics.get("accepted_intervention_trace", ())
        if bool(row.get("execution_effective", row.get("request_accepted", False)))
    )
    switches = tuple(
        row for row in effective if str(row.get("override_kind")) == "switch"
    )
    stays = tuple(
        row for row in effective if str(row.get("override_kind")) == "stay"
    )
    stays_during_cooldown = tuple(
        row
        for row in stays
        if bool(row.get("global_cooldown_active_before", False))
    )
    switch_times = np.asarray(
        [float(row["time_sec"]) for row in switches], dtype=float
    )
    switch_gaps = np.diff(switch_times)
    violations = switch_gaps < float(expected_horizon_sec) - 1e-9
    return {
        "protocol": ALIGNMENT_PROTOCOL,
        "scope": "network_global_phase_transitions",
        "expected_minimum_switch_gap_sec": int(expected_horizon_sec),
        "effective_override_count": len(effective),
        "switch_override_count": len(switches),
        "stay_override_count": len(stays),
        "stay_during_global_cooldown_count": len(stays_during_cooldown),
        "minimum_observed_switch_gap_sec": (
            float(np.min(switch_gaps)) if switch_gaps.size else None
        ),
        "switch_gap_violation_count": int(np.sum(violations)),
        "passed": bool(
            (not switch_gaps.size or not np.any(violations))
            and not stays_during_cooldown
        ),
    }


def _validate_v89_parent(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v88_protocol_path: Path,
    v88_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], InterferenceExecutionCandidate]:
    protocol = _read_json(protocol_path)
    v86 = _read_json(v86_protocol_path)
    v88 = _read_json(v88_protocol_path)
    v88_audit = _read_json(v88_audit_path)
    parent = dict(protocol["confirmation_parent"])
    if not (
        protocol.get("protocol") == PROTOCOL
        and v86.get("protocol") == V86_PROTOCOL
        and v88.get("protocol") == V88_PROTOCOL
        and v88_audit.get("protocol") == V88_AUDIT_PROTOCOL
        and v88_audit.get("status") == "PASS"
        and v88_audit.get("decision") == V88_DECISION
        and v88_audit.get("selection_gate", {}).get("selected_candidate")
        == METHOD_POLICY
        and parent["v86_protocol"]["sha256"] == _sha256(v86_protocol_path)
        and parent["v88_protocol"]["sha256"] == _sha256(v88_protocol_path)
        and parent["v88_independent_audit"]["sha256"]
        == _sha256(v88_audit_path)
    ):
        raise ValueError("v89 parent evidence changed")
    return protocol, v86, selected_candidate(protocol, v86)


def run_fresh_confirmation(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v88_protocol_path: Path,
    v88_audit_path: Path,
    v83_protocol_path: Path,
    v83_audit_path: Path,
    v84_protocol_path: Path,
    v84_audit_path: Path,
    v81_protocol_path: Path,
    v81_audit_path: Path,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
    artifact_audit_path: Path,
    artifact_root: Path,
    parent_runtime_protocol_path: Path,
    hierarchical_parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
    scenario: str,
    seed: int,
    policy: str,
    tripinfo_out: Path,
) -> dict[str, Any]:
    protocol, _, candidate = _validate_v89_parent(
        protocol_path=protocol_path,
        v86_protocol_path=v86_protocol_path,
        v88_protocol_path=v88_protocol_path,
        v88_audit_path=v88_audit_path,
    )
    identity = ("jinan", str(scenario), int(seed), str(policy))
    if identity not in set(all_rollout_identities(protocol)):
        raise ValueError("rollout identity is outside frozen v89 confirmation")

    result = run_interference_aware_redevelopment(
        protocol_path=v86_protocol_path,
        v83_protocol_path=v83_protocol_path,
        v83_audit_path=v83_audit_path,
        v84_protocol_path=v84_protocol_path,
        v84_audit_path=v84_audit_path,
        v81_protocol_path=v81_protocol_path,
        v81_audit_path=v81_audit_path,
        artifact_protocol_path=artifact_protocol_path,
        artifact_result_path=artifact_result_path,
        artifact_audit_path=artifact_audit_path,
        artifact_root=artifact_root,
        parent_runtime_protocol_path=parent_runtime_protocol_path,
        hierarchical_parent_path=hierarchical_parent_path,
        offline_result_path=offline_result_path,
        support_audit_path=support_audit_path,
        hierarchical_freeze_audit_path=hierarchical_freeze_audit_path,
        hierarchical_freeze_root=hierarchical_freeze_root,
        environment_protocol_path=environment_protocol_path,
        environment_parent_path=environment_parent_path,
        external_manifest_path=external_manifest_path,
        conversion_root=conversion_root,
        conversion_manifest_path=conversion_manifest_path,
        scenario=str(scenario),
        seed=int(seed),
        policy=str(policy),
        tripinfo_out=tripinfo_out,
        result_protocol=RESULT_PROTOCOL,
        analysis_stage=ANALYSIS_STAGE,
        selection_eligible_evidence=False,
        diagnostic_only_evidence=False,
        externally_frozen_identity=identity,
    )
    metrics = dict(result["metrics"])
    alignment = None
    if policy == METHOD_POLICY:
        guard = dict(metrics["guard_audit"])
        alignment = global_switch_alignment(
            metrics, expected_horizon_sec=candidate.prediction_horizon_sec
        )
        if not (
            alignment["passed"] is True
            and int(alignment["effective_override_count"])
            == int(guard["effective_overrides"])
            and metrics["residual_deployment"]["global_cooldown_contract"]
            == "one_executed_residual_then_network_prior_only"
            and metrics["residual_deployment"]["execution_trust_region"][
                "max_stay_overrides_per_global_cooldown"
            ]
            is None
        ):
            raise RuntimeError("v89 frozen execution semantics changed")
    if not (
        result.get("identity_authorization")
        == "external_frozen_protocol_exact_identity"
        and int(metrics["starting_teleports"]) == 0
        and int(metrics["ending_teleports"]) == 0
    ):
        raise RuntimeError("v89 rollout authorization or teleport gate failed")

    result["v86_parent_protocol_sha256"] = result["protocol_sha256"]
    result["protocol_sha256"] = _sha256(protocol_path)
    result["v88_parent_protocol_sha256"] = _sha256(v88_protocol_path)
    result["v88_parent_audit_sha256"] = _sha256(v88_audit_path)
    result["intervention_alignment"] = alignment
    result["selection_eligible_evidence"] = False
    result["untouched_validation_evidence"] = True
    result["prospective_evidence"] = True
    result["confirmation_evidence"] = True
    result["confirmation_scope"] = "same_network_fresh_simulator_seeds"
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v88-protocol", type=Path, required=True)
    parser.add_argument("--v88-audit", type=Path, required=True)
    parser.add_argument("--v83-protocol", type=Path, required=True)
    parser.add_argument("--v83-audit", type=Path, required=True)
    parser.add_argument("--v84-protocol", type=Path, required=True)
    parser.add_argument("--v84-audit", type=Path, required=True)
    parser.add_argument("--v81-protocol", type=Path, required=True)
    parser.add_argument("--v81-audit", type=Path, required=True)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-runtime-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite v89 rollout")
    result = run_fresh_confirmation(
        protocol_path=args.protocol,
        v86_protocol_path=args.v86_protocol,
        v88_protocol_path=args.v88_protocol,
        v88_audit_path=args.v88_audit,
        v83_protocol_path=args.v83_protocol,
        v83_audit_path=args.v83_audit,
        v84_protocol_path=args.v84_protocol,
        v84_audit_path=args.v84_audit,
        v81_protocol_path=args.v81_protocol,
        v81_audit_path=args.v81_audit,
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
        artifact_audit_path=args.artifact_audit,
        artifact_root=args.artifact_root,
        parent_runtime_protocol_path=args.parent_runtime_protocol,
        hierarchical_parent_path=args.hierarchical_parent,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        scenario=str(args.scenario),
        seed=int(args.seed),
        policy=str(args.policy),
        tripinfo_out=args.tripinfo_out,
    )
    _atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "policy": result["policy"],
                "mean_waiting_time": result["metrics"][
                    "mean_tripinfo_waiting_time"
                ],
                "collision_incidents": result["metrics"]["collision_incidents"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
