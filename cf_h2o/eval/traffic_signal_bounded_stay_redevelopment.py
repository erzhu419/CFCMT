"""Run one frozen v88 bounded-stay development rollout."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    stay_aware_execution_candidate,
)
from scripts.cluster.audit_tsc_external_phase_transition_aware_redevelopment import (
    AUDIT_PROTOCOL as V87_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (
    PROTOCOL as V86_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (
    BOUNDED_STAY_POLICY,
    BOUNDED_STAY_ROLE,
    LEGACY_POLICY,
    PROTOCOL,
    V86_AUDIT_PROTOCOL,
    V86_REJECTION,
    V87_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_phase_transition_aware_redevelopment import (
    PROTOCOL as V87_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v88r84-bounded-stay-redevelopment-rollout-v1"
ANALYSIS_STAGE = "post_v87_bounded_stay_adaptive_redevelopment"
ALIGNMENT_PROTOCOL = "bounded-stay-global-horizon-alignment-v1"


def bounded_stay_candidate(
    protocol: Mapping[str, Any], v86_protocol: Mapping[str, Any]
) -> InterferenceExecutionCandidate:
    row = dict(protocol["development"]["method_candidate"])
    candidate = InterferenceExecutionCandidate(
        key=str(row["key"]),
        artifact_key=str(row["artifact_key"]),
        candidate_role=str(row["candidate_role"]),
        selection_eligible_closed_loop=bool(row["selection_eligible_closed_loop"]),
        prediction_horizon_sec=int(row["prediction_horizon_sec"]),
        runtime_policy=str(row["runtime_policy"]),
        coordination_mode=str(row["coordination_mode"]),
        cooldown_intervals=int(row["cooldown_intervals"]),
        execution_trust_region=redevelopment_candidates(v86_protocol)[
            0
        ].execution_trust_region,
    )
    candidate = replace(
        candidate,
        execution_trust_region=type(candidate.execution_trust_region)(
            **dict(row["execution_trust_region"])
        ),
    )
    base = {
        value.key: value for value in redevelopment_candidates(v86_protocol)
    }[LEGACY_POLICY]
    expected = stay_aware_execution_candidate(
        base,
        key=BOUNDED_STAY_POLICY,
        candidate_role=BOUNDED_STAY_ROLE,
        max_stay_overrides_per_global_cooldown=1,
    )
    if candidate != expected or candidate_payload(candidate) != row:
        raise ValueError("v88 bounded-stay candidate changed")
    return candidate


def all_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    development = dict(protocol["development"])
    identities = tuple(
        ("jinan", "jinan_3x4_real", int(seed), BOUNDED_STAY_POLICY)
        for seed in development["seeds"]
    )
    if not (
        development["rollout_policies"] == [BOUNDED_STAY_POLICY]
        and len(identities) == len(set(identities)) == 54
        and int(development["new_rollout_count"]) == 54
    ):
        raise ValueError("v88 rollout identity set changed")
    return identities


def phase_transition_alignment(
    metrics: Mapping[str, Any], *, expected_horizon_sec: int
) -> dict[str, Any]:
    effective = tuple(
        dict(row)
        for row in metrics.get("accepted_intervention_trace", ())
        if bool(row.get("execution_effective", row.get("request_accepted", False)))
    )
    switches = tuple(
        row
        for row in effective
        if str(row.get("override_kind", "switch")) == "switch"
    )
    stays = tuple(
        row for row in effective if str(row.get("override_kind")) == "stay"
    )
    switch_times = np.asarray(
        [float(row["time_sec"]) for row in switches], dtype=float
    )
    switch_gaps = np.diff(switch_times)
    violations = switch_gaps < float(expected_horizon_sec) - 1e-9
    bounded_stays = tuple(
        row
        for row in stays
        if bool(row.get("global_cooldown_active_before", False))
    )
    stay_budget_violations = tuple(
        row
        for row in bounded_stays
        if int(row.get("stay_bypass_remaining_before", 0)) < 1
    )
    return {
        "protocol": ALIGNMENT_PROTOCOL,
        "scope": "network_global_phase_transitions",
        "expected_minimum_switch_gap_sec": int(expected_horizon_sec),
        "effective_override_count": len(effective),
        "switch_override_count": len(switches),
        "stay_override_count": len(stays),
        "switch_times_sec": switch_times.tolist(),
        "minimum_observed_switch_gap_sec": (
            float(np.min(switch_gaps)) if switch_gaps.size else None
        ),
        "switch_gap_violation_count": int(np.sum(violations)),
        "stay_overrides_bypass_phase_transition_cooldown": True,
        "max_stay_overrides_per_global_cooldown": 1,
        "bounded_stay_override_count": len(bounded_stays),
        "stay_budget_violation_count": len(stay_budget_violations),
        "passed": bool(
            (not switch_gaps.size or not np.any(violations))
            and not stay_budget_violations
        ),
    }


def _validate_v88_parent(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v86_audit_path: Path,
    v87_protocol_path: Path,
    v87_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], InterferenceExecutionCandidate]:
    protocol = _read_json(protocol_path)
    v86 = _read_json(v86_protocol_path)
    v86_audit = _read_json(v86_audit_path)
    v87 = _read_json(v87_protocol_path)
    v87_audit = _read_json(v87_audit_path)
    parent = dict(protocol["redevelopment_parent"])
    if not (
        protocol.get("protocol") == PROTOCOL
        and v86.get("protocol") == V86_PROTOCOL
        and v86_audit.get("protocol") == V86_AUDIT_PROTOCOL
        and v86_audit.get("status") == "PASS"
        and v86_audit.get("decision") == V86_REJECTION
        and v87.get("protocol") == V87_PROTOCOL
        and v87_audit.get("protocol") == V87_AUDIT_PROTOCOL
        and v87_audit.get("status") == "PASS"
        and v87_audit.get("decision") == V87_DECISION
        and parent["v86_protocol"]["sha256"] == _sha256(v86_protocol_path)
        and parent["v86_independent_audit"]["sha256"] == _sha256(v86_audit_path)
        and parent["v87_protocol"]["sha256"] == _sha256(v87_protocol_path)
        and parent["v87_independent_audit"]["sha256"]
        == _sha256(v87_audit_path)
    ):
        raise ValueError("v88 parent evidence changed")
    return protocol, v86, bounded_stay_candidate(protocol, v86)


def run_bounded_stay_redevelopment(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v86_audit_path: Path,
    v87_protocol_path: Path,
    v87_audit_path: Path,
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
    protocol, _, candidate = _validate_v88_parent(
        protocol_path=protocol_path,
        v86_protocol_path=v86_protocol_path,
        v86_audit_path=v86_audit_path,
        v87_protocol_path=v87_protocol_path,
        v87_audit_path=v87_audit_path,
    )
    identity = ("jinan", str(scenario), int(seed), str(policy))
    if identity not in set(all_rollout_identities(protocol)):
        raise ValueError("rollout identity is outside frozen v88 development")

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
        policy=LEGACY_POLICY,
        tripinfo_out=tripinfo_out,
        candidate_override=candidate,
        result_protocol=RESULT_PROTOCOL,
        analysis_stage=ANALYSIS_STAGE,
        reported_policy=BOUNDED_STAY_POLICY,
        selection_eligible_evidence=True,
        diagnostic_only_evidence=False,
    )
    metrics = dict(result["metrics"])
    guard = dict(metrics["guard_audit"])
    alignment = phase_transition_alignment(
        metrics, expected_horizon_sec=candidate.prediction_horizon_sec
    )
    if not (
        alignment["passed"] is True
        and int(alignment["effective_override_count"])
        == int(guard["effective_overrides"])
        and int(alignment["switch_override_count"])
        == int(guard["executed_switch_overrides"])
        and int(alignment["stay_override_count"])
        == int(guard["executed_stay_overrides"])
        and metrics["residual_deployment"]["global_cooldown_contract"]
        == "phase_transition_only_cooldown_with_bounded_stay_override_bypass"
        and metrics["residual_deployment"]["execution_trust_region"][
            "max_stay_overrides_per_global_cooldown"
        ]
        == 1
        and int(metrics["starting_teleports"]) == 0
        and int(metrics["ending_teleports"]) == 0
    ):
        raise RuntimeError("v88 execution semantics or intervention trace is incomplete")

    result["v86_parent_protocol_sha256"] = result["protocol_sha256"]
    result["protocol_sha256"] = _sha256(protocol_path)
    result["v86_parent_audit_sha256"] = _sha256(v86_audit_path)
    result["v87_parent_protocol_sha256"] = _sha256(v87_protocol_path)
    result["v87_parent_audit_sha256"] = _sha256(v87_audit_path)
    result["intervention_alignment"] = alignment
    result["execution_semantics"] = {
        "protocol": "bounded-stay-residual-execution-v1",
        "stay_override": "retain_current_green_against_a_prior_switch",
        "switch_override": "request_a_green_state_different_from_current_green",
        "global_cooldown_applies_to": "switch_overrides_only",
        "stay_overrides_are_effective_but_do_not_consume_switch_budget": True,
        "stay_overrides_consume_bounded_stay_budget": True,
        "maximum_stay_overrides_per_active_global_cooldown": 1,
    }
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v86-audit", type=Path, required=True)
    parser.add_argument("--v87-protocol", type=Path, required=True)
    parser.add_argument("--v87-audit", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite v88 rollout")
    result = run_bounded_stay_redevelopment(
        protocol_path=args.protocol,
        v86_protocol_path=args.v86_protocol,
        v86_audit_path=args.v86_audit,
        v87_protocol_path=args.v87_protocol,
        v87_audit_path=args.v87_audit,
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
                "effective_overrides": result["metrics"]["guard_audit"][
                    "effective_overrides"
                ],
                "collision_incidents": result["metrics"]["collision_incidents"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
