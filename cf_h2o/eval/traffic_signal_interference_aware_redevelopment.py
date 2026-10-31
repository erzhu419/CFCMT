"""Run one frozen v86 partial-interference closed-loop rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import _read_json
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    _build_runtime_models,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
    evaluate_policy_v3,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze import (
    load_uncertainty_horizon_originator,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_closed_loop_development import (
    _validate_evidence as _validate_v83_evidence,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (
    ARTIFACT_KEY,
    CANDIDATE_SPECS,
    METHOD_KEYS,
    PHASE_POLICY,
    PROTOCOL,
    V83_AUDIT_PROTOCOL,
    V83_AUTHORIZATION,
    V83_PROTOCOL,
    V84_AUDIT_PROTOCOL,
    V84_PROTOCOL,
    V84_REJECTION,
)


RESULT_PROTOCOL = "tsc-v86r82-interference-aware-redevelopment-rollout-v1"
ANALYSIS_STAGE = "post_v84_interference_aware_redevelopment"


@dataclass(frozen=True)
class InterferenceExecutionCandidate:
    key: str
    artifact_key: str
    candidate_role: str
    selection_eligible_closed_loop: bool
    prediction_horizon_sec: int
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def redevelopment_candidates(
    protocol: Mapping[str, Any],
) -> tuple[InterferenceExecutionCandidate, ...]:
    rows = tuple(protocol["development"]["method_candidates"])
    candidates = tuple(
        InterferenceExecutionCandidate(
            key=str(row["key"]),
            artifact_key=str(row["artifact_key"]),
            candidate_role=str(row["candidate_role"]),
            selection_eligible_closed_loop=bool(
                row["selection_eligible_closed_loop"]
            ),
            prediction_horizon_sec=int(row["prediction_horizon_sec"]),
            runtime_policy=str(row["runtime_policy"]),
            coordination_mode=str(row["coordination_mode"]),
            cooldown_intervals=int(row["cooldown_intervals"]),
            execution_trust_region=ResidualExecutionTrustRegionConfig(
                **dict(row["execution_trust_region"])
            ),
        )
        for row in rows
    )
    observed = tuple(
        (
            row.key,
            row.coordination_mode,
            row.cooldown_intervals,
            row.execution_trust_region.global_cooldown_intervals,
            row.execution_trust_region.max_simultaneous_overrides,
        )
        for row in candidates
    )
    if not (
        observed == CANDIDATE_SPECS
        and tuple(row.key for row in candidates) == METHOD_KEYS
        and all(
            row.artifact_key == ARTIFACT_KEY
            and row.prediction_horizon_sec == 120
            and row.runtime_policy == HIERARCHICAL_RUNTIME_POLICY
            and row.selection_eligible_closed_loop
            and row.execution_trust_region.enabled
            and row.execution_trust_region.min_priority == 0.0
            and row.execution_trust_region.min_total_vehicles == 1.0
            for row in candidates
        )
    ):
        raise ValueError("v86 interference candidate grid changed")
    return candidates


def all_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    development = dict(protocol["development"])
    seeds = tuple(int(value) for value in development["seeds"])
    policies = tuple(str(value) for value in development["policies"])
    identities = tuple(
        (str(city), str(scenario), seed, policy)
        for city, scenarios in sorted(development["city_scenarios"].items())
        for scenario in scenarios
        for seed in seeds
        for policy in policies
    )
    if not (
        policies == (PHASE_POLICY, *METHOD_KEYS)
        and len(seeds) == 54
        and len(identities) == int(development["matrix_size"]) == 270
        and len(identities) == len(set(identities))
    ):
        raise ValueError("v86 rollout matrix changed")
    return identities


def candidate_payload(candidate: InterferenceExecutionCandidate) -> dict[str, Any]:
    execution_trust_region = asdict(candidate.execution_trust_region)
    if execution_trust_region["max_stay_overrides_per_global_cooldown"] is None:
        execution_trust_region.pop("max_stay_overrides_per_global_cooldown")
    return {
        "key": candidate.key,
        "artifact_key": candidate.artifact_key,
        "candidate_role": candidate.candidate_role,
        "selection_eligible_closed_loop": candidate.selection_eligible_closed_loop,
        "prediction_horizon_sec": candidate.prediction_horizon_sec,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "execution_trust_region": execution_trust_region,
    }


def stay_aware_execution_candidate(
    candidate: InterferenceExecutionCandidate,
    *,
    key: str,
    candidate_role: str,
    max_stay_overrides_per_global_cooldown: int | None = None,
) -> InterferenceExecutionCandidate:
    """Enable phase-continuation overrides without changing the fitted model."""

    return replace(
        candidate,
        key=str(key),
        candidate_role=str(candidate_role),
        execution_trust_region=replace(
            candidate.execution_trust_region,
            allow_stay_during_global_cooldown=True,
            max_stay_overrides_per_global_cooldown=(
                max_stay_overrides_per_global_cooldown
            ),
        ),
    )


def _global_alignment(
    metrics: Mapping[str, Any], *, expected_horizon_sec: int
) -> dict[str, Any]:
    accepted = tuple(
        row
        for row in metrics.get("accepted_intervention_trace", ())
        if bool(row.get("request_accepted", False))
    )
    times = np.asarray([float(row["time_sec"]) for row in accepted], dtype=float)
    gaps = np.diff(times)
    minimum_gap = float(np.min(gaps)) if gaps.size else None
    return {
        "protocol": "network-global-horizon-alignment-v1",
        "scope": "network_global",
        "expected_minimum_gap_sec": int(expected_horizon_sec),
        "accepted_request_count": len(accepted),
        "accepted_request_times_sec": times.tolist(),
        "minimum_observed_conflicting_gap_sec": minimum_gap,
        "independent_overlap_pair_count": 0,
        "violation_count": int(
            np.sum(gaps < float(expected_horizon_sec) - 1e-9)
        ),
        "passed": bool(
            not gaps.size
            or np.all(gaps >= float(expected_horizon_sec) - 1e-9)
        ),
    }


def _partial_interference_alignment(
    metrics: Mapping[str, Any], *, expected_horizon_sec: int
) -> dict[str, Any]:
    accepted = tuple(
        row
        for row in metrics.get("accepted_intervention_trace", ())
        if bool(row.get("request_accepted", False))
    )
    deployment = dict(metrics.get("residual_deployment", {}))
    graph = dict(deployment.get("intervention_graph", {}))
    conflicts = {
        str(tls_id): {str(value) for value in values}
        for tls_id, values in dict(graph.get("conflicts", {})).items()
    }
    if not conflicts:
        raise ValueError("partial-interference alignment lacks a conflict graph")
    violations = []
    conflicting_gaps = []
    independent_overlaps = 0
    for left_index, left in enumerate(accepted):
        left_time = float(left["time_sec"])
        left_tls = str(left["tls_id"])
        for right in accepted[left_index + 1 :]:
            right_time = float(right["time_sec"])
            gap = right_time - left_time
            if gap < -1e-9:
                raise ValueError("intervention trace is not time ordered")
            right_tls = str(right["tls_id"])
            conflict = bool(
                left_tls == right_tls
                or right_tls in conflicts.get(left_tls, set())
                or left_tls in conflicts.get(right_tls, set())
            )
            if conflict:
                conflicting_gaps.append(gap)
                if gap < float(expected_horizon_sec) - 1e-9:
                    violations.append(
                        {
                            "left_time_sec": left_time,
                            "right_time_sec": right_time,
                            "gap_sec": gap,
                            "left_tls": left_tls,
                            "right_tls": right_tls,
                        }
                    )
            elif gap < float(expected_horizon_sec) - 1e-9:
                independent_overlaps += 1
    return {
        "protocol": "conflict-graph-partial-interference-horizon-alignment-v1",
        "scope": "same_or_conflicting_tls_neighborhood",
        "expected_minimum_gap_sec": int(expected_horizon_sec),
        "accepted_request_count": len(accepted),
        "minimum_observed_conflicting_gap_sec": (
            float(min(conflicting_gaps)) if conflicting_gaps else None
        ),
        "independent_overlap_pair_count": int(independent_overlaps),
        "violation_count": len(violations),
        "violations": violations[:16],
        "passed": not violations,
    }


def intervention_alignment(
    metrics: Mapping[str, Any], *, candidate: InterferenceExecutionCandidate
) -> dict[str, Any]:
    if candidate.coordination_mode == "direct":
        return _global_alignment(
            metrics, expected_horizon_sec=candidate.prediction_horizon_sec
        )
    return _partial_interference_alignment(
        metrics, expected_horizon_sec=candidate.prediction_horizon_sec
    )


def _validate_redevelopment_parents(
    *,
    protocol: Mapping[str, Any],
    v83_protocol_path: Path,
    v83_audit_path: Path,
    v84_protocol_path: Path,
    v84_audit_path: Path,
) -> None:
    v83 = _read_json(v83_protocol_path)
    v83_audit = _read_json(v83_audit_path)
    v84 = _read_json(v84_protocol_path)
    v84_audit = _read_json(v84_audit_path)
    parent = dict(protocol["redevelopment_parent"])
    if not (
        v83.get("protocol") == V83_PROTOCOL
        and v83_audit.get("protocol") == V83_AUDIT_PROTOCOL
        and v83_audit.get("decision") == V83_AUTHORIZATION
        and v84.get("protocol") == V84_PROTOCOL
        and v84_audit.get("protocol") == V84_AUDIT_PROTOCOL
        and v84_audit.get("decision") == V84_REJECTION
        and v84_audit.get("confirmation_gate", {}).get("passed") is False
        and _sha256(v83_protocol_path) == parent["v83_protocol"]["sha256"]
        and _sha256(v83_audit_path) == parent["v83_independent_audit"]["sha256"]
        and _sha256(v84_protocol_path) == parent["v84_protocol"]["sha256"]
        and _sha256(v84_audit_path) == parent["v84_independent_audit"]["sha256"]
    ):
        raise ValueError("v86 redevelopment parent evidence changed")


def run_interference_aware_redevelopment(
    *,
    protocol_path: Path,
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
    candidate_override: InterferenceExecutionCandidate | None = None,
    result_protocol: str = RESULT_PROTOCOL,
    analysis_stage: str = ANALYSIS_STAGE,
    reported_policy: str | None = None,
    selection_eligible_evidence: bool = True,
    diagnostic_only_evidence: bool = False,
    externally_frozen_identity: tuple[str, str, int, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v86 protocol changed")
    _validate_redevelopment_parents(
        protocol=protocol,
        v83_protocol_path=v83_protocol_path,
        v83_audit_path=v83_audit_path,
        v84_protocol_path=v84_protocol_path,
        v84_audit_path=v84_audit_path,
    )
    _validate_v83_evidence(
        protocol=protocol,
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
        environment_protocol_path=environment_protocol_path,
        environment_parent_path=environment_parent_path,
        external_manifest_path=external_manifest_path,
        conversion_manifest_path=conversion_manifest_path,
    )
    identity = ("jinan", str(scenario), int(seed), str(policy))
    identity_in_v86_matrix = identity in set(all_rollout_identities(protocol))
    if not identity_in_v86_matrix and identity != externally_frozen_identity:
        raise ValueError("rollout identity is outside frozen v86 redevelopment")
    candidates = {row.key: row for row in redevelopment_candidates(protocol)}
    if candidate_override is not None and policy == PHASE_POLICY:
        raise ValueError("phase-pressure baseline cannot accept a residual candidate override")

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    environment = dict(protocol["environment"])
    if not (
        scenario in manifest.sumocfgs
        and manifest.city_groups[scenario] == "jinan"
        and conversion_root in Path(manifest.sumocfgs[scenario]).resolve().parents
        and libsumo_version() == str(environment["sumo_version"])
    ):
        raise ValueError("v86 SUMO environment changed")

    runtime_models = None
    runtime_policy = PHASE_POLICY
    execution_region = None
    cooldown_intervals = None
    coordination_mode = "direct"
    hierarchy_model_hash = None
    hierarchy_certificate_hash = None
    originator_diagnostics = None
    candidate = None
    artifact_model_sha = None
    artifact_certificate_sha = None
    if policy != PHASE_POLICY:
        hierarchical_parent = _read_json(hierarchical_parent_path)
        hierarchical_freeze_audit = _read_json(hierarchical_freeze_audit_path)
        _, frozen_model = _load_frozen_city_model(
            city="jinan",
            freeze_root=hierarchical_freeze_root,
            protocol=hierarchical_parent,
            freeze_audit=hierarchical_freeze_audit,
        )
        hierarchy_model_hash = str(
            hierarchical_parent["city_artifacts"]["jinan"]["model"]["sha256"]
        )
        hierarchy_certificate_hash = str(
            hierarchical_parent["city_artifacts"]["jinan"]["certificate"][
                "sha256"
            ]
        )
        candidate = candidates[policy]
        if candidate_override is not None:
            expected_override = stay_aware_execution_candidate(
                candidate,
                key=candidate_override.key,
                candidate_role=candidate_override.candidate_role,
                max_stay_overrides_per_global_cooldown=(
                    candidate_override.execution_trust_region.max_stay_overrides_per_global_cooldown
                ),
            )
            if not (
                candidate_override == expected_override
                and candidate_override.execution_trust_region.allow_stay_during_global_cooldown
            ):
                raise ValueError(
                    "candidate override changed more than the declared stay-aware execution rule"
                )
            candidate = candidate_override
        _, runtime_models = _build_runtime_models(
            model=frozen_model,
            prediction_horizon_sec=candidate.prediction_horizon_sec,
        )
        originator, _, _ = load_uncertainty_horizon_originator(
            artifact_root, candidate.artifact_key
        )
        runtime_models = replace(runtime_models, action_originator=originator)
        runtime_policy = candidate.runtime_policy
        execution_region = candidate.execution_trust_region
        cooldown_intervals = candidate.cooldown_intervals
        coordination_mode = candidate.coordination_mode
        originator_diagnostics = originator.diagnostics()
        artifact_record = protocol["artifact"]["artifacts"][candidate.artifact_key]
        artifact_model_sha = str(artifact_record["model"]["sha256"])
        artifact_certificate_sha = str(artifact_record["certificate"]["sha256"])
        if not (
            runtime_models.action_originator is originator
            and runtime_models.execution_veto is None
            and runtime_models.prediction_horizon_sec == 120
            and originator.config.rollout_value_horizon_sec == 120
        ):
            raise RuntimeError("v86 originator injection changed")

    temporary_tripinfo = tripinfo_out.with_suffix(
        tripinfo_out.suffix + f".tmp-{os.getpid()}"
    )
    alignment = None
    try:
        metrics = evaluate_policy_v3(
            sumo_api=load_libsumo(),
            sumocfg=manifest.sumocfgs[scenario],
            scenario=scenario,
            policy=runtime_policy,
            models=runtime_models,
            duration_sec=float(environment["duration_sec"]),
            control_interval_sec=int(environment["control_interval_sec"]),
            warmup_sec=float(environment["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode=coordination_mode,
            residual_cooldown_intervals_override=cooldown_intervals,
            residual_execution_trust_region=execution_region,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"v86 rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("v86 rollout contains teleports")
        if policy != PHASE_POLICY:
            guard = dict(metrics.get("guard_audit", {}))
            if int(guard.get("target_originator_decisions", 0)) <= 0:
                raise RuntimeError("v86 rollout did not enter the originator")
            alignment = intervention_alignment(metrics, candidate=candidate)
            if alignment["passed"] is not True:
                raise RuntimeError("v86 rollout violated partial-interference horizon")
            if int(alignment["accepted_request_count"]) != int(
                guard.get("executed_overrides", 0)
            ):
                raise RuntimeError("v86 intervention trace is incomplete")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("v86 rollout lacks tripinfo evidence")
        temporary_tripinfo.replace(tripinfo_out)
    except Exception:
        temporary_tripinfo.unlink(missing_ok=True)
        raise

    return {
        "protocol": str(result_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "analysis_stage": str(analysis_stage),
        "selection_eligible_evidence": bool(selection_eligible_evidence),
        "diagnostic_only_evidence": bool(diagnostic_only_evidence),
        "identity_authorization": (
            "v86_frozen_development_matrix"
            if identity_in_v86_matrix
            else "external_frozen_protocol_exact_identity"
        ),
        "untouched_validation_evidence": False,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "v83_protocol_sha256": _sha256(v83_protocol_path),
        "v83_audit_sha256": _sha256(v83_audit_path),
        "v84_protocol_sha256": _sha256(v84_protocol_path),
        "v84_audit_sha256": _sha256(v84_audit_path),
        "artifact_key": candidate.artifact_key if candidate else None,
        "artifact_model_sha256": artifact_model_sha,
        "artifact_certificate_sha256": artifact_certificate_sha,
        "parent_runtime_protocol_sha256": _sha256(parent_runtime_protocol_path),
        "hierarchical_parent_sha256": _sha256(hierarchical_parent_path),
        "hierarchical_model_sha256": hierarchy_model_hash,
        "hierarchical_certificate_sha256": hierarchy_certificate_hash,
        "environment_protocol_sha256": _sha256(environment_protocol_path),
        "environment_parent_sha256": _sha256(environment_parent_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": str(environment["conversion_tree_sha256"]),
        "sumo_version": libsumo_version(),
        "city": "jinan",
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(reported_policy or policy),
        "runtime_policy": runtime_policy,
        "controller_refit": False,
        "target_labels_used_at_fit": policy != PHASE_POLICY,
        "target_outcomes_used_online": False,
        "prediction_horizon_sec": 120 if candidate else None,
        "proposal_origin": (
            "target_uncertainty_aware_state_action_latent"
            if candidate
            else "phase_pressure"
        ),
        "candidate_role": candidate.candidate_role if candidate else "baseline",
        "execution_candidate": candidate_payload(candidate) if candidate else None,
        "originator_diagnostics": originator_diagnostics,
        "intervention_alignment": alignment,
        "tripinfo_evidence": {
            "filename": tripinfo_out.name,
            "sha256": _sha256(tripinfo_out),
            "size_bytes": int(tripinfo_out.stat().st_size),
        },
        "metrics": metrics,
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite v86 rollout")
    result = run_interference_aware_redevelopment(
        protocol_path=args.protocol,
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
                "executed_overrides": result["metrics"].get(
                    "guard_audit", {}
                ).get("executed_overrides", 0),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
