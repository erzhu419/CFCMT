"""Run one v52 estimand-aligned global-cooldown development candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import _read_json
from cf_h2o.eval.traffic_signal_external_execution_trust_region_development import (
    ExecutionTrustRegionCandidate,
    PROTOCOL as V51_PROTOCOL,
    trust_region_candidates,
)
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (
    build_candidate_runtime,
    policy_candidates,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    PROTOCOL as PRESSURE_PROTOCOL,
    _load_artifacts,
)


PROTOCOL = "tsc-v52r48-external-estimand-aligned-global-cooldown-development-v1"
RESULT_PROTOCOL = (
    "tsc-v52r48-estimand-aligned-global-cooldown-development-rollout-v1"
)
FAILED_V51_AUDIT_PROTOCOL = (
    "tsc-v51r47-execution-trust-region-development-selection-audit-v1"
)
FAILED_V51_DECISION = "reject_v51_and_preserve_new_validation_and_prospective_seeds"


def global_cooldown_candidates(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> tuple[ExecutionTrustRegionCandidate, ...]:
    candidates = trust_region_candidates(protocol, adaptation)
    if any(
        not candidate.execution_trust_region.enabled
        or candidate.execution_trust_region.max_simultaneous_overrides != 1
        or candidate.execution_trust_region.global_cooldown_intervals <= 0
        for candidate in candidates
    ):
        raise ValueError("v52 candidate violates the global-cooldown contract")
    return candidates


def all_rollout_identities(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> tuple[tuple[str, str, int, str], ...]:
    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    scenario = str(grid["scenario"])
    identities = tuple(
        (city, scenario, int(seed), candidate.key)
        for seed in grid["seeds"]
        for candidate in global_cooldown_candidates(protocol, adaptation)
    )
    expected = int(grid["matrix_size"])
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError("v52 global-cooldown matrix changed")
    return identities


def run_global_cooldown_candidate(
    *,
    global_protocol_path: Path,
    v51_protocol_path: Path,
    adaptation_protocol_path: Path,
    failed_v51_audit_path: Path,
    expected_failed_v51_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    candidate_key: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(global_protocol_path)
    v51_protocol = _read_json(v51_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    failed = _read_json(failed_v51_audit_path)
    pressure = _read_json(pressure_protocol_path)
    joint = _read_json(pressure_joint_audit_path)
    v51_sha = _sha256(v51_protocol_path)
    failed_sha = _sha256(failed_v51_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v52 global-cooldown protocol changed")
    if (
        v51_protocol.get("protocol") != V51_PROTOCOL
        or v51_sha
        != str(protocol["parent_execution_trust_region_protocol"]["sha256"])
        or failed_sha != str(expected_failed_v51_audit_sha256)
        or failed_sha != str(protocol["failed_v51_selection_audit"]["sha256"])
        or failed.get("protocol") != FAILED_V51_AUDIT_PROTOCOL
        or failed.get("status") != "PASS"
        or failed.get("decision") != FAILED_V51_DECISION
        or failed.get("trust_protocol_sha256") != v51_sha
        or _sha256(adaptation_protocol_path)
        != str(protocol["candidate_grid_protocol"]["sha256"])
        or _sha256(adaptation_protocol_path)
        != str(v51_protocol["candidate_grid_protocol"]["sha256"])
        or _sha256(pressure_protocol_path)
        != str(adaptation["parent_pressure_protocol"]["sha256"])
        or pressure.get("protocol") != PRESSURE_PROTOCOL
        or joint_sha != str(expected_pressure_joint_audit_sha256)
        or joint_sha != str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        or joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or joint.get("status") != "PASS"
        or joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v52 lacks frozen failed-development/model evidence")

    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    if scenario != str(grid["scenario"]):
        raise ValueError(f"scenario {scenario} is outside v52 development")
    allowed_seeds = tuple(int(value) for value in grid["seeds"])
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is outside v52 development")
    candidates = {
        candidate.key: candidate
        for candidate in global_cooldown_candidates(protocol, adaptation)
    }
    if candidate_key not in candidates:
        raise ValueError(f"unknown v52 candidate: {candidate_key}")
    candidate = candidates[candidate_key]
    base = {
        value.key: value for value in policy_candidates(adaptation, city)
    }[candidate.base_candidate_key]
    _, model = _load_artifacts(
        protocol=pressure,
        joint=joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )
    source_policy, runtime_override, cooldown = build_candidate_runtime(
        candidate=base,
        model=model,
        rollout_kwargs=rollout_kwargs,
        provenance={
            "global_cooldown_protocol": PROTOCOL,
            "failed_v51_audit_sha256": failed_sha,
            "pressure_joint_audit_sha256": joint_sha,
            "city": city,
        },
    )
    if cooldown != candidate.cooldown_intervals:
        raise ValueError("v52 base cooldown changed")
    diagnostic = ClosedLoopDiagnosticSpec(
        name=candidate.key,
        source_policy=source_policy,
        coordination_mode=candidate.coordination_mode,
        result_protocol=RESULT_PROTOCOL,
        cooldown_intervals_override=candidate.cooldown_intervals,
        execution_trust_region=candidate.execution_trust_region,
        allowed_seeds=allowed_seeds,
        analysis_status="post_v51_estimand_aligned_development_not_validation",
    )
    result = run_external_closed_loop_rollout(
        scenario=scenario,
        seed=int(seed),
        policy=candidate.key,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
        **dict(rollout_kwargs),
    )
    residual = dict(result["metrics"]["residual_deployment"])
    expected_global = int(candidate.execution_trust_region.global_cooldown_intervals)
    if (
        residual.get("global_cooldown_contract")
        != "one_executed_residual_then_network_prior_only"
        or int(residual.get("global_cooldown_intervals", -1)) != expected_global
    ):
        raise RuntimeError("v52 rollout did not execute the frozen global cooldown")
    result.update(
        {
            "analysis_stage": "estimand_aligned_global_cooldown_development",
            "global_cooldown_protocol_sha256": _sha256(global_protocol_path),
            "parent_v51_protocol_sha256": v51_sha,
            "candidate_grid_protocol_sha256": _sha256(adaptation_protocol_path),
            "failed_v51_selection_audit_sha256": failed_sha,
            "pressure_regularized_joint_audit_sha256": joint_sha,
            "pressure_regularized_certificate_sha256": str(
                pressure["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "pressure_regularized_model_sha256": str(
                pressure["city_artifacts"][city]["model"]["sha256"]
            ),
            "global_cooldown_candidate": {
                **asdict(candidate),
                "execution_trust_region": asdict(
                    candidate.execution_trust_region
                ),
                "base_candidate": asdict(base),
            },
            "selection_eligible_evidence": True,
            "untouched_validation_evidence": False,
            "prospective_evidence": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-protocol", type=Path, required=True)
    parser.add_argument("--v51-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--failed-v51-audit", type=Path, required=True)
    parser.add_argument("--expected-failed-v51-audit-sha256", required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--expected-offline-authorization-sha256", required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--expected-conversion-manifest-sha256", required=True)
    parser.add_argument("--expected-conversion-tree-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite v52 development evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_global_cooldown_candidate(
        global_protocol_path=args.global_protocol,
        v51_protocol_path=args.v51_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        failed_v51_audit_path=args.failed_v51_audit,
        expected_failed_v51_audit_sha256=args.expected_failed_v51_audit_sha256,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
        candidate_key=args.candidate,
        scenario=args.scenario,
        seed=args.seed,
        rollout_kwargs={
            "protocol_spec_path": args.protocol_spec,
            "offline_authorization_path": args.offline_authorization,
            "expected_offline_authorization_sha256": args.expected_offline_authorization_sha256,
            "joint_freeze_audit_path": args.joint_freeze_audit,
            "expected_joint_freeze_sha256": args.expected_joint_freeze_sha256,
            "freeze_root": args.freeze_root,
            "external_manifest_path": args.external_manifest,
            "conversion_root": args.conversion_root,
            "expected_external_manifest_sha256": args.expected_external_manifest_sha256,
            "conversion_manifest_path": args.conversion_manifest,
            "expected_conversion_manifest_sha256": args.expected_conversion_manifest_sha256,
            "expected_conversion_tree_sha256": args.expected_conversion_tree_sha256,
            "expected_sumo_version": args.expected_sumo_version,
            "tripinfo_out": args.tripinfo_out,
        },
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "candidate": args.candidate,
                "mean_waiting_time": payload["metrics"][
                    "mean_tripinfo_waiting_time"
                ],
                "executed_overrides": payload["metrics"]["guard_audit"][
                    "executed_overrides"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
