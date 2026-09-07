"""Run one v51 sparse execution-trust-region development candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import _read_json
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (
    PHASE_POLICY,
    build_candidate_runtime,
    policy_candidates,
)
from cf_h2o.eval.traffic_signal_external_policy_horizon_robustness import (
    PROTOCOL as ROBUSTNESS_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    PROTOCOL as PRESSURE_PROTOCOL,
    _load_artifacts,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
)


PROTOCOL = "tsc-v51r47-external-execution-trust-region-development-v1"
RESULT_PROTOCOL = "tsc-v51r47-execution-trust-region-development-rollout-v1"
FAILED_V50_DECISION = "reject_v50_and_preserve_new_validation_and_prospective_seeds"


@dataclass(frozen=True)
class ExecutionTrustRegionCandidate:
    key: str
    base_candidate_key: str
    coordination_mode: str
    cooldown_intervals: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def trust_region_candidates(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> tuple[ExecutionTrustRegionCandidate, ...]:
    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    base = {
        candidate.key: candidate
        for candidate in policy_candidates(adaptation, city)
        if candidate.key != PHASE_POLICY
    }
    expected_base = tuple(str(value) for value in grid["base_candidates"])
    if any(key not in base for key in expected_base):
        raise ValueError("v51 base candidate changed")
    candidates = []
    for base_key in expected_base:
        for raw_region in grid["execution_trust_regions"]:
            region = dict(raw_region)
            region_key = str(region.pop("key"))
            candidates.append(
                ExecutionTrustRegionCandidate(
                    key=f"{base_key}__{region_key}",
                    base_candidate_key=base_key,
                    coordination_mode=str(grid["coordination_mode"]),
                    cooldown_intervals=int(base[base_key].cooldown_intervals),
                    execution_trust_region=ResidualExecutionTrustRegionConfig(
                        **region
                    ),
                )
            )
    keys = [candidate.key for candidate in candidates]
    expected = int(grid["candidate_count"])
    if len(candidates) != expected or len(set(keys)) != expected:
        raise ValueError("v51 execution trust-region candidate grid changed")
    return tuple(candidates)


def all_rollout_identities(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> tuple[tuple[str, str, int, str], ...]:
    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    scenario = str(grid["scenario"])
    identities = tuple(
        (city, scenario, int(seed), candidate.key)
        for seed in grid["seeds"]
        for candidate in trust_region_candidates(protocol, adaptation)
    )
    expected = int(grid["matrix_size"])
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError("v51 execution trust-region matrix changed")
    return identities


def run_execution_trust_region_candidate(
    *,
    trust_protocol_path: Path,
    expanded_protocol_path: Path,
    adaptation_protocol_path: Path,
    robustness_protocol_path: Path,
    failed_v50_audit_path: Path,
    expected_failed_v50_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    candidate_key: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(trust_protocol_path)
    expanded = _read_json(expanded_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    robustness = _read_json(robustness_protocol_path)
    failed = _read_json(failed_v50_audit_path)
    pressure = _read_json(pressure_protocol_path)
    joint = _read_json(pressure_joint_audit_path)
    failed_sha = _sha256(failed_v50_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v51 execution trust-region protocol changed")
    if (
        failed_sha != str(expected_failed_v50_audit_sha256)
        or failed_sha != str(protocol["failed_v50_selection_audit"]["sha256"])
        or failed.get("status") != "PASS"
        or failed.get("decision") != FAILED_V50_DECISION
        or _sha256(expanded_protocol_path)
        != str(protocol["parent_expanded_protocol"]["sha256"])
        or failed.get("expanded_protocol_sha256")
        != _sha256(expanded_protocol_path)
        or _sha256(adaptation_protocol_path)
        != str(protocol["candidate_grid_protocol"]["sha256"])
        or _sha256(robustness_protocol_path)
        != str(expanded["parent_robustness_protocol"]["sha256"])
        or robustness.get("protocol") != ROBUSTNESS_PROTOCOL
        or _sha256(pressure_protocol_path)
        != str(adaptation["parent_pressure_protocol"]["sha256"])
        or pressure.get("protocol") != PRESSURE_PROTOCOL
        or joint_sha != str(expected_pressure_joint_audit_sha256)
        or joint_sha != str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        or joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or joint.get("status") != "PASS"
        or joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v51 lacks frozen failed-development/model evidence")
    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    if scenario != str(grid["scenario"]):
        raise ValueError(f"scenario {scenario} is outside v51 development")
    allowed_seeds = tuple(int(value) for value in grid["seeds"])
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is outside v51 development")
    candidates = {
        candidate.key: candidate
        for candidate in trust_region_candidates(protocol, adaptation)
    }
    if candidate_key not in candidates:
        raise ValueError(f"unknown v51 candidate: {candidate_key}")
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
            "execution_trust_region_protocol": PROTOCOL,
            "failed_v50_audit_sha256": failed_sha,
            "pressure_joint_audit_sha256": joint_sha,
            "city": city,
        },
    )
    if cooldown != candidate.cooldown_intervals:
        raise ValueError("v51 base cooldown changed")
    diagnostic = ClosedLoopDiagnosticSpec(
        name=candidate.key,
        source_policy=source_policy,
        coordination_mode=candidate.coordination_mode,
        result_protocol=RESULT_PROTOCOL,
        cooldown_intervals_override=candidate.cooldown_intervals,
        execution_trust_region=candidate.execution_trust_region,
        allowed_seeds=allowed_seeds,
        analysis_status="post_v50_execution_trust_region_development_not_validation",
    )
    result = run_external_closed_loop_rollout(
        scenario=scenario,
        seed=int(seed),
        policy=candidate.key,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
        **dict(rollout_kwargs),
    )
    result.update(
        {
            "analysis_stage": "execution_trust_region_development",
            "execution_trust_region_protocol_sha256": _sha256(
                trust_protocol_path
            ),
            "parent_expanded_protocol_sha256": _sha256(expanded_protocol_path),
            "candidate_grid_protocol_sha256": _sha256(adaptation_protocol_path),
            "failed_v50_selection_audit_sha256": failed_sha,
            "pressure_regularized_joint_audit_sha256": joint_sha,
            "pressure_regularized_certificate_sha256": str(
                pressure["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "pressure_regularized_model_sha256": str(
                pressure["city_artifacts"][city]["model"]["sha256"]
            ),
            "execution_trust_region_candidate": {
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
    parser.add_argument("--trust-protocol", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--failed-v50-audit", type=Path, required=True)
    parser.add_argument("--expected-failed-v50-audit-sha256", required=True)
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
        raise FileExistsError("refusing to overwrite v51 development evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_execution_trust_region_candidate(
        trust_protocol_path=args.trust_protocol,
        expanded_protocol_path=args.expanded_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        robustness_protocol_path=args.robustness_protocol,
        failed_v50_audit_path=args.failed_v50_audit,
        expected_failed_v50_audit_sha256=args.expected_failed_v50_audit_sha256,
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
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
