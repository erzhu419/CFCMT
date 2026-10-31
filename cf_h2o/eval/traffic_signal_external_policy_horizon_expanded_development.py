"""Run one v50 expanded full-horizon development candidate."""

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


PROTOCOL = "tsc-v50r46-external-policy-horizon-expanded-development-v1"
FAILED_V49_DECISION = "reject_v49_and_preserve_prospective_seeds"
RESULT_PROTOCOL = "tsc-v50r46-policy-horizon-expanded-development-rollout-v1"


def scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["city_scenarios"].items()
        if scenario in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError(f"v50 scenario has no unique city: {scenario}")
    return matches[0]


def active_candidate_keys(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any], city: str
) -> tuple[str, ...]:
    keys = tuple(
        candidate.key
        for candidate in policy_candidates(adaptation, city)
        if candidate.key != PHASE_POLICY
    )
    if len(keys) != int(protocol["candidate_grid_protocol"]["active_candidate_count_per_city"]):
        raise ValueError("v50 active candidate grid changed")
    return keys


def all_rollout_identities(
    protocol: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> tuple[tuple[str, str, int, str], ...]:
    seeds = tuple(
        int(value) for value in protocol["expanded_development_grid"]["additional_seeds"]
    )
    identities = tuple(
        (str(city), str(scenario), seed, candidate)
        for city, scenarios in protocol["city_scenarios"].items()
        for scenario in scenarios
        for seed in seeds
        for candidate in active_candidate_keys(protocol, adaptation, str(city))
    )
    expected = int(protocol["expanded_development_grid"]["matrix_size"])
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError("v50 expanded development matrix changed")
    return identities


def run_expanded_development_candidate(
    *,
    expanded_protocol_path: Path,
    adaptation_protocol_path: Path,
    robustness_protocol_path: Path,
    failed_v49_audit_path: Path,
    expected_failed_v49_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    candidate_key: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(expanded_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    robustness = _read_json(robustness_protocol_path)
    failed = _read_json(failed_v49_audit_path)
    pressure_protocol = _read_json(pressure_protocol_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    failed_sha = _sha256(failed_v49_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v50 expanded development protocol changed")
    if (
        _sha256(adaptation_protocol_path)
        != str(protocol["candidate_grid_protocol"]["sha256"])
        or _sha256(robustness_protocol_path)
        != str(protocol["parent_robustness_protocol"]["sha256"])
        or robustness.get("protocol") != ROBUSTNESS_PROTOCOL
        or _sha256(adaptation_protocol_path)
        != str(robustness["parent_adaptation_protocol"]["sha256"])
        or _sha256(pressure_protocol_path)
        != str(adaptation["parent_pressure_protocol"]["sha256"])
        or pressure_protocol.get("protocol") != PRESSURE_PROTOCOL
        or failed_sha != str(expected_failed_v49_audit_sha256)
        or failed_sha != str(protocol["failed_v49_robustness_audit"]["sha256"])
        or failed.get("status") != "PASS"
        or failed.get("decision") != FAILED_V49_DECISION
        or joint_sha != str(expected_pressure_joint_audit_sha256)
        or joint_sha != str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        or pressure_joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v50 expanded development lacks frozen failure/model evidence")
    city = scenario_city(protocol, scenario)
    candidates = {
        candidate.key: candidate
        for candidate in policy_candidates(adaptation, city)
        if candidate.key != PHASE_POLICY
    }
    if candidate_key not in candidates:
        raise ValueError(f"unknown v50 candidate: {candidate_key}")
    allowed_seeds = tuple(
        int(value) for value in protocol["expanded_development_grid"]["additional_seeds"]
    )
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is not a v50 expanded-development seed")
    candidate = candidates[candidate_key]
    certificate, model = _load_artifacts(
        protocol=pressure_protocol,
        joint=pressure_joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )
    source_policy, runtime_override, cooldown = build_candidate_runtime(
        candidate=candidate,
        model=model,
        rollout_kwargs=rollout_kwargs,
        provenance={
            "expanded_protocol": PROTOCOL,
            "city": city,
            "failed_v49_audit_sha256": failed_sha,
            "pressure_joint_audit_sha256": joint_sha,
        },
    )
    diagnostic = ClosedLoopDiagnosticSpec(
        name=candidate.key,
        source_policy=source_policy,
        coordination_mode="direct",
        result_protocol=RESULT_PROTOCOL,
        cooldown_intervals_override=cooldown,
        allowed_seeds=allowed_seeds,
        analysis_status="post_v49_failure_expanded_development_not_confirmation",
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
            "analysis_stage": "expanded_policy_horizon_development",
            "expanded_development_protocol_sha256": _sha256(expanded_protocol_path),
            "candidate_grid_protocol_sha256": _sha256(adaptation_protocol_path),
            "failed_robustness_protocol_sha256": _sha256(
                robustness_protocol_path
            ),
            "failed_v49_robustness_audit_sha256": failed_sha,
            "pressure_regularized_joint_audit_sha256": joint_sha,
            "pressure_regularized_certificate_sha256": str(
                pressure_protocol["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "pressure_regularized_model_sha256": str(
                pressure_protocol["city_artifacts"][city]["model"]["sha256"]
            ),
            "policy_horizon_candidate": asdict(candidate),
            "selection_eligible_evidence": True,
            "prospective_evidence": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--failed-v49-audit", type=Path, required=True)
    parser.add_argument("--expected-failed-v49-audit-sha256", required=True)
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
        raise FileExistsError("refusing to overwrite v50 expanded-development evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_expanded_development_candidate(
        expanded_protocol_path=args.expanded_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        robustness_protocol_path=args.robustness_protocol,
        failed_v49_audit_path=args.failed_v49_audit,
        expected_failed_v49_audit_sha256=args.expected_failed_v49_audit_sha256,
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
    print(json.dumps({"scenario": args.scenario, "seed": args.seed, "candidate": args.candidate, "mean_waiting_time": payload["metrics"]["mean_tripinfo_waiting_time"], "out": str(args.out)}, sort_keys=True))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
