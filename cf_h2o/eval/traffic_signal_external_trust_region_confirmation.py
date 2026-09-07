"""Run development or prospective closed-loop trust-region rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    RESULT_PROTOCOL as TRUST_FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig


TRUST_PROTOCOL = "tsc-v44r40-external-conformal-trust-region-development-v1"
DEVELOPMENT_RESULT_PROTOCOL = "tsc-v44r40-trust-region-contaminated-development-rollout-v1"
CONFIRMATION_RESULT_PROTOCOL = "tsc-v44r40-trust-region-prospective-confirmation-rollout-v1"
METHOD = "cfcmt_trust_region_mpc"
POLICY_TO_SOURCE = {
    "selected_source_prior": "selected_source_prior",
    "fixed_time": "fixed_time",
    "phase_pressure": "phase_pressure",
    "max_pressure": "max_pressure",
    "rigid_anchor_mpc": "rigid_anchor_mpc",
    "h2oplus_style_dense_residual_mpc": "h2oplus_style_dense_residual_mpc",
}
POLICIES = (METHOD, *POLICY_TO_SOURCE)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].items()
        if str(scenario) in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError(f"trust-region scenario has no unique city: {scenario}")
    return matches[0]


def _guard(payload: Mapping[str, Any]) -> ContrastGuardConfig:
    row = dict(payload["deployment"]["guard"])
    return ContrastGuardConfig(
        enabled=bool(row["enabled"]),
        risk_multiplier=float(row["risk_multiplier"]),
        min_context_trust=float(row["min_context_trust"]),
        margin=float(row["margin"]),
        max_relative_rule_gap=float(row["max_relative_rule_gap"]),
    )


def run_trust_region_rollout(
    *,
    trust_protocol_path: Path,
    trust_joint_audit_path: Path,
    expected_trust_joint_audit_sha256: str,
    certificate_root: Path,
    analysis_stage: str,
    policy: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(trust_protocol_path)
    joint = _read_json(trust_joint_audit_path)
    if protocol.get("protocol") != TRUST_PROTOCOL:
        raise ValueError("trust-region rollout protocol changed")
    protocol_spec_path = Path(str(rollout_kwargs["protocol_spec_path"]))
    if (
        _sha256(protocol_spec_path) != str(protocol["parent_protocol"]["sha256"])
        or
        _sha256(trust_joint_audit_path) != str(expected_trust_joint_audit_sha256)
        or str(protocol["trust_region_joint_audit"]["sha256"])
        != str(expected_trust_joint_audit_sha256)
        or joint.get("status") != "PASS"
    ):
        raise ValueError("trust-region rollout lacks a valid joint guard audit")
    if policy not in POLICIES:
        raise ValueError(f"unknown trust-region rollout policy: {policy}")
    city = _scenario_city(protocol, scenario)
    certificate_path = Path(certificate_root) / city / "trust_region_freeze.json"
    certificate = _read_json(certificate_path)
    expected_certificate_sha256 = str(
        protocol["city_certificates"][city]["sha256"]
    )
    joint_city = joint["cities"][city]
    if (
        _sha256(certificate_path) != expected_certificate_sha256
        or joint_city["certificate_sha256"] != expected_certificate_sha256
        or certificate.get("protocol") != TRUST_FREEZE_PROTOCOL
        or certificate.get("city") != city
        or not bool(certificate.get("freeze_gate", {}).get("passed", False))
    ):
        raise ValueError(f"trust-region certificate changed for {city}")

    if analysis_stage == "development":
        allowed_seeds = tuple(int(value) for value in protocol["development"]["closed_loop_seeds"])
        analysis_status = "contaminated_redesign_development_not_confirmation"
        result_protocol = DEVELOPMENT_RESULT_PROTOCOL
    elif analysis_stage == "confirmation":
        allowed_seeds = tuple(
            int(value)
            for value in protocol["prospective_confirmation_reservation"][
                "closed_loop_seeds"
            ]
        )
        analysis_status = "prospective_confirmation_after_method_and_seed_freeze"
        result_protocol = CONFIRMATION_RESULT_PROTOCOL
    else:
        raise ValueError("analysis stage must be development or confirmation")
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is not frozen for {analysis_stage}")

    guard = _guard(certificate)
    if policy == METHOD:
        source_policy = METHOD_POLICY if guard.enabled else "selected_source_prior"
        guard_config = guard if guard.enabled else None
    else:
        source_policy = POLICY_TO_SOURCE[policy]
        guard_config = None
    diagnostic = ClosedLoopDiagnosticSpec(
        name=policy,
        source_policy=source_policy,
        coordination_mode="sparse",
        result_protocol=result_protocol,
        guard_config=guard_config,
        allowed_seeds=allowed_seeds,
        analysis_status=analysis_status,
    )
    result = run_external_closed_loop_rollout(
        scenario=scenario,
        seed=int(seed),
        policy=policy,
        diagnostic_spec=diagnostic,
        **dict(rollout_kwargs),
    )
    result.update(
        {
            "analysis_stage": analysis_stage,
            "trust_protocol_sha256": _sha256(trust_protocol_path),
            "trust_joint_audit_sha256": _sha256(trust_joint_audit_path),
            "trust_region_certificate_sha256": _sha256(certificate_path),
            "trust_region_decision": certificate["selection"]["decision"],
            "trust_region_guard": certificate["deployment"]["guard"],
            "prospective_confirmation_seeds": list(
                protocol["prospective_confirmation_reservation"]["closed_loop_seeds"]
            ),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-protocol", type=Path, required=True)
    parser.add_argument("--trust-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-trust-joint-audit-sha256", required=True)
    parser.add_argument("--certificate-root", type=Path, required=True)
    parser.add_argument("--analysis-stage", choices=("development", "confirmation"), required=True)
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite trust-region rollout evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_trust_region_rollout(
        trust_protocol_path=args.trust_protocol,
        trust_joint_audit_path=args.trust_joint_audit,
        expected_trust_joint_audit_sha256=args.expected_trust_joint_audit_sha256,
        certificate_root=args.certificate_root,
        analysis_stage=args.analysis_stage,
        policy=args.policy,
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
                "analysis_stage": args.analysis_stage,
                "scenario": args.scenario,
                "seed": args.seed,
                "policy": args.policy,
                "mean_waiting_time": payload["metrics"]["mean_tripinfo_waiting_time"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
