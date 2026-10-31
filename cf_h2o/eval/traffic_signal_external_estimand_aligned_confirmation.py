"""Run estimand-aligned development or prospective external-city rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
    ClosedLoopDiagnosticSpec,
    ClosedLoopMethodRuntimeOverride,
    FrozenAnchoredBlendModel,
    _fitted_runtime_bundle,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (
    ANCHOR_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
    V9_MODEL_PROTOCOL,
    V9_RESULT_PROTOCOL as V9_FREEZE_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


ALIGNED_PROTOCOL = "tsc-v46r42-external-oof-validated-support-development-v1"
ALIGNED_AUDIT_PROTOCOL = "tsc-v46r42-external-oof-validated-support-joint-audit-v1"
ALIGNED_AUDIT_DECISION = (
    "authorize_oof_validated_support_contaminated_seed_development_only"
)
V9_ALIGNED_PROTOCOL = "tsc-v55r51-external-v9-oof-validated-support-development-v1"
V9_ALIGNED_AUDIT_PROTOCOL = (
    "tsc-v55r51-external-v9-oof-validated-support-joint-audit-v1"
)
V9_ALIGNED_AUDIT_DECISION = (
    "authorize_v9_estimand_aligned_frozen_development_replay"
)
DEVELOPMENT_RESULT_PROTOCOL = (
    "tsc-v46r42-oof-validated-support-contaminated-development-rollout-v1"
)
CONFIRMATION_RESULT_PROTOCOL = (
    "tsc-v46r42-oof-validated-support-prospective-confirmation-rollout-v1"
)
V9_DEVELOPMENT_RESULT_PROTOCOL = (
    "tsc-v55r51-v9-oof-validated-support-development-replay-rollout-v1"
)
V9_CONFIRMATION_RESULT_PROTOCOL = (
    "tsc-v55r51-v9-oof-validated-support-prospective-confirmation-rollout-v1"
)
EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL = (
    "tsc-v46r42-oof-validated-support-execution-factorial-diagnostic-v1"
)
METHOD = "cfcmt_oof_validated_mpc"
POLICY_TO_SOURCE = {
    "phase_pressure": "phase_pressure",
    "max_pressure": "max_pressure",
    "selected_source_prior": "selected_source_prior",
    "fixed_time": "fixed_time",
    "rigid_anchor_mpc": "rigid_anchor_mpc",
    "simulator_only_mpc": "simulator_only_mpc",
    "h2oplus_style_dense_residual_mpc": "h2oplus_style_dense_residual_mpc",
}
POLICIES = (METHOD, *POLICY_TO_SOURCE)
EXECUTION_DIAGNOSTICS = {
    "direct_raw": {
        "guarded": False,
        "coordination_mode": "direct",
        "cooldown_intervals_override": 0,
    },
    "direct_guarded": {
        "guarded": True,
        "coordination_mode": "direct",
        "cooldown_intervals_override": 0,
    },
    "spatial_raw": {
        "guarded": False,
        "coordination_mode": "spatial_only",
        "cooldown_intervals_override": 0,
    },
    "spatial_guarded": {
        "guarded": True,
        "coordination_mode": "spatial_only",
        "cooldown_intervals_override": 0,
    },
    "sparse_raw": {
        "guarded": False,
        "coordination_mode": "sparse",
        "cooldown_intervals_override": None,
    },
}
EXECUTION_VARIANTS = tuple(EXECUTION_DIAGNOSTICS)


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
        raise ValueError(f"estimand-aligned scenario has no unique city: {scenario}")
    return matches[0]


def _resolve_protocol_reference(protocol_path: Path, reference: str) -> Path:
    path = Path(str(reference))
    if path.is_absolute():
        return path
    return Path(protocol_path).resolve().parents[2] / path


def _prediction_horizon_sec(parent_protocol_path: Path) -> int:
    parent = _read_json(parent_protocol_path)
    upstream = _resolve_protocol_reference(
        parent_protocol_path, str(parent["parent_protocol"]["path"])
    )
    if _sha256(upstream) != str(parent["parent_protocol"]["sha256"]):
        raise ValueError("estimand-aligned upstream protocol changed")
    upstream_payload = _read_json(upstream)
    cache = dict(upstream_payload["counterfactual_cache"])
    return int(cache["control_interval_sec"]) * int(
        cache["counterfactual_horizon_intervals"]
    )


def _guard(payload: Mapping[str, Any]) -> ContrastGuardConfig:
    row = dict(payload["deployment"]["guard"])
    return ContrastGuardConfig(
        enabled=bool(row["enabled"]),
        risk_multiplier=float(row["risk_multiplier"]),
        min_context_trust=float(row["min_context_trust"]),
        margin=float(row["margin"]),
        max_relative_rule_gap=float(row["max_relative_rule_gap"]),
    )


def _load_aligned_artifacts(
    *,
    protocol: Mapping[str, Any],
    joint: Mapping[str, Any],
    aligned_freeze_root: Path,
    city: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name == ALIGNED_PROTOCOL:
        freeze_result_protocol = FREEZE_RESULT_PROTOCOL
        model_protocol = MODEL_PROTOCOL
    elif protocol_name == V9_ALIGNED_PROTOCOL:
        freeze_result_protocol = V9_FREEZE_RESULT_PROTOCOL
        model_protocol = V9_MODEL_PROTOCOL
    else:
        raise ValueError("unknown estimand-aligned artifact protocol")
    city_spec = dict(protocol["city_artifacts"][city])
    joint_city = dict(joint["cities"][city])
    city_root = Path(aligned_freeze_root) / city
    certificate_path = city_root / "freeze.json"
    model_path = city_root / "model.pkl"
    certificate_sha = _sha256(certificate_path)
    model_sha = _sha256(model_path)
    if (
        certificate_sha != str(city_spec["certificate"]["sha256"])
        or certificate_sha != str(joint_city["certificate"]["sha256"])
        or model_sha != str(city_spec["model"]["sha256"])
        or model_sha != str(joint_city["model"]["sha256"])
    ):
        raise ValueError(f"estimand-aligned artifact identity changed for {city}")

    certificate = _read_json(certificate_path)
    model_payload = pickle.loads(model_path.read_bytes())
    models = model_payload.get("models")
    support = model_payload.get("target_support")
    guard = model_payload.get("guard")
    if (
        certificate.get("protocol") != freeze_result_protocol
        or certificate.get("city") != city
        or not bool(certificate.get("freeze_gate", {}).get("passed", False))
        or model_payload.get("protocol") != model_protocol
        or model_payload.get("city") != city
        or model_payload.get("anchor_policy") != ANCHOR_POLICY
        or model_payload.get("selected_candidate")
        != certificate.get("selected_candidate")
        or not isinstance(models, OfflineScreeningModels)
        or set(models.family_models) != set(BASE_FAMILIES)
        or str(getattr(models.prior_spec, "key", "")) != ANCHOR_POLICY
        or not isinstance(support, TargetActionSupport)
        or not isinstance(guard, ContrastGuardConfig)
        or bool(guard.enabled)
        != bool(certificate.get("deployment", {}).get("guard", {}).get("enabled"))
    ):
        raise ValueError(f"estimand-aligned artifact contract failed for {city}")
    return certificate, model_payload


def run_estimand_aligned_rollout(
    *,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
    expected_aligned_joint_audit_sha256: str,
    aligned_freeze_root: Path,
    analysis_stage: str,
    policy: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
    execution_variant: str | None = None,
) -> dict[str, Any]:
    protocol = _read_json(aligned_protocol_path)
    joint = _read_json(aligned_joint_audit_path)
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name not in {ALIGNED_PROTOCOL, V9_ALIGNED_PROTOCOL}:
        raise ValueError("estimand-aligned rollout protocol changed")
    expected_audit_protocol = (
        ALIGNED_AUDIT_PROTOCOL
        if protocol_name == ALIGNED_PROTOCOL
        else V9_ALIGNED_AUDIT_PROTOCOL
    )
    expected_audit_decision = (
        ALIGNED_AUDIT_DECISION
        if protocol_name == ALIGNED_PROTOCOL
        else V9_ALIGNED_AUDIT_DECISION
    )
    model_protocol = (
        MODEL_PROTOCOL if protocol_name == ALIGNED_PROTOCOL else V9_MODEL_PROTOCOL
    )
    parent_protocol_path = Path(str(rollout_kwargs["protocol_spec_path"]))
    joint_sha = _sha256(aligned_joint_audit_path)
    if (
        _sha256(parent_protocol_path) != str(protocol["parent_protocol"]["sha256"])
        or joint_sha != str(expected_aligned_joint_audit_sha256)
        or joint_sha
        != str(protocol["estimand_aligned_joint_audit"]["sha256"])
        or joint.get("protocol") != expected_audit_protocol
        or joint.get("status") != "PASS"
        or joint.get("decision") != expected_audit_decision
    ):
        raise ValueError("estimand-aligned rollout lacks a valid joint audit")
    if policy not in POLICIES:
        raise ValueError(f"unknown estimand-aligned rollout policy: {policy}")
    if execution_variant is not None and (
        execution_variant not in EXECUTION_DIAGNOSTICS
        or analysis_stage != "development"
        or policy != METHOD
    ):
        raise ValueError(
            "execution diagnostics are frozen-method development analyses only"
        )
    city = _scenario_city(protocol, scenario)
    certificate, model_payload = _load_aligned_artifacts(
        protocol=protocol,
        joint=joint,
        aligned_freeze_root=aligned_freeze_root,
        city=city,
    )

    if analysis_stage == "development":
        allowed_seeds = tuple(int(value) for value in protocol["development"]["closed_loop_seeds"])
        allowed_policies = tuple(str(value) for value in protocol["development"]["policies"])
        analysis_status = (
            "contaminated_estimand_alignment_falsification_not_confirmation"
            if protocol_name == ALIGNED_PROTOCOL
            else "repaired_network_frozen_development_replay_not_confirmation"
        )
        result_protocol = (
            DEVELOPMENT_RESULT_PROTOCOL
            if protocol_name == ALIGNED_PROTOCOL
            else V9_DEVELOPMENT_RESULT_PROTOCOL
        )
    elif analysis_stage == "confirmation":
        allowed_seeds = tuple(
            int(value)
            for value in protocol["prospective_confirmation_reservation"][
                "closed_loop_seeds"
            ]
        )
        allowed_policies = POLICIES
        analysis_status = "prospective_confirmation_after_development_and_analysis_freeze"
        result_protocol = (
            CONFIRMATION_RESULT_PROTOCOL
            if protocol_name == ALIGNED_PROTOCOL
            else V9_CONFIRMATION_RESULT_PROTOCOL
        )
    else:
        raise ValueError("analysis stage must be development or confirmation")
    if int(seed) not in set(allowed_seeds) or policy not in set(allowed_policies):
        raise ValueError(f"{policy}/seed {seed} is not frozen for {analysis_stage}")

    guard = _guard(certificate)
    execution_spec = (
        dict(EXECUTION_DIAGNOSTICS[execution_variant])
        if execution_variant is not None
        else None
    )
    if execution_spec is not None and not guard.enabled:
        raise ValueError(
            "execution diagnostics require a city with an enabled frozen guard"
        )
    runtime_override: ClosedLoopMethodRuntimeOverride | None = None
    if policy == METHOD and guard.enabled:
        runtime_guard = (
            guard
            if execution_spec is None or bool(execution_spec["guarded"])
            else None
        )
        offline: OfflineScreeningModels = model_payload["models"]
        wrapper = FrozenAnchoredBlendModel(
            anchor_model=offline.family_models[ANCHOR_FAMILY],
            correction_model=offline.family_models[CORRECTION_FAMILY],
            candidate=str(model_payload["selected_candidate"]),
            anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
        )
        runtime_models = _fitted_runtime_bundle(
            offline_payload={"model_ensemble": [offline]},
            family=ANCHOR_FAMILY,
            model=wrapper,
            prediction_horizon_sec=_prediction_horizon_sec(parent_protocol_path),
            guard_config=runtime_guard,
            target_support=(
                model_payload["target_support"] if runtime_guard is not None else None
            ),
        )
        source_policy = METHOD_POLICY
        runtime_override = ClosedLoopMethodRuntimeOverride(
            runtime_policy=(
                f"{ANCHOR_FAMILY}_contrast_guard"
                if runtime_guard is not None
                else f"{ANCHOR_FAMILY}_contrast_raw"
            ),
            models=runtime_models,
            selected_candidate=str(model_payload["selected_candidate"]),
            anchored_blend_model=wrapper,
            provenance={
                "protocol": model_protocol,
                "city": city,
                "anchor_policy": ANCHOR_POLICY,
                "joint_audit_sha256": joint_sha,
                "certificate_sha256": str(
                    protocol["city_artifacts"][city]["certificate"]["sha256"]
                ),
                "model_sha256": str(protocol["city_artifacts"][city]["model"]["sha256"]),
                "target_support_installed": runtime_guard is not None,
                "execution_variant": execution_variant,
            },
        )
        guard_config = runtime_guard
    elif policy == METHOD:
        source_policy = ANCHOR_POLICY
        guard_config = None
    else:
        source_policy = POLICY_TO_SOURCE[policy]
        guard_config = None

    deployed_policy = (
        f"{METHOD}_{execution_variant}"
        if execution_variant is not None
        else policy
    )
    diagnostic = ClosedLoopDiagnosticSpec(
        name=deployed_policy,
        source_policy=source_policy,
        coordination_mode=(
            str(execution_spec["coordination_mode"])
            if execution_spec is not None
            else str(protocol["deployment"]["coordination_mode"])
        ),
        result_protocol=(
            EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL
            if execution_spec is not None
            else result_protocol
        ),
        guard_config=guard_config,
        cooldown_intervals_override=(
            execution_spec["cooldown_intervals_override"]
            if execution_spec is not None
            else None
        ),
        allowed_seeds=allowed_seeds,
        analysis_status=(
            "post_failure_contaminated_execution_factorial_not_confirmation"
            if execution_spec is not None
            else analysis_status
        ),
    )
    result = run_external_closed_loop_rollout(
        scenario=scenario,
        seed=int(seed),
        policy=deployed_policy,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
        **dict(rollout_kwargs),
    )
    result.update(
        {
            "analysis_stage": analysis_stage,
            "estimand_aligned_protocol_sha256": _sha256(aligned_protocol_path),
            "estimand_aligned_joint_audit_sha256": joint_sha,
            "estimand_aligned_certificate_sha256": str(
                protocol["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "estimand_aligned_model_sha256": str(
                protocol["city_artifacts"][city]["model"]["sha256"]
            ),
            "estimand_alignment": certificate["estimand_alignment"],
            "estimand_aligned_selected_candidate": certificate["selected_candidate"],
            "estimand_aligned_guard_decision": certificate["guard_selection"]["decision"],
            "estimand_aligned_guard": certificate["deployment"]["guard"],
            "exact_phase_fallback": bool(
                policy == METHOD and not guard.enabled and execution_spec is None
            ),
            "execution_diagnostic": (
                {
                    "variant": execution_variant,
                    "guarded": bool(execution_spec["guarded"]),
                    "coordination_mode": str(execution_spec["coordination_mode"]),
                    "cooldown_intervals_override": execution_spec[
                        "cooldown_intervals_override"
                    ],
                    "same_frozen_model_as_v46": True,
                    "prospective_evidence": False,
                }
                if execution_spec is not None
                else None
            ),
            "old_v43_offline_authorization_role": protocol["deployment"][
                "old_v43_offline_authorization_role"
            ],
            "prospective_confirmation_seeds": list(
                protocol["prospective_confirmation_reservation"]["closed_loop_seeds"]
            ),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-aligned-joint-audit-sha256", required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
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
    parser.add_argument("--execution-variant", choices=EXECUTION_VARIANTS)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite estimand-aligned rollout evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_estimand_aligned_rollout(
        aligned_protocol_path=args.aligned_protocol,
        aligned_joint_audit_path=args.aligned_joint_audit,
        expected_aligned_joint_audit_sha256=args.expected_aligned_joint_audit_sha256,
        aligned_freeze_root=args.aligned_freeze_root,
        analysis_stage=args.analysis_stage,
        policy=args.policy,
        scenario=args.scenario,
        seed=args.seed,
        execution_variant=args.execution_variant,
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
                "policy": payload["policy"],
                "execution_variant": args.execution_variant,
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
