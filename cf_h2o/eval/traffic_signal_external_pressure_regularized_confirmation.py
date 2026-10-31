"""Run v47 pressure-regularized external-city closed-loop evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
    ClosedLoopDiagnosticSpec,
    ClosedLoopMethodRuntimeOverride,
    FrozenAnchoredBlendModel,
    _fitted_runtime_bundle,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    POLICY_TO_SOURCE,
    _prediction_horizon_sec,
    _read_json,
    _resolve_protocol_reference,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import ANCHOR_POLICY
from cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze import (
    DEPLOYMENT_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
    V9_MODEL_PROTOCOL,
    V9_RESULT_PROTOCOL as V9_FREEZE_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import PriorRegularizationConfig
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


PROTOCOL = "tsc-v47r43-external-oof-pressure-regularized-development-v1"
V9_PROTOCOL = "tsc-v56r52-external-v9-oof-pressure-regularized-development-v1"
JOINT_AUDIT_PROTOCOL = (
    "tsc-v47r43-external-oof-pressure-regularized-joint-audit-v1"
)
JOINT_AUDIT_DECISION = "authorize_pressure_regularized_contaminated_development_only"
V9_JOINT_AUDIT_PROTOCOL = (
    "tsc-v56r52-external-v9-oof-pressure-regularized-joint-audit-v1"
)
V9_JOINT_AUDIT_DECISION = (
    "authorize_v9_pressure_regularized_frozen_development_replay"
)
FALSIFICATION_RESULT_PROTOCOL = (
    "tsc-v47r43-pressure-regularized-contaminated-falsification-rollout-v1"
)
ROBUSTNESS_RESULT_PROTOCOL = (
    "tsc-v47r43-pressure-regularized-untouched-robustness-rollout-v1"
)
CONFIRMATION_RESULT_PROTOCOL = (
    "tsc-v47r43-pressure-regularized-prospective-confirmation-rollout-v1"
)
METHOD = DEPLOYMENT_POLICY
POLICIES = (METHOD, *POLICY_TO_SOURCE)


def _scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].items()
        if str(scenario) in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError(f"v47 scenario has no unique city: {scenario}")
    return matches[0]


def _load_artifacts(
    *,
    protocol: Mapping[str, Any],
    joint: Mapping[str, Any],
    freeze_root: Path,
    city: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name == PROTOCOL:
        freeze_result_protocol = FREEZE_RESULT_PROTOCOL
        model_protocol = MODEL_PROTOCOL
    elif protocol_name == V9_PROTOCOL:
        freeze_result_protocol = V9_FREEZE_RESULT_PROTOCOL
        model_protocol = V9_MODEL_PROTOCOL
    else:
        raise ValueError("unknown pressure-regularized artifact protocol")
    city_spec = dict(protocol["city_artifacts"][city])
    joint_city = dict(joint["cities"][city])
    root = Path(freeze_root) / city
    certificate_path = root / "freeze.json"
    model_path = root / "model.pkl"
    certificate_sha = _sha256(certificate_path)
    model_sha = _sha256(model_path)
    if (
        certificate_sha != str(city_spec["certificate"]["sha256"])
        or certificate_sha != str(joint_city["certificate"]["sha256"])
        or model_sha != str(city_spec["model"]["sha256"])
        or model_sha != str(joint_city["model"]["sha256"])
    ):
        raise ValueError(f"pressure-regularized artifact identity changed: {city}")
    certificate = _read_json(certificate_path)
    model = pickle.loads(model_path.read_bytes())
    regularizer = model.get("regularizer")
    support = model.get("target_support")
    models = model.get("models")
    if (
        certificate.get("protocol") != freeze_result_protocol
        or certificate.get("city") != city
        or not bool(certificate.get("freeze_gate", {}).get("passed", False))
        or model.get("protocol") != model_protocol
        or model.get("city") != city
        or model.get("anchor_policy") != ANCHOR_POLICY
        or model.get("selected_candidate") != certificate.get("selected_candidate")
        or not isinstance(models, OfflineScreeningModels)
        or str(models.prior_spec.key) != ANCHOR_POLICY
        or not isinstance(regularizer, PriorRegularizationConfig)
        or not isinstance(support, TargetActionSupport)
        or not bool(support.label_free)
        or bool(regularizer.enabled)
        != bool(certificate.get("deployment", {}).get("regularizer", {}).get("enabled"))
    ):
        raise ValueError(f"pressure-regularized artifact contract failed: {city}")
    return certificate, model


def run_pressure_regularized_rollout(
    *,
    protocol_path: Path,
    joint_audit_path: Path,
    expected_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    analysis_stage: str,
    policy: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if analysis_stage != "falsification":
        raise ValueError(
            "v47 failed its frozen falsification gate; robustness and "
            "confirmation stages are permanently unauthorized"
        )
    protocol = _read_json(protocol_path)
    joint = _read_json(joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v47 pressure-regularized rollout protocol changed")
    parent_path = _resolve_protocol_reference(
        protocol_path, str(protocol["parent_protocol"]["path"])
    )
    joint_sha = _sha256(joint_audit_path)
    if (
        _sha256(parent_path) != str(protocol["parent_protocol"]["sha256"])
        or joint_sha != str(expected_joint_audit_sha256)
        or joint_sha
        != str(protocol["pressure_regularized_joint_audit"]["sha256"])
        or joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or joint.get("status") != "PASS"
        or joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v47 rollout lacks a valid pressure-regularized joint audit")
    if policy not in POLICIES:
        raise ValueError(f"unknown v47 rollout policy: {policy}")
    city = _scenario_city(protocol, scenario)
    certificate, model = _load_artifacts(
        protocol=protocol,
        joint=joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )

    if analysis_stage == "falsification":
        stage_spec = protocol["contaminated_falsification"]
        analysis_status = "contaminated_falsification_not_confirmation"
        result_protocol = FALSIFICATION_RESULT_PROTOCOL
    else:
        raise AssertionError("non-falsification v47 stages are rejected above")
    allowed_seeds = tuple(int(value) for value in stage_spec["closed_loop_seeds"])
    allowed_policies = (
        POLICIES
        if analysis_stage == "confirmation"
        else tuple(str(value) for value in stage_spec["policies"])
    )
    if int(seed) not in set(allowed_seeds) or policy not in set(allowed_policies):
        raise ValueError(f"{policy}/seed {seed} is not frozen for {analysis_stage}")

    regularizer: PriorRegularizationConfig = model["regularizer"]
    runtime_override: ClosedLoopMethodRuntimeOverride | None = None
    if policy == METHOD and regularizer.enabled:
        offline: OfflineScreeningModels = model["models"]
        wrapper = FrozenAnchoredBlendModel(
            anchor_model=offline.family_models[ANCHOR_FAMILY],
            correction_model=offline.family_models[CORRECTION_FAMILY],
            candidate=str(model["selected_candidate"]),
            anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
        )
        runtime_models = _fitted_runtime_bundle(
            offline_payload={"model_ensemble": [offline]},
            family=ANCHOR_FAMILY,
            model=wrapper,
            prediction_horizon_sec=_prediction_horizon_sec(
                Path(str(rollout_kwargs["protocol_spec_path"]))
            ),
            regularizer_config=regularizer,
            target_support=model["target_support"],
        )
        source_policy = METHOD_POLICY
        runtime_override = ClosedLoopMethodRuntimeOverride(
            runtime_policy=f"{ANCHOR_FAMILY}_contrast_pressure_gate",
            models=runtime_models,
            selected_candidate=str(model["selected_candidate"]),
            anchored_blend_model=wrapper,
            provenance={
                "protocol": MODEL_PROTOCOL,
                "city": city,
                "anchor_policy": ANCHOR_POLICY,
                "joint_audit_sha256": joint_sha,
                "certificate_sha256": str(
                    protocol["city_artifacts"][city]["certificate"]["sha256"]
                ),
                "model_sha256": str(
                    protocol["city_artifacts"][city]["model"]["sha256"]
                ),
                "pressure_regularizer": certificate["deployment"]["regularizer"],
                "label_free_target_support_installed": True,
            },
        )
    elif policy == METHOD:
        source_policy = ANCHOR_POLICY
    else:
        source_policy = POLICY_TO_SOURCE[policy]

    diagnostic = ClosedLoopDiagnosticSpec(
        name=policy,
        source_policy=source_policy,
        coordination_mode="direct" if runtime_override is not None else "sparse",
        result_protocol=result_protocol,
        cooldown_intervals_override=0 if runtime_override is not None else None,
        allowed_seeds=allowed_seeds,
        analysis_status=analysis_status,
    )
    result = run_external_closed_loop_rollout(
        scenario=scenario,
        seed=int(seed),
        policy=policy,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
        **dict(rollout_kwargs),
    )
    result.update(
        {
            "analysis_stage": analysis_stage,
            "pressure_regularized_protocol_sha256": _sha256(protocol_path),
            "pressure_regularized_joint_audit_sha256": joint_sha,
            "pressure_regularized_certificate_sha256": str(
                protocol["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "pressure_regularized_model_sha256": str(
                protocol["city_artifacts"][city]["model"]["sha256"]
            ),
            "pressure_regularizer_selection": certificate[
                "regularizer_selection"
            ]["selected"],
            "pressure_regularized_deployment": certificate["deployment"],
            "exact_phase_fallback": bool(policy == METHOD and not regularizer.enabled),
            "prospective_confirmation_seeds": list(
                protocol["prospective_confirmation_reservation"][
                    "closed_loop_seeds"
                ]
            ),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-stage",
        choices=("falsification", "robustness", "confirmation"),
        required=True,
    )
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
        raise FileExistsError("refusing to overwrite v47 rollout evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_pressure_regularized_rollout(
        protocol_path=args.pressure_protocol,
        joint_audit_path=args.pressure_joint_audit,
        expected_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
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
