"""Run one v48 full-horizon target-adaptation policy candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
    _prediction_horizon_sec,
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import ANCHOR_POLICY
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    _load_artifacts,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze import (
    MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import PriorRegularizationConfig
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)


PROTOCOL = "tsc-v48r44-external-policy-horizon-adaptation-v1"
RESULT_PROTOCOL = "tsc-v48r44-policy-horizon-adaptation-rollout-v1"
FAILED_V47_DECISION = (
    "reject_v47_and_preserve_robustness_and_prospective_seeds_for_successor_method"
)
PHASE_POLICY = "phase_pressure"


@dataclass(frozen=True)
class HorizonPolicyCandidate:
    key: str
    family: str
    blend_weight: float | None
    risk_multiplier: float | None
    min_context_trust: float | None
    cooldown_intervals: int


def _float_token(value: float) -> str:
    return format(float(value), "g").replace(".", "p")


def scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["city_scenarios"].items()
        if scenario in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError(f"v48 scenario has no unique city: {scenario}")
    return matches[0]


def policy_candidates(
    protocol: Mapping[str, Any], city: str
) -> tuple[HorizonPolicyCandidate, ...]:
    adaptation = dict(protocol["policy_horizon_adaptation"])
    fixed = dict(adaptation["city_fixed_oof_parameters"][city])
    candidates = [
        HorizonPolicyCandidate(
            key=PHASE_POLICY,
            family=PHASE_POLICY,
            blend_weight=None,
            risk_multiplier=None,
            min_context_trust=None,
            cooldown_intervals=0,
        )
    ]
    cooldowns = tuple(int(value) for value in adaptation["cooldown_intervals"])
    for cooldown in cooldowns:
        candidates.append(
            HorizonPolicyCandidate(
                key=f"cfcmt_raw_cd{cooldown}",
                family="cfcmt_raw_direct",
                blend_weight=None,
                risk_multiplier=None,
                min_context_trust=None,
                cooldown_intervals=cooldown,
            )
        )
    for blend in adaptation["blend_weights"]:
        for cooldown in cooldowns:
            candidates.append(
                HorizonPolicyCandidate(
                    key=f"cfcmt_pr_b{_float_token(float(blend))}_cd{cooldown}",
                    family="cfcmt_pressure_regularized_direct",
                    blend_weight=float(blend),
                    risk_multiplier=float(fixed["risk_multiplier"]),
                    min_context_trust=float(fixed["min_context_trust"]),
                    cooldown_intervals=cooldown,
                )
            )
    keys = [candidate.key for candidate in candidates]
    expected_count = int(adaptation["candidate_count_per_city"])
    if len(candidates) != expected_count or len(set(keys)) != len(keys):
        raise ValueError("v48 policy candidate grid changed")
    return tuple(candidates)


def all_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    seeds = tuple(int(value) for value in protocol["policy_horizon_adaptation"]["seeds"])
    identities = []
    for city, scenarios in protocol["city_scenarios"].items():
        candidate_keys = [candidate.key for candidate in policy_candidates(protocol, str(city))]
        for scenario in scenarios:
            for seed in seeds:
                for key in candidate_keys:
                    identities.append((str(city), str(scenario), seed, key))
    expected = int(protocol["policy_horizon_adaptation"]["matrix_size"])
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError("v48 policy-horizon matrix changed")
    return tuple(identities)


def build_candidate_runtime(
    *,
    candidate: HorizonPolicyCandidate,
    model: Mapping[str, Any],
    rollout_kwargs: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[str, ClosedLoopMethodRuntimeOverride | None, int]:
    """Instantiate one frozen candidate without performing stage selection."""

    if candidate.family == PHASE_POLICY:
        return PHASE_POLICY, None, 0
    offline: OfflineScreeningModels = model["models"]
    wrapper = FrozenAnchoredBlendModel(
        anchor_model=offline.family_models[ANCHOR_FAMILY],
        correction_model=offline.family_models[CORRECTION_FAMILY],
        candidate=str(model["selected_candidate"]),
        anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
        correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
    )
    if candidate.family == "cfcmt_pressure_regularized_direct":
        regularizer = PriorRegularizationConfig(
            enabled=True,
            blend_weight=float(candidate.blend_weight),
            risk_multiplier=float(candidate.risk_multiplier),
            min_context_trust=float(candidate.min_context_trust),
        )
        target_support = model["target_support"]
        runtime_policy = f"{ANCHOR_FAMILY}_contrast_pressure_gate"
    elif candidate.family == "cfcmt_raw_direct":
        regularizer = None
        target_support = None
        runtime_policy = f"{ANCHOR_FAMILY}_contrast_raw"
    else:
        raise ValueError(f"unsupported policy-horizon candidate: {candidate.family}")
    runtime_models = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [offline]},
        family=ANCHOR_FAMILY,
        model=wrapper,
        prediction_horizon_sec=_prediction_horizon_sec(
            Path(str(rollout_kwargs["protocol_spec_path"]))
        ),
        regularizer_config=regularizer,
        target_support=target_support,
    )
    override = ClosedLoopMethodRuntimeOverride(
        runtime_policy=runtime_policy,
        models=runtime_models,
        selected_candidate=str(model["selected_candidate"]),
        anchored_blend_model=wrapper,
        provenance={
            "protocol": str(model.get("protocol", MODEL_PROTOCOL)),
            "candidate": asdict(candidate),
            "anchor_policy": ANCHOR_POLICY,
            **dict(provenance),
        },
    )
    return METHOD_POLICY, override, int(candidate.cooldown_intervals)


def run_policy_horizon_candidate(
    *,
    policy_protocol_path: Path,
    failed_v47_audit_path: Path,
    expected_failed_v47_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    candidate_key: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(policy_protocol_path)
    pressure_protocol = _read_json(pressure_protocol_path)
    failed = _read_json(failed_v47_audit_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    failed_sha = _sha256(failed_v47_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v48 policy-horizon protocol changed")
    if (
        _sha256(pressure_protocol_path)
        != str(protocol["parent_pressure_protocol"]["sha256"])
        or failed_sha != str(expected_failed_v47_audit_sha256)
        or failed_sha
        != str(protocol["failed_v47_falsification_audit"]["sha256"])
        or failed.get("status") != "PASS"
        or failed.get("decision") != FAILED_V47_DECISION
        or joint_sha != str(expected_pressure_joint_audit_sha256)
        or joint_sha != str(protocol["model_artifacts"]["pressure_joint_audit_sha256"])
        or pressure_joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v48 policy search lacks frozen failure/model evidence")
    city = scenario_city(protocol, scenario)
    candidates = {candidate.key: candidate for candidate in policy_candidates(protocol, city)}
    if candidate_key not in candidates:
        raise ValueError(f"unknown v48 policy candidate: {candidate_key}")
    allowed_seeds = tuple(
        int(value) for value in protocol["policy_horizon_adaptation"]["seeds"]
    )
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is not a v48 policy-adaptation seed")
    candidate = candidates[candidate_key]
    certificate, model = _load_artifacts(
        protocol=pressure_protocol,
        joint=pressure_joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )

    runtime_override: ClosedLoopMethodRuntimeOverride | None = None
    if candidate.family == PHASE_POLICY:
        source_policy = PHASE_POLICY
    else:
        offline: OfflineScreeningModels = model["models"]
        wrapper = FrozenAnchoredBlendModel(
            anchor_model=offline.family_models[ANCHOR_FAMILY],
            correction_model=offline.family_models[CORRECTION_FAMILY],
            candidate=str(model["selected_candidate"]),
            anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
        )
        if candidate.family == "cfcmt_pressure_regularized_direct":
            regularizer = PriorRegularizationConfig(
                enabled=True,
                blend_weight=float(candidate.blend_weight),
                risk_multiplier=float(candidate.risk_multiplier),
                min_context_trust=float(candidate.min_context_trust),
            )
            target_support = model["target_support"]
            runtime_policy = f"{ANCHOR_FAMILY}_contrast_pressure_gate"
        elif candidate.family == "cfcmt_raw_direct":
            regularizer = None
            target_support = None
            runtime_policy = f"{ANCHOR_FAMILY}_contrast_raw"
        else:
            raise AssertionError(candidate.family)
        runtime_models = _fitted_runtime_bundle(
            offline_payload={"model_ensemble": [offline]},
            family=ANCHOR_FAMILY,
            model=wrapper,
            prediction_horizon_sec=_prediction_horizon_sec(
                Path(str(rollout_kwargs["protocol_spec_path"]))
            ),
            regularizer_config=regularizer,
            target_support=target_support,
        )
        source_policy = METHOD_POLICY
        runtime_override = ClosedLoopMethodRuntimeOverride(
            runtime_policy=runtime_policy,
            models=runtime_models,
            selected_candidate=str(model["selected_candidate"]),
            anchored_blend_model=wrapper,
            provenance={
                "protocol": MODEL_PROTOCOL,
                "search_protocol": PROTOCOL,
                "candidate": asdict(candidate),
                "city": city,
                "anchor_policy": ANCHOR_POLICY,
                "failed_v47_audit_sha256": failed_sha,
                "pressure_joint_audit_sha256": joint_sha,
            },
        )

    diagnostic = ClosedLoopDiagnosticSpec(
        name=candidate.key,
        source_policy=source_policy,
        coordination_mode="direct",
        result_protocol=RESULT_PROTOCOL,
        cooldown_intervals_override=int(candidate.cooldown_intervals),
        allowed_seeds=allowed_seeds,
        analysis_status="target_adaptation_policy_horizon_search_not_confirmation",
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
            "analysis_stage": "policy_horizon_adaptation",
            "policy_horizon_protocol_sha256": _sha256(policy_protocol_path),
            "failed_v47_falsification_audit_sha256": failed_sha,
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
    parser.add_argument("--policy-protocol", type=Path, required=True)
    parser.add_argument("--failed-v47-audit", type=Path, required=True)
    parser.add_argument("--expected-failed-v47-audit-sha256", required=True)
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
        raise FileExistsError("refusing to overwrite v48 policy-horizon evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_policy_horizon_candidate(
        policy_protocol_path=args.policy_protocol,
        failed_v47_audit_path=args.failed_v47_audit,
        expected_failed_v47_audit_sha256=args.expected_failed_v47_audit_sha256,
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
