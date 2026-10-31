"""Run one frozen v49 city-selector robustness rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (
    PHASE_POLICY,
    HorizonPolicyCandidate,
    policy_candidates,
)
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


PROTOCOL = "tsc-v49r45-external-policy-horizon-robustness-v1"
ADAPTATION_AUDIT_DECISION = (
    "authorize_frozen_v48_selector_for_untouched_robustness"
)
RESULT_PROTOCOL = "tsc-v49r45-policy-horizon-untouched-robustness-rollout-v1"
METHOD = "cfcmt_policy_horizon_selected_mpc"
POLICIES = (METHOD, PHASE_POLICY)


def scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["city_scenarios"].items()
        if scenario in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError(f"v49 scenario has no unique city: {scenario}")
    return matches[0]


def all_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    seeds = tuple(int(value) for value in protocol["untouched_robustness"]["seeds"])
    policies = tuple(str(value) for value in protocol["untouched_robustness"]["policies"])
    identities = tuple(
        (str(city), str(scenario), seed, policy)
        for city, scenarios in protocol["city_scenarios"].items()
        for scenario in scenarios
        for seed in seeds
        for policy in policies
    )
    expected = int(protocol["untouched_robustness"]["matrix_size"])
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError("v49 robustness matrix changed")
    return identities


def _selected_candidate(
    *,
    protocol: Mapping[str, Any],
    adaptation_protocol: Mapping[str, Any],
    selection: Mapping[str, Any],
    city: str,
) -> HorizonPolicyCandidate:
    frozen = dict(protocol["frozen_city_deployment"][city])
    selected = dict(selection["city_selection"][city]["selected"])
    fields = (
        "key",
        "family",
        "blend_weight",
        "risk_multiplier",
        "min_context_trust",
        "cooldown_intervals",
    )
    if any(frozen.get(field) != selected.get(field) for field in fields):
        raise ValueError(f"v49 selected deployment changed for {city}")
    candidates = {
        candidate.key: candidate
        for candidate in policy_candidates(adaptation_protocol, city)
    }
    key = str(frozen["key"])
    if key not in candidates:
        raise ValueError(f"v49 selected candidate is not in the frozen grid: {city}")
    return candidates[key]


def run_policy_horizon_robustness(
    *,
    robustness_protocol_path: Path,
    adaptation_protocol_path: Path,
    adaptation_selection_audit_path: Path,
    expected_adaptation_selection_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    policy: str,
    scenario: str,
    seed: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(robustness_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    selection = _read_json(adaptation_selection_audit_path)
    pressure_protocol = _read_json(pressure_protocol_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    selection_sha = _sha256(adaptation_selection_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v49 robustness protocol changed")
    if (
        _sha256(adaptation_protocol_path)
        != str(protocol["parent_adaptation_protocol"]["sha256"])
        or selection_sha != str(expected_adaptation_selection_sha256)
        or selection_sha != str(protocol["adaptation_selection_audit"]["sha256"])
        or selection.get("status") != "PASS"
        or selection.get("decision") != ADAPTATION_AUDIT_DECISION
        or joint_sha != str(expected_pressure_joint_audit_sha256)
        or joint_sha
        != str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        or pressure_joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != JOINT_AUDIT_DECISION
    ):
        raise ValueError("v49 robustness lacks frozen selector/model evidence")
    if policy not in POLICIES:
        raise ValueError(f"unknown v49 robustness policy: {policy}")
    allowed_seeds = tuple(
        int(value) for value in protocol["untouched_robustness"]["seeds"]
    )
    if int(seed) not in set(allowed_seeds):
        raise ValueError(f"seed {seed} is not a v49 robustness seed")
    city = scenario_city(protocol, scenario)
    candidate = _selected_candidate(
        protocol=protocol,
        adaptation_protocol=adaptation,
        selection=selection,
        city=city,
    )
    certificate, model = _load_artifacts(
        protocol=pressure_protocol,
        joint=pressure_joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )

    runtime_override: ClosedLoopMethodRuntimeOverride | None = None
    if policy == PHASE_POLICY or candidate.family == PHASE_POLICY:
        source_policy = PHASE_POLICY
        cooldown = 0
    else:
        offline: OfflineScreeningModels = model["models"]
        wrapper = FrozenAnchoredBlendModel(
            anchor_model=offline.family_models[ANCHOR_FAMILY],
            correction_model=offline.family_models[CORRECTION_FAMILY],
            candidate=str(model["selected_candidate"]),
            anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
        )
        regularizer = PriorRegularizationConfig(
            enabled=True,
            blend_weight=float(candidate.blend_weight),
            risk_multiplier=float(candidate.risk_multiplier),
            min_context_trust=float(candidate.min_context_trust),
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
        cooldown = int(candidate.cooldown_intervals)
        runtime_override = ClosedLoopMethodRuntimeOverride(
            runtime_policy=f"{ANCHOR_FAMILY}_contrast_pressure_gate",
            models=runtime_models,
            selected_candidate=str(model["selected_candidate"]),
            anchored_blend_model=wrapper,
            provenance={
                "protocol": MODEL_PROTOCOL,
                "robustness_protocol": PROTOCOL,
                "adaptation_selection_sha256": selection_sha,
                "candidate": asdict(candidate),
                "city": city,
                "anchor_policy": ANCHOR_POLICY,
            },
        )

    diagnostic = ClosedLoopDiagnosticSpec(
        name=policy,
        source_policy=source_policy,
        coordination_mode="direct",
        result_protocol=RESULT_PROTOCOL,
        cooldown_intervals_override=cooldown,
        allowed_seeds=allowed_seeds,
        analysis_status="untouched_robustness_development_not_confirmation",
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
            "analysis_stage": "untouched_robustness",
            "policy_horizon_robustness_protocol_sha256": _sha256(
                robustness_protocol_path
            ),
            "adaptation_protocol_sha256": _sha256(adaptation_protocol_path),
            "adaptation_selection_audit_sha256": selection_sha,
            "pressure_regularized_joint_audit_sha256": joint_sha,
            "frozen_city_candidate": asdict(candidate),
            "exact_phase_fallback": bool(
                policy == METHOD and candidate.family == PHASE_POLICY
            ),
            "prospective_evidence": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-selection-audit", type=Path, required=True)
    parser.add_argument("--expected-adaptation-selection-sha256", required=True)
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite v49 robustness evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_policy_horizon_robustness(
        robustness_protocol_path=args.robustness_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        adaptation_selection_audit_path=args.adaptation_selection_audit,
        expected_adaptation_selection_sha256=args.expected_adaptation_selection_sha256,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
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
