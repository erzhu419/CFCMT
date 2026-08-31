"""Run one frozen v64 hierarchical closed-loop development rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
    _fitted_runtime_bundle,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
    evaluate_policy_v3,
)
from cf_h2o.eval.traffic_signal_target_support_equivalence_audit import (
    AUDIT_DECISION as SUPPORT_AUDIT_DECISION,
    RESULT_PROTOCOL as SUPPORT_AUDIT_PROTOCOL,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


PROTOCOL = "tsc-v64r60-external-v9-hierarchical-closed-loop-development-v2"
RESULT_PROTOCOL = (
    "tsc-v64r60-external-v9-hierarchical-closed-loop-development-rollout-v2"
)
PHASE_POLICY = "phase_pressure"
HIERARCHICAL_RUNTIME_POLICY = (
    f"{ANCHOR_FAMILY}_contrast_hierarchical_guard"
)


@dataclass(frozen=True)
class HierarchicalExecutionCandidate:
    key: str
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    execution_region_key: str
    execution_trust_region: ResidualExecutionTrustRegionConfig


def development_candidates(
    protocol: Mapping[str, Any],
) -> tuple[HierarchicalExecutionCandidate, ...]:
    rows = tuple(protocol["development"]["method_candidates"])
    candidates = tuple(
        HierarchicalExecutionCandidate(
            key=str(row["key"]),
            runtime_policy=str(row["runtime_policy"]),
            coordination_mode=str(row["coordination_mode"]),
            cooldown_intervals=int(row["cooldown_intervals"]),
            execution_region_key=str(row["execution_trust_region"]["key"]),
            execution_trust_region=ResidualExecutionTrustRegionConfig(
                **{
                    key: value
                    for key, value in dict(row["execution_trust_region"]).items()
                    if key != "key"
                }
            ),
        )
        for row in rows
    )
    if (
        len(candidates) != int(protocol["development"]["candidate_count"])
        or len(candidates) != 12
        or len({candidate.key for candidate in candidates}) != len(candidates)
        or any(
            candidate.runtime_policy != HIERARCHICAL_RUNTIME_POLICY
            or candidate.coordination_mode != "direct"
            or candidate.cooldown_intervals
            != candidate.execution_trust_region.global_cooldown_intervals
            for candidate in candidates
        )
    ):
        raise ValueError("hierarchical closed-loop candidate grid changed")
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
    if (
        policies != (PHASE_POLICY, *[row.key for row in development_candidates(protocol)])
        or len(identities) != int(development["matrix_size"])
        or len(identities) != len(set(identities))
    ):
        raise ValueError("hierarchical closed-loop development matrix changed")
    return identities


def _build_runtime_models(
    *, model: Mapping[str, Any], prediction_horizon_sec: int
) -> tuple[FrozenAnchoredBlendModel, Any]:
    offline = model["models"]
    wrapper = FrozenAnchoredBlendModel(
        anchor_model=offline.family_models[ANCHOR_FAMILY],
        correction_model=offline.family_models[CORRECTION_FAMILY],
        candidate=str(model["selected_candidate"]),
        anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
        correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
    )
    runtime = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [offline]},
        family=ANCHOR_FAMILY,
        model=wrapper,
        prediction_horizon_sec=int(prediction_horizon_sec),
        guard_config=model["guard"],
        regularizer_config=model["regularizer"],
        target_support=model["target_support"],
    )
    if (
        runtime.guards[ANCHOR_FAMILY] != model["guard"]
        or runtime.regularizers[ANCHOR_FAMILY] != model["regularizer"]
        or runtime.target_support is not model["target_support"]
    ):
        raise RuntimeError("hierarchical runtime changed a frozen gate")
    return wrapper, runtime


def candidate_payload(
    candidate: HierarchicalExecutionCandidate,
) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "execution_trust_region": {
            "key": candidate.execution_region_key,
            **asdict(candidate.execution_trust_region),
        },
    }


def run_hierarchical_closed_loop_development(
    *,
    protocol_path: Path,
    parent_protocol_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
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
    started = time.monotonic()
    protocol = _read_json(protocol_path)
    parent = _read_json(parent_protocol_path)
    offline = _read_json(offline_result_path)
    support = _read_json(support_audit_path)
    freeze_audit = _read_json(freeze_audit_path)
    environment = _read_json(environment_protocol_path)
    environment_parent = _read_json(environment_parent_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or _sha256(parent_protocol_path) != str(protocol["parent_protocol"]["sha256"])
        or _sha256(offline_result_path)
        != str(protocol["offline_authorization"]["sha256"])
        or offline.get("protocol") != OFFLINE_RESULT_PROTOCOL
        or offline.get("decision") != AUTHORIZATION_DECISION
        or not bool(offline.get("offline_confirmation_gate", {}).get("passed", False))
        or _sha256(support_audit_path)
        != str(protocol["support_runtime_equivalence"]["sha256"])
        or support.get("protocol") != SUPPORT_AUDIT_PROTOCOL
        or support.get("status") != "PASS"
        or support.get("decision") != SUPPORT_AUDIT_DECISION
        or _sha256(freeze_audit_path)
        != str(protocol["hierarchical_freeze_audit"]["sha256"])
        or freeze_audit.get("status") != "PASS"
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
        or _sha256(environment_protocol_path)
        != str(protocol["environment"]["protocol"]["sha256"])
        or _sha256(environment_parent_path)
        != str(protocol["environment"]["network_parent_protocol"]["sha256"])
        or _sha256(external_manifest_path)
        != str(protocol["environment"]["external_manifest"]["sha256"])
        or _sha256(conversion_manifest_path)
        != str(protocol["environment"]["conversion_manifest"]["sha256"])
        or environment.get("protocol")
        != "tsc-v54r50-external-v9-full-budget-seed-blocked-refit-confirmation-v1"
        or environment_parent.get("protocol")
        != "tsc-v53r49-external-v9-network-repair-confirmation-v1"
    ):
        raise ValueError("hierarchical closed-loop evidence chain changed")
    if not bool(
        protocol.get("scientific_amendment", {}).get(
            "no_development_validation_or_prospective_outcome_inspected", False
        )
    ):
        raise ValueError("hierarchical closed-loop amendment is not outcome independent")
    development = dict(protocol["development"])
    if int(seed) not in {int(value) for value in development["seeds"]}:
        raise ValueError("seed is outside hierarchical development")
    matching_cities = [
        str(city)
        for city, scenarios in development["city_scenarios"].items()
        if scenario in {str(value) for value in scenarios}
    ]
    if len(matching_cities) != 1:
        raise ValueError("scenario lacks one hierarchical development city")
    city = matching_cities[0]
    candidates = {row.key: row for row in development_candidates(protocol)}
    if policy != PHASE_POLICY and policy not in candidates:
        raise ValueError("policy is outside hierarchical development")

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    if (
        scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != city
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
        or libsumo_version() != str(protocol["environment"]["sumo_version"])
    ):
        raise ValueError("hierarchical closed-loop environment changed")

    wrapper = None
    runtime_models = None
    execution_region = None
    cooldown_intervals = None
    runtime_policy = PHASE_POLICY
    model_hash = None
    certificate_hash = None
    if policy != PHASE_POLICY:
        _, model = _load_frozen_city_model(
            city=city,
            freeze_root=freeze_root,
            protocol=parent,
            freeze_audit=freeze_audit,
        )
        model_hash = str(parent["city_artifacts"][city]["model"]["sha256"])
        certificate_hash = str(
            parent["city_artifacts"][city]["certificate"]["sha256"]
        )
        wrapper, runtime_models = _build_runtime_models(
            model=model,
            prediction_horizon_sec=int(
                protocol["environment"]["prediction_horizon_sec"]
            ),
        )
        candidate = candidates[policy]
        runtime_policy = candidate.runtime_policy
        execution_region = candidate.execution_trust_region
        cooldown_intervals = candidate.cooldown_intervals

    temporary_tripinfo = tripinfo_out.with_suffix(
        tripinfo_out.suffix + f".tmp-{os.getpid()}"
    )
    try:
        metrics = evaluate_policy_v3(
            sumo_api=load_libsumo(),
            sumocfg=manifest.sumocfgs[scenario],
            scenario=scenario,
            policy=runtime_policy,
            models=runtime_models,
            duration_sec=float(protocol["environment"]["duration_sec"]),
            control_interval_sec=int(protocol["environment"]["control_interval_sec"]),
            warmup_sec=float(protocol["environment"]["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode="direct",
            residual_cooldown_intervals_override=cooldown_intervals,
            residual_execution_trust_region=execution_region,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"hierarchical rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("hierarchical closed-loop rollout contains teleports")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("hierarchical rollout lacks tripinfo evidence")
        temporary_tripinfo.replace(tripinfo_out)
    except Exception:
        temporary_tripinfo.unlink(missing_ok=True)
        raise
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "analysis_stage": "hierarchical_closed_loop_development",
        "selection_eligible_evidence": True,
        "untouched_validation_evidence": False,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "offline_authorization_sha256": _sha256(offline_result_path),
        "support_equivalence_audit_sha256": _sha256(support_audit_path),
        "hierarchical_freeze_audit_sha256": _sha256(freeze_audit_path),
        "environment_protocol_sha256": _sha256(environment_protocol_path),
        "environment_parent_sha256": _sha256(environment_parent_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": str(
            protocol["environment"]["conversion_tree_sha256"]
        ),
        "sumo_version": libsumo_version(),
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(policy),
        "runtime_policy": runtime_policy,
        "model_sha256": model_hash,
        "certificate_sha256": certificate_hash,
        "controller_refit": False,
        "execution_candidate": (
            candidate_payload(candidates[policy])
            if policy != PHASE_POLICY
            else None
        ),
        "anchored_blend_audit": wrapper.audit.to_dict() if wrapper else None,
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
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite hierarchical rollout")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_hierarchical_closed_loop_development(
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        scenario=args.scenario,
        seed=args.seed,
        policy=args.policy,
        tripinfo_out=args.tripinfo_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": payload["city"],
                "scenario": payload["scenario"],
                "seed": payload["seed"],
                "policy": payload["policy"],
                "mean_waiting_time": payload["metrics"][
                    "mean_tripinfo_waiting_time"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
