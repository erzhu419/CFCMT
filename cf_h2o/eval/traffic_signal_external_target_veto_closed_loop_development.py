"""Run one frozen v68 proposal-conditional closed-loop development rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
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
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalTargetVeto,
)
from scripts.cluster.audit_tsc_external_proposal_conditional_veto_successor import (
    AUDIT_DECISION as VETO_AUDIT_DECISION,
    AUDIT_PROTOCOL as VETO_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_closed_loop_protocol import (
    PROTOCOL as PROPOSAL_PARENT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (
    PROTOCOL as COLLECTION_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_proposal_conditional_veto_successor import (
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_target_veto_closed_loop_protocol import (
    PROTOCOL,
)


RESULT_PROTOCOL = (
    "tsc-v68r64-external-v9-proposal-conditional-closed-loop-development-rollout-v1"
)
ANALYSIS_STAGE = "proposal_conditional_closed_loop_development"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TargetVetoDevelopmentCandidate:
    key: str
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    max_simultaneous_overrides: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def development_candidates(
    protocol: Mapping[str, Any],
) -> tuple[TargetVetoDevelopmentCandidate, ...]:
    rows = tuple(protocol["development"]["method_candidates"])
    candidates = tuple(
        TargetVetoDevelopmentCandidate(
            key=str(row["key"]),
            runtime_policy=str(row["runtime_policy"]),
            coordination_mode=str(row["coordination_mode"]),
            cooldown_intervals=int(row["cooldown_intervals"]),
            max_simultaneous_overrides=int(row["max_simultaneous_overrides"]),
            execution_trust_region=ResidualExecutionTrustRegionConfig(
                enabled=True,
                min_priority=0.0,
                min_total_queue=0.0,
                min_total_vehicles=1.0,
                max_mean_speed=None,
                max_simultaneous_overrides=int(row["max_simultaneous_overrides"]),
                global_cooldown_intervals=int(row["cooldown_intervals"]),
            ),
        )
        for row in rows
    )
    if (
        len(candidates) != 3
        or len({candidate.key for candidate in candidates}) != 3
        or any(
            candidate.runtime_policy != HIERARCHICAL_RUNTIME_POLICY
            or candidate.coordination_mode != "direct"
            or candidate.max_simultaneous_overrides != 1
            or candidate.cooldown_intervals
            != candidate.execution_trust_region.global_cooldown_intervals
            for candidate in candidates
        )
    ):
        raise ValueError("target-veto closed-loop candidate grid changed")
    return candidates


def all_method_rollout_identities(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    development = dict(protocol["development"])
    candidates = development_candidates(protocol)
    identities = tuple(
        (str(city), str(scenario), int(seed), candidate.key)
        for city, scenarios in sorted(development["city_scenarios"].items())
        for scenario in scenarios
        for seed in development["seeds"]
        for candidate in candidates
    )
    expected = int(development["method_only_matrix_size"])
    if len(identities) != expected or len(identities) != 96:
        raise ValueError("target-veto method-only rollout matrix changed")
    return identities


def candidate_payload(candidate: TargetVetoDevelopmentCandidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "max_simultaneous_overrides": candidate.max_simultaneous_overrides,
        "execution_trust_region": asdict(candidate.execution_trust_region),
    }


def _load_target_veto(
    *, city: str, artifact_root: Path, joint_audit: Mapping[str, Any]
) -> tuple[dict[str, Any], ProposalConditionalTargetVeto]:
    audited = dict(joint_audit["cities"][city])
    model_path = Path(artifact_root) / city / "model.pkl"
    certificate_path = Path(artifact_root) / city / "freeze.json"
    if (
        _sha256(model_path) != audited["model"]["sha256"]
        or _sha256(certificate_path) != audited["certificate"]["sha256"]
    ):
        raise ValueError(f"target-veto artifact identity changed for {city}")
    artifact = pickle.loads(model_path.read_bytes())
    if (
        artifact.get("protocol")
        != "cfcmt-proposal-conditional-target-veto-model-v1"
        or artifact.get("city") != city
        or not isinstance(
            artifact.get("proposal_conditional_veto"),
            ProposalConditionalTargetVeto,
        )
        or artifact.get("zero_shot") is not False
        or artifact.get("veto_role")
        != "proposal_conditional_veto_only_never_action_originator"
    ):
        raise ValueError(f"target-veto model contract changed for {city}")
    return artifact, artifact["proposal_conditional_veto"]


def run_target_veto_closed_loop_development(
    *,
    protocol_path: Path,
    parent_protocol_path: Path,
    collection_protocol_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    veto_joint_audit_path: Path,
    veto_artifact_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
    scenario: str,
    seed: int,
    policy: str,
    tripinfo_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    parent = json.loads(Path(parent_protocol_path).read_text(encoding="utf-8"))
    collection = json.loads(Path(collection_protocol_path).read_text(encoding="utf-8"))
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    freeze_audit = json.loads(Path(freeze_audit_path).read_text(encoding="utf-8"))
    veto_audit = json.loads(Path(veto_joint_audit_path).read_text(encoding="utf-8"))
    if (
        protocol.get("protocol") != PROTOCOL
        or parent.get("protocol") != PARENT_PROTOCOL
        or _sha256(parent_protocol_path) != protocol["parent_protocol"]["sha256"]
        or collection.get("protocol") != COLLECTION_PROTOCOL
        or _sha256(collection_protocol_path)
        != protocol["collection_protocol"]["sha256"]
        or _sha256(collection_protocol_path)
        != parent["collection_protocol"]["sha256"]
        or proposal_parent.get("protocol") != PROPOSAL_PARENT_PROTOCOL
        or _sha256(proposal_parent_protocol_path)
        != protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(proposal_parent_protocol_path)
        != collection["parent_protocol"]["sha256"]
    ):
        raise ValueError("target-veto closed-loop parent evidence changed")
    runtime_source_gate = {
        str(relative): (PROJECT_ROOT / str(relative)).is_file()
        and _sha256(PROJECT_ROOT / str(relative)) == str(expected)
        for relative, expected in protocol.get(
            "runtime_executable_sources", {}
        ).items()
    }
    if not runtime_source_gate or not all(runtime_source_gate.values()):
        raise ValueError(
            f"target-veto runtime executable freeze changed: {runtime_source_gate}"
        )
    if (
        _sha256(freeze_audit_path)
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or _sha256(freeze_audit_path)
        != proposal_parent["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
        or veto_audit.get("protocol") != VETO_AUDIT_PROTOCOL
        or veto_audit.get("status") != "PASS"
        or veto_audit.get("decision") != VETO_AUDIT_DECISION
        or not bool(veto_audit.get("integrity_gate", {}).get("passed", False))
        or _sha256(veto_joint_audit_path)
        != protocol["target_veto_joint_audit"]["sha256"]
        or _sha256(external_manifest_path)
        != protocol["environment"]["external_manifest"]["sha256"]
        or _sha256(conversion_manifest_path)
        != protocol["environment"]["conversion_manifest"]["sha256"]
    ):
        raise ValueError("target-veto closed-loop evidence chain changed")
    development = dict(protocol["development"])
    if int(seed) not in {int(value) for value in development["seeds"]}:
        raise ValueError("seed is outside target-veto development")
    matching_cities = [
        str(city)
        for city, scenarios in development["city_scenarios"].items()
        if str(scenario) in {str(value) for value in scenarios}
    ]
    if len(matching_cities) != 1:
        raise ValueError("scenario lacks one target-veto development city")
    city = matching_cities[0]
    candidates = {row.key: row for row in development_candidates(protocol)}
    if policy not in candidates:
        raise ValueError("policy is outside target-veto development")

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    if (
        scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != city
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
        or libsumo_version() != str(protocol["environment"]["sumo_version"])
    ):
        raise ValueError("target-veto closed-loop environment changed")
    _, frozen_model = _load_frozen_city_model(
        city=city,
        freeze_root=freeze_root,
        protocol=proposal_parent,
        freeze_audit=freeze_audit,
    )
    veto_artifact, target_veto = _load_target_veto(
        city=city,
        artifact_root=veto_artifact_root,
        joint_audit=veto_audit,
    )
    wrapper, runtime_models = _build_runtime_models(
        model=frozen_model,
        prediction_horizon_sec=int(protocol["proposal"]["prediction_horizon_sec"]),
    )
    runtime_models = replace(runtime_models, execution_veto=target_veto)
    candidate = candidates[policy]
    temporary_tripinfo = tripinfo_out.with_suffix(
        tripinfo_out.suffix + f".tmp-{os.getpid()}"
    )
    try:
        metrics = evaluate_policy_v3(
            sumo_api=load_libsumo(),
            sumocfg=manifest.sumocfgs[scenario],
            scenario=scenario,
            policy=candidate.runtime_policy,
            models=runtime_models,
            duration_sec=float(development["duration_sec"]),
            control_interval_sec=int(protocol["environment"]["control_interval_sec"]),
            warmup_sec=float(protocol["environment"]["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode=candidate.coordination_mode,
            residual_cooldown_intervals_override=candidate.cooldown_intervals,
            residual_execution_trust_region=candidate.execution_trust_region,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"target-veto rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("target-veto rollout contains teleports")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("target-veto rollout lacks tripinfo evidence")
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
        "runtime_executable_source_gate": runtime_source_gate,
        "analysis_stage": ANALYSIS_STAGE,
        "selection_eligible_evidence": True,
        "untouched_validation_evidence": False,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "collection_protocol_sha256": _sha256(collection_protocol_path),
        "proposal_parent_protocol_sha256": _sha256(
            proposal_parent_protocol_path
        ),
        "hierarchical_freeze_audit_sha256": _sha256(freeze_audit_path),
        "target_veto_joint_audit_sha256": _sha256(veto_joint_audit_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": protocol["environment"]["conversion_tree_sha256"],
        "sumo_version": libsumo_version(),
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(policy),
        "runtime_policy": candidate.runtime_policy,
        "proposal_model_sha256": protocol["proposal"]["city_artifacts"][city][
            "model"
        ]["sha256"],
        "proposal_certificate_sha256": protocol["proposal"]["city_artifacts"][city][
            "certificate"
        ]["sha256"],
        "target_veto_model_sha256": veto_audit["cities"][city]["model"]["sha256"],
        "target_veto_certificate_sha256": veto_audit["cities"][city]["certificate"][
            "sha256"
        ],
        "target_veto_active": bool(target_veto.config.enabled),
        "target_veto_config": asdict(target_veto.config),
        "target_veto_classification": veto_artifact["classification"],
        "controller_refit": False,
        "execution_candidate": candidate_payload(candidate),
        "anchored_blend_audit": wrapper.audit.to_dict(),
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
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--veto-artifact-root", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite target-veto rollout")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_target_veto_closed_loop_development(
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        collection_protocol_path=args.collection_protocol,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        veto_joint_audit_path=args.veto_joint_audit,
        veto_artifact_root=args.veto_artifact_root,
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
                "executed_overrides": payload["metrics"]["guard_audit"].get(
                    "executed_overrides", 0
                ),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
