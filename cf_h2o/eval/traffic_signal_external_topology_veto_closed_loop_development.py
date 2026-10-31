"""Closed-loop development for the frozen v77 topology-veto execution veto."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    _build_runtime_models,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_external_topology_veto_freeze import (
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
    evaluate_policy_v3,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.topology_target_veto import (
    ARTIFACT_PROTOCOL,
    TopologyTargetVeto,
)


PROTOCOL = "tsc-v78r74-external-v9-topology-veto-closed-loop-development-v2"
RESULT_PROTOCOL = (
    "tsc-v78r74-external-v9-topology-veto-closed-loop-development-rollout-v2"
)
ANALYSIS_STAGE = "topology_veto_closed_loop_development"
CANDIDATE_KEY = "cfcmt_topology_veto_cd450"
PHASE_POLICY = "phase_pressure"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TopologyVetoDevelopmentCandidate:
    key: str
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    max_simultaneous_overrides: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def development_candidate(protocol: Mapping[str, Any]) -> TopologyVetoDevelopmentCandidate:
    row = protocol["development"]["candidate"]
    candidate = TopologyVetoDevelopmentCandidate(
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
    if (
        candidate.key != CANDIDATE_KEY
        or candidate.runtime_policy != HIERARCHICAL_RUNTIME_POLICY
        or candidate.coordination_mode != "direct"
        or candidate.cooldown_intervals != 44
        or candidate.max_simultaneous_overrides != 1
    ):
        raise ValueError("v78 development candidate changed")
    return candidate


def candidate_payload(candidate: TopologyVetoDevelopmentCandidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "max_simultaneous_overrides": candidate.max_simultaneous_overrides,
        "execution_trust_region": asdict(candidate.execution_trust_region),
    }


def all_rollout_identities(
    protocol: Mapping[str, Any]
) -> tuple[tuple[str, str, int, str], ...]:
    development = protocol["development"]
    policies = (PHASE_POLICY, development_candidate(protocol).key)
    return tuple(
        (str(city), str(scenario), int(seed), policy)
        for city, scenarios in sorted(development["city_scenarios"].items())
        for scenario in scenarios
        for seed in development["seeds"]
        for policy in policies
    )


def _load_veto(
    *,
    city: str,
    scenario: str,
    artifact_root: Path,
    joint_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], TopologyTargetVeto]:
    audited = dict(joint_audit["cities"][city])
    model_path = Path(artifact_root) / city / "model.pkl"
    certificate_path = Path(artifact_root) / city / "freeze.json"
    if (
        _sha256(model_path) != audited["model"]["sha256"]
        or _sha256(certificate_path) != audited["certificate"]["sha256"]
    ):
        raise ValueError(f"v78 topology-veto artifact changed for {city}")
    artifact = pickle.loads(model_path.read_bytes())
    scenario_vetoes = artifact.get("scenario_vetoes", {})
    veto = scenario_vetoes.get(scenario)
    if (
        artifact.get("protocol") != ARTIFACT_PROTOCOL
        or artifact.get("city") != city
        or set(scenario_vetoes) != set(audited["scenarios"])
        or not isinstance(veto, TopologyTargetVeto)
        or artifact.get("zero_shot") is not False
        or artifact.get("scenario_id_is_model_feature") is not False
        or artifact.get("veto_role")
        != "static_topology_veto_only_never_action_originator"
        or bool(veto.config.enabled) != bool(audited["active"])
    ):
        raise ValueError(f"v78 topology-veto model contract changed for {city}")
    return artifact, veto


def run_topology_veto_closed_loop_development(
    *,
    protocol_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    veto_freeze_result_path: Path,
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
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    hierarchical_audit = json.loads(
        Path(hierarchical_freeze_audit_path).read_text(encoding="utf-8")
    )
    veto_freeze_result = json.loads(
        Path(veto_freeze_result_path).read_text(encoding="utf-8")
    )
    veto_audit = json.loads(Path(veto_joint_audit_path).read_text(encoding="utf-8"))
    if (
        protocol.get("protocol") != PROTOCOL
        or _sha256(proposal_parent_protocol_path)
        != protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(hierarchical_freeze_audit_path)
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or hierarchical_audit.get("status") != "PASS"
        or veto_freeze_result.get("protocol") != FREEZE_RESULT_PROTOCOL
        or veto_freeze_result.get("status") != "PASS"
        or _sha256(veto_freeze_result_path)
        != protocol["topology_veto_freeze_result"]["sha256"]
        or veto_audit.get("status") != "PASS"
        or veto_audit.get("decision")
        != "authorize_topology_veto_closed_loop_development_only"
        or _sha256(veto_joint_audit_path)
        != protocol["topology_veto_joint_audit"]["sha256"]
    ):
        raise ValueError("v78 closed-loop evidence chain changed")
    source_gate = {
        str(relative): (PROJECT_ROOT / str(relative)).is_file()
        and _sha256(PROJECT_ROOT / str(relative)) == str(expected)
        for relative, expected in protocol["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"v78 runtime source freeze changed: {source_gate}")
    if (
        _sha256(external_manifest_path)
        != protocol["environment"]["external_manifest"]["sha256"]
        or _sha256(conversion_manifest_path)
        != protocol["environment"]["conversion_manifest"]["sha256"]
    ):
        raise ValueError("v78 environment evidence changed")
    development = protocol["development"]
    if int(seed) not in {int(value) for value in development["seeds"]}:
        raise ValueError("seed is outside v78 development")
    matching_cities = [
        str(city)
        for city, scenarios in development["city_scenarios"].items()
        if str(scenario) in {str(value) for value in scenarios}
    ]
    if len(matching_cities) != 1:
        raise ValueError("scenario lacks one v78 development city")
    city = matching_cities[0]
    candidate = development_candidate(protocol)
    if str(policy) not in {PHASE_POLICY, candidate.key}:
        raise ValueError("policy is outside v78 development")

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    if (
        scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != city
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
        or libsumo_version() != str(protocol["environment"]["sumo_version"])
    ):
        raise ValueError("v78 closed-loop environment changed")
    _, frozen_model = _load_frozen_city_model(
        city=city,
        freeze_root=hierarchical_freeze_root,
        protocol=proposal_parent,
        freeze_audit=hierarchical_audit,
    )
    veto_artifact, target_veto = _load_veto(
        city=city,
        scenario=scenario,
        artifact_root=veto_artifact_root,
        joint_audit=veto_audit,
    )
    wrapper, runtime_models = _build_runtime_models(
        model=frozen_model,
        prediction_horizon_sec=int(protocol["proposal"]["prediction_horizon_sec"]),
    )
    is_method = str(policy) == candidate.key
    if is_method:
        runtime_models = replace(runtime_models, execution_veto=target_veto)
        runtime_policy = candidate.runtime_policy
    else:
        runtime_models = replace(runtime_models, execution_veto=None)
        runtime_policy = PHASE_POLICY
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
            raise RuntimeError(f"v78 rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("v78 rollout contains teleports")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("v78 rollout lacks tripinfo evidence")
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
        "runtime_executable_source_gate": source_gate,
        "analysis_stage": ANALYSIS_STAGE,
        "selection_eligible_evidence": True,
        "untouched_validation_evidence": False,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "proposal_parent_protocol_sha256": _sha256(
            proposal_parent_protocol_path
        ),
        "hierarchical_freeze_audit_sha256": _sha256(
            hierarchical_freeze_audit_path
        ),
        "topology_veto_freeze_result_sha256": _sha256(veto_freeze_result_path),
        "topology_veto_joint_audit_sha256": _sha256(veto_joint_audit_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": protocol["environment"]["conversion_tree_sha256"],
        "sumo_version": libsumo_version(),
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(policy),
        "policy_role": "topology_veto_cfcmt" if is_method else "paired_phase_pressure",
        "runtime_policy": runtime_policy,
        "target_veto_active": bool(target_veto.config.enabled) if is_method else False,
        "target_veto_artifact_active": bool(target_veto.config.enabled),
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
