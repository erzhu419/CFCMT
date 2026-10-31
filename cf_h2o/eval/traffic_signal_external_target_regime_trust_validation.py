"""Untouched-seed validation of the frozen target-regime trust successor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    _build_runtime_models,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_target_regime_trust_freeze import (
    FREEZE_PROTOCOL as REGIME_FREEZE_PROTOCOL,
    load_frozen_regime_trust,
)
from cf_h2o.eval.traffic_signal_external_topology_veto_closed_loop_development import (
    _load_veto,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
    evaluate_policy_v3,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


PROTOCOL = "tsc-v81r77-external-v9-target-regime-trust-validation-v1"
RESULT_PROTOCOL = "tsc-v81r77-target-regime-trust-validation-rollout-v1"
ANALYSIS_STAGE = "untouched_target_regime_trust_validation"
CANDIDATE_KEY = "cfcmt_target_regime_trust_cd450"
PHASE_POLICY = "phase_pressure"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidationCandidate:
    key: str
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    max_simultaneous_overrides: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def validation_candidate(protocol: Mapping[str, Any]) -> ValidationCandidate:
    row = protocol["validation"]["candidate"]
    candidate = ValidationCandidate(
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
        raise ValueError("target regime validation candidate changed")
    return candidate


def candidate_payload(candidate: ValidationCandidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "max_simultaneous_overrides": candidate.max_simultaneous_overrides,
        "execution_trust_region": asdict(candidate.execution_trust_region),
    }


def _scenario_city(protocol: Mapping[str, Any], scenario: str) -> str:
    matches = [
        str(city)
        for city, scenarios in protocol["validation"]["city_scenarios"].items()
        if str(scenario) in {str(value) for value in scenarios}
    ]
    if len(matches) != 1:
        raise ValueError("validation scenario lacks one target city")
    return matches[0]


def frozen_regime_decision(
    *,
    protocol: Mapping[str, Any],
    artifact: Mapping[str, Any],
    model: Any,
    city: str,
    scenario: str,
) -> tuple[bool, dict[str, Any]]:
    """Resolve the policy before SUMO starts and without validation-day inputs."""

    declared = bool(protocol["validation"]["scenario_gate"][str(scenario)])
    if str(city) != str(artifact["city"]):
        if declared:
            raise ValueError("fallback city cannot activate target regime trust")
        return False, {
            "source": "frozen_city_fallback",
            "historical_phase_mean_queue_per_lane": None,
            "predicted_relative_delta": None,
            "history_source_seeds": [],
            "heldout_or_validation_metrics_used": False,
        }
    row = dict(artifact["scenario_history"][str(scenario)])
    feature = float(row["historical_phase_mean_queue_per_lane"])
    predicted = float(model.predict_relative_delta(feature))
    active = bool(predicted < 0.0)
    if (
        abs(predicted - float(row["predicted_relative_delta"])) > 1e-12
        or active != bool(row["trust_local_controller"])
        or active != declared
        or set(int(value) for value in row["history_source_seeds"])
        != set(int(value) for value in protocol["development_evidence"]["seeds"])
    ):
        raise ValueError("frozen target regime decision changed")
    return active, {
        "source": "frozen_target_history_artifact",
        "historical_phase_mean_queue_per_lane": feature,
        "predicted_relative_delta": predicted,
        "history_source_seeds": [int(value) for value in row["history_source_seeds"]],
        "heldout_or_validation_metrics_used": False,
    }


def all_rollout_identities(
    protocol: Mapping[str, Any]
) -> tuple[tuple[str, str, int, str], ...]:
    validation = protocol["validation"]
    policies = (PHASE_POLICY, validation_candidate(protocol).key)
    return tuple(
        (str(city), str(scenario), int(seed), policy)
        for city, scenarios in sorted(validation["city_scenarios"].items())
        for scenario in scenarios
        for seed in validation["seeds"]
        for policy in policies
    )


def run_target_regime_trust_validation(
    *,
    protocol_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    veto_freeze_result_path: Path,
    veto_joint_audit_path: Path,
    veto_artifact_root: Path,
    regime_freeze_result_path: Path,
    regime_artifact_path: Path,
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
    veto_audit = json.loads(Path(veto_joint_audit_path).read_text(encoding="utf-8"))
    regime_freeze = json.loads(
        Path(regime_freeze_result_path).read_text(encoding="utf-8")
    )
    evidence = protocol["execution_evidence"]
    if (
        protocol.get("protocol") != PROTOCOL
        or _sha256(proposal_parent_protocol_path)
        != evidence["proposal_parent_protocol"]["sha256"]
        or _sha256(hierarchical_freeze_audit_path)
        != evidence["hierarchical_freeze_audit"]["sha256"]
        or hierarchical_audit.get("status") != "PASS"
        or _sha256(veto_freeze_result_path)
        != evidence["topology_veto_freeze_result"]["sha256"]
        or _sha256(veto_joint_audit_path)
        != evidence["topology_veto_joint_audit"]["sha256"]
        or veto_audit.get("status") != "PASS"
        or regime_freeze.get("protocol") != REGIME_FREEZE_PROTOCOL
        or regime_freeze.get("status") != "PASS"
        or _sha256(regime_freeze_result_path)
        != protocol["regime_trust_freeze"]["sha256"]
    ):
        raise ValueError("target regime validation evidence chain changed")
    source_gate = {
        str(relative): (PROJECT_ROOT / str(relative)).is_file()
        and _sha256(PROJECT_ROOT / str(relative)) == str(expected)
        for relative, expected in protocol["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"target regime validation source freeze changed: {source_gate}")
    validation = protocol["validation"]
    validation_seeds = {int(value) for value in validation["seeds"]}
    development_seeds = {
        int(value) for value in protocol["development_evidence"]["seeds"]
    }
    if int(seed) not in validation_seeds or validation_seeds & development_seeds:
        raise ValueError("seed is outside untouched target regime validation")
    city = _scenario_city(protocol, str(scenario))
    candidate = validation_candidate(protocol)
    if str(policy) not in {PHASE_POLICY, candidate.key}:
        raise ValueError("policy is outside target regime validation")

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    if (
        _sha256(external_manifest_path) != protocol["environment"]["external_manifest"]["sha256"]
        or _sha256(conversion_manifest_path)
        != protocol["environment"]["conversion_manifest"]["sha256"]
        or scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != city
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
        or libsumo_version() != str(protocol["environment"]["sumo_version"])
    ):
        raise ValueError("target regime validation environment changed")
    _, frozen_model = _load_frozen_city_model(
        city=city,
        freeze_root=hierarchical_freeze_root,
        protocol=proposal_parent,
        freeze_audit=hierarchical_audit,
    )
    _, target_veto = _load_veto(
        city=city,
        scenario=scenario,
        artifact_root=veto_artifact_root,
        joint_audit=veto_audit,
    )
    regime_artifact, regime_model = load_frozen_regime_trust(
        artifact_path=regime_artifact_path,
        expected_sha256=protocol["regime_trust_artifact"]["sha256"],
    )
    regime_active, regime_evidence = frozen_regime_decision(
        protocol=protocol,
        artifact=regime_artifact,
        model=regime_model,
        city=city,
        scenario=scenario,
    )
    if regime_active and not bool(target_veto.config.enabled):
        raise ValueError("regime gate activated a city without a topology veto")

    wrapper, runtime_models = _build_runtime_models(
        model=frozen_model,
        prediction_horizon_sec=int(protocol["proposal"]["prediction_horizon_sec"]),
    )
    is_method = str(policy) == candidate.key
    execute_cfcmt = bool(is_method and regime_active)
    if execute_cfcmt:
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
            duration_sec=float(validation["duration_sec"]),
            control_interval_sec=int(protocol["environment"]["control_interval_sec"]),
            warmup_sec=float(protocol["environment"]["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode=candidate.coordination_mode,
            residual_cooldown_intervals_override=candidate.cooldown_intervals,
            residual_execution_trust_region=candidate.execution_trust_region,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"target regime validation rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("target regime validation rollout contains teleports")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("target regime validation lacks tripinfo evidence")
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
        "selection_eligible_evidence": False,
        "untouched_validation_evidence": True,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "regime_trust_freeze_sha256": _sha256(regime_freeze_result_path),
        "regime_trust_artifact_sha256": _sha256(regime_artifact_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": protocol["environment"]["conversion_tree_sha256"],
        "sumo_version": libsumo_version(),
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(policy),
        "policy_role": (
            "target_regime_gated_cfcmt" if is_method else "paired_phase_pressure"
        ),
        "runtime_policy": runtime_policy,
        "regime_trust_active": bool(regime_active) if is_method else False,
        "regime_trust_artifact_decision": bool(regime_active),
        "regime_trust_evidence": regime_evidence,
        "target_veto_active": execute_cfcmt,
        "target_veto_artifact_active": bool(target_veto.config.enabled),
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
