"""Run one frozen v81 multihorizon estimand-aligned closed-loop rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    _build_runtime_models,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (
    load_multihorizon_state_originator,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ResidualExecutionTrustRegionConfig,
    evaluate_policy_v3,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (
    PRIMARY_ARTIFACT_KEY,
    SECONDARY_ARTIFACT_KEY,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_estimand_aligned_closed_loop_development import (
    DIAGNOSTIC_CANDIDATE_SPECS,
    DIAGNOSTIC_METHOD_KEYS,
    METHOD_KEYS,
    PHASE_POLICY,
    PRIMARY_CANDIDATE_SPECS,
    PRIMARY_METHOD_KEYS,
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-estimand-aligned-closed-loop-rollout-v1"
ANALYSIS_STAGE = "multihorizon_estimand_aligned_closed_loop_development"


@dataclass(frozen=True)
class MultihorizonExecutionCandidate:
    key: str
    artifact_key: str
    candidate_role: str
    selection_eligible_closed_loop: bool
    runtime_policy: str
    coordination_mode: str
    cooldown_intervals: int
    execution_trust_region: ResidualExecutionTrustRegionConfig


def development_candidates(
    protocol: Mapping[str, Any],
) -> tuple[MultihorizonExecutionCandidate, ...]:
    rows = tuple(protocol["development"]["method_candidates"])
    candidates = tuple(
        MultihorizonExecutionCandidate(
            key=str(row["key"]),
            artifact_key=str(row["artifact_key"]),
            candidate_role=str(row["candidate_role"]),
            selection_eligible_closed_loop=bool(
                row["selection_eligible_closed_loop"]
            ),
            runtime_policy=str(row["runtime_policy"]),
            coordination_mode=str(row["coordination_mode"]),
            cooldown_intervals=int(row["cooldown_intervals"]),
            execution_trust_region=ResidualExecutionTrustRegionConfig(
                **dict(row["execution_trust_region"])
            ),
        )
        for row in rows
    )
    expected_specs = (*PRIMARY_CANDIDATE_SPECS, *DIAGNOSTIC_CANDIDATE_SPECS)
    primary_count = len(PRIMARY_CANDIDATE_SPECS)
    if (
        tuple(candidate.key for candidate in candidates) != METHOD_KEYS
        or tuple(candidate.cooldown_intervals for candidate in candidates)
        != (0,) * len(expected_specs)
        or tuple(
            candidate.execution_trust_region.global_cooldown_intervals
            for candidate in candidates
        )
        != tuple(row[1] for row in expected_specs)
        or tuple(
            candidate.execution_trust_region.min_priority
            for candidate in candidates
        )
        != tuple(row[2] for row in expected_specs)
        or tuple(candidate.artifact_key for candidate in candidates[:primary_count])
        != (PRIMARY_ARTIFACT_KEY,) * primary_count
        or tuple(candidate.artifact_key for candidate in candidates[primary_count:])
        != (SECONDARY_ARTIFACT_KEY,) * len(DIAGNOSTIC_CANDIDATE_SPECS)
        or any(
            candidate.selection_eligible_closed_loop is not (index < primary_count)
            or candidate.runtime_policy != HIERARCHICAL_RUNTIME_POLICY
            or candidate.coordination_mode != "direct"
            or candidate.execution_trust_region.max_simultaneous_overrides != 1
            for index, candidate in enumerate(candidates)
        )
    ):
        raise ValueError("v81 execution candidate grid changed")
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
        policies != (PHASE_POLICY, *METHOD_KEYS)
        or len(identities) != int(development["matrix_size"])
        or len(identities) != 198
        or len(identities) != len(set(identities))
    ):
        raise ValueError("v81 rollout matrix changed")
    return identities


def candidate_payload(candidate: MultihorizonExecutionCandidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "artifact_key": candidate.artifact_key,
        "candidate_role": candidate.candidate_role,
        "selection_eligible_closed_loop": candidate.selection_eligible_closed_loop,
        "runtime_policy": candidate.runtime_policy,
        "coordination_mode": candidate.coordination_mode,
        "cooldown_intervals": candidate.cooldown_intervals,
        "execution_trust_region": asdict(candidate.execution_trust_region),
    }


def _validate_evidence(
    *,
    protocol: Mapping[str, Any],
    v80_protocol_path: Path,
    v80_audit_path: Path,
    alignment_diagnostic_path: Path,
    artifact_result_path: Path,
    artifact_audit_path: Path,
    artifact_root: Path,
    parent_runtime_protocol_path: Path,
    hierarchical_parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    hierarchical_freeze_audit_path: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_manifest_path: Path,
) -> None:
    artifact = dict(protocol["artifact"])
    runtime = dict(protocol["runtime_parent"])
    environment = dict(protocol["environment"])
    development_parent = dict(protocol["development_parent"])
    checks = [
        (
            _sha256(v80_protocol_path),
            development_parent["v80_protocol"]["sha256"],
        ),
        (
            _sha256(v80_audit_path),
            development_parent["v80_independent_audit"]["sha256"],
        ),
        (
            _sha256(alignment_diagnostic_path),
            development_parent["estimand_alignment_diagnostic"]["sha256"],
        ),
        (_sha256(artifact_result_path), artifact["freeze_result"]["sha256"]),
        (_sha256(artifact_audit_path), artifact["independent_audit"]["sha256"]),
        (_sha256(parent_runtime_protocol_path), runtime["protocol"]["sha256"]),
        (_sha256(hierarchical_parent_path), runtime["hierarchical_parent"]["sha256"]),
        (_sha256(offline_result_path), runtime["offline_result"]["sha256"]),
        (_sha256(support_audit_path), runtime["support_audit"]["sha256"]),
        (
            _sha256(hierarchical_freeze_audit_path),
            runtime["hierarchical_freeze_audit"]["sha256"],
        ),
        (_sha256(environment_protocol_path), environment["protocol"]["sha256"]),
        (
            _sha256(environment_parent_path),
            environment["network_parent_protocol"]["sha256"],
        ),
        (_sha256(external_manifest_path), environment["external_manifest"]["sha256"]),
        (
            _sha256(conversion_manifest_path),
            environment["conversion_manifest"]["sha256"],
        ),
    ]
    for key, row in artifact["artifacts"].items():
        checks.extend(
            (
                (
                    _sha256(Path(artifact_root) / key / "model.pkl"),
                    row["model"]["sha256"],
                ),
                (
                    _sha256(Path(artifact_root) / key / "freeze.json"),
                    row["certificate"]["sha256"],
                ),
            )
        )
    if any(str(observed) != str(expected) for observed, expected in checks):
        raise ValueError("v81 closed-loop evidence changed")


def run_multihorizon_estimand_aligned_closed_loop_development(
    *,
    protocol_path: Path,
    v80_protocol_path: Path,
    v80_audit_path: Path,
    alignment_diagnostic_path: Path,
    artifact_result_path: Path,
    artifact_audit_path: Path,
    artifact_root: Path,
    parent_runtime_protocol_path: Path,
    hierarchical_parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
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
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v81 closed-loop protocol changed")
    _validate_evidence(
        protocol=protocol,
        v80_protocol_path=v80_protocol_path,
        v80_audit_path=v80_audit_path,
        alignment_diagnostic_path=alignment_diagnostic_path,
        artifact_result_path=artifact_result_path,
        artifact_audit_path=artifact_audit_path,
        artifact_root=artifact_root,
        parent_runtime_protocol_path=parent_runtime_protocol_path,
        hierarchical_parent_path=hierarchical_parent_path,
        offline_result_path=offline_result_path,
        support_audit_path=support_audit_path,
        hierarchical_freeze_audit_path=hierarchical_freeze_audit_path,
        environment_protocol_path=environment_protocol_path,
        environment_parent_path=environment_parent_path,
        external_manifest_path=external_manifest_path,
        conversion_manifest_path=conversion_manifest_path,
    )
    identity = ("jinan", str(scenario), int(seed), str(policy))
    if identity not in set(all_rollout_identities(protocol)):
        raise ValueError("rollout identity is outside frozen v81 development")
    candidates = {row.key: row for row in development_candidates(protocol)}

    conversion_root = Path(conversion_root).resolve()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    environment = dict(protocol["environment"])
    if (
        scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != "jinan"
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
        or libsumo_version() != str(environment["sumo_version"])
        or int(environment["prediction_horizon_sec"]) != 60
    ):
        raise ValueError("v81 SUMO environment changed")

    runtime_models = None
    runtime_policy = PHASE_POLICY
    execution_region = None
    cooldown_intervals = None
    hierarchy_model_hash = None
    hierarchy_certificate_hash = None
    originator_diagnostics = None
    candidate = None
    artifact_key = None
    artifact_model_sha = None
    artifact_certificate_sha = None
    if policy != PHASE_POLICY:
        parent_runtime = _read_json(parent_runtime_protocol_path)
        hierarchical_parent = _read_json(hierarchical_parent_path)
        hierarchical_freeze_audit = _read_json(hierarchical_freeze_audit_path)
        _, frozen_model = _load_frozen_city_model(
            city="jinan",
            freeze_root=hierarchical_freeze_root,
            protocol=hierarchical_parent,
            freeze_audit=hierarchical_freeze_audit,
        )
        hierarchy_model_hash = str(
            hierarchical_parent["city_artifacts"]["jinan"]["model"]["sha256"]
        )
        hierarchy_certificate_hash = str(
            hierarchical_parent["city_artifacts"]["jinan"]["certificate"][
                "sha256"
            ]
        )
        _, runtime_models = _build_runtime_models(
            model=frozen_model,
            prediction_horizon_sec=int(environment["prediction_horizon_sec"]),
        )
        candidate = candidates[policy]
        artifact_key = candidate.artifact_key
        originator, _, _ = load_multihorizon_state_originator(
            artifact_root, artifact_key
        )
        runtime_models = replace(runtime_models, action_originator=originator)
        originator_diagnostics = originator.diagnostics()
        runtime_policy = candidate.runtime_policy
        execution_region = candidate.execution_trust_region
        cooldown_intervals = candidate.cooldown_intervals
        artifact_record = protocol["artifact"]["artifacts"][artifact_key]
        artifact_model_sha = str(artifact_record["model"]["sha256"])
        artifact_certificate_sha = str(artifact_record["certificate"]["sha256"])
        if (
            runtime_models.action_originator is not originator
            or runtime_models.execution_veto is not None
            or int(runtime_models.prediction_horizon_sec) != 60
        ):
            raise RuntimeError("v81 originator runtime injection changed")

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
            duration_sec=float(environment["duration_sec"]),
            control_interval_sec=int(environment["control_interval_sec"]),
            warmup_sec=float(environment["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode="direct",
            residual_cooldown_intervals_override=cooldown_intervals,
            residual_execution_trust_region=execution_region,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"v81 rollout failed: {metrics.get('error')}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            raise RuntimeError("v81 rollout contains teleports")
        guard = dict(metrics.get("guard_audit", {}))
        if policy != PHASE_POLICY and int(
            guard.get("target_originator_decisions", 0)
        ) <= 0:
            raise RuntimeError("v81 rollout did not enter the originator")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("v81 rollout lacks tripinfo evidence")
        temporary_tripinfo.replace(tripinfo_out)
    except Exception:
        temporary_tripinfo.unlink(missing_ok=True)
        raise
    is_primary = policy == PHASE_POLICY or policy in PRIMARY_METHOD_KEYS
    proposal_origin = "phase_pressure"
    if artifact_key == PRIMARY_ARTIFACT_KEY:
        proposal_origin = "target_multihorizon_compact_state_latent"
    elif artifact_key == SECONDARY_ARTIFACT_KEY:
        proposal_origin = "target_multihorizon_state_action_latent"
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "analysis_stage": ANALYSIS_STAGE,
        "selection_eligible_evidence": bool(is_primary),
        "diagnostic_only_evidence": bool(policy in DIAGNOSTIC_METHOD_KEYS),
        "untouched_validation_evidence": False,
        "prospective_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "v80_protocol_sha256": _sha256(v80_protocol_path),
        "v80_audit_sha256": _sha256(v80_audit_path),
        "alignment_diagnostic_sha256": _sha256(alignment_diagnostic_path),
        "artifact_result_sha256": _sha256(artifact_result_path),
        "artifact_audit_sha256": _sha256(artifact_audit_path),
        "artifact_key": artifact_key,
        "artifact_model_sha256": artifact_model_sha,
        "artifact_certificate_sha256": artifact_certificate_sha,
        "parent_runtime_protocol_sha256": _sha256(parent_runtime_protocol_path),
        "hierarchical_parent_sha256": _sha256(hierarchical_parent_path),
        "hierarchical_model_sha256": hierarchy_model_hash,
        "hierarchical_certificate_sha256": hierarchy_certificate_hash,
        "environment_protocol_sha256": _sha256(environment_protocol_path),
        "environment_parent_sha256": _sha256(environment_parent_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": str(environment["conversion_tree_sha256"]),
        "sumo_version": libsumo_version(),
        "city": "jinan",
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": str(policy),
        "runtime_policy": runtime_policy,
        "controller_refit": False,
        "target_labels_used_at_fit": policy != PHASE_POLICY,
        "target_outcomes_used_online": False,
        "prediction_horizon_sec": 60 if policy != PHASE_POLICY else None,
        "proposal_origin": proposal_origin,
        "candidate_role": candidate.candidate_role if candidate else "baseline",
        "execution_candidate": candidate_payload(candidate) if candidate else None,
        "originator_diagnostics": originator_diagnostics,
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
    parser.add_argument("--v80-protocol", type=Path, required=True)
    parser.add_argument("--v80-audit", type=Path, required=True)
    parser.add_argument("--alignment-diagnostic", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-runtime-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite v81 rollout")
    result = run_multihorizon_estimand_aligned_closed_loop_development(
        protocol_path=args.protocol,
        v80_protocol_path=args.v80_protocol,
        v80_audit_path=args.v80_audit,
        alignment_diagnostic_path=args.alignment_diagnostic,
        artifact_result_path=args.artifact_result,
        artifact_audit_path=args.artifact_audit,
        artifact_root=args.artifact_root,
        parent_runtime_protocol_path=args.parent_runtime_protocol,
        hierarchical_parent_path=args.hierarchical_parent,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        scenario=str(args.scenario),
        seed=int(args.seed),
        policy=str(args.policy),
        tripinfo_out=args.tripinfo_out,
    )
    _atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "policy": result["policy"],
                "mean_waiting_time": result["metrics"][
                    "mean_tripinfo_waiting_time"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
