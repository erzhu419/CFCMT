#!/usr/bin/env python3
"""Freeze the v80 multihorizon state-latent closed-loop development matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    HIERARCHICAL_RUNTIME_POLICY,
    PROTOCOL as PARENT_RUNTIME_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_audit import (  # noqa: E402
    AUTHORIZATION_DECISION as ARTIFACT_AUDIT_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as ARTIFACT_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as ARTIFACT_FREEZE_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as ARTIFACT_FREEZE_RESULT_PROTOCOL,
    load_multihorizon_state_originator,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (  # noqa: E402
    ADAPTATION_CLASSIFICATION,
    ARTIFACT_KEYS,
    PRIMARY_ARTIFACT_KEY,
    PROTOCOL as ARTIFACT_PROTOCOL,
    SECONDARY_ARTIFACT_KEY,
)


PROTOCOL = "tsc-v83r79-external-v9-multihorizon-state-latent-closed-loop-v1"
PARTITION_PROTOCOL = "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
PHASE_POLICY = "phase_pressure"
PRIMARY_METHOD_KEYS = (
    "mh_compact_cd0",
    "mh_compact_cd2",
    "mh_compact_cd5",
)
DIAGNOSTIC_METHOD_KEYS = (
    "mh_state_action_cd0",
    "mh_state_action_cd2",
    "mh_state_action_cd5",
)
METHOD_KEYS = (*PRIMARY_METHOD_KEYS, *DIAGNOSTIC_METHOD_KEYS)
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def freeze_protocol(
    *,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
    artifact_audit_path: Path,
    artifact_root: Path,
    partition_path: Path,
    parent_runtime_protocol_path: Path,
    hierarchical_parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_manifest_path: Path,
    conversion_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v80 protocol: {output_path}")
    artifact_protocol = _read_json(artifact_protocol_path)
    artifact_result = _read_json(artifact_result_path)
    artifact_audit = _read_json(artifact_audit_path)
    partition = _read_json(partition_path)
    parent_runtime = _read_json(parent_runtime_protocol_path)
    if (
        artifact_protocol.get("protocol") != ARTIFACT_PROTOCOL
        or artifact_result.get("protocol") != ARTIFACT_FREEZE_RESULT_PROTOCOL
        or artifact_result.get("status") != "PASS"
        or artifact_result.get("decision")
        != ARTIFACT_FREEZE_AUTHORIZATION_DECISION
        or artifact_result.get("integrity_gate", {}).get("passed") is not True
        or artifact_result.get("artifact_protocol_sha256")
        != _sha256(artifact_protocol_path)
        or artifact_audit.get("protocol") != ARTIFACT_AUDIT_PROTOCOL
        or artifact_audit.get("status") != "PASS"
        or artifact_audit.get("decision")
        != ARTIFACT_AUDIT_AUTHORIZATION_DECISION
        or artifact_audit.get("integrity_gate", {}).get("passed") is not True
        or artifact_audit.get("artifact_protocol_sha256")
        != _sha256(artifact_protocol_path)
        or artifact_audit.get("artifact_result_sha256")
        != _sha256(artifact_result_path)
        or partition.get("protocol") != PARTITION_PROTOCOL
        or parent_runtime.get("protocol") != PARENT_RUNTIME_PROTOCOL
    ):
        raise ValueError("v80 artifact or runtime authorization evidence changed")

    artifact_records = {}
    protocol_specs = {
        str(row["artifact_key"]): dict(row) for row in artifact_protocol["artifacts"]
    }
    for key in ARTIFACT_KEYS:
        originator, certificate, artifact = load_multihorizon_state_originator(
            artifact_root, key
        )
        result_record = dict(artifact_result["artifacts"][key])
        model_path = Path(artifact_root) / key / "model.pkl"
        certificate_path = Path(artifact_root) / key / "freeze.json"
        if (
            result_record["model"]["sha256"] != _sha256(model_path)
            or result_record["certificate"]["sha256"]
            != _sha256(certificate_path)
            or certificate["prediction_sha256"]
            != result_record["prediction_sha256"]
            or artifact["model_key"] != protocol_specs[key]["model_key"]
            or originator.config.rollout_value_horizon_sec != 60
        ):
            raise ValueError(f"v80 artifact {key!r} identity changed")
        artifact_records[key] = {
            "artifact_key": key,
            "role": str(protocol_specs[key]["role"]),
            "selection_eligible_closed_loop": bool(
                protocol_specs[key]["selection_eligible_closed_loop"]
            ),
            "model": _path_record(model_path),
            "certificate": _path_record(certificate_path),
            "prediction_sha256": str(certificate["prediction_sha256"]),
            "model_key": str(artifact["model_key"]),
            "target_name": str(artifact["target_name"]),
            "prediction_horizon_sec": int(artifact["prediction_horizon_sec"]),
        }
    if (
        tuple(artifact_records) != ARTIFACT_KEYS
        or artifact_records[PRIMARY_ARTIFACT_KEY][
            "selection_eligible_closed_loop"
        ]
        is not True
        or artifact_records[SECONDARY_ARTIFACT_KEY][
            "selection_eligible_closed_loop"
        ]
        is not False
    ):
        raise ValueError("v80 primary/diagnostic artifact roles changed")

    parent_refs = {
        "hierarchical_parent": hierarchical_parent_path,
        "offline_result": offline_result_path,
        "support_audit": support_audit_path,
        "hierarchical_freeze_audit": hierarchical_freeze_audit_path,
        "environment_protocol": environment_protocol_path,
        "environment_parent": environment_parent_path,
        "external_manifest": external_manifest_path,
        "conversion_manifest": conversion_manifest_path,
    }
    expected_hashes = {
        "hierarchical_parent": parent_runtime["parent_protocol"]["sha256"],
        "offline_result": parent_runtime["offline_authorization"]["sha256"],
        "support_audit": parent_runtime["support_runtime_equivalence"]["sha256"],
        "hierarchical_freeze_audit": parent_runtime[
            "hierarchical_freeze_audit"
        ]["sha256"],
        "environment_protocol": parent_runtime["environment"]["protocol"][
            "sha256"
        ],
        "environment_parent": parent_runtime["environment"][
            "network_parent_protocol"
        ]["sha256"],
        "external_manifest": parent_runtime["environment"]["external_manifest"][
            "sha256"
        ],
        "conversion_manifest": parent_runtime["environment"][
            "conversion_manifest"
        ]["sha256"],
    }
    if any(
        _sha256(path) != str(expected_hashes[key])
        for key, path in parent_refs.items()
    ):
        raise ValueError("v80 parent runtime evidence changed")

    development_seeds = tuple(int(value) for value in partition["redevelopment"]["seeds"])
    confirmatory_seeds = tuple(int(value) for value in partition["confirmatory"]["seeds"])
    prospective_seeds = tuple(int(value) for value in partition["prospective"]["seeds"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        len(development_seeds) != 22
        or len(confirmatory_seeds) != 32
        or len(prospective_seeds) != 8
        or set(development_seeds).intersection(confirmatory_seeds)
        or set(development_seeds).intersection(prospective_seeds)
        or set(confirmatory_seeds).intersection(prospective_seeds)
        or any(path.exists() for path in sealed_roots)
        or tuple(artifact_protocol["training"]["seeds"]) != development_seeds
    ):
        raise ValueError("v80 development partition or sealed roots changed")

    execution_candidates = []
    for artifact_key, keys in (
        (PRIMARY_ARTIFACT_KEY, PRIMARY_METHOD_KEYS),
        (SECONDARY_ARTIFACT_KEY, DIAGNOSTIC_METHOD_KEYS),
    ):
        artifact_record = artifact_records[artifact_key]
        for key, cooldown in zip(keys, (0, 2, 5), strict=True):
            execution_candidates.append(
                {
                    "key": key,
                    "artifact_key": artifact_key,
                    "candidate_role": artifact_record["role"],
                    "selection_eligible_closed_loop": bool(
                        artifact_record["selection_eligible_closed_loop"]
                    ),
                    "runtime_policy": HIERARCHICAL_RUNTIME_POLICY,
                    "coordination_mode": "direct",
                    "cooldown_intervals": cooldown,
                    "execution_trust_region": {
                        "enabled": True,
                        "min_priority": 0.0,
                        "min_total_queue": 0.0,
                        "min_total_vehicles": 1.0,
                        "max_mean_speed": None,
                        "max_simultaneous_overrides": 1,
                        "global_cooldown_intervals": 0,
                    },
                }
            )
    policies = (PHASE_POLICY, *METHOD_KEYS)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v79_artifact_audit_pre_v80_closed_loop_outcomes",
        "artifact": {
            "protocol": _path_record(artifact_protocol_path),
            "freeze_result": _path_record(artifact_result_path),
            "independent_audit": _path_record(artifact_audit_path),
            "root": str(Path(artifact_root).resolve()),
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
            "artifacts": artifact_records,
        },
        "partition": _path_record(partition_path),
        "runtime_parent": {
            "protocol": _path_record(parent_runtime_protocol_path),
            "hierarchical_parent": _path_record(hierarchical_parent_path),
            "offline_result": _path_record(offline_result_path),
            "support_audit": _path_record(support_audit_path),
            "hierarchical_freeze_audit": _path_record(
                hierarchical_freeze_audit_path
            ),
            "hierarchical_freeze_root": str(Path(hierarchical_freeze_root).resolve()),
        },
        "environment": {
            "protocol": _path_record(environment_protocol_path),
            "network_parent_protocol": _path_record(environment_parent_path),
            "external_manifest": _path_record(external_manifest_path),
            "conversion_manifest": _path_record(conversion_manifest_path),
            "conversion_root": str(Path(conversion_root)),
            "conversion_tree_sha256": str(
                parent_runtime["environment"]["conversion_tree_sha256"]
            ),
            "sumo_version": str(parent_runtime["environment"]["sumo_version"]),
            "duration_sec": 3600,
            "control_interval_sec": 10,
            "warmup_sec": 60,
            "prediction_horizon_sec": 60,
        },
        "development": {
            "classification": "development_only_never_confirmatory",
            "city_scenarios": {"jinan": ["jinan_3x4_real"]},
            "seeds": list(development_seeds),
            "sealed_confirmatory_seeds": list(confirmatory_seeds),
            "sealed_prospective_seeds": list(prospective_seeds),
            "policies": list(policies),
            "phase_policy": PHASE_POLICY,
            "method_candidates": execution_candidates,
            "primary_method_keys": list(PRIMARY_METHOD_KEYS),
            "diagnostic_method_keys": list(DIAGNOSTIC_METHOD_KEYS),
            "candidate_count": len(execution_candidates),
            "policy_count": len(policies),
            "matrix_size": len(development_seeds) * len(policies),
            "pairing_unit": "same_scenario_same_simulator_seed",
            "primary_metric": "mean_tripinfo_waiting_time",
            "selection_rule": {
                "maximum_equal_seed_mean_relative_delta": 0.0,
                "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.0,
                "maximum_worst_seed_relative_delta": 0.05,
                "minimum_improved_seed_fraction": 0.5,
                "minimum_active_seed_fraction": 0.75,
                "method_teleports_must_not_exceed_paired_phase_pressure": True,
                "method_collision_incidents_must_not_exceed_paired_phase_pressure": True,
                "fallback": "phase_pressure_if_no_primary_candidate_is_feasible",
                "selector": (
                    "primary_compact_only_lowest_mean_relative_delta_then_lower_"
                    "bootstrap_upper_then_fewer_overrides_then_longer_cooldown"
                ),
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "paired_simulator_seed",
            },
        },
        "output_roots": {
            "remote_results_root": str(remote_results_root),
            "local_results_root": str(Path(local_results_root).resolve()),
        },
        "scientific_status": {
            "proposal_origin": "target_multihorizon_state_conditioned_action_latent",
            "old_hierarchical_score_path_bypassed": True,
            "safe_phase_executor_retained": True,
            "coordination_and_cooldown_retained": True,
            "target_outcomes_available_online": False,
            "primary_model_preregistered_from_full_v77_development": True,
            "secondary_model_is_nested_stability_diagnostic": True,
            "closed_loop_outcomes_inspected_before_freeze": False,
        },
        "claim_boundary": (
            "This 154-rollout matrix is adaptive closed-loop development on the "
            "same 22 seeds used to fit both target-simulator latents. Only the "
            "three preregistered compact-h60 cooldown candidates may authorize an "
            "untouched v84 experiment. State-action-h60 is diagnostic only; a "
            "diagnostic-only success requires a new development protocol. No "
            "efficacy or generalization claim is authorized by v80."
        ),
    }
    if payload["development"]["matrix_size"] != 154:
        raise RuntimeError("v80 closed-loop matrix size changed")
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--parent-runtime-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
        artifact_audit_path=args.artifact_audit,
        artifact_root=args.artifact_root,
        partition_path=args.partition,
        parent_runtime_protocol_path=args.parent_runtime_protocol,
        hierarchical_parent_path=args.hierarchical_parent,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_manifest_path=args.conversion_manifest,
        conversion_root=args.conversion_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "matrix_size": payload["development"]["matrix_size"],
                "primary_candidates": len(PRIMARY_METHOD_KEYS),
                "diagnostic_candidates": len(DIAGNOSTIC_METHOD_KEYS),
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
