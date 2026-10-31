#!/usr/bin/env python3
"""Freeze the 22-seed waiting-aligned latent closed-loop development matrix."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_freeze import (  # noqa: E402
    RESULT_PROTOCOL as ARTIFACT_RESULT_PROTOCOL,
    load_waiting_aligned_originator,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_artifact import (  # noqa: E402
    ADAPTATION_CLASSIFICATION,
    PROTOCOL as ARTIFACT_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-latent-closed-loop-v1"
PARTITION_PROTOCOL = "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
PHASE_POLICY = "phase_pressure"
METHOD_KEYS = (
    "waiting_latent_cd0",
    "waiting_latent_cd2",
    "waiting_latent_cd5",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def freeze_protocol(
    *,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
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
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite latent closed-loop protocol: {output_path}")
    artifact_protocol = _read_json(artifact_protocol_path)
    artifact_result = _read_json(artifact_result_path)
    partition = _read_json(partition_path)
    parent_runtime = _read_json(parent_runtime_protocol_path)
    if (
        artifact_protocol.get("protocol") != ARTIFACT_PROTOCOL
        or artifact_result.get("protocol") != ARTIFACT_RESULT_PROTOCOL
        or artifact_result.get("status") != "PASS"
        or artifact_result.get("decision")
        != "authorize_waiting_aligned_latent_closed_loop_redevelopment"
        or artifact_result.get("integrity_gate", {}).get("passed") is not True
        or artifact_result.get("artifact_protocol_sha256")
        != _sha256(artifact_protocol_path)
        or partition.get("protocol") != PARTITION_PROTOCOL
        or parent_runtime.get("protocol") != PARENT_RUNTIME_PROTOCOL
    ):
        raise ValueError("latent closed-loop authorization evidence changed")
    model_path = Path(artifact_root) / "model.pkl"
    certificate_path = Path(artifact_root) / "freeze.json"
    _, certificate = load_waiting_aligned_originator(artifact_root)
    if (
        artifact_result["artifact"]["model"]["sha256"] != _sha256(model_path)
        or artifact_result["artifact"]["certificate"]["sha256"]
        != _sha256(certificate_path)
        or certificate.get("adaptation_classification")
        != ADAPTATION_CLASSIFICATION
    ):
        raise ValueError("latent artifact identity changed before closed-loop freeze")

    parent_refs = {
        "hierarchical_parent": (hierarchical_parent_path, "parent_protocol"),
        "offline_result": (offline_result_path, "offline_authorization"),
        "support_audit": (support_audit_path, "support_runtime_equivalence"),
        "hierarchical_freeze_audit": (
            hierarchical_freeze_audit_path,
            "hierarchical_freeze_audit",
        ),
        "environment_protocol": (environment_protocol_path, "environment.protocol"),
        "environment_parent": (
            environment_parent_path,
            "environment.network_parent_protocol",
        ),
        "external_manifest": (
            external_manifest_path,
            "environment.external_manifest",
        ),
        "conversion_manifest": (
            conversion_manifest_path,
            "environment.conversion_manifest",
        ),
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
        for key, (path, _) in parent_refs.items()
    ):
        raise ValueError("latent closed-loop parent runtime evidence changed")

    development_seeds = tuple(int(value) for value in partition["redevelopment"]["seeds"])
    confirmatory_seeds = tuple(int(value) for value in partition["confirmatory"]["seeds"])
    prospective_seeds = tuple(int(value) for value in partition["prospective"]["seeds"])
    if (
        len(development_seeds) != 22
        or len(confirmatory_seeds) != 32
        or len(prospective_seeds) != 8
        or set(development_seeds).intersection(confirmatory_seeds)
        or set(development_seeds).intersection(prospective_seeds)
        or set(confirmatory_seeds).intersection(prospective_seeds)
    ):
        raise ValueError("latent closed-loop seed partition changed")
    artifact_training_seeds = tuple(
        int(value) for value in artifact_protocol["training"]["seeds"]
    )
    if artifact_training_seeds != development_seeds:
        raise ValueError("latent artifact and closed-loop development seeds differ")

    execution_candidates = []
    for key, cooldown in zip(METHOD_KEYS, (0, 2, 5), strict=True):
        execution_candidates.append(
            {
                "key": key,
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
        "stage": "post_v70_oof_post_v71_artifact_pre_closed_loop_outcome_development",
        "artifact": {
            "protocol": _path_record(artifact_protocol_path),
            "freeze_result": _path_record(artifact_result_path),
            "model": _path_record(model_path),
            "certificate": _path_record(certificate_path),
            "root": str(Path(artifact_root).resolve()),
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
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
            "prediction_horizon_sec": 450,
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
                "fallback": "phase_pressure_if_no_candidate_is_feasible",
                "selector": (
                    "lowest_mean_relative_delta_then_lower_bootstrap_upper_then_"
                    "fewer_overrides_then_longer_cooldown"
                ),
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "paired_simulator_seed",
            },
        },
        "scientific_status": {
            "proposal_origin": "target_spatiotemporal_action_latent",
            "old_hierarchical_score_path_bypassed": True,
            "safe_phase_executor_retained": True,
            "coordination_and_cooldown_retained": True,
            "target_outcomes_available_online": False,
            "closed_loop_outcomes_inspected_before_freeze": False,
        },
        "claim_boundary": (
            "This is adaptive 22-seed closed-loop development on the same seed "
            "partition used to fit the target latent. It can select a runtime "
            "candidate but cannot support an efficacy or generalization claim."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
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
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "matrix_size": payload["development"]["matrix_size"],
                "policies": payload["development"]["policies"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
