#!/usr/bin/env python3
"""Freeze the v83 uncertainty-aware h60/h120 closed-loop development matrix."""

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
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_audit import (  # noqa: E402
    AUTHORIZATION_DECISION as ARTIFACT_AUDIT_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as ARTIFACT_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as ARTIFACT_FREEZE_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as ARTIFACT_FREEZE_RESULT_PROTOCOL,
    load_uncertainty_horizon_originator,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_multihorizon_estimand_aligned_closed_loop import (  # noqa: E402
    AUDIT_PROTOCOL as V81_AUDIT_PROTOCOL,
    REJECTION_DECISION as V81_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_estimand_aligned_closed_loop_development import (  # noqa: E402
    PROTOCOL as V81_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (  # noqa: E402
    ADAPTATION_CLASSIFICATION,
    ARTIFACT_KEYS,
    MINIMUM_CONTEXT_TRUST,
    PROTOCOL as ARTIFACT_PROTOCOL,
    RISK_MULTIPLIER,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-uncertainty-horizon-closed-loop-v1"
PHASE_POLICY = "phase_pressure"
CANDIDATE_SPECS = (
    ("uh_compact_h60", "compact_h60_r025_t010", 60, 5),
    ("uh_state_action_h60", "state_action_h60_r025_t010", 60, 5),
    ("uh_compact_h120", "compact_h120_r025_t010", 120, 11),
    ("uh_state_action_h120", "state_action_h120_r025_t010", 120, 11),
)
METHOD_KEYS = tuple(row[0] for row in CANDIDATE_SPECS)
PRIMARY_METHOD_KEYS = METHOD_KEYS
DIAGNOSTIC_METHOD_KEYS: tuple[str, ...] = ()
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
    v81_protocol_path: Path,
    v81_audit_path: Path,
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
        raise FileExistsError(f"refusing to overwrite v83 protocol: {output_path}")
    v81_protocol = _read_json(v81_protocol_path)
    v81_audit = _read_json(v81_audit_path)
    artifact_protocol = _read_json(artifact_protocol_path)
    artifact_result = _read_json(artifact_result_path)
    artifact_audit = _read_json(artifact_audit_path)
    partition = _read_json(partition_path)
    parent_runtime = _read_json(parent_runtime_protocol_path)
    if not (
        v81_protocol.get("protocol") == V81_PROTOCOL
        and v81_audit.get("protocol") == V81_AUDIT_PROTOCOL
        and v81_audit.get("status") == "PASS"
        and v81_audit.get("decision") == V81_REJECTION_DECISION
        and v81_audit.get("integrity_gate", {}).get("passed") is True
        and v81_audit.get("protocol_sha256") == _sha256(v81_protocol_path)
        and artifact_protocol.get("protocol") == ARTIFACT_PROTOCOL
        and artifact_result.get("protocol") == ARTIFACT_FREEZE_RESULT_PROTOCOL
        and artifact_result.get("status") == "PASS"
        and artifact_result.get("decision")
        == ARTIFACT_FREEZE_AUTHORIZATION_DECISION
        and artifact_result.get("integrity_gate", {}).get("passed") is True
        and artifact_result.get("artifact_protocol_sha256")
        == _sha256(artifact_protocol_path)
        and artifact_audit.get("protocol") == ARTIFACT_AUDIT_PROTOCOL
        and artifact_audit.get("status") == "PASS"
        and artifact_audit.get("decision")
        == ARTIFACT_AUDIT_AUTHORIZATION_DECISION
        and artifact_audit.get("integrity_gate", {}).get("passed") is True
        and artifact_audit.get("artifact_protocol_sha256")
        == _sha256(artifact_protocol_path)
        and artifact_audit.get("artifact_result_sha256")
        == _sha256(artifact_result_path)
        and partition.get("protocol") == PARTITION_PROTOCOL
        and parent_runtime.get("protocol") == PARENT_RUNTIME_PROTOCOL
    ):
        raise ValueError("v83 parent, artifact, or runtime authorization changed")

    artifact_records: dict[str, dict[str, Any]] = {}
    artifact_specs = {
        str(row["artifact_key"]): dict(row) for row in artifact_protocol["artifacts"]
    }
    for key in ARTIFACT_KEYS:
        originator, certificate, artifact = load_uncertainty_horizon_originator(
            artifact_root, key
        )
        result_record = dict(artifact_result["artifacts"][key])
        audit_record = dict(artifact_audit["artifacts"][key])
        spec = artifact_specs[key]
        model_path = Path(artifact_root) / key / "model.pkl"
        certificate_path = Path(artifact_root) / key / "freeze.json"
        horizon = int(spec["prediction_horizon_sec"])
        if not (
            result_record["model"]["sha256"] == _sha256(model_path)
            and result_record["certificate"]["sha256"]
            == _sha256(certificate_path)
            and certificate["prediction_sha256"]
            == result_record["prediction_sha256"]
            and audit_record.get("exact") is True
            and audit_record.get("prediction_sha256")
            == certificate["prediction_sha256"]
            and artifact["model_key"] == spec["model_key"]
            and int(artifact["prediction_horizon_sec"]) == horizon
            and int(originator.config.rollout_value_horizon_sec) == horizon
            and float(originator.config.risk_multiplier) == RISK_MULTIPLIER
            and float(originator.config.minimum_context_trust)
            == MINIMUM_CONTEXT_TRUST
            and bool(spec["selection_eligible_closed_loop"])
        ):
            raise ValueError(f"v83 artifact {key!r} identity changed")
        artifact_records[key] = {
            "artifact_key": key,
            "role": str(spec["role"]),
            "selection_eligible_closed_loop": True,
            "model": _path_record(model_path),
            "certificate": _path_record(certificate_path),
            "prediction_sha256": str(certificate["prediction_sha256"]),
            "model_key": str(artifact["model_key"]),
            "target_name": str(artifact["target_name"]),
            "prediction_horizon_sec": horizon,
            "risk_multiplier": float(originator.config.risk_multiplier),
            "minimum_context_trust": float(
                originator.config.minimum_context_trust
            ),
            "fit_mode": str(artifact["fit_mode"]),
        }
    if tuple(artifact_records) != ARTIFACT_KEYS:
        raise ValueError("v83 artifact key order changed")

    parent_refs = {
        "parent_runtime_protocol": parent_runtime_protocol_path,
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
        "parent_runtime_protocol": v81_protocol["runtime_parent"]["protocol"][
            "sha256"
        ],
        "hierarchical_parent": v81_protocol["runtime_parent"][
            "hierarchical_parent"
        ]["sha256"],
        "offline_result": v81_protocol["runtime_parent"]["offline_result"][
            "sha256"
        ],
        "support_audit": v81_protocol["runtime_parent"]["support_audit"][
            "sha256"
        ],
        "hierarchical_freeze_audit": v81_protocol["runtime_parent"][
            "hierarchical_freeze_audit"
        ]["sha256"],
        "environment_protocol": v81_protocol["environment"]["protocol"][
            "sha256"
        ],
        "environment_parent": v81_protocol["environment"][
            "network_parent_protocol"
        ]["sha256"],
        "external_manifest": v81_protocol["environment"]["external_manifest"][
            "sha256"
        ],
        "conversion_manifest": v81_protocol["environment"][
            "conversion_manifest"
        ]["sha256"],
    }
    if any(
        _sha256(path) != str(expected_hashes[key])
        for key, path in parent_refs.items()
    ):
        raise ValueError("v83 inherited runtime evidence changed")
    if (
        _sha256(partition_path) != v81_protocol["partition"]["sha256"]
        or str(Path(hierarchical_freeze_root).resolve())
        != str(
            Path(
                v81_protocol["runtime_parent"]["hierarchical_freeze_root"]
            ).resolve()
        )
    ):
        raise ValueError("v83 inherited partition or freeze root changed")

    development_seeds = tuple(
        int(value) for value in partition["redevelopment"]["seeds"]
    )
    confirmatory_seeds = tuple(
        int(value) for value in partition["confirmatory"]["seeds"]
    )
    prospective_seeds = tuple(
        int(value) for value in partition["prospective"]["seeds"]
    )
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if not (
        len(development_seeds) == 22
        and len(confirmatory_seeds) == 32
        and len(prospective_seeds) == 8
        and not set(development_seeds).intersection(confirmatory_seeds)
        and not set(development_seeds).intersection(prospective_seeds)
        and not set(confirmatory_seeds).intersection(prospective_seeds)
        and not any(path.exists() for path in sealed_roots)
        and tuple(artifact_protocol["training"]["seeds"]) == development_seeds
        and tuple(v81_protocol["development"]["seeds"]) == development_seeds
    ):
        raise ValueError("v83 development partition or sealed roots changed")

    execution_candidates: list[dict[str, Any]] = []
    for key, artifact_key, horizon, global_cooldown in CANDIDATE_SPECS:
        artifact_record = artifact_records[artifact_key]
        if int(artifact_record["prediction_horizon_sec"]) != int(horizon):
            raise ValueError(f"v83 candidate {key!r} horizon changed")
        execution_candidates.append(
            {
                "key": key,
                "artifact_key": artifact_key,
                "candidate_role": artifact_record["role"],
                "selection_eligible_closed_loop": True,
                "prediction_horizon_sec": int(horizon),
                "runtime_policy": HIERARCHICAL_RUNTIME_POLICY,
                "coordination_mode": "direct",
                "cooldown_intervals": 0,
                "execution_trust_region": {
                    "enabled": True,
                    "min_priority": 0.0,
                    "min_total_queue": 0.0,
                    "min_total_vehicles": 1.0,
                    "max_mean_speed": None,
                    "max_simultaneous_overrides": 1,
                    "global_cooldown_intervals": int(global_cooldown),
                },
            }
        )

    environment = dict(v81_protocol["environment"])
    environment.pop("prediction_horizon_sec", None)
    environment["prediction_horizons_sec"] = [60, 120]
    policies = (PHASE_POLICY, *METHOD_KEYS)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v81_rejection_post_v82_artifact_audit_pre_v83_outcomes",
        "development_parent": {
            "v81_protocol": _path_record(v81_protocol_path),
            "v81_independent_audit": _path_record(v81_audit_path),
            "v81_decision": V81_REJECTION_DECISION,
        },
        "artifact": {
            "protocol": _path_record(artifact_protocol_path),
            "freeze_result": _path_record(artifact_result_path),
            "independent_audit": _path_record(artifact_audit_path),
            "root": str(Path(artifact_root).resolve()),
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
            "risk_multiplier": RISK_MULTIPLIER,
            "minimum_context_trust": MINIMUM_CONTEXT_TRUST,
            "artifacts": artifact_records,
        },
        "partition": _path_record(partition_path),
        "runtime_parent": dict(v81_protocol["runtime_parent"]),
        "environment": environment,
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
            "diagnostic_method_keys": [],
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
                    "lowest_mean_relative_delta_then_lower_bootstrap_upper_"
                    "then_fewer_overrides_then_shorter_horizon_then_method_key"
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
            "proposal_origin": "target_uncertainty_aware_multihorizon_latent",
            "old_hierarchical_score_path_bypassed": True,
            "safe_phase_executor_retained": True,
            "target_outcomes_available_online": False,
            "all_four_candidates_selection_eligible": True,
            "v81_closed_loop_outcomes_inspected_before_freeze": True,
            "v83_closed_loop_outcomes_inspected_before_freeze": False,
            "risk_and_trust_fixed_from_v77_oof_screen": True,
            "horizons_fixed_from_v77_oof_screen": True,
            "global_cooldown_exactly_matches_each_estimand": True,
        },
        "claim_boundary": (
            "This 110-rollout matrix is adaptive successor development after the "
            "v81 rejection. It tests four v82-audited uncertainty-aware h60/h120 "
            "originators under exact horizon-aligned execution. All outcomes use "
            "the disclosed 22-seed redevelopment partition; v84 confirmatory and "
            "v85 prospective roots remain sealed. No efficacy or generalization "
            "claim is authorized by v83 alone."
        ),
    }
    if payload["development"]["matrix_size"] != 110:
        raise RuntimeError("v83 closed-loop matrix size changed")
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81-protocol", type=Path, required=True)
    parser.add_argument("--v81-audit", type=Path, required=True)
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
        v81_protocol_path=args.v81_protocol,
        v81_audit_path=args.v81_audit,
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
                "candidate_count": len(METHOD_KEYS),
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
