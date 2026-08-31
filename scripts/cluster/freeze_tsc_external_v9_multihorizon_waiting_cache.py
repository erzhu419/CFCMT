#!/usr/bin/env python3
"""Freeze the v76 multihorizon waiting-cost redevelopment cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (  # noqa: E402
    MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    NODES,
    PROTOCOL as PARENT_CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-multihorizon-waiting-cache-v1"
PARENT_CACHE_AUDIT_PROTOCOL = "tsc-v83r79-waiting-aligned-cache-audit-v1"
PARTITION_PROTOCOL = (
    "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
)
RESULT_REQUIREMENTS = {
    "state_conditioned": {
        "protocol": "tsc-v83r79-state-conditioned-latent-result-v1",
        "decision": "retain_phase_pressure_and_reject_state_conditioned_latent",
    },
    "distributional": {
        "protocol": "tsc-v83r79-distributional-action-state-result-v1",
        "decision": (
            "retain_phase_pressure_and_reject_distributional_action_state"
        ),
    },
    "hierarchical": {
        "protocol": "tsc-v83r79-hierarchical-action-state-residual-result-v1",
        "decision": (
            "retain_phase_pressure_and_reject_hierarchical_action_state_residual"
        ),
    },
    "closed_loop": {
        "protocol": "tsc-v83r79-waiting-aligned-latent-closed-loop-audit-v1",
        "decision": "retain_phase_pressure_and_reject_waiting_aligned_latent",
    },
}
PREFIX_HORIZONS_SEC = (10, 30, 60, 120, 450)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_rejection(
    path: Path,
    *,
    requirement: dict[str, str],
) -> dict[str, Any]:
    value = _read_json(path)
    gate = value.get("integrity_gate", value.get("gate", {}))
    if (
        value.get("protocol") != requirement["protocol"]
        or value.get("status") != "PASS"
        or value.get("decision") != requirement["decision"]
        or (gate and gate.get("passed") is not True)
    ):
        raise ValueError(f"rejection evidence changed: {path}")
    return value


def build_protocol(
    *,
    parent_cache_protocol_path: Path,
    parent_cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    state_result_path: Path,
    distributional_result_path: Path,
    hierarchical_result_path: Path,
    closed_loop_audit_path: Path,
    remote_cache_root: Path,
    remote_task_root: Path,
) -> dict[str, Any]:
    parent = _read_json(parent_cache_protocol_path)
    parent_audit = _read_json(parent_cache_audit_path)
    partition = _read_json(partition_path)
    if (
        parent.get("protocol") != PARENT_CACHE_PROTOCOL
        or parent_audit.get("protocol") != PARENT_CACHE_AUDIT_PROTOCOL
        or parent_audit.get("status") != "PASS"
        or parent_audit.get("gate", {}).get("passed") is not True
        or partition.get("protocol") != PARTITION_PROTOCOL
        or parent.get("frozen_inputs", {}).get("manifest_sha256")
        != _sha256(manifest_path)
    ):
        raise ValueError("multihorizon parent evidence changed")
    evidence_paths = {
        "state_conditioned": Path(state_result_path),
        "distributional": Path(distributional_result_path),
        "hierarchical": Path(hierarchical_result_path),
        "closed_loop": Path(closed_loop_audit_path),
    }
    evidence = {
        name: _validate_rejection(
            path,
            requirement=RESULT_REQUIREMENTS[name],
        )
        for name, path in evidence_paths.items()
    }
    seeds = tuple(int(value) for value in partition["redevelopment"]["seeds"])
    collection = dict(parent["collection"])
    if (
        len(seeds) != 22
        or len(set(seeds)) != 22
        or tuple(int(value) for value in collection["seeds"]) != seeds
        or int(collection["control_interval_sec"]) != 10
        or int(collection["counterfactual_horizon_intervals"]) != 45
        or int(collection["collection_shards_per_seed"]) != 32
        or str(collection["counterfactual_cost_mode"]) != "halted_queue"
        or str(collection["behavior_policy"]) != "phase_pressure"
    ):
        raise ValueError("multihorizon redevelopment collection contract changed")
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if any(path.exists() for path in sealed_roots):
        raise ValueError("confirmatory or prospective root is no longer sealed")
    assignments = {
        str(node): [int(value) for value in values]
        for node, values in collection["node_assignments"].items()
    }
    if tuple(assignments) != NODES or sorted(
        value for values in assignments.values() for value in values
    ) != sorted(seeds):
        raise ValueError("multihorizon node assignment changed")
    shards = int(collection["collection_shards_per_seed"])
    workers_by_node = {
        node: min(len(values) * shards, 128)
        for node, values in assignments.items()
    }
    frozen_inputs = {
        "parent_cache_protocol": parent_cache_protocol_path,
        "parent_cache_audit": parent_cache_audit_path,
        "partition": partition_path,
        "manifest": manifest_path,
        **{
            f"{name}_result": path for name, path in evidence_paths.items()
        },
    }
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "adaptive_development_multihorizon_estimand_recollection",
        "frozen_inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in frozen_inputs.items()
        },
        "environment": dict(parent["environment"]),
        "collection": {
            **{
                key: value
                for key, value in collection.items()
                if key not in {"workers_by_node", "expected_cache_file_count"}
            },
            "cache_version": MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
            "rollout_prefix_horizons_sec": list(PREFIX_HORIZONS_SEC),
            "rollout_prefix_aggregation": (
                "arithmetic_mean_of_one_second_global_halted_queue_per_"
                "controlled_lane_from_branch_start"
            ),
            "expected_cache_file_count": len(seeds) * shards,
            "workers_by_node": workers_by_node,
        },
        "output_roots": {
            "remote_cache_root": str(Path(remote_cache_root)),
            "remote_task_root": str(Path(remote_task_root)),
        },
        "estimand_contract": {
            "action": "candidate_phase_for_first_10_seconds",
            "continuation_policy": "phase_pressure_after_first_control_interval",
            "primary_deployment_interval_sec": 10,
            "saved_horizons_sec": list(PREFIX_HORIZONS_SEC),
            "full_horizon_equivalence": (
                "prefix_mean_cost_450s_equals_legacy_interval_cost"
            ),
            "selection": (
                "nested_leave_one_seed_out_horizon_selection_without_heldout_"
                "seed_labels"
            ),
        },
        "post_collection_gate": {
            "all_cache_identities_valid": True,
            "all_seed_shard_sets_complete": True,
            "behavior_trace_consistent_within_seed": True,
            "full_tls_coverage_within_seed": True,
            "replay_and_safety_pass": True,
            "finite_nonnegative_prefix_targets": True,
            "prefix_targets_are_monotone_in_sample_count_not_value": True,
            "full_horizon_equivalence_pass": True,
        },
        "sealed_evidence": {
            "confirmatory_root": str(sealed_roots[0]),
            "prospective_root": str(sealed_roots[1]),
            "confirmatory_seed_count": int(partition["confirmatory"]["seed_count"]),
            "prospective_seed_count": int(partition["prospective"]["seed_count"]),
        },
        "adaptive_rationale": {
            name: {
                "status": value["status"],
                "decision": value["decision"],
            }
            for name, value in evidence.items()
        },
        "claim_boundary": (
            "All multihorizon labels use the disclosed 22-seed redevelopment "
            "partition. Horizon and model selection are adaptive development; "
            "v84 confirmatory and v85 prospective evidence remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-protocol", type=Path, required=True)
    parser.add_argument("--parent-cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-result", type=Path, required=True)
    parser.add_argument("--distributional-result", type=Path, required=True)
    parser.add_argument("--hierarchical-result", type=Path, required=True)
    parser.add_argument("--closed-loop-audit", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-task-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite multihorizon protocol: {args.out}")
    payload = build_protocol(
        parent_cache_protocol_path=args.parent_cache_protocol,
        parent_cache_audit_path=args.parent_cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        state_result_path=args.state_result,
        distributional_result_path=args.distributional_result,
        hierarchical_result_path=args.hierarchical_result,
        closed_loop_audit_path=args.closed_loop_audit,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "cache_files": payload["collection"]["expected_cache_file_count"],
                "horizons_sec": payload["collection"]["rollout_prefix_horizons_sec"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
