#!/usr/bin/env python3
"""Freeze the 22-seed waiting-aligned redevelopment cache matrix."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_pilot import (  # noqa: E402
    PILOT_PROTOCOL,
    RESULT_PROTOCOL as PILOT_RESULT_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-redevelopment-cache-v1"
PARTITION_PROTOCOL = (
    "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
)
SOURCE_PROTOCOL = (
    "tsc-v69r65-external-v9-estimand-aligned-450s-veto-cache-v1"
)
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _node_assignments(seeds: Sequence[int]) -> dict[str, list[int]]:
    values = [int(seed) for seed in seeds]
    base, extra = divmod(len(values), len(NODES))
    assignments: dict[str, list[int]] = {}
    offset = 0
    for index, node in enumerate(NODES):
        count = base + int(index < extra)
        assignments[node] = values[offset : offset + count]
        offset += count
    if offset != len(values) or any(not value for value in assignments.values()):
        raise ValueError("waiting-aligned seed assignment is incomplete")
    return assignments


def build_protocol(
    *,
    partition_path: Path,
    pilot_protocol_path: Path,
    pilot_result_path: Path,
    source_protocol_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    remote_cache_root: Path,
    remote_task_root: Path,
    confirmatory_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    partition = _read_json(partition_path)
    pilot_protocol = _read_json(pilot_protocol_path)
    pilot = _read_json(pilot_result_path)
    source = _read_json(source_protocol_path)
    if (
        partition.get("protocol") != PARTITION_PROTOCOL
        or pilot_protocol.get("protocol") != PILOT_PROTOCOL
        or pilot.get("protocol") != PILOT_RESULT_PROTOCOL
        or source.get("protocol") != SOURCE_PROTOCOL
        or pilot.get("status") != "PASS"
        or pilot.get("decision")
        != "authorize_waiting_aligned_redevelopment_collection"
        or pilot.get("gates", {}).get("passed") is not True
        or pilot.get("frozen_protocol", {}).get("sha256")
        != _sha256(pilot_protocol_path)
    ):
        raise ValueError("waiting-aligned redevelopment parent evidence changed")
    if Path(confirmatory_root).exists() or Path(prospective_root).exists():
        raise ValueError("confirmatory or prospective root was opened before redevelopment")
    seeds = [int(value) for value in partition["redevelopment"]["seeds"]]
    if len(seeds) != 22 or len(seeds) != len(set(seeds)):
        raise ValueError("waiting-aligned redevelopment seed partition changed")
    collection = dict(source["long_horizon_collection"])
    if (
        int(collection["collection_shards_per_scenario_seed"]) != 32
        or int(collection["counterfactual_horizon_intervals"]) != 45
        or float(collection["duration_sec"]) != 3600.0
        or str(collection["behavior_policy"]) != "phase_pressure"
    ):
        raise ValueError("waiting-aligned source collection contract changed")
    assignments = _node_assignments(seeds)
    shards = int(collection["collection_shards_per_scenario_seed"])
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_waiting_aligned_counterfactual_collection",
        "frozen_inputs": {
            "partition_path": str(Path(partition_path).resolve()),
            "partition_sha256": _sha256(partition_path),
            "pilot_protocol_path": str(Path(pilot_protocol_path).resolve()),
            "pilot_protocol_sha256": _sha256(pilot_protocol_path),
            "pilot_result_path": str(Path(pilot_result_path).resolve()),
            "pilot_result_sha256": _sha256(pilot_result_path),
            "source_protocol_path": str(Path(source_protocol_path).resolve()),
            "source_protocol_sha256": _sha256(source_protocol_path),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "environment": {
            "sumo_version": str(source["environment"]["sumo_version"]),
            "conversion_root": str(Path(conversion_root)),
            "conversion_manifest_sha256": source["environment"][
                "conversion_manifest"
            ]["sha256"],
            "conversion_tree_sha256": source["environment"][
                "conversion_tree_sha256"
            ],
        },
        "collection": {
            "scenario": "jinan_3x4_real",
            "seeds": seeds,
            "seed_count": len(seeds),
            "seed_classification": partition["redevelopment"]["classification"],
            "duration_sec": float(collection["duration_sec"]),
            "control_interval_sec": int(collection["control_interval_sec"]),
            "warmup_sec": float(collection["warmup_sec"]),
            "max_focal_tls": int(collection["max_focal_tls"]),
            "counterfactual_horizon_intervals": int(
                collection["counterfactual_horizon_intervals"]
            ),
            "collection_shards_per_seed": shards,
            "behavior_policy": str(collection["behavior_policy"]),
            "counterfactual_cost_mode": "halted_queue",
            "expected_cache_file_count": len(seeds) * shards,
            "node_assignments": assignments,
            "workers_by_node": {
                node: len(node_seeds) * shards
                for node, node_seeds in assignments.items()
            },
        },
        "output_roots": {
            "remote_cache_root": str(Path(remote_cache_root)),
            "remote_task_root": str(Path(remote_task_root)),
        },
        "training_boundary": {
            "use": "proposal_conditional_veto_target_only",
            "target": "450_second_global_halted_queue_integral_per_controlled_lane",
            "mechanism_world_model_refit": False,
            "confirmatory_seeds_used": False,
            "prospective_seeds_used": False,
        },
        "post_collection_gate": {
            "all_cache_identities_valid": True,
            "all_seed_shard_sets_complete": True,
            "behavior_trace_consistent_within_seed": True,
            "full_tls_coverage_within_seed": True,
            "replay_and_safety_pass": True,
            "finite_nonnegative_waiting_targets": True,
        },
        "claim_boundary": (
            "All 22 seeds are disclosed redevelopment evidence and can never be "
            "reclassified as confirmatory evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--pilot-protocol", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-task-root", type=Path, required=True)
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite redevelopment protocol: {args.out}")
    payload = build_protocol(
        partition_path=args.partition,
        pilot_protocol_path=args.pilot_protocol,
        pilot_result_path=args.pilot_result,
        source_protocol_path=args.source_protocol,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
        confirmatory_root=args.confirmatory_root,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "cache_files": payload["collection"]["expected_cache_file_count"],
                "workers_by_node": payload["collection"]["workers_by_node"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
