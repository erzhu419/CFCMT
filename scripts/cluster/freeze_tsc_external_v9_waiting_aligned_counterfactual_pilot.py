#!/usr/bin/env python3
"""Freeze the development-only waiting-aligned counterfactual pilot."""

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
)


PARTITION_PROTOCOL = (
    "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
)
SOURCE_PROTOCOL = (
    "tsc-v69r65-external-v9-estimand-aligned-450s-veto-cache-v1"
)
SOURCE_AUDIT_PROTOCOL = "tsc-v69r65-external-v9-450s-veto-cache-audit-v1"
SOURCE_AUDIT_DECISION = (
    "authorize_demand_conditioned_450s_veto_oof_training_only"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def build_protocol(
    *,
    partition_path: Path,
    source_protocol_path: Path,
    source_cache_audit_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    legacy_cache_root: Path,
) -> dict[str, Any]:
    partition = _read_json(partition_path)
    source = _read_json(source_protocol_path)
    audit = _read_json(source_cache_audit_path)
    if (
        partition.get("protocol") != PARTITION_PROTOCOL
        or source.get("protocol") != SOURCE_PROTOCOL
        or audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision") != SOURCE_AUDIT_DECISION
    ):
        raise ValueError("waiting-aligned pilot parent evidence changed")
    seed = 80314
    redevelopment = {int(value) for value in partition["redevelopment"]["seeds"]}
    confirmation = {int(value) for value in partition["confirmatory"]["seeds"]}
    prospective = {int(value) for value in partition["prospective"]["seeds"]}
    if seed not in redevelopment or seed in confirmation or seed in prospective:
        raise ValueError("waiting-aligned pilot seed partition changed")
    collection = dict(source["long_horizon_collection"])
    if (
        int(collection["counterfactual_horizon_intervals"]) != 45
        or int(collection["collection_shards_per_scenario_seed"]) != 32
        or str(collection["behavior_policy"]) != "phase_pressure"
    ):
        raise ValueError("waiting-aligned pilot source collection changed")
    return {
        "protocol": PILOT_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v81_failed_validation_development_only_estimand_repair",
        "frozen_inputs": {
            "partition_path": str(Path(partition_path).resolve()),
            "partition_sha256": _sha256(partition_path),
            "source_protocol_path": str(Path(source_protocol_path).resolve()),
            "source_protocol_sha256": _sha256(source_protocol_path),
            "source_cache_audit_path": str(Path(source_cache_audit_path).resolve()),
            "source_cache_audit_sha256": _sha256(source_cache_audit_path),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "environment": {
            "sumo_version": str(source["environment"]["sumo_version"]),
            "conversion_root": str(Path(conversion_root)),
            "legacy_cache_root": str(Path(legacy_cache_root)),
        },
        "pilot": {
            "scenario": "jinan_3x4_real",
            "seed": seed,
            "seed_classification": partition["redevelopment"]["classification"],
            "duration_sec": float(collection["duration_sec"]),
            "control_interval_sec": int(collection["control_interval_sec"]),
            "warmup_sec": float(collection["warmup_sec"]),
            "max_focal_tls": int(collection["max_focal_tls"]),
            "counterfactual_horizon_intervals": int(
                collection["counterfactual_horizon_intervals"]
            ),
            "collection_shard_index": 0,
            "collection_shard_count": int(
                collection["collection_shards_per_scenario_seed"]
            ),
            "behavior_policy": str(collection["behavior_policy"]),
            "legacy_cost_mode": "system_vehicle_load",
            "waiting_cost_mode": "halted_queue",
        },
        "implementation_gate": {
            "exact_action_key_set": True,
            "identical_behavior_trace": True,
            "trajectory_invariance_tolerance": 1e-8,
            "waiting_label_finite_and_nonnegative": True,
            "minimum_informative_waiting_groups": 1,
            "replay_and_safety": True,
        },
        "use_boundary": {
            "waiting_target_used_for": "proposal_conditional_veto_only",
            "mechanism_world_model_refit": False,
            "reason": (
                "The analytic interval-cost prior remains a vehicle-load prior; "
                "the halted-queue label is not used to refit that mechanism."
            ),
        },
        "claim_boundary": (
            "This is a disclosed development-only implementation pilot. It cannot "
            "support efficacy or confirmatory claims."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite pilot protocol: {args.out}")
    payload = build_protocol(
        partition_path=args.partition,
        source_protocol_path=args.source_protocol,
        source_cache_audit_path=args.source_cache_audit,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        legacy_cache_root=args.legacy_cache_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "seed": payload["pilot"]["seed"],
                "scenario": payload["pilot"]["scenario"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
