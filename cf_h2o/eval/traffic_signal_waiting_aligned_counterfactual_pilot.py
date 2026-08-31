"""Run one matched legacy/waiting-aligned SUMO counterfactual shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _collect_worker,
    _counterfactual_cache_identity,
    _counterfactual_cache_path,
    _validate_counterfactual_cache_dataset,
)
from cf_h2o.sumo_runtime import libsumo_version
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PILOT_PROTOCOL = (
    "tsc-v83r79-external-v9-waiting-aligned-counterfactual-pilot-v1"
)
RESULT_PROTOCOL = (
    "tsc-v83r79-external-v9-waiting-aligned-counterfactual-pilot-result-v1"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _row_keys(dataset: MechanismDataset) -> list[tuple[str, str]]:
    groups = [str(value) for value in dataset.metadata.get("action_group_ids", ())]
    states = [str(value) for value in dataset.metadata.get("candidate_states", ())]
    if len(groups) != dataset.size or len(states) != dataset.size:
        raise ValueError("counterfactual row metadata is not aligned")
    keys = list(zip(groups, states))
    if len(keys) != len(set(keys)):
        raise ValueError("counterfactual action keys are not unique")
    return keys


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_counterfactual_costs(
    legacy: MechanismDataset,
    waiting: MechanismDataset,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compare two estimands on exactly matched action trajectories."""

    legacy_keys = _row_keys(legacy)
    waiting_keys = _row_keys(waiting)
    exact_key_set = set(legacy_keys) == set(waiting_keys)
    if not exact_key_set:
        return {
            "exact_action_key_set": False,
            "legacy_rows": legacy.size,
            "waiting_rows": waiting.size,
            "passed": False,
        }
    waiting_index = {key: index for index, key in enumerate(waiting_keys)}
    order = np.asarray([waiting_index[key] for key in legacy_keys], dtype=int)
    legacy_cost = np.asarray(legacy.targets["interval_cost"], dtype=float)
    waiting_cost = np.asarray(waiting.targets["interval_cost"], dtype=float)[order]
    groups = np.asarray([key[0] for key in legacy_keys], dtype=str)

    invariant_targets = sorted(
        set(legacy.targets)
        & set(waiting.targets)
        - {"interval_cost", "terminal_system_load"}
    )
    target_differences = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(legacy.targets[name], dtype=float)
                    - np.asarray(waiting.targets[name], dtype=float)[order]
                )
            )
        )
        for name in invariant_targets
    }
    feature_difference = float(
        np.max(
            np.abs(
                np.asarray(legacy.features, dtype=float)
                - np.asarray(waiting.features, dtype=float)[order]
            )
        )
    )
    context_difference = float(
        np.max(
            np.abs(
                np.asarray(legacy.context, dtype=float)
                - np.asarray(waiting.context, dtype=float)[order]
            )
        )
    )

    legacy_centered = np.zeros_like(legacy_cost)
    waiting_centered = np.zeros_like(waiting_cost)
    informative_groups = 0
    best_action_agreements = 0
    pairwise_agreements = 0
    pairwise_comparisons = 0
    group_rows = []
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        old = legacy_cost[rows]
        new = waiting_cost[rows]
        legacy_centered[rows] = old - np.mean(old)
        waiting_centered[rows] = new - np.mean(new)
        old_range = float(np.ptp(old))
        new_range = float(np.ptp(new))
        informative_groups += int(new_range > tolerance)
        best_action_agreements += int(int(np.argmin(old)) == int(np.argmin(new)))
        for left in range(rows.size):
            for right in range(left + 1, rows.size):
                old_delta = float(old[left] - old[right])
                new_delta = float(new[left] - new[right])
                if abs(old_delta) <= tolerance or abs(new_delta) <= tolerance:
                    continue
                pairwise_comparisons += 1
                pairwise_agreements += int(np.sign(old_delta) == np.sign(new_delta))
        group_rows.append(
            {
                "group_id": str(group),
                "actions": int(rows.size),
                "legacy_range": old_range,
                "waiting_range": new_range,
                "same_best_action": bool(int(np.argmin(old)) == int(np.argmin(new))),
            }
        )

    max_invariant_difference = max(
        [feature_difference, context_difference, *target_differences.values()],
        default=0.0,
    )
    values_finite = bool(
        np.all(np.isfinite(legacy_cost)) and np.all(np.isfinite(waiting_cost))
    )
    values_nonnegative = bool(
        np.all(legacy_cost >= -tolerance) and np.all(waiting_cost >= -tolerance)
    )
    group_count = len(group_rows)
    result = {
        "exact_action_key_set": True,
        "exact_action_key_order": legacy_keys == waiting_keys,
        "legacy_rows": int(legacy.size),
        "waiting_rows": int(waiting.size),
        "group_count": int(group_count),
        "informative_waiting_group_count": int(informative_groups),
        "informative_waiting_group_fraction": float(
            informative_groups / max(group_count, 1)
        ),
        "same_best_action_fraction": float(
            best_action_agreements / max(group_count, 1)
        ),
        "pairwise_sign_agreement": (
            float(pairwise_agreements / pairwise_comparisons)
            if pairwise_comparisons
            else None
        ),
        "pairwise_comparison_count": int(pairwise_comparisons),
        "row_cost_pearson": _finite_pearson(legacy_cost, waiting_cost),
        "within_group_centered_cost_pearson": _finite_pearson(
            legacy_centered, waiting_centered
        ),
        "legacy_cost": {
            "minimum": float(np.min(legacy_cost)),
            "mean": float(np.mean(legacy_cost)),
            "maximum": float(np.max(legacy_cost)),
        },
        "waiting_cost": {
            "minimum": float(np.min(waiting_cost)),
            "mean": float(np.mean(waiting_cost)),
            "maximum": float(np.max(waiting_cost)),
        },
        "trajectory_invariance": {
            "feature_max_abs_difference": feature_difference,
            "context_max_abs_difference": context_difference,
            "target_max_abs_difference": target_differences,
            "maximum_abs_difference": max_invariant_difference,
            "passed": bool(max_invariant_difference <= tolerance),
        },
        "values_finite": values_finite,
        "values_nonnegative": values_nonnegative,
        "groups": group_rows,
    }
    result["passed"] = bool(
        exact_key_set
        and values_finite
        and values_nonnegative
        and group_count > 0
        and informative_groups > 0
        and result["trajectory_invariance"]["passed"]
    )
    return result


def run_pilot(
    *,
    protocol_path: Path,
    partition_path: Path,
    source_protocol_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    legacy_cache_root: Path,
    waiting_cache_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    partition = _read_json(partition_path)
    source_protocol = _read_json(source_protocol_path)
    if protocol.get("protocol") != PILOT_PROTOCOL:
        raise ValueError("waiting-aligned pilot protocol changed")
    inputs = dict(protocol["frozen_inputs"])
    if (
        _sha256(partition_path) != inputs["partition_sha256"]
        or _sha256(source_protocol_path) != inputs["source_protocol_sha256"]
        or _sha256(manifest_path) != inputs["manifest_sha256"]
    ):
        raise ValueError("waiting-aligned pilot frozen input changed")
    pilot = dict(protocol["pilot"])
    seed = int(pilot["seed"])
    if (
        seed not in {int(value) for value in partition["redevelopment"]["seeds"]}
        or seed in {int(value) for value in partition["confirmatory"]["seeds"]}
        or seed in {int(value) for value in partition["prospective"]["seeds"]}
    ):
        raise ValueError("waiting-aligned pilot seed is not development-only")
    collection = dict(source_protocol["long_horizon_collection"])
    expected = {
        "duration_sec": float(collection["duration_sec"]),
        "control_interval_sec": int(collection["control_interval_sec"]),
        "warmup_sec": float(collection["warmup_sec"]),
        "max_focal_tls": int(collection["max_focal_tls"]),
        "counterfactual_horizon_intervals": int(
            collection["counterfactual_horizon_intervals"]
        ),
        "collection_shard_count": int(
            collection["collection_shards_per_scenario_seed"]
        ),
        "behavior_policy": str(collection["behavior_policy"]),
    }
    for name, value in expected.items():
        if pilot.get(name) != value:
            raise ValueError(f"waiting-aligned pilot {name} changed")
    if str(libsumo_version()) != str(protocol["environment"]["sumo_version"]):
        raise RuntimeError("waiting-aligned pilot SUMO version changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(pilot["scenario"])
    sumocfg = Path(manifest.sumocfgs[scenario])
    common = {
        "sumocfg": sumocfg,
        "scenario": scenario,
        "duration_sec": expected["duration_sec"],
        "control_interval_sec": expected["control_interval_sec"],
        "warmup_sec": expected["warmup_sec"],
        "seed": seed,
        "max_focal_tls": expected["max_focal_tls"],
        "counterfactual_horizon_intervals": expected[
            "counterfactual_horizon_intervals"
        ],
        "behavior_policy": expected["behavior_policy"],
        "collection_shard_index": int(pilot["collection_shard_index"]),
        "collection_shard_count": expected["collection_shard_count"],
    }
    legacy_identity = _counterfactual_cache_identity(**common)
    legacy_path = _counterfactual_cache_path(
        cache_root=legacy_cache_root,
        **common,
    )
    if not legacy_path.is_file():
        raise FileNotFoundError(f"matched legacy cache is missing: {legacy_path}")
    legacy = load_mechanism_dataset(legacy_path)
    _validate_counterfactual_cache_dataset(legacy, legacy_identity, legacy_path)

    waiting_identity = _counterfactual_cache_identity(
        **common,
        counterfactual_cost_mode="halted_queue",
    )
    waiting_path = _counterfactual_cache_path(
        cache_root=waiting_cache_root,
        **common,
        counterfactual_cost_mode="halted_queue",
    )
    started = time.perf_counter()
    _, _, _, waiting = _collect_worker(
        {
            **common,
            "counterfactual_cost_mode": "halted_queue",
            "cache_identity": waiting_identity,
            "cache_path": str(waiting_path),
        }
    )
    elapsed = float(time.perf_counter() - started)
    comparison = compare_counterfactual_costs(legacy, waiting)
    replay = dict(waiting.metadata.get("counterfactual_replay_audit", {}))
    safety = dict(waiting.metadata.get("counterfactual_safety_audit", {}))
    behavior = dict(waiting.metadata.get("behavior_safety_audit", {}))
    metadata_gate = bool(
        waiting.metadata.get("counterfactual_estimand")
        == WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5
        and waiting.metadata.get("counterfactual_cost_mode") == "halted_queue"
        and waiting.metadata.get("counterfactual_cost_scope")
        == "global_halted_vehicles_per_controlled_lane"
        and int(waiting.metadata.get("rollout_value_horizon_sec", -1))
        == expected["control_interval_sec"]
        * expected["counterfactual_horizon_intervals"]
    )
    safety_gate = bool(
        replay.get("passed", False)
        and safety.get("no_teleport_passed", False)
        and safety.get("symmetric_group_censoring_passed", False)
        and behavior.get("no_teleport_passed", False)
    )
    trace_gate = bool(
        legacy.metadata.get("behavior_trace_sha256")
        == waiting.metadata.get("behavior_trace_sha256")
    )
    passed = bool(metadata_gate and safety_gate and trace_gate and comparison["passed"])
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "authorize_waiting_aligned_redevelopment_collection"
            if passed
            else "reject_waiting_aligned_cache_implementation"
        ),
        "frozen_protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": _sha256(protocol_path),
        },
        "seed_classification": partition["redevelopment"]["classification"],
        "setting": {
            **pilot,
            "sumocfg": str(sumocfg),
            "legacy_cache": str(legacy_path),
            "waiting_cache": str(waiting_path),
        },
        "runtime": {
            "libsumo_version": str(libsumo_version()),
            "elapsed_seconds": elapsed,
        },
        "gates": {
            "metadata_contract": metadata_gate,
            "replay_and_safety": safety_gate,
            "identical_behavior_trace": trace_gate,
            "matched_trajectory_comparison": bool(comparison["passed"]),
            "passed": passed,
        },
        "comparison": comparison,
        "new_cache_audits": {
            "replay": replay,
            "counterfactual_safety": safety,
            "behavior_safety": behavior,
        },
        "claim_boundary": (
            "This development-only pilot validates the waiting-aligned label "
            "implementation. It is not efficacy or confirmatory evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--waiting-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite pilot result: {args.out}")
    result = run_pilot(
        protocol_path=args.protocol,
        partition_path=args.partition,
        source_protocol_path=args.source_protocol,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        legacy_cache_root=args.legacy_cache_root,
        waiting_cache_root=args.waiting_cache_root,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "groups": result["comparison"].get("group_count", 0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
