"""Audit the complete v76 multihorizon waiting-cost cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    _counterfactual_cache_identity,
    _counterfactual_cache_path,
    _validate_counterfactual_cache_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-waiting-cache-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def audit_cache(
    *,
    protocol_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    cache_root: Path,
    task_root: Path,
    read_workers: int,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("multihorizon cache audit protocol changed")
    collection = dict(protocol["collection"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(collection["scenario"])
    sumocfg = Path(manifest.sumocfgs[scenario])
    seeds = [int(value) for value in collection["seeds"]]
    shard_count = int(collection["collection_shards_per_seed"])
    horizons = tuple(
        int(value) for value in collection["rollout_prefix_horizons_sec"]
    )
    target_names = tuple(rollout_prefix_target_name(value) for value in horizons)
    common = {
        "sumocfg": sumocfg,
        "scenario": scenario,
        "duration_sec": float(collection["duration_sec"]),
        "control_interval_sec": int(collection["control_interval_sec"]),
        "warmup_sec": float(collection["warmup_sec"]),
        "max_focal_tls": int(collection["max_focal_tls"]),
        "counterfactual_horizon_intervals": int(
            collection["counterfactual_horizon_intervals"]
        ),
        "behavior_policy": str(collection["behavior_policy"]),
        "counterfactual_cost_mode": str(collection["counterfactual_cost_mode"]),
        "collection_shard_count": shard_count,
        "rollout_prefix_horizons_sec": horizons,
    }
    jobs = []
    for seed in seeds:
        for shard in range(shard_count):
            identity = _counterfactual_cache_identity(
                **common,
                seed=seed,
                collection_shard_index=shard,
            )
            path = _counterfactual_cache_path(
                cache_root=cache_root,
                **common,
                seed=seed,
                collection_shard_index=shard,
            )
            jobs.append((seed, shard, identity, path))
    missing = [str(path) for _, _, _, path in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"multihorizon cache is incomplete ({len(missing)} missing): {missing[:8]}"
        )

    def read(job: tuple[int, int, dict[str, Any], Path]) -> dict[str, Any]:
        seed, shard, identity, path = job
        dataset = load_mechanism_dataset(path)
        _validate_counterfactual_cache_dataset(dataset, identity, path)
        prefix = {
            name: np.asarray(dataset.targets[name], dtype=float)
            for name in target_names
        }
        groups = [str(value) for value in dataset.metadata["action_group_ids"]]
        states = [str(value) for value in dataset.metadata["candidate_states"]]
        replay = dict(dataset.metadata.get("counterfactual_replay_audit", {}))
        safety = dict(dataset.metadata.get("counterfactual_safety_audit", {}))
        behavior = dict(dataset.metadata.get("behavior_safety_audit", {}))
        full_equal = bool(
            np.allclose(
                prefix[rollout_prefix_target_name(horizons[-1])],
                np.asarray(dataset.targets["interval_cost"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
        passed = bool(
            identity.get("version") == MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            and dataset.metadata.get("counterfactual_cost_mode") == "halted_queue"
            and tuple(dataset.metadata.get("rollout_prefix_horizons_sec", ()))
            == horizons
            and replay.get("passed", False)
            and safety.get("no_teleport_passed", False)
            and safety.get("symmetric_group_censoring_passed", False)
            and behavior.get("no_teleport_passed", False)
            and all(values.shape == (dataset.size,) for values in prefix.values())
            and all(np.all(np.isfinite(values)) for values in prefix.values())
            and all(np.all(values >= -1e-8) for values in prefix.values())
            and full_equal
            and len(groups) == dataset.size
            and len(states) == dataset.size
        )
        return {
            "seed": seed,
            "shard": shard,
            "path": str(path),
            "rows": int(dataset.size),
            "groups": len(set(groups)),
            "action_keys": list(zip(groups, states)),
            "behavior_trace": dataset.metadata.get("behavior_trace_sha256"),
            "eligible_opportunities": int(
                dataset.metadata.get("eligible_collection_opportunities", -1)
            ),
            "controllable_tls": set(
                dataset.metadata.get("controllable_tls_ids", ())
            ),
            "covered_tls": set(dataset.metadata.get("covered_tls_ids", ())),
            "full_horizon_equivalence_passed": full_equal,
            "passed": passed,
        }

    with ThreadPoolExecutor(max_workers=max(1, int(read_workers))) as pool:
        rows = list(pool.map(read, jobs))
    by_seed = {seed: [] for seed in seeds}
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seed_audits = []
    for seed, items in by_seed.items():
        indices = sorted(int(item["shard"]) for item in items)
        traces = {str(item["behavior_trace"]) for item in items}
        opportunities = {int(item["eligible_opportunities"]) for item in items}
        controllable = set().union(*(item["controllable_tls"] for item in items))
        covered = set().union(*(item["covered_tls"] for item in items))
        action_keys = [key for item in items for key in item["action_keys"]]
        passed = bool(
            indices == list(range(shard_count))
            and len(traces) == 1
            and "None" not in traces
            and len(opportunities) == 1
            and next(iter(opportunities), -1) >= 0
            and controllable == covered
            and len(action_keys) == len(set(action_keys))
            and all(bool(item["passed"]) for item in items)
        )
        seed_audits.append(
            {
                "seed": seed,
                "passed": passed,
                "shard_count": len(items),
                "rows": sum(int(item["rows"]) for item in items),
                "groups": sum(int(item["groups"]) for item in items),
                "behavior_trace_count": len(traces),
                "controllable_tls_count": len(controllable),
                "covered_tls_count": len(covered),
                "full_horizon_equivalence_passed": all(
                    bool(item["full_horizon_equivalence_passed"])
                    for item in items
                ),
            }
        )
    task_audits = []
    for node, assigned in collection["node_assignments"].items():
        path = Path(task_root) / str(node) / "cache_summary.json"
        summary = _read_json(path)
        setting = dict(summary.get("setting", {}))
        passed = bool(
            setting.get("cache_version")
            == MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            and setting.get("counterfactual_cost_mode") == "halted_queue"
            and tuple(setting.get("rollout_prefix_horizons_sec", ())) == horizons
            and [int(value) for value in setting.get("seeds", ())]
            == [int(value) for value in assigned]
            and setting.get("scenarios") == [scenario]
        )
        task_audits.append(
            {"node": str(node), "path": str(path), "passed": passed}
        )
    passed = bool(
        len(rows) == int(collection["expected_cache_file_count"])
        and all(item["passed"] for item in seed_audits)
        and all(item["passed"] for item in task_audits)
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "authorize_multihorizon_nested_seed_oof_development"
            if passed
            else "reject_multihorizon_cache_and_prohibit_training"
        ),
        "cache_file_count": len(rows),
        "expected_cache_file_count": int(collection["expected_cache_file_count"]),
        "rollout_prefix_horizons_sec": list(horizons),
        "total_rows": sum(int(item["rows"]) for item in rows),
        "total_groups": sum(int(item["groups"]) for item in rows),
        "seed_audits": seed_audits,
        "task_audits": task_audits,
        "gate": {"passed": passed},
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite cache audit: {args.out}")
    result = audit_cache(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        cache_root=args.cache_root,
        task_root=args.task_root,
        read_workers=args.read_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "cache_files": result["cache_file_count"],
                "rows": result["total_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
