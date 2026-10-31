"""Collect an explicit subset of counterfactual cache seed/shard pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    counterfactual_cost_contract_v5,
    normalize_rollout_prefix_horizons,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _collect_worker,
    _counterfactual_cache_identity,
    _counterfactual_cache_path,
    _map_fresh_sumo_processes,
)
from cf_h2o.sumo_runtime import libsumo_version
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-counterfactual-explicit-seed-shard-subset-v1"


def parse_seed_shards(values: Sequence[str], *, collection_shards: int) -> tuple[tuple[int, int], ...]:
    pairs = []
    for value in values:
        seed_text, separator, shard_text = str(value).partition(":")
        if not separator:
            raise ValueError(f"seed/shard pair must use SEED:SHARD syntax: {value!r}")
        pair = (int(seed_text), int(shard_text))
        if not 0 <= pair[1] < int(collection_shards):
            raise ValueError(f"shard index is outside [0, {int(collection_shards)}): {pair[1]}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("at least one explicit seed/shard pair is required")
    if len(pairs) != len(set(pairs)):
        raise ValueError("explicit seed/shard pairs contain duplicates")
    return tuple(pairs)


def _payload(
    *,
    scenario: str,
    sumocfg: Path,
    seed: int,
    shard_index: int,
    collection_shards: int,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    behavior_policy: str,
    counterfactual_cost_mode: str,
    rollout_prefix_horizons_sec: Sequence[int] | None,
    cache_root: Path,
) -> dict[str, Any]:
    identity_args = {
        "sumocfg": Path(sumocfg),
        "scenario": str(scenario),
        "duration_sec": float(duration_sec),
        "control_interval_sec": int(control_interval_sec),
        "warmup_sec": float(warmup_sec),
        "seed": int(seed),
        "max_focal_tls": int(max_focal_tls),
        "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
        "behavior_policy": str(behavior_policy),
        "counterfactual_cost_mode": str(counterfactual_cost_mode),
        "collection_shard_index": int(shard_index),
        "collection_shard_count": int(collection_shards),
        "rollout_prefix_horizons_sec": rollout_prefix_horizons_sec,
    }
    return {
        "scenario": str(scenario),
        "sumocfg": str(Path(sumocfg)),
        "duration_sec": float(duration_sec),
        "control_interval_sec": int(control_interval_sec),
        "warmup_sec": float(warmup_sec),
        "seed": int(seed),
        "max_focal_tls": int(max_focal_tls),
        "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
        "behavior_policy": str(behavior_policy),
        "counterfactual_cost_mode": str(counterfactual_cost_mode),
        "collection_shard_index": int(shard_index),
        "collection_shard_count": int(collection_shards),
        "rollout_prefix_horizons_sec": list(
            normalize_rollout_prefix_horizons(
                rollout_prefix_horizons_sec,
                rollout_horizon_sec=int(control_interval_sec)
                * max(int(counterfactual_horizon_intervals), 1),
            )
        ),
        "cache_identity": _counterfactual_cache_identity(**identity_args),
        "cache_path": str(
            _counterfactual_cache_path(cache_root=cache_root, **identity_args)
        ),
    }


def collect_shard_subset(
    *,
    manifest_path: Path,
    scenario: str,
    seed_shards: Sequence[tuple[int, int]],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    collection_shards: int,
    workers: int,
    cache_root: Path,
    behavior_policy: str,
    counterfactual_cost_mode: str,
    rollout_prefix_horizons_sec: Sequence[int] | None,
    expected_sumo_version: str | None,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(scenario)
    if scenario not in manifest.sumocfgs:
        raise ValueError(f"scenario is absent from manifest: {scenario}")
    pairs = tuple((int(seed), int(shard)) for seed, shard in seed_shards)
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("seed_shards must be non-empty and unique")
    if any(not 0 <= shard < int(collection_shards) for _, shard in pairs):
        raise ValueError("seed_shards contain an out-of-range shard")
    actual_sumo_version = libsumo_version()
    if expected_sumo_version and actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            f"counterfactual shard SUMO version mismatch: {actual_sumo_version!r}"
        )
    payloads = [
        _payload(
            scenario=scenario,
            sumocfg=manifest.sumocfgs[scenario],
            seed=seed,
            shard_index=shard,
            collection_shards=collection_shards,
            duration_sec=duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=warmup_sec,
            max_focal_tls=max_focal_tls,
            counterfactual_horizon_intervals=counterfactual_horizon_intervals,
            behavior_policy=behavior_policy,
            counterfactual_cost_mode=counterfactual_cost_mode,
            rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
            cache_root=cache_root,
        )
        for seed, shard in pairs
    ]
    cached_payloads = [item for item in payloads if Path(item["cache_path"]).is_file()]
    fresh_payloads = [item for item in payloads if not Path(item["cache_path"]).is_file()]
    started = time.perf_counter()
    cached = [_collect_worker(item) for item in cached_payloads]
    fresh = _map_fresh_sumo_processes(_collect_worker, fresh_payloads, workers)
    rows = sorted((*cached, *fresh), key=lambda item: (item[1], item[2]))
    observed_pairs = tuple((int(seed), int(shard)) for _, seed, shard, _ in rows)
    if observed_pairs != tuple(sorted(pairs)):
        raise RuntimeError("counterfactual shard workers returned the wrong pair set")
    diagnostics = []
    for _, seed, shard, dataset in rows:
        replay = dict(dataset.metadata.get("counterfactual_replay_audit", {}))
        safety = dict(dataset.metadata.get("counterfactual_safety_audit", {}))
        if not replay.get("passed", False) or not safety.get("no_teleport_passed", False):
            raise RuntimeError(f"counterfactual shard failed replay/safety: {seed}:{shard}")
        diagnostics.append(
            {
                "seed": int(seed),
                "shard_index": int(shard),
                "rows": int(dataset.size),
                "groups": len(set(dataset.metadata.get("action_group_ids", ()))),
                "cache_path": next(
                    item["cache_path"]
                    for item in payloads
                    if int(item["seed"]) == int(seed)
                    and int(item["collection_shard_index"]) == int(shard)
                ),
                "replay_checks": int(replay.get("checks", 0)),
                "retained_groups": int(safety.get("retained_groups", 0)),
                "censored_groups": int(safety.get("censored_groups", 0)),
            }
        )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scenario": scenario,
        "seed_shards": [f"{seed}:{shard}" for seed, shard in pairs],
        "collection_shards": int(collection_shards),
        "counterfactual_cost_contract": counterfactual_cost_contract_v5(
            counterfactual_cost_mode
        ),
        "cached_pair_count": len(cached_payloads),
        "fresh_pair_count": len(fresh_payloads),
        "elapsed_seconds": float(time.perf_counter() - started),
        "libsumo_version": actual_sumo_version,
        "diagnostics": diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed-shards", nargs="+", required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--control-interval-sec", type=int, required=True)
    parser.add_argument("--warmup-sec", type=float, required=True)
    parser.add_argument("--max-focal-tls", type=int, required=True)
    parser.add_argument("--counterfactual-horizon-intervals", type=int, required=True)
    parser.add_argument("--collection-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--behavior-policy",
        choices=("phase_pressure", "exploratory_mixture"),
        default="phase_pressure",
    )
    parser.add_argument(
        "--counterfactual-cost-mode",
        choices=("system_vehicle_load", "halted_queue"),
        default="system_vehicle_load",
    )
    parser.add_argument("--rollout-prefix-horizons-sec", nargs="+", type=int)
    parser.add_argument("--expected-sumo-version")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite shard-subset result: {args.out}")
    pairs = parse_seed_shards(
        args.seed_shards, collection_shards=args.collection_shards
    )
    result = collect_shard_subset(
        manifest_path=args.manifest,
        scenario=args.scenario,
        seed_shards=pairs,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        max_focal_tls=args.max_focal_tls,
        counterfactual_horizon_intervals=args.counterfactual_horizon_intervals,
        collection_shards=args.collection_shards,
        workers=args.workers,
        cache_root=args.cache_root,
        behavior_policy=args.behavior_policy,
        counterfactual_cost_mode=args.counterfactual_cost_mode,
        rollout_prefix_horizons_sec=args.rollout_prefix_horizons_sec,
        expected_sumo_version=args.expected_sumo_version,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs": len(result["seed_shards"]),
                "fresh": result["fresh_pair_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
