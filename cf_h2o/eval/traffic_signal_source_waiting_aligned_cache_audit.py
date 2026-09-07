"""Audit the full V112 waiting-aligned source counterfactual cache."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    counterfactual_cost_contract_v5,
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    _counterfactual_cache_identity,
    _counterfactual_cache_path,
    _validate_counterfactual_cache_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)


COLLECTION_PROTOCOLS = {
    "tsc-v114-source-pure-waiting-cache-v1",
    "tsc-v114-source-pure-waiting-cache-v2",
}
ADMISSION_PROTOCOLS = {
    "tsc-v114-source-pure-waiting-admission-v1",
    "tsc-v114-source-pure-waiting-admission-v2",
}
NATIVE_SAFETY_PROTOCOL = "tsc-native-sumo-collision-safety-audit-v1"
RESULT_PROTOCOL = "tsc-v114-source-pure-waiting-cache-admission-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_admission_protocol(
    admission_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    admission = _read_json(admission_path)
    if admission.get("protocol") not in ADMISSION_PROTOCOLS:
        raise ValueError("source waiting-aligned admission protocol changed")
    reference = dict(admission["collection_protocol"])
    collection_path = Path(admission_path).parent / str(reference["path"])
    if _sha256(collection_path) != str(reference["sha256"]):
        raise ValueError("source collection protocol identity changed")
    collection = _read_json(collection_path)
    if (
        collection.get("protocol") not in COLLECTION_PROTOCOLS
        or collection.get("protocol") != reference.get("protocol")
    ):
        raise ValueError("source collection protocol name changed")
    return admission, collection


def _scenario_seed_jobs(protocol: Mapping[str, Any]) -> tuple[
    dict[str, dict[str, list[Any]]], list[tuple[str, int]]
]:
    jobs = {
        str(node): {
            "scenarios": [str(value) for value in job["scenarios"]],
            "seeds": [int(value) for value in job["seeds"]],
        }
        for node, job in protocol["node_jobs"].items()
    }
    pairs = [
        (scenario, seed)
        for job in jobs.values()
        for scenario in job["scenarios"]
        for seed in job["seeds"]
    ]
    return jobs, pairs


def _scenario_shard_count(protocol: Mapping[str, Any], scenario: str) -> int:
    collection = dict(protocol["collection"])
    overrides = {
        str(name): int(value)
        for name, value in dict(
            collection.get("scenario_collection_shards_per_seed", {})
        ).items()
    }
    return int(overrides.get(str(scenario), collection["collection_shards_per_seed"]))


def _validate_protocol_contract(
    protocol: Mapping[str, Any],
    manifest: TrafficSignalBenchmarkManifest,
) -> None:
    if protocol.get("protocol") not in COLLECTION_PROTOCOLS:
        raise ValueError("source waiting-aligned cache protocol changed")
    collection = dict(protocol["collection"])
    jobs, scenario_seed_pairs = _scenario_seed_jobs(protocol)
    scenarios = sorted({scenario for scenario, _ in scenario_seed_pairs})
    manifest_scenarios = [item.scenario for item in manifest.scenarios]
    manifest_groups = {item.city_group for item in manifest.scenarios}
    expected_groups = {str(value) for value in protocol["source_city_groups"]}
    seeds = [int(value) for value in collection["seeds"]]
    shard_overrides = {
        str(name): int(value)
        for name, value in dict(
            collection.get("scenario_collection_shards_per_seed", {})
        ).items()
    }
    if set(shard_overrides) - set(scenarios) or any(
        value < 1 for value in shard_overrides.values()
    ):
        raise ValueError("source cache shard overrides are invalid")
    expected_files = sum(
        _scenario_shard_count(protocol, scenario)
        for scenario, _ in scenario_seed_pairs
    )
    if tuple(jobs) != tuple(f"node00{index}" for index in range(1, 7)):
        raise ValueError("source cache must use node001 through node006 once")
    if set(scenarios) != set(manifest_scenarios):
        raise ValueError("source cache assignments do not match the full manifest")
    if (
        len(scenario_seed_pairs) != len(set(scenario_seed_pairs))
        or any(
            {seed for name, seed in scenario_seed_pairs if name == scenario}
            != set(seeds)
            for scenario in scenarios
        )
    ):
        raise ValueError("source cache scenario-seed jobs are duplicated or incomplete")
    if expected_groups != manifest_groups:
        raise ValueError("source city groups do not match the full manifest")
    if len(scenarios) != int(protocol["expected_scenario_count"]):
        raise ValueError("source scenario count changed")
    if expected_files != int(protocol["expected_cache_file_count"]):
        raise ValueError("source cache file-count contract changed")
    if len(scenario_seed_pairs) != int(protocol["expected_scenario_seed_count"]):
        raise ValueError("source scenario-seed count changed")
    if collection.get("rollout_prefix_horizons_sec") != [450]:
        raise ValueError("source cache must contain the frozen 450-second target")
    if int(collection["counterfactual_horizon_intervals"]) != 45:
        raise ValueError("source cache rollout horizon changed")
    if str(collection["counterfactual_cost_mode"]) != "halted_queue":
        raise ValueError("source cache estimand changed")
    if collection.get("counterfactual_cost_contract") != counterfactual_cost_contract_v5(
        "halted_queue"
    ):
        raise ValueError("source pure halted-queue cost contract changed")


def _scenario_seed_audit(
    *,
    scenario: str,
    city_group: str,
    seed: int,
    shard_count: int,
    items: Sequence[Mapping[str, Any]],
    excluded: bool = False,
) -> dict[str, Any]:
    indices = sorted(int(item["shard"]) for item in items)
    traces = {str(item["behavior_trace"]) for item in items}
    opportunities = {int(item["eligible_opportunities"]) for item in items}
    controllable = set().union(*(set(item["controllable_tls"]) for item in items))
    covered = set().union(*(set(item["covered_tls"]) for item in items))
    action_keys = [key for item in items for key in item["action_keys"]]
    common_passed = bool(
        indices == list(range(shard_count))
        and len(traces) == 1
        and "None" not in traces
        and len(opportunities) == 1
        and next(iter(opportunities), -1) >= 0
        and bool(controllable)
        and len(action_keys) == len(set(action_keys))
        and all(bool(item["passed"]) for item in items)
    )
    retained_groups = sum(int(item["retained_groups"]) for item in items)
    candidate_groups = sum(int(item["candidate_groups"]) for item in items)
    behavior_collision_steps = max(
        (int(item["behavior_collision_steps"]) for item in items),
        default=0,
    )
    passed = bool(
        common_passed
        and (
            (
                retained_groups == 0
                and candidate_groups > 0
                and not covered
                and behavior_collision_steps > 0
            )
            if excluded
            else (retained_groups > 0 and controllable == covered)
        )
    )
    return {
        "scenario": str(scenario),
        "city_group": str(city_group),
        "seed": int(seed),
        "passed": passed,
        "training_admitted": not excluded,
        "admission_status": (
            "complete_but_zero_retained_groups"
            if excluded
            else "admitted_for_waiting_aligned_training"
        ),
        "shard_count": len(items),
        "rows": sum(int(item["rows"]) for item in items),
        "groups": sum(int(item["groups"]) for item in items),
        "behavior_trace_count": len(traces),
        "eligible_collection_opportunities": next(iter(opportunities), -1),
        "controllable_tls_count": len(controllable),
        "covered_tls_count": len(covered),
        "unique_action_key_count": len(set(action_keys)),
        "candidate_group_count": candidate_groups,
        "retained_group_count": retained_groups,
        "behavior_collision_event_steps": behavior_collision_steps,
        "full_horizon_equivalence_passed": all(
            bool(item["full_horizon_equivalence_passed"]) for item in items
        ),
    }


def audit_cache(
    *,
    protocol_path: Path,
    native_safety_audit_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    cache_root: Path,
    task_root: Path,
    read_workers: int,
) -> dict[str, Any]:
    admission_protocol, protocol = _load_admission_protocol(protocol_path)
    collection = dict(protocol["collection"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    _validate_protocol_contract(protocol, manifest)
    admission = dict(admission_protocol["admission"])
    exclusions = {
        str(name): dict(value)
        for name, value in admission["excluded_scenarios"].items()
    }
    if not exclusions or set(exclusions) - set(manifest.sumocfgs):
        raise ValueError("source admission exclusions are empty or unknown")
    if (
        int(admission["expected_collected_scenario_count"])
        != len(manifest.scenarios)
        or int(admission["expected_training_scenario_count"])
        != len(manifest.scenarios) - len(exclusions)
    ):
        raise ValueError("source admission scenario counts changed")
    native_safety = _read_json(native_safety_audit_path)
    if (
        native_safety.get("protocol") != NATIVE_SAFETY_PROTOCOL
        or native_safety.get("status") != "UNSAFE"
        or set(native_safety.get("scenarios", ())) != set(exclusions)
        or set(int(value) for value in native_safety.get("seeds", ()))
        != set(int(value) for value in collection["seeds"])
        or int(native_safety.get("unsafe_run_count", -1))
        != len(exclusions) * len(collection["seeds"])
        or any(
            row.get("status") != "UNSAFE"
            for row in native_safety.get("runs", ())
        )
    ):
        raise ValueError("native collision audit does not authorize exclusions")

    jobs_by_node, _ = _scenario_seed_jobs(protocol)
    scenarios = [item.scenario for item in manifest.scenarios]
    seeds = [int(value) for value in collection["seeds"]]
    horizons = tuple(
        int(value) for value in collection["rollout_prefix_horizons_sec"]
    )
    target_names = tuple(rollout_prefix_target_name(value) for value in horizons)
    jobs: list[tuple[str, str, int, int, dict[str, Any], Path]] = []
    for scenario in scenarios:
        shard_count = _scenario_shard_count(protocol, scenario)
        common = {
            "sumocfg": Path(manifest.sumocfgs[scenario]),
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
                jobs.append(
                    (
                        scenario,
                        manifest.city_groups[scenario],
                        seed,
                        shard,
                        identity,
                        path,
                    )
                )

    missing = [str(path) for *_, path in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"source cache is incomplete ({len(missing)} missing): {missing[:8]}"
        )

    def read(
        job: tuple[str, str, int, int, dict[str, Any], Path],
    ) -> dict[str, Any]:
        scenario, city_group, seed, shard, identity, path = job
        dataset = load_mechanism_dataset(path)
        _validate_counterfactual_cache_dataset(dataset, identity, path)
        prefixes = {
            name: np.asarray(dataset.targets[name], dtype=float)
            for name in target_names
        }
        interval_cost = np.asarray(dataset.targets["interval_cost"], dtype=float)
        groups = [str(value) for value in dataset.metadata["action_group_ids"]]
        states = [str(value) for value in dataset.metadata["candidate_states"]]
        replay = dict(dataset.metadata.get("counterfactual_replay_audit", {}))
        safety = dict(dataset.metadata.get("counterfactual_safety_audit", {}))
        behavior = dict(dataset.metadata.get("behavior_safety_audit", {}))
        full_equal = bool(
            np.allclose(
                prefixes[target_names[-1]],
                interval_cost,
                rtol=0.0,
                atol=1e-12,
            )
        )
        passed = bool(
            identity.get("version") == MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            and dataset.metadata.get("counterfactual_cost_mode") == "halted_queue"
            and dataset.metadata.get("counterfactual_estimand")
            == WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
            and dataset.metadata.get("counterfactual_cost_population")
            == "sumo_last_step_halting_number_on_controlled_lanes"
            and tuple(dataset.metadata.get("rollout_prefix_horizons_sec", ()))
            == horizons
            and replay.get("passed", False)
            and safety.get("no_teleport_passed", False)
            and safety.get("symmetric_group_censoring_passed", False)
            and behavior.get("no_teleport_passed", False)
            and interval_cost.shape == (dataset.size,)
            and np.all(np.isfinite(interval_cost))
            and np.all(interval_cost >= -1e-8)
            and all(values.shape == (dataset.size,) for values in prefixes.values())
            and all(np.all(np.isfinite(values)) for values in prefixes.values())
            and all(np.all(values >= -1e-8) for values in prefixes.values())
            and full_equal
            and len(groups) == dataset.size
            and len(states) == dataset.size
        )
        return {
            "scenario": scenario,
            "city_group": city_group,
            "seed": seed,
            "shard": shard,
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
            "candidate_groups": int(safety.get("candidate_groups", -1)),
            "retained_groups": int(safety.get("retained_groups", -1)),
            "behavior_collision_steps": int(
                behavior.get("collision_event_steps", 0)
            ),
            "passed": passed,
        }

    with ThreadPoolExecutor(max_workers=max(1, int(read_workers))) as pool:
        shard_audits = list(pool.map(read, jobs))

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {
        (scenario, seed): [] for scenario in scenarios for seed in seeds
    }
    for item in shard_audits:
        grouped[(str(item["scenario"]), int(item["seed"]))].append(item)
    scenario_seed_audits = [
        _scenario_seed_audit(
            scenario=scenario,
            city_group=manifest.city_groups[scenario],
            seed=seed,
            shard_count=_scenario_shard_count(protocol, scenario),
            items=grouped[(scenario, seed)],
            excluded=scenario in exclusions,
        )
        for scenario in scenarios
        for seed in seeds
    ]

    require_task_summaries = (
        protocol.get("task_summary_gate") != "direct_cache_audit_only"
    )
    task_audits = []
    for node, job in jobs_by_node.items() if require_task_summaries else ():
        assigned = job["scenarios"]
        assigned_seeds = job["seeds"]
        path = Path(task_root) / node / "cache_summary.json"
        summary = _read_json(path)
        setting = dict(summary.get("setting", {}))
        diagnostics = dict(summary.get("diagnostics", {}))
        admitted_assigned = [
            scenario for scenario in assigned if scenario not in exclusions
        ]
        passed = bool(
            setting.get("cache_version")
            == MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            and setting.get("counterfactual_cost_mode") == "halted_queue"
            and tuple(setting.get("rollout_prefix_horizons_sec", ())) == horizons
            and setting.get("counterfactual_cost_contract")
            == collection["counterfactual_cost_contract"]
            and [int(value) for value in setting.get("seeds", ())]
            == assigned_seeds
            and [str(value) for value in setting.get("scenarios", ())] == assigned
            and set(setting.get("allowed_zero_retained_scenarios", ()))
            == (set(assigned) & set(exclusions))
            and int(setting.get("collection_shards", -1))
            == int(collection["collection_shards_per_seed"])
            and int(setting.get("counterfactual_horizon_intervals", -1))
            == int(collection["counterfactual_horizon_intervals"])
            and str(setting.get("behavior_policy")) == collection["behavior_policy"]
            and str(setting.get("libsumo_version")) == collection["sumo_version"]
            and set(diagnostics) == set(assigned)
            and all(
                (
                    diagnostics[scenario].get("collection_admission_status")
                    == "complete_but_zero_retained_groups"
                    if scenario in exclusions
                    else float(
                        diagnostics[scenario].get("tls_coverage_fraction", 0.0)
                    )
                    >= float(collection["minimum_tls_coverage"])
                )
                for scenario in assigned
            )
        )
        task_audits.append(
            {
                "node": node,
                "path": str(path),
                "passed": passed,
                "scenarios": admitted_assigned,
                "seeds": assigned_seeds,
                "excluded_scenarios": [
                    scenario for scenario in assigned if scenario in exclusions
                ],
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )

    city_group_counts = Counter(
        manifest.city_groups[scenario] for scenario in scenarios
    )
    admitted_scenarios = [scenario for scenario in scenarios if scenario not in exclusions]
    admitted_city_groups = {
        manifest.city_groups[scenario] for scenario in admitted_scenarios
    }
    passed = bool(
        len(shard_audits) == int(protocol["expected_cache_file_count"])
        and len(admitted_scenarios)
        == int(admission["expected_training_scenario_count"])
        and admitted_city_groups == set(protocol["source_city_groups"])
        and all(item["passed"] for item in scenario_seed_audits)
        and (
            all(item["passed"] for item in task_audits)
            if require_task_summaries
            else True
        )
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "authorize_waiting_aligned_source_mechanism_training"
            if passed
            else "reject_source_cache_and_prohibit_training"
        ),
        "cache_version": MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
        "cache_file_count": len(shard_audits),
        "expected_cache_file_count": int(protocol["expected_cache_file_count"]),
        "scenario_count": len(scenarios),
        "training_scenario_count": len(admitted_scenarios),
        "training_scenarios": admitted_scenarios,
        "excluded_scenarios": exclusions,
        "city_group_count": len(city_group_counts),
        "training_city_group_count": len(admitted_city_groups),
        "city_group_scenario_counts": dict(sorted(city_group_counts.items())),
        "seeds": seeds,
        "shards_per_scenario_seed": {
            scenario: _scenario_shard_count(protocol, scenario)
            for scenario in scenarios
        },
        "rollout_prefix_horizons_sec": list(horizons),
        "total_rows": sum(int(item["rows"]) for item in shard_audits),
        "total_groups": sum(int(item["groups"]) for item in shard_audits),
        "scenario_seed_audits": scenario_seed_audits,
        "task_audits": task_audits,
        "native_safety_audit": {
            "path": str(Path(native_safety_audit_path).resolve()),
            "sha256": _sha256(native_safety_audit_path),
            "status": native_safety["status"],
            "unsafe_run_count": native_safety["unsafe_run_count"],
        },
        "gate": {"passed": passed},
        "claim_boundary": admission_protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-safety-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite source cache audit: {args.out}")
    result = audit_cache(
        protocol_path=args.protocol,
        native_safety_audit_path=args.native_safety_audit,
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
                "scenarios": result["scenario_count"],
                "rows": result["total_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
