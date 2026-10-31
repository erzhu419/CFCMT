"""Audit the full V113 waiting-aligned target adaptation cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
from cf_h2o.eval.traffic_signal_source_waiting_aligned_cache_audit import (
    _scenario_seed_audit,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)


PROTOCOL = "tsc-v114-target-pure-waiting-adaptation-cache-v1"
RESULT_PROTOCOL = "tsc-v114-target-pure-waiting-adaptation-cache-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _flatten_assignments(
    assignments: Mapping[str, Sequence[str]],
) -> list[str]:
    return [str(scenario) for scenarios in assignments.values() for scenario in scenarios]


def _validate_protocol_contract(
    protocol: Mapping[str, Any],
    manifest: TrafficSignalBenchmarkManifest,
) -> None:
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("target waiting-aligned cache protocol changed")
    collection = dict(protocol["collection"])
    assignments = {
        str(node): [str(value) for value in scenarios]
        for node, scenarios in protocol["node_scenario_assignments"].items()
    }
    scenarios = _flatten_assignments(assignments)
    manifest_scenarios = [item.scenario for item in manifest.scenarios]
    manifest_groups = {item.city_group for item in manifest.scenarios}
    expected_groups = {str(value) for value in protocol["target_city_groups"]}
    expected_files = (
        len(scenarios)
        * len(collection["seeds"])
        * int(collection["collection_shards_per_seed"])
    )
    if len(assignments) != 4 or len(set(assignments)) != 4:
        raise ValueError("target cache must use four distinct nodes")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("target cache scenario assignments contain duplicates")
    if set(scenarios) != set(manifest_scenarios):
        raise ValueError("target cache assignments do not match the full manifest")
    if expected_groups != manifest_groups:
        raise ValueError("target city groups do not match the full manifest")
    if len(scenarios) != int(protocol["expected_scenario_count"]):
        raise ValueError("target scenario count changed")
    if expected_files != int(protocol["expected_cache_file_count"]):
        raise ValueError("target cache file-count contract changed")
    if [int(value) for value in collection["seeds"]] != [5057, 6067]:
        raise ValueError("target adaptation seeds changed")
    if collection.get("rollout_prefix_horizons_sec") != [450]:
        raise ValueError("target cache must contain the frozen 450-second target")
    if int(collection["counterfactual_horizon_intervals"]) != 45:
        raise ValueError("target cache rollout horizon changed")
    if str(collection["counterfactual_cost_mode"]) != "halted_queue":
        raise ValueError("target cache estimand changed")
    if collection.get("counterfactual_cost_contract") != counterfactual_cost_contract_v5(
        "halted_queue"
    ):
        raise ValueError("target pure halted-queue cost contract changed")


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
    collection = dict(protocol["collection"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    _validate_protocol_contract(protocol, manifest)
    assignments = {
        str(node): [str(value) for value in scenarios]
        for node, scenarios in protocol["node_scenario_assignments"].items()
    }
    scenarios = _flatten_assignments(assignments)
    seeds = [int(value) for value in collection["seeds"]]
    shard_count = int(collection["collection_shards_per_seed"])
    horizons = tuple(
        int(value) for value in collection["rollout_prefix_horizons_sec"]
    )
    target_names = tuple(rollout_prefix_target_name(value) for value in horizons)
    jobs: list[tuple[str, str, int, int, dict[str, Any], Path]] = []
    for scenario in scenarios:
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
            f"target cache is incomplete ({len(missing)} missing): {missing[:8]}"
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
            "candidate_groups": int(safety.get("candidate_groups", -1)),
            "retained_groups": int(safety.get("retained_groups", -1)),
            "behavior_collision_steps": int(
                behavior.get("collision_event_steps", 0)
            ),
            "full_horizon_equivalence_passed": full_equal,
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
            shard_count=shard_count,
            items=grouped[(scenario, seed)],
        )
        for scenario in scenarios
        for seed in seeds
    ]

    task_audits = []
    for node, assigned in assignments.items():
        path = Path(task_root) / node / "cache_summary.json"
        summary = _read_json(path)
        setting = dict(summary.get("setting", {}))
        diagnostics = dict(summary.get("diagnostics", {}))
        passed = bool(
            setting.get("cache_version")
            == MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            and setting.get("counterfactual_cost_mode") == "halted_queue"
            and tuple(setting.get("rollout_prefix_horizons_sec", ())) == horizons
            and setting.get("counterfactual_cost_contract")
            == collection["counterfactual_cost_contract"]
            and [int(value) for value in setting.get("seeds", ())] == seeds
            and [str(value) for value in setting.get("scenarios", ())] == assigned
            and int(setting.get("collection_shards", -1)) == shard_count
            and int(setting.get("counterfactual_horizon_intervals", -1))
            == int(collection["counterfactual_horizon_intervals"])
            and str(setting.get("behavior_policy")) == collection["behavior_policy"]
            and str(setting.get("libsumo_version")) == collection["sumo_version"]
            and set(diagnostics) == set(assigned)
            and all(
                float(diagnostics[scenario].get("tls_coverage_fraction", 0.0))
                >= float(collection["minimum_tls_coverage"])
                for scenario in assigned
            )
        )
        task_audits.append(
            {
                "node": node,
                "path": str(path),
                "passed": passed,
                "scenarios": assigned,
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )
    passed = bool(
        len(shard_audits) == int(protocol["expected_cache_file_count"])
        and all(item["passed"] for item in scenario_seed_audits)
        and all(item["passed"] for item in task_audits)
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "authorize_equal_information_b100_waiting_aligned_fit"
            if passed
            else "reject_target_cache_and_prohibit_b100_fit"
        ),
        "cache_version": MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
        "cache_file_count": len(shard_audits),
        "expected_cache_file_count": int(protocol["expected_cache_file_count"]),
        "scenario_count": len(scenarios),
        "city_group_count": len(set(manifest.city_groups.values())),
        "seeds": seeds,
        "shards_per_scenario_seed": shard_count,
        "rollout_prefix_horizons_sec": list(horizons),
        "total_rows": sum(int(item["rows"]) for item in shard_audits),
        "total_groups": sum(int(item["groups"]) for item in shard_audits),
        "scenario_seed_audits": scenario_seed_audits,
        "task_audits": task_audits,
        "information_budget": protocol["information_budget"],
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
        raise FileExistsError(f"refusing to overwrite target cache audit: {args.out}")
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
                "scenarios": result["scenario_count"],
                "rows": result["total_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
