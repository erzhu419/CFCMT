"""Benchmark-aware junction-collision survey for TSC reference policies."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    STRICT_SAFETY_MONITORING_V3,
    evaluate_policy_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _load_libsumo,
    _map_fresh_sumo_processes,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    SUMO_EXECUTION_PROTOCOL,
)
from cf_h2o.sumo_runtime import libsumo_version
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


DEFAULT_POLICIES = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
)


def _safety_survey_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = evaluate_policy_v3(
        sumo_api=_load_libsumo(),
        sumocfg=Path(payload["sumocfg"]),
        scenario=str(payload["scenario"]),
        policy=str(payload["policy"]),
        models=None,
        duration_sec=float(payload["duration_sec"]),
        control_interval_sec=int(payload["control_interval_sec"]),
        warmup_sec=float(payload["warmup_sec"]),
        seed=int(payload["seed"]),
        tripinfo_output=Path(payload["tripinfo_output"]),
    )
    return {
        "scenario": str(payload["scenario"]),
        "city_group": str(payload["city_group"]),
        "policy": str(payload["policy"]),
        "seed": int(payload["seed"]),
        "metrics": metrics,
    }


def _aggregate_safety_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    policies = sorted({str(row["policy"]) for row in rows})
    fields = (
        "collision_events",
        "collision_event_steps",
        "collision_incidents",
        "collision_rate",
        "collision_incident_rate",
        "emergency_stops",
        "starting_teleports",
        "ending_teleports",
        "mean_system_vehicles_per_controlled_lane",
    )
    return {
        scenario: {
            policy: {
                field: float(
                    np.mean(
                        [
                            float(row["metrics"][field])
                            for row in rows
                            if str(row["scenario"]) == scenario
                            and str(row["policy"]) == policy
                        ]
                    )
                )
                for field in fields
            }
            for policy in policies
        }
        for scenario in scenarios
    }


def run_safety_survey(
    *,
    manifest_path: Path,
    scenarios: Sequence[str] | None,
    policies: Sequence[str],
    seeds: Sequence[int],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    workers: int,
    tripinfo_root: Path,
    expected_sumo_version: str | None = None,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(str(name) for name in (scenarios or tuple(manifest.sumocfgs)))
    unknown = set(selected) - set(manifest.sumocfgs)
    if unknown:
        raise ValueError(f"scenarios are absent from manifest: {sorted(unknown)}")
    selected_policies = tuple(str(policy) for policy in policies)
    unknown_policies = set(selected_policies) - set(DEFAULT_POLICIES)
    if unknown_policies:
        raise ValueError(f"unsupported safety-survey policies: {sorted(unknown_policies)}")
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected or not selected_policies or not selected_seeds:
        raise ValueError("safety survey requires scenarios, policies, and seeds")
    actual_sumo_version = libsumo_version()
    if expected_sumo_version and actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            "safety survey SUMO version mismatch: "
            f"expected {expected_sumo_version!r}, found {actual_sumo_version!r}"
        )
    jobs = [
        {
            "scenario": scenario,
            "city_group": manifest.city_groups[scenario],
            "sumocfg": str(manifest.sumocfgs[scenario]),
            "policy": policy,
            "seed": seed,
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "tripinfo_output": str(
                Path(tripinfo_root)
                / f"{scenario}__{policy}__seed{int(seed)}.xml"
            ),
        }
        for scenario in selected
        for policy in selected_policies
        for seed in selected_seeds
    ]
    started = time.perf_counter()
    rows = _map_fresh_sumo_processes(_safety_survey_worker, jobs, workers)
    rows.sort(key=lambda row: (row["scenario"], row["policy"], row["seed"]))
    errors: list[str] = []
    required_metrics = (
        "collision_events",
        "collision_event_steps",
        "collision_incidents",
        "collision_rate",
        "collision_incident_rate",
        "emergency_stops",
        "starting_teleports",
        "ending_teleports",
        "mean_system_vehicles_per_controlled_lane",
    )
    for row in rows:
        location = f"{row['scenario']}/{row['policy']}/seed={row['seed']}"
        metrics = dict(row.get("metrics", {}))
        if metrics.get("ok") is not True:
            errors.append(f"{location}: rollout failed: {metrics.get('error')}")
            continue
        if metrics.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
            errors.append(f"{location}: SUMO execution protocol mismatch")
        if metrics.get("strict_safety_monitoring") != STRICT_SAFETY_MONITORING_V3:
            errors.append(f"{location}: safety monitoring protocol mismatch")
        for field in required_metrics:
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{location}: invalid metric {field}")
        if int(metrics.get("starting_teleports", -1)) != 0 or int(
            metrics.get("ending_teleports", -1)
        ) != 0:
            errors.append(f"{location}: teleport event")
        if int(metrics.get("collision_incidents", -1)) > int(
            metrics.get("collision_events", -1)
        ):
            errors.append(f"{location}: collision incident count exceeds events")

    expected_rows = len(jobs)
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    successful_rows = [row for row in rows if row.get("metrics", {}).get("ok")]
    aggregate = (
        _aggregate_safety_rows(successful_rows)
        if not errors and len(successful_rows) == len(rows) and rows
        else {}
    )
    raw_collision_rollouts = sum(
        int(row.get("metrics", {}).get("collision_incidents", 0)) > 0
        for row in successful_rows
    )
    return {
        "experiment": "traffic_signal_benchmark_aware_safety_survey",
        "runtime": _runtime_metadata(),
        "setting": {
            "manifest": manifest.to_dict(),
            "scenarios": list(selected),
            "policies": list(selected_policies),
            "seeds": list(selected_seeds),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "workers": int(workers),
            "libsumo_version": actual_sumo_version,
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "collision_interpretation": (
                "raw_benchmark_events_audited_not_blanket_controller_failure"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "expected_row_count": expected_rows,
        "row_count": len(rows),
        "raw_collision_rollout_count": int(raw_collision_rollouts),
        "raw_collision_rollout_fraction": float(
            raw_collision_rollouts / max(len(successful_rows), 1)
        ),
        "passed": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cf_h2o/config/traffic_signal_cross_city_v1.json"),
    )
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1709, 2027, 3037])
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--tripinfo-root", type=Path, required=True)
    parser.add_argument("--expected-sumo-version")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_safety_survey(
        manifest_path=args.manifest,
        scenarios=args.scenarios,
        policies=args.policies,
        seeds=args.seeds,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        workers=args.workers,
        tripinfo_root=args.tripinfo_root,
        expected_sumo_version=args.expected_sumo_version,
    )
    atomic_write_json(args.out, result)
    if not result["passed"]:
        raise RuntimeError("safety survey failed: " + "; ".join(result["errors"]))
    print(
        f"audited {result['row_count']} rollouts; "
        f"raw-collision rollouts={result['raw_collision_rollout_count']}"
    )


if __name__ == "__main__":
    main()
