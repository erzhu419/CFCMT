"""Run isolated libsumo admission rollouts for derived Salt Lake scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _evaluate_worker,
    _map_fresh_sumo_processes,
)
from cf_h2o.sumo_runtime import runtime_metadata


SUPPORTED_POLICIES = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scenario_record(sumocfg: Path) -> dict[str, Any]:
    provenance_path = sumocfg.parent / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )
    expected_vehicles = (
        dict(provenance.get("demand_audit", {})).get("generated_vehicle_total")
    )
    return {
        "scenario": sumocfg.stem,
        "sumocfg": str(sumocfg.resolve()),
        "provenance": str(provenance_path.resolve()) if provenance else None,
        "expected_vehicles": int(expected_vehicles) if expected_vehicles is not None else None,
        "construction_protocol": provenance.get("construction_protocol"),
        "output_route_sha256": dict(provenance.get("hashes", {})).get("output_route_sha256"),
    }


def run_admission(
    sumocfgs: Sequence[Path],
    *,
    policies: Sequence[str] = SUPPORTED_POLICIES,
    seeds: Sequence[int] = (20260808,),
    duration_sec: float = 3600.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    safety_policy: str = "phase_pressure",
    workers: int = 4,
    tripinfo_root: Path,
) -> dict[str, Any]:
    requested_policies = tuple(dict.fromkeys(str(policy) for policy in policies))
    unknown = set(requested_policies) - set(SUPPORTED_POLICIES)
    if unknown:
        raise ValueError(f"unsupported admission policies: {sorted(unknown)}")
    if safety_policy not in requested_policies:
        raise ValueError("safety_policy must be included in policies")
    scenario_records = [_scenario_record(Path(path)) for path in sumocfgs]
    if not scenario_records:
        raise ValueError("at least one sumocfg is required")
    tripinfo_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scenario in scenario_records:
        for seed in seeds:
            for policy in requested_policies:
                jobs.append(
                    {
                        "scenario": scenario["scenario"],
                        "sumocfg": scenario["sumocfg"],
                        "policy": policy,
                        "models": None,
                        "duration_sec": float(duration_sec),
                        "control_interval_sec": int(control_interval_sec),
                        "warmup_sec": float(warmup_sec),
                        "seed": int(seed),
                        "tripinfo_output": str(
                            tripinfo_root
                            / f"{scenario['scenario']}__{policy}__seed{int(seed)}.xml"
                        ),
                    }
                )
    rows = _map_fresh_sumo_processes(
        _evaluate_worker,
        jobs,
        workers=max(int(workers), 1),
    )
    rows.sort(key=lambda row: (row["target"], row["seed"], row["policy"]))
    expected_by_scenario = {
        str(record["scenario"]): record["expected_vehicles"] for record in scenario_records
    }
    errors: list[str] = []
    warnings: list[str] = []
    for row in rows:
        scenario = str(row["target"])
        policy = str(row["policy"])
        metrics = dict(row.get("metrics", {}))
        if metrics.get("ok") is not True:
            errors.append(f"{scenario}/{policy}/seed{row['seed']}: {metrics.get('error', 'failed')}")
            continue
        expected = expected_by_scenario[scenario]
        pending = int(metrics.get("pending_vehicles_at_horizon", 0))
        demand_population = int(
            metrics.get(
                "demand_population_at_horizon",
                int(metrics.get("departed", -1)) + pending,
            )
        )
        if expected is not None and demand_population != int(expected):
            errors.append(
                f"{scenario}/{policy}/seed{row['seed']}: observed demand population "
                f"{demand_population} vehicles; expected {expected}"
            )
        if expected is not None and int(metrics.get("loaded", -1)) != int(expected):
            warnings.append(
                f"{scenario}/{policy}/seed{row['seed']}: SUMO loaded-event count "
                f"{metrics.get('loaded')} differs from route demand {expected}; "
                "admission uses departed plus horizon-pending population"
            )
        collisions = int(metrics.get("collision_events", 0))
        teleports = int(metrics.get("starting_teleports", 0)) + int(
            metrics.get("ending_teleports", 0)
        )
        if policy != "fixed_program" and (collisions or teleports):
            errors.append(
                f"{scenario}/{policy}/seed{row['seed']}: collisions={collisions}, teleports={teleports}"
            )
        elif policy == "fixed_program" and (collisions or teleports):
            warnings.append(
                f"{scenario}/fixed_program/seed{row['seed']}: native action-catalog program is unsafe "
                f"(collisions={collisions}, teleports={teleports})"
            )

    safety_rows = [row for row in rows if row["policy"] != "fixed_program"]
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "experiment": "salt_lake_deterministic_demand_libsumo_admission",
        "runtime": runtime_metadata(),
        "setting": {
            "policies": list(requested_policies),
            "safety_policy": safety_policy,
            "safety_policies": [
                policy for policy in requested_policies if policy != "fixed_program"
            ],
            "seeds": [int(seed) for seed in seeds],
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "process_isolation": "spawn_pool_maxtasksperchild_1",
            "workers": max(int(workers), 1),
        },
        "scenarios": scenario_records,
        "safety_summary": [
            {
                "scenario": row["target"],
                "seed": row["seed"],
                "loaded": row["metrics"].get("loaded"),
                "departed": row["metrics"].get("departed"),
                "arrived": row["metrics"].get("arrived"),
                "pending_at_horizon": row["metrics"].get("pending_vehicles_at_horizon"),
                "demand_population_at_horizon": row["metrics"].get(
                    "demand_population_at_horizon",
                    int(row["metrics"].get("departed", -1))
                    + int(row["metrics"].get("pending_vehicles_at_horizon", 0)),
                ),
                "expected_vehicles": expected_by_scenario[str(row["target"])],
                "active_at_horizon": row["metrics"].get("active_vehicles_at_horizon"),
                "departure_service_ratio": row["metrics"].get("departure_service_ratio"),
                "completion_ratio": row["metrics"].get("completion_ratio"),
                "mean_system_vehicles_per_controlled_lane": row["metrics"].get(
                    "mean_system_vehicles_per_controlled_lane"
                ),
                "collision_events": row["metrics"].get("collision_events"),
                "starting_teleports": row["metrics"].get("starting_teleports"),
                "ending_teleports": row["metrics"].get("ending_teleports"),
                "phase_execution_audit": row["metrics"].get("phase_execution_audit"),
            }
            for row in safety_rows
        ],
        "demand_admission_definition": (
            "cumulative_departed_plus_sumo_pending_insertion_vehicles_at_horizon"
        ),
        "rows": rows,
        "errors": errors,
        "warnings": warnings,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfgs", nargs="+", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=SUPPORTED_POLICIES, default=list(SUPPORTED_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260808])
    parser.add_argument("--duration-sec", type=float, default=3600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--safety-policy", choices=SUPPORTED_POLICIES, default="phase_pressure")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tripinfo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_admission(
        args.sumocfgs,
        policies=args.policies,
        seeds=args.seeds,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        safety_policy=args.safety_policy,
        workers=args.workers,
        tripinfo_root=args.tripinfo_root,
    )
    _atomic_write_json(args.out, result)
    print(json.dumps({key: result[key] for key in ("status", "errors", "warnings", "safety_summary")}, indent=2))
    if result["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
