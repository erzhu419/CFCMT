"""Measure native SUMO collision safety without external signal control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _start_sumo
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-native-sumo-collision-safety-audit-v1"


def audit_native_safety(
    *,
    manifest_path: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    duration_sec: float,
    expected_sumo_version: str,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(str(value) for value in scenarios)
    unknown = set(selected) - set(manifest.sumocfgs)
    if unknown:
        raise ValueError(f"native safety audit has unknown scenarios: {sorted(unknown)}")
    ordered_seeds = tuple(int(value) for value in seeds)
    if not ordered_seeds or len(ordered_seeds) != len(set(ordered_seeds)):
        raise ValueError("native safety audit seeds must be nonempty and unique")
    actual_sumo_version = str(libsumo_version())
    if actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            f"native safety audit SUMO version changed: {actual_sumo_version}"
        )

    sumo = load_libsumo()
    runs = []
    try:
        for scenario in selected:
            for seed in ordered_seeds:
                _start_sumo(sumo, manifest.sumocfgs[scenario], seed)
                begin = float(sumo.simulation.getTime())
                end = begin + float(duration_sec)
                collision_steps = 0
                raw_colliding_vehicle_ids = 0
                incidents: set[tuple[str, ...]] = set()
                first_collision = None
                starting_teleports = 0
                ending_teleports = 0
                while (
                    float(sumo.simulation.getTime()) < end
                    and int(sumo.simulation.getMinExpectedNumber()) > 0
                ):
                    sumo.simulationStep()
                    starting_teleports += int(
                        sumo.simulation.getStartingTeleportNumber()
                    )
                    ending_teleports += int(
                        sumo.simulation.getEndingTeleportNumber()
                    )
                    colliding = tuple(
                        sorted(
                            str(value)
                            for value in sumo.simulation.getCollidingVehiclesIDList()
                        )
                    )
                    if not colliding:
                        continue
                    collision_steps += 1
                    raw_colliding_vehicle_ids += len(colliding)
                    incidents.add(colliding)
                    if first_collision is None:
                        first_collision = {
                            "time_sec": float(sumo.simulation.getTime()),
                            "elapsed_sec": float(sumo.simulation.getTime()) - begin,
                            "vehicle_ids": list(colliding),
                        }
                runs.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "begin_time_sec": begin,
                        "end_time_sec": float(sumo.simulation.getTime()),
                        "collision_event_steps": collision_steps,
                        "raw_colliding_vehicle_ids": raw_colliding_vehicle_ids,
                        "unique_colliding_vehicle_sets": len(incidents),
                        "first_collision": first_collision,
                        "starting_teleports": starting_teleports,
                        "ending_teleports": ending_teleports,
                        "status": "UNSAFE" if collision_steps else "SAFE",
                    }
                )
    finally:
        try:
            sumo.close()
        except Exception:
            pass
    unsafe_count = sum(row["status"] == "UNSAFE" for row in runs)
    status = "UNSAFE" if unsafe_count else "SAFE"
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "native_collision_detected_counterfactual_admission_required"
            if status == "UNSAFE"
            else "native_collision_safety_passed"
        ),
        "sumo_version": actual_sumo_version,
        "manifest": str(Path(manifest_path).resolve()),
        "duration_sec": float(duration_sec),
        "scenarios": list(selected),
        "seeds": list(ordered_seeds),
        "run_count": len(runs),
        "unsafe_run_count": unsafe_count,
        "runs": runs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--expected-sumo-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite native safety audit: {args.out}")
    result = audit_native_safety(
        manifest_path=args.manifest,
        scenarios=args.scenarios,
        seeds=args.seeds,
        duration_sec=args.duration_sec,
        expected_sumo_version=args.expected_sumo_version,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_count": result["run_count"],
                "unsafe_run_count": result["unsafe_run_count"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
