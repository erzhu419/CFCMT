"""Aggregate the frozen three-seed Cologne comparison, including one reused cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


PROTOCOL = "tsc-v154f-cologne-repaired-controller-comparison-v1"
SEEDS = (52282, 41242, 63792)
ARMS = ("rigid_target_only", "phase_pressure")
SERVICE_KEYS = (
    "mean_tripinfo_waiting_time", "system_vehicle_hours", "active_vehicle_hours",
    "pending_vehicle_hours", "mean_tripinfo_depart_delay", "mean_queue",
    "departed", "arrived", "active_vehicles_at_horizon", "pending_vehicles_at_horizon",
)


def reused_cell(result, source_path):
    if result["protocol"] != "tsc-v154e-cologne-bilateral-full-validation-v1" or result["status"] != "PASS":
        raise ValueError("Reuse the passing V154E full result")
    arm = result["repair"]
    return {
        "protocol": PROTOCOL, "arm": "rigid_target_only", "seed": 41242,
        "run_valid": True, "status": "PASS", "zero_incident_status": "PASS",
        "metrics": arm["metrics"], "collision_inventory": arm["collision_inventory"],
        "focal_passages": arm["focal_passages"], "source_bindings": result["source_bindings"],
        "repaired_sumocfg": result["repaired_sumocfg"],
        "source_result": str(source_path), "source_branch": "repair", "task_id": "t90988",
        "reused": True, "elapsed_sec": None,
    }


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    result = {}
    for metric in SERVICE_KEYS:
        rigid = [by_key[seed, ARMS[0]]["metrics"][metric] for seed in seeds]
        pressure = [by_key[seed, ARMS[1]]["metrics"][metric] for seed in seeds]
        differences = [r - p for r, p in zip(rigid, pressure)]
        result[metric] = {
            "rigid_mean": mean(rigid), "phase_pressure_mean": mean(pressure),
            "paired_difference_mean": mean(differences),
            "relative_difference_of_means_percent": 100 * (mean(rigid) / mean(pressure) - 1)
            if mean(pressure) != 0 else None,
            "paired_differences": [{"seed": seed, "rigid_minus_phase_pressure": delta}
                                   for seed, delta in zip(seeds, differences)],
        }
    return {"seeds": list(seeds), "weighting": "equal_seed", "metrics": result}


def aggregate(cells):
    keys = [(row["seed"], row["arm"]) for row in cells]
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("The aggregate requires exactly the six frozen cells, with no duplicates or omissions")
    by_key = dict(zip(keys, cells))
    ordered = [by_key[seed, arm] for seed in SEEDS for arm in ARMS]
    reference = by_key[41242, "rigid_target_only"]
    if any(row["protocol"] != PROTOCOL or row["repaired_sumocfg"] != reference["repaired_sumocfg"]
           or row["source_bindings"] != reference["source_bindings"] for row in ordered):
        raise ValueError("Comparison cells must use the frozen protocol, same repaired network and controller source")
    complete = all(row["run_valid"] for row in ordered)
    safety = {}
    for arm in ARMS:
        rows = [by_key[seed, arm] for seed in SEEDS]
        valid = all(row["run_valid"] for row in rows)
        safety[arm] = {
            "all_seed_runs_valid": valid,
            "zero_collision_status": "UNAVAILABLE" if not valid else
            ("PASS" if all(row["metrics"]["collision_incidents"] == 0 and row["metrics"]["collision_events"] == 0
                           for row in rows) else "FAIL"),
            "zero_collision_seed_count": sum(row["run_valid"] and row["metrics"]["collision_incidents"] == 0
                                             and row["metrics"]["collision_events"] == 0 for row in rows),
            "collision_incidents_total": sum(row["metrics"]["collision_incidents"] for row in rows) if valid else None,
            "collision_events_total": sum(row["metrics"]["collision_events"] for row in rows) if valid else None,
        }
    rows = []
    for row in ordered:
        fields = {key: row[key] for key in ("seed", "arm", "run_valid", "status", "zero_incident_status")}
        fields.update(
            metrics={key: row["metrics"].get(key) for key in (*SERVICE_KEYS, "demand_population_at_horizon",
                     "collision_events", "collision_incidents", "starting_teleports", "ending_teleports")},
            collision_lane_pairs=row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"],
            focal_passages=row["focal_passages"], source_result=row["source_result"],
            reused=row.get("reused", False), task_id=row.get("task_id"), elapsed_sec=row.get("elapsed_sec"),
        )
        rows.append(fields)
    return {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        "cells": rows, "safety_by_arm": safety,
        "all_seed_service": service_summary(ordered, SEEDS) if complete else None,
        "additional_seed_service": service_summary(ordered, (52282, 63792)) if complete else None,
        "paired_collision_differences": [
            {"seed": seed, "rigid_minus_phase_pressure_incidents":
             by_key[seed, ARMS[0]]["metrics"]["collision_incidents"] - by_key[seed, ARMS[1]]["metrics"]["collision_incidents"]}
            for seed in SEEDS] if complete else None,
        "new_simulation_count": sum(not row.get("reused", False) for row in ordered),
        "reused_cell_count": sum(row.get("reused", False) for row in ordered),
        "new_simulation_elapsed_seconds_sum": sum(row.get("elapsed_sec") or 0 for row in ordered if not row.get("reused", False)),
        "repaired_sumocfg": reference["repaired_sumocfg"], "source_bindings": reference["source_bindings"],
        "interpretation": "Keep complete-run validity, per-arm zero-collision outcomes and paired service effects separate. Colliding valid cells remain in every mean. All-three-seed results include the geometry-development seed; the additional-two-seed summary is prespecified. No formal superiority claim from three seeds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--reuse-result", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads(args.launch_record.read_text())
    submitted = json.loads(launch["scheduler_stdout"])["submitted"]
    task_ids = {row["signature"]: row["id"] for row in submitted}
    cells = [reused_cell(json.loads(args.reuse_result.read_text()), args.reuse_result)]
    retrieved_bytes = 0
    for seed in SEEDS:
        for arm in ARMS:
            if (seed, arm) == (41242, "rigid_target_only"):
                continue
            path = args.results_root / f"seed_{seed}" / arm / "result.json"
            row = json.loads(path.read_text())
            if (row["seed"], row["arm"]) != (seed, arm):
                raise ValueError(f"Result cell does not match its frozen path: {path}")
            row.update(source_result=str(path), reused=False,
                       task_id=task_ids[f"CFCMT/v154f/repaired-controller-v1/cologne1/{seed}/{arm}"])
            cells.append(row)
            retrieved_bytes += path.stat().st_size
    result = aggregate(cells)
    result.update(retrieved_result_bytes=retrieved_bytes, launch_record=str(args.launch_record))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"matrix_status": result["matrix_status"], "safety_by_arm": result["safety_by_arm"],
                      "retrieved_result_bytes": retrieved_bytes,
                      "all_seed_service": result["all_seed_service"],
                      "additional_seed_service": result["additional_seed_service"]}, indent=2))


if __name__ == "__main__":
    main()
