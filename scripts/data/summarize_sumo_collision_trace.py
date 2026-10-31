#!/usr/bin/env python3
"""Summarize a bounded read-only SUMO collision trace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "sumo-collision-trace-summary-v2"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vehicle_rows(
    samples: Sequence[Mapping[str, Any]], vehicle_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        time_sec = _finite_float(sample.get("time_sec"))
        for vehicle in sample.get("vehicles", ()):
            vehicle = dict(vehicle)
            if str(vehicle.get("vehicle_id", "")) != str(vehicle_id):
                continue
            if vehicle.get("present") is not True:
                continue
            rows.append({"time_sec": time_sec, **vehicle})
            break
    return rows


def _transitions(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous: Any = None
    initialized = False
    for row in rows:
        current = row.get(field)
        if initialized and current != previous:
            transitions.append(
                {
                    "time_sec": row.get("time_sec"),
                    "from": previous,
                    "to": current,
                }
            )
        previous = current
        initialized = True
    return transitions


def _participant_summary(
    samples: Sequence[Mapping[str, Any]], vehicle_id: str
) -> dict[str, Any]:
    rows = _vehicle_rows(samples, vehicle_id)
    speeds = [
        number
        for row in rows
        if (number := _finite_float(row.get("Speed"))) is not None
    ]
    accelerations = [
        number
        for row in rows
        if (number := _finite_float(row.get("Acceleration"))) is not None
    ]
    declared_decels = [
        number
        for row in rows
        if (number := _finite_float(row.get("Decel"))) is not None
    ]
    emergency_decels = [
        number
        for row in rows
        if (number := _finite_float(row.get("EmergencyDecel"))) is not None
    ]
    lane_ids = [str(row.get("LaneID")) for row in rows if row.get("LaneID")]
    road_ids = [str(row.get("RoadID")) for row in rows if row.get("RoadID")]
    minimum_acceleration = min(accelerations) if accelerations else None
    declared_decel = min(declared_decels) if declared_decels else None
    emergency_decel = min(emergency_decels) if emergency_decels else None
    return {
        "vehicle_id": str(vehicle_id),
        "present_sample_count": len(rows),
        "first_present_time_sec": rows[0].get("time_sec") if rows else None,
        "last_present_time_sec": rows[-1].get("time_sec") if rows else None,
        "observed_lane_ids": list(dict.fromkeys(lane_ids)),
        "observed_road_ids": list(dict.fromkeys(road_ids)),
        "lane_transitions": _transitions(rows, "LaneID"),
        "road_transitions": _transitions(rows, "RoadID"),
        "minimum_speed_mps": min(speeds) if speeds else None,
        "maximum_speed_mps": max(speeds) if speeds else None,
        "minimum_acceleration_mps2": minimum_acceleration,
        "maximum_braking_mps2": (
            -minimum_acceleration if minimum_acceleration is not None else None
        ),
        "declared_decel_mps2": declared_decel,
        "declared_emergency_decel_mps2": emergency_decel,
        "braking_exceeded_declared_decel": (
            minimum_acceleration is not None
            and declared_decel is not None
            and minimum_acceleration < -declared_decel - 1e-9
        ),
        "braking_reached_emergency_decel": (
            minimum_acceleration is not None
            and emergency_decel is not None
            and minimum_acceleration <= -emergency_decel + 1e-9
        ),
    }


def summarize_collision_trace(
    payload: Mapping[str, Any],
    *,
    collider_id: str,
    victim_id: str,
    watched_lane_ids: Sequence[str] = (),
) -> dict[str, Any]:
    admission = dict(payload.get("admission_result", payload))
    observed = dict(admission.get("observed", {}))
    trace = dict(observed.get("diagnostic_trace", {}))
    samples = [dict(value) for value in trace.get("samples", ())]
    if not samples:
        raise ValueError("diagnostic trace contains no samples")
    collider_rows = _vehicle_rows(samples, collider_id)
    pair_rows: list[dict[str, Any]] = []
    for row in collider_rows:
        leader = row.get("LeaderWithin100m")
        if not isinstance(leader, (list, tuple)) or len(leader) < 2:
            continue
        if str(leader[0]) != str(victim_id):
            continue
        raw_clearance = _finite_float(leader[1])
        min_gap = _finite_float(row.get("MinGap"))
        # SUMO getLeader excludes the follower's minGap from the bumper gap.
        bumper_gap = (
            raw_clearance + min_gap
            if raw_clearance is not None and min_gap is not None
            else None
        )
        pair_rows.append(
            {
                "time_sec": row.get("time_sec"),
                "raw_leader_clearance_m": raw_clearance,
                "bumper_gap_m": bumper_gap,
                "collider_min_gap_m": min_gap,
                "below_min_gap": (
                    bumper_gap is not None
                    and min_gap is not None
                    and bumper_gap < min_gap
                ),
            }
        )
    finite_bumper_gaps = [
        float(row["bumper_gap_m"])
        for row in pair_rows
        if row["bumper_gap_m"] is not None
    ]
    finite_raw_clearances = [
        float(row["raw_leader_clearance_m"])
        for row in pair_rows
        if row["raw_leader_clearance_m"] is not None
    ]

    watched: dict[str, Any] = {}
    participants = {str(collider_id), str(victim_id)}
    for lane_id in watched_lane_ids:
        occupancy = [
            {
                "time_sec": sample.get("time_sec"),
                "vehicle_ids": list(
                    dict(sample.get("vehicle_ids_by_lane", {})).get(str(lane_id), ())
                ),
            }
            for sample in samples
        ]
        watched[str(lane_id)] = {
            "occupied_sample_count": sum(bool(row["vehicle_ids"]) for row in occupancy),
            "nonparticipant_vehicle_ids": sorted(
                {
                    str(vehicle_id)
                    for row in occupancy
                    for vehicle_id in row["vehicle_ids"]
                    if str(vehicle_id) not in participants
                }
            ),
            "final_vehicle_ids": occupancy[-1]["vehicle_ids"],
        }

    collision_samples = list(observed.get("collision_samples", ()))
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_protocol": trace.get("protocol"),
        "trace_read_only": trace.get("read_only"),
        "trace_sample_count": len(samples),
        "trace_time_range_sec": [samples[0].get("time_sec"), samples[-1].get("time_sec")],
        "collision_samples": collision_samples,
        "participants": {
            "collider": _participant_summary(samples, collider_id),
            "victim": _participant_summary(samples, victim_id),
        },
        "longitudinal_pair": {
            "distance_definition": (
                "SUMO getLeader returns raw_leader_clearance_m, the distance "
                "from the follower front plus its minGap to the leader rear. "
                "bumper_gap_m = raw_leader_clearance_m + collider_min_gap_m."
            ),
            "collider_followed_victim_sample_count": len(pair_rows),
            "first_bumper_gap_m": finite_bumper_gaps[0] if finite_bumper_gaps else None,
            "final_bumper_gap_m": finite_bumper_gaps[-1] if finite_bumper_gaps else None,
            "minimum_bumper_gap_m": min(finite_bumper_gaps) if finite_bumper_gaps else None,
            "first_raw_leader_clearance_m": (
                finite_raw_clearances[0] if finite_raw_clearances else None
            ),
            "final_raw_leader_clearance_m": (
                finite_raw_clearances[-1] if finite_raw_clearances else None
            ),
            "minimum_raw_leader_clearance_m": (
                min(finite_raw_clearances) if finite_raw_clearances else None
            ),
            "below_min_gap_times_sec": [
                row["time_sec"] for row in pair_rows if row["below_min_gap"]
            ],
            "samples": pair_rows,
        },
        "watched_lanes": watched,
        "interpretation_boundary": (
            "This summary reports observed kinematics and occupancy. Braking, "
            "lane transitions, or concurrent foe-lane occupancy do not by "
            "themselves identify the simulator's internal collision cause."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--collider-id", required=True)
    parser.add_argument("--victim-id", required=True)
    parser.add_argument("--watched-lane-id", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.out}")
    payload = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    result = summarize_collision_trace(
        payload,
        collider_id=args.collider_id,
        victim_id=args.victim_id,
        watched_lane_ids=args.watched_lane_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
