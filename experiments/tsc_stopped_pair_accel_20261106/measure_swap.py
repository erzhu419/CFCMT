"""Replay a frozen high arm and swap the nominal accel of its two stopped leaders."""
import json
from pathlib import Path

from collect_tsc_physical_response_trial import write_json
from measure_arm import measure_arm as measure_hangzhou
from measure_target import measure_arm as measure_jinan


def set_pair_accel(api, vehicles, lane, length, near_count, native, swapped):
    limit = api.lane.getMaxSpeed(lane)
    ranked = sorted((length - v["lane_position"], vid) for vid, v in vehicles.items()
                    if v["lane_id"] == lane and length - v["lane_position"] <= 10 * limit)
    if api.lane.getLength(lane) != length or len(ranked) != near_count or len(ranked) < 2:
        raise ValueError("Straight lane geometry or near count differs from the original case")
    targets = [vid for _, vid in ranked[:2]]
    if any(vehicles[vid]["speed"] != 0 for vid in targets):
        raise ValueError("The two original straight leaders must both be at rest")
    vehicle = api.vehicle
    before = {vid: vehicle.getAccel(vid) for vid in vehicles}
    observed = [before[vid] for vid in targets]
    if observed != [native, native]:
        raise ValueError(f"Native nominal accelerations {observed} differ from expected {native}")

    original_types = [vehicle.getTypeID(vid) for vid in targets]
    type_accels = {kind: api.vehicletype.getAccel(kind) for kind in original_types}
    for vid in targets:
        vehicle.setAccel(vid, swapped)
    applied = [vehicle.getAccel(vid) for vid in targets]
    if applied != [swapped, swapped]:
        raise ValueError(f"Observed post-set acceleration values {applied} differ from {swapped}")
    if any(vehicle.getAccel(vid) != value for vid, value in before.items() if vid not in targets):
        raise ValueError("Changing the two leaders altered another present vehicle's accel")
    if any(api.vehicletype.getAccel(kind) != value for kind, value in type_accels.items()):
        raise ValueError("A shared vehicle type was changed by the intervention")

    return dict(lane_id=lane, lane_length_m=length, target_ids=targets,
                native_accel_mps2=observed, swapped_accel_mps2=applied,
                native_type_ids=original_types, unaffected_vehicles=len(vehicles)-2)


def measure(city, case, sumocfg, registry_path, archived_high, output, on_branch_start=None):
    from cf_h2o.sumo_runtime import load_libsumo

    native, swapped = {"hangzhou": (2.6, 2.0), "jinan": (2.0, 2.6)}[city]
    original = Path(archived_high)
    out = Path(output)
    baseline = json.loads((original / "arm.json").read_text())
    if baseline != case["high"] or baseline["status"] != "COMPLETE":
        raise ValueError("Archived high arm disagrees with its frozen case")
    saved_start = json.loads((original / "physical_start_server_only.json").read_text())
    observations = json.loads((original / "observations_server_only.json").read_text())
    lane, = observations["catalog"][case["straight_id"]]["incoming_lanes"]
    length = observations["lane_geometry"][lane]["length_m"]

    def swap_at_checkpoint(start):
        if start != saved_start:
            keys = [k for k in start.keys() | saved_start.keys() if start.get(k) != saved_start.get(k)]
            raise ValueError(f"Replay start differs from the archived high arm in {sorted(keys)}")
        details = set_pair_accel(load_libsumo(), start["vehicles"], lane, length,
                                 case["straight_near"], native, swapped)
        write_json(out / "intervention_server_only.json", details)

    kwargs = dict(on_checkpoint=swap_at_checkpoint, on_branch_start=on_branch_start)
    if city == "jinan":
        actual = measure_jinan(case, "high", sumocfg, out, registry_path=registry_path, **kwargs)
    else:
        actual = measure_hangzhou(case, "high", sumocfg, out, **kwargs)
    if any(actual[k] != baseline[k] for k in ("candidate_index", "right_nominal", "straight_nominal")):
        raise ValueError("Swapped and original high arms have different nominal actions")
    return actual
