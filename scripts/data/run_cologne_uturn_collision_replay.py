"""Read-only replay of the V154F straight/U-turn incident, with a 46-second trace."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location(
    "cologne_v154f_tooling", Path(__file__).with_name("run_cologne_repaired_controller_comparison.py"))
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)
full = comparison.full
TLS = full.short.TLS
PROTOCOL = "tsc-v154g-cologne-uturn-collision-replay-v1"
BEGIN, END = 25200, 27486
TRACE_BEGIN, TRACE_END = 27440, 27485
IDS = ("200684_437_0", "201382_438_0")
PATHS = {
    IDS[0]: ["27115123#3_1", f":{TLS}_16_1", "32324544#0_1"],
    IDS[1]: ["23429231#1_1", f":{TLS}_9_0", f":{TLS}_23_0", "32324544#0_1"],
}
FOE_FIELDS = ("foeId", "egoDist", "foeDist", "egoExitDist", "foeExitDist",
              "egoLane", "foeLane", "egoResponse", "foeResponse")
# SUMO 1.22 libsumo parseConnectionList order (differs from the TraCI docstring).
LINK_FIELDS = ("lane", "priority", "opened", "foe", "via", "state", "direction", "length")


def complete_rear_pose(api, runtime, pose, path):
    """SUMO getBackPosition along this fixed route; verify its native heading."""
    pose = dict(pose)
    if pose["polygon_unavailable_reason"] != "rear_on_preceding_lane":
        return pose
    if pose["lateral_position_m"] != 0 or pose["lane_id"] not in path:
        return pose
    index = path.index(pose["lane_id"])
    offset = pose["lane_position_m"] - pose["length_m"]
    while offset < 0 and index > 0:
        index -= 1
        offset += float(api.lane.getLength(path[index]))
    if offset < 0:
        return pose
    edge, lane_index = path[index].rsplit("_", 1)
    back = list(map(float, api.simulation.convert2D(edge, offset, int(lane_index), False)))
    front = pose["position_xy_m"]
    heading = math.degrees(math.atan2(front[0] - back[0], front[1] - back[1])) % 360
    error = abs((heading - pose["angle_deg"] + 180) % 360 - 180)
    pose["rear_reconstruction"] = {"lane_id": path[index], "lane_position_m": offset,
        "position_xy_m": back, "heading_error_deg": error,
        "basis": "Fixed legal preceding lanes and SUMO logical lane offsets; checked against native vehicle heading."}
    if error > 1e-5:
        pose["polygon_unavailable_reason"] = "preceding_path_native_heading_mismatch"
        return pose
    pose.update(back_position_xy_m=back,
                passenger_polygon_xy_m=runtime.passenger_polygon(front, back, pose["width_m"]),
                polygon_unavailable_reason=None)
    return pose


class WindowObserver:
    def __init__(self, api, runtime):
        self.api, self.runtime = api, runtime
        self.samples, self.lane_catalog, self.parameters = [], {}, {}

    def __call__(self, *, stage, time_sec, executors):
        if not TRACE_BEGIN <= time_sec <= TRACE_END:
            return
        api, vehicles = self.api, {}
        if not self.lane_catalog:
            self.lane_catalog = {lane: {"length_m": api.lane.getLength(lane),
                "width_m": api.lane.getWidth(lane), "shape_xy_m": api.lane.getShape(lane)}
                for lane in sorted({lane for path in PATHS.values() for lane in path})}
        active = set(api.vehicle.getIDList())
        for vehicle in IDS:
            if vehicle not in active:
                continue
            pose = complete_rear_pose(api, self.runtime, self.runtime.vehicle_pose(api, vehicle), PATHS[vehicle])
            pose["junction_foes_100m"] = [dict(zip(FOE_FIELDS, row))
                for row in api.vehicle.getJunctionFoes(vehicle, 100.)]
            pose["next_links"] = [dict(zip(LINK_FIELDS, row)) for row in api.vehicle.getNextLinks(vehicle)]
            pose["leader"] = api.vehicle.getLeader(vehicle, 100.)
            pose["speed_without_traci_mps"] = api.vehicle.getSpeedWithoutTraCI(vehicle)
            if vehicle not in self.parameters:
                self.parameters[vehicle] = {name: getattr(api.vehicle, getter)(vehicle)
                    for name, getter in (("min_gap_m", "getMinGap"), ("tau_sec", "getTau"),
                        ("accel_mps2", "getAccel"), ("decel_mps2", "getDecel"),
                        ("emergency_decel_mps2", "getEmergencyDecel"), ("speed_mode", "getSpeedMode"))}
            vehicles[vehicle] = pose
        polygons = [vehicles.get(vehicle, {}).get("passenger_polygon_xy_m") for vehicle in IDS]
        self.samples.append({"time_sec": time_sec, "stage": stage,
            "tls_state": api.trafficlight.getRedYellowGreenState(TLS),
            "executor": executors[TLS].snapshot(), "vehicles": vehicles,
            "lane_vehicles": {lane: list(api.lane.getLastStepVehicleIDs(lane)) for lane in self.lane_catalog},
            "body_axis_penetration_m": self.runtime.polygon_axis_penetration(*polygons) if all(polygons) else None})


def replay_checks(reference, inventory, metrics, samples, step_times):
    expected_reports = [row for row in reference["metrics"]["collision_samples"] if row["time"] <= END]
    times = [(row["time_sec"], row["stage"]) for row in samples]
    expected_times = {(t, stage) for t in range(TRACE_BEGIN, TRACE_END + 1) for stage in full.short.STAGES}
    return {
        "runner_completed": bool(metrics.get("ok")),
        "original_incident_inventory_exact": inventory == reference["collision_inventory"],
        "original_capped_collision_reports_exact": metrics.get("collision_samples") == expected_reports,
        "full_prefix_steps": step_times == list(range(BEGIN + 1, END + 1)),
        "three_stage_window_complete": len(times) == len(expected_times) and set(times) == expected_times,
        "native_counts_agree": (inventory["native_event_count"], inventory["native_incident_count"], inventory["collision_event_steps"])
            == (metrics.get("collision_events"), metrics.get("collision_incidents"), metrics.get("collision_event_steps")),
        "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
    }


def run_replay(reference_result, geometry_reference_result, tripinfo):
    started = time.monotonic()
    reference = json.loads(reference_result.read_text())
    geometry_reference = json.loads(geometry_reference_result.read_text())
    if (reference["protocol"], reference["arm"], reference["seed"], reference["run_valid"]) != (
            comparison.PROTOCOL, "phase_pressure", 52282, True):
        raise ValueError("Use the retained V154F PhasePressure/52282 result")
    runtime, bindings = comparison._load_runtime()
    if bindings != reference["source_bindings"]:
        raise ValueError("Controller source differs from the retained run")
    sumocfg = Path(reference["repaired_sumocfg"])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != geometry_reference["bilateral_static_geometry"]:
        raise ValueError("Use the unchanged bilateral geometry")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = geometry_reference["original_runtime_arguments"]["--conversion-root"]
    api = runtime.load_libsumo()
    if "1.22.0" not in str(runtime.libsumo_version()):
        raise ValueError("Use the retained SUMO 1.22.0")
    v3 = importlib.import_module("cf_h2o.eval.traffic_signal_resco_cfcmt_v3")
    detailed = WindowObserver(api, runtime)
    old_routes = {vehicle: geometry_reference["focal_demand"]["vehicles"][vehicle]["route"] for vehicle in full.FOCAL_IDS}
    observer = full.FullObserver(api, detailed, old_routes, runtime, v3, geometry)
    if tripinfo.exists() or "CFCMT_SCRATCH" not in tripinfo.parts:
        raise ValueError("Use a fresh server scratch tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario="cologne1",
            policy="phase_pressure", models=None, duration_sec=END-BEGIN, control_interval_sec=10,
            warmup_sec=60, seed=52282, tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0, step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    inventory = observer.inventory()
    checks = replay_checks(reference, inventory, metrics, detailed.samples, observer.step_times)
    return {"protocol": PROTOCOL, "status": "PASS" if all(checks.values()) else "REPLAY_FAILED",
        "checks": checks, "reference_result": str(reference_result),
        "reference_safety_status": reference["status"], "geometry_reference_result": str(geometry_reference_result),
        "repaired_sumocfg": str(sumocfg), "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
        "simulation_window_sec": [BEGIN, END], "trace_window_sec": [TRACE_BEGIN, TRACE_END],
        "seed": 52282, "arm": "phase_pressure", "metrics": metrics,
        "collision_inventory": inventory, "samples": detailed.samples, "focal_paths": PATHS,
        "native_lane_catalog": detailed.lane_catalog, "vehicle_parameters": detailed.parameters,
        "elapsed_sec": time.monotonic() - started,
        "geometry_method_source": "https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L1455",
        "interpretation": "PASS means exact reproduction and trace completeness. The V154F collision FAIL is retained; no geometry or controller change."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-result", "geometry-reference-result", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_replay(args.reference_result, args.geometry_reference_result, args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks", "elapsed_sec")}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
