"""Bounded read-only replay of the first corrected Cologne rigid collision."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from cf_h2o.eval.traffic_signal_cologne_service_replay import TLS_ID, launch_inputs
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import evaluate_policy_v3
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    RUNTIME_POLICY, StateConditionedSourceUtilityOriginator, _runtime_models,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.right_of_way_context import (
    net_file_from_sumocfg, read_network_right_of_way_context,
)


PROTOCOL = "tsc-v154-cologne-first-collision-read-only-replay-v1"
BEGIN_SEC, END_SEC = 25200, 25666
TRACE_BEGIN_SEC, TRACE_END_SEC = 25630, 25665
EXPECTED_TIME_SEC = 25657
FOCAL_IDS = ("102219_396_0", "129962_409_0")
EXPECTED_LANE = f":{TLS_ID}_1_1"
STAGES = (
    "before_simulation_step", "after_simulation_step_before_executor",
    "after_executor_advance",
)
PASSENGER_SHAPES = {
    "passenger", "passenger/sedan", "passenger/hatchback", "passenger/wagon",
    "passenger/van",
}


def static_geometry(net_file: Path) -> dict[str, Any]:
    root = ET.parse(net_file).getroot()
    prefix = f":{TLS_ID}_"
    return {
        "net_file": str(net_file), "net_attributes": dict(root.attrib),
        "internal_edges": [
            {"attrs": dict(edge.attrib), "lanes": [dict(lane.attrib) for lane in edge.findall("lane")]}
            for edge in root.findall("edge") if edge.get("id", "").startswith(prefix)
        ],
        "junctions": [
            {"attrs": dict(junction.attrib), "requests": [dict(row.attrib) for row in junction.findall("request")]}
            for junction in root.findall("junction")
            if junction.get("id") == TLS_ID or junction.get("id", "").startswith(prefix)
        ],
        "connections": [dict(row.attrib) for row in root.findall("connection")
                        if row.get("tl") == TLS_ID or row.get("from", "").startswith(prefix)
                        or row.get("to", "").startswith(prefix)],
    }


def passenger_polygon(front, back, width: float) -> list[list[float]]:
    """SUMO 1.22 MSVehicle::getBoundingPoly passenger shape, zero extra margin."""
    dx, dy = back[0] - front[0], back[1] - front[1]
    norm = math.hypot(dx, dy)
    nx, ny = dy / norm, -dx / norm
    # The full-width sides use 80% of the front-to-back centre line.
    result = []
    for along, side in ((0, .3), (.1, .5), (.9, .5), (1, .3),
                        (1, -.3), (.9, -.5), (.1, -.5), (0, -.3)):
        result.append([front[0] + along * dx + side * width * nx,
                       front[1] + along * dy + side * width * ny])
    return result


def polygon_axis_penetration(a, b) -> float:
    """Positive overlap on every separating axis; negative means separation."""
    margins = []
    for polygon in (a, b):
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = math.hypot(dx, dy)
            axis = (-dy / length, dx / length)
            pa = [v[0] * axis[0] + v[1] * axis[1] for v in a]
            pb = [v[0] * axis[0] + v[1] * axis[1] for v in b]
            margins.append(min(max(pa), max(pb)) - max(min(pa), min(pb)))
    return min(margins)


def vehicle_pose(api, vehicle_id: str) -> dict[str, Any]:
    vehicle = api.vehicle
    lane = str(vehicle.getLaneID(vehicle_id))
    road = str(vehicle.getRoadID(vehicle_id))
    position = float(vehicle.getLanePosition(vehicle_id))
    length, width = float(vehicle.getLength(vehicle_id)), float(vehicle.getWidth(vehicle_id))
    lateral = float(vehicle.getLateralLanePosition(vehicle_id))
    front = list(map(float, vehicle.getPosition(vehicle_id)))
    shape = str(api.vehicletype.getShapeClass(vehicle.getTypeID(vehicle_id)))
    row = {
        "id": vehicle_id, "type_id": str(vehicle.getTypeID(vehicle_id)), "shape_class": shape,
        "lane_id": lane, "road_id": road, "lane_index": int(vehicle.getLaneIndex(vehicle_id)),
        "lane_position_m": position, "lateral_position_m": lateral,
        "position_xy_m": front, "angle_deg": float(vehicle.getAngle(vehicle_id)),
        "speed_mps": float(vehicle.getSpeed(vehicle_id)),
        "acceleration_mps2": float(vehicle.getAcceleration(vehicle_id)),
        "waiting_sec": float(vehicle.getWaitingTime(vehicle_id)),
        "length_m": length, "width_m": width,
        "route": list(vehicle.getRoute(vehicle_id)), "route_index": int(vehicle.getRouteIndex(vehicle_id)),
        "next_tls": [[str(tls), int(link), float(distance), str(state)]
                     for tls, link, distance, state in vehicle.getNextTLS(vehicle_id)],
        "back_position_xy_m": None, "passenger_polygon_xy_m": None,
        "polygon_unavailable_reason": None,
    }
    if shape not in PASSENGER_SHAPES:
        row["polygon_unavailable_reason"] = "non_passenger_shape"
    elif position < length:
        row["polygon_unavailable_reason"] = "rear_on_preceding_lane"
    elif lateral != 0.0:
        row["polygon_unavailable_reason"] = "nonzero_lateral_position"
    else:
        # This is the native getBackPosition branch for a vehicle fully on its
        # current lane. convert2D uses that same geometryPositionAtOffset call.
        back = list(map(float, api.simulation.convert2D(
            road, position - length, row["lane_index"], False,
        )))
        row["back_position_xy_m"] = back
        row["passenger_polygon_xy_m"] = passenger_polygon(front, back, width)
    return row


class CollisionObserver:
    def __init__(self, api, geometry):
        self.api = api
        self.internal_lanes = sorted(lane["id"] for edge in geometry["internal_edges"] for lane in edge["lanes"])
        self.lane_catalog = {}
        self.samples = []

    def __call__(self, *, stage: str, time_sec: float, executors):
        if not TRACE_BEGIN_SEC <= time_sec <= TRACE_END_SEC:
            return
        api = self.api
        if not self.lane_catalog:
            for lane in self.internal_lanes:
                self.lane_catalog[lane] = {
                    "length_m": float(api.lane.getLength(lane)), "width_m": float(api.lane.getWidth(lane)),
                    "shape_xy_m": [list(map(float, point)) for point in api.lane.getShape(lane)],
                }
        lane_vehicles = {lane: sorted(api.lane.getLastStepVehicleIDs(lane)) for lane in self.internal_lanes}
        active = set(api.vehicle.getIDList())
        observed_ids = set(FOCAL_IDS) & active
        observed_ids.update(vehicle for vehicles in lane_vehicles.values() for vehicle in vehicles)
        vehicles = {vehicle: vehicle_pose(api, vehicle) for vehicle in sorted(observed_ids)}
        polygons = [vehicles.get(vehicle, {}).get("passenger_polygon_xy_m") for vehicle in FOCAL_IDS]
        collisions = None
        if stage == "after_simulation_step_before_executor":
            collisions = [{name: getattr(event, name) for name in (
                "collider", "victim", "colliderType", "victimType", "colliderSpeed",
                "victimSpeed", "type", "lane", "pos",
            )} for event in api.simulation.getCollisions()]
        self.samples.append({
            "time_sec": time_sec, "stage": stage,
            "tls_state": str(api.trafficlight.getRedYellowGreenState(TLS_ID)),
            "executor": executors[TLS_ID].snapshot(),
            "active_inserted_vehicles": int(api.vehicle.getIDCount()),
            "pending_insertion_vehicles": len(api.simulation.getPendingVehicles()),
            "lane_vehicles": lane_vehicles,
            "vehicles": vehicles, "native_collisions": collisions,
            "focal_polygon_minimum_axis_penetration_m": (
                polygon_axis_penetration(*polygons) if all(polygons) else None
            ),
        })


def exact_prefix(reference: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected = [row for row in reference["metrics"]["accepted_intervention_trace"] if row["time_sec"] < END_SEC]
    actual = metrics.get("accepted_intervention_trace", [])
    return {"passed": expected == actual and bool(expected), "expected_count": len(expected),
            "actual_count": len(actual), "comparison": "all_retained_fields_exact_no_tolerance"}


def run_replay(launch_record: Path, reference_result: Path, tripinfo: Path) -> dict[str, Any]:
    started = time.monotonic()
    options = launch_inputs(json.loads(launch_record.read_text()))
    reference = json.loads(reference_result.read_text())
    if (reference["city"], reference["scenario"], reference["seed"], reference["arm"], reference["feature_binding"]) != (
        "cologne", "cologne1", 41242, "rigid_target_only", "stored_feature_names",
    ):
        raise ValueError("Reference must be the corrected t90756 Cologne rigid rollout")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = load_traffic_signal_manifest(Path(options["--manifest"]))
    sumocfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    net_file = net_file_from_sumocfg(sumocfg)
    geometry = static_geometry(net_file)
    originator = StateConditionedSourceUtilityOriginator.load(
        Path(options["--runtime-model"]), expected_sha256=options["--runtime-model-sha256"],
        expected_city="cologne", network_context=read_network_right_of_way_context(net_file), arm="rigid_target_only",
    )
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("The original SUMO 1.22.0 runtime is required")
    observer = CollisionObserver(api, geometry)
    if "CFCMT_SCRATCH" not in tripinfo.parts or tripinfo.exists():
        raise ValueError("Use a fresh remote CFCMT_SCRATCH tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario="cologne1", policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=450),
            duration_sec=END_SEC - BEGIN_SEC, control_interval_sec=10, warmup_sec=60, seed=41242,
            tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0, step_observer=observer,
        )
    finally:
        tripinfo.unlink(missing_ok=True)
    expected_collisions = [row for row in reference["metrics"]["collision_samples"] if row["time"] <= END_SEC]
    observed_collisions = metrics.get("collision_samples", [])
    first = observed_collisions[0] if observed_collisions else {}
    prefix = exact_prefix(reference, metrics)
    times = {(row["time_sec"], row["stage"]) for row in observer.samples}
    checks = {
        "runner_completed": bool(metrics.get("ok")),
        "exact_original_intervention_prefix": prefix["passed"],
        "exact_retained_collision_samples": expected_collisions == observed_collisions and bool(expected_collisions),
        "expected_first_collision": (first.get("time"), first.get("collider"), first.get("victim"), first.get("lane"))
        == (EXPECTED_TIME_SEC, *FOCAL_IDS, EXPECTED_LANE),
        "three_stage_window_complete": times == {(t, stage) for t in range(TRACE_BEGIN_SEC, TRACE_END_SEC + 1) for stage in STAGES}
        and len(observer.samples) == 3 * (TRACE_END_SEC - TRACE_BEGIN_SEC + 1),
        "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
    }
    endpoint_keys = ("ok", "error", "departed", "arrived", "active_vehicles_at_horizon", "pending_vehicles_at_horizon",
                     "collision_events", "collision_incidents", "starting_teleports", "ending_teleports", "phase_execution_audit")
    return {
        "protocol": PROTOCOL, "status": "PASS" if all(checks.values()) else "REPRODUCTION_FAILED", "checks": checks,
        "reference_result": str(reference_result), "launch_record": str(launch_record), "original_runtime_arguments": options,
        "sumo_version": libsumo_version(), "elapsed_sec": time.monotonic() - started,
        "simulation_window_sec": [BEGIN_SEC, END_SEC], "observation_window_sec_inclusive": [TRACE_BEGIN_SEC, TRACE_END_SEC],
        "expected_first_collision": {"time_sec": EXPECTED_TIME_SEC, "collider": FOCAL_IDS[0], "victim": FOCAL_IDS[1], "lane": EXPECTED_LANE},
        "intervention_prefix": prefix, "retained_collision_samples": observed_collisions,
        "endpoint": {key: metrics[key] for key in endpoint_keys if key in metrics},
        "static_geometry": geometry, "native_lane_catalog": observer.lane_catalog, "samples": observer.samples,
        "semantics": {
            "PASS": "unchanged failing trajectory reproduced and observed; not safety admission",
            "native_collisions": "queried only immediately after simulationStep; null at other stages avoids duplicate counting",
            "before_simulation_step": "after any control decision/request at this time; signal used for the following simulation step",
            "passenger_polygon": "SUMO1.22 zero-margin passenger octagon from actual front and native converted back; unavailable if rear crosses a lane or lateral position is nonzero",
            "axis_penetration": "derived convex polygon overlap margin, not a replacement for native collision records",
            "physics": "original controller/model, seed, collision warn, demand, vehicle and executor parameters retained",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", required=True, type=Path)
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--tripinfo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_replay(args.launch_record, args.reference_result, args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "elapsed_sec": result["elapsed_sec"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
