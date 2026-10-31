"""Validate the bilateral Cologne waiting split with the frozen original runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET


# Load the tooling sibling without adding its repository root to PYTHONPATH:
# cf_h2o must continue to resolve from the separately frozen source 85 snapshot.
_spec = importlib.util.spec_from_file_location(
    "cologne_v154c_validation_tooling",
    Path(__file__).with_name("run_cologne_waiting_geometry_validation.py"),
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

PROTOCOL = "tsc-v154d-cologne-bilateral-waiting-geometry-validation-v1"
MIRROR_REFERENCE_PROTOCOL = base.PROTOCOL
BEGIN, END = base.BEGIN, base.END
TRACE_BEGIN, TRACE_END = base.TRACE_BEGIN, base.TRACE_END
STAGES, TLS = base.STAGES, base.TLS
ORIGINAL_IDS = base.FOCAL_IDS
MIRROR_IDS = ("121463_406_0", "168358_425_0")
FOCAL_IDS = ORIGINAL_IDS + MIRROR_IDS
FOCAL_LANES = {
    **base.FOCAL_LANES,
    MIRROR_IDS[0]: (f":{TLS}_11_1",),
    MIRROR_IDS[1]: (f":{TLS}_3_0", f":{TLS}_20_0"),
}
WAITING_STRAIGHT_LANES = {
    f":{TLS}_13_0": FOCAL_LANES[ORIGINAL_IDS[0]][0],
    f":{TLS}_3_0": FOCAL_LANES[MIRROR_IDS[0]][0],
}
_load_runtime = base._load_runtime


class FocalPassageTracker:
    def __init__(self, routes):
        self.routes = routes
        self.transitions = {vehicle: [] for vehicle in FOCAL_IDS}
        self.route_matches = {vehicle: True for vehicle in FOCAL_IDS}

    def observe(self, api, time_sec):
        active = set(api.vehicle.getIDList())
        for vehicle in FOCAL_IDS:
            if vehicle not in active:
                continue
            self.route_matches[vehicle] &= list(api.vehicle.getRoute(vehicle)) == self.routes[vehicle]
            lane = str(api.vehicle.getLaneID(vehicle))
            records = self.transitions[vehicle]
            if not records or records[-1]["lane_id"] != lane:
                records.append({"time_sec": time_sec, "lane_id": lane,
                                "road_id": str(api.vehicle.getRoadID(vehicle)),
                                "route_index": int(api.vehicle.getRouteIndex(vehicle)),
                                "lane_position_m": float(api.vehicle.getLanePosition(vehicle))})

    def summary(self):
        result = {}
        for vehicle, records in self.transitions.items():
            incoming, outgoing = self.routes[vehicle]
            required = [incoming, *FOCAL_LANES[vehicle], outgoing]
            milestones, next_required = [], 0
            for row in records:
                if next_required == len(required):
                    break
                field = "road_id" if next_required in (0, len(required) - 1) else "lane_id"
                if row[field] == required[next_required]:
                    milestones.append(row)
                    next_required += 1
            result[vehicle] = {
                "seen": bool(records), "original_route": self.routes[vehicle],
                "route_unchanged": self.route_matches[vehicle],
                "required_incoming_internal_outgoing_sequence": required,
                "matched_milestones": milestones, "lane_transitions": records,
                "passed_controlled_junction": next_required == len(required) and self.route_matches[vehicle],
            }
        return result


class ValidationObserver:
    def __init__(self, api, detailed, routes):
        self.api, self.detailed = api, detailed
        self.tracker = FocalPassageTracker(routes)
        self.step_times = []

    def __call__(self, *, stage, time_sec, executors):
        self.detailed(stage=stage, time_sec=time_sec, executors=executors)
        if stage == STAGES[1]:
            self.step_times.append(time_sec)
            self.tracker.observe(self.api, time_sec)


def repair_checks(arm):
    metrics = arm["metrics"]
    return {
        "runner_completed": bool(metrics.get("ok")), **base.window_checks(arm),
        "no_native_collisions_in_full_window": metrics.get("collision_events") == 0
        and metrics.get("collision_incidents") == 0,
        "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
        "original_pair_passed_controlled_junction": all(
            arm["focal_passages"][vehicle]["passed_controlled_junction"] for vehicle in ORIGINAL_IDS),
        "mirror_pair_passed_controlled_junction": all(
            arm["focal_passages"][vehicle]["passed_controlled_junction"] for vehicle in MIRROR_IDS),
    }


def waiting_body_evidence(arm, through_widths):
    """Actual body-to-strip distances, retaining speed and unavailable bodies."""
    result = {}
    for waiting_lane, straight_lane in WAITING_STRAIGHT_LANES.items():
        centerline = arm["native_lane_catalog"][straight_lane]["shape_xy_m"]
        width = through_widths[straight_lane]
        observations = []
        for sample in arm["samples"]:
            if sample["stage"] != STAGES[1]:
                continue
            for vehicle_id, vehicle in sample["vehicles"].items():
                if vehicle["lane_id"] != waiting_lane:
                    continue
                polygon = vehicle["passenger_polygon_xy_m"]
                row = {
                    "time_sec": sample["time_sec"], "vehicle_id": vehicle_id,
                    "lane_position_m": vehicle["lane_position_m"],
                    "speed_mps": vehicle["speed_mps"],
                    "stationary_at_sample": vehicle["speed_mps"] <= 1e-8,
                    "length_m": vehicle["length_m"], "width_m": vehicle["width_m"],
                    "shape_class": vehicle["shape_class"],
                    "passenger_polygon_xy_m": polygon,
                    "polygon_unavailable_reason": vehicle["polygon_unavailable_reason"],
                }
                if polygon is not None:
                    row.update(base.body_strip_clearance(polygon, centerline, width))
                observations.append(row)
        available = [row for row in observations if row.get("body_within_straight_lane_span")]
        stationary = [row["clearance_m"] for row in available if row["stationary_at_sample"]]
        result[waiting_lane] = {
            "opposing_straight_lane": straight_lane,
            "through_vehicle_width_m": width, "through_width_source": "original demand vehicle type",
            "observation_count": len(observations),
            "body_reconstructable_count": sum(row["passenger_polygon_xy_m"] is not None for row in observations),
            "minimum_clearance_m": min((row["clearance_m"] for row in available), default=None),
            "stationary_minimum_clearance_m": min(stationary, default=None),
            "observations": observations,
        }
    return {"waiting_lanes": result, "hard_gate": False,
            "interpretation": "Auxiliary measured vehicle bodies in the frozen detailed window. Speed is retained; a moving or nearly stationary sample is not stationary-body evidence. Missing reconstructable bodies do not establish clearance."}


def read_focal_demand(sumocfg):
    cfg = ET.parse(sumocfg).getroot()
    route_value = cfg.find("input/route-files").get("value")
    route_path = (sumocfg.parent / route_value).resolve()
    root = ET.parse(route_path).getroot()
    types = {row.get("id"): dict(row.attrib) for row in root.findall("vType")}
    vehicles = {row.get("id"): row for row in root.findall("trip") if row.get("id") in FOCAL_IDS}
    routes, demand = {}, {}
    for vehicle in FOCAL_IDS:
        row = vehicles[vehicle]
        routes[vehicle] = [row.get("from"), row.get("to")]
        type_row = types[row.get("type")]
        demand[vehicle] = {"vehicle": dict(row.attrib), "vehicle_type": type_row, "route": routes[vehicle]}
    # These four fixed trips traverse this one junction with two-edge routes;
    # the passage tracker checks SUMO's complete actual route in each arm.
    return routes, {"route_file": str(route_path), "vehicles": demand}


def _run_arm(runtime, options, sumocfg, routes, tripinfo):
    net_file = runtime.net_file_from_sumocfg(sumocfg)
    geometry = runtime.static_geometry(net_file)
    originator = runtime.StateConditionedSourceUtilityOriginator.load(
        Path(options["--runtime-model"]), expected_sha256=options["--runtime-model-sha256"],
        expected_city="cologne", network_context=runtime.read_network_right_of_way_context(net_file),
        arm="rigid_target_only")
    api = runtime.load_libsumo()
    if "1.22.0" not in str(runtime.libsumo_version()):
        raise ValueError("The original SUMO 1.22.0 runtime is required")
    detailed = runtime.CollisionObserver(api, geometry)
    observer = ValidationObserver(api, detailed, routes)
    if "CFCMT_SCRATCH" not in tripinfo.parts or tripinfo.exists():
        raise ValueError("Use a fresh remote CFCMT_SCRATCH tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario="cologne1", policy=runtime.RUNTIME_POLICY,
            models=runtime._runtime_models(originator, prediction_horizon_sec=int(options["--prediction-horizon-sec"])),
            duration_sec=END - BEGIN, control_interval_sec=int(options["--control-interval-sec"]),
            warmup_sec=float(options["--warmup-sec"]), seed=int(options["--seed"]),
            tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0, step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    return {"metrics": metrics, "samples": detailed.samples, "static_geometry": geometry,
            "native_lane_catalog": detailed.lane_catalog, "step_times": observer.step_times,
            "focal_passages": observer.tracker.summary(), "originator_diagnostics": originator.diagnostics()}


def run_validation(*, sumocfg, geometry_report, launch_record, reference_result,
                   mirror_reference_result, tripinfo):
    started = time.monotonic()
    runtime, bindings = _load_runtime()
    reference = json.loads(reference_result.read_text())
    if reference["protocol"] != base.REFERENCE_PROTOCOL or reference["status"] != "PASS":
        raise ValueError("Use the passing original t90912 collision replay as reference")
    mirror = json.loads(mirror_reference_result.read_text())
    if mirror["protocol"] != MIRROR_REFERENCE_PROTOCOL or mirror["status"] != "FAIL":
        raise ValueError("Use the retained failing t90940 single-side repair as mirror reference")
    if not any({row["collider"], row["victim"]} == set(MIRROR_IDS)
               for row in mirror["repair"]["retained_collision_samples"]):
        raise ValueError("The frozen mirror pair is absent from the retained V154C collision reports")
    original = json.loads(Path(reference["reference_result"]).read_text())
    expected_actions = [row for row in original["metrics"]["accepted_intervention_trace"] if row["time_sec"] < END]
    options = runtime.launch_inputs(json.loads(launch_record.read_text()))
    if (options["--seed"], options["--warmup-sec"], options["--control-interval-sec"], options["--prediction-horizon-sec"]) != ("41242", "60", "10", "450"):
        raise ValueError("The original frozen seed and controller parameters changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = runtime.load_traffic_signal_manifest(Path(options["--manifest"]))
    original_cfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    routes, demand = read_focal_demand(original_cfg)
    for vehicle in ORIGINAL_IDS:
        observed_route = next(row["vehicles"][vehicle]["route"] for row in reference["samples"]
                              if vehicle in row["vehicles"])
        if routes[vehicle] != observed_route:
            raise ValueError(f"Original demand route differs from retained replay for {vehicle}")
    mirror_left_route = next(row["vehicles"][MIRROR_IDS[1]]["route"] for row in mirror["repair"]["samples"]
                             if MIRROR_IDS[1] in row["vehicles"])
    if routes[MIRROR_IDS[1]] != mirror_left_route:
        raise ValueError("Mirror left demand route differs from the retained V154C observation")
    baseline = _run_arm(runtime, options, original_cfg, routes,
                        tripinfo.with_name(tripinfo.stem + ".baseline" + tripinfo.suffix))
    audit = base.baseline_audit(reference, expected_actions, baseline)
    result = {"protocol": PROTOCOL, "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
              "reference_result": str(reference_result), "mirror_reference_result": str(mirror_reference_result),
              "launch_record": str(launch_record), "geometry_report": str(geometry_report),
              "original_sumocfg": str(original_cfg), "repaired_sumocfg": str(sumocfg),
              "original_runtime_arguments": options, "focal_demand": demand,
              "simulation_window_sec": [BEGIN, END], "observation_window_sec_inclusive": [TRACE_BEGIN, TRACE_END],
              "baseline_reproduction": audit}
    if not audit["passed"]:
        return {**result, "status": "BASELINE_REPRODUCTION_FAILED", "repair_executed": False,
                "elapsed_sec": time.monotonic() - started}
    repair = _run_arm(runtime, options, sumocfg, routes, tripinfo)
    checks = repair_checks(repair)
    endpoint_keys = ("ok", "error", "departed", "arrived", "active_vehicles_at_horizon",
                     "pending_vehicles_at_horizon", "collision_events", "collision_incidents",
                     "starting_teleports", "ending_teleports", "phase_execution_audit")
    # Native width is retained in actual observations, including the mirrored
    # left vehicle whose type is the same pkw as its prespecified through car.
    through_widths = {}
    for vehicle in (ORIGINAL_IDS[0], MIRROR_IDS[0]):
        type_id = demand["vehicles"][vehicle]["vehicle"]["type"]
        measured = [pose["width_m"] for row in baseline["samples"] for pose in row["vehicles"].values()
                    if pose["type_id"] == type_id]
        if not measured or set(measured) != {1.8}:
            raise ValueError("The frozen passenger width was not reproduced in native observations")
        through_widths[FOCAL_LANES[vehicle][0]] = measured[0]
    return {**result, "status": "PASS" if all(checks.values()) else "FAIL", "repair_executed": True,
            "checks": checks, "elapsed_sec": time.monotonic() - started,
            "early_prefix_diagnostic": {
                "retained_intervention_trace": base.sequence_comparison(expected_actions, repair["metrics"].get("accepted_intervention_trace", [])),
                "window_observations": base.sequence_comparison(reference["samples"], repair["samples"]),
                "hard_gate": False,
                "interpretation": "Both waiting geometries may affect earlier traffic and state feedback under the fixed controller. No unchanged prefix is assumed."},
            "baseline_waiting_body_evidence": waiting_body_evidence(baseline, through_widths),
            "repair_waiting_body_evidence": waiting_body_evidence(repair, through_widths),
            "repair": {"endpoint": {key: repair["metrics"][key] for key in endpoint_keys if key in repair["metrics"]},
                       "accepted_intervention_trace": repair["metrics"].get("accepted_intervention_trace", []),
                       "retained_collision_samples": repair["metrics"].get("collision_samples", []),
                       "focal_passages": repair["focal_passages"], "originator_diagnostics": repair["originator_diagnostics"],
                       "static_geometry": repair["static_geometry"], "native_lane_catalog": repair["native_lane_catalog"],
                       "samples": repair["samples"]},
            "claim_boundary": "One original-versus-bilateral comparison in the frozen collision window. All four original and mirror focal vehicles must pass; no full-rollout safety, occupancy-unit or source-admission claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "geometry-report", "launch-record", "reference-result", "mirror-reference-result", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_validation(sumocfg=args.sumocfg, geometry_report=args.geometry_report,
                            launch_record=args.launch_record, reference_result=args.reference_result,
                            mirror_reference_result=args.mirror_reference_result, tripinfo=args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "baseline": result["baseline_reproduction"]["passed"],
                      "repair_executed": result["repair_executed"], "checks": result.get("checks"),
                      "elapsed_sec": result["elapsed_sec"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
