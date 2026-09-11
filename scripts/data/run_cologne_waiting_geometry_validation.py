"""Validate one Cologne waiting-point repair with the frozen pre-unit-change runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import time


PROTOCOL = "tsc-v154c-cologne-waiting-geometry-validation-v1"
SOURCE_ID = "85b2b43cd12832c5a1dd"
REFERENCE_PROTOCOL = "tsc-v154-cologne-first-collision-read-only-replay-v1"
BEGIN, END = 25200, 25666
TRACE_BEGIN, TRACE_END = 25630, 25665
TLS = "cluster_357187_359543"
FOCAL_IDS = ("102219_396_0", "129962_409_0")
STAGES = ("before_simulation_step", "after_simulation_step_before_executor", "after_executor_advance")
FOCAL_LANES = {
    FOCAL_IDS[0]: (f":{TLS}_1_1",),
    FOCAL_IDS[1]: (f":{TLS}_13_0", f":{TLS}_24_0"),
}


def sequence_comparison(expected, actual):
    common = 0
    for left, right in zip(expected, actual):
        if left != right:
            break
        common += 1
    same = expected == actual
    first = None
    if not same:
        left = expected[common] if common < len(expected) else None
        right = actual[common] if common < len(actual) else None
        first = {
            "index": common,
            "reference_time_sec": left.get("time_sec", left.get("time")) if left else None,
            "actual_time_sec": right.get("time_sec", right.get("time")) if right else None,
            "changed_fields": sorted(key for key in set(left or {}) | set(right or {})
                                     if (left or {}).get(key) != (right or {}).get(key)),
        }
    return {"passed": same, "expected_count": len(expected), "actual_count": len(actual),
            "exact_leading_row_count": common, "first_difference": first}


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
            route = list(api.vehicle.getRoute(vehicle))
            self.route_matches[vehicle] &= route == self.routes[vehicle]
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
            milestones = []
            next_required = 0
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
    """Keep the original detailed observer and add read-only passage milestones."""

    def __init__(self, api, detailed, routes):
        self.api, self.detailed = api, detailed
        self.tracker = FocalPassageTracker(routes)
        self.step_times = []

    def __call__(self, *, stage, time_sec, executors):
        self.detailed(stage=stage, time_sec=time_sec, executors=executors)
        if stage == "after_simulation_step_before_executor":
            self.step_times.append(time_sec)
            self.tracker.observe(self.api, time_sec)


def window_checks(arm):
    rows = arm["samples"]
    expected = {(t, stage) for t in range(TRACE_BEGIN, TRACE_END + 1) for stage in STAGES}
    return {
        "full_466_one_second_steps": arm["step_times"] == list(range(BEGIN + 1, END + 1)),
        "complete_108_three_stage_observations": len(rows) == 108
        and {(row["time_sec"], row["stage"]) for row in rows} == expected,
    }


def baseline_audit(reference, expected_actions, arm):
    metrics = arm["metrics"]
    comparisons = {
        "intervention_trace": sequence_comparison(expected_actions, metrics.get("accepted_intervention_trace", [])),
        "retained_collision_reports": sequence_comparison(reference["retained_collision_samples"], metrics.get("collision_samples", [])),
        "original_108_observations": sequence_comparison(reference["samples"], arm["samples"]),
    }
    checks = {"runner_completed": bool(metrics.get("ok")), **window_checks(arm),
              **{name: comparison["passed"] for name, comparison in comparisons.items()},
              "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0}
    return {"passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "native_collision_events": metrics.get("collision_events"),
            "native_collision_incidents": metrics.get("collision_incidents")}


def repair_checks(arm):
    metrics = arm["metrics"]
    return {"runner_completed": bool(metrics.get("ok")), **window_checks(arm),
            "no_native_collisions_in_full_window": metrics.get("collision_events") == 0
            and metrics.get("collision_incidents") == 0,
            "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
            "both_focal_vehicles_passed_controlled_junction": all(
                arm["focal_passages"][vehicle]["passed_controlled_junction"] for vehicle in FOCAL_IDS)}


def body_strip_clearance(polygon, centerline, through_width):
    """Lateral body clearance to this measured straight lane's swept strip."""
    if len(centerline) != 2:
        raise ValueError("The frozen straight lane must retain its two-point centerline")
    start, end = centerline
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    along = [((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / length for p in polygon]
    lateral = [(-(p[0] - start[0]) * dy + (p[1] - start[1]) * dx) / length for p in polygon]
    distance = 0.0 if min(lateral) <= 0 <= max(lateral) else min(abs(value) for value in lateral)
    return {"clearance_m": distance - through_width / 2,
            "body_longitudinal_range_m": [min(along), max(along)],
            "body_within_straight_lane_span": min(along) >= 0 and max(along) <= length}


def stationary_body_evidence(arm):
    samples = arm["samples"]
    straight = next((row["vehicles"][FOCAL_IDS[0]] for row in samples
                     if FOCAL_IDS[0] in row["vehicles"]), None)
    centerline = arm["native_lane_catalog"].get(FOCAL_LANES[FOCAL_IDS[0]][0], {}).get("shape_xy_m")
    observations = []
    if straight and centerline:
        for row in samples:
            left = row["vehicles"].get(FOCAL_IDS[1])
            if row["stage"] != STAGES[1] or not left or left["lane_id"] != FOCAL_LANES[FOCAL_IDS[1]][0]:
                continue
            if left["speed_mps"] > 1e-8 or left["passenger_polygon_xy_m"] is None:
                continue
            evidence = body_strip_clearance(left["passenger_polygon_xy_m"], centerline, straight["width_m"])
            observations.append({"time_sec": row["time_sec"], "lane_position_m": left["lane_position_m"],
                                 "through_vehicle_width_m": straight["width_m"], **evidence})
    available = [row["clearance_m"] for row in observations if row["body_within_straight_lane_span"]]
    return {"observations": observations, "minimum_clearance_m": min(available) if available else None,
            "interpretation": "Auxiliary actual stationary-body evidence; absent stationary samples do not imply clearance was observed."}


def _load_runtime():
    source = Path(os.environ["CFCMT_SOURCE_ROOT"]).resolve()
    if source.name != SOURCE_ID:
        raise ValueError(f"Use the original frozen runtime source {SOURCE_ID}")
    names = (
        "cf_h2o.eval.traffic_signal_cologne_collision_replay",
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3",
        "cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop",
        "cf_h2o.traffic_signal.action_ranker",
    )
    modules = [importlib.import_module(name) for name in names]
    bindings = {name: str(Path(module.__file__).resolve()) for name, module in zip(names, modules)}
    if any(not Path(path).is_relative_to(source) for path in bindings.values()):
        raise ValueError("The validation loaded cf_h2o outside the frozen original source")
    return modules[0], bindings


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


def run_validation(*, sumocfg, geometry_report, launch_record, reference_result, tripinfo):
    started = time.monotonic()
    runtime, bindings = _load_runtime()
    reference = json.loads(reference_result.read_text())
    if reference["protocol"] != REFERENCE_PROTOCOL or reference["status"] != "PASS":
        raise ValueError("Use the passing original t90912 collision replay as reference")
    original = json.loads(Path(reference["reference_result"]).read_text())
    expected_actions = [row for row in original["metrics"]["accepted_intervention_trace"] if row["time_sec"] < END]
    options = runtime.launch_inputs(json.loads(launch_record.read_text()))
    if (options["--seed"], options["--warmup-sec"], options["--control-interval-sec"], options["--prediction-horizon-sec"]) != ("41242", "60", "10", "450"):
        raise ValueError("The original frozen seed and controller parameters changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = runtime.load_traffic_signal_manifest(Path(options["--manifest"]))
    original_cfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    routes = {vehicle: next(row["vehicles"][vehicle]["route"] for row in reference["samples"]
                           if vehicle in row["vehicles"]) for vehicle in FOCAL_IDS}
    baseline = _run_arm(runtime, options, original_cfg, routes,
                        tripinfo.with_name(tripinfo.stem + ".baseline" + tripinfo.suffix))
    audit = baseline_audit(reference, expected_actions, baseline)
    result = {"protocol": PROTOCOL, "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
              "reference_result": str(reference_result), "launch_record": str(launch_record),
              "geometry_report": str(geometry_report), "original_sumocfg": str(original_cfg),
              "repaired_sumocfg": str(sumocfg), "original_runtime_arguments": options,
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
    return {**result, "status": "PASS" if all(checks.values()) else "FAIL", "repair_executed": True,
            "checks": checks, "elapsed_sec": time.monotonic() - started,
            "early_prefix_diagnostic": {
                "retained_intervention_trace": sequence_comparison(expected_actions, repair["metrics"].get("accepted_intervention_trace", [])),
                "window_observations": sequence_comparison(reference["samples"], repair["samples"]),
                "hard_gate": False,
                "interpretation": "Geometry can affect preceding vehicles and approach braking; no arbitrary time is certified as unaffected. Later action differences reflect state feedback under fixed controller parameters."},
            "baseline_stationary_body_evidence": stationary_body_evidence(baseline),
            "repair_stationary_body_evidence": stationary_body_evidence(repair),
            "repair": {"endpoint": {key: repair["metrics"][key] for key in endpoint_keys if key in repair["metrics"]},
                       "accepted_intervention_trace": repair["metrics"].get("accepted_intervention_trace", []),
                       "retained_collision_samples": repair["metrics"].get("collision_samples", []),
                       "focal_passages": repair["focal_passages"], "originator_diagnostics": repair["originator_diagnostics"],
                       "static_geometry": repair["static_geometry"], "native_lane_catalog": repair["native_lane_catalog"],
                       "samples": repair["samples"]},
            "claim_boundary": "Same-window original-model geometry validation; no unit-formula change, source admission or full-rollout safety claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "geometry-report", "launch-record", "reference-result", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_validation(sumocfg=args.sumocfg, geometry_report=args.geometry_report,
                            launch_record=args.launch_record, reference_result=args.reference_result, tripinfo=args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "baseline": result["baseline_reproduction"]["passed"],
                      "repair_executed": result["repair_executed"], "checks": result.get("checks"),
                      "elapsed_sec": result["elapsed_sec"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
