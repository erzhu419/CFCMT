"""Full 3600-second confirmation and uncapped collision inventory for V154D."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time


_spec = importlib.util.spec_from_file_location(
    "cologne_v154d_validation_tooling",
    Path(__file__).with_name("run_cologne_bilateral_waiting_geometry_validation.py"),
)
short = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(short)
base = short.base
PROTOCOL = "tsc-v154e-cologne-bilateral-full-validation-v1"
BEGIN, END = 25200, 28800
PREFIX_END = short.END
FOCAL_IDS = short.FOCAL_IDS
_load_runtime = short._load_runtime

TRAFFIC_METRIC_KEYS = (
    "departed", "arrived", "loaded", "demand_population_at_horizon",
    "active_vehicles_at_horizon", "pending_vehicles_at_horizon",
    "completion_ratio", "departure_service_ratio", "throughput_ratio",
    "active_vehicle_seconds", "active_vehicle_hours", "pending_vehicle_seconds", "pending_vehicle_hours",
    "system_vehicle_seconds", "system_vehicle_hours", "halted_vehicle_seconds", "halted_vehicle_hours",
    "mean_active_vehicles", "mean_system_vehicles", "mean_queue", "p90_queue",
    "mean_queue_proxy", "fixed_horizon_sample_seconds", "controlled_lane_count",
    "tripinfo_count", "tripinfo_completed_count", "tripinfo_unfinished_count", "tripinfo_unfinished_fraction",
    "mean_tripinfo_duration", "mean_tripinfo_waiting_time", "mean_tripinfo_time_loss", "mean_tripinfo_depart_delay",
    "p90_tripinfo_duration", "p90_tripinfo_waiting_time", "p90_tripinfo_time_loss", "p90_tripinfo_depart_delay",
    "mean_completed_tripinfo_duration", "mean_completed_tripinfo_waiting_time", "mean_completed_tripinfo_time_loss",
    "p90_completed_tripinfo_duration", "p90_completed_tripinfo_waiting_time", "p90_completed_tripinfo_time_loss",
    "starting_teleports", "ending_teleports", "collision_events", "collision_event_steps", "collision_incidents",
    "emergency_stops", "phase_execution_audit",
)


def movement_catalog(geometry):
    """Associate internal waiting/continuation lanes with their original TLS movement."""
    connections = geometry["connections"]
    continuations = {f"{row['from']}_{row['fromLane']}": row["via"]
                     for row in connections if row["from"].startswith(":") and row.get("via")}
    result = {}
    for row in connections:
        if row.get("tl") != short.TLS:
            continue
        movement = {key: row[key] for key in ("from", "to", "fromLane", "toLane", "linkIndex", "dir")}
        lane = row["via"]
        result[lane] = movement
        if lane in continuations:
            result[continuations[lane]] = movement
    return result


class FullObserver:
    def __init__(self, api, detailed, routes, runtime, v3, geometry):
        self.api, self.detailed, self.runtime, self.v3 = api, detailed, runtime, v3
        self.tracker = short.FocalPassageTracker(routes)
        self.movements = movement_catalog(geometry)
        self.step_times = []
        self.events = []
        self.incidents = {}
        self.collision_event_steps = 0
        self.prefix_passages = None

    def __call__(self, *, stage, time_sec, executors):
        self.detailed(stage=stage, time_sec=time_sec, executors=executors)
        if stage != short.STAGES[1]:
            return
        self.step_times.append(time_sec)
        self.tracker.observe(self.api, time_sec)
        if time_sec == PREFIX_END:
            self.prefix_passages = deepcopy(self.tracker.summary())
        collisions = tuple(self.api.simulation.getCollisions())
        if not collisions:
            return
        self.collision_event_steps += 1
        rows = self.v3._collision_debug_samples(
            sumo_api=self.api, collisions=collisions, executors=executors,
            time_sec=time_sec, max_samples=len(collisions))
        active = set(self.api.vehicle.getIDList())
        for collision, row in zip(collisions, rows):
            key = self.v3._collision_incident_key(collision)
            row["incident_key"] = list(key)
            row["observation_stage"] = stage
            row["participant_poses"] = {
                role: self.runtime.vehicle_pose(self.api, str(getattr(collision, role)))
                for role in ("collider", "victim") if str(getattr(collision, role)) in active}
            row["unavailable_participant_poses"] = {
                role: {"vehicle_id": str(getattr(collision, role)),
                       "reason": "vehicle_not_active_at_observation"}
                for role in ("collider", "victim") if str(getattr(collision, role)) not in active}
            row["participant_movements"] = {
                role: self.movements.get(pose["lane_id"])
                for role, pose in row["participant_poses"].items()}
            index = len(self.events)
            self.events.append(row)
            if key not in self.incidents:
                self.incidents[key] = {
                    "incident_key": list(key), "first_time_sec": time_sec, "last_time_sec": time_sec,
                    "event_indices": [], "first_participant_lanes": {
                        role: pose["lane_id"] for role, pose in row["participant_poses"].items()},
                    "first_participant_speeds_mps": {role: pose["speed_mps"]
                                                   for role, pose in row["participant_poses"].items()},
                    "first_unavailable_participant_poses": row["unavailable_participant_poses"],
                    "participant_movements": row["participant_movements"],
                }
            incident = self.incidents[key]
            incident["last_time_sec"] = time_sec
            incident["event_indices"].append(index)

    def inventory(self):
        incidents = list(self.incidents.values())
        lane_pairs = Counter((tuple(sorted(row["first_participant_lanes"].values())),
                              tuple(sorted(row["first_unavailable_participant_poses"]))) for row in incidents)
        return {
            "native_event_count": len(self.events), "native_incident_count": len(incidents),
            "collision_event_steps": self.collision_event_steps,
            "event_sample_cap": None,
            "incident_definition": "Original v3._collision_incident_key: sorted vehicle IDs, native type, native reported lane; repeated reports share one incident.",
            "events": self.events, "incidents": incidents,
            "incident_counts_by_first_participant_lane_pair": [
                {"lane_pair": list(pair), "unavailable_roles": list(missing), "incident_count": count}
                for (pair, missing), count in sorted(lane_pairs.items())],
            "observation_stage": short.STAGES[1],
            "state_timing": "Executor and TLS states are read before executor.advance; the original capped metric reports are sampled after advance and remain separately retained.",
        }


def completeness_checks(arm):
    metrics, inventory = arm["metrics"], arm["collision_inventory"]
    return {
        "runner_completed": bool(metrics.get("ok")),
        "full_3600_one_second_steps": arm["step_times"] == list(range(BEGIN + 1, END + 1)),
        "uncapped_events_match_native_total": inventory["native_event_count"] == metrics.get("collision_events"),
        "uncapped_incidents_match_native_total": inventory["native_incident_count"] == metrics.get("collision_incidents"),
        "collision_event_steps_match_native_total": inventory["collision_event_steps"] == metrics.get("collision_event_steps"),
        "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
    }


def baseline_audit(reference, arm):
    metrics = arm["metrics"]
    comparisons = {
        "retained_actions": base.sequence_comparison(reference["accepted_intervention_trace"], metrics.get("accepted_intervention_trace", [])),
        "retained_20_collision_reports": base.sequence_comparison(reference["collision_samples"], metrics.get("collision_samples", [])),
    }
    differences = {key: {"expected": reference[key], "actual": metrics.get(key)}
                   for key in TRAFFIC_METRIC_KEYS if reference[key] != metrics.get(key)}
    checks = {**completeness_checks(arm),
              **{key: value["passed"] for key, value in comparisons.items()},
              "scientific_traffic_and_safety_metrics_exact": not differences,
              "all_original_metrics_exact": reference == metrics}
    return {"passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "full_metric_expected_count": len(reference), "full_metric_actual_count": len(metrics),
            "full_metric_changed_keys": sorted(key for key in set(reference) | set(metrics)
                                              if key not in reference or key not in metrics or reference[key] != metrics[key]),
            "scientific_metric_keys": list(TRAFFIC_METRIC_KEYS), "scientific_metric_differences": differences}


def short_prefix_audit(reference, arm):
    original = reference["repair"]
    actions = [row for row in arm["metrics"].get("accepted_intervention_trace", []) if row["time_sec"] < PREFIX_END]
    comparisons = {
        "retained_466_second_actions": base.sequence_comparison(original["accepted_intervention_trace"], actions),
        "original_108_physical_observations": base.sequence_comparison(original["samples"], arm["samples"]),
    }
    checks = {**{key: value["passed"] for key, value in comparisons.items()},
              "all_four_passage_prefixes_exact": original["focal_passages"] == arm["prefix_passages"]}
    return {"passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "prior_v154d_status": reference["status"],
            "interpretation": "The first466 seconds reproduce V154D within the new full run; its original FAIL and fixed endpoint remain unchanged."}


def repair_checks(arm, prefix_audit):
    metrics = arm["metrics"]
    return {**completeness_checks(arm), "v154d_466_second_prefix_reproduced": prefix_audit["passed"],
            "zero_native_collisions_over_3600_seconds": metrics.get("collision_events") == 0
            and metrics.get("collision_incidents") == 0,
            "original_pair_passed_controlled_junction": all(
                arm["focal_passages"][vehicle]["passed_controlled_junction"] for vehicle in short.ORIGINAL_IDS),
            "mirror_pair_passed_controlled_junction": all(
                arm["focal_passages"][vehicle]["passed_controlled_junction"] for vehicle in short.MIRROR_IDS)}


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
    v3 = importlib.import_module("cf_h2o.eval.traffic_signal_resco_cfcmt_v3")
    detailed = runtime.CollisionObserver(api, geometry)
    observer = FullObserver(api, detailed, routes, runtime, v3, geometry)
    if "CFCMT_SCRATCH" not in tripinfo.parts or tripinfo.exists():
        raise ValueError("Use a fresh remote CFCMT_SCRATCH tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario="cologne1", policy=runtime.RUNTIME_POLICY,
            models=runtime._runtime_models(originator, prediction_horizon_sec=450),
            duration_sec=END - BEGIN, control_interval_sec=10, warmup_sec=60, seed=41242,
            tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0, step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    return {"metrics": metrics, "collision_inventory": observer.inventory(),
            "step_times": observer.step_times, "focal_passages": observer.tracker.summary(),
            "prefix_passages": observer.prefix_passages, "samples": detailed.samples,
            "static_geometry": geometry, "native_lane_catalog": detailed.lane_catalog,
            "originator_diagnostics": originator.diagnostics()}


def _retained_arm(arm):
    return {key: arm[key] for key in ("metrics", "collision_inventory", "step_times",
            "focal_passages", "prefix_passages", "originator_diagnostics")}


def run_validation(*, sumocfg, launch_record, reference_result, short_reference_result, tripinfo):
    started = time.monotonic()
    runtime, bindings = _load_runtime()
    reference = json.loads(reference_result.read_text())
    if (reference["city"], reference["scenario"], reference["seed"], reference["arm"],
        reference["feature_binding"], reference["duration_sec"]) != (
            "cologne", "cologne1", 41242, "rigid_target_only", "stored_feature_names", 3600):
        raise ValueError("Use the original full corrected t90756 Cologne rigid rollout")
    short_reference = json.loads(short_reference_result.read_text())
    if short_reference["protocol"] != short.PROTOCOL or short_reference["status"] != "FAIL":
        raise ValueError("Use the retained V154D failing fixed-window result")
    if sumocfg.resolve() != Path(short_reference["repaired_sumocfg"]).resolve():
        raise ValueError("Use the unchanged V154D bilateral network configuration")
    options = runtime.launch_inputs(json.loads(launch_record.read_text()))
    if (options["--seed"], options["--warmup-sec"], options["--control-interval-sec"], options["--prediction-horizon-sec"]) != ("41242", "60", "10", "450"):
        raise ValueError("The original frozen seed and controller parameters changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = runtime.load_traffic_signal_manifest(Path(options["--manifest"]))
    original_cfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    routes, demand = short.read_focal_demand(original_cfg)
    if any(routes[vehicle] != short_reference["focal_demand"]["vehicles"][vehicle]["route"] for vehicle in FOCAL_IDS):
        raise ValueError("The four focal routes changed from V154D")
    baseline = _run_arm(runtime, options, original_cfg, routes,
                        tripinfo.with_name(tripinfo.stem + ".baseline" + tripinfo.suffix))
    audit = baseline_audit(reference["metrics"], baseline)
    audit["checks"]["originator_diagnostics_exact"] = reference["originator_diagnostics"] == baseline["originator_diagnostics"]
    audit["passed"] = all(audit["checks"].values())
    result = {"protocol": PROTOCOL, "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
              "reference_result": str(reference_result), "short_reference_result": str(short_reference_result),
              "launch_record": str(launch_record), "original_sumocfg": str(original_cfg),
              "repaired_sumocfg": str(sumocfg), "original_runtime_arguments": options,
              "focal_demand": demand, "simulation_window_sec": [BEGIN, END],
              "reproduced_v154d_window_sec": [BEGIN, PREFIX_END], "baseline_reproduction": audit,
              "baseline": _retained_arm(baseline)}
    if not audit["passed"]:
        return {**result, "status": "BASELINE_REPRODUCTION_FAILED", "repair_executed": False,
                "elapsed_sec": time.monotonic() - started}
    repair = _run_arm(runtime, options, sumocfg, routes, tripinfo)
    prefix = short_prefix_audit(short_reference, repair)
    checks = repair_checks(repair, prefix)
    return {**result, "status": "PASS" if all(checks.values()) else "FAIL", "repair_executed": True,
            "checks": checks, "v154d_prefix_reproduction": prefix, "repair": _retained_arm(repair),
            "bilateral_static_geometry": repair["static_geometry"],
            "bilateral_native_lane_catalog": repair["native_lane_catalog"],
            "elapsed_sec": time.monotonic() - started,
            "service_comparison": {key: {"original": baseline["metrics"].get(key),
                                         "bilateral": repair["metrics"].get(key)} for key in TRAFFIC_METRIC_KEYS},
            "claim_boundary": "One prespecified3600-second original-versus-unchanged-bilateral run at seed41242, with all native incidents retained. V154D remains FAIL; service superiority, other seeds/cities, occupancy-unit changes and source-transfer benefit require separate evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "launch-record", "reference-result", "short-reference-result", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_validation(sumocfg=args.sumocfg, launch_record=args.launch_record,
                            reference_result=args.reference_result, short_reference_result=args.short_reference_result,
                            tripinfo=args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "baseline": result["baseline_reproduction"]["passed"],
                      "repair_executed": result["repair_executed"], "checks": result.get("checks"),
                      "elapsed_sec": result["elapsed_sec"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
