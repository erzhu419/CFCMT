"""Change only the 27480 yellow priority encoding in the retained Cologne replay."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location(
    "cologne_v154g_tooling", Path(__file__).with_name("run_cologne_uturn_collision_replay.py"))
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
PROTOCOL = "tsc-v154h-cologne-yellow-priority-counterfactual-v1"
SWITCH_TIME = 27480
TRIGGER_STAGE = "before_simulation_step"


def priority_yellow(green, yellow):
    expected = "".join("y" if char in "Gg" else "r" for char in green)
    if yellow != expected:
        raise ValueError("The intervention must start from the retained all-minor yellow mapping")
    return "".join("Y" if char == "G" else actual for char, actual in zip(green, yellow))


class CounterfactualObserver(replay.WindowObserver):
    def __init__(self, api, runtime):
        super().__init__(api, runtime)
        self.interventions = []
        self.before_intervention_sample = None
        self.passages = {}

    def __call__(self, *, stage, time_sec, executors):
        if time_sec == SWITCH_TIME and stage == TRIGGER_STAGE:
            if self.interventions:
                raise ValueError("Only one yellow transition may be changed")
            super().__call__(stage=stage, time_sec=time_sec, executors=executors)
            self.before_intervention_sample = self.samples.pop()
            executor = executors[replay.TLS]
            before = executor.snapshot()
            if before["mode"] != "yellow" or before["remaining_sec"] != 3.0:
                raise ValueError("The retained 27480 yellow transition was not reached")
            changed = priority_yellow(before["current_green_state"], before["current_state"])
            executor._set_state(changed)
            self.interventions.append({"time_sec": time_sec, "stage": stage,
                "before": before, "after": executor.snapshot(),
                "changed_indices": [i for i, (a, b) in enumerate(zip(before["current_state"], changed)) if a != b]})
        super().__call__(stage=stage, time_sec=time_sec, executors=executors)
        if stage == replay.full.short.STAGES[1] and time_sec >= SWITCH_TIME:
            active = set(self.api.vehicle.getIDList())
            for vehicle in replay.IDS:
                if vehicle not in self.passages and vehicle in active:
                    lane = self.api.vehicle.getLaneID(vehicle)
                    if lane == replay.PATHS[vehicle][-1]:
                        self.passages[vehicle] = {"time_sec": time_sec, "lane_id": lane,
                            "lane_position_m": self.api.vehicle.getLanePosition(vehicle),
                            "route": list(self.api.vehicle.getRoute(vehicle))}


def sample_key(row):
    return row["time_sec"], row["stage"]


def timing_view(row):
    executor = deepcopy(row["executor"])
    executor["current_state"] = executor["current_state"].replace("Y", "y")
    return row["tls_state"].replace("Y", "y"), executor


def counterfactual_checks(reference, metrics, inventory, observer, step_times):
    expected = {sample_key(row): row for row in reference["samples"]}
    actual = {sample_key(row): row for row in observer.samples}
    trigger = SWITCH_TIME, TRIGGER_STAGE
    before = [row for row in reference["samples"] if row["time_sec"] < SWITCH_TIME
              or (row["time_sec"] == SWITCH_TIME and row["stage"] != TRIGGER_STAGE)]
    prefix = [row for row in observer.samples if sample_key(row) in {sample_key(x) for x in before}]
    differences = [{"time_sec": key[0], "stage": key[1]} for key in expected
                   if key not in actual or timing_view(expected[key]) != timing_view(actual[key])]
    comparisons = {"prefix_samples": len(before), "timing_differences": differences}
    native = (inventory["native_event_count"], inventory["native_incident_count"], inventory["collision_event_steps"])
    checks = {
        "runner_completed": bool(metrics.get("ok")),
        # Native leader values are tuples; retained JSON represents them as lists.
        # Compare the exact persisted representation without relaxing numbers or order.
        "exact_original_preintervention_samples": json.loads(json.dumps(prefix)) == before
            and json.loads(json.dumps(observer.before_intervention_sample)) == expected[trigger],
        "one_intervention": len(observer.interventions) == 1,
        "complete_same_window": len(observer.samples) == len(reference["samples"]) and set(actual) == set(expected),
        "same_signal_and_executor_timing": not differences,
        "full_prefix_steps": step_times == list(range(replay.BEGIN + 1, replay.END + 1)),
        "native_inventory_agrees": native == (metrics.get("collision_events"), metrics.get("collision_incidents"), metrics.get("collision_event_steps")),
        "zero_native_collisions": native == (0, 0, 0),
        "no_teleports": metrics.get("starting_teleports") == 0 and metrics.get("ending_teleports") == 0,
        "both_focal_vehicles_passed": all(vehicle in observer.passages for vehicle in replay.IDS),
    }
    return checks, comparisons


def run_counterfactual(reference_result, geometry_reference_result, tripinfo):
    started = time.monotonic()
    reference = json.loads(reference_result.read_text())
    geometry_reference = json.loads(geometry_reference_result.read_text())
    if reference["protocol"] != replay.PROTOCOL or reference["status"] != "PASS":
        raise ValueError("Use the passing V154G exact collision replay as the retained baseline")
    runtime, bindings = replay.comparison._load_runtime()
    if bindings != reference["source_bindings"]:
        raise ValueError("Controller source changed")
    sumocfg = Path(reference["repaired_sumocfg"])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != geometry_reference["bilateral_static_geometry"]:
        raise ValueError("The fixed bilateral network changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = geometry_reference["original_runtime_arguments"]["--conversion-root"]
    api = runtime.load_libsumo()
    if "1.22.0" not in str(runtime.libsumo_version()):
        raise ValueError("Use SUMO 1.22.0")
    detailed = CounterfactualObserver(api, runtime)
    v3 = importlib.import_module("cf_h2o.eval.traffic_signal_resco_cfcmt_v3")
    routes = {v: geometry_reference["focal_demand"]["vehicles"][v]["route"] for v in replay.full.FOCAL_IDS}
    observer = replay.full.FullObserver(api, detailed, routes, runtime, v3, geometry)
    if tripinfo.exists() or "CFCMT_SCRATCH" not in tripinfo.parts:
        raise ValueError("Use a fresh server scratch tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario="cologne1",
            policy="phase_pressure", models=None, duration_sec=replay.END-replay.BEGIN,
            control_interval_sec=10, warmup_sec=60, seed=52282, tripinfo_output=tripinfo,
            residual_coordination_mode="direct", residual_cooldown_intervals_override=0,
            step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    inventory = observer.inventory()
    checks, comparisons = counterfactual_checks(reference, metrics, inventory, detailed, observer.step_times)
    return {"protocol": PROTOCOL, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "comparisons": comparisons, "reference_result": str(reference_result),
        "original_safety_status_retained": reference["reference_safety_status"],
        "geometry_reference_result": str(geometry_reference_result), "repaired_sumocfg": str(sumocfg),
        "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
        "simulation_window_sec": [replay.BEGIN, replay.END], "trace_window_sec": [replay.TRACE_BEGIN, replay.TRACE_END],
        "seed": 52282, "interventions": detailed.interventions,
        "before_intervention_sample": detailed.before_intervention_sample,
        "samples": detailed.samples, "focal_passages": detailed.passages,
        "metrics": metrics, "collision_inventory": inventory, "elapsed_sec": time.monotonic()-started,
        "interpretation": "One yellow transition only. Later executor behavior remains native; timing differences fail the frozen comparison. The prior baseline collision is retained."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-result", "geometry-reference-result", "tripinfo", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_counterfactual(args.reference_result, args.geometry_reference_result, args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":"))+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks", "elapsed_sec")}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
