"""One frozen V154F controller/seed cell on the existing bilateral Cologne net."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time


_spec = importlib.util.spec_from_file_location(
    "cologne_v154e_validation_tooling",
    Path(__file__).with_name("run_cologne_bilateral_full_validation.py"),
)
full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(full)
short = full.short
PROTOCOL = "tsc-v154f-cologne-repaired-controller-comparison-v1"
SEED_ROSTER = (52282, 41242, 63792)
ARMS = ("rigid_target_only", "phase_pressure")
ROSTER_CONFIG = Path(__file__).resolve().parents[2] / "cf_h2o/config/traffic_signal_tsc_v150l_state_conditioned_source_closed_loop.json"
_load_runtime = full._load_runtime


def legal_internal_sequences(geometry, routes):
    connections = geometry["connections"]
    continuations = {f"{row['from']}_{row['fromLane']}": row["via"]
                     for row in connections if row["from"].startswith(":") and row.get("via")}
    result = {}
    for vehicle, (incoming, outgoing) in routes.items():
        alternatives = []
        for row in connections:
            if row.get("tl") != short.TLS or (row["from"], row["to"]) != (incoming, outgoing):
                continue
            sequence = [row["via"]]
            if row["via"] in continuations:
                sequence.append(continuations[row["via"]])
            if sequence not in alternatives:
                alternatives.append(sequence)
        result[vehicle] = alternatives
    return result


class RouteAwarePassageTracker(short.FocalPassageTracker):
    """Diagnostic passage on any actual legal lane for the frozen two-edge route."""

    def __init__(self, routes, geometry):
        super().__init__(routes)
        self.alternatives = legal_internal_sequences(geometry, routes)

    def summary(self):
        result = {}
        for vehicle, records in self.transitions.items():
            incoming, outgoing = self.routes[vehicle]
            attempts = []
            for internal in self.alternatives[vehicle]:
                required = [incoming, *internal, outgoing]
                milestones, next_required = [], 0
                for row in records:
                    if next_required == len(required):
                        break
                    field = "road_id" if next_required in (0, len(required) - 1) else "lane_id"
                    if row[field] == required[next_required]:
                        milestones.append(row)
                        next_required += 1
                attempts.append({"internal_lane_sequence": internal,
                                 "matched_milestones": milestones,
                                 "complete": next_required == len(required)})
            passed = next((attempt for attempt in attempts if attempt["complete"]), None)
            result[vehicle] = {
                "seen": bool(records), "original_route": self.routes[vehicle],
                "route_unchanged": self.route_matches[vehicle],
                "legal_internal_lane_sequences": self.alternatives[vehicle],
                "matched_internal_lane_sequence": passed["internal_lane_sequence"] if passed else None,
                "matched_milestones": passed["matched_milestones"] if passed else [],
                "lane_transitions": records,
                "passed_controlled_junction": passed is not None and self.route_matches[vehicle],
                "hard_gate": False,
            }
        return result


def comparison_validity(arm, due_demand):
    metrics = arm["metrics"]
    departed, arrived = metrics.get("departed"), metrics.get("arrived")
    active, pending = metrics.get("active_vehicles_at_horizon"), metrics.get("pending_vehicles_at_horizon")
    counts_available = all(value is not None for value in (departed, arrived, active, pending))
    return {
        **full.completeness_checks(arm),
        "fixed_due_demand_population": metrics.get("demand_population_at_horizon") == due_demand,
        "departed_equals_arrived_plus_active": counts_available and departed == arrived + active,
        "due_equals_departed_plus_pending": counts_available and due_demand == departed + pending,
        "tripinfo_covers_all_departed": counts_available and metrics.get("tripinfo_count") == departed,
        "tripinfo_completed_matches_arrivals": counts_available and metrics.get("tripinfo_completed_count") == arrived,
        "tripinfo_unfinished_matches_active": counts_available and metrics.get("tripinfo_unfinished_count") == active,
        "complete_3541_post_warmup_samples": metrics.get("fixed_horizon_sample_seconds") == 3541,
    }


def _run_arm(runtime, options, sumocfg, routes, geometry, arm, seed, tripinfo):
    originator, models = None, None
    policy = "phase_pressure"
    if arm == "rigid_target_only":
        originator = runtime.StateConditionedSourceUtilityOriginator.load(
            Path(options["--runtime-model"]), expected_sha256=options["--runtime-model-sha256"],
            expected_city="cologne", network_context=runtime.read_network_right_of_way_context(
                runtime.net_file_from_sumocfg(sumocfg)), arm=arm)
        models = runtime._runtime_models(originator, prediction_horizon_sec=450)
        policy = runtime.RUNTIME_POLICY
    api = runtime.load_libsumo()
    if "1.22.0" not in str(runtime.libsumo_version()):
        raise ValueError("The original SUMO 1.22.0 runtime is required")
    v3 = importlib.import_module("cf_h2o.eval.traffic_signal_resco_cfcmt_v3")
    observer = full.FullObserver(api, lambda **_: None, routes, runtime, v3, geometry)
    observer.tracker = RouteAwarePassageTracker(routes, geometry)
    if "CFCMT_SCRATCH" not in tripinfo.parts or tripinfo.exists():
        raise ValueError("Use a fresh remote CFCMT_SCRATCH tripinfo path")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario="cologne1", policy=policy, models=models,
            duration_sec=full.END - full.BEGIN, control_interval_sec=10, warmup_sec=60, seed=seed,
            tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0, step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    return {
        "metrics": metrics, "collision_inventory": observer.inventory(),
        "step_times": observer.step_times, "focal_passages": observer.tracker.summary(),
        "originator_diagnostics": originator.diagnostics() if originator is not None else None,
        "deployed_policy": policy,
    }


def run_cell(*, sumocfg, reference_result, arm, seed, tripinfo):
    started = time.monotonic()
    roster = json.loads(ROSTER_CONFIG.read_text())["full"]["seeds"]
    if roster != list(SEED_ROSTER) or seed not in roster or arm not in ARMS:
        raise ValueError("Use the frozen V150L full seed roster and the two specified controllers")
    if (arm, seed) == ("rigid_target_only", 41242):
        raise ValueError("Reuse the existing V154E rigid_target_only/41242 result without another simulation")
    reference = json.loads(reference_result.read_text())
    if reference["protocol"] != full.PROTOCOL or reference["status"] != "PASS":
        raise ValueError("Use the passing V154E original-duration validation as the fixed reference")
    if sumocfg.resolve() != Path(reference["repaired_sumocfg"]).resolve():
        raise ValueError("Use the unchanged V154D bilateral network configuration")
    runtime, bindings = _load_runtime()
    if bindings != reference["source_bindings"]:
        raise ValueError("The controller module bindings differ from the frozen V154E runtime")
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != reference["bilateral_static_geometry"]:
        raise ValueError("The bilateral geometry differs from the retained V154E network")
    options = reference["original_runtime_arguments"]
    if (options["--seed"], options["--warmup-sec"], options["--control-interval-sec"], options["--prediction-horizon-sec"]) != ("41242", "60", "10", "450"):
        raise ValueError("The retained original controller parameters differ from the frozen setup")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    routes = {vehicle: reference["focal_demand"]["vehicles"][vehicle]["route"] for vehicle in full.FOCAL_IDS}
    actual = _run_arm(runtime, options, sumocfg, routes, geometry, arm, seed, tripinfo)
    due_demand = reference["repair"]["metrics"]["demand_population_at_horizon"]
    checks = comparison_validity(actual, due_demand)
    valid = all(checks.values())
    zero_collisions = actual["metrics"].get("collision_events") == 0 and actual["metrics"].get("collision_incidents") == 0
    status = "INVALID" if not valid else "PASS" if zero_collisions else "FAIL"
    return {
        "protocol": PROTOCOL, "status": status, "run_valid": valid,
        "zero_incident_status": "PASS" if valid and zero_collisions else "FAIL" if valid else "UNAVAILABLE",
        "validity_checks": checks, "arm": arm, "seed": seed,
        "seed_roster": roster, "seed_roster_source": str(ROSTER_CONFIG),
        "scenario": "cologne1", "city": "cologne", "simulation_window_sec": [full.BEGIN, full.END],
        "repaired_sumocfg": str(sumocfg), "reference_result": str(reference_result),
        "geometry_reference": "V154E bilateral_static_geometry, exact parsed equality before simulation",
        "source_bindings": bindings, "driver_file": str(Path(__file__).resolve()),
        "runtime_model": options["--runtime-model"] if arm == "rigid_target_only" else None,
        "fixed_runtime": {"duration_sec": 3600, "warmup_sec": 60, "control_interval_sec": 10,
                          "rigid_prediction_horizon_sec": 450, "residual_coordination_mode": "direct",
                          "residual_cooldown_intervals_override": 0, "models_none_for_phase_pressure": arm == "phase_pressure"},
        **actual, "elapsed_sec": time.monotonic() - started,
        "focal_passage_interpretation": "Diagnostic only. Each fixed two-edge route permits its actual legal internal lane sequence; other seeds may use an alternate straight lane or pass later.",
        "claim_boundary": "One cell of the fixed three-seed, same-network rigid versus PhasePressure comparison. Collisions remain observed failures; no tuned service or focal-passage threshold. Reused V154E rigid/41242 is a separate retained cell.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEED_ROSTER, required=True)
    for name in ("sumocfg", "reference-result", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_cell(sumocfg=args.sumocfg, reference_result=args.reference_result,
                      arm=args.arm, seed=args.seed, tripinfo=args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "run_valid", "zero_incident_status", "arm", "seed", "elapsed_sec")}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
