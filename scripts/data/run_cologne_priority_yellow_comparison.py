"""V154I cells: frozen V154F runtime with the single shared yellow-priority fix."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "cologne_v154f_tooling", Path(__file__).with_name("run_cologne_repaired_controller_comparison.py"))
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)
PROTOCOL = "tsc-v154i-cologne-priority-yellow-comparison-v1"
ARMS, SEED_ROSTER = comparison.ARMS, comparison.SEED_ROSTER
EXECUTOR_MODULE = "cf_h2o.traffic_signal.safe_phase_controller"
_OLD_BRANCH = 'if char == "G":\n            chars.append("y")'
_NEW_BRANCH = ('if char == "G":\n'
               '            # SUMO retains major-link priority only for uppercase yellow.\n'
               '            chars.append("Y")')


def _load_priority_runtime(executor_source, expected_bindings):
    source = Path(os.environ["CFCMT_SOURCE_ROOT"]).resolve()
    original = (source / "cf_h2o/traffic_signal/safe_phase_controller.py").read_text()
    executor_source = executor_source.resolve()
    corrected = executor_source.read_text()
    if original.count(_OLD_BRANCH) != 1 or corrected != original.replace(_OLD_BRANCH, _NEW_BRANCH):
        raise ValueError("The executor must differ from frozen 85 only by the approved G-to-Y branch")
    spec = importlib.util.spec_from_file_location(EXECUTOR_MODULE, executor_source)
    executor_module = importlib.util.module_from_spec(spec)
    sys.modules[EXECUTOR_MODULE] = executor_module
    spec.loader.exec_module(executor_module)
    runtime, bindings = comparison._load_runtime()
    if bindings != expected_bindings:
        raise ValueError("The four original runtime module bindings changed")
    v3 = importlib.import_module("cf_h2o.eval.traffic_signal_resco_cfcmt_v3")
    if (v3.SafePhaseExecutor is not executor_module.SafePhaseExecutor
            or v3.build_safe_phase_executors is not executor_module.build_safe_phase_executors):
        raise ValueError("The evaluator did not bind the corrected shared executor")
    writes = []
    api = SimpleNamespace(trafficlight=SimpleNamespace(
        setRedYellowGreenState=lambda tls, state: writes.append(state)))
    timings = [executor_module.PhaseTiming(state, 0., 3., 1.) for state in ("Gg", "rG")]
    probe = v3.SafePhaseExecutor(sumo_api=api, tls_id="priority_probe", timings=timings, initial_state="Gg")
    if not probe.request("rG") or writes != ["Yy"] or probe.mode != "yellow" or probe.remaining_sec != 3.:
        raise ValueError("The actual shared executor did not preserve Gg-to-Yy and yellow timing")
    return runtime, bindings, {"initial_state": "Gg", "requested_state": "rG",
                              "observed_state": writes[0], "remaining_sec": probe.remaining_sec}


def run_cell(*, sumocfg, reference_result, executor_source, arm, seed, tripinfo):
    started = time.monotonic()
    roster = json.loads(comparison.ROSTER_CONFIG.read_text())["full"]["seeds"]
    if roster != list(SEED_ROSTER) or seed not in roster or arm not in ARMS:
        raise ValueError("Use the frozen V150L full seed roster and the two specified controllers")
    reference = json.loads(reference_result.read_text())
    if reference["protocol"] != comparison.full.PROTOCOL or reference["status"] != "PASS":
        raise ValueError("Use the passing V154E fixed-geometry reference")
    if sumocfg.resolve() != Path(reference["repaired_sumocfg"]).resolve():
        raise ValueError("Use the unchanged bilateral network configuration")
    runtime, bindings, probe = _load_priority_runtime(executor_source, reference["source_bindings"])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != reference["bilateral_static_geometry"]:
        raise ValueError("The bilateral geometry differs from the retained V154E network")
    options = reference["original_runtime_arguments"]
    if (options["--seed"], options["--warmup-sec"], options["--control-interval-sec"],
            options["--prediction-horizon-sec"]) != ("41242", "60", "10", "450"):
        raise ValueError("The frozen controller parameters changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    routes = {v: reference["focal_demand"]["vehicles"][v]["route"] for v in comparison.full.FOCAL_IDS}
    actual = comparison._run_arm(runtime, options, sumocfg, routes, geometry, arm, seed, tripinfo)
    checks = comparison.comparison_validity(actual, reference["repair"]["metrics"]["demand_population_at_horizon"])
    valid = all(checks.values())
    zero = actual["metrics"].get("collision_events") == 0 and actual["metrics"].get("collision_incidents") == 0
    return {
        "protocol": PROTOCOL, "status": "INVALID" if not valid else "PASS" if zero else "FAIL",
        "run_valid": valid, "zero_incident_status": "UNAVAILABLE" if not valid else "PASS" if zero else "FAIL",
        "validity_checks": checks, "arm": arm, "seed": seed, "seed_roster": roster,
        "seed_roster_source": str(comparison.ROSTER_CONFIG), "scenario": "cologne1", "city": "cologne",
        "simulation_window_sec": [comparison.full.BEGIN, comparison.full.END],
        "repaired_sumocfg": str(sumocfg), "reference_result": str(reference_result),
        "source_bindings": bindings, "executor_source": str(executor_source.resolve()),
        "executor_change": "Exact approved G-to-Y branch replacement relative to frozen 85; g remains y.",
        "shared_executor_probe": probe, "driver_file": str(Path(__file__).resolve()),
        "runtime_model": options["--runtime-model"] if arm == "rigid_target_only" else None,
        "fixed_runtime": {"duration_sec": 3600, "warmup_sec": 60, "control_interval_sec": 10,
                          "rigid_prediction_horizon_sec": 450, "residual_coordination_mode": "direct",
                          "residual_cooldown_intervals_override": 0, "models_none_for_phase_pressure": arm == "phase_pressure"},
        **actual, "elapsed_sec": time.monotonic() - started,
        "claim_boundary": "All six cells are new runs with the same shared executor change. Valid collisions remain FAIL; focal passage is diagnostic only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEED_ROSTER, required=True)
    for name in ("sumocfg", "reference-result", "executor-source", "tripinfo", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_cell(sumocfg=args.sumocfg, reference_result=args.reference_result,
                      executor_source=args.executor_source, arm=args.arm, seed=args.seed, tripinfo=args.tripinfo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "run_valid", "zero_incident_status", "arm", "seed", "elapsed_sec")}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
