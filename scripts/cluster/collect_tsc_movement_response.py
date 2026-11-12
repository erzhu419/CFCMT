#!/usr/bin/env python3
"""Four fixed native replays to test movement-response measurement, without fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROTOCOL = "tsc-movement-response-measurement-v1"
WINDOWS = (10, 60, 450)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def validate_spec(spec):
    if (spec["protocol"] != PROTOCOL or spec["horizon_sec"] != 450
            or tuple(spec["windows_sec"]) != WINDOWS or spec["max_new_branches"] != 4
            or spec["new_fits"] != 0 or len(spec["cases"]) != 2
            or {case["city"] for case in spec["cases"]} != {"cologne", "jinan"}):
        raise ValueError("Use the fixed two-city, four-branch, zero-fit measurement pilot")
    for case in spec["cases"]:
        states = case["candidate_states"]
        if (len(states) != 8 or len(set(states)) != 8
                or case["reference_index"] == case["candidate_index"]
                or not all(0 <= case[key] < 8 for key in ("reference_index", "candidate_index"))
                or set(case["expected_costs"]) != {"pp", "candidate"}):
            raise ValueError("Keep the frozen eight actions and distinct PP/candidate pair")


def select_bank_group(bank, case):
    rows = np.flatnonzero(np.asarray(bank.metadata["action_group_ids"]) == case["group_id"])
    if len(rows) != 8 or [bank.metadata["candidate_states"][i] for i in rows] != case["candidate_states"]:
        raise ValueError("Frozen group or candidate order differs from original bank")
    if not np.all(np.asarray(bank.metadata["row_times"])[rows] == case["checkpoint_sec"]):
        raise ValueError("Frozen checkpoint differs from original bank")
    column = bank.feature_names.index("switch_indicator")
    if bank.features[rows[case["candidate_index"]], column] <= 0:
        raise ValueError("Measurement candidate must be the frozen non-stay request")
    return rows


def applied_states(api, executors):
    return {tls: str(api.trafficlight.getRedYellowGreenState(tls)) for tls in sorted(executors)}


def advance_observed(api, v3, executors, actions, lanes, ledger, observer):
    """Original _advance_with_actions ordering, adding only response getters."""
    for tls in sorted(actions):
        executors[tls].request(actions[tls])
    costs = []
    for _ in range(10):
        before = float(api.simulation.getTime())
        applied = applied_states(api, executors)
        api.simulationStep()
        now = float(api.simulation.getTime())
        for tls in sorted(executors):
            executors[tls].advance(max(now - before, 0.0))
        starting = int(api.simulation.getStartingTeleportNumber())
        ending = int(api.simulation.getEndingTeleportNumber())
        collisions = tuple(api.simulation.getCollisions())
        v3._record_safety_step(ledger=ledger, sumo_api=api, executors=executors,
                              starting_teleports=starting, ending_teleports=ending, collisions=collisions)
        if starting or ending:
            raise ValueError("Native response window teleported; measurement is invalid")
        costs.append(float(sum(api.lane.getLastStepHaltingNumber(lane) for lane in lanes)) / len(lanes))
        observer.sample(now, applied_states=applied)
    return costs


def save_observations(output, response, start, preaction, costs):
    if response is not None:
        write_json(output / "observations_server_only.json", {
            "preaction": preaction, "physical_start": start, "events": response.events,
            "frames": response.frames, "costs": costs})


def source_branch(api, v3, bank, case, candidate_index, response_class, output):
    # These unchanged helpers are staged beside this script if absent from the snapshot.
    from run_cologne_training_label_restore_audit import native_prefix, feature_match
    from cologne_action_branch_checks import physical_snapshot

    rows = select_bank_group(bank, case)
    infos, context, executors, states, prefix = native_prefix(
        api, v3, Path(case["sumocfg"]), case["seed"], case["interval_index"])
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    tls = case["tls_id"]
    start = physical_snapshot(api, executors)
    start["pending_ids"] = sorted(api.simulation.getPendingVehicles())
    features = np.vstack([v3.candidate_features_v3(states[tls], v3._candidate_for_state(infos[tls], state),
        executors[tls], control_interval_sec=10) for state in case["candidate_states"]])
    features_match = feature_match(features, bank.features[rows], bank.feature_names)
    checks = {
        "checkpoint_exact": start["time_sec"] == case["checkpoint_sec"],
        "one_signal_eight_lanes": list(infos) == [tls] and len(lanes) == 8,
        "candidate_roster_exact": list(executors[tls].feasible_states_now()) == case["candidate_states"],
        "preaction_features_exact": features_match["passed"],
        "pp_reference_exact": v3._rule_candidate(states[tls], executors[tls], "phase_pressure").state
                              == case["candidate_states"][case["reference_index"]],
        "prefix_no_teleport": prefix["no_teleport_passed"],
    }
    if not all(checks.values()):
        raise ValueError(f"Cologne native prefix mismatch: {checks}")
    response = response_class(api, infos, case["checkpoint_sec"], windows=WINDOWS)
    response.sample(case["checkpoint_sec"])
    costs, ledger = [], v3._new_safety_ledger()
    preaction = {"feature_names": list(bank.feature_names), "features": features.tolist(),
                 "focal_executor": start["executors"][tls]}
    try:
        for interval in range(45):
            current = states if interval == 0 else {
                name: v3._read_local_state(api, name, info, context) for name, info in sorted(infos.items())}
            actions = {name: v3._rule_candidate(current[name], executors[name], "phase_pressure").state
                       for name in sorted(infos)}
            if interval == 0:
                actions[tls] = case["candidate_states"][candidate_index]
            costs.extend(advance_observed(api, v3, executors, actions, lanes, ledger, response))
    finally:
        save_observations(output, response, start, preaction, costs)
    return {"checks": checks, "start": start, "costs": costs, "response": response,
            "safety": v3._finalize_safety_ledger(ledger), "prefix_safety": prefix,
            "preaction": preaction}


def target_branch(api, v3, bank, case, candidate_index, response_class, output):
    from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
    from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import (
        physical_snapshot, compare_physical, _new_window_record, _sample, _record_step_diagnostics, _window_summary)

    rows = select_bank_group(bank, case)
    parent = Path(case["parent_root"])
    old = json.loads((parent / "development_v1" / f"seed_{case['seed']}" / "result.json").read_text())
    group = dict(next(row for row in old["groups"] if row["group_id"] == case["group_id"]))
    names = tuple(name for name in bank.feature_names if name not in v158.RAW_PREDICTION_FEATURES)
    columns = [bank.feature_names.index(name) for name in names]
    group["dataset"] = SimpleNamespace(feature_names=names, context_names=bank.context_names,
        features=bank.features[rows][:, columns], context=bank.context[rows])
    if (group["reference_index"] != case["reference_index"] or group["tls_id"] != case["tls_id"]
            or group["candidate_states"] != case["candidate_states"]):
        raise ValueError("Jinan native parent differs from the frozen case")
    original = pickle.loads(Path(old["full_traces_server_only"]).read_bytes())[case["group_id"]]

    class Originator(v158.FixedActionBranchOriginator):
        def prepare_interval(self, **kwargs):
            super().prepare_interval(**kwargs)
            self.infos = {tls: state.info for tls, state in kwargs["states"].items()}

    originator = Originator(group, candidate_index)
    record, start, response, applied = _new_window_record(), {}, None, None

    def observe(*, stage, time_sec, executors):
        nonlocal start, response, applied
        checkpoint = case["checkpoint_sec"]
        if stage == "before_simulation_step" and checkpoint <= time_sec < checkpoint + 450:
            applied = (time_sec, applied_states(api, executors))
        if stage != "after_executor_advance":
            return
        if time_sec == checkpoint:
            start = physical_snapshot(api, executors)
            response = response_class(api, originator.infos, checkpoint, windows=WINDOWS)
            response.sample(time_sec)
        if checkpoint < time_sec <= checkpoint + 450:
            if response is None or applied is None or applied[0] != time_sec - 1:
                raise ValueError("Missing preaction state or applied signal in native measurement")
            response.sample(time_sec, applied_states=applied[1])
            record["samples"].append(_sample(api, originator.controlled_lanes, time_sec))
            _record_step_diagnostics(api, record)

    preaction = {"feature_names": list(names), "features": group["dataset"].features.tolist()}
    try:
        runtime = v158._evaluate_originator(api=api, v3=v3, sumocfg=Path(case["sumocfg"]),
            scenario=case["scenario"], seed=case["seed"], duration_sec=case["checkpoint_sec"] + 450,
            originator=originator, observer=observe, tripinfo_path=output / "tripinfo.xml")
    finally:
        if start:
            preaction["focal_executor"] = start["executors"][case["tls_id"]]
        save_observations(output, response, start, preaction, [row["cost"] for row in record["samples"]])
    if runtime.get("ok") is not True or not start or response is None:
        raise ValueError(f"Jinan native replay failed: {runtime.get('error')}")
    summary = _window_summary(record, start)
    trace = runtime.get("accepted_intervention_trace", [])
    checks = {
        "native_parent_physical_start_exact": compare_physical(original["baseline_snapshot"], start, tolerance=0)["passed"],
        "preaction_dataset_exact": bool(originator.dataset_match and all(originator.dataset_match.values())),
        "population_accounting": summary["active_population_accounting_passed"],
        "no_teleport": runtime.get("starting_teleports") == runtime.get("ending_teleports") == 0,
        "one_frozen_request": not trace if candidate_index == case["reference_index"] else (
            len(trace) == 1 and trace[0]["time_sec"] == case["checkpoint_sec"]
            and trace[0]["tls_id"] == case["tls_id"] and trace[0]["execution_effective"] is True),
    }
    return {"checks": checks, "start": start, "costs": [row["cost"] for row in record["samples"]],
            "response": response, "safety": {"raw_collision_events": summary["collision_events"],
                "starting_teleports": summary["starting_teleports"], "ending_teleports": summary["ending_teleports"]},
            "preaction": preaction}


def run(spec, output):
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = spec["conversion_root"]
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from tsc_movement_response import MovementResponseObserver
    from cologne_action_branch_checks import compare_physical as compare_source
    from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import compare_physical as compare_target

    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("Use the fixed SUMO 1.22.0 runtime")
    api, results, attempted = load_libsumo(), [], 0
    started = time.monotonic()
    for case in spec["cases"]:
        bank = load_mechanism_dataset(Path(case["bank_path"]))
        pair_start = None
        for arm in ("pp", "candidate"):
            candidate_index = case["reference_index"] if arm == "pp" else case["candidate_index"]
            branch_dir = output / case["city"] / arm
            branch_dir.mkdir(parents=True)
            attempted += 1
            item = {"city": case["city"], "group_id": case["group_id"], "arm": arm,
                    "candidate_index": candidate_index, "status": "INVALID"}
            try:
                branch = (source_branch(api, v3, bank, case, candidate_index, MovementResponseObserver, branch_dir)
                          if case["city"] == "cologne" else
                          target_branch(api, v3, bank, case, candidate_index, MovementResponseObserver, branch_dir))
                if pair_start is None:
                    pair_start = branch["start"]
                costs, response = branch["costs"], branch["response"]
                compare = compare_source if case["city"] == "cologne" else compare_target
                checks = {**branch["checks"],
                    "paired_start_exact": compare(pair_start, branch["start"], tolerance=0)["passed"],
                    "paired_pending_ids_exact": pair_start["pending_ids"] == branch["start"]["pending_ids"],
                    "window_complete": len(costs) == 450,
                    "original_cost_reproduced": abs(float(np.mean(costs)) - case["expected_costs"][arm]) < 1e-12}
                measurement = response.summary()
                checks["movement_accounting_complete"] = measurement["complete"]
                checks["nondegenerate_measured_crossings"] = sum(row["stopline_entries"]
                    for row in measurement["windows"]["450"]["movements"].values()) > 0
                item.update(checks=checks, measurement=measurement, cost=float(np.mean(costs)),
                            safety=branch["safety"], prefix_safety=branch.get("prefix_safety"),
                            status="PASS_MEASUREMENT" if all(checks.values()) else "INVALID")
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
            finally:
                try:
                    api.close()
                except Exception:
                    pass  # evaluate_policy_v3 closes its own connection.
            results.append(item)
            write_json(branch_dir / "result.json", item)
            print("SCHEDULEURM_PROGRESS " + json.dumps({"current": attempted, "total": 4, "unit": "branch",
                "city": case["city"], "arm": arm, "status": item["status"]}), flush=True)
            if item["status"] == "INVALID":
                break
        if results[-1]["status"] == "INVALID":
            break
    result = {"protocol": PROTOCOL, "status": "PASS_MEASUREMENT" if len(results) == 4 and
        all(item["status"] == "PASS_MEASUREMENT" for item in results) else "HOLD_MEASUREMENT",
        "new_fits": 0, "new_branches_attempted": attempted, "branches": results,
        "elapsed_sec": time.monotonic() - started,
        "scope": "Measurement pilot only; no response predictor, transfer benefit or control efficacy tested."}
    write_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.spec.read_text()), args.output)
    print(json.dumps({key: result[key] for key in ("status", "new_fits", "new_branches_attempted")}), flush=True)
