"""A 60-second same-state hold-versus-safe-switch mechanism diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from cf_h2o.eval.traffic_signal_cologne_service_replay import TLS_ID, observe_lane
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    _finalize_safety_ledger, _new_safety_ledger, _record_safety_step,
    _restore_executors,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    _reload_sumo_state, _start_sumo, _tls_phase_infos,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors


PROTOCOL = "tsc-v154-cologne-same-state-service-intervention-v1"
START_SEC = 26700.0
HORIZON_SEC = 60
HOLD_PHASE = "rrrGGrrrrrrrrGGrrrrr"
SWITCH_PHASE = "GGGggrrrrrGGGggrrrrr"
HEAD_LANES = ("-32038056#3_1", "28198821#3_1")
POSITION_ROUNDING_BOUND_M = 0.5e-8


def state_has_rng(path: Path) -> bool:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("start",)):
            if str(element.tag).rsplit("}", 1)[-1] == "rngState":
                return True
    return False


def key_observation(sample: Mapping[str, Any], *, legacy: bool) -> dict[str, Any]:
    lanes = {}
    for name, row in sample["lanes"].items():
        lane = {key: row[key] for key in ("vehicles", "halting", "mean_speed_mps")}
        # V154 replay-v1's pct label is incorrect: the stored getter value is a fraction.
        lane["occupancy_fraction"] = row["occupancy_pct" if legacy else "occupancy_fraction"]
        if "head" in row:
            lane["head"] = row["head"]
            lane["next_tls_link_counts"] = row["next_tls_link_counts"]
        lanes[name] = lane
    return {
        "time_sec": sample["time_sec"],
        "active_inserted_vehicles": sample["active_inserted_vehicles"],
        "pending_insertion_vehicles": sample["pending_insertion_vehicles"],
        "executor_before": sample["executor_before"],
        "lanes": lanes,
    }


def compare_hold(reference_samples, observed_samples) -> dict[str, Any]:
    expected = {float(row["time_sec"]): key_observation(row, legacy=True)
                for row in reference_samples
                if START_SEC <= float(row["time_sec"]) <= START_SEC + HORIZON_SEC}
    actual = {float(row["time_sec"]): key_observation(row, legacy=False)
              for row in observed_samples}
    mismatches: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []

    def visit(old, new, path):
        if isinstance(old, Mapping):
            if not isinstance(new, Mapping) or set(old) != set(new):
                mismatches.append({"path": path, "reason": "mapping_keys_or_type"})
                return
            for key in old:
                visit(old[key], new[key], f"{path}.{key}")
        elif isinstance(old, list):
            if not isinstance(new, list) or len(old) != len(new):
                mismatches.append({"path": path, "reason": "list_length_or_type"})
                return
            for index, value in enumerate(old):
                visit(value, new[index], f"{path}[{index}]")
        elif old != new:
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                delta = abs(float(old) - float(new))
                position_field = path.endswith(".position_m") or path.endswith(".next_tls[1]")
                within_precision = position_field and delta <= POSITION_ROUNDING_BOUND_M
                difference = {"path": path, "original": old, "restored": new,
                              "absolute_difference": delta,
                              "within_declared_position_rounding": within_precision}
                differences.append(difference)
                if not within_precision:
                    mismatches.append(difference)
            else:
                mismatches.append({"path": path, "original": old, "restored": new})

    if set(expected) != set(actual) or len(expected) != 7:
        mismatches.append({"path": "sample_times", "expected": sorted(expected), "actual": sorted(actual)})
    for now in sorted(set(expected) & set(actual)):
        visit(expected[now], actual[now], str(now))
    return {
        "passed": not mismatches,
        "exact_match": not mismatches and not differences,
        "sample_count": len(actual),
        "state_save_precision_decimal_places": 8,
        "position_and_next_tls_distance_bound_m": POSITION_ROUNDING_BOUND_M,
        "other_fields": "exact equality",
        "numeric_difference_count": len(differences),
        "maximum_numeric_absolute_difference": max(
            (row["absolute_difference"] for row in differences), default=0.0
        ),
        "numeric_differences": differences,
        "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
    }


def observe_key_state(api, executors, lane_catalog):
    lanes = {}
    for lane, metadata in lane_catalog.items():
        row, _ = observe_lane(api, lane, None, incoming=bool(metadata["incoming"]))
        row.pop("sampled_lane_enters")
        row.pop("sampled_lane_leaves")
        lanes[lane] = row
    executor = executors[TLS_ID].snapshot()
    executor.pop("audit")
    return {
        "time_sec": float(api.simulation.getTime()),
        "active_inserted_vehicles": int(api.vehicle.getIDCount()),
        "pending_insertion_vehicles": len(api.simulation.getPendingVehicles()),
        "executor_before": executor, "lanes": lanes,
    }


def incoming_membership(api, lanes):
    return {str(vehicle): lane for lane in lanes
            for vehicle in api.lane.getLastStepVehicleIDs(lane)}


def crossing_events(previous, current, active_ids, locations, *, now, receiving_lanes):
    events = []
    for vehicle in sorted(set(previous) - set(current)):
        if vehicle not in active_ids:
            continue
        lane, road = locations(vehicle)
        if lane.startswith(f":{TLS_ID}_"):
            destination = "internal"
        elif lane in receiving_lanes:
            destination = "receiving"
        else:
            continue
        events.append({"time_sec": now, "vehicle_id": vehicle,
                       "from_lane": previous[vehicle], "to_lane": lane,
                       "to_road": road, "destination": destination})
    return events


def run_branch(api, *, sumocfg, state_path, executor_context, executors,
               replay, phase_state, branch):
    _reload_sumo_state(api, sumocfg, 41242, state_path,
                       begin_time=START_SEC, save_state_rng=True)
    _restore_executors(executors, executor_context["executors"])
    if float(api.simulation.getTime()) != START_SEC:
        raise RuntimeError("Restored SUMO time differs from the captured state")
    incoming = [lane for lane, metadata in replay["lane_catalog"].items() if metadata["incoming"]]
    receiving = set(replay["lane_catalog"]) - set(incoming)
    first = observe_key_state(api, executors, replay["lane_catalog"])
    observations = [first]
    heads = {lane: {"vehicle_id": first["lanes"][lane]["head"]["id"],
                    "initial_next_tls": first["lanes"][lane]["head"]["next_tls"],
                    "expected_receiving_edge": first["lanes"][lane]["head"]["next_route_edge"],
                    "first_internal_time_sec": None,
                    "first_expected_receiving_edge_time_sec": None,
                    "arrival_time_sec": None}
             for lane in HEAD_LANES}
    request_accepted = executors[TLS_ID].request(phase_state)
    previous = incoming_membership(api, incoming)
    entries = []
    departures = arrivals = 0
    seconds = []
    ledger = _new_safety_ledger()
    for elapsed in range(1, HORIZON_SEC + 1):
        before = float(api.simulation.getTime())
        api.simulationStep()
        now = float(api.simulation.getTime())
        for executor in executors.values():
            executor.advance(now - before)
        starting = int(api.simulation.getStartingTeleportNumber())
        ending = int(api.simulation.getEndingTeleportNumber())
        _record_safety_step(ledger=ledger, sumo_api=api, executors=executors,
                            starting_teleports=starting, ending_teleports=ending,
                            collisions=tuple(api.simulation.getCollisions()))
        departures += int(api.simulation.getDepartedNumber())
        arrivals += int(api.simulation.getArrivedNumber())
        arrived_ids = set(api.simulation.getArrivedIDList())
        current = incoming_membership(api, incoming)
        active_ids = set(api.vehicle.getIDList())
        locations = lambda vehicle: (str(api.vehicle.getLaneID(vehicle)), str(api.vehicle.getRoadID(vehicle)))
        entries.extend(crossing_events(previous, current, active_ids, locations,
                                      now=now, receiving_lanes=receiving))
        for head in heads.values():
            vehicle = head["vehicle_id"]
            if vehicle in active_ids:
                lane, road = locations(vehicle)
                if lane.startswith(f":{TLS_ID}_") and head["first_internal_time_sec"] is None:
                    head["first_internal_time_sec"] = now
                if road == head["expected_receiving_edge"] and head["first_expected_receiving_edge_time_sec"] is None:
                    head["first_expected_receiving_edge_time_sec"] = now
            if vehicle in arrived_ids and head["arrival_time_sec"] is None:
                head["arrival_time_sec"] = now
        previous = current
        seconds.append({"time_sec": now, "signal": executors[TLS_ID].current_state,
                        "active_inserted": len(active_ids),
                        "pending": len(api.simulation.getPendingVehicles()),
                        "departed_since_branch_start": departures,
                        "arrived_since_branch_start": arrivals})
        if elapsed % 10 == 0:
            observations.append(observe_key_state(api, executors, replay["lane_catalog"]))
        if starting or ending:
            break
    return {
        "branch": branch, "requested_phase_state": phase_state,
        "switch_request_accepted": bool(request_accepted),
        "elapsed_simulation_sec": float(api.simulation.getTime()) - START_SEC,
        "key_observations": observations,
        "original_shared_lane_heads": heads,
        "incoming_to_internal_or_receiving_events": entries,
        "intersection_entry_count": len(entries),
        "intersection_entry_counts_by_from_lane": dict(Counter(row["from_lane"] for row in entries)),
        "one_second_state": seconds,
        "endpoint": seconds[-1],
        "safety": _finalize_safety_ledger(ledger),
    }


def run_diagnostic(replay_path: Path):
    started = time.monotonic()
    replay = json.loads(replay_path.read_text())
    if replay["status"] != "PASS" or (replay["scenario"], replay["seed"]) != ("cologne1", 41242):
        raise ValueError("The intervention requires the accepted original Cologne replay")
    options = replay["original_runtime_arguments"]
    saved = replay["saved_snapshot"]
    state_path = Path(saved["sumo_state_path"])
    context = json.loads(Path(saved["executor_context_path"]).read_text())
    if not state_has_rng(state_path):
        raise ValueError("Saved SUMO state does not contain the original RNG state")
    if context["executors"][TLS_ID]["current_green_state"] != HOLD_PHASE:
        raise ValueError("Snapshot no longer contains the diagnosed held phase")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = load_traffic_signal_manifest(Path(options["--manifest"]))
    sumocfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    api = load_libsumo()
    if libsumo_version() != "1.22.0":
        raise ValueError("Same-state diagnostic requires the original SUMO 1.22.0 runtime")
    _start_sumo(api, sumocfg, 41242, save_state_rng=True)
    try:
        infos = _tls_phase_infos(api)
        executors = build_safe_phase_executors(api, infos)
        if set(executors) != {TLS_ID} or SWITCH_PHASE not in executors[TLS_ID].feasible_states:
            raise ValueError("Original TLS candidate set does not contain the frozen intervention")
        hold = run_branch(api, sumocfg=sumocfg, state_path=state_path,
                          executor_context=context, executors=executors, replay=replay,
                          phase_state=HOLD_PHASE, branch="hold_current")
        reproduction = compare_hold(replay["samples"], hold["key_observations"])
        reproduction["hold_complete_and_incident_free"] = bool(
            hold["elapsed_simulation_sec"] == HORIZON_SEC
            and hold["safety"]["no_teleport_passed"]
            and hold["safety"]["raw_collision_events"] == 0
        )
        reproduction["passed"] = bool(
            reproduction["passed"] and reproduction["hold_complete_and_incident_free"]
        )
        switch = None
        if reproduction["passed"]:
            switch = run_branch(api, sumocfg=sumocfg, state_path=state_path,
                                executor_context=context, executors=executors, replay=replay,
                                phase_state=SWITCH_PHASE, branch="safe_switch_to_straight")
    finally:
        api.close()
    status = "REPRODUCTION_FAILED"
    if switch is not None:
        status = "PASS" if (
            switch["elapsed_simulation_sec"] == HORIZON_SEC
            and switch["safety"]["no_teleport_passed"]
        ) else "INVALID_BRANCH"
    return {
        "protocol": PROTOCOL, "status": status,
        "replay_result": str(replay_path), "state_path_remote_only": str(state_path),
        "window_sec": [START_SEC, START_SEC + HORIZON_SEC], "seed": 41242,
        "sumo_version": libsumo_version(), "rng_state_present": True,
        "hold_reproduction": reproduction,
        "mechanism_contrast_incident_free": bool(
            switch is not None and switch["safety"]["raw_collision_events"] == 0
            and switch["safety"]["no_teleport_passed"]
        ),
        "branches": {"hold_current": hold, "safe_switch_to_straight": switch},
        "elapsed_sec": time.monotonic() - started,
        "interpretation_scope": "fixed-state mechanism intervention; no learned policy efficacy claim",
        "measurement_notes": {
            "occupancy_fraction": "unaltered SUMO getter values; legacy replay-v1 incorrectly named these occupancy_pct",
            "crossing_events": "per-second observed incoming-to-internal/receiving transitions; incoming lane changes excluded",
            "arrived_since_branch_start": "new completed trips during the 60-second branch, not the full-run total",
            "state_precision": "only head positions and next-TLS distances may differ by at most half of 10^-8 m; other reproduction fields exact",
            "controller": "one original executor request at 26700, then hold; no action reselection",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(args.replay_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "hold_reproduction": result["hold_reproduction"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
