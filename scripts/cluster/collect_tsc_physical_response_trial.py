#!/usr/bin/env python3
"""Collect fixed native action groups for short physical-response prediction.

Every action restarts the original seed and follows PP to the declared state.
The movement observer runs for ten seconds; the unchanged PP continuation
retains the 450-second network cost. Full observations stay on the server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROTOCOL = "tsc-physical-response-development-v1"
CONTROL_SEC = 10
COST_SEC = 450


class IneligibleCase(ValueError):
    """The fixed TLS has no eight-action epoch inside its declared window."""


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def validate_spec(spec, indices):
    cases = spec["cases"]
    if (spec["protocol"] != PROTOCOL or spec["response_horizon_sec"] != CONTROL_SEC
            or spec["cost_horizon_sec"] != COST_SEC
            or spec["checkpoint_search_sec"] != 60
            or spec["max_new_branches"] != len(cases) * 8
            or len({case["case_id"] for case in cases}) != len(cases)):
        raise ValueError("Use the frozen eight-action native response collection protocol")
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(cases) for i in indices):
        raise ValueError("Case indices must be a distinct subset of the frozen roster")
    for case in cases:
        if case["city"] not in {"cologne", "jinan"} or int(case["checkpoint_sec"]) % CONTROL_SEC:
            raise ValueError("Fixed cases require a supported city and ten-second checkpoint")


def native_prefix(api, v3, case):
    """Use the original PP collector's step ordering without state restoration."""
    v3._start_sumo(api, Path(case["sumocfg"]), int(case["seed"]))
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=Path(case["sumocfg"]), infos=infos,
                                     control_interval_sec=CONTROL_SEC)
    executors = v3.build_safe_phase_executors(api, infos)
    begin = float(api.simulation.getTime())
    requested = float(case.get("resolved_checkpoint_sec", case["checkpoint_sec"]))
    remaining = requested - begin
    if remaining < 0 or remaining % CONTROL_SEC:
        raise ValueError("Frozen checkpoint is not a native ten-second control epoch")
    ledger = v3._new_safety_ledger()
    for _ in range(int(remaining // CONTROL_SEC)):
        states = v3._read_graph_states_v3(api, infos, context, graph)
        actions = {tls: v3._rule_candidate(states[tls], executors[tls], "phase_pressure").state
                   for tls in sorted(infos)}
        v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
            control_interval_sec=CONTROL_SEC, collision_mode="audit", safety_ledger=ledger)
    states = v3._read_graph_states_v3(api, infos, context, graph)
    if case["tls_id"] not in executors:
        raise ValueError("Frozen focal TLS is absent from the network")
    if "resolved_checkpoint_sec" not in case:
        deadline = float(case["checkpoint_sec"]) + case["checkpoint_search_sec"]
        while len(executors[case["tls_id"]].feasible_states_now()) != 8:
            if float(api.simulation.getTime()) >= deadline:
                raise IneligibleCase("Fixed TLS never became eight-action eligible within anchor+60 seconds")
            actions = {tls: v3._rule_candidate(states[tls], executors[tls], "phase_pressure").state
                       for tls in sorted(infos)}
            v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                control_interval_sec=CONTROL_SEC, collision_mode="audit", safety_ledger=ledger)
            states = v3._read_graph_states_v3(api, infos, context, graph)
    return infos, graph, context, executors, states, v3._finalize_safety_ledger(ledger)


def candidate_roster(v3, states, executors, case):
    tls = case["tls_id"]
    if tls not in states:
        raise ValueError("Frozen focal TLS is absent from the network")
    candidates = list(executors[tls].feasible_states_now())
    if len(candidates) != 8 or len(set(candidates)) != 8:
        raise ValueError("Each fixed action group must retain eight distinct candidate phases")
    pp = v3._rule_candidate(states[tls], executors[tls], "phase_pressure").state
    return candidates, candidates.index(pp)


def network_preaction(v3, states, executors, graph):
    """Save observed target topology/state for the later frozen network readout."""
    return {
        "feature_names": list(v3.FEATURE_NAMES_V3),
        "tls_ids": sorted(states),
        "phase_pressure_features": [v3.candidate_features_v3(states[tls],
            v3._rule_candidate(states[tls], executors[tls], "phase_pressure"),
            executors[tls], control_interval_sec=CONTROL_SEC).tolist() for tls in sorted(states)],
        "adjacency": {tls: list(neighbors) for tls, neighbors in graph.adjacency.items()},
        "downstream_tls_by_lane": graph.downstream_tls_by_lane,
    }


def unobserved_first_receiving_count(frames, movements):
    """Detect first observations that may have skipped an entire junction step."""
    receiving = {movement["outgoing_edge"] for movement in movements.values()}
    seen = set(frames[0]["vehicles"]) if frames else set()
    count = 0
    for frame in frames[1:]:
        for vehicle in set(frame["vehicles"]) - seen:
            row = frame["vehicles"][vehicle]
            count += int(row["road_id"] in receiving and row["route_index"] > 0)
        seen.update(frame["vehicles"])
    return count


def collect_branch(api, v3, case, index, output, response_class, build_rows, extract_labels):
    from collect_tsc_movement_response import advance_observed
    from cologne_action_branch_checks import physical_snapshot

    infos, graph, context, executors, states, prefix = native_prefix(api, v3, case)
    tls = case["tls_id"]
    candidates, reference = candidate_roster(v3, states, executors, case)
    index = reference if index is None else index
    checkpoint = int(api.simulation.getTime())
    start = physical_snapshot(api, executors)
    start["pending_ids"] = sorted(api.simulation.getPendingVehicles())
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    actions = {name: v3._rule_candidate(states[name], executors[name], "phase_pressure").state
               for name in sorted(infos)}
    actions[tls] = candidates[index]
    response = response_class(api, infos, checkpoint, windows=(CONTROL_SEC,))
    preaction = build_rows(api, infos, executors, actions, response.movements,
                          focal_tls=tls, horizon_sec=CONTROL_SEC)
    network = network_preaction(v3, states, executors, graph)
    focal_features = v3.candidate_features_v3(states[tls],
        v3._candidate_for_state(infos[tls], candidates[index]), executors[tls],
        control_interval_sec=CONTROL_SEC).tolist()
    response.sample(checkpoint)
    costs, ledger = [], v3._new_safety_ledger()
    try:
        costs.extend(advance_observed(api, v3, executors, actions, lanes, ledger, response))
        for _ in range(COST_SEC // CONTROL_SEC - 1):
            states = v3._read_graph_states_v3(api, infos, context, graph)
            actions = {name: v3._rule_candidate(states[name], executors[name], "phase_pressure").state
                       for name in sorted(infos)}
            raw_costs = v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                control_interval_sec=CONTROL_SEC, cost_lanes=lanes, collision_mode="audit", safety_ledger=ledger)
            costs.extend(float(value) / len(lanes) for value in raw_costs)
    finally:
        write_json(output / "observations_server_only.json", {
            "preaction": preaction, "events": response.events, "frames": response.frames,
            "costs": costs, "catalog": response.movements, "lane_geometry": response.lane_geometry})
    measurement = response.summary()
    first_receiving_count = unobserved_first_receiving_count(response.frames, response.movements)
    labels = extract_labels(measurement, preaction["movement_ids"], horizon_sec=CONTROL_SEC)
    expected_checkpoint = case.get("resolved_checkpoint_sec", checkpoint)
    checks = {"native_checkpoint_exact": start["time_sec"] == expected_checkpoint,
              "inside_frozen_checkpoint_window": case["checkpoint_sec"] <= checkpoint <= case["checkpoint_sec"] + case["checkpoint_search_sec"],
              "response_samples_complete": measurement["sample_count"] == CONTROL_SEC,
              "movement_accounting_complete": measurement["complete"],
              "first_receiving_entries_observed": first_receiving_count == 0,
              "full_cost_horizon": len(costs) == COST_SEC,
              "end_time_exact": float(api.simulation.getTime()) == checkpoint + COST_SEC,
              "prefix_no_teleport": prefix["no_teleport_passed"]}
    return {"start": start, "candidate_states": candidates, "reference_index": reference,
            "candidate_index": index, "resolved_checkpoint_sec": checkpoint,
            "preaction": preaction, "network_preaction": network, "focal_features": focal_features,
            "labels": labels, "label_names": ["stopline_entries_10s", "receiving_entries_10s"],
            "cost": float(np.mean(costs)), "costs": costs, "controlled_lane_count": len(lanes),
            "checks": checks, "measurement": measurement,
            "unobserved_first_receiving_count": first_receiving_count,
            "prefix_safety": prefix, "safety": v3._finalize_safety_ledger(ledger)}


def collect_case(api, v3, case, output, response_class, build_rows, extract_labels, progress):
    from cologne_action_branch_checks import compare_physical

    output.mkdir(parents=True, exist_ok=False)
    branches, summaries, first = [], [], None
    indices, resolved_case = [None], dict(case)
    for execution_index in range(8):
        index = indices[execution_index]
        branch_dir = output / ("first_pp" if index is None else f"action_{index}")
        branch_dir.mkdir()
        summary = {"candidate_index": index, "status": "INVALID"}
        try:
            branch = collect_branch(api, v3, resolved_case, index, branch_dir, response_class, build_rows, extract_labels)
            index = branch["candidate_index"]
            summary["candidate_index"] = index
            if first is None:
                first = branch
                resolved_case["resolved_checkpoint_sec"] = branch["resolved_checkpoint_sec"]
                indices.extend(i for i in range(8) if i != branch["reference_index"])
                write_json(output / "physical_start_server_only.json", first["start"])
            checks = {**branch["checks"],
                      "same_native_state": compare_physical(first["start"], branch["start"], tolerance=0)["passed"],
                      "same_pending_population": first["start"]["pending_ids"] == branch["start"]["pending_ids"],
                      "same_candidate_roster": first["candidate_states"] == branch["candidate_states"],
                      "same_pp_reference": first["reference_index"] == branch["reference_index"],
                      "same_movement_roster": first["preaction"]["movement_ids"] == branch["preaction"]["movement_ids"],
                      "same_network_preaction": first["network_preaction"] == branch["network_preaction"]}
            summary.update(status="COMPLETE" if all(checks.values()) else "INVALID", checks=checks,
                cost=branch["cost"], movement_rows=len(branch["labels"]),
                measured_stopline_entries=sum(row[0] for row in branch["labels"]),
                measured_receiving_entries=sum(row[1] for row in branch["labels"]),
                unresolved_events=branch["measurement"]["unresolved_count"],
                unobserved_first_receiving_count=branch["unobserved_first_receiving_count"],
                prefix_safety=branch["prefix_safety"], safety=branch["safety"])
            branches.append({"candidate_index": index, "candidate_state": branch["candidate_states"][index],
                "preaction": branch["preaction"], "labels": branch["labels"],
                "focal_features": branch["focal_features"], "cost": branch["cost"], "costs": branch["costs"]})
        except IneligibleCase as error:
            summary.update(status="INELIGIBLE", error=f"{type(error).__name__}: {error}")
        except Exception as error:
            summary["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                api.close()
            except Exception:
                pass
        summaries.append(summary)
        write_json(branch_dir / "result.json", summary)
        progress(case, summary)
        if summary["status"] != "COMPLETE":
            break
    complete = len(summaries) == 8 and all(b["status"] == "COMPLETE" for b in summaries)
    result = {"case_id": case["case_id"], "city": case["city"], "seed": case["seed"],
              "status": "COMPLETE" if complete else ("INELIGIBLE" if summaries[-1]["status"] == "INELIGIBLE" else "INVALID"),
              "anchor_sec": case["checkpoint_sec"], "resolved_checkpoint_sec": resolved_case.get("resolved_checkpoint_sec"),
              "new_branches_attempted": len(summaries), "branches": summaries}
    if first is not None:
        branches.sort(key=lambda row: row["candidate_index"])
        data = {"case": case, "candidate_states": first["candidate_states"],
                "case_id": case["case_id"], "city": case["city"], "seed": case["seed"],
                "anchor_sec": case["checkpoint_sec"], "resolved_checkpoint_sec": resolved_case["resolved_checkpoint_sec"],
                "reference_index": first["reference_index"], "controlled_lane_count": first["controlled_lane_count"],
                "network_preaction": first["network_preaction"], "label_names": first["label_names"],
                "feature_names": first["preaction"]["feature_names"],
                "movement_ids": first["preaction"]["movement_ids"],
                "X": [b["preaction"]["features"] for b in branches],
                "Y": [b["labels"] for b in branches],
                "costs_450": [b["cost"] for b in branches],
                "branches": branches}
        write_json(output / "group_data_server_only.json", data)
    write_json(output / "result.json", result)
    return result


def run(spec, indices, output):
    validate_spec(spec, indices)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = spec["conversion_root"]
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from tsc_movement_response import MovementResponseObserver
    from tsc_physical_response_data import build_preaction_rows, extract_response_labels

    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("Use the frozen SUMO 1.22.0 runtime")
    started, attempted, groups = time.monotonic(), 0, []

    def progress(case=None, branch=None):
        nonlocal attempted
        attempted += int(branch is not None)
        print("SCHEDULEURM_PROGRESS " + json.dumps({"current": attempted, "total": len(indices) * 8,
              "unit": "branch", "elapsed_seconds": time.monotonic() - started,
              "case_id": case["case_id"] if case else None,
              "status": branch["status"] if branch else "START"}), flush=True)

    progress()
    api = load_libsumo()
    for index in indices:
        case = {**spec["cases"][index], "checkpoint_search_sec": spec["checkpoint_search_sec"]}
        result = collect_case(api, v3, case, output / case["case_id"], MovementResponseObserver,
                              build_preaction_rows, extract_response_labels, progress)
        groups.append(result)
        if result["status"] != "COMPLETE":
            break
    result = {"protocol": PROTOCOL, "status": "COMPLETE" if len(groups) == len(indices)
              and all(g["status"] == "COMPLETE" for g in groups) else "HOLD_COLLECTION",
              "case_indices": indices, "groups": groups, "new_fits": 0,
              "new_branches_attempted": attempted, "elapsed_sec": time.monotonic() - started}
    write_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--case-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.spec.read_text()), [int(i) for i in args.case_indices.split(",")], args.output)
    print(json.dumps({key: result[key] for key in ("status", "new_fits", "new_branches_attempted")}), flush=True)
