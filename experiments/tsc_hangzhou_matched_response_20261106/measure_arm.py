"""Measure one frozen ten-second Hangzhou action from its original PP prefix."""
import json
from pathlib import Path

from collect_tsc_movement_response import applied_states
from collect_tsc_physical_response_trial import unobserved_first_receiving_count, write_json
from cologne_action_branch_checks import physical_snapshot
from source_pilot import TLS_IDS, observe
from tsc_clearance_execution_audit import audit_execution
from tsc_movement_response import MovementResponseObserver
from tsc_physical_response_data import nominal_signal_plan


def pp_actions(v3, states, executors):
    return {tls: v3._rule_candidate(states[tls], executors[tls], "phase_pressure").state
            for tls in TLS_IDS}


def native_start(api, v3, cfg, endpoint):
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
    executors = v3.build_safe_phase_executors(api, infos)
    if set(executors) != set(TLS_IDS) or any(len(e.feasible_states) != 8 for e in executors.values()):
        raise ValueError("Native signals differ from the admitted sixteen-signal package")
    ledger = v3._new_safety_ledger()
    for _ in range(0, endpoint, 10):
        states = v3._read_graph_states_v3(api, infos, context, graph)
        v3._advance_with_actions(sumo_api=api, executors=executors, actions=pp_actions(v3, states, executors),
                                 control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
    if api.simulation.getTime() != endpoint:
        raise ValueError("The native checkpoint differs from the frozen time")
    states = v3._read_graph_states_v3(api, infos, context, graph)
    return infos, executors, pp_actions(v3, states, executors), v3._finalize_safety_ledger(ledger)


def signal_trace(whole_states, links):
    traces = {"".join(state[index] for state in whole_states) for index in links}
    if len(traces) != 1:
        raise ValueError("The movement's links have different signal sequences")
    return traces.pop()


def measure_arm(case, arm, sumocfg, output, on_checkpoint=None, on_branch_start=None):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    cfg, output = Path(sumocfg), Path(output)
    if arm not in ("high", "low"):
        raise ValueError("Expected a high or low arm")
    tls, checkpoint = case["tls_id"], case["time_sec"]
    registry = json.loads((cfg.parent / "movement_pairs.json").read_text())
    admission = json.loads((cfg.parent / "static_admission.json").read_text())
    if (admission["status"] != "PASS_STATIC_ADMISSION" or registry["protocol"] != "right-straight-same-lane-pairs-v1"
            or set(registry["tls"]) != set(TLS_IDS)):
        raise ValueError("Expected the admitted Hangzhou movement registry")
    pair, = (p for p in registry["tls"][tls]
             if p["right_id"] == case["right_id"] and p["straight_id"] == case["straight_id"])
    api = load_libsumo()
    if libsumo_version(api) != "1.22.0":
        raise ValueError("Expected SUMO 1.22.0")
    write_json(output / "prefix_started.json", {"prefix_started": True})
    v3._start_sumo(api, cfg, case["seed"])
    try:
        infos, executors, requests, prefix = native_start(api, v3, cfg, checkpoint)
        executor = executors[tls]
        candidates = list(executor.feasible_states_now())
        if len(set(candidates)) != 8 or len(candidates) != 8:
            raise ValueError("Checkpoint no longer has eight feasible actions")
        supply = observe(api, executor, [pair], candidates)["pairs"][0]
        contrast = supply["contrast"]
        if (any(supply[k] != case[k] for k in ("right_near", "straight_near"))
                or not contrast or any(contrast[k] != case[k] for k in ("high_candidate", "low_candidate"))
                or (contrast["high_exposure_s"], contrast["low_exposure_s"]) != (4, 0)):
            raise ValueError(f"Replayed supply or action contrast differs from the fixed case: {supply}")
        start = physical_snapshot(api, executors)
        start.update(pending_ids=sorted(api.simulation.getPendingVehicles()), candidate_states=candidates,
                     pp_actions=requests.copy(), traffic_lights=applied_states(api, executors),
                     supply=supply)
        write_json(output / "physical_start_server_only.json", start)
        if on_checkpoint is not None:
            on_checkpoint(start)

        index = case[arm + "_candidate"]
        requests[tls] = candidates[index]
        nominal = nominal_signal_plan(executor, requests[tls])["states"]
        observer = MovementResponseObserver(api, {tls: infos[tls]}, checkpoint, windows=(10,))
        for side in ("right", "straight"):
            if set(observer.movements[pair[side + "_id"]]["link_indices"]) != set(pair[side + "_links"]):
                raise ValueError(f"Observed movement links differ from the registry for {side}")
        observer.sample(checkpoint)
        ledger = v3._new_safety_ledger()
        if on_branch_start is not None:
            on_branch_start()
        try:
            for name in sorted(requests):
                executors[name].request(requests[name])
            # Read the state that governs the next second; the executor advances after that step.
            for _ in range(10):
                applied = applied_states(api, executors)
                v3._advance_with_actions(sumo_api=api, executors=executors, actions={},
                                         control_interval_sec=1, collision_mode="fail", safety_ledger=ledger)
                observer.sample(api.simulation.getTime(), applied_states=applied)
        finally:
            observations = dict(frames=observer.frames, events=observer.events, catalog=observer.movements,
                                lane_geometry=observer.lane_geometry, nominal=nominal)
            write_json(output / "observations_server_only.json", observations)

        measurement = observer.summary()
        missed = unobserved_first_receiving_count(observer.frames, observer.movements)
        if (api.simulation.getTime() != checkpoint + 10 or measurement["sample_count"] != 10
                or not measurement["complete"] or missed):
            raise ValueError(f"Incomplete native measurement: samples={measurement['sample_count']}, "
                             f"unresolved={measurement['unresolved_count']}, missing_receiving={missed}")
        audit = audit_execution(observations, nominal, executor.clearance_lane_ids, tls=tls)
        if not audit["matches"]:
            raise ValueError(f"Actual focal signals conflict with the clearance rule: {audit}")
        actual = [frame["applied_states"][tls] for frame in observer.frames[1:]]
        response = measurement["windows"]["10"]["movements"]
        compact = dict(status="COMPLETE", seed=case["seed"], time_sec=checkpoint, arm=arm,
                       candidate_index=index, right_near=case["right_near"], straight_near=case["straight_near"],
                       prefix_safety=prefix, safety=v3._finalize_safety_ledger(ledger),
                       clearance_extension_steps=audit["clearance_extension_steps"], sample_count=10,
                       unresolved_events=measurement["unresolved_count"], unobserved_first_receiving=missed)
        for side in ("right", "straight"):
            links = pair[side + "_links"]
            compact[side + "_nominal"] = signal_trace(nominal, links)
            compact[side + "_actual"] = signal_trace(actual, links)
            compact[side + "_entries"] = response[pair[side + "_id"]]["stopline_entries"]
        compact["exposure_s"] = sum(a == "g" and b == "G" for a, b in
                                    zip(compact["right_actual"], compact["straight_actual"]))
        write_json(output / "arm.json", compact)
        return compact
    finally:
        api.close()
