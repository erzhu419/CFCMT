"""Run the single admitted Jinan PP safety regression and trace the archived incident vehicles."""
import argparse
import json
from pathlib import Path


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def focal_connections(api, tls, participants):
    links = {role: [] for role in participants}
    for index, group in enumerate(api.trafficlight.getControlledLinks(tls)):
        for origin, destination, via in group:
            edges = [api.lane.getEdgeID(origin), api.lane.getEdgeID(destination)]
            for role, vehicle in participants.items():
                if edges == vehicle["route_transition"]:
                    links[role].append(dict(index=index, origin=origin, via=via, destination=destination))
    if any(not values for values in links.values()):
        raise ValueError("The incident movements cannot be bound to the candidate network")
    return links


def sample(api, tls, participants, links, second, applied, compact, stream):
    active = set(api.vehicle.getIDList())
    observations = {}
    in_vicinity = False
    for role, participant in participants.items():
        vehicle = participant["vehicle_id"]
        if vehicle not in active:
            observations[role] = None
            continue
        edge = api.vehicle.getRoadID(vehicle)
        lane = api.vehicle.getLaneID(vehicle)
        approach, outlet = participant["route_transition"]
        if edge == approach:
            location = "approach"
        elif edge == outlet:
            location = "outlet"
        elif edge.startswith(":" + tls + "_"):
            location = "internal"
        else:
            location = "elsewhere"
        observations[role] = dict(location=location, lane=lane, speed_mps=api.vehicle.getSpeed(vehicle),
                                  lane_position_m=api.vehicle.getLanePosition(vehicle))
        if location != "elsewhere":
            in_vicinity = True
            compact["first_observed_sec"][role].setdefault(location, second)
    if in_vicinity or 229 <= second <= 269:
        row = dict(time_sec=second, signal_applied=applied,
                   signal_after_executor=api.trafficlight.getRedYellowGreenState(tls),
                   movement_signals={role: {x["index"]: applied[x["index"]] for x in group}
                                     for role, group in links.items()}, vehicles=observations)
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        if second == 249:
            compact["at_249"] = row


def run(cfg, admission, incident, output):
    gate = json.loads(admission.read_text())
    if gate["status"] != "PASS_STATIC_PRIORITY" or Path(gate["candidate_sumocfg"]).resolve() != cfg:
        raise ValueError("The candidate has not passed its bound static priority audit")
    previous = json.loads(incident.read_text())
    seed, tls = previous["seed"], previous["tls_id"]
    if (seed, tls, previous["time_s"]) != (34110704, "intersection_4_3", 249):
        raise ValueError("Expected the registered native PP incident")
    participants = {role: previous[role] for role in ("right", "straight")}

    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    api = load_libsumo()
    if libsumo_version(api) != "1.22.0":
        raise ValueError("Expected the frozen SUMO 1.22.0 runtime")
    output.mkdir(parents=True, exist_ok=False)
    ledger = v3._new_safety_ledger()
    record = dict(status="INCOMPLETE", seed=seed, sumocfg=str(cfg), static_admission=str(admission),
                  sumo_version="1.22.0", controller="phase_pressure", horizon_sec=3600,
                  started=0, pp_decisions=0, prediction_branches=0, fits=0,
                  vehicle_ids={r: p["vehicle_id"] for r, p in participants.items()},
                  first_observed_sec={r: {} for r in participants}, at_249=None)
    try:
        v3._start_sumo(api, cfg, seed)
        record["started"] = 1
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
        executors = v3.build_safe_phase_executors(api, infos)
        if len(executors) != 12 or any(len(e.feasible_states) != 8 for e in executors.values()):
            raise ValueError("Expected 12 native Jinan junctions with eight stable choices")
        links = focal_connections(api, tls, participants)
        record["focal_links"] = links
        with (output / "focal_seconds_server_only.jsonl").open("x") as stream:
            for epoch in range(0, 3600, 10):
                if api.simulation.getTime() != epoch:
                    raise ValueError("Unexpected PP decision time")
                states = v3._read_graph_states_v3(api, infos, context, graph)
                actions = {t: v3._rule_candidate(states[t], executors[t], "phase_pressure").state
                           for t in sorted(executors)}
                for t in sorted(actions):
                    executors[t].request(actions[t])
                record["pp_decisions"] += 1
                # Match _advance_with_actions' request-once, advance-ten-steps order.
                for _ in range(10):
                    applied = api.trafficlight.getRedYellowGreenState(tls)
                    try:
                        v3._advance_with_actions(sumo_api=api, executors=executors, actions={},
                                                 control_interval_sec=1, collision_mode="fail", safety_ledger=ledger)
                    except v3.UnsafeRolloutError:
                        sample(api, tls, participants, links, int(api.simulation.getTime()), applied, record, stream)
                        raise
                    sample(api, tls, participants, links, int(api.simulation.getTime()), applied, record, stream)
                if (epoch + 10) % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=epoch+10, total=3600, unit="s")), flush=True)
        if api.simulation.getTime() != 3600 or record["at_249"] is None:
            raise ValueError("Incomplete native regression hour or missing incident-time observation")
        record["status"] = "PASS_PP_REGRESSION"
    except v3.UnsafeRolloutError:
        record["status"] = "HOLD_PP_SAFETY"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["safety"] = v3._finalize_safety_ledger(ledger)
        record["final_time_sec"] = api.simulation.getTime() if record["started"] else None
        if record["started"]:
            api.close()
        write(output / "result.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sumocfg", "admission", "incident", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.sumocfg.resolve(), args.admission.resolve(), args.incident, args.output)
    print(json.dumps({k: result[k] for k in ("status", "seed", "final_time_sec")}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS_PP_REGRESSION" else 2)
