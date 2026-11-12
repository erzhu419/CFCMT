"""One of the four frozen native/PP safety and preaction-supply trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path


TLS_IDS = tuple(f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 5))
ANCHORS = (600, 1800)
SEEDS = (31061101, 31061102)


def action_contrast(plans, pair):
    """Match all ten own signal steps and vary protected straight service during own g."""
    if len(plans) != 8 or any(len(plan) != 10 for plan in plans):
        raise ValueError("Expected eight nominal actions of ten seconds")
    views = []
    for index, plan in enumerate(plans):
        right = [set(state[i] for i in pair["right_links"]) for state in plan]
        straight = [set(state[i] for i in pair["straight_links"]) for state in plan]
        # Compare movement-level service only when all its physical links agree,
        # as in the Jinan roster. A mixed protected/yielding movement differs.
        if any(len(chars) != 1 for chars in right + straight):
            continue
        own_seq = "".join(next(iter(chars)) for chars in right)
        priority_seq = "".join(next(iter(chars)) for chars in straight)
        seconds = sum(a == "g" and b == "G" for a, b in zip(own_seq, priority_seq))
        views.append((index, own_seq, priority_seq, seconds))
    eligible = [
        (lo[3] - hi[3], hi[0], lo[0], hi, lo)
        for hi in views for lo in views if hi[1] == lo[1] and hi[3] > lo[3]
    ]
    if not eligible:
        return None
    negative_gap, _, _, high, low = min(eligible)
    return dict(high_candidate=high[0], low_candidate=low[0], right_sequence=high[1],
                high_straight_sequence=high[2], low_straight_sequence=low[2],
                high_exposure_s=high[3], low_exposure_s=low[3], gap_s=-negative_gap)


def observe(api, executor, pairs, candidates):
    from tsc_physical_response_data import nominal_signal_plan

    tls = executor.tls_id
    inlets = sorted(set(api.trafficlight.getControlledLanes(tls)))
    near = {}
    for lane in inlets:
        length, limit = api.lane.getLength(lane), api.lane.getMaxSpeed(lane)
        near[lane] = sum(
            api.vehicle.getLaneID(v) == lane and length - api.vehicle.getLanePosition(v) <= 10 * limit
            for v in set(api.lane.getLastStepVehicleIDs(lane))
        )
    plans = [nominal_signal_plan(executor, candidate)["states"] for candidate in candidates]
    measured = []
    for index, pair in enumerate(pairs):
        if not pair["same_lane_link_pairs"]:
            raise ValueError(f"Static merge registry has no directed same-lane connection at {tls}")
        measured.append(dict(
            registry_index=index, right_id=pair["right_id"], straight_id=pair["straight_id"],
            right_near=sum(near[lane] for lane in set(pair["right_lanes"])),
            straight_near=sum(near[lane] for lane in set(pair["straight_lanes"])),
            contrast=action_contrast(plans, pair),
        ))
    return dict(status="MEASURED", checkpoint_sec=int(api.simulation.getTime()),
                near_by_inlet=near, candidate_states=list(candidates), pairs=measured)


def supply_summary(states):
    observed = [s for s in states if s["status"] == "MEASURED"]
    pairs = [p for s in observed for p in s["pairs"]]
    qualifying = [
        (s, p) for s in observed for p in s["pairs"]
        if p["contrast"] is not None and p["right_near"] > 0 and p["straight_near"] > 0
    ]
    return dict(scheduled_states=len(states),
                state_status=dict(Counter(s["status"] for s in states)),
                directed_static_group_pairs=len(pairs),
                switchable_group_pairs=sum(p["contrast"] is not None for p in pairs),
                supplied_switchable_group_pairs=len(qualifying),
                supplied_switchable_states=len({(s["anchor_sec"], s["tls_id"]) for s, p in qualifying}),
                supplied_switchable_tls=sorted({s["tls_id"] for s, p in qualifying}),
                near_points=sorted({(p["right_near"], p["straight_near"]) for s, p in qualifying}))


def run(cfg, output, seed, controller):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    roster = json.loads((cfg.parent / "movement_pairs.json").read_text())
    if roster["protocol"] != "right-straight-same-lane-pairs-v1" or set(roster["tls"]) != set(TLS_IDS):
        raise ValueError("Expected the admitted physical pair roster for all 16 Hangzhou signals")
    api = load_libsumo()
    version = libsumo_version(api)
    if version != "1.22.0":
        raise ValueError(f"Expected SUMO 1.22.0, got {version}")
    states = [dict(anchor_sec=a, tls_id=t, status="NOT_REACHED") for a in ANCHORS for t in TLS_IDS
              if controller == "pp"]
    output.mkdir(parents=True, exist_ok=False)
    ledger = v3._new_safety_ledger()
    result = dict(protocol="hangzhou-aligned-supply-pilot-v1", seed=seed, controller=controller,
                  sumocfg=str(cfg), sumo_version=version, status="INVALID", trajectories_started=0,
                  horizon_sec=3600, new_alternative_branches=0, new_fits=0)
    clock = None
    try:
        v3._start_sumo(api, cfg, seed)
        result["trajectories_started"] = 1
        clock = float(api.simulation.getTime())
        if clock != 0 or set(api.trafficlight.getIDList()) != set(TLS_IDS):
            raise ValueError("Expected absolute 0s and the fixed sixteen-signal network")
        executors, actions = {}, {}
        if controller == "pp":
            infos = v3._tls_phase_infos(api)
            graph = v3.build_signal_routing_graph(api, infos)
            context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
            executors = v3.build_safe_phase_executors(api, infos)
            if set(executors) != set(TLS_IDS) or any(len(e.feasible_states) != 8 for e in executors.values()):
                raise ValueError("Runtime PP phase repertoire differs from the admitted topology")
        # Empty executor/action dictionaries leave the native program in control.
        for epoch in range(0, 3600, 10):
            if clock != epoch:
                raise ValueError(f"Unexpected epoch {clock}, expected {epoch}")
            if controller == "pp":
                for state in states:
                    offset = epoch - state["anchor_sec"]
                    if state["status"] != "NOT_REACHED" or not 0 <= offset <= 60:
                        continue
                    tls = state["tls_id"]
                    candidates = executors[tls].feasible_states_now()
                    if len(candidates) == 8:
                        state.update(observe(api, executors[tls], roster["tls"][tls], candidates))
                    elif offset == 60:
                        state["status"] = "NO_ELIGIBLE_EPOCH"
                observed = v3._read_graph_states_v3(api, infos, context, graph)
                actions = {t: v3._rule_candidate(observed[t], executors[t], "phase_pressure").state for t in TLS_IDS}
            v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                                     control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
            clock = float(api.simulation.getTime())
            if clock % 600 == 0:
                print("SCHEDULEURM_PROGRESS " + json.dumps(dict(controller=controller, seed=seed,
                      current=int(clock), total=3600, unit="s")), flush=True)
        if clock != 3600 or any(s["status"] == "NOT_REACHED" for s in states):
            raise ValueError("Incomplete hour or unaccounted measurement window")
        result["status"] = "COMPLETE"
    except v3.UnsafeRolloutError as exc:
        result.update(status="HOLD_SAFETY", error=str(exc).split("safety_audit=")[0].rstrip(", "))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if result["trajectories_started"]:
            clock = float(api.simulation.getTime())
            api.close()
        result.update(final_time_sec=clock, safety=v3._finalize_safety_ledger(ledger))
        if controller == "pp":
            result["supply"] = supply_summary(states)
            (output / "states.json").write_text(json.dumps(states, separators=(",", ":")) + "\n")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--controller", choices=("native", "pp"), required=True)
    args = parser.parse_args()
    cell = run(args.sumocfg.resolve(), args.output, args.seed, args.controller)
    print(json.dumps(cell), flush=True)
    raise SystemExit({"COMPLETE": 0, "HOLD_SAFETY": 2, "INVALID": 1}[cell["status"]])
