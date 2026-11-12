"""Collect one native PP hour of physical arrivals and pre-request signal opportunities."""
import argparse
from collections import Counter
import gzip
import json
from math import ceil
from pathlib import Path

INTERVAL = 10
HORIZON = 3600
NETWORKS = {"hangzhou": (16, 48), "jinan": (12, 34)}


def dump_line(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def decision_row(v3, nominal_plan, gp_spec, api, state, executor, tracked, measurement, t):
    """All controller calls here only read the live state. Candidate plans run on isolated copies."""
    legal = list(executor.feasible_states_now())
    pp = gp = 0
    if len(legal) > 1:
        pp = legal.index(v3._rule_candidate(state, executor, "phase_pressure").state)
        gp_state = v3._pressure_candidate_v3(state, executor, gp_spec, control_interval_sec=INTERVAL).state
        gp = legal.index(gp_state)

    groups = state.info.controlled_links
    halted = {lane: int(api.lane.getLastStepHaltingNumber(lane))
              for lane in {connection[0] for group in groups for connection in group}}
    phase_queue = {}
    base = []
    masks = {edge: [] for edge in tracked}
    for request in legal:
        phases = nominal_plan(executor, request, horizon_sec=INTERVAL)["states"]
        for phase in set(phases):
            if phase not in phase_queue:
                served_lanes = {connection[0] for i, group in enumerate(groups)
                                if phase[i] in "Gg" for connection in group}
                phase_queue[phase] = sum(halted[lane] for lane in served_lanes)
        base.append(sum(phase_queue[phase] for phase in phases))
        for edge, indices in tracked.items():
            masks[edge].append(sum(1 << bit for bit, phase in enumerate(phases)
                                   if all(phase[i] == "G" for i in indices)))
    return dict(t=t, tls=executor.tls_id, states=legal, pp=pp, gp=gp, base=base,
                protected=masks, **measurement)


def opportunity(row):
    """Find an observable moving vehicle that can receive different nominal service this interval."""
    if len(row["states"]) == 1 or row["spillback"] or row["unmatched"]:
        return None
    for flight in row["flights"]:
        if not flight["on_straight_lane"]:
            continue
        eta = flight["analytic_eta_s"] + flight["tail_s"]
        first = max(1, ceil(eta))
        if first > INTERVAL:
            continue
        masks = row["protected"][flight["edge"]]
        credits = [(mask >> (first - 1)).bit_count() for mask in masks]
        if max(credits) != min(credits):
            return dict(t=row["t"], tls=row["tls"], edge=flight["edge"], vid=flight["vid"],
                        entry_s=flight["entry_s"], age_s=flight["age_s"], eta_s=eta,
                        first_credit_second=first, credits=credits)
    return None


def hour(city, cfg, seed, output):
    from arrival_sensor import ArrivalSensor
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
    from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
    from tsc_physical_response_data import nominal_signal_plan

    output.mkdir(parents=True, exist_ok=False)
    counts = Counter(dict(tls_epochs=0, controllable_epochs=0, flight_epochs=0,
                          spillback_epochs=0, unmatched_epochs=0, opportunity_epochs=0))
    ledger = v3._new_safety_ledger()
    result = dict(city=city, seed=seed, sumocfg=str(cfg), status="INCOMPLETE", started=0,
                  elapsed_sec=None, horizon_sec=HORIZON, decision_interval_sec=INTERVAL,
                  controller="phase_pressure", counts=counts, opportunity_example=None)
    sensor = None
    sampled = -1
    try:
        api = load_libsumo()
        result["sumo_version"] = libsumo_version(api)
        if result["sumo_version"] != "1.22.0":
            raise ValueError("The corridor study binds SUMO 1.22.0")
        sensor = ArrivalSensor(cfg)
        result["links"] = sensor.links
        gp_spec = PressurePolicySpec(downstream_occupancy_weight=0.02)
        result["gp_spec"] = gp_spec.to_dict()
        v3._start_sumo(api, cfg, seed)
        result["started"] = 1
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=INTERVAL)
        executors = v3.build_safe_phase_executors(api, infos)
        if ((len(executors), len(sensor.links)) != NETWORKS[city]
                or any(len(ex.feasible_states) != 8 for ex in executors.values())):
            raise ValueError("Junction, feasible-state or monitored-link count differs from the repaired network")
        tracked = {tls: {} for tls in executors}
        for edge, link in sensor.links.items():
            tls, lane, indices = link["tls"], link["lane"], link["indices"]
            groups = infos[tls].controlled_links
            if not indices or not all(any(c[0] == lane for c in groups[i]) for i in indices):
                raise ValueError(f"Unbound straight signal indices: {tls}, {edge}, {lane}")
            tracked[tls][edge] = indices

        sensor.sample(api, 0)
        sampled = 0
        with gzip.open(output / "decisions.jsonl.gz", "xt", compresslevel=1) as decisions:
            for t in range(0, HORIZON, INTERVAL):
                if api.simulation.getTime() != t:
                    raise ValueError(f"PP decision and physical observation clocks differ at {t}")
                states = v3._read_graph_states_v3(api, infos, context, graph)
                requests = {}
                for tls in sorted(executors):
                    row = decision_row(v3, nominal_signal_plan, gp_spec, api, states[tls], executors[tls],
                                       tracked[tls], sensor.decision(tls), t)
                    dump_line(decisions, row)
                    requests[tls] = row["states"][row["pp"]]
                    counts["tls_epochs"] += 1
                    counts["controllable_epochs"] += len(row["states"]) > 1
                    counts["flight_epochs"] += bool(row["flights"])
                    counts["spillback_epochs"] += row["spillback"]
                    counts["unmatched_epochs"] += row["unmatched"]
                    example = opportunity(row)
                    if example is not None:
                        counts["opportunity_epochs"] += 1
                        if result["opportunity_example"] is None:
                            result["opportunity_example"] = example
                # Match _advance_with_actions' request-once, then ten audited native steps.
                for tls in sorted(requests):
                    executors[tls].request(requests[tls])
                for sec in range(t + 1, t + INTERVAL + 1):
                    v3._advance_with_actions(sumo_api=api, executors=executors, actions={},
                                             control_interval_sec=1, collision_mode="fail", safety_ledger=ledger)
                    sensor.sample(api, sec)
                    sampled = sec
                if sampled % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(dict(city=city, seed=seed, current=sampled,
                                                                    total=HORIZON, unit="s")), flush=True)
        if api.simulation.getTime() != HORIZON or counts["tls_epochs"] != len(executors) * HORIZON // INTERVAL:
            raise ValueError("Missing native clock time or TLS decision epochs")
        result["status"] = "PASS_HOUR"
    except v3.UnsafeRolloutError as exc:
        result.update(status="HOLD_PP_SAFETY", error=str(exc))
    except Exception as exc:
        result.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    finally:
        if result["started"]:
            result["elapsed_sec"] = api.simulation.getTime()
            api.close()
        result["safety"] = v3._finalize_safety_ledger(ledger)
        try:
            if sampled >= 0:
                labels, summary = sensor.finish(sampled)
                result["sensor"] = summary
                with gzip.open(output / "labels.jsonl.gz", "xt", compresslevel=1) as records:
                    for label in labels:
                        dump_line(records, label)
                if result["status"] == "PASS_HOUR" and not (
                        summary["accounting_ok"] and summary["frames"] == HORIZON + 1
                        and summary["clock_from_s"] == 0 and summary["clock_to_s"] == HORIZON
                        and summary["unresolved_entries"] == 0 and summary["counts_by_status"]["unknown"] == 0
                        and len(labels) == summary["entry_count"]):
                    result["status"] = "HOLD_OBSERVABILITY"
        except Exception as exc:
            result["sensor_error"] = f"{type(exc).__name__}: {exc}"
            if result["status"] == "PASS_HOUR":
                result["status"] = "INCOMPLETE"
        (output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=NETWORKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sumocfg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = hour(args.city, args.sumocfg.resolve(), args.seed, args.output.resolve())
    print(json.dumps({key: result.get(key) for key in ("city", "seed", "status", "elapsed_sec", "counts", "error",
                                                        "sensor_error")}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS_HOUR" else 2)
