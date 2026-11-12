"""Run the two admitted Hangzhou development hours and preserve preaction opportunities."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from census_stopped import choose, stopped_leaders
from collect_tsc_movement_response import applied_states
from cologne_action_branch_checks import physical_snapshot
from measure_arm import pp_actions
from run_census import pair_rows
from source_pilot import TLS_IDS, observe


SEEDS = (32061101, 32061102)
POINTS = ((1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 2))


def save(path, record):
    path.write_text(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def hour(cfg, pairs, seed, output):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    api = load_libsumo()
    if libsumo_version(api) != "1.22.0":
        raise ValueError("Expected the bound SUMO 1.22.0 runtime")
    output.mkdir()
    record = dict(seed=seed, status="INCOMPLETE", started=0, decisions=0, tls_epochs=0,
                  movement_rows=0, final_time_sec=None, alternative_branches=0, fits=0)
    ledger = v3._new_safety_ledger()
    candidates_by_point = {p: [] for p in POINTS}
    high4 = Counter()
    visits, last = {}, {}
    selected = []
    try:
        v3._start_sumo(api, cfg, seed)
        record["started"] = 1
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
        executors = v3.build_safe_phase_executors(api, infos)
        if set(executors) != set(pairs) or any(len(e.feasible_states) != 8 for e in executors.values()):
            raise ValueError("Runtime signals differ from the admitted Hangzhou network")
        with TemporaryDirectory(dir=output) as tmp, (output / "rows.jsonl").open("x") as stream:
            scratch = Path(tmp)
            for epoch in range(0, 3600, 10):
                if api.simulation.getTime() != epoch:
                    raise ValueError("Unexpected PP decision time")
                states = v3._read_graph_states_v3(api, infos, context, graph)
                actions = pp_actions(v3, states, executors)
                record["decisions"] += 1
                keep_epoch = False
                for tls in TLS_IDS:
                    ex = executors[tls]
                    record["tls_epochs"] += 1
                    for pair, row in zip(pairs[tls], pair_rows(api, ex, pairs[tls], seed, epoch)):
                        record["movement_rows"] += 1
                        key = tls, row["right_id"], row["straight_id"]
                        point = row["right_near"], row["straight_near"]
                        if min(point) > 0:
                            if last.get(key) != epoch - 10:
                                visits[key] = epoch
                            last[key] = epoch
                        if (point in candidates_by_point and row["eight_eligible"]
                                and (row["high_exposure_s"], row["low_exposure_s"]) == (4, 0)):
                            high4[point] += 1
                            lane, = pair["straight_lanes"]
                            ids = stopped_leaders(api, lane, point[1])
                            row.update(straight_lane=lane, straight_target_ids=ids,
                                       both_supplied_visit_start_sec=visits[key])
                            if ids:
                                choices = list(ex.feasible_states_now())
                                supply = observe(api, ex, [pair], choices)["pairs"][0]
                                candidates_by_point[point].append(dict(row, candidate_states=choices, supply=supply))
                                keep_epoch = True
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if keep_epoch:
                    start = physical_snapshot(api, executors)
                    start.update(pending_ids=sorted(api.simulation.getPendingVehicles()), pp_actions=actions,
                                 traffic_lights=applied_states(api, executors))
                    save(scratch / f"{epoch}.json", start)
                v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                                         control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
                if (epoch + 10) % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(dict(seed=seed, current=epoch+10, total=3600, unit="s")),
                          flush=True)
            if (api.simulation.getTime(), record["decisions"], record["tls_epochs"], record["movement_rows"]) != (
                    3600, 360, 5760, 11520):
                raise ValueError("Incomplete Hangzhou hour or registered-pair census")
            selected = choose(POINTS, candidates_by_point, output, scratch)
        record["status"] = "COMPLETE"
    except v3.UnsafeRolloutError:
        record["status"] = "HOLD_PP_SAFETY"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["safety"] = v3._finalize_safety_ledger(ledger)
        if record["started"]:
            record["final_time_sec"] = api.simulation.getTime()
            api.close()
        present = {tuple(row["point"]) for row in selected}
        record["points"] = [
            dict(point=p, high4_low0=high4[p], both_stopped=len(candidates_by_point[p]), selected=p in present)
            for p in POINTS]
        save(output / "result.json", record)
    return record, selected


def run(cfg, admission, output):
    gate = json.loads(admission.read_text())
    if gate["status"] != "PASS_STATIC_PRIORITY" or Path(gate["candidate_sumocfg"]).resolve() != cfg:
        raise ValueError("Hangzhou candidate is not admitted")
    registry_path = (admission.parent / gate["registry"]).resolve()
    registry = json.loads(registry_path.read_text())
    pairs = registry["tls"]
    if (registry["protocol"] != "right-straight-same-lane-pairs-v1" or set(pairs) != set(TLS_IDS)
            or any(len(rows) != 2 for rows in pairs.values())):
        raise ValueError("Expected exactly the original two pairs on each Hangzhou signal")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="RUNNING", sumocfg=str(cfg), rebound_registry=str(registry_path), seeds=SEEDS,
                   points=POINTS, hours=[], observed_cases=[], pp_hours_started=0, pp_hours_complete=0,
                   alternative_branches=0, fits=0, policy_comparisons=0)
    for seed in SEEDS:
        receipt, cases = hour(cfg, pairs, seed, output / f"pp_{seed}")
        summary["hours"].append(receipt)
        summary["observed_cases"].extend(cases)
        summary["pp_hours_started"] += receipt["started"]
        if receipt["status"] != "COMPLETE":
            summary["status"] = receipt["status"]
            break
        summary["pp_hours_complete"] += 1
        save(output / "result.json", summary)
    if summary["pp_hours_complete"] == len(SEEDS):
        summary["status"] = "PASS_SOURCE_PP"
    summary["observed_seed_point_cells"] = len(summary["observed_cases"])
    save(output / "result.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "pp_hours_complete", "observed_seed_point_cells")}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "admission", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.sumocfg.resolve(), args.admission.resolve(), args.output.resolve())
    raise SystemExit(0 if result["status"] == "PASS_SOURCE_PP" else 2)
