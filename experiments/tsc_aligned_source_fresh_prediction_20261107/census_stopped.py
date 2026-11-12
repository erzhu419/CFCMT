"""Freeze six preaction points in each new native Jinan seed before opening any arm."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from collect_tsc_movement_response import applied_states
from cologne_action_branch_checks import physical_snapshot
from measure_target import pp_actions
from run_census import pair_rows
from source_pilot import observe


def save(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def stopped_leaders(api, lane, near_count):
    length = api.lane.getLength(lane)
    threshold = 10 * api.lane.getMaxSpeed(lane)
    ranked = sorted((length - api.vehicle.getLanePosition(v), v)
                    for v in set(api.lane.getLastStepVehicleIDs(lane)) if api.vehicle.getLaneID(v) == lane)
    near = [v for distance, v in ranked if distance <= threshold]
    if len(near) != near_count:
        raise ValueError("Census supply disagrees with the candidate's straight vehicles")
    return near[:2] if len(near) >= 2 and all(api.vehicle.getSpeed(v) == 0 for v in near[:2]) else []


def choose(points, options, directory, scratch):
    cases, times, visits = [], set(), set()
    for point in sorted(points):
        for row in sorted(options[point], key=lambda r: (r["time_sec"], r["tls_id"], r["right_id"], r["straight_id"])):
            visit = (row["tls_id"], row["right_id"], row["straight_id"], row["both_supplied_visit_start_sec"])
            epoch = row["time_sec"]
            if epoch in times or visit in visits:
                continue
            times.add(epoch)
            visits.add(visit)
            case = {k: v for k, v in row.items() if k not in ("supply", "candidate_states")}
            start = json.loads((scratch / f"{epoch}.json").read_text())
            start.update(supply=row["supply"], candidate_states=row["candidate_states"])
            target = directory / f"start_{epoch}.json"
            save(target, start)
            case.update(start_file=str(target.resolve()), point=list(point))
            cases.append(case)
            break
    return cases


def census(cfg, registry, output, seed, points):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    pairs = registry["tls"]
    api = load_libsumo()
    if libsumo_version(api) != "1.22.0":
        raise ValueError("Expected the frozen SUMO 1.22.0 runtime")
    output.mkdir()
    ledger = v3._new_safety_ledger()
    last, starts = {}, {}
    options = {p: [] for p in points}
    point_exposures = Counter()
    record = dict(seed=seed, status="INCOMPLETE", tls_epochs=0, pair_rows=0, final_time_sec=None)
    v3._start_sumo(api, cfg, seed)
    try:
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
        executors = v3.build_safe_phase_executors(api, infos)
        if set(executors) != set(pairs) or any(len(e.feasible_states) != 8 for e in executors.values()):
            raise ValueError("Native signals differ from the admitted Jinan registry")
        with TemporaryDirectory(dir=output) as tmp, (output / "rows.jsonl").open("x") as stream:
            scratch = Path(tmp)
            for epoch in range(0, 3600, 10):
                if api.simulation.getTime() != epoch:
                    raise ValueError("Census epoch does not match native simulator time")
                states = v3._read_graph_states_v3(api, infos, context, graph)
                requests = pp_actions(v3, states, executors)
                admitted = False
                for tls in sorted(pairs):
                    executor = executors[tls]
                    record["tls_epochs"] += 1
                    for pair, row in zip(pairs[tls], pair_rows(api, executor, pairs[tls], seed, epoch)):
                        record["pair_rows"] += 1
                        point = row["right_near"], row["straight_near"]
                        key = tls, row["right_id"], row["straight_id"]
                        if min(point) > 0:
                            if last.get(key) != epoch - 10:
                                starts[key] = epoch
                            last[key] = epoch
                        if (point in options and row["eight_eligible"]
                                and (row["high_exposure_s"], row["low_exposure_s"]) == (4, 0)):
                            point_exposures[point] += 1
                            lane, = pair["straight_lanes"]
                            target_ids = stopped_leaders(api, lane, row["straight_near"])
                            row.update(straight_lane=lane, straight_target_ids=target_ids,
                                       both_supplied_visit_start_sec=starts[key])
                            if target_ids:
                                candidates = list(executor.feasible_states_now())
                                supply = observe(api, executor, [pair], candidates)["pairs"][0]
                                options[point].append(dict(row, candidate_states=candidates, supply=supply))
                                admitted = True
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if admitted:
                    start = physical_snapshot(api, executors)
                    start.update(pending_ids=sorted(api.simulation.getPendingVehicles()),
                                 pp_actions=requests, traffic_lights=applied_states(api, executors))
                    save(scratch / f"{epoch}.json", start)
                v3._advance_with_actions(sumo_api=api, executors=executors, actions=requests,
                                         control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
                if (epoch + 10) % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(dict(seed=seed, current=epoch + 10,
                                                                   total=3600, unit="s")), flush=True)
            record["final_time_sec"] = api.simulation.getTime()
            if (record["final_time_sec"], record["tls_epochs"], record["pair_rows"]) != (3600, 4320, 8640):
                raise ValueError("Incomplete preaction census")
            record["safety"] = v3._finalize_safety_ledger(ledger)
            if any(record["safety"][k] for k in ("raw_collision_events", "starting_teleports", "ending_teleports")):
                raise ValueError("Unsafe PP trajectory")
            cases = choose(points, options, output, scratch)
        present = {tuple(case["point"]) for case in cases}
        record.update(status="COMPLETE", missing_points=[p for p in points if p not in present],
                      points=[dict(point=p, high4_low0=point_exposures[p], both_stopped=len(options[p]),
                                   selected=p in present) for p in points])
        return record, cases
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["final_time_sec"] = api.simulation.getTime()
        record["safety"] = v3._finalize_safety_ledger(ledger)
        api.close()
        save(output / "result.json", record)


def main(plan_path, cfg, registry_path, root):
    plan = json.loads(plan_path.read_text())
    registry = json.loads(registry_path.read_text())
    admission = json.loads((registry_path.parent / "static_admission.json").read_text())
    if (admission["status"] != "PASS_STATIC_ADMISSION" or Path(admission["sumocfg"]) != cfg
            or registry["protocol"] != "right-straight-same-lane-pairs-v1" or len(registry["tls"]) != 12
            or any(len(ps) != 2 for ps in registry["tls"].values())):
        raise ValueError("Expected the admitted native Jinan movement registry and matching network")
    seeds = plan["target_seeds"]
    points = sorted(tuple(r[:2]) for r in plan["development_counts"]["rows"])
    if len(set(seeds)) != len(seeds) or len(seeds) != 4 or len(set(points)) != len(points) or len(points) != 6:
        raise ValueError("Expected the frozen four-seed, six-point plan")
    root.mkdir(parents=True, exist_ok=True)
    summaries, roster = [], []
    for seed in seeds:
        summary, cases = census(cfg, registry, root / f"pp_{seed}", seed, points)
        summaries.append(summary)
        roster.extend(cases)
    complete = all(not s["missing_points"] for s in summaries)
    result = dict(status="PASS_BALANCED_COVERAGE" if complete else "HOLD_FRESH_TARGET_COVERAGE",
                  seeds=seeds, points=points, per_seed=summaries, selected_cells=len(roster),
                  cases=roster if complete else [], alternative_branches=0, fits=0)
    save(root / "selection.json", result)
    print(json.dumps(dict(status=result["status"], selected_cells=len(roster))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sumocfg", "registry", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    main(args.plan, args.sumocfg.resolve(), args.registry.resolve(), args.output_root.resolve())
