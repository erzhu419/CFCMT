"""Single-action interventions at fixed free-event ordinals after exact native prefixes."""

import argparse
import json
from pathlib import Path
import random
import time

from native_runtime import (
    HERE, CausalArrivalRates, bootstrap, causal_event, city_line_mapping,
    load_environment, observe_arrivals, project_executed_action,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tracker_state(tracker):
    return (tracker.seen.copy(), {key: tuple(value) for key, value in tracker.arrivals.items()})


def candidates(api, torch, actors, state, raw, guard, config):
    with torch.inference_mode():
        states = torch.from_numpy(state[None])
        requested = {name: actor(states, deterministic=True)[0].numpy()[0]
            for name, actor in actors.items()}
        requested["masked_positive"] = actors["masked"].positive_action_and_log_prob(
            states, deterministic=True)[0].numpy()[0]
    requested["zero"] = api.np.asarray([-1.0], api.np.float32)
    executed = {name: project_executed_action(api, raw, action, guard, config)
        for name, action in requested.items()}
    holds = {name: float(api.contract.policy_action_to_hold_seconds_v7(action[0], config))
        for name, action in executed.items()}
    return requested, executed, holds


def run_day(api, torch, actors, protocol, native, seed, assets, reference=None, candidate=None):
    random.seed(seed)
    api.np.random.seed(seed)
    torch.manual_seed(seed)
    horizon, config = protocol["episode_seconds"], api.contract.DEFAULT_CONFIG_V7
    guard = native["smoke"]["completion_guard"]
    env, manifest = load_environment(api, assets, horizon, protocol["case"]["sumo_seed"])
    require(manifest["demand_seed"] == protocol["case"]["demand_seed"], "Demand seed differs")
    tracker = CausalArrivalRates(native["observation"]["arrival_rate_window_seconds"])
    trace, selected, holds = [], [], []
    event_index = batch_index = terminal_events = prefix_terminals = free_ordinal = 0
    target, reached = (reference["selected"] if reference else None), False
    started = time.monotonic()
    try:
        env.reset()
        version = list(api.traci.getVersion())
        require("SUMO " + native["smoke"]["sumo_version"] in version[1], "SUMO version differs")
        line_ids, stations = city_line_mapping(env)
        env.step_to_event({})
        while not env.done:
            observe_arrivals(api, tracker, env)
            events = sorted(env._current_events, key=api.event_sort_key)
            signature = [(float(e.sim_time), str(e.bus_id), str(e.line_id), str(e.stop_id),
                bool(api.is_terminal_event(env, e))) for e in events]
            controls, actions = [], {}
            for event in events:
                if api.is_terminal_event(env, event):
                    actions[event.bus_id] = [0.0, 1.0]
                    terminal_events += 1
                else:
                    controls.append((event, causal_event(api, tracker, env, event)))
            if target and batch_index <= target["batch_index"]:
                require(signature == reference["trace"][batch_index]["events"],
                    f"Native prefix event mismatch at batch {batch_index}")
                prefix_terminals += len(events)-len(controls)
            executed_actions = []
            if controls:
                batch = api.np.stack([raw for _, raw in controls])
                prior = api.np.full(len(controls), -1.0, api.np.float32)
                for slot, (event, _) in enumerate(controls):
                    raw = api.contract.pack_event_batch_v7(batch, slot, prior, config)
                    state = api.contract.raw_state_to_canonical_v7(raw, stations, config)
                    bounds = [float(api.completion_safety_action(raw,
                        api.np.asarray([a], api.np.float32), guard, config)[0]) for a in (-1.0, 1.0)]
                    is_free = bounds[0] < bounds[1]
                    capture = not reference and is_free and free_ordinal in protocol["free_ordinals"]
                    intervene = target and event_index == target["event_index"]
                    if capture or intervene:
                        requested, executed, seconds = candidates(api, torch, actors, state, raw, guard, config)
                    else:
                        with torch.inference_mode():
                            action = actors["adapted"](torch.from_numpy(state[None]), deterministic=True)[0].numpy()[0]
                        executed = {"adapted": project_executed_action(api, raw, action, guard, config)}
                        seconds = {"adapted": float(api.contract.policy_action_to_hold_seconds_v7(executed["adapted"][0], config))}
                    pair = (float(executed["adapted"][0]), seconds["adapted"])
                    if target and event_index < target["event_index"]:
                        require(pair == reference["trace"][batch_index]["actions"][slot],
                            f"Native prefix action mismatch at event {event_index}")
                    if capture or intervene:
                        current = dict(free_ordinal=free_ordinal, event_index=event_index,
                            batch_index=batch_index, batch_slot=slot, time_seconds=float(event.sim_time),
                            bus_id=str(event.bus_id), line_id=str(event.line_id), stop_id=str(event.stop_id),
                            raw=raw.tolist(), canonical=state.tolist(), prior=prior.tolist(), bounds=bounds,
                            requested={name: value.tolist() for name, value in requested.items()},
                            executed={name: value.tolist() for name, value in executed.items()}, holds=seconds,
                            ledger=env._bridge.passenger_time_ledger.snapshot().to_dict(), tracker=tracker_state(tracker))
                        if capture:
                            selected.append(current)
                        else:
                            require(current == target, "Selected native state, tracker, ledger or candidates differ")
                            reached = True
                    name = candidate if intervene else "adapted"
                    prior[slot] = executed[name][0]
                    actions[event.bus_id] = [seconds[name], 1.0]
                    executed_actions.append((float(executed[name][0]), seconds[name]))
                    holds.append(seconds[name])
                    event_index += 1
                    free_ordinal += int(is_free)
            if not reference:
                trace.append(dict(events=signature, actions=executed_actions))
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
            batch_index += 1
        ledger = env._bridge.passenger_time_ledger.snapshot()
        passengers = api.summarize_passengers(env._bridge, ledger)
        require(ledger.time_seconds == horizon and not passengers["active_state_count_balance_error"],
            "Final native ledger horizon or passenger balance differs")
        require(reached if target else [row["free_ordinal"] for row in selected] == protocol["free_ordinals"],
            "Replay failed to reach every fixed selected event")
        cost = 2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds
        result = dict(horizon_seconds=ledger.time_seconds, simulator_version=version, line_ids=line_ids,
            ledger=ledger.to_dict(), passengers=passengers, service=api.summarize_service(env._bridge),
            terminal_events=terminal_events, event_batches=batch_index, control_events=event_index,
            generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"])
        checks = dict(exact=True, event_batches=target["batch_index"]+1,
            executed_control_actions=target["event_index"], terminal_actions=prefix_terminals,
            selected_identity=True, selected_raw=True, selected_canonical=True, selected_batch_prior=True,
            selected_arrival_tracker=True, selected_ledger=True, selected_candidate_actions=True) if target else None
        return dict(result=result, selected=selected, trace=trace, same_prefix_checks=checks,
            mean_hold_seconds=float(api.np.mean(holds)), positive_holds=int(api.np.sum(api.np.asarray(holds)>0)),
            elapsed_seconds=time.monotonic()-started)
    finally:
        env.close()


def attach_existing_scores(protocol, seed, selected, baseline, actors, assets):
    source = HERE / protocol["source_preference_folder"] / f"seed{seed}"
    summary = json.loads((source / "summary.json").read_text())
    require(summary["complete"] and summary["training_seed"] == seed and summary["case"] == protocol["case"],
        "Existing native preference identity differs")
    require(summary["actors"] == actors and summary["asset_manifest"] == str(assets), "Existing actor or assets differ")
    require(summary["baseline"] == baseline, "Native baseline differs from existing complete preference baseline")
    wanted = {row["event_index"]: row for row in selected}
    source_fields = ("event_index", "batch_index", "batch_slot", "time_seconds", "bus_id", "line_id",
        "stop_id", "canonical", "prior", "bounds", "requested", "executed", "holds")
    found = set()
    with (source / "events.jsonl").open() as stream:
        for line in stream:
            original = json.loads(line)
            if original["event_index"] not in wanted:
                continue
            row = wanted[original["event_index"]]
            require(all(row[key] == original[key] for key in source_fields), "Existing selected event metadata differs")
            require(set(original["critic"]) == {"adapted", "unmasked", "masked"}, "Existing critic stages differ")
            for values in original["critic"].values():
                require(set(values) == set(protocol["candidate_names"]) and all(
                    item["min"] == min(item["heads"]) for item in values.values()), "Existing critic candidates or min differ")
            row["critic_scores"] = original["critic"]
            found.add(original["event_index"])
    require(found == set(wanted), "Existing preference event records are missing")
    return dict(summary=str(source / "summary.json"), events=str(source / "events.jsonl"),
        baseline_exact=True, selected_metadata_exact=True, critics=summary["critics"],
        critic_reloads=0)


def effects(result, baseline):
    return dict(generalized_passenger_seconds=result["generalized_passenger_seconds"]-baseline["generalized_passenger_seconds"],
        generalized_seconds_per_departed_passenger=result["generalized_seconds_per_departed_passenger"]-
            baseline["generalized_seconds_per_departed_passenger"],
        wait_passenger_seconds=result["ledger"]["accrued_wait_seconds"]-baseline["ledger"]["accrued_wait_seconds"],
        in_vehicle_passenger_seconds=result["ledger"]["accrued_in_vehicle_seconds"]-baseline["ledger"]["accrued_in_vehicle_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "native_fixed_action_replay_protocol.json")
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    require(args.seed in protocol["training_seeds"], "Seed is not one of the frozen training seeds")
    native = json.loads((HERE / "protocol.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    assets = Path(runtime["assets_root"]) / protocol["asset_folder"] / "case00/asset_manifest.json"
    api = bootstrap()
    import torch
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    torch.set_num_threads(1)
    actors, paths = {}, {}
    for name, spec in protocol["actors"].items():
        path = HERE / spec["folder"] / f"seed{args.seed}" / spec["filename"]
        actors[name], _ = load_actor_checkpoint_v7(path, "cpu", require_policy_artifact=True)
        actors[name].eval().requires_grad_(False)
        paths[name] = str(path)
    baseline = run_day(api, torch, actors, protocol, native, args.seed, assets)
    provenance = attach_existing_scores(protocol, args.seed, baseline["selected"], baseline["result"], paths, assets)
    report = dict(complete=False, protocol=protocol["name"], training_seed=args.seed, case=protocol["case"],
        updates=0, free_ordinals=protocol["free_ordinals"], candidate_names=protocol["candidate_names"],
        asset_manifest=str(assets), actors=paths, baseline=baseline["result"], selected=[],
        provenance=provenance, actual_simulations=1,
        interpretation="Negative cost differences favor intervention; conditional on adapted deterministic continuation. Q is a scaled undiscounted IQL/H2O+ continuation value, not a calibrated cost label under that continuation.",
        trace_storage="Exact baseline trace and tracker snapshots retained only in server process memory")
    output = HERE / "native_fixed_action_replay" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    def write_progress():
        report["elapsed_seconds"] = time.monotonic()-started
        (output / "progress.json").write_text(json.dumps(report, indent=2)+"\n")
    write_progress()
    for selected in baseline["selected"]:
        visible = {key: value for key, value in selected.items() if key != "tracker"}
        visible["arrival_tracker_counts"] = dict(seen_person_stop_pairs=len(selected["tracker"][0]),
            streams=len(selected["tracker"][1]), recorded_line_arrivals=sum(map(len, selected["tracker"][1].values())))
        visible["candidates"] = {}
        report["selected"].append(visible)
        for name in protocol["candidate_names"]:
            previous = next((other for other in visible["candidates"] if selected["executed"][name] ==
                selected["executed"][other]), None)
            if name == "adapted":
                branch = dict(result=baseline["result"], same_prefix_checks=None, elapsed_seconds=0.0)
                reused_from, actual = "baseline", False
            elif previous is not None:
                branch = visible["candidates"][previous]
                reused_from, actual = previous, False
            else:
                reference = dict(selected={key: value for key, value in selected.items() if key != "critic_scores"}, trace=baseline["trace"])
                branch = run_day(api, torch, actors, protocol, native, args.seed, assets, reference, name)
                require(branch["result"]["line_ids"] == baseline["result"]["line_ids"] and
                    branch["result"]["passengers"]["departed_passengers"] == baseline["result"]["passengers"]["departed_passengers"],
                    "Counterfactual service mapping or departed population differs")
                report["actual_simulations"] += 1
                reused_from, actual = None, True
            visible["candidates"][name] = dict(complete=True, requested=selected["requested"][name],
                executed=selected["executed"][name], hold_seconds=selected["holds"][name],
                result=branch["result"], actual_effect_vs_adapted=effects(branch["result"], baseline["result"]),
                same_prefix_checks=branch["same_prefix_checks"], actual_simulation=actual,
                identical_execution=(previous is not None or name == "adapted"), reused_from=reused_from,
                elapsed_seconds=branch["elapsed_seconds"])
            write_progress()
            print(json.dumps(dict(seed=args.seed, free_ordinal=selected["free_ordinal"], candidate=name,
                actual_simulation=actual, reused_from=reused_from,
                effect=visible["candidates"][name]["actual_effect_vs_adapted"])), flush=True)
    report["complete"] = True
    write_progress()
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True, seed=args.seed, actual_simulations=report["actual_simulations"])), flush=True)


if __name__ == "__main__":
    main()
