"""Full-horizon action interventions after an exactly replayed native prefix."""

import argparse
import json
from pathlib import Path
import random
import time

from native_runtime import (
    HERE, CausalArrivalRates, bootstrap, causal_event, city_line_mapping,
    load_environment, observe_arrivals, project_executed_action,
)


def tracker_state(tracker):
    return (tracker.seen.copy(), {key: tuple(value) for key, value in tracker.arrivals.items()})


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_day(api, torch, actors, protocol, assets, native, mode, reference=None):
    seed, case = protocol["training_seed"], protocol["case"]
    random.seed(seed)
    api.np.random.seed(seed)
    torch.manual_seed(seed)
    horizon, config = protocol["episode_seconds"], api.contract.DEFAULT_CONFIG_V7
    env, manifest = load_environment(api, assets, horizon, case["sumo_seed"])
    require(manifest["demand_seed"] == case["demand_seed"], "Demand seed differs from frozen case")
    tracker = CausalArrivalRates(native["observation"]["arrival_rate_window_seconds"])
    trace, selected, holds = [], None, []
    event_index = batch_index = prefix_terminals = 0
    reached = False
    started = time.monotonic()
    target = reference["selected"] if reference else None
    guard = native["smoke"]["completion_guard"]
    try:
        env.reset()
        version = api.traci.getVersion()
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
                else:
                    controls.append((event, causal_event(api, tracker, env, event)))
            if target and batch_index <= target["batch_index"]:
                require(signature == reference["trace"][batch_index]["events"],
                    f"Native prefix event mismatch at batch {batch_index}")
                prefix_terminals += len(events) - len(controls)
            executed_actions = []
            if controls:
                batch = api.np.stack([raw for _, raw in controls])
                prior = api.np.full(len(controls), -1.0, api.np.float32)
                for index, (event, _) in enumerate(controls):
                    raw = api.contract.pack_event_batch_v7(batch, index, prior, config)
                    state = api.contract.raw_state_to_canonical_v7(raw, stations, config)
                    with torch.inference_mode():
                        requested = {name: actor(torch.from_numpy(state[None]), deterministic=True)[0].numpy()[0]
                            for name, actor in actors.items()}
                    executed = {name: project_executed_action(api, raw, action, guard, config)
                        for name, action in requested.items()}
                    seconds = {name: float(api.contract.policy_action_to_hold_seconds_v7(action[0], config))
                        for name, action in executed.items()}
                    adapted_pair = (float(executed["adapted"][0]), seconds["adapted"])
                    if target and event_index < target["event_index"]:
                        require(adapted_pair == reference["trace"][batch_index]["actions"][index],
                            f"Native prefix executed action mismatch at event {event_index}")
                    if target and event_index == target["event_index"]:
                        require(api.np.array_equal(raw, target["raw"]), "Selected raw state mismatch")
                        require(api.np.array_equal(state, target["canonical"]), "Selected canonical state mismatch")
                        require(api.np.array_equal(prior, target["prior"]), "Selected batch prior mismatch")
                        require(env._bridge.passenger_time_ledger.snapshot().to_dict() == target["ledger"],
                            "Selected passenger ledger mismatch")
                        require(tracker_state(tracker) == target["tracker"], "Selected arrival tracker mismatch")
                        require(seconds == target["holds"] and all(api.np.array_equal(executed[name],
                            target["executed"][name]) for name in actors), "Selected candidate action mismatch")
                        reached = True
                    if reference is None and horizon-float(event.sim_time) >= protocol["minimum_remaining_seconds"]:
                        bounds = [api.completion_safety_action(raw, api.np.asarray([a], api.np.float32),
                            guard, config)[0] for a in (-1.0, 1.0)]
                        gap = abs(seconds["masked"]-seconds["adapted"])
                        if bounds[1] > bounds[0] and (selected is None or gap > selected["gap_seconds"]):
                            selected = dict(event_index=event_index, batch_index=batch_index, batch_slot=index,
                                event=signature[[e.bus_id for e in events].index(event.bus_id)],
                                gap_seconds=gap, raw=raw.copy(), canonical=state.copy(), prior=prior.copy(),
                                executed={name: value.copy() for name, value in executed.items()},
                                holds=seconds.copy(), bounds=[float(value) for value in bounds],
                                ledger=env._bridge.passenger_time_ledger.snapshot().to_dict(),
                                tracker=tracker_state(tracker))
                    use_mask = target and (event_index == target["event_index"] or
                        (mode == "continuous" and event_index > target["event_index"]))
                    name = "masked" if use_mask else "adapted"
                    prior[index] = executed[name][0]
                    actions[event.bus_id] = [seconds[name], 1.0]
                    executed_actions.append((float(executed[name][0]), seconds[name]))
                    holds.append(seconds[name])
                    event_index += 1
            if reference is None:
                trace.append(dict(events=signature, actions=executed_actions))
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
            batch_index += 1
        end = env._bridge.passenger_time_ledger.snapshot()
        passengers = api.summarize_passengers(env._bridge, end)
        require(end.time_seconds == horizon, f"Final ledger horizon mismatch: {end.time_seconds}")
        require(not passengers["active_state_count_balance_error"], "Final passenger ledger does not balance")
        if target:
            require(reached, "Replay did not reach the selected native event")
        else:
            require(selected is not None and selected["gap_seconds"] > 0,
                "No eligible free native event has a nonzero executed action difference")
        cost = 2*end.accrued_wait_seconds+end.accrued_in_vehicle_seconds
        result = dict(complete=True, mode=mode, horizon_seconds=end.time_seconds,
            simulator_version=version, line_ids=line_ids, ledger=end.to_dict(), passengers=passengers,
            service=api.summarize_service(env._bridge), generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            decision_events=len(holds), mean_hold_seconds=float(api.np.mean(holds)),
            positive_holds=int(api.np.sum(api.np.asarray(holds)>0)), elapsed_seconds=time.monotonic()-started)
        if target:
            result["same_prefix_checks"] = dict(exact=True, event_batches=target["batch_index"]+1,
                executed_control_actions=target["event_index"], terminal_actions=prefix_terminals,
                selected_raw=True, selected_canonical=True, selected_batch_prior=True,
                selected_arrival_tracker=True, selected_ledger=True, selected_candidate_actions=True)
        print(json.dumps(dict(mode=mode, complete=True, cost=result["generalized_seconds_per_departed_passenger"])), flush=True)
        return dict(result=result, selected=selected, trace=trace)
    finally:
        env.close()


def critic_scores(torch, protocol, selected):
    import h5py
    from diagnose_values import learner_from_payload
    from train_native import runtime_api
    base, config_class, _, _ = runtime_api()
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        contract = json.loads(data.attrs["config"])
    scores = {}
    states = torch.as_tensor(selected["canonical"], dtype=torch.float32)[None].repeat(2, 1)
    actions = torch.as_tensor([selected["executed"][name].tolist() for name in ("adapted", "masked")])
    for name, spec in protocol["critic_stages"].items():
        path = HERE / spec["folder"] / f"seed{protocol['training_seed']}" / spec["filename"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["total_updates"] == spec["stored_updates"], f"Unexpected {name} critic stage")
        learner = learner_from_payload(base, config_class, contract, payload, protocol["training_seed"])
        with torch.inference_mode():
            q = learner.critic(states, actions).min(0).values.reshape(-1).tolist()
        scores[name] = dict(checkpoint=str(path), adapted_q=q[0], masked_q=q[1],
            masked_minus_adapted_q=q[1]-q[0], prefers_masked=q[1]>q[0],
            reward_scale=learner.config.reward_scale)
        del learner, payload
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "native_action_replay_protocol.json")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "native_action_replay")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    native = json.loads((HERE / "protocol.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    assets = args.assets or Path(runtime["assets_root"]) / protocol["asset_folder"] / "case00/asset_manifest.json"
    api = bootstrap()
    import torch
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    torch.set_num_threads(1)
    actors, paths = {}, {}
    for name, spec in protocol["actors"].items():
        path = HERE / spec["folder"] / f"seed{protocol['training_seed']}" / spec["filename"]
        actors[name], _ = load_actor_checkpoint_v7(path, "cpu", require_policy_artifact=True)
        actors[name].eval()
        paths[name] = str(path)
    baseline = run_day(api, torch, actors, protocol, assets, native, "adapted")
    branches = {"adapted": baseline["result"]}
    for mode in ("single", "continuous"):
        branches[mode] = run_day(api, torch, actors, protocol, assets, native, mode, baseline)["result"]
        require(branches[mode]["line_ids"] == branches["adapted"]["line_ids"], "Line mapping differs")
        require(branches[mode]["passengers"]["departed_passengers"] ==
            branches["adapted"]["passengers"]["departed_passengers"], "Counterfactual passenger population differs")
    selected = baseline["selected"]
    q = critic_scores(torch, protocol, selected)
    visible_selected = {key: value for key, value in selected.items() if key != "tracker"}
    for key in ("raw", "canonical", "prior"):
        visible_selected[key] = selected[key].tolist()
    visible_selected["executed"] = {key: value.tolist() for key, value in selected["executed"].items()}
    visible_selected["arrival_tracker_counts"] = dict(seen_person_stop_pairs=len(selected["tracker"][0]),
        streams=len(selected["tracker"][1]), recorded_line_arrivals=sum(map(len, selected["tracker"][1].values())))
    effects = {}
    for mode in ("single", "continuous"):
        effects[mode] = dict(generalized_passenger_seconds=branches[mode]["generalized_passenger_seconds"]-
            branches["adapted"]["generalized_passenger_seconds"],
            generalized_seconds_per_departed_passenger=branches[mode]["generalized_seconds_per_departed_passenger"]-
            branches["adapted"]["generalized_seconds_per_departed_passenger"],
            wait_passenger_seconds=branches[mode]["ledger"]["accrued_wait_seconds"]-
            branches["adapted"]["ledger"]["accrued_wait_seconds"],
            in_vehicle_passenger_seconds=branches[mode]["ledger"]["accrued_in_vehicle_seconds"]-
            branches["adapted"]["ledger"]["accrued_in_vehicle_seconds"])
    report = dict(complete=True, protocol=protocol["name"], training_seed=protocol["training_seed"],
        case=protocol["case"], updates=0, asset_manifest=str(assets.resolve()), actors=paths,
        selected=visible_selected, branches=branches, actual_effects_vs_adapted=effects, critic_scores=q,
        trace_storage="Baseline prefix retained only in server process memory; no trajectory files",
        effect_sign="Negative cost difference favors intervention; positive Q difference favors masked action",
        q_interpretation="Scaled undiscounted IQL/H2O+ continuation value (V7 gamma=1); not a calibrated passenger cost label under adapted deterministic continuation")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True, protocol=protocol["name"], effects=effects)), flush=True)


if __name__ == "__main__":
    main()
