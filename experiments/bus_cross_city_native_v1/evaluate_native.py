"""Evaluate frozen policies on held-out generated Halifax demand days."""

import argparse
import json
from pathlib import Path
import time

from native_runtime import (
    HERE, CausalArrivalRates, bootstrap, causal_event, city_line_mapping,
    load_environment, observe_arrivals, project_executed_action,
)


def forward_rule(api, raw, config, gain):
    event = api.contract.current_event_v7(raw)
    idx = api.contract.RAW_EVENT_STATE_INDEX
    target = float(event[idx["target_forward_headway_seconds"]])
    forward = (float(event[idx["forward_headway_seconds"]])
               if event[idx["forward_bus_present"]] else target)
    hold = min(config.max_hold_seconds, max(0.0, gain * (target - forward)))
    return api.np.asarray([api.contract.hold_seconds_to_policy_action_v7(hold, config)], api.np.float32)


def evaluate(args, *, actor=None, result_extra=None):
    api = bootstrap()
    import torch
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7

    protocol = json.loads(args.evaluation_protocol.read_text())
    spec = protocol["evaluation"]
    case = next(row for row in spec["cases"] if row["case"] == args.case)
    native = json.loads((HERE / "protocol.json").read_text())
    config = api.contract.DEFAULT_CONFIG_V7
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.method not in ("forward_proportional", "zero"):
        if actor is None:
            if args.checkpoint is None:
                raise ValueError("trained-policy evaluation requires its final checkpoint")
            actor, _ = load_actor_checkpoint_v7(args.checkpoint, "cpu", require_policy_artifact=True)
        actor.eval()
    torch.set_num_threads(1)
    env, assets = load_environment(api, args.assets, spec["episode_seconds"], case["sumo_seed"])
    if assets["demand_seed"] != case["demand_seed"]:
        raise ValueError("evaluation asset demand seed differs from the frozen test case")
    tracker = CausalArrivalRates(native["observation"]["arrival_rate_window_seconds"])
    holds, proposed_holds, legacy_holds = [], [], []
    guard_increased_holds = guard_decreased_holds = 0
    started = time.monotonic()
    try:
        env.reset()
        version = api.traci.getVersion()
        if "SUMO " + native["smoke"]["sumo_version"] not in version[1]:
            raise RuntimeError("evaluation SUMO version differs from collection")
        line_ids, stations = city_line_mapping(env)
        env.step_to_event({})
        while not env.done:
            observe_arrivals(api, tracker, env)
            controls, actions = [], {}
            for event in sorted(env._current_events, key=api.event_sort_key):
                if api.is_terminal_event(env, event):
                    actions[event.bus_id] = [0.0, 1.0]
                else:
                    controls.append((event, causal_event(api, tracker, env, event)))
            if controls:
                batch = api.np.stack([raw for _, raw in controls])
                prior = api.np.full(len(controls), -1.0, api.np.float32)
                for index, (event, _) in enumerate(controls):
                    raw = api.contract.pack_event_batch_v7(batch, index, prior, config)
                    if args.method == "zero":
                        proposal = api.np.asarray([
                            api.contract.hold_seconds_to_policy_action_v7(0.0, config)], api.np.float32)
                    elif actor is None:
                        proposal = forward_rule(api, raw, config, spec["rule_gain"])
                    else:
                        state = api.contract.raw_state_to_canonical_v7(raw, stations, config)
                        with torch.inference_mode():
                            proposal, _ = actor(torch.from_numpy(state[None]), deterministic=True)
                        proposal = proposal.numpy()[0]
                    proposed_hold = float(api.contract.policy_action_to_hold_seconds_v7(proposal[0], config))
                    legacy = api.completion_safety_action(raw, proposal,
                        native["smoke"]["completion_guard"], config)
                    executed = (project_executed_action(api, raw, proposal,
                        native["smoke"]["completion_guard"], config)
                        if args.actor_action_space == "executed" else legacy)
                    hold = float(api.contract.policy_action_to_hold_seconds_v7(executed[0], config))
                    legacy_holds.append(float(api.contract.policy_action_to_hold_seconds_v7(legacy[0], config)))
                    proposed_holds.append(proposed_hold)
                    guard_increased_holds += int(hold > proposed_hold)
                    guard_decreased_holds += int(hold < proposed_hold)
                    prior[index] = executed[0]
                    actions[event.bus_id] = [hold, 1.0]
                    holds.append(hold)
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
        end = env._bridge.passenger_time_ledger.snapshot()
        passengers = api.summarize_passengers(env._bridge, end)
        if end.time_seconds != spec["episode_seconds"]:
            raise RuntimeError(f"evaluation horizon mismatch: ledger={end.time_seconds}, "
                f"bridge={env._bridge.current_time}, SUMO={api.traci.simulation.getTime()}, "
                f"expected={spec['episode_seconds']}, passengers={passengers}")
        if passengers["active_state_count_balance_error"]:
            raise RuntimeError("evaluation passenger ledger does not balance")
        cost = 2 * end.accrued_wait_seconds + end.accrued_in_vehicle_seconds
        result = dict(complete=True, method=args.method, training_seed=args.seed,
            **case, horizon_seconds=end.time_seconds, simulator_version=version,
            line_ids=line_ids, generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            passengers=passengers, service=api.summarize_service(env._bridge),
            decision_events=len(holds), mean_hold_seconds=float(api.np.mean(holds)),
            positive_holds=int(api.np.sum(api.np.asarray(holds) > 0)),
            actor_action_space=args.actor_action_space,
            mean_requested_hold_seconds=float(api.np.mean(proposed_holds)),
            mean_proposed_hold_seconds=float(api.np.mean(proposed_holds)) if args.actor_action_space == "proposal" else None,
            positive_proposed_holds=int(api.np.sum(api.np.asarray(proposed_holds) > 0)) if args.actor_action_space == "proposal" else None,
            max_proposed_hold_seconds=float(api.np.max(proposed_holds)) if args.actor_action_space == "proposal" else None,
            same_state_legacy_mean_hold_seconds=float(api.np.mean(legacy_holds)),
            same_state_legacy_different_events=int(api.np.sum(api.np.abs(api.np.asarray(holds)-legacy_holds)>1e-4)),
            guard_increased_holds=guard_increased_holds, guard_decreased_holds=guard_decreased_holds,
            execution_layer=native["smoke"]["execution_layer"],
            completion_guard=native["smoke"]["completion_guard"],
            elapsed_seconds=time.monotonic()-started,
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            protocol=protocol["name"], evaluation_protocol=str(args.evaluation_protocol.resolve()),
            asset_manifest=str(args.assets.resolve()),
            interpretation="Held-out generated corridor demand day; not measured real-city deployment")
        if result_extra is not None:
            result.update(result_extra)
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "method", "training_seed",
            "case", "generalized_seconds_per_departed_passenger")}), flush=True)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("zero", "forward_proportional", "target_only", "target_model", "cross_city"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, default=HERE / "training_protocol.json",
        help="Frozen evaluation cases; fresh confirmation uses its separately saved protocol.")
    parser.add_argument("--actor-action-space", choices=("proposal", "executed"), default="proposal",
        help="Existing comparison uses proposals; fixed adapted actor intervention uses executed holding.")
    args = parser.parse_args()
    if args.actor_action_space == "executed" and args.method in ("zero", "forward_proportional"):
        parser.error("Rules and zero retain their original proposal action space")
    evaluate(args)
