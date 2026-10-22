"""Collect native chronological Halifax transitions; data stays on the server."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from native_runtime import (
    HERE, CausalArrivalRates, bootstrap, causal_event, city_line_mapping,
    load_environment, observe_arrivals,
)


TRANSITION_FIELDS = (
    "observations", "actions", "rewards", "next_observations", "discounts",
    "terminals", "delta_t_seconds", "trajectory_local_id", "wait_passenger_seconds",
    "in_vehicle_passenger_seconds", "event_bus_id", "event_line_index",
    "event_station_index", "event_decision_time_seconds", "event_base_dwell_seconds",
)


def proposed_hold(method, rng, smoke):
    if method == "zero" or rng.random() < smoke["mixture_zero_probability"]:
        return 0.0
    alpha, beta = smoke["mixture_positive_beta"]
    return float(rng.beta(alpha, beta) * smoke["max_proposed_hold_seconds"])


def collection_spec(args, protocol):
    smoke = protocol["smoke"]
    if args.case is None:
        if args.method is None:
            raise ValueError("smoke collection requires --method")
        return dict(smoke, method=args.method,
            episode_seconds=1800 if args.episode_seconds is None else args.episode_seconds,
            case=None, split="smoke", global_trajectory_id=None)
    case = next(row for row in protocol["collection_cases"] if row["case"] == args.case)
    if args.method is not None and args.method != case["method"]:
        raise ValueError("--method differs from the frozen collection case")
    if args.episode_seconds is not None and args.episode_seconds != case["episode_seconds"]:
        raise ValueError("--episode-seconds differs from the frozen collection case")
    return dict(smoke, **case, global_trajectory_id=case["trajectory_id"])


def save_transitions(api, out, rows, metadata):
    import h5py
    with h5py.File(out / "transitions.h5", "w") as data:
        for name, values in rows.items():
            if name == "event_bus_id":
                data.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))
            else:
                array = api.np.asarray(values)
                data.create_dataset(name, data=array, compression="gzip")
        for name, value in metadata.items():
            data.attrs[name] = json.dumps(value)


def collect(args):
    api = bootstrap()
    np, c = api.np, api.contract
    protocol = json.loads((HERE / "protocol.json").read_text())
    smoke = collection_spec(args, protocol)
    method, horizon = smoke["method"], smoke["episode_seconds"]
    config = c.DEFAULT_CONFIG_V7
    rng = np.random.default_rng(smoke["action_seed"])
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    env, assets = load_environment(api, args.assets, horizon, smoke["sumo_seed"])
    if assets["demand_seed"] != smoke["demand_seed"]:
        raise ValueError("the asset demand seed differs from the selected collection case")
    tracker = CausalArrivalRates(protocol["observation"]["arrival_rate_window_seconds"])
    rows = {name: [] for name in TRANSITION_FIELDS}
    rows.update(proposed_actions=[], proposed_hold_seconds=[], executed_hold_seconds=[])
    previous = None
    first_snapshot = None
    terminal_endpoints = 0
    batch_sizes = []
    started = time.monotonic()
    try:
        env.reset()
        simulator_version = api.traci.getVersion()
        actual_sumo_version = re.search(r"\bSUMO\s+(\d+\.\d+\.\d+)", simulator_version[1]).group(1)
        if actual_sumo_version != smoke["sumo_version"]:
            raise RuntimeError(f"connected SUMO version {actual_sumo_version} differs from {smoke['sumo_version']}")
        line_ids, station_max = city_line_mapping(env)
        env.step_to_event({})
        while not env.done:
            observe_arrivals(api, tracker, env)
            events = sorted(env._current_events, key=api.event_sort_key)
            controls = []
            actions = {}
            for event in events:
                if api.is_terminal_event(env, event):
                    actions[event.bus_id] = [0.0, 1.0]
                    terminal_endpoints += 1
                else:
                    controls.append((event, causal_event(api, tracker, env, event),
                        c.passenger_time_snapshot_v5(event.metadata["passenger_time_at_decision"])))
            if controls:
                local_batch = np.stack([row[1] for row in controls])
                reference = controls[0][2]
                if any(row[2] != reference for row in controls[1:]):
                    raise RuntimeError("simultaneous decisions disagree on the real passenger ledger")
                batch_sizes.append(len(controls))
                prior = np.full(len(controls), -1.0, np.float32)
                for index, (event, raw_event, snapshot) in enumerate(controls):
                    raw = c.pack_event_batch_v7(local_batch, index, prior, config)
                    if previous is not None:
                        api.append_transition(rows, previous, raw, snapshot, terminal=False)
                    else:
                        first_snapshot = snapshot
                    hold = proposed_hold(method, rng, smoke)
                    proposal = np.asarray([c.hold_seconds_to_policy_action_v7(hold, config)], np.float32)
                    executed = api.completion_safety_action(raw, proposal,
                        smoke["completion_guard"], config)
                    executed_hold = float(c.policy_action_to_hold_seconds_v7(executed[0], config))
                    # These columns align with the transition emitted at the next decision.
                    rows["proposed_actions"].append(proposal.copy())
                    rows["proposed_hold_seconds"].append(hold)
                    rows["executed_hold_seconds"].append(executed_hold)
                    previous = dict(raw=raw.copy(), action=executed.copy(), passenger_time=snapshot,
                        event_bus_id=str(event.bus_id),
                        event_line_index=int(raw_event[c.RAW_EVENT_STATE_INDEX["line_id"]]),
                        event_station_index=int(raw_event[c.RAW_EVENT_STATE_INDEX["station_index"]]),
                        event_decision_time_seconds=float(event.sim_time),
                        event_base_dwell_seconds=float(raw_event[c.RAW_EVENT_STATE_INDEX["base_dwell_seconds"]]))
                    prior[index] = executed[0]
                    actions[event.bus_id] = [executed_hold, 1.0]
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
        end = env._bridge.passenger_time_ledger.snapshot()
        if end.time_seconds != horizon:
            raise RuntimeError(f"collection horizon mismatch: ledger={end.time_seconds}, expected={horizon}")
        if previous is None:
            raise RuntimeError("the fixed city subset emitted no controllable decision")
        api.append_transition(rows, previous, c.terminal_raw_state_v7(previous["raw"], end),
            end, terminal=True)
        passengers = api.summarize_passengers(env._bridge, end)
        service = api.summarize_service(env._bridge)
        weighted_cost = 2 * end.accrued_wait_seconds + end.accrued_in_vehicle_seconds
        collected_cost = (2 * (end.accrued_wait_seconds - first_snapshot.accrued_wait_seconds)
            + end.accrued_in_vehicle_seconds - first_snapshot.accrued_in_vehicle_seconds)
        reward_sum = float(np.sum(rows["rewards"], dtype=np.float64))
        if not np.isclose(reward_sum, -collected_cost, rtol=0.0, atol=1e-6):
            raise RuntimeError("chronological rewards do not telescope to the passenger ledger")
        if passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("real SUMO passenger counts do not balance")
        canonical = c.raw_state_to_canonical_v7(np.stack(rows["observations"]), station_max, config)
        if not np.isfinite(canonical).all():
            raise RuntimeError("new-city canonical states are not finite")
        dt = np.asarray(rows["delta_t_seconds"])
        metadata = dict(schema="bus-cross-city-native-v1", method=method,
            case=smoke["case"], split=smoke["split"], global_trajectory_id=smoke["global_trajectory_id"],
            episode_seconds=horizon, diagnostic=args.case is None,
            asset_manifest=str(args.assets.resolve()), scheduled_service_date=assets["scheduled_service_date"],
            city=assets["city"], line_ids=line_ids, station_max_by_line=station_max,
            frozen_snapshot=str(api.snapshot), config=config.__dict__,
            source_start_hour=assets["start_hour"], simulation_clock="relative seconds; t=0 is source start_hour",
            demand_seed=smoke["demand_seed"], sumo_seed=smoke["sumo_seed"], action_seed=smoke["action_seed"],
            simulator_version=simulator_version, sumo_version=actual_sumo_version,
            execution_layer=smoke["execution_layer"], completion_guard=smoke["completion_guard"],
            arrival_rate="actual departed or observed stop arrivals, rolling1800s; no future APC demand",
            target_total_unique_budget=protocol["planned_training"]["target_unique_transition_budget"],
            planned_target_splits=protocol["planned_training"]["target_splits"],
            world_model_fit_rows_within_fit=protocol["planned_training"]["world_model_fit_rows_within_fit"],
            world_model_fit_eligibility=protocol["planned_training"]["world_model_fit_eligibility"],
            training_launched=False)
        save_transitions(api, out, rows, metadata)
        summary = dict(metadata, complete=True, diagnostic=args.case is None, episode_seconds=horizon,
            horizon_seconds=float(end.time_seconds), total_rows=len(dt), physical_dt_positive_rows=int(np.sum(dt > 0)),
            simultaneous_dt_zero_rows=int(np.sum(dt == 0)), terminal_rows=int(np.sum(rows["terminals"])),
            terminal_stop_events_skipped=terminal_endpoints, raw_state_dimension=245,
            canonical_state_dimension=int(canonical.shape[1]), max_simultaneous_batch=max(batch_sizes),
            finite_states_and_rewards=bool(np.isfinite(np.stack(rows["observations"])).all()
                and np.isfinite(rows["rewards"]).all()),
            positive_proposed_holds=int(np.sum(np.asarray(rows["proposed_hold_seconds"]) > 0)),
            positive_executed_holds=int(np.sum(np.asarray(rows["executed_hold_seconds"]) > 0)),
            executed_hold_total_seconds=float(np.sum(rows["executed_hold_seconds"])),
            nonzero_eligible_rate_events=int(sum(
                c.current_event_v7(raw)[c.RAW_EVENT_STATE_INDEX["eligible_arrival_rate_per_second"]] > 0
                for raw in rows["observations"])),
            observed_person_stop_arrivals=len(tracker.seen),
            collected_reward_sum=reward_sum, collected_weighted_cost=collected_cost,
            uncollected_initial_weighted_cost=weighted_cost-collected_cost,
            ledger_reward_telescope_error=reward_sum+collected_cost,
            generalized_seconds_per_departed_passenger=(weighted_cost/passengers["departed_passengers"]
                if passengers["departed_passengers"] else None),
            passengers=passengers, service=service, elapsed_wall_seconds=time.monotonic()-started,
            dataset_file="transitions.h5", claim="Native target data collection; no trained-policy comparison")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: summary[key] for key in ("method", "complete", "total_rows",
            "physical_dt_positive_rows", "generalized_seconds_per_departed_passenger")}), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--case", type=int)
    parser.add_argument("--method", choices=("zero", "mixture"))
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
