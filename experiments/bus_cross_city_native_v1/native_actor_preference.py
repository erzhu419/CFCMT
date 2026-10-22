"""Read-only actor/critic preferences along each adapted actor's native prefix."""

import argparse
import json
from pathlib import Path
import random
import time

from native_runtime import (
    HERE, CausalArrivalRates, bootstrap, causal_event, city_line_mapping,
    load_environment, observe_arrivals, project_executed_action,
)


def describe(np, values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()),
        p05=float(np.quantile(values, .05)), median=float(np.median(values)),
        p95=float(np.quantile(values, .95)), max=float(values.max()),
        positive=int((values > 0).sum()), negative=int((values < 0).sum()),
        zero=int((values == 0).sum()))


def gap_summary(np, values, heads):
    return dict(describe(np, values), all_heads_positive=int((heads > 0).all(0).sum()),
        all_heads_negative=int((heads < 0).all(0).sum()),
        heads=[describe(np, row) for row in heads])


def collect_prefix(api, torch, actors, protocol, native, seed, assets):
    random.seed(seed)
    api.np.random.seed(seed)
    torch.manual_seed(seed)
    config = api.contract.DEFAULT_CONFIG_V7
    guard = native["smoke"]["completion_guard"]
    env, manifest = load_environment(api, assets, protocol["episode_seconds"],
        protocol["case"]["sumo_seed"])
    if manifest["demand_seed"] != protocol["case"]["demand_seed"]:
        raise RuntimeError("Existing asset demand seed differs from frozen case")
    tracker = CausalArrivalRates(native["observation"]["arrival_rate_window_seconds"])
    rows, batch_index, terminal_events = [], 0, 0
    try:
        env.reset()
        version = api.traci.getVersion()
        if "SUMO " + native["smoke"]["sumo_version"] not in version[1]:
            raise RuntimeError("Simulator version differs from frozen runtime")
        line_ids, stations = city_line_mapping(env)
        env.step_to_event({})
        while not env.done:
            observe_arrivals(api, tracker, env)
            controls, actions = [], {}
            for event in sorted(env._current_events, key=api.event_sort_key):
                if api.is_terminal_event(env, event):
                    actions[event.bus_id] = [0.0, 1.0]
                    terminal_events += 1
                else:
                    controls.append((event, causal_event(api, tracker, env, event)))
            if controls:
                batch = api.np.stack([raw for _, raw in controls])
                prior = api.np.full(len(controls), -1.0, api.np.float32)
                for slot, (event, _) in enumerate(controls):
                    raw = api.contract.pack_event_batch_v7(batch, slot, prior, config)
                    canonical = api.contract.raw_state_to_canonical_v7(raw, stations, config)
                    states = torch.from_numpy(canonical[None])
                    requested, probabilities, positive_actions = {}, {}, {}
                    with torch.inference_mode():
                        for name, actor in actors.items():
                            requested[name] = actor(states, deterministic=True)[0].numpy()[0]
                            gate, _, _ = actor.distribution_parameters(states)
                            probabilities[name] = float(gate.sigmoid()[0])
                            positive_actions[name] = actor.positive_action_and_log_prob(
                                states, deterministic=True)[0].numpy()[0]
                    requested["masked_positive"] = positive_actions["masked"]
                    requested["zero"] = api.np.asarray([-1.0], api.np.float32)
                    executed = {name: project_executed_action(api, raw, action, guard, config)
                        for name, action in requested.items()}
                    holds = {name: float(api.contract.policy_action_to_hold_seconds_v7(
                        action[0], config)) for name, action in executed.items()}
                    bounds = [float(api.completion_safety_action(raw,
                        api.np.asarray([action], api.np.float32), guard, config)[0])
                        for action in (-1.0, 1.0)]
                    rows.append(dict(event_index=len(rows), batch_index=batch_index, batch_slot=slot,
                        time_seconds=float(event.sim_time), bus_id=str(event.bus_id),
                        line_id=str(event.line_id), stop_id=str(event.stop_id),
                        canonical=canonical.tolist(), prior=prior.tolist(), bounds=bounds,
                        requested={name: value.tolist() for name, value in requested.items()},
                        executed={name: value.tolist() for name, value in executed.items()},
                        holds=holds, hold_probability=probabilities,
                        conditional_positive_requested={name: value.tolist()
                            for name, value in positive_actions.items()}))
                    prior[slot] = executed["adapted"][0]
                    actions[event.bus_id] = [holds["adapted"], 1.0]
            env.step_to_event(actions)
            env._bridge.consume_completed_trip_passenger_times()
            batch_index += 1
        ledger = env._bridge.passenger_time_ledger.snapshot()
        passengers = api.summarize_passengers(env._bridge, ledger)
        if ledger.time_seconds != protocol["episode_seconds"] or passengers[
                "active_state_count_balance_error"]:
            raise RuntimeError("Full-horizon native ledger did not complete or balance")
        cost = 2*ledger.accrued_wait_seconds+ledger.accrued_in_vehicle_seconds
        baseline = dict(horizon_seconds=ledger.time_seconds, simulator_version=version,
            line_ids=line_ids, ledger=ledger.to_dict(), passengers=passengers,
            service=api.summarize_service(env._bridge), terminal_events=terminal_events,
            event_batches=batch_index, control_events=len(rows), generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"])
        return rows, baseline
    finally:
        env.close()


def critics(torch, np, protocol, seed, rows):
    import h5py
    from diagnose_values import learner_from_payload
    from train_native import runtime_api
    trainer_class, config_class, _, _ = runtime_api()
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        contract = json.loads(data.attrs["config"])
    names, stages, provenance = protocol["candidate_names"], {}, {}
    for name, spec in protocol["critic_stages"].items():
        path = HERE / spec["folder"] / f"seed{seed}" / spec["filename"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["total_updates"] != spec["stored_updates"]:
            raise RuntimeError(f"Unexpected stored updates for {name} critic")
        learner = learner_from_payload(trainer_class, config_class, contract, payload, seed)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(rows), protocol["critic_batch_size"]):
                chunk = rows[start:start+protocol["critic_batch_size"]]
                states = torch.tensor([row["canonical"] for row in chunk], dtype=torch.float32)
                actions = torch.tensor([[row["executed"][candidate] for candidate in names]
                    for row in chunk], dtype=torch.float32).reshape(-1, 1)
                q = learner.critic(states.repeat_interleave(len(names), 0), actions)
                scores.append(q.reshape(q.shape[0], len(chunk), len(names)).numpy())
        heads = np.concatenate(scores, axis=1)
        stages[name] = heads
        provenance[name] = dict(checkpoint=str(path), stored_updates=payload["total_updates"],
            heads=heads.shape[0], reward_scale=learner.config.reward_scale)
        for index, row in enumerate(rows):
            row.setdefault("critic", {})[name] = {candidate: dict(
                heads=heads[:, index, slot].tolist(), min=float(heads[:, index, slot].min()))
                for slot, candidate in enumerate(names)}
        del learner, payload
    return stages, provenance


def summaries(np, protocol, rows, stages):
    names = protocol["candidate_names"]
    holds = {name: np.asarray([row["holds"][name] for row in rows]) for name in names}
    probability = {name: np.asarray([row["hold_probability"][name] for row in rows])
        for name in ("adapted", "masked")}
    free = np.asarray([row["bounds"][1] > row["bounds"][0] for row in rows])
    different = holds["masked"] != holds["adapted"]
    groups = dict(all=np.ones(len(rows), bool), free=free, collapsed=~free,
        free_actor_different=free & different, free_actor_same=free & ~different,
        free_masked_zero=free & (holds["masked"] == 0),
        free_masked_positive=free & (holds["masked"] > 0),
        free_gate_zero=free & (probability["masked"] < .5),
        free_gate_positive=free & (probability["masked"] >= .5))
    output = {}
    for group, selected in groups.items():
        result = dict(events=int(selected.sum()),
            hold_seconds={name: describe(np, values[selected]) for name, values in holds.items()},
            hold_probability={name: describe(np, values[selected]) for name, values in probability.items()},
            masked_minus_adapted_hold_seconds=describe(np,
                (holds["masked"]-holds["adapted"])[selected]), critics={})
        for name, heads in stages.items():
            minimum = heads.min(0)
            adapted, masked, positive, zero = [names.index(candidate) for candidate in
                ("adapted", "masked", "masked_positive", "zero")]
            qgap = minimum[:, masked]-minimum[:, adapted]
            gate_gap = minimum[:, positive]-minimum[:, zero]
            gate_heads = heads[:, :, positive]-heads[:, :, zero]
            gate_candidates_differ = holds["masked_positive"] != holds["zero"]
            zero_gate = probability["masked"] < .5
            conflict = selected & gate_candidates_differ & (
                (zero_gate & (gate_gap > 0)) | (~zero_gate & (gate_gap < 0)))
            result["critics"][name] = dict(
                masked_minus_adapted=gap_summary(np, qgap[selected],
                    (heads[:, :, masked]-heads[:, :, adapted])[:, selected]),
                positive_minus_zero=gap_summary(np, gate_gap[selected], gate_heads[:, selected]),
                gate_candidates_differ=int((selected & gate_candidates_differ).sum()),
                gate_preference_conflicts=int(conflict.sum()),
                conflict_all_heads=int((conflict & (
                    (zero_gate & (gate_heads > 0).all(0)) |
                    (~zero_gate & (gate_heads < 0).all(0)))).sum()))
        output[group] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "native_actor_preference_protocol.json")
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    if args.seed not in protocol["training_seeds"]:
        raise ValueError("Seed must be one of the three frozen existing training seeds")
    native = json.loads((HERE / "protocol.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    api = bootstrap()
    import torch
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    torch.set_num_threads(1)
    assets = Path(runtime["assets_root"]) / protocol["asset_folder"] / "case00/asset_manifest.json"
    actors, paths = {}, {}
    for name, spec in protocol["actors"].items():
        path = HERE / spec["folder"] / f"seed{args.seed}" / spec["filename"]
        actors[name], _ = load_actor_checkpoint_v7(path, "cpu", require_policy_artifact=True)
        actors[name].eval().requires_grad_(False)
        paths[name] = str(path)
    rows, baseline = collect_prefix(api, torch, actors, protocol, native, args.seed, assets)
    stages, provenance = critics(torch, api.np, protocol, args.seed, rows)
    output = HERE / "native_actor_preference" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    report = dict(complete=True, protocol=protocol["name"], training_seed=args.seed,
        case=protocol["case"], updates=0, selection=protocol["selection"],
        actors=paths, critics=provenance, asset_manifest=str(assets), baseline=baseline,
        groups=summaries(api.np, protocol, rows, stages), elapsed_seconds=time.monotonic()-started,
        trace_storage=str(output / "events.jsonl"),
        execution="Adapted deterministic actions only; same-time prior uses actual adapted execution",
        positive_branch="Actor positive_action_and_log_prob(deterministic=True): 2*sigmoid(mean)-1, epsilon-clipped",
        interpretation="Strict signs, including tiny gaps; learned scaled undiscounted IQL/H2O+ Q is not a calibrated cost label under adapted deterministic continuation; correlated native events are descriptive")
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True, seed=args.seed, events=len(rows), output=str(output))), flush=True)


if __name__ == "__main__":
    main()
