"""Replay case00's recorded proposals and verify its full-horizon terminal tail."""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import h5py
import numpy as np

import collect_native as collector
from native_runtime import HERE


def main():
    case = 0
    original = HERE / "runs" / "case00"
    output = HERE / "native_terminal_repair" / "case00_replay"
    old = json.loads((original / "summary.json").read_text())
    with h5py.File(original / "transitions.h5", "r") as data:
        proposals = data["proposed_hold_seconds"][:].tolist()
    proposed = iter(proposals)
    used, old_stop = [], {}

    def recorded_hold(method, rng, smoke):
        value = next(proposed)
        used.append(value)
        return value

    original_loader = collector.load_environment

    def instrumented_loader(api, *args):
        env, assets = original_loader(api, *args)
        full_horizon = env._bridge._check_termination

        def observe_old_stop(bridge):
            snapshot = bridge.passenger_time_ledger.snapshot()
            if snapshot.time_seconds == old["horizon_seconds"] and not old_stop:
                old_stop.update(ledger=snapshot.to_dict(),
                    sumo_time_seconds=api.traci.simulation.getTime(),
                    sumo_expected_agents=api.traci.simulation.getMinExpectedNumber(),
                    sumo_vehicle_ids=list(api.traci.vehicle.getIDList()))
            full_horizon()

        env._bridge._check_termination = MethodType(observe_old_stop, env._bridge)
        return env, assets

    collector.proposed_hold = recorded_hold
    collector.load_environment = instrumented_loader
    collector.collect(SimpleNamespace(case=case, method=None, episode_seconds=None,
        assets=Path(old["asset_manifest"]), output=output))
    new = json.loads((output / "summary.json").read_text())
    if len(used) != len(proposals):
        raise RuntimeError("Replay did not consume exactly the recorded proposals")
    tail = new["horizon_seconds"] - old["horizon_seconds"]
    count = old["passengers"]["state_counts_at_horizon"]["waiting"]
    expected_wait = count * tail
    changed_terminal_fields = {"rewards", "delta_t_seconds", "wait_passenger_seconds", "next_observations"}
    checks, terminal = {}, {}
    with h5py.File(original / "transitions.h5", "r") as before, h5py.File(output / "transitions.h5", "r") as after:
        if set(before) != set(after):
            raise RuntimeError("Replay transition fields differ")
        for key in before:
            a, b = before[key][:], after[key][:]
            checks[key] = dict(same_shape=a.shape == b.shape,
                nonterminal_exact=bool(np.array_equal(a[:-1], b[:-1])),
                terminal_exact=bool(np.array_equal(a[-1], b[-1])))
            if not checks[key]["same_shape"] or not checks[key]["nonterminal_exact"]:
                raise RuntimeError("Original nonterminal transition changed: " + key)
            if key not in changed_terminal_fields and not checks[key]["terminal_exact"]:
                raise RuntimeError("Unexpected terminal field changed: " + key)
        for key, delta in (("rewards", -2 * expected_wait),
                           ("wait_passenger_seconds", expected_wait), ("delta_t_seconds", tail)):
            a, b = float(before[key][-1]), float(after[key][-1])
            terminal[key] = dict(before=a, after=b, actual_delta=b-a, expected_delta=delta)
            if b-a != delta:
                raise RuntimeError("Analytic terminal-tail mismatch: " + key)
        terminal["raw_next_before"] = before["next_observations"][-1].tolist()
        terminal["raw_next_after"] = after["next_observations"][-1].tolist()
    old_ledger = old_stop["ledger"]
    if (old_ledger["accrued_wait_seconds"] != old["passengers"]["accrued_wait_seconds_total"]
            or old_ledger["accrued_in_vehicle_seconds"] != old["passengers"]["accrued_in_vehicle_seconds_total"]
            or old_ledger["waiting_passengers"] != count or old_stop["sumo_vehicle_ids"]
            or old_stop["sumo_expected_agents"] != 0):
        raise RuntimeError("Replay does not reproduce the original natural termination state")
    if (new["horizon_seconds"] != 24000 or new["ledger_reward_telescope_error"] != 0
            or new["passengers"]["state_counts_at_horizon"] != old["passengers"]["state_counts_at_horizon"]
            or new["passengers"]["departed_passengers"] != old["passengers"]["departed_passengers"]):
        raise RuntimeError("Full-horizon passenger accounting changed beyond the waiting tail")
    report = dict(complete=True,case=case,new_training_updates=0,actual_simulations=1,
        replay="Original recorded proposals, original seeds/assets, unchanged executed actions",
        original=str(original),output=str(output),controls=len(used),field_checks=checks,
        old_stop=old_stop,old_horizon_seconds=old["horizon_seconds"],new_horizon_seconds=new["horizon_seconds"],
        existing_waiter_tail_passenger_seconds=2*expected_wait,terminal=terminal,
        old_service=old["service"],new_service=new["service"],
        ledger_reward_telescope_error=new["ledger_reward_telescope_error"])
    (output / "comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("complete","case","controls","existing_waiter_tail_passenger_seconds")}),flush=True)


if __name__ == "__main__":
    main()
