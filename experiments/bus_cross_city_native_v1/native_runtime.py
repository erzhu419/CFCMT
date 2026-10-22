"""New-city observations on the already frozen transit-v7 SUMO runtime."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent


class CausalArrivalRates:
    """Count actual departures/observed stop arrivals, once per person and stop."""

    def __init__(self, window_seconds=1800.0):
        self.window_seconds = float(window_seconds)
        self.seen = set()
        self.arrivals = defaultdict(list)

    def record(self, person_id, stop_id, line_ids, arrival_time, now):
        key = (str(person_id), str(stop_id))
        if key in self.seen or float(arrival_time) > float(now):
            return
        self.seen.add(key)
        for line_id in line_ids:
            self.arrivals[(str(stop_id), str(line_id))].append(float(arrival_time))

    def observe(self, bridge, now, waiting_by_stop):
        # Objects are consulted only after SUMO reports their actual departure.
        # The scheduled demand/rate table is never consulted here.
        for person_id, departed_time in bridge.departed_passenger_times.items():
            if float(departed_time) > now:
                continue
            person = bridge.passenger_obj_dic[person_id]
            self.record(person_id, person.start_stop_id_s,
                person.passable_line_l, departed_time, now)
        for stop_id, person_ids in waiting_by_stop.items():
            for person_id in person_ids:
                person = bridge.passenger_obj_dic[person_id]
                self.record(person_id, stop_id, person.passable_line_l,
                    person.arriver_time_n, now)

    def rate(self, stop_id, line_id, now):
        elapsed = min(float(now), self.window_seconds)
        if elapsed <= 0.0:
            return 0.0
        start = max(0.0, float(now) - self.window_seconds)
        count = sum(start < t <= now or (start == 0.0 and t == 0.0)
            for t in self.arrivals[(str(stop_id), str(line_id))])
        return float(count / elapsed)


def bootstrap():
    """Bind libsumo before importing the owned sim_obj supplement or bridge."""
    runtime = json.loads((HERE / "runtime.json").read_text())
    snapshot = Path(runtime["frozen_snapshot"])
    for key, value in runtime["env"].items():
        os.environ[key] = str(value)
    h2o = snapshot / "runtime/H2Oplus"
    os.environ["THESIS_H2O_ROOT"] = str(h2o)
    os.environ["LIBSUMO_AS_TRACI"] = "quiet"
    overlay = runtime["env"]["PYTHONPATH"]
    for entry in reversed(overlay.split(os.pathsep)):
        if entry:
            sys.path.insert(0, entry)
    sys.path.insert(0, str(h2o))
    sys.path.insert(0, str(HERE / "runtime_supplement"))
    import numpy as np
    import traci
    from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
    from bus_h2o import mdp_contract_v7 as contract
    from collect_policy.collect_transit_v7 import (
        append_transition, event_sort_key, is_terminal_event, raw_event_observation,
    )
    from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
    from SimpleSAC.transit_v3.evaluation import summarize_service
    from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
    return SimpleNamespace(np=np, traci=traci, env_class=AllLinesSumoEnv,
        contract=contract, append_transition=append_transition,
        event_sort_key=event_sort_key, is_terminal_event=is_terminal_event,
        raw_event_observation=raw_event_observation,
        completion_safety_action=completion_safety_action,
        summarize_service=summarize_service,
        summarize_passengers=summarize_passengers_v5, snapshot=snapshot)


def project_executed_action(api, raw_state, desired_action, variant, config):
    """Project a learned execution action onto the frozen guard's image.

    The quarter-load-sixty guard is continuous and nondecreasing in holding:
    its reachable execution interval is determined by zero/full proposals.
    Existing execution labels in that interval remain unchanged.
    """
    lower = api.completion_safety_action(raw_state,
        api.np.asarray([-1.0], api.np.float32), variant, config)[0]
    upper = api.completion_safety_action(raw_state,
        api.np.asarray([1.0], api.np.float32), variant, config)[0]
    return api.np.clip(api.np.asarray(desired_action, api.np.float32), lower, upper)


def _full_horizon_termination(bridge):
    """Keep empty-system ticks so final arrivals and passenger counts synchronize."""
    bridge.done = bridge.steps >= bridge.max_steps
    if bridge.done:
        bridge._cleanup_departed_buses()


def load_environment(api, asset_manifest, episode_seconds, sumo_seed):
    path = Path(asset_manifest).resolve()
    assets = json.loads(path.read_text())
    def asset_path(name):
        return str((path.parent / assets[name]).resolve())
    os.environ["THESIS_SUMO_ROOT"] = str(Path(asset_path("sumo_dir")).parent)
    env = api.env_class(asset_path("sumo_dir"), asset_path("edge_xml"),
        sumo_cfg=asset_path("sumo_cfg"),
        passenger_obj_xml=asset_path("passenger_obj_xml"),
        episode_seconds=int(episode_seconds), sumo_seed=int(sumo_seed),
        observation_schema="v5", sumo_backend="libsumo")
    # The frozen bridge also stops when SUMO has no expected agents. The fixed
    # service-day objective needs its normal accounting loop through the horizon.
    # reset() reuses this bridge instance, including this owned termination rule.
    env._ensure_bridge()
    env._bridge._check_termination = MethodType(_full_horizon_termination, env._bridge)
    return env, assets


def city_line_mapping(env):
    ids = [None] * len(env._line_idx_map)
    station_max = [None] * len(ids)
    for line_id, index in env._line_idx_map.items():
        ids[index] = str(line_id)
        station_max[index] = len(env._bridge.line_obj_dic[line_id].stop_id_l) - 1
    if any(value is None for value in ids):
        raise RuntimeError("new-city line indices are not contiguous")
    return ids, station_max


def observe_arrivals(api, tracker, env):
    now = float(env._bridge.current_time)
    waiting = {stop_id: api.traci.busstop.getPersonIDs(stop_id)
        for stop_id in env._bridge.stop_obj_dic}
    tracker.observe(env._bridge, now, waiting)


def causal_event(api, tracker, env, event):
    raw = api.raw_event_observation(env, event.bus_id).copy()
    raw[api.contract.RAW_EVENT_STATE_INDEX["eligible_arrival_rate_per_second"]] = (
        tracker.rate(event.stop_id, event.line_id, float(event.sim_time)))
    return raw
