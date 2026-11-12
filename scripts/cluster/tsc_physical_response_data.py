"""Pre-action inputs and measured ten-second movement responses.

The signal schedule is nominal: the copied executor assumes that its future
clearance lanes are empty. Actual occupancy can extend all-red, so current
internal/clearance counts are features and actual future signal states are not.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cf_h2o.traffic_signal.safe_phase_controller import SafePhaseExecutor

try:
    from .tsc_movement_response import movement_catalog
except ImportError:
    from tsc_movement_response import movement_catalog


FEATURE_NAMES = (
    "incoming_lane_count", "outgoing_lane_count",
    "incoming_length_mean_m", "outgoing_length_mean_m",
    "incoming_speed_limit_mean_mps", "outgoing_speed_limit_mean_mps",
    "incoming_vehicle_count", "incoming_halted_count",
    "incoming_occupancy_raw_mean", "incoming_speed_mean_mps",
    "outgoing_vehicle_count", "outgoing_halted_count",
    "outgoing_occupancy_raw_mean", "outgoing_speed_mean_mps",
    "turn_allocated_vehicle_count", "turn_allocated_halted_count",
    "movement_lane_equivalent", "outgoing_free_vehicle_spaces",
    "internal_vehicle_count", "internal_length_sum_m",
    "internal_speed_limit_mean_mps", "movement_link_count",
    "clearance_vehicle_count", "executor_green", "executor_yellow",
    "executor_all_red", "executor_green_elapsed_sec", "executor_remaining_sec",
    "current_min_green_sec", "request_differs", "nominal_request_accepted",
    "nominal_major_green_sec", "nominal_minor_green_sec", "nominal_stop_go_sec",
    "nominal_off_priority_sec", "nominal_yellow_sec", "nominal_red_sec",
)
LABEL_NAMES = ("stopline_entries_10s", "receiving_entries_10s")
NOMINAL_ASSUMPTION = "No future occupancy clearance extension; current occupancy remains an input."


def nominal_signal_plan(executor, request, *, horizon_sec=10):
    """Run an isolated copy; never call a setter or future getter on SUMO."""
    if horizon_sec != 10:
        raise ValueError("This response task fixes the first control interval to ten seconds")
    writes = []
    memory_api = SimpleNamespace(trafficlight=SimpleNamespace(
        setRedYellowGreenState=lambda tls, state: writes.append((tls, state))))
    clone = SafePhaseExecutor(
        sumo_api=memory_api, tls_id=executor.tls_id,
        timings=list(executor.timings.values()), initial_state=executor.current_state,
        clearance_lane_ids=(), clearance_poll_sec=executor.clearance_poll_sec)
    clone.restore(executor.snapshot(), write_state=False)
    accepted = clone.request(request)
    states = []
    for _ in range(horizon_sec):
        states.append(clone.current_state)
        clone.advance(1.0)
    return {"states": states, "request_accepted": bool(accepted),
            "assumption": NOMINAL_ASSUMPTION}


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _lane_rows(api, lanes):
    return {lane: {
        "length": float(api.lane.getLength(lane)),
        "limit": float(api.lane.getMaxSpeed(lane)),
        "vehicles": int(api.lane.getLastStepVehicleNumber(lane)),
        "halted": int(api.lane.getLastStepHaltingNumber(lane)),
        "occupancy": float(api.lane.getLastStepOccupancy(lane)),
        "speed": float(api.lane.getLastStepMeanSpeed(lane)),
    } for lane in sorted(lanes)}


def _aggregate(lanes, rows):
    return {
        "count": len(lanes), "length": _mean([rows[lane]["length"] for lane in lanes]),
        "limit": _mean([rows[lane]["limit"] for lane in lanes]),
        "vehicles": sum(rows[lane]["vehicles"] for lane in lanes),
        "halted": sum(rows[lane]["halted"] for lane in lanes),
        "occupancy": _mean([rows[lane]["occupancy"] for lane in lanes]),
        # Empty-lane speed conventions do not describe an observed vehicle.
        "speed": _mean([rows[lane]["speed"] for lane in lanes if rows[lane]["vehicles"] > 0]),
    }


def build_preaction_rows(api, infos, executors, actions, movements=None, *,
                         focal_tls=None, horizon_sec=10):
    """Capture movement rows before requesting any action.

    Route, next-TLS and post-action observations are intentionally absent. The
    equal-turn allocation is a topology-only proxy, not hidden vehicle intent.
    The caller may retain all movements for accounting and fit only focal rows.
    """
    movements = movement_catalog(api, infos) if movements is None else movements
    selected = {key: movement for key, movement in movements.items()
                if focal_tls is None or movement["tls_id"] == focal_tls}
    if not selected:
        raise ValueError("No movements for the requested focal signal")
    signal_ids = sorted({movement["tls_id"] for movement in selected.values()})
    plans = {tls: nominal_signal_plan(executors[tls], actions[tls], horizon_sec=horizon_sec)
             for tls in signal_ids}
    lanes = {lane for movement in selected.values()
             for name in ("incoming_lanes", "outgoing_lanes", "internal_lanes")
             for lane in movement[name]}
    lanes.update(lane for tls in signal_ids for lane in executors[tls].clearance_lane_ids)
    lane_rows = _lane_rows(api, lanes)
    destinations = {}
    for movement in movements.values():
        for lane in movement["incoming_lanes"]:
            destinations.setdefault(lane, set()).add(movement["outgoing_edge"])
    features = []
    for key, movement in selected.items():
        tls = movement["tls_id"]
        ex, plan = executors[tls], plans[tls]
        incoming = _aggregate(movement["incoming_lanes"], lane_rows)
        outgoing = _aggregate(movement["outgoing_lanes"], lane_rows)
        internal = _aggregate(movement["internal_lanes"], lane_rows)
        allocation = {lane: 1.0 / len(destinations[lane]) for lane in movement["incoming_lanes"]}
        links = movement["link_indices"]
        if not links:
            raise ValueError("A controlled movement requires its signal-link indices")
        seconds = {chars: sum(state[index] in chars for state in plan["states"] for index in links)
                   / len(links) for chars in ("G", "g", "s", "O", "yY", "r")}
        values = [
            incoming["count"], outgoing["count"], incoming["length"], outgoing["length"],
            incoming["limit"], outgoing["limit"], incoming["vehicles"], incoming["halted"],
            incoming["occupancy"], incoming["speed"], outgoing["vehicles"], outgoing["halted"],
            outgoing["occupancy"], outgoing["speed"],
            sum(lane_rows[lane]["vehicles"] * share for lane, share in allocation.items()),
            sum(lane_rows[lane]["halted"] * share for lane, share in allocation.items()),
            sum(allocation.values()),
            sum(max(lane_rows[lane]["length"] / 7.5 - lane_rows[lane]["vehicles"], 0.0)
                for lane in movement["outgoing_lanes"]),
            internal["vehicles"], sum(lane_rows[lane]["length"] for lane in movement["internal_lanes"]),
            internal["limit"], len(links),
            sum(lane_rows[lane]["vehicles"] for lane in ex.clearance_lane_ids),
            float(ex.mode == "green"), float(ex.mode == "yellow"), float(ex.mode == "all_red"),
            ex.green_elapsed_sec, ex.remaining_sec,
            ex.timings[ex.current_green_state].min_green_sec,
            float(actions[tls] != ex.current_green_state), float(plan["request_accepted"]),
            seconds["G"], seconds["g"], seconds["s"], seconds["O"], seconds["yY"], seconds["r"],
        ]
        if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError("Nonfinite or incomplete pre-action response features")
        features.append([float(value) for value in values])
    return {"feature_names": list(FEATURE_NAMES), "movement_ids": list(selected),
            "features": features, "horizon_sec": horizon_sec,
            "nominal_states": {tls: plan["states"] for tls, plan in plans.items()},
            "nominal_request_accepted": {tls: plan["request_accepted"] for tls, plan in plans.items()},
            "nominal_assumption": NOMINAL_ASSUMPTION,
            "turn_allocation": "Equal share among network-feasible outgoing edges from each incoming lane."}


def extract_response_labels(measurement, movement_ids, *, horizon_sec=10):
    """Extract true measured counts, including receipts of initial internal vehicles."""
    if horizon_sec != 10 or not measurement["complete"]:
        raise ValueError("Labels require complete ten-second movement accounting")
    rows = measurement["windows"][str(horizon_sec)]["movements"]
    labels = []
    for key in movement_ids:
        row = rows[key]
        if not row["accounting_balanced"] or row["unresolved_exits"] or row["unobserved_entries"]:
            raise ValueError("Unresolved movement is not a zero-response label")
        labels.append([float(row["stopline_entries"]), float(row["receiving_entries"])])
    return labels


def analytic_baseline(features, feature_names=FEATURE_NAMES):
    """Fixed queue/headway rule; no fitting or future signal/flow observations.

    Two-second headway, 7.5-m vehicle storage, uniform feasible-turn allocation.
    s/O are permissive states in the executor, treated at the same nominal
    headway as G/g. Internal distance and speed supply a crude receiving lag.
    """
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("The fixed analytic rule requires the declared feature roster")
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Analytic baseline requires finite movement feature rows")
    col = {name: x[:, i] for i, name in enumerate(FEATURE_NAMES)}
    service_seconds = sum(col[name] for name in (
        "nominal_major_green_sec", "nominal_minor_green_sec",
        "nominal_stop_go_sec", "nominal_off_priority_sec"))
    offered = service_seconds * col["movement_lane_equivalent"] / 2.0
    entries = np.minimum(col["turn_allocated_vehicle_count"], offered)
    entries = np.minimum(entries, col["outgoing_free_vehicle_spaces"])
    distance = col["internal_length_sum_m"] / np.maximum(col["movement_link_count"], 1.0)
    lag = distance / np.maximum(col["internal_speed_limit_mean_mps"], 1.0)
    receipts = (col["internal_vehicle_count"] + entries) * np.clip((10.0 - lag) / 10.0, 0.0, 1.0)
    receipts = np.minimum(receipts, col["outgoing_free_vehicle_spaces"])
    return np.column_stack((np.maximum(entries, 0.0), np.maximum(receipts, 0.0)))
