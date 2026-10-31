from copy import deepcopy
from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_cologne_service_counterfactual import (
    TLS_ID, compare_hold, crossing_events,
    observe_key_state,
)


def _samples():
    expected = [{"time_sec": float(now), "active_inserted_vehicles": 2,
                 "pending_insertion_vehicles": 5, "executor_before": {"current_state": "rrrGG"},
                 "lanes": {"in_1": {"vehicles": 2, "halting": 2, "occupancy_pct": 0.2,
                                       "mean_speed_mps": 0.0, "next_tls_link_counts": {"2": 2},
                                       "head": {"id": "straight_head", "position_m": 90.123456789,
                                                "speed_mps": 0.0, "waiting_sec": 200.0,
                                                "next_tls": [2, 1.000000001, "r"],
                                                "next_route_edge": "out"}}}}
                for now in range(26700, 26761, 10)]
    actual = deepcopy(expected)
    for row in actual:
        lane = row["lanes"]["in_1"]
        lane["occupancy_fraction"] = lane.pop("occupancy_pct")
    return expected, actual


def test_live_observer_uses_replay_fraction_field(monkeypatch):
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_cologne_service_counterfactual.observe_lane",
        lambda *args, **kwargs: ({"occupancy_fraction": 0.28158186943028773,
                                 "sampled_lane_enters": 0, "sampled_lane_leaves": 0}, set()),
    )
    api = SimpleNamespace(
        simulation=SimpleNamespace(getTime=lambda: 26700.0, getPendingVehicles=lambda: []),
        vehicle=SimpleNamespace(getIDCount=lambda: 23),
    )
    executor = SimpleNamespace(snapshot=lambda: {"audit": {}})
    result = observe_key_state(api, {TLS_ID: executor}, {"lane": {"incoming": True}})
    assert result["lanes"]["lane"] == {"occupancy_fraction": 0.28158186943028773}


def test_hold_gate_allows_only_declared_position_serialization_rounding():
    expected, actual = _samples()
    actual[0]["lanes"]["in_1"]["head"]["position_m"] += 1e-9
    result = compare_hold(expected, actual)
    assert result["passed"] and not result["exact_match"]
    assert result["numeric_difference_count"] == 1
    actual[0]["lanes"]["in_1"]["head"]["position_m"] += 1e-7
    assert not compare_hold(expected, actual)["passed"]


def test_restart_changes_in_speed_head_identity_or_occupancy_reject_comparison():
    for field, value in [("speed_mps", 1e-12), ("id", "other_head")]:
        expected, actual = _samples()
        actual[0]["lanes"]["in_1"]["head"][field] = value
        assert not compare_hold(expected, actual)["passed"]
    expected, actual = _samples()
    actual[0]["lanes"]["in_1"]["occupancy_fraction"] += 1e-12
    assert not compare_hold(expected, actual)["passed"]


def test_crossing_events_exclude_changes_between_incoming_lanes_and_unseen_arrivals():
    previous = {"cross": "in_1", "lane_change": "in_1", "arrived": "in_0"}
    current = {"lane_change": "in_0"}
    result = crossing_events(previous, current, {"cross", "lane_change"},
                             lambda vehicle: (f":{TLS_ID}_1_0", f":{TLS_ID}_1"),
                             now=26705.0, receiving_lanes={"out_0"})
    assert [row["vehicle_id"] for row in result] == ["cross"]
    assert result[0]["destination"] == "internal"
