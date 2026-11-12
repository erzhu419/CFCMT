import json
from types import SimpleNamespace

import pytest

from scripts.cluster.tsc_movement_response import (
    MovementCounter, MovementResponseObserver, movement_catalog,
)


def catalog():
    return {"tls|in|out": dict(movement_id="tls|in|out", tls_id="tls",
        incoming_edge="in", outgoing_edge="out", incoming_lanes=["in_0", "in_1"],
        outgoing_lanes=["out_0"], internal_lanes=[":tls_0", ":tls_1"], link_indices=[0, 1])}


def position(lane, road=None, **extra):
    return dict(lane_id=lane, road_id=road or lane.rsplit("_", 1)[0], **extra)


def kinds(counter):
    return [row["kind"] for row in counter.events]


def test_lane_change_is_not_discharge_and_internal_transit_stays_pending():
    c = MovementCounter(catalog())
    c.observe(0, {"v": position("in_0")})
    c.observe(1, {"v": position("in_1")})
    assert not c.events
    c.observe(2, {"v": position(":tls_0")})
    c.observe(3, {"v": position(":tls_1")})
    assert kinds(c) == ["stopline_entry"]
    assert c.pending == {"v": ("tls|in|out", False)}
    c.observe(4, {"v": position("out_0")})
    c.observe(5, {"v": position("out_1")})
    assert kinds(c) == ["stopline_entry", "receiving_entry"]
    assert not c.pending


def test_one_step_whole_junction_and_receiving_lane_change_keep_edge_movement():
    c = MovementCounter(catalog())
    c.observe(0, {"v": position("in_0")})
    c.observe(1, {"v": position("out_1")})
    assert kinds(c) == ["stopline_entry", "receiving_entry"]
    assert {row["movement_id"] for row in c.events} == {"tls|in|out"}


def test_initial_internal_receipt_does_not_create_new_stopline_entry():
    c = MovementCounter(catalog())
    c.observe(0, {"v": position(":tls_1")})
    c.observe(1, {"v": position("out_0")})
    assert kinds(c) == ["receiving_entry"]
    assert c.events[0]["initial_internal"] is True
    assert c.initial_pending["tls|in|out"] == 1


@pytest.mark.parametrize("terminal", [{}, {"v": "arrived"}, {"v": "teleported"}])
def test_lost_incoming_vehicle_is_unresolved_not_success(terminal):
    c = MovementCounter(catalog())
    c.observe(0, {"v": position("in_0", route_ends_here=False)})
    c.observe(1, {}, terminal)
    assert kinds(c) == ["unresolved_transition"]
    assert c.diagnostics["unresolved_incoming_transition"] == 1


def test_arrival_on_final_incoming_edge_is_not_a_junction_crossing():
    c = MovementCounter(catalog())
    c.observe(0, {"v": position("in_0", route_ends_here=True)})
    c.observe(1, {}, {"v": "arrived"})
    assert not c.events
    assert c.diagnostics["arrived_before_controlled_junction"] == 1


def test_unseen_internal_vehicle_and_wrong_junction_are_not_fabricated_flow():
    c = MovementCounter(catalog())
    c.observe(0, {"v": position("in_0")})
    c.observe(1, {"v": position(":other_0"), "new": position(":tls_1")})
    assert kinds(c) == ["unresolved_transition", "unresolved_entry"]
    assert c.pending == {"new": ("tls|in|out", False)}


class GetterAPI:
    """No set/step methods: the observer cannot move vehicles or signals."""
    def __init__(self):
        self.rows = {}
        self.arrived = []
        self.internal_links = {":tls_0": [("out_0", True, True, False, ":tls_1", "G", "s", 8)],
                               ":tls_1": [("out_0", True, True, False, "", "G", "s", 8)]}
        self.lane = SimpleNamespace(
            getEdgeID=lambda lane: lane.rsplit("_", 1)[0],
            getLinks=lambda lane: self.internal_links.get(lane, []),
            getLength=lambda lane: 100.,
            getLastStepVehicleIDs=lambda lane: [v for v, r in self.rows.items() if r["lane_id"] == lane],
            getLastStepHaltingNumber=lambda lane: 2,
            getLastStepOccupancy=lambda lane: .2,
            getLastStepMeanSpeed=lambda lane: 5.)
        self.vehicle = SimpleNamespace(
            getIDList=lambda: list(self.rows), getRoute=lambda v: ("in", "out"),
            getRouteIndex=lambda v: int(self.rows[v]["road_id"] == "out"),
            getLaneID=lambda v: self.rows[v]["lane_id"],
            getRoadID=lambda v: self.rows[v]["road_id"], getLanePosition=lambda v: 2.)
        self.simulation = SimpleNamespace(getArrivedIDList=lambda: self.arrived,
                                         getStartingTeleportIDList=lambda: [])


def infos():
    return {"tls": SimpleNamespace(controlled_links=[[("in_0", "out_0", ":tls_0")],
                                                     [("in_1", "out_0", ":tls_0")]])}


def test_catalog_groups_lane_connections_and_follows_second_internal_segment():
    m = movement_catalog(GetterAPI(), infos())["tls|in|out"]
    assert m["incoming_lanes"] == ["in_0", "in_1"]
    assert m["internal_lanes"] == [":tls_0", ":tls_1"]
    assert m["link_indices"] == [0, 1]


def test_observer_windows_account_initial_and_unfinished_crossings():
    api = GetterAPI()
    o = MovementResponseObserver(api, infos(), checkpoint_sec=100, windows=(1, 3))
    api.rows = {"old": position(":tls_1"), "new": position("in_0")}
    o.sample(100, {"tls": "G"})
    api.rows = {"old": position("out_0"), "new": position("in_1")}
    o.sample(101)
    api.rows = {"old": position("out_0"), "new": position(":tls_0")}
    o.sample(102)
    api.rows = {"old": position("out_0"), "new": position(":tls_1")}
    o.sample(103)
    r = o.summary()
    assert r["complete"] and r["sample_count"] == 3
    m = r["windows"]["3"]["movements"]["tls|in|out"]
    assert (m["initial_internal"], m["stopline_entries"], m["receiving_entries"], m["pending_internal"]) == (1, 1, 1, 1)
    lane = r["windows"]["3"]["receiving_lanes"]["out_0"]
    assert lane["halted_vehicle_seconds"] == 6
    assert lane["vehicle_seconds"] == 3
    with pytest.raises(ValueError, match="consecutive"):
        o.sample(105)


def test_unresolved_pending_blocks_complete_even_if_accounting_balances():
    api = GetterAPI()
    o = MovementResponseObserver(api, infos(), checkpoint_sec=0, windows=(1,))
    api.rows = {"v": position(":tls_0")}
    o.sample(0)
    api.rows = {}
    o.sample(1)
    r = o.summary()
    assert not r["complete"] and r["unresolved_count"] == 1
    assert r["windows"]["1"]["movements"]["tls|in|out"]["accounting_balanced"]


def test_saved_frames_replay_counter_events_diagnostics_and_pending_exactly():
    api = GetterAPI()
    observer = MovementResponseObserver(api, infos(), checkpoint_sec=0, windows=(4,))
    trajectories = [
        {"cross": position("in_0"), "lost": position(":tls_0")},
        {"cross": position("in_1"), "lost": position(":tls_1")},
        {"cross": position(":tls_0")},
        {"cross": position("out_0")},
        {"cross": position("elsewhere_0")},
    ]
    for second, rows in enumerate(trajectories):
        api.rows = rows
        api.arrived = ["lost"] if second == 2 else []
        observer.sample(second)
    # Round-trip the archive representation rather than relying on shared dicts.
    archived = json.loads(json.dumps(observer.frames))
    replay = MovementCounter(observer.movements)
    for frame in archived:
        replay.observe(frame["time_sec"], frame["vehicles"], frame["terminal"])
        # The thin API sampler retains vehicles outside its lane roster for one
        # exit observation; this recorded carry state reproduces that pruning.
        replay.previous = {vehicle: frame["vehicles"][vehicle]
                           for vehicle in frame["counter_retained_vehicle_ids"]}
    assert replay.events == observer.events
    assert replay.diagnostics == observer.counter.diagnostics
    assert replay.pending == observer.counter.pending
    assert replay.initial_pending == observer.counter.initial_pending
    assert archived[2]["terminal"] == {"lost": "arrived"}
    assert archived[-1]["vehicles"]["cross"]["road_id"] == "elsewhere"
    assert archived[-1]["counter_retained_vehicle_ids"] == []
