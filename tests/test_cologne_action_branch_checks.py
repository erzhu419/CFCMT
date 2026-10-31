from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.data.cologne_action_branch_checks import compare_physical, physical_snapshot


def snapshot():
    vehicle = SimpleNamespace(
        getIDList=lambda: ("b", "a"), getLaneID=lambda vehicle: f"lane_{vehicle}",
        getRouteIndex=lambda vehicle: 1, getRoute=lambda vehicle: ("in", "out"),
        getLanePosition=lambda vehicle: 9.0, getSpeed=lambda vehicle: 2.0,
    )
    api = SimpleNamespace(vehicle=vehicle, simulation=SimpleNamespace(
        getTime=lambda: 25800.0, getPendingVehicles=lambda: ("pending_a", "pending_b")))
    executor = SimpleNamespace(snapshot=lambda: {
        "tls_id": "tls", "mode": "green", "current_state": "Gg",
        "current_green_state": "Gg", "target_green_state": "Gg", "green_elapsed_sec": 10.0,
        "remaining_sec": 0.0, "pending_all_red_sec": 0.0, "audit": {"requests": 80},
    })
    return physical_snapshot(api, {"tls": executor})


def test_snapshot_records_physical_fields_and_excludes_executor_audit():
    state = snapshot()
    assert list(state["vehicles"]) == ["a", "b"]
    assert state["vehicles"]["a"] == {
        "lane_id": "lane_a", "route_index": 1, "route": ["in", "out"], "lane_position": 9, "speed": 2,
    }
    assert "audit" not in state["executors"]["tls"]
    assert "tls_id" not in state["executors"]["tls"]
    assert state["pending_raw"] == 2
    assert compare_physical(state, deepcopy(state))["passed"]


def test_save_precision_is_allowed_but_exact_branch_comparison_rejects_it():
    before = snapshot()
    after = deepcopy(before)
    after["pending_raw"] = 0
    after["vehicles"]["a"]["lane_position"] += 5e-9
    after["vehicles"]["b"]["speed"] -= 4e-9
    result = compare_physical(before, after)
    assert result["passed"]
    assert result["max_position_or_speed_difference"] == pytest.approx(5e-9, abs=1e-15)
    assert compare_physical(before, after, tolerance=0)["mismatch_count"] == 2
    after["vehicles"]["a"]["lane_position"] += 2e-8
    assert compare_physical(before, after)["mismatch_count"] == 1


@pytest.mark.parametrize("field,value", [("lane_id", "other"), ("route_index", 2), ("route", ["in", "other"])])
def test_lane_and_route_identity_require_exact_equality(field, value):
    before = snapshot()
    after = deepcopy(before)
    after["vehicles"]["a"][field] = value
    result = compare_physical(before, after)
    assert not result["passed"]
    assert result["mismatches"][0]["path"] == f"vehicles.a.{field}"


def test_time_executor_state_and_vehicle_membership_are_not_precision_tolerant():
    before = snapshot()
    after = deepcopy(before)
    after["time_sec"] += 1e-9
    after["executors"]["tls"]["remaining_sec"] += 1e-12
    after["executors"]["tls"]["target_green_state"] = "rr"
    del after["vehicles"]["b"]
    result = compare_physical(before, after)
    assert result["mismatch_count"] == 4
    assert {row["path"] for row in result["mismatches"]} == {
        "time_sec", "vehicle_ids", "executors.tls.remaining_sec", "executors.tls.target_green_state",
    }


def test_mismatch_details_are_capped_without_truncating_count_or_maximum():
    before = snapshot()
    before["vehicles"] = {f"vehicle{i:02d}": deepcopy(before["vehicles"]["a"]) for i in range(25)}
    after = deepcopy(before)
    for i, vehicle in enumerate(after["vehicles"].values(), 1):
        vehicle["speed"] += i
    result = compare_physical(before, after)
    assert not result["passed"]
    assert result["mismatch_count"] == 25
    assert len(result["mismatches"]) == 20
    assert result["max_position_or_speed_difference"] == 25
