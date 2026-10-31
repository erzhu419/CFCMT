from copy import deepcopy
from types import SimpleNamespace

import pytest

from cf_h2o.eval.traffic_signal_cologne_collision_replay import passenger_polygon
from scripts.data import run_cologne_uturn_collision_replay as replay


def rear_fixture(angle=90.):
    calls = []
    lengths = {":junction_23_0": 2., ":junction_9_0": 10.}
    api = SimpleNamespace(
        lane=SimpleNamespace(getLength=lambda lane: lengths[lane]),
        simulation=SimpleNamespace(convert2D=lambda *args: calls.append(args) or (6., 5.)),
    )
    pose = {
        "lane_id": "out_1", "lane_position_m": 1., "length_m": 4.3,
        "width_m": 1.8, "lateral_position_m": 0., "position_xy_m": [10., 5.],
        "angle_deg": angle, "back_position_xy_m": None, "passenger_polygon_xy_m": None,
        "polygon_unavailable_reason": "rear_on_preceding_lane",
    }
    path = ["in_1", ":junction_9_0", ":junction_23_0", "out_1"]
    return api, SimpleNamespace(passenger_polygon=passenger_polygon), pose, path, calls


def test_rear_traverses_both_preceding_internal_lanes_with_logical_offsets():
    api, runtime, pose, path, calls = rear_fixture()
    original = deepcopy(pose)
    result = replay.complete_rear_pose(api, runtime, pose, path)
    assert pose == original
    assert len(calls) == 1
    assert calls[0][0] == ":junction_9"
    assert calls[0][1] == pytest.approx(8.7)
    assert calls[0][2:] == (0, False)
    assert result["rear_reconstruction"]["lane_id"] == ":junction_9_0"
    assert result["rear_reconstruction"]["heading_error_deg"] == 0.
    assert result["back_position_xy_m"] == [6., 5.]
    assert result["passenger_polygon_xy_m"] == passenger_polygon([10., 5.], [6., 5.], 1.8)
    assert result["polygon_unavailable_reason"] is None


def test_preceding_path_disagreeing_with_native_heading_cannot_supply_polygon():
    api, runtime, pose, path, _ = rear_fixture(angle=100.)
    result = replay.complete_rear_pose(api, runtime, pose, path)
    assert result["rear_reconstruction"]["heading_error_deg"] == 10.
    assert result["back_position_xy_m"] is None
    assert result["passenger_polygon_xy_m"] is None
    assert result["polygon_unavailable_reason"] == "preceding_path_native_heading_mismatch"


def replay_fixture():
    report = {"time": 27476, "collider": replay.IDS[0], "victim": replay.IDS[1], "lane": "out_1"}
    inventory = {"native_event_count": 2, "native_incident_count": 1, "collision_event_steps": 2,
                 "incidents": [{"collider": replay.IDS[0], "victim": replay.IDS[1]}]}
    reference = {"metrics": {"collision_samples": [deepcopy(report)]},
                 "collision_inventory": deepcopy(inventory)}
    metrics = {"ok": True, "collision_samples": [report], "collision_events": 2,
               "collision_incidents": 1, "collision_event_steps": 2,
               "starting_teleports": 0, "ending_teleports": 0}
    samples = [{"time_sec": t, "stage": stage}
               for t in range(replay.TRACE_BEGIN, replay.TRACE_END + 1)
               for stage in replay.full.short.STAGES]
    return reference, inventory, metrics, samples, list(range(replay.BEGIN + 1, replay.END + 1))


@pytest.mark.parametrize("change,failed_check", [
    ("report", "original_capped_collision_reports_exact"),
    ("participant", "original_incident_inventory_exact"),
    ("count", "native_counts_agree"),
    ("duplicate_stage", "three_stage_window_complete"),
    ("missing_step", "full_prefix_steps"),
    ("teleport", "no_teleports"),
])
def test_frozen_replay_rejects_changed_event_or_incomplete_observation(change, failed_check):
    reference, inventory, metrics, samples, steps = replay_fixture()
    assert all(replay.replay_checks(reference, inventory, metrics, samples, steps).values())
    if change == "report":
        metrics["collision_samples"][0]["time"] += 1
    elif change == "participant":
        inventory["incidents"][0]["victim"] = "different_vehicle"
    elif change == "count":
        metrics["collision_events"] += 1
    elif change == "duplicate_stage":
        samples[0] = deepcopy(samples[1])
    elif change == "missing_step":
        steps.pop()
    else:
        metrics["starting_teleports"] = 1
    assert not replay.replay_checks(reference, inventory, metrics, samples, steps)[failed_check]
