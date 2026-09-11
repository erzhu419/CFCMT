from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_bilateral_waiting_geometry_validation as validation


def routes_fixture():
    return {
        "102219_396_0": ["-32038056#3", "-28198821#4"],
        "129962_409_0": ["28198821#3", "32038051#0"],
        "121463_406_0": ["28198821#3", "32038056#0"],
        "168358_425_0": ["-32038056#3", "32324544#0"],
    }


def arm_fixture():
    routes = routes_fixture()
    samples = [{"time_sec": time, "stage": stage, "vehicles": {
        vehicle: {"route": route, "type_id": "pkw", "width_m": 1.8, "lane_id": "incoming"}
        for vehicle, route in routes.items()}}
        for time in range(validation.TRACE_BEGIN, validation.TRACE_END + 1)
        for stage in validation.STAGES]
    return {
        "metrics": {"ok": True, "accepted_intervention_trace": [
            {"time_sec": 25260 + i * 10, "score": .125} for i in range(23)],
            "collision_samples": [{"time": 25657}, {"time": 25658}],
            "collision_events": 2, "collision_incidents": 1,
            "starting_teleports": 0, "ending_teleports": 0},
        "step_times": list(range(validation.BEGIN + 1, validation.END + 1)),
        "samples": samples,
        "native_lane_catalog": {lane: {"shape_xy_m": [[0, 0], [10, 0]]}
                                for lane in validation.WAITING_STRAIGHT_LANES.values()},
        "static_geometry": {}, "originator_diagnostics": {},
        "focal_passages": {vehicle: {"passed_controlled_junction": True}
                           for vehicle in validation.FOCAL_IDS},
    }


@pytest.mark.parametrize("changed", ["action", "physical_sample", "collision_report", None])
def test_baseline_remains_exact_and_bilateral_feedback_is_allowed(tmp_path, monkeypatch, changed):
    baseline = arm_fixture()
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"metrics": baseline["metrics"]}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "protocol": validation.base.REFERENCE_PROTOCOL, "status": "PASS",
        "reference_result": str(original), "samples": baseline["samples"],
        "retained_collision_samples": baseline["metrics"]["collision_samples"],
    }))
    mirror_reference = tmp_path / "mirror.json"
    mirror_reference.write_text(json.dumps({
        "protocol": validation.MIRROR_REFERENCE_PROTOCOL, "status": "FAIL",
        "repair": {"retained_collision_samples": [{"collider": validation.MIRROR_IDS[1],
                                                   "victim": validation.MIRROR_IDS[0], "time": 25629}],
                   "samples": [{"vehicles": {validation.MIRROR_IDS[1]: {
                       "route": routes_fixture()[validation.MIRROR_IDS[1]]}}}]},
    }))
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    options = {"--seed": "41242", "--warmup-sec": "60", "--control-interval-sec": "10",
               "--prediction-horizon-sec": "450", "--conversion-root": str(tmp_path),
               "--manifest": "manifest.json"}
    runtime = SimpleNamespace(launch_inputs=lambda _: options,
        load_traffic_signal_manifest=lambda _: SimpleNamespace(sumocfgs={"cologne1": "old.sumocfg"}))
    monkeypatch.setattr(validation, "_load_runtime", lambda: (runtime, {"runtime": "frozen85"}))
    monkeypatch.setattr(validation, "read_focal_demand", lambda _: (routes_fixture(), {
        "vehicles": {vehicle: {"vehicle": {"type": "pkw"}} for vehicle in validation.FOCAL_IDS}}))
    monkeypatch.setenv("CFCMT_EXTERNAL_CONVERSION_ROOT", "before-test")
    repair = deepcopy(baseline)
    repair["metrics"].update(collision_events=0, collision_incidents=0, collision_samples=[])
    repair["metrics"]["accepted_intervention_trace"][0]["score"] = .25
    repair["samples"][0]["new_physical_state"] = True
    if changed == "action":
        baseline["metrics"]["accepted_intervention_trace"][2]["score"] = .2
    elif changed == "physical_sample":
        baseline["samples"][0]["different_position"] = True
    elif changed == "collision_report":
        baseline["metrics"]["collision_samples"].pop()
    calls = []

    def run_arm(*args):
        calls.append(args[2])
        assert args[3] == routes_fixture()
        return baseline if len(calls) == 1 else repair

    monkeypatch.setattr(validation, "_run_arm", run_arm)
    result = validation.run_validation(sumocfg=tmp_path / "repair.sumocfg",
        geometry_report=tmp_path / "geometry.json", launch_record=launch,
        reference_result=reference, mirror_reference_result=mirror_reference,
        tripinfo=tmp_path / "tripinfo.xml")
    if changed:
        assert result["status"] == "BASELINE_REPRODUCTION_FAILED"
        assert not result["repair_executed"]
        assert len(calls) == 1
    else:
        assert result["status"] == "PASS"
        assert len(calls) == 2
        prefix = result["early_prefix_diagnostic"]
        assert not prefix["hard_gate"]
        assert prefix["retained_intervention_trace"]["exact_leading_row_count"] == 0
        assert not prefix["window_observations"]["passed"]
        assert result["checks"]["original_pair_passed_controlled_junction"]
        assert result["checks"]["mirror_pair_passed_controlled_junction"]


@pytest.mark.parametrize("failure", ["original_late", "mirror_late", "collision_before_trace",
                                    "teleport", "short_run", "missing_stage"])
def test_whole_window_and_each_prespecified_pair_are_independent_gates(failure):
    arm = arm_fixture()
    arm["metrics"].update(collision_events=0, collision_incidents=0, collision_samples=[])
    assert all(validation.repair_checks(arm).values())
    if failure == "original_late":
        arm["focal_passages"][validation.ORIGINAL_IDS[0]]["passed_controlled_junction"] = False
    elif failure == "mirror_late":
        arm["focal_passages"][validation.MIRROR_IDS[0]]["passed_controlled_junction"] = False
    elif failure == "collision_before_trace":
        arm["metrics"].update(collision_events=1, collision_incidents=1,
                              collision_samples=[{"time": 25629}])
    elif failure == "teleport":
        arm["metrics"]["starting_teleports"] = 1
    elif failure == "short_run":
        arm["step_times"].pop()
    else:
        arm["samples"].pop()
    checks = validation.repair_checks(arm)
    assert not all(checks.values())
    if failure.endswith("late"):
        assert checks["no_native_collisions_in_full_window"]
        assert checks["original_pair_passed_controlled_junction"] == (failure == "mirror_late")
        assert checks["mirror_pair_passed_controlled_junction"] == (failure == "original_late")


def test_four_routes_require_both_left_continuations_and_all_outgoing_edges():
    routes = routes_fixture()
    tracker = validation.FocalPassageTracker(routes)
    current = {}
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getIDList=lambda: list(current), getRoute=lambda vehicle: routes[vehicle],
        getLaneID=lambda vehicle: current[vehicle][0], getRoadID=lambda vehicle: current[vehicle][1],
        getRouteIndex=lambda vehicle: int(current[vehicle][1] == routes[vehicle][1]),
        getLanePosition=lambda _: 1.0))
    for vehicle in routes:
        current[vehicle] = (routes[vehicle][0] + "_0", routes[vehicle][0])
    tracker.observe(api, 25250)
    for vehicle in routes:
        current[vehicle] = (validation.FOCAL_LANES[vehicle][0], "internal")
    tracker.observe(api, 25630)
    for vehicle in (validation.ORIGINAL_IDS[0], validation.MIRROR_IDS[0]):
        current[vehicle] = (routes[vehicle][1] + "_0", routes[vehicle][1])
    tracker.observe(api, 25631)
    assert sum(row["passed_controlled_junction"] for row in tracker.summary().values()) == 2
    for vehicle in (validation.ORIGINAL_IDS[1], validation.MIRROR_IDS[1]):
        current[vehicle] = (validation.FOCAL_LANES[vehicle][1], "continuation")
    tracker.observe(api, 25632)
    assert sum(row["passed_controlled_junction"] for row in tracker.summary().values()) == 2
    for vehicle in routes:
        current[vehicle] = (routes[vehicle][1] + "_0", routes[vehicle][1])
    tracker.observe(api, 25633)
    assert all(row["passed_controlled_junction"] for row in tracker.summary().values())


def test_clearance_keeps_moving_speed_both_sides_and_unavailable_body():
    arm = arm_fixture()
    lanes = list(validation.WAITING_STRAIGHT_LANES)

    def pose(lane, speed, polygon):
        return {"lane_id": lane, "speed_mps": speed, "lane_position_m": 8.3,
                "length_m": 4.3, "width_m": 1.8, "shape_class": "passenger",
                "passenger_polygon_xy_m": polygon,
                "polygon_unavailable_reason": None if polygon else "rear_on_preceding_lane"}

    arm["samples"] = [{"time_sec": 25665, "stage": validation.STAGES[1], "vehicles": {
        "moving": pose(lanes[0], .000365, [(2, 1), (3, 1), (3, 2), (2, 2)]),
        "stopped": pose(lanes[1], 0, [(2, .8), (3, .8), (3, 2), (2, 2)]),
        "unavailable": pose(lanes[0], 1, None),
    }}]
    evidence = validation.waiting_body_evidence(arm, {
        lane: 1.8 for lane in validation.WAITING_STRAIGHT_LANES.values()})
    assert not evidence["hard_gate"]
    first, mirror = (evidence["waiting_lanes"][lane] for lane in lanes)
    assert first["observation_count"] == 2
    assert first["body_reconstructable_count"] == 1
    assert first["minimum_clearance_m"] == pytest.approx(.1)
    assert first["stationary_minimum_clearance_m"] is None
    assert first["observations"][0]["speed_mps"] == .000365
    assert not first["observations"][0]["stationary_at_sample"]
    assert mirror["minimum_clearance_m"] == pytest.approx(-.1)
    assert mirror["stationary_minimum_clearance_m"] == pytest.approx(-.1)


def test_actual_trip_demand_format_reads_four_routes_without_missing_sample_inference(tmp_path):
    cfg = tmp_path / "cologne1.sumocfg"
    cfg.write_text('<configuration><input><route-files value="cologne1.rou.xml"/></input></configuration>')
    routes = routes_fixture()
    demand = tmp_path / "cologne1.rou.xml"
    demand.write_text('<routes><vType id="pkw" length="4.3"/>' + ''.join(
        f'<trip id="{vehicle}" from="{edges[0]}" to="{edges[1]}" type="pkw" depart="25600"/>'
        for vehicle, edges in routes.items()) + '</routes>')
    actual, record = validation.read_focal_demand(cfg)
    assert actual == routes
    assert record["route_file"] == str(demand)
    assert record["vehicles"][validation.MIRROR_IDS[0]]["route"] == routes[validation.MIRROR_IDS[0]]
