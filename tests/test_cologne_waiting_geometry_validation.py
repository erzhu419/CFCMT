from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_waiting_geometry_validation as validation


def arm_fixture():
    routes = {validation.FOCAL_IDS[0]: ["straight_in", "straight_out"],
              validation.FOCAL_IDS[1]: ["left_in", "left_out"]}
    samples = [{"time_sec": time, "stage": stage, "vehicles": {
        vehicle: {"route": route} for vehicle, route in routes.items()}}
        for time in range(validation.TRACE_BEGIN, validation.TRACE_END + 1)
        for stage in validation.STAGES]
    return {
        "metrics": {"ok": True, "accepted_intervention_trace": [
            {"time_sec": 25260 + i * 10, "score": .125} for i in range(23)],
            "collision_samples": [{"time": 25657}, {"time": 25658}],
            "collision_events": 2, "collision_incidents": 1,
            "starting_teleports": 0, "ending_teleports": 0},
        "step_times": list(range(validation.BEGIN + 1, validation.END + 1)),
        "samples": samples, "native_lane_catalog": {}, "static_geometry": {},
        "originator_diagnostics": {},
        "focal_passages": {vehicle: {"passed_controlled_junction": True}
                           for vehicle in validation.FOCAL_IDS},
    }


@pytest.mark.parametrize("changed", ["action", "physical_sample", "collision_report", None])
def test_baseline_is_a_prerequisite_but_repair_feedback_may_change(tmp_path, monkeypatch, changed):
    baseline = arm_fixture()
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"metrics": baseline["metrics"]}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "protocol": validation.REFERENCE_PROTOCOL, "status": "PASS",
        "reference_result": str(original), "samples": baseline["samples"],
        "retained_collision_samples": baseline["metrics"]["collision_samples"],
    }))
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    options = {"--seed": "41242", "--warmup-sec": "60", "--control-interval-sec": "10",
               "--prediction-horizon-sec": "450", "--conversion-root": str(tmp_path),
               "--manifest": "manifest.json"}
    runtime = SimpleNamespace(launch_inputs=lambda _: options,
        load_traffic_signal_manifest=lambda _: SimpleNamespace(sumocfgs={"cologne1": "old.sumocfg"}))
    monkeypatch.setattr(validation, "_load_runtime", lambda: (runtime, {"runtime": "frozen85"}))
    monkeypatch.setenv("CFCMT_EXTERNAL_CONVERSION_ROOT", "before-test")
    repair = deepcopy(baseline)
    repair["metrics"].update(collision_events=0, collision_incidents=0, collision_samples=[])
    repair["metrics"]["accepted_intervention_trace"][2]["score"] = .25
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
        return baseline if len(calls) == 1 else repair
    monkeypatch.setattr(validation, "_run_arm", run_arm)
    result = validation.run_validation(sumocfg=tmp_path / "repair.sumocfg",
        geometry_report=tmp_path / "geometry.json", launch_record=launch,
        reference_result=reference, tripinfo=tmp_path / "tripinfo.xml")
    if changed:
        assert result["status"] == "BASELINE_REPRODUCTION_FAILED"
        assert not result["repair_executed"]
        assert len(calls) == 1
    else:
        assert result["status"] == "PASS"
        assert len(calls) == 2
        prefix = result["early_prefix_diagnostic"]
        assert not prefix["hard_gate"]
        assert prefix["retained_intervention_trace"]["exact_leading_row_count"] == 2
        assert prefix["retained_intervention_trace"]["first_difference"]["changed_fields"] == ["score"]
        assert not prefix["window_observations"]["passed"]


@pytest.mark.parametrize("failure", ["not_passed", "collision", "teleport", "short_run", "missing_stage"])
def test_zero_collisions_alone_cannot_pass_incomplete_repair(failure):
    arm = arm_fixture()
    arm["metrics"].update(collision_events=0, collision_incidents=0, collision_samples=[])
    assert all(validation.repair_checks(arm).values())
    if failure == "not_passed":
        arm["focal_passages"][validation.FOCAL_IDS[1]]["passed_controlled_junction"] = False
    elif failure == "collision":
        arm["metrics"]["collision_events"] = 1
    elif failure == "teleport":
        arm["metrics"]["starting_teleports"] = 1
    elif failure == "short_run":
        arm["step_times"].pop()
    else:
        arm["samples"].pop()
    assert not all(validation.repair_checks(arm).values())


def test_passage_requires_waiting_continuation_and_outgoing_and_clearance_is_signed():
    straight, left = validation.FOCAL_IDS
    routes = {straight: ["straight_in", "straight_out"], left: ["left_in", "left_out"]}
    tracker = validation.FocalPassageTracker(routes)
    current = {}
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getIDList=lambda: list(current), getRoute=lambda vehicle: routes[vehicle],
        getLaneID=lambda vehicle: current[vehicle][0], getRoadID=lambda vehicle: current[vehicle][1],
        getRouteIndex=lambda vehicle: int(current[vehicle][1] == routes[vehicle][1]),
        getLanePosition=lambda _: 1.0))
    assert not any(row["passed_controlled_junction"] for row in tracker.summary().values())
    current.update({straight: ("straight_in_0", "straight_in"), left: ("left_in_0", "left_in")})
    tracker.observe(api, 1)
    current.update({straight: (validation.FOCAL_LANES[straight][0], "internal_straight"),
                    left: (validation.FOCAL_LANES[left][0], "internal_left")})
    tracker.observe(api, 2)
    current[straight] = ("straight_out_0", "straight_out")
    tracker.observe(api, 3)
    assert tracker.summary()[straight]["passed_controlled_junction"]
    assert not tracker.summary()[left]["passed_controlled_junction"]
    current[left] = (validation.FOCAL_LANES[left][1], "internal_continuation")
    tracker.observe(api, 4)
    assert not tracker.summary()[left]["passed_controlled_junction"]
    current[left] = ("left_out_0", "left_out")
    tracker.observe(api, 5)
    assert tracker.summary()[left]["passed_controlled_junction"]

    polygon = [(2, .8), (3, .8), (3, 2), (2, 2)]
    overlap = validation.body_strip_clearance(polygon, [(0, 0), (10, 0)], 1.8)
    assert overlap["clearance_m"] == pytest.approx(-.1)
    assert overlap["body_within_straight_lane_span"]
    clear = validation.body_strip_clearance([(x, y + .2) for x, y in polygon], [(0, 0), (10, 0)], 1.8)
    assert clear["clearance_m"] == pytest.approx(.1)
