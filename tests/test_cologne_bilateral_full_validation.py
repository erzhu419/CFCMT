from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_bilateral_full_validation as validation


def arm_fixture():
    metrics = {key: 0 for key in validation.TRAFFIC_METRIC_KEYS}
    times = list(range(25260, 25490, 10)) + list(range(25700, 28740, 20))
    metrics.update(ok=True, collision_events=36, collision_incidents=18, collision_event_steps=36,
                   mean_tripinfo_waiting_time=19.712655086848635,
                   accepted_intervention_trace=[{"time_sec": t, "score": .125} for t in times],
                   collision_samples=[{"time": 25657 + i} for i in range(20)],
                   guard_audit={"decisions": 1})
    passages = {vehicle: {"passed_controlled_junction": True} for vehicle in validation.FOCAL_IDS}
    return {"metrics": metrics, "collision_inventory": {
                "native_event_count": 36, "native_incident_count": 18, "collision_event_steps": 36},
            "step_times": list(range(validation.BEGIN + 1, validation.END + 1)),
            "focal_passages": deepcopy(passages), "prefix_passages": deepcopy(passages),
            "samples": [{"time_sec": t, "stage": stage, "vehicles": {}}
                        for t in range(25630, 25666) for stage in validation.short.STAGES],
            "originator_diagnostics": {"decision_count": 354},
            "static_geometry": {}, "native_lane_catalog": {}}


def collision_free_arm():
    arm = arm_fixture()
    arm["metrics"].update(collision_events=0, collision_incidents=0, collision_event_steps=0, collision_samples=[])
    arm["collision_inventory"].update(native_event_count=0, native_incident_count=0, collision_event_steps=0)
    return arm


@pytest.mark.parametrize("changed", ["action", "retained_collision", "traffic", "guard_audit", "originator", None])
def test_exact_full_baseline_is_required_before_bilateral(tmp_path, monkeypatch, changed):
    baseline = arm_fixture()
    full_reference = tmp_path / "full.json"
    full_reference.write_text(json.dumps({
        "city": "cologne", "scenario": "cologne1", "seed": 41242, "arm": "rigid_target_only",
        "feature_binding": "stored_feature_names", "duration_sec": 3600,
        "metrics": baseline["metrics"], "originator_diagnostics": baseline["originator_diagnostics"],
    }))
    repair = collision_free_arm()
    routes = {vehicle: [f"in_{i}", f"out_{i}"] for i, vehicle in enumerate(validation.FOCAL_IDS)}
    demand = {"vehicles": {vehicle: {"route": route} for vehicle, route in routes.items()}}
    short_reference = tmp_path / "short.json"
    repaired_cfg = tmp_path / "existing_v154d.sumocfg"
    short_reference.write_text(json.dumps({
        "protocol": validation.short.PROTOCOL, "status": "FAIL", "repaired_sumocfg": str(repaired_cfg),
        "focal_demand": demand, "repair": {
            "accepted_intervention_trace": [row for row in repair["metrics"]["accepted_intervention_trace"]
                                            if row["time_sec"] < validation.PREFIX_END],
            "samples": repair["samples"], "focal_passages": repair["prefix_passages"]},
    }))
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    options = {"--seed": "41242", "--warmup-sec": "60", "--control-interval-sec": "10",
               "--prediction-horizon-sec": "450", "--conversion-root": str(tmp_path), "--manifest": "manifest.json"}
    runtime = SimpleNamespace(launch_inputs=lambda _: options,
        load_traffic_signal_manifest=lambda _: SimpleNamespace(sumocfgs={"cologne1": "original.sumocfg"}))
    monkeypatch.setattr(validation, "_load_runtime", lambda: (runtime, {"runtime": "source85"}))
    monkeypatch.setattr(validation.short, "read_focal_demand", lambda _: (routes, demand))
    monkeypatch.setenv("CFCMT_EXTERNAL_CONVERSION_ROOT", "before-test")
    if changed == "action":
        baseline["metrics"]["accepted_intervention_trace"][-1]["score"] += 1
    elif changed == "retained_collision":
        baseline["metrics"]["collision_samples"].pop()
    elif changed == "traffic":
        baseline["metrics"]["mean_tripinfo_waiting_time"] += .01
    elif changed == "guard_audit":
        baseline["metrics"]["guard_audit"]["decisions"] += 1
    elif changed == "originator":
        baseline["originator_diagnostics"]["decision_count"] += 1
    calls = []

    def run_arm(*args):
        calls.append(args[2])
        return baseline if len(calls) == 1 else repair

    monkeypatch.setattr(validation, "_run_arm", run_arm)
    result = validation.run_validation(sumocfg=repaired_cfg, launch_record=launch,
        reference_result=full_reference, short_reference_result=short_reference, tripinfo=tmp_path / "trip.xml")
    if changed:
        assert result["status"] == "BASELINE_REPRODUCTION_FAILED"
        assert not result["repair_executed"]
        assert len(calls) == 1
    else:
        assert result["status"] == "PASS"
        assert len(calls) == 2
        assert result["v154d_prefix_reproduction"]["prior_v154d_status"] == "FAIL"
        assert "samples" not in result["repair"]
        assert "samples" not in result["baseline"]
        assert len(result["baseline"]["metrics"]["accepted_intervention_trace"]) == 175


@pytest.mark.parametrize("failure", ["missing_step", "capped_inventory", "wrong_incident_key",
                                    "event_steps", "late_collision", "teleport", "focal_late", "prefix_drift"])
def test_full_completion_inventory_and_passage_gates(failure):
    arm = collision_free_arm()
    prefix = {"passed": True}
    if failure == "missing_step":
        arm["step_times"].pop()
    elif failure == "capped_inventory":
        arm["metrics"].update(collision_events=36, collision_incidents=18, collision_event_steps=36)
        arm["collision_inventory"].update(native_event_count=20, native_incident_count=10, collision_event_steps=36)
    elif failure == "wrong_incident_key":
        arm["metrics"].update(collision_events=2, collision_incidents=2, collision_event_steps=1)
        arm["collision_inventory"].update(native_event_count=2, native_incident_count=1, collision_event_steps=1)
    elif failure == "event_steps":
        arm["collision_inventory"]["collision_event_steps"] = 1
    elif failure == "late_collision":
        arm["metrics"].update(collision_events=1, collision_incidents=1, collision_event_steps=1)
        arm["collision_inventory"].update(native_event_count=1, native_incident_count=1, collision_event_steps=1)
    elif failure == "teleport":
        arm["metrics"]["ending_teleports"] = 1
    elif failure == "focal_late":
        arm["focal_passages"][validation.FOCAL_IDS[0]]["passed_controlled_junction"] = False
    else:
        prefix["passed"] = False
    assert not all(validation.repair_checks(arm, prefix).values())
    if failure == "late_collision":
        assert all(validation.completeness_checks(arm).values())


def test_short_prefix_keeps_incomplete_v154d_passage_without_blocking_later_passage():
    arm = collision_free_arm()
    vehicle = validation.FOCAL_IDS[0]
    arm["prefix_passages"][vehicle]["passed_controlled_junction"] = False
    reference = {"status": "FAIL", "repair": {
        "accepted_intervention_trace": [row for row in arm["metrics"]["accepted_intervention_trace"]
                                        if row["time_sec"] < validation.PREFIX_END],
        "samples": deepcopy(arm["samples"]), "focal_passages": deepcopy(arm["prefix_passages"])}}
    assert validation.short_prefix_audit(reference, arm)["passed"]
    assert all(validation.repair_checks(arm, validation.short_prefix_audit(reference, arm)).values())
    arm["samples"][0]["changed_position"] = True
    assert not validation.short_prefix_audit(reference, arm)["passed"]


def test_uncapped_inventory_reuses_native_keys_and_first_incident_lanes():
    waiting, through, continuation = ":tls_3_0", ":tls_11_1", ":tls_20_0"
    geometry = {"connections": [
        {"from": "west", "to": "north", "fromLane": "1", "toLane": "1", "via": waiting,
         "tl": validation.short.TLS, "linkIndex": "3", "dir": "l"},
        {"from": ":tls_3", "to": "north", "fromLane": "0", "toLane": "1", "via": continuation},
        {"from": "east", "to": "west_out", "fromLane": "1", "toLane": "1", "via": through,
         "tl": validation.short.TLS, "linkIndex": "12", "dir": "s"},
    ]}
    assert validation.movement_catalog(geometry)[continuation]["linkIndex"] == "3"
    vehicles = [f"car{i}" for i in range(24)]
    current = {vehicle: {"lane_id": waiting if i % 2 == 0 else through, "speed_mps": 0 if i % 2 == 0 else 8}
               for i, vehicle in enumerate(vehicles)}
    collisions = [SimpleNamespace(collider=vehicles[2*i], victim=vehicles[2*i+1],
                  native_key=(vehicles[2*i], vehicles[2*i+1], "junction", through)) for i in range(12)]
    queried_caps, queried_keys = [], []

    def debug_samples(**kwargs):
        queried_caps.append(kwargs["max_samples"])
        return [{"time": kwargs["time_sec"], "collider": c.collider, "victim": c.victim}
                for c in kwargs["collisions"][:kwargs["max_samples"]]]

    def native_key(collision):
        queried_keys.append(collision.native_key)
        return collision.native_key

    api = SimpleNamespace(vehicle=SimpleNamespace(getIDList=lambda: vehicles),
                          simulation=SimpleNamespace(getCollisions=lambda: collisions))
    runtime = SimpleNamespace(vehicle_pose=lambda _, vehicle: deepcopy(current[vehicle]))
    v3 = SimpleNamespace(_collision_debug_samples=debug_samples, _collision_incident_key=native_key)
    observer = validation.FullObserver(api, lambda **_: None,
        {vehicle: ["in", "out"] for vehicle in validation.FOCAL_IDS}, runtime, v3, geometry)
    observer(stage=validation.short.STAGES[0], time_sec=26700, executors={})
    observer(stage=validation.short.STAGES[1], time_sec=26701, executors={})
    observer(stage=validation.short.STAGES[2], time_sec=26701, executors={})
    for pose in current.values():
        pose["lane_id"] = "outgoing"
    observer(stage=validation.short.STAGES[1], time_sec=26702, executors={})
    inventory = observer.inventory()
    assert queried_caps == [12, 12]
    assert len(queried_keys) == 24
    assert inventory["native_event_count"] == 24
    assert inventory["native_incident_count"] == 12
    assert inventory["collision_event_steps"] == 2
    assert inventory["event_sample_cap"] is None
    assert len(inventory["incident_counts_by_first_participant_lane_pair"]) == 1
    assert inventory["incident_counts_by_first_participant_lane_pair"][0]["incident_count"] == 12
    assert "outgoing" not in inventory["incident_counts_by_first_participant_lane_pair"][0]["lane_pair"]
    assert inventory["incidents"][0]["event_indices"] == [0, 12]
    assert inventory["events"][0]["participant_poses"]["collider"]["speed_mps"] == 0
    assert inventory["events"][-1]["participant_poses"]["collider"]["lane_id"] == "outgoing"
    assert inventory["observation_stage"] == "after_simulation_step_before_executor"
    missing = vehicles.pop()
    observer(stage=validation.short.STAGES[1], time_sec=26703, executors={})
    assert observer.inventory()["events"][-1]["unavailable_participant_poses"] == {
        "victim": {"vehicle_id": missing, "reason": "vehicle_not_active_at_observation"}}
