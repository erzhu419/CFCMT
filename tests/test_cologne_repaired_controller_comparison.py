from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_repaired_controller_comparison as comparison


def routes_fixture():
    return {
        "102219_396_0": ["-32038056#3", "-28198821#4"],
        "129962_409_0": ["28198821#3", "32038051#0"],
        "121463_406_0": ["28198821#3", "32038056#0"],
        "168358_425_0": ["-32038056#3", "32324544#0"],
    }


def geometry_fixture():
    tls = comparison.short.TLS
    routes = routes_fixture()
    rows = []
    for vehicle, lanes in {
        "102219_396_0": [("1_0", "1"), ("1_1", "2")],
        "121463_406_0": [("11_0", "11"), ("11_1", "12")],
        "129962_409_0": [("13_0", "13")],
        "168358_425_0": [("3_0", "3")],
    }.items():
        for lane, link in lanes:
            rows.append({"from": routes[vehicle][0], "to": routes[vehicle][1],
                         "fromLane": lane[-1], "toLane": lane[-1], "tl": tls,
                         "via": f":{tls}_{lane}", "linkIndex": link,
                         "dir": "l" if lane in ("13_0", "3_0") else "s"})
    for lane, continuation in (("13", "24"), ("3", "20")):
        rows.append({"from": f":{tls}_{lane}", "fromLane": "0", "via": f":{tls}_{continuation}_0"})
    return {"connections": rows}


def arm_fixture():
    return {
        "metrics": {"ok": True, "collision_events": 0, "collision_incidents": 0,
                    "collision_event_steps": 0, "starting_teleports": 0, "ending_teleports": 0,
                    "demand_population_at_horizon": 2015, "departed": 2014, "arrived": 1993,
                    "active_vehicles_at_horizon": 21, "pending_vehicles_at_horizon": 1,
                    "tripinfo_count": 2014., "tripinfo_completed_count": 1993.,
                    "tripinfo_unfinished_count": 21., "fixed_horizon_sample_seconds": 3541},
        "collision_inventory": {"native_event_count": 0, "native_incident_count": 0,
                                "collision_event_steps": 0},
        "step_times": list(range(25201, 28801)),
        "focal_passages": {vehicle: {"passed_controlled_junction": False, "hard_gate": False}
                           for vehicle in comparison.full.FOCAL_IDS},
        "originator_diagnostics": None, "deployed_policy": "phase_pressure",
    }


@pytest.mark.parametrize("straight_suffix", ["0", "1"])
def test_actual_legal_straight_lane_alternatives_and_left_continuations(straight_suffix):
    routes, geometry = routes_fixture(), geometry_fixture()
    tracker = comparison.RouteAwarePassageTracker(routes, geometry)
    tls = comparison.short.TLS
    current = {vehicle: (route[0] + "_0", route[0]) for vehicle, route in routes.items()}
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getIDList=lambda: list(current), getRoute=lambda vehicle: routes[vehicle],
        getLaneID=lambda vehicle: current[vehicle][0], getRoadID=lambda vehicle: current[vehicle][1],
        getRouteIndex=lambda vehicle: int(current[vehicle][1] == routes[vehicle][1]),
        getLanePosition=lambda _: 1.0))
    tracker.observe(api, 25500)
    actual_lanes = {"102219_396_0": f"1_{straight_suffix}", "121463_406_0": f"11_{straight_suffix}",
                    "129962_409_0": "13_0", "168358_425_0": "3_0"}
    for vehicle, lane in actual_lanes.items():
        current[vehicle] = (f":{tls}_{lane}", "internal")
    tracker.observe(api, 25700)
    for vehicle in ("102219_396_0", "121463_406_0"):
        current[vehicle] = (routes[vehicle][1] + "_1", routes[vehicle][1])
    for vehicle, lane in (("129962_409_0", "24_0"), ("168358_425_0", "20_0")):
        current[vehicle] = (f":{tls}_{lane}", "continuation")
    tracker.observe(api, 25701)
    assert sum(row["passed_controlled_junction"] for row in tracker.summary().values()) == 2
    for vehicle in routes:
        current[vehicle] = (routes[vehicle][1] + "_1", routes[vehicle][1])
    tracker.observe(api, 25704)
    summary = tracker.summary()
    assert all(row["passed_controlled_junction"] for row in summary.values())
    assert all(not row["hard_gate"] for row in summary.values())
    assert summary["102219_396_0"]["matched_internal_lane_sequence"] == [f":{tls}_1_{straight_suffix}"]
    assert summary["129962_409_0"]["matched_internal_lane_sequence"] == [f":{tls}_13_0", f":{tls}_24_0"]


@pytest.mark.parametrize("failure", ["population", "conservation", "tripinfo", "duration", "inventory", "teleport"])
def test_validity_catches_missing_population_or_observations_without_focal_gate(failure):
    arm = arm_fixture()
    assert all(comparison.comparison_validity(arm, 2015).values())
    assert not any(row["passed_controlled_junction"] for row in arm["focal_passages"].values())
    if failure == "population":
        arm["metrics"]["demand_population_at_horizon"] -= 1
    elif failure == "conservation":
        arm["metrics"]["arrived"] -= 1
    elif failure == "tripinfo":
        arm["metrics"]["tripinfo_count"] -= 1
    elif failure == "duration":
        arm["step_times"].pop()
    elif failure == "inventory":
        arm["collision_inventory"]["native_incident_count"] = 1
    else:
        arm["metrics"]["starting_teleports"] = 1
    assert not all(comparison.comparison_validity(arm, 2015).values())


@pytest.mark.parametrize("arm", comparison.ARMS)
def test_actual_policy_branch_and_runtime_arguments_are_frozen(tmp_path, monkeypatch, arm):
    calls, loaded = [], []
    originator = SimpleNamespace(diagnostics=lambda: {"decisions": 354})

    def load(*args, **kwargs):
        loaded.append(kwargs)
        return originator

    def evaluate(**kwargs):
        calls.append(kwargs)
        assert isinstance(kwargs["step_observer"].tracker, comparison.RouteAwarePassageTracker)
        kwargs["tripinfo_output"].write_text("temporary")
        return {"ok": True}

    runtime = SimpleNamespace(
        StateConditionedSourceUtilityOriginator=SimpleNamespace(load=load),
        _runtime_models=lambda model, prediction_horizon_sec: (model, prediction_horizon_sec),
        RUNTIME_POLICY="frozen_rigid_policy", load_libsumo=lambda: SimpleNamespace(),
        libsumo_version=lambda: "SUMO1.22.0", evaluate_policy_v3=evaluate,
        net_file_from_sumocfg=lambda _: Path("net.xml"), read_network_right_of_way_context=lambda _: {},
    )
    monkeypatch.setattr(comparison.importlib, "import_module", lambda _: SimpleNamespace())
    tripinfo = tmp_path / "CFCMT_SCRATCH" / "trip.xml"
    options = {"--runtime-model": "frozen_model.json", "--runtime-model-sha256": "existing-model-identifier"}
    result = comparison._run_arm(runtime, options, tmp_path / "fixed.sumocfg", routes_fixture(),
                                 geometry_fixture(), arm, 63792, tripinfo)
    assert not tripinfo.exists()
    call = calls[0]
    assert (call["duration_sec"], call["warmup_sec"], call["control_interval_sec"], call["seed"]) == (3600, 60, 10, 63792)
    assert (call["residual_coordination_mode"], call["residual_cooldown_intervals_override"]) == ("direct", 0)
    if arm == "phase_pressure":
        assert not loaded
        assert call["policy"] == "phase_pressure"
        assert call["models"] is None
        assert result["originator_diagnostics"] is None
    else:
        assert len(loaded) == 1
        assert loaded[0]["arm"] == "rigid_target_only"
        assert call["policy"] == "frozen_rigid_policy"
        assert call["models"] == (originator, 450)


@pytest.mark.parametrize("outcome", ["clean", "collision", "invalid", "geometry_changed", "bindings_changed"])
def test_cell_references_fixed_geometry_and_separates_valid_failure(tmp_path, monkeypatch, outcome):
    geometry, routes = geometry_fixture(), routes_fixture()
    cfg = tmp_path / "fixed.sumocfg"
    reference = {"protocol": comparison.full.PROTOCOL, "status": "PASS", "repaired_sumocfg": str(cfg),
                 "source_bindings": {"runtime": "frozen85"}, "bilateral_static_geometry": geometry,
                 "original_runtime_arguments": {"--seed": "41242", "--warmup-sec": "60", "--control-interval-sec": "10",
                                                "--prediction-horizon-sec": "450", "--conversion-root": str(tmp_path),
                                                "--runtime-model": "frozen_model.json"},
                 "focal_demand": {"vehicles": {vehicle: {"route": route} for vehicle, route in routes.items()}},
                 "repair": {"metrics": {"demand_population_at_horizon": 2015}}}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    actual_geometry = deepcopy(geometry)
    bindings = {"runtime": "frozen85"}
    if outcome == "geometry_changed":
        actual_geometry["connections"][0]["toLane"] = "different"
    elif outcome == "bindings_changed":
        bindings["runtime"] = "other_source"
    runtime = SimpleNamespace(net_file_from_sumocfg=lambda _: Path("net.xml"), static_geometry=lambda _: actual_geometry)
    monkeypatch.setattr(comparison, "_load_runtime", lambda: (runtime, bindings))
    monkeypatch.setenv("CFCMT_EXTERNAL_CONVERSION_ROOT", "before-test")
    actual = arm_fixture()
    if outcome == "collision":
        actual["metrics"].update(collision_events=2, collision_incidents=1, collision_event_steps=2)
        actual["collision_inventory"].update(native_event_count=2, native_incident_count=1, collision_event_steps=2)
    elif outcome == "invalid":
        actual["step_times"].pop()
    calls = []

    def run(*args):
        calls.append(args)
        return actual

    monkeypatch.setattr(comparison, "_run_arm", run)
    kwargs = dict(sumocfg=cfg, reference_result=reference_path, arm="phase_pressure", seed=41242,
                  tripinfo=tmp_path / "trip.xml")
    if outcome in ("geometry_changed", "bindings_changed"):
        with pytest.raises(ValueError):
            comparison.run_cell(**kwargs)
        assert not calls
    else:
        result = comparison.run_cell(**kwargs)
        assert len(calls) == 1
        assert result["seed_roster"] == [52282, 41242, 63792]
        assert result["seed"] == 41242
        assert result["runtime_model"] is None
        assert result["status"] == {"clean": "PASS", "collision": "FAIL", "invalid": "INVALID"}[outcome]
        assert result["run_valid"] == (outcome != "invalid")
        assert result["zero_incident_status"] == {"clean": "PASS", "collision": "FAIL", "invalid": "UNAVAILABLE"}[outcome]


@pytest.mark.parametrize("arm,seed", [("rigid_target_only", 41242), ("phase_pressure", 99999)])
def test_reused_cell_and_new_seed_are_rejected_before_runtime(tmp_path, monkeypatch, arm, seed):
    def unexpected_runtime():
        pytest.fail("Invalid or reused cells must not load the runtime")

    monkeypatch.setattr(comparison, "_load_runtime", unexpected_runtime)
    with pytest.raises(ValueError):
        comparison.run_cell(sumocfg=tmp_path / "cfg", reference_result=tmp_path / "missing.json",
                            arm=arm, seed=seed, tripinfo=tmp_path / "trip.xml")
