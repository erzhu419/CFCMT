from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_priority_yellow_comparison as comparison


@pytest.mark.parametrize("change", [None, "timer", "old_bindings", "stale_executor"])
def test_only_approved_executor_overlay_binds_to_actual_runtime(tmp_path, monkeypatch, change):
    corrected = (Path(__file__).resolve().parents[1] / "cf_h2o/traffic_signal/safe_phase_controller.py").read_text()
    assert corrected.count(comparison._NEW_BRANCH) == 1
    original = corrected.replace(comparison._NEW_BRANCH, comparison._OLD_BRANCH)
    source = tmp_path / "85b2b43cd12832c5a1dd"
    original_path = source / "cf_h2o/traffic_signal/safe_phase_controller.py"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(original)
    patch_path = tmp_path / "corrected_executor.py"
    if change == "timer":
        corrected = corrected.replace("self.remaining_sec = 0.0", "self.remaining_sec = 1.0", 1)
    patch_path.write_text(corrected)
    monkeypatch.setenv("CFCMT_SOURCE_ROOT", str(source))
    monkeypatch.setitem(sys.modules, comparison.EXECUTOR_MODULE, None)
    expected_bindings = {"runtime": str(source / "runtime.py")}
    bindings = {"runtime": "other_source.py"} if change == "old_bindings" else expected_bindings
    runtime = SimpleNamespace()
    monkeypatch.setattr(comparison.comparison, "_load_runtime", lambda: (runtime, bindings))

    def import_runtime(name):
        assert name == "cf_h2o.eval.traffic_signal_resco_cfcmt_v3"
        executor = sys.modules[comparison.EXECUTOR_MODULE]
        return SimpleNamespace(
            SafePhaseExecutor=object if change == "stale_executor" else executor.SafePhaseExecutor,
            build_safe_phase_executors=executor.build_safe_phase_executors)

    monkeypatch.setattr(comparison.importlib, "import_module", import_runtime)
    if change:
        with pytest.raises(ValueError):
            comparison._load_priority_runtime(patch_path, expected_bindings)
    else:
        loaded, actual_bindings, probe = comparison._load_priority_runtime(patch_path, expected_bindings)
        assert loaded is runtime
        assert actual_bindings == expected_bindings
        assert probe == {"initial_state": "Gg", "requested_state": "rG", "observed_state": "Yy", "remaining_sec": 3.}
        assert Path(sys.modules[comparison.EXECUTOR_MODULE].__file__) == patch_path


def reference_fixture(tmp_path):
    cfg = tmp_path / "fixed.sumocfg"
    geometry = {"connections": []}
    reference = {"protocol": comparison.comparison.full.PROTOCOL, "status": "PASS",
        "repaired_sumocfg": str(cfg), "source_bindings": {"runtime": "frozen85"},
        "bilateral_static_geometry": geometry,
        "original_runtime_arguments": {"--seed": "41242", "--warmup-sec": "60",
            "--control-interval-sec": "10", "--prediction-horizon-sec": "450",
            "--conversion-root": str(tmp_path), "--runtime-model": "frozen_model.json"},
        "focal_demand": {"vehicles": {vehicle: {"route": ["in", "out"]}
            for vehicle in comparison.comparison.full.FOCAL_IDS}},
        "repair": {"metrics": {"demand_population_at_horizon": 2015}}}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    return cfg, path, reference


def result_fixture():
    return {
        "metrics": {"ok": True, "collision_events": 0, "collision_incidents": 0,
            "collision_event_steps": 0, "starting_teleports": 0, "ending_teleports": 0,
            "demand_population_at_horizon": 2015, "departed": 2014, "arrived": 1993,
            "active_vehicles_at_horizon": 21, "pending_vehicles_at_horizon": 1,
            "tripinfo_count": 2014., "tripinfo_completed_count": 1993.,
            "tripinfo_unfinished_count": 21., "fixed_horizon_sample_seconds": 3541},
        "collision_inventory": {"native_event_count": 0, "native_incident_count": 0, "collision_event_steps": 0},
        "step_times": list(range(25201, 28801)), "focal_passages": {},
        "originator_diagnostics": None, "deployed_policy": "test_policy"}


@pytest.mark.parametrize("arm,seed,status", [
    ("rigid_target_only", 52282, "PASS"), ("rigid_target_only", 41242, "PASS"),
    ("rigid_target_only", 63792, "FAIL"), ("phase_pressure", 52282, "PASS"),
    ("phase_pressure", 41242, "INVALID"), ("phase_pressure", 63792, "PASS"),
])
def test_all_six_cells_run_anew_and_retain_valid_collision_failures(tmp_path, monkeypatch, arm, seed, status):
    cfg, reference_path, reference = reference_fixture(tmp_path)
    runtime = SimpleNamespace(net_file_from_sumocfg=lambda _: Path("net.xml"),
                              static_geometry=lambda _: deepcopy(reference["bilateral_static_geometry"]))
    probe = {"observed_state": "Yy"}
    monkeypatch.setattr(comparison, "_load_priority_runtime", lambda *_: (runtime, reference["source_bindings"], probe))
    monkeypatch.setenv("CFCMT_EXTERNAL_CONVERSION_ROOT", "before-test")
    actual, calls = result_fixture(), []
    if status == "FAIL":
        actual["metrics"].update(collision_events=2, collision_incidents=1, collision_event_steps=2)
        actual["collision_inventory"].update(native_event_count=2, native_incident_count=1, collision_event_steps=2)
    elif status == "INVALID":
        actual["metrics"]["arrived"] -= 1

    def run(*args):
        calls.append(args)
        return actual

    monkeypatch.setattr(comparison.comparison, "_run_arm", run)
    executor_source = tmp_path / "corrected_executor.py"
    result = comparison.run_cell(sumocfg=cfg, reference_result=reference_path, executor_source=executor_source,
                                  arm=arm, seed=seed, tripinfo=tmp_path / "trip.xml")
    assert len(calls) == 1
    assert calls[0][0] is runtime
    assert calls[0][1] == reference["original_runtime_arguments"]
    assert calls[0][2] == cfg
    assert calls[0][5:7] == (arm, seed)
    assert result["status"] == status
    assert result["run_valid"] == (status != "INVALID")
    assert result["zero_incident_status"] == {"PASS": "PASS", "FAIL": "FAIL", "INVALID": "UNAVAILABLE"}[status]
    assert result["source_bindings"] == reference["source_bindings"]
    assert result["executor_source"] == str(executor_source)
    assert result["runtime_model"] == ("frozen_model.json" if arm == "rigid_target_only" else None)
    assert result["seed_roster"] == [52282, 41242, 63792]


def test_network_change_is_rejected_before_simulation(tmp_path, monkeypatch):
    cfg, reference_path, reference = reference_fixture(tmp_path)
    runtime = SimpleNamespace(net_file_from_sumocfg=lambda _: Path("net.xml"),
                              static_geometry=lambda _: {"connections": ["changed"]})
    monkeypatch.setattr(comparison, "_load_priority_runtime", lambda *_: (runtime, reference["source_bindings"], {}))
    monkeypatch.setattr(comparison.comparison, "_run_arm", lambda *_: pytest.fail("Changed network must not run"))
    with pytest.raises(ValueError, match="geometry"):
        comparison.run_cell(sumocfg=cfg, reference_result=reference_path,
                            executor_source=tmp_path / "executor.py", arm="phase_pressure", seed=52282,
                            tripinfo=tmp_path / "trip.xml")
