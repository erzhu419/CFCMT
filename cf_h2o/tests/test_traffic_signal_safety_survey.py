from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    SUMO_EXECUTION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_safety_survey import run_safety_survey


def _metrics(*, collision_events=0, collision_incidents=0, teleports=0):
    return {
        "ok": True,
        "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        "collision_events": collision_events,
        "collision_event_steps": collision_events,
        "collision_incidents": collision_incidents,
        "collision_rate": float(collision_events),
        "collision_incident_rate": float(collision_incidents),
        "emergency_stops": 0,
        "starting_teleports": teleports,
        "ending_teleports": 0,
        "mean_system_vehicles_per_controlled_lane": 2.0,
    }


def _patch_survey(monkeypatch, *, teleports=0):
    manifest = SimpleNamespace(
        sumocfgs={"network": Path("network.sumocfg")},
        city_groups={"network": "city"},
        to_dict=lambda: {"network": "network.sumocfg"},
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_safety_survey.load_traffic_signal_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_safety_survey.libsumo_version",
        lambda: "1.22.0",
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_safety_survey._runtime_metadata",
        lambda: {"source_tree_sha256": "hash"},
    )

    def fake_map(worker, jobs, workers):
        return [
            {
                "scenario": job["scenario"],
                "city_group": job["city_group"],
                "policy": job["policy"],
                "seed": job["seed"],
                "metrics": _metrics(
                    collision_events=3 if job["policy"] == "fixed_program" else 0,
                    collision_incidents=1 if job["policy"] == "fixed_program" else 0,
                    teleports=teleports,
                ),
            }
            for job in jobs
        ]

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_safety_survey._map_fresh_sumo_processes",
        fake_map,
    )


def test_safety_survey_reports_native_collision_without_failing(tmp_path, monkeypatch):
    _patch_survey(monkeypatch)
    result = run_safety_survey(
        manifest_path=Path("manifest.json"),
        scenarios=("network",),
        policies=("fixed_program", "phase_pressure"),
        seeds=(17,),
        duration_sec=600.0,
        control_interval_sec=10,
        warmup_sec=60.0,
        workers=2,
        tripinfo_root=tmp_path,
        expected_sumo_version="1.22.0",
    )

    assert result["passed"] is True
    assert result["raw_collision_rollout_count"] == 1
    assert result["aggregate"]["network"]["fixed_program"][
        "collision_incidents"
    ] == 1.0


def test_safety_survey_rejects_any_teleport(tmp_path, monkeypatch):
    _patch_survey(monkeypatch, teleports=1)
    result = run_safety_survey(
        manifest_path=Path("manifest.json"),
        scenarios=("network",),
        policies=("fixed_program",),
        seeds=(17,),
        duration_sec=600.0,
        control_interval_sec=10,
        warmup_sec=60.0,
        workers=1,
        tripinfo_root=tmp_path,
        expected_sumo_version="1.22.0",
    )

    assert result["passed"] is False
    assert any("teleport event" in error for error in result["errors"])
