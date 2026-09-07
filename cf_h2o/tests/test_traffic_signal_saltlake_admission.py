from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval import traffic_signal_saltlake_admission as admission


def _scenario(tmp_path: Path, expected: int = 10) -> Path:
    root = tmp_path / "saltlake_test"
    root.mkdir()
    cfg = root / "saltlake_test.sumocfg"
    cfg.write_text("<configuration/>\n", encoding="utf-8")
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "construction_protocol": "test-v1",
                "demand_audit": {"generated_vehicle_total": expected},
                "hashes": {"output_route_sha256": "abc"},
            }
        ),
        encoding="utf-8",
    )
    return cfg


def _metrics(
    *,
    loaded: int = 10,
    departed: int | None = None,
    pending: int = 0,
    collisions: int = 0,
) -> dict[str, object]:
    departed = loaded if departed is None else departed
    return {
        "ok": True,
        "loaded": loaded,
        "departed": departed,
        "arrived": max(departed - 1, 0),
        "pending_vehicles_at_horizon": pending,
        "demand_population_at_horizon": departed + pending,
        "active_vehicles_at_horizon": 1,
        "departure_service_ratio": 1.0,
        "completion_ratio": 0.9,
        "mean_system_vehicles_per_controlled_lane": 2.0,
        "collision_events": collisions,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "phase_execution_audit": {},
    }


def test_native_fixed_collision_is_warning_when_safe_policy_passes(tmp_path: Path, monkeypatch) -> None:
    cfg = _scenario(tmp_path)

    def fake_map(worker, jobs, workers):
        return [
            {
                "target": job["scenario"],
                "policy": job["policy"],
                "seed": job["seed"],
                "metrics": _metrics(collisions=2 if job["policy"] == "fixed_program" else 0),
            }
            for job in jobs
        ]

    monkeypatch.setattr(admission, "_map_fresh_sumo_processes", fake_map)
    monkeypatch.setattr(admission, "runtime_metadata", lambda: {"libsumo_version": "1.22.0"})

    result = admission.run_admission(
        [cfg],
        policies=("fixed_program", "phase_pressure"),
        safety_policy="phase_pressure",
        tripinfo_root=tmp_path / "tripinfo",
    )

    assert result["status"] == "PASS"
    assert not result["errors"]
    assert len(result["warnings"]) == 1


def test_safe_policy_collision_and_population_mismatch_fail(tmp_path: Path, monkeypatch) -> None:
    cfg = _scenario(tmp_path, expected=10)

    def fake_map(worker, jobs, workers):
        return [
            {
                "target": job["scenario"],
                "policy": job["policy"],
                "seed": job["seed"],
                "metrics": _metrics(loaded=9, collisions=1),
            }
            for job in jobs
        ]

    monkeypatch.setattr(admission, "_map_fresh_sumo_processes", fake_map)
    monkeypatch.setattr(admission, "runtime_metadata", lambda: {"libsumo_version": "1.22.0"})

    result = admission.run_admission(
        [cfg],
        policies=("phase_pressure",),
        safety_policy="phase_pressure",
        tripinfo_root=tmp_path / "tripinfo",
    )

    assert result["status"] == "FAIL"
    assert any("observed demand population 9 vehicles" in error for error in result["errors"])
    assert any("collisions=1" in error for error in result["errors"])


def test_loaded_boundary_event_mismatch_is_diagnostic_only(tmp_path: Path, monkeypatch) -> None:
    cfg = _scenario(tmp_path, expected=10)

    def fake_map(worker, jobs, workers):
        return [
            {
                "target": job["scenario"],
                "policy": job["policy"],
                "seed": job["seed"],
                "metrics": _metrics(loaded=9, departed=8, pending=2),
            }
            for job in jobs
        ]

    monkeypatch.setattr(admission, "_map_fresh_sumo_processes", fake_map)
    monkeypatch.setattr(admission, "runtime_metadata", lambda: {"libsumo_version": "1.22.0"})

    result = admission.run_admission(
        [cfg],
        policies=("phase_pressure",),
        safety_policy="phase_pressure",
        tripinfo_root=tmp_path / "tripinfo",
    )

    assert result["status"] == "PASS"
    assert not result["errors"]
    assert any("loaded-event count 9" in warning for warning in result["warnings"])
    assert result["safety_summary"][0]["demand_population_at_horizon"] == 10
