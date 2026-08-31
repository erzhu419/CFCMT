from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_intas_full_day_admission import (
    EXPECTED_BUS_VEHICLES,
    EXPECTED_CAR_VEHICLES,
    EXPECTED_PERSONS,
    EXPECTED_TRAFFIC_LIGHTS,
    MAXIMUM_TIME_SEC,
    PACKAGE_PROTOCOL,
    SCENARIO,
    SEED,
    _admission_checks,
    _final_summary_step,
    _validate_package,
)


def _package(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "package"
    root.mkdir()
    (root / "strict_full_day.sumocfg").write_text(
        "<configuration/>", encoding="utf-8"
    )
    payload = {
        "protocol": PACKAGE_PROTOCOL,
        "scenario": SCENARIO,
        "passed": True,
        "static_failures": [],
        "workers": 8,
        "car_demand_inventory": {"vehicle_count": EXPECTED_CAR_VEHICLES},
        "bus_demand_inventory": {
            "expanded_vehicle_count": EXPECTED_BUS_VEHICLES
        },
        "pedestrian_demand_inventory": {"person_count": EXPECTED_PERSONS},
        "network_inventory": {"traffic_light_count": EXPECTED_TRAFFIC_LIGHTS},
        "strict_execution": {
            "config": "strict_full_day.sumocfg",
            "backend": "libsumo",
            "sumo_version": "1.22.0",
            "seed": SEED,
            "begin_sec": 0,
            "end_sec": int(MAXIMUM_TIME_SEC),
            "step_length_sec": 0.1,
            "time_to_teleport_sec": -1,
            "max_depart_delay_sec": -1,
            "demand_scale": 1.0,
            "collision_check_junctions": True,
            "collision_action": "warn",
        },
    }
    manifest = root / "package_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return root, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_validate_package_requires_frozen_identity_and_complete_demand(
    tmp_path: Path,
) -> None:
    root, identity = _package(tmp_path)

    config, payload, observed_identity = _validate_package(
        root, expected_manifest_sha256=identity
    )

    assert config.name == "strict_full_day.sumocfg"
    assert payload["car_demand_inventory"]["vehicle_count"] == 185_923
    assert observed_identity == identity
    with pytest.raises(ValueError, match="identity changed"):
        _validate_package(root, expected_manifest_sha256="0" * 64)


def test_final_summary_step_reads_cumulative_vehicle_safety_counts(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.xml"
    summary.write_text(
        "<summary><step time='60' loaded='4' inserted='3' running='2' "
        "waiting='1' ended='1' arrived='1' collisions='0' teleports='0'/>"
        "<step time='120' loaded='4' inserted='4' running='0' waiting='0' "
        "ended='4' arrived='4' collisions='0' teleports='0'/></summary>",
        encoding="utf-8",
    )

    final = _final_summary_step(summary)

    assert final["time"] == 120.0
    assert final["loaded"] == 4
    assert final["arrived"] == 4
    assert final["collisions"] == 0
    assert final["teleports"] == 0


def test_admission_checks_require_vehicle_and_person_conservation() -> None:
    observed = {
        "termination_reason": "all_entities_completed",
        "departed_vehicles": 4,
        "arrived_vehicles": 4,
        "departed_persons": 2,
        "arrived_persons": 2,
        "final_active_vehicles": 0,
        "final_active_persons": 0,
        "final_pending_vehicles": 0,
        "final_min_expected": 0,
        "unique_collision_incidents": 0,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "controllable_tls_count": 3,
    }
    summary = {
        "loaded": 4,
        "arrived": 4,
        "running": 0,
        "waiting": 0,
        "collisions": 0,
        "teleports": 0,
    }

    checks = _admission_checks(
        expected_vehicle_count=4,
        expected_person_count=2,
        expected_traffic_lights=3,
        observed=observed,
        summary=summary,
        error=None,
    )

    assert all(checks.values())
    observed["arrived_persons"] = 1
    failed = _admission_checks(
        expected_vehicle_count=4,
        expected_person_count=2,
        expected_traffic_lights=3,
        observed=observed,
        summary=summary,
        error=None,
    )
    assert failed["exact_person_demand_arrived"] is False
