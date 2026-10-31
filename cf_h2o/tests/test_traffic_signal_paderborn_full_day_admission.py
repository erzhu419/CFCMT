from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_paderborn_full_day_admission import (
    PACKAGE_PROTOCOL,
    SCENARIO,
    _admission_checks,
    _final_summary_step,
    _validate_package,
)


def _package(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "package"
    root.mkdir()
    (root / "strict_full_day.sumo.cfg").write_text("<configuration/>", encoding="utf-8")
    payload = {
        "protocol": PACKAGE_PROTOCOL,
        "scenario": SCENARIO,
        "passed": True,
        "gates": {"a": True, "b": True},
        "route_inventory": {"vehicle_count": 203_387},
        "microscopic_instantiation": {"config": "strict_full_day.sumo.cfg"},
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

    assert config.name == "strict_full_day.sumo.cfg"
    assert payload["route_inventory"]["vehicle_count"] == 203_387
    assert observed_identity == identity
    with pytest.raises(ValueError, match="identity changed"):
        _validate_package(root, expected_manifest_sha256="0" * 64)


def test_final_summary_step_reads_cumulative_safety_counts(tmp_path: Path) -> None:
    summary = tmp_path / "summary.xml"
    summary.write_text(
        "<summary><step time='60' loaded='4' inserted='3' running='2' "
        "waiting='1' ended='1' arrived='1' collisions='0' teleports='0'/>"
        "<step time='120' loaded='4' inserted='4' running='0' waiting='0' "
        "ended='4' arrived='4' collisions='0' teleports='0'/></summary>",
        encoding="utf-8",
    )

    final = _final_summary_step(summary)

    assert final == {
        "time": 120.0,
        "loaded": 4,
        "inserted": 4,
        "running": 0,
        "waiting": 0,
        "ended": 4,
        "arrived": 4,
        "collisions": 0,
        "teleports": 0,
    }


def test_admission_checks_require_completion_not_only_disabled_teleports() -> None:
    observed = {
        "termination_reason": "all_vehicles_completed",
        "departed": 4,
        "arrived": 4,
        "final_active_vehicles": 0,
        "final_pending_vehicles": 0,
        "final_min_expected": 0,
        "unique_collision_incidents": 0,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "controllable_tls_count": 2,
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
        expected_minimum_tls=2,
        observed=observed,
        summary=summary,
        error=None,
    )

    assert all(checks.values())
    observed["termination_reason"] = "fixed_completion_cap_reached"
    observed["final_min_expected"] = 1
    failed = _admission_checks(
        expected_vehicle_count=4,
        expected_minimum_tls=2,
        observed=observed,
        summary=summary,
        error=None,
    )
    assert failed["all_vehicles_completed_before_cap"] is False
    assert failed["no_expected_vehicles"] is False
