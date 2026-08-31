from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from cf_h2o.eval.traffic_signal_chicago_full_week_admission import (
    EXPECTED_TLS_COUNT,
    _admission_checks,
    _final_summary_step,
    _validate_route_package,
)
from scripts.data.acquire_chicago_for_hire_unseen import DATES
from scripts.data.prepare_chicago_for_hire_demand import EXPECTED_NETWORK_SHA256
from scripts.data.route_chicago_for_hire_demand import (
    EXPECTED_INCLUDED_TRIPS,
    PROTOCOL as ROUTE_PROTOCOL,
    SAFETY_SEEDS,
    strict_config_tree,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "package"
    root.mkdir()
    network_root = tmp_path / "network"
    network_root.mkdir()
    network = network_root / "chicago.net.xml.gz"
    network.write_bytes(b"network")
    (network_root / "network_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "network_sha256": EXPECTED_NETWORK_SHA256,
                "network_inventory": {"traffic_light_count": EXPECTED_TLS_COUNT},
            }
        ),
        encoding="utf-8",
    )
    daily = []
    for day_index, (day, seed) in enumerate(zip(DATES, SAFETY_SEEDS, strict=True)):
        route = root / f"routes_{day.isoformat()}.rou.xml.gz"
        with gzip.open(route, "wt", encoding="utf-8") as handle:
            handle.write("<routes/>")
        config = root / f"strict_{day.isoformat()}.sumocfg"
        strict_config_tree(
            network_file=network,
            route_file_name=route.name,
            seed=seed,
        ).write(config, encoding="utf-8", xml_declaration=True)
        daily.append(
            {
                "date": day.isoformat(),
                "day_index": day_index,
                "vehicle_count": day_index + 1,
                "route_file": route.name,
                "route_size_bytes": route.stat().st_size,
                "route_sha256": _sha256(route),
                "strict_config": config.name,
                "strict_seed": seed,
            }
        )
    payload = {
        "protocol": ROUTE_PROTOCOL,
        "status": "PASS",
        "sumo_version": "1.22.0",
        "network_root": str(network_root),
        "network_sha256": EXPECTED_NETWORK_SHA256,
        "routed_vehicle_count": EXPECTED_INCLUDED_TRIPS,
        "bulk_routing": False,
        "repair": False,
        "ignore_errors": False,
        "strict_execution": {
            "step_length_sec": 1.0,
            "end_sec": 108_000,
            "demand_scale": 1.0,
            "max_depart_delay_sec": -1,
            "time_to_teleport_sec": -1,
            "collision_action": "warn",
            "collision_check_junctions": True,
            "route_errors_fatal": True,
        },
        "daily_routes": daily,
    }
    manifest = root / "route_package_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return root, _sha256(manifest)


def test_validate_route_package_requires_exact_manifest_route_and_config(
    tmp_path: Path,
) -> None:
    root, identity = _package(tmp_path)

    config, _, day, observed = _validate_route_package(
        root,
        expected_manifest_sha256=identity,
        day_index=3,
    )

    assert observed == identity
    assert config.name == "strict_2023-08-24.sumocfg"
    assert day["strict_seed"] == 5204
    with pytest.raises(ValueError, match="manifest identity changed"):
        _validate_route_package(root, expected_manifest_sha256="0" * 64, day_index=3)


def test_validate_route_package_rejects_route_content_change(tmp_path: Path) -> None:
    root, identity = _package(tmp_path)
    route = root / "routes_2023-08-21.rou.xml.gz"
    route.write_bytes(b"changed")

    with pytest.raises(ValueError, match="daily route file changed"):
        _validate_route_package(root, expected_manifest_sha256=identity, day_index=0)


def test_final_summary_and_checks_require_exact_clean_completion(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.xml"
    summary_path.write_text(
        "<summary><step time='90000' loaded='4' inserted='4' running='0' "
        "waiting='0' ended='4' arrived='4' collisions='0' teleports='0'/>"
        "</summary>",
        encoding="utf-8",
    )
    summary = _final_summary_step(summary_path)
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
        "controllable_tls_count": EXPECTED_TLS_COUNT,
    }

    assert all(
        _admission_checks(
            expected_vehicle_count=4,
            observed=observed,
            summary=summary,
            error=None,
        ).values()
    )
    summary["collisions"] = 1
    assert not _admission_checks(
        expected_vehicle_count=4,
        observed=observed,
        summary=summary,
        error=None,
    )["zero_collisions"]


def test_strict_config_fixture_is_well_formed(tmp_path: Path) -> None:
    root, _ = _package(tmp_path)
    assert ET.parse(root / "strict_2023-08-21.sumocfg").getroot().tag == "configuration"
