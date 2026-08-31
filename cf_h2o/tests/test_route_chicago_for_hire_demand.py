from __future__ import annotations

import gzip
from pathlib import Path
import xml.etree.ElementTree as ET

import scripts.data.route_chicago_for_hire_demand as route_module
from scripts.data.route_chicago_for_hire_demand import (
    ROUTING_ALGORITHM,
    ROUTING_SEED,
    ROUTING_THREADS,
    _split_and_audit,
    departure_matches_input,
    duarouter_command,
    parse_day_source,
    route_matches_endpoints,
    strict_config_tree,
)


def test_duarouter_command_is_parallel_ch_without_unsupported_bulk_or_repair() -> None:
    command = duarouter_command(
        duarouter=Path("/sumo/duarouter"),
        network_file=Path("/data/chicago.net.xml.gz"),
        trip_files=[Path("/data/day0.xml.gz"), Path("/data/day1.xml.gz")],
        output_file=Path("/out/routes.xml.gz"),
    )

    assert command[command.index("--routing-algorithm") + 1] == ROUTING_ALGORITHM
    assert "--bulk-routing" not in command
    assert command[command.index("--routing-threads") + 1] == str(ROUTING_THREADS)
    assert command[command.index("--seed") + 1] == str(ROUTING_SEED)
    assert "--repair" not in command
    assert "--ignore-errors" not in command


def test_route_endpoint_and_identity_audits_are_exact() -> None:
    assert route_matches_endpoints(["a", "x", "b"], "a", "b")
    assert not route_matches_endpoints(["x", "b"], "a", "b")
    assert not route_matches_endpoints([], "a", "b")
    assert parse_day_source("d6_tnp_g0000001_r00000") == (6, "tnp")
    assert departure_matches_input(964.29, 964.285714)
    assert not departure_matches_input(964.30, 964.285714)


def test_strict_config_disables_loss_and_teleportation() -> None:
    tree = strict_config_tree(
        network_file=Path("/network/chicago.net.xml.gz"),
        route_file_name="routes_2023-08-21.rou.xml.gz",
        seed=5201,
    )
    root = tree.getroot()
    values = {
        element.tag: element.attrib.get("value") for element in root.iter()
    }

    assert values["route-files"] == "routes_2023-08-21.rou.xml.gz"
    assert values["scale"] == "1"
    assert values["max-depart-delay"] == "-1"
    assert values["time-to-teleport"] == "-1"
    assert values["collision.action"] == "warn"
    assert values["collision.check-junctions"] == "true"
    assert values["ignore-route-errors"] == "false"
    assert values["seed"] == "5201"


def test_split_audit_preserves_route_and_restores_local_departure(
    tmp_path: Path, monkeypatch
) -> None:
    combined = tmp_path / "combined.rou.xml.gz"
    with gzip.open(combined, "wt", encoding="utf-8") as handle:
        handle.write(
            '<?xml version="1.0"?><routes>'
            '<vehicle id="d0_tnp_g0000001_r00000" depart="12.5">'
            '<route edges="a x b"/></vehicle>'
            '<vehicle id="d1_taxi_g0000002_r00000" depart="86420">'
            '<route edges="c d"/></vehicle>'
            "</routes>"
        )
    monkeypatch.setattr(route_module, "EXPECTED_INCLUDED_TRIPS", 2)
    input_endpoints = {
        "d0_tnp_g0000001_r00000": ("a", "b", 12.504545, 0, "tnp"),
        "d1_taxi_g0000002_r00000": ("c", "d", 86_420.0, 1, "taxi"),
    }

    rows = _split_and_audit(
        combined_path=combined,
        input_endpoints=input_endpoints,
        staging=tmp_path,
    )

    assert rows[0]["vehicle_count"] == 1
    assert abs(rows[0]["maximum_departure_delta_sec"] - 0.004545) < 1e-12
    assert rows[1]["vehicle_count"] == 1
    with gzip.open(tmp_path / rows[1]["route_file"], "rb") as handle:
        vehicle = ET.parse(handle).getroot().find("vehicle")
    assert vehicle is not None
    assert vehicle.attrib["depart"] == "20.000000"
    assert vehicle.find("route").attrib["edges"] == "c d"
