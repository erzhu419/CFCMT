from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.package_eth_city_sumo import (
    _network_compatibility_audit,
    _route_inventory,
    _unsupported_zipper_junctions,
    _write_zipper_compatibility_repair,
    _write_microscopic_config,
    _write_microscopic_routes,
    expected_city_entries,
)


def test_expected_city_entries_are_complete_and_city_scoped() -> None:
    assert expected_city_entries("BOS") == (
        "BOS/",
        "BOS/additional.add.xml",
        "BOS/BOS.net.xml",
        "BOS/meso.sumo.cfg",
        "BOS/taz.xml",
        "BOS/trips24h_smoothed.rou.xml",
    )
    with pytest.raises(ValueError, match="unsupported ETH city code"):
        expected_city_entries("XXX")


def test_microscopic_config_is_structured_copy_with_strict_overrides(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meso.sumo.cfg"
    source.write_text(
        """<?xml version="1.0"?>
<configuration>
  <input>
    <net-file value="BOS.net.xml"/>
    <route-files value="trips24h_smoothed.rou.xml"/>
    <additional-files value="additional.add.xml,taz.xml"/>
  </input>
  <output>
    <output-prefix value="TIME"/>
    <tripinfo-output value="trips.xml"/>
    <vehroute-output value="routes.xml"/>
  </output>
  <time><begin value="14400"/><end value="49800"/><step-length value="2"/></time>
  <processing>
    <ignore-route-errors value="true"/>
    <collision.action value="none"/>
    <time-to-teleport value="360"/>
    <max-depart-delay value="1200"/>
    <random-depart-offset value="1200"/>
  </processing>
  <routing><device.rerouting.probability value="0.5"/></routing>
  <mesoscopic><mesosim value="true"/></mesoscopic>
  <random_number><seed value="2136"/></random_number>
</configuration>
""",
        encoding="utf-8",
    )
    destination = tmp_path / "microscopic.sumo.cfg"
    output_change = _write_microscopic_config(
        source=source,
        destination=destination,
        city_code="BOS",
        begin_sec=7201.78,
        end_sec=71995.74,
        seed=5057,
    )
    root = ET.parse(destination).getroot()

    def value(section: str, tag: str) -> str | None:
        element = root.find(f"./{section}/{tag}")
        return None if element is None else element.attrib.get("value")

    assert root.find("./mesoscopic") is None
    assert value("input", "net-file") == "source/BOS/BOS.net.xml"
    assert value("input", "route-files") == (
        "microscopic_trips.rou.xml"
    )
    assert root.find("./output") is None
    assert output_change["removed_output_option_count"] == 3
    assert output_change["demand_or_network_elements_removed"] == 0
    assert value("time", "begin") == "7201"
    assert value("time", "end") == "71996"
    assert value("time", "step-length") == "1"
    assert value("processing", "ignore-route-errors") == "false"
    assert value("processing", "collision.action") == "warn"
    assert value("processing", "collision.check-junctions") == "true"
    assert value("processing", "time-to-teleport") == "-1"
    assert value("processing", "max-depart-delay") == "-1"
    assert value("processing", "random-depart-offset") == "1200"
    assert value("routing", "device.rerouting.probability") == "0.5"
    assert value("random_number", "seed") == "5057"


def test_route_inventory_accepts_published_trip_to_taz_attribute(
    tmp_path: Path,
) -> None:
    route = tmp_path / "routes.xml"
    route.write_text(
        """<routes>
  <trip id="a" depart="1.5" from="e1" to="e2" fromTaz="z1" trip_toTaz="z2"/>
  <trip id="b" depart="2.5" from="e2" to="e1" fromTaz="z2" trip_toTaz="z1"/>
</routes>
""",
        encoding="utf-8",
    )
    inventory = _route_inventory(
        route,
        edge_ids={"e1", "e2"},
        taz_ids={"z1", "z2"},
    )
    assert inventory["trip_count"] == 2
    assert inventory["unique_trip_id_count"] == 2
    assert inventory["missing_edge_anchor_count"] == 0
    assert inventory["missing_taz_anchor_count"] == 0
    assert inventory["nonmonotonic_departure_count"] == 0
    assert inventory["custom_trip_to_taz_attribute_count"] == 2
    assert inventory["canonical_to_taz_attribute_count"] == 0
    assert inventory["conflicting_destination_taz_attribute_count"] == 0


def test_microscopic_routes_canonicalize_taz_schema_without_changing_trips(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.rou.xml"
    source.write_text(
        """<routes>
  <trip id="a" depart="1.5" from="e1" to="e2" fromTaz="z1" trip_toTaz="z2"/>
  <trip id="b" depart="2.5" from="e2" to="e1" fromTaz="z2" trip_toTaz="z1"/>
</routes>
""",
        encoding="utf-8",
    )
    destination = tmp_path / "microscopic.rou.xml"
    audit = _write_microscopic_routes(source, destination)
    trips = list(ET.parse(destination).getroot())
    assert audit["trip_count"] == 2
    assert audit["renamed_trip_to_taz_count"] == 2
    assert audit["trip_anchor_edges_changed"] == 0
    assert audit["trips_removed"] == 0
    assert audit["route_repair_applied"] is False
    assert [trip.attrib for trip in trips] == [
        {
            "id": "a",
            "depart": "1.5",
            "from": "e1",
            "to": "e2",
            "fromTaz": "z1",
            "toTaz": "z2",
        },
        {
            "id": "b",
            "depart": "2.5",
            "from": "e2",
            "to": "e1",
            "fromTaz": "z2",
            "toTaz": "z1",
        },
    ]


def test_microscopic_routes_reject_conflicting_destination_taz(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.rou.xml"
    source.write_text(
        '<routes><trip id="a" depart="1" from="e1" to="e2" '
        'fromTaz="z1" trip_toTaz="z2" toTaz="z3"/></routes>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting destination TAZ"):
        _write_microscopic_routes(source, tmp_path / "out.rou.xml")


def test_unsupported_zipper_detection_uses_sumo_response_foe_links(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <junction id="safe" type="zipper" incLanes="a_0 b_0">
    <request index="0" response="01" foes="01" cont="0"/>
    <request index="1" response="10" foes="10" cont="0"/>
  </junction>
  <junction id="unsupported" type="zipper" incLanes="c_0 d_0 e_0">
    <request index="0" response="011" foes="011" cont="0"/>
    <request index="1" response="100" foes="100" cont="0"/>
    <request index="2" response="100" foes="100" cont="0"/>
  </junction>
  <junction id="ordinary" type="priority" incLanes="f_0 g_0">
    <request index="0" response="01" foes="01" cont="0"/>
  </junction>
</net>
""",
        encoding="utf-8",
    )

    audit = _unsupported_zipper_junctions(network)

    assert audit["zipper_junction_count"] == 2
    assert audit["unsupported_candidate_count"] == 1
    assert audit["unsupported_candidates"] == [
        {
            "id": "unsupported",
            "incoming_lane_count": 3,
            "request_count": 3,
            "max_response_foe_count": 2,
            "violating_request_indices": [0],
        }
    ]


def test_structured_network_repair_allows_only_declared_type_change(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before.net.xml"
    after = tmp_path / "after.net.xml"

    def network(
        junction_type: str,
        *,
        destination: str = "out",
        connection_state: str = "Z",
    ) -> str:
        return f"""<net>
  <edge id="in" from="origin" to="merge" priority="1">
    <lane id="in_0" index="0" speed="13.9" length="100"/>
  </edge>
  <edge id="{destination}" from="merge" to="sink" priority="1">
    <lane id="{destination}_0" index="0" speed="13.9" length="100"/>
  </edge>
  <tlLogic id="tls" type="static" programID="0" offset="0">
    <phase duration="30" state="G"/>
  </tlLogic>
  <junction id="merge" type="{junction_type}" incLanes="in_0">
    <request index="0" response="0" foes="0" cont="0"/>
  </junction>
  <connection from="in" to="{destination}" fromLane="0" toLane="0" state="{connection_state}"/>
</net>
"""

    before.write_text(network("zipper"), encoding="utf-8")
    repair, connection_changes = _write_zipper_compatibility_repair(
        before,
        after,
        junction_ids=["merge"],
    )
    assert repair == {
        "method": (
            "structured_xml_declared_junction_type_and_link_state_substitution"
        ),
        "changed_junction_count": 1,
        "changed_connection_state_count": 1,
        "connection_state_change_histogram": {"Z->M": 1},
    }
    audit = _network_compatibility_audit(
        before,
        after,
        repaired_junction_ids=["merge"],
        repaired_connection_states=connection_changes,
    )
    assert audit["passed"] is True

    after.write_text(
        network("priority", destination="changed", connection_state="M"),
        encoding="utf-8",
    )
    assert not _network_compatibility_audit(
        before,
        after,
        repaired_junction_ids=["merge"],
        repaired_connection_states=connection_changes,
    )["passed"]


def test_structured_network_repair_rewrites_multi_foe_zipper_states(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before.net.xml"
    after = tmp_path / "after.net.xml"
    before.write_text(
        """<net>
  <edge id="a" from="a0" to="merge"><lane id="a_0" index="0"/></edge>
  <edge id="b" from="b0" to="merge"><lane id="b_0" index="0"/></edge>
  <edge id="c" from="c0" to="merge"><lane id="c_0" index="0"/></edge>
  <edge id="out" from="merge" to="sink"><lane id="out_0" index="0"/></edge>
  <junction id="merge" type="zipper" incLanes="a_0 b_0 c_0">
    <request index="0" response="011" foes="011" cont="0"/>
    <request index="1" response="101" foes="101" cont="0"/>
    <request index="2" response="110" foes="110" cont="0"/>
  </junction>
  <connection from="a" to="out" fromLane="0" toLane="0" state="Z"/>
  <connection from="b" to="out" fromLane="0" toLane="0" state="Z"/>
  <connection from="c" to="out" fromLane="0" toLane="0" state="Z"/>
</net>
""",
        encoding="utf-8",
    )

    repair, connection_changes = _write_zipper_compatibility_repair(
        before,
        after,
        junction_ids=["merge"],
    )

    root = ET.parse(after).getroot()
    assert root.find("junction").attrib["type"] == "priority"
    assert [row.attrib["state"] for row in root.iter("connection")] == [
        "m",
        "m",
        "m",
    ]
    assert repair["connection_state_change_histogram"] == {"Z->m": 3}
    assert _network_compatibility_audit(
        before,
        after,
        repaired_junction_ids=["merge"],
        repaired_connection_states=connection_changes,
    )["passed"]
