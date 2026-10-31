from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.package_zenodo_paderborn_sumo import (
    _external_tls_inventory,
    _network_inventory,
    _route_inventory,
    _trip_inventory,
    _vtype_inventory,
    _write_strict_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _network(tmp_path: Path):
    path = _write(
        tmp_path / "network.xml",
        """<net>
        <edge id="a"><lane id="a_0"/></edge>
        <edge id="b"><lane id="b_0"/></edge>
        <junction id="j"/><connection from="a" to="b"/>
        </net>""",
    )
    return _network_inventory(path)


def test_trip_and_route_inventories_preserve_identity(tmp_path: Path) -> None:
    _, edges, lanes, _ = _network(tmp_path)
    trips_path = _write(
        tmp_path / "trips.xml",
        """<routes><trip id="x" type="default" depart="0" from="a" to="b">
        <stop lane="a_0" duration="1"/></trip></routes>""",
    )
    routes_path = _write(
        tmp_path / "routes.xml",
        """<routes><vehicle id="x" type="default" depart="0">
        <route edges="a b"/><stop lane="a_0" duration="1"/>
        </vehicle></routes>""",
    )

    trips, trip_ids = _trip_inventory(
        trips_path, edge_ids=edges, lane_ids=lanes
    )
    routes, route_ids = _route_inventory(
        routes_path, edge_ids=edges, lane_ids=lanes
    )

    assert trip_ids == route_ids == {"x"}
    assert trips["missing_edge_anchor_count"] == 0
    assert trips["missing_stop_lane_count"] == 0
    assert routes["missing_route_edge_reference_count"] == 0
    assert routes["route_edge_reference_count"] == 2


def test_route_inventory_reports_missing_edge(tmp_path: Path) -> None:
    _, edges, lanes, _ = _network(tmp_path)
    path = _write(
        tmp_path / "routes.xml",
        """<routes><vehicle id="x" depart="0">
        <route edges="a missing"/></vehicle></routes>""",
    )

    routes, _ = _route_inventory(path, edge_ids=edges, lane_ids=lanes)

    assert routes["missing_route_edge_reference_count"] == 1


def test_vtypes_tls_and_strict_config(tmp_path: Path) -> None:
    _, _, _, junctions = _network(tmp_path)
    vtypes_path = _write(
        tmp_path / "vtypes.xml",
        "<additional><vType id=\"default\"/><vType id=\"bus\"/></additional>",
    )
    tls_path = _write(
        tmp_path / "tls.xml",
        """<additional><tlLogic id="j" programID="p">
        <phase duration="30" state="G"/></tlLogic></additional>""",
    )

    vtypes, identifiers = _vtype_inventory(vtypes_path)
    tls = _external_tls_inventory(tls_path, junction_ids=junctions)
    config = tmp_path / "strict.sumocfg"
    _write_strict_config(
        destination=config,
        begin_sec=0,
        end_sec=108000,
        seed=5057,
    )

    assert identifiers == {"default", "bus"}
    assert vtypes["duplicate_vehicle_type_id_count"] == 0
    assert tls["controlled_intersection_count"] == 1
    assert tls["missing_network_junction_count"] == 0
    root = ET.parse(config).getroot()
    assert root.find("./processing/ignore-route-errors").attrib["value"] == "false"
    assert root.find("./processing/time-to-teleport").attrib["value"] == "-1"
    assert root.find("./processing/collision.check-junctions").attrib["value"] == "true"
    assert root.find("./random_number/seed").attrib["value"] == "5057"
