from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.package_eth_lisbon_sumo import (
    _native_config_inventory,
    _network_inventory,
    _route_inventory,
    _taz_inventory,
    _write_microscopic_additional,
    _write_microscopic_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_static_inventories_preserve_trip_and_signal_identity(tmp_path: Path) -> None:
    network_path = _write(
        tmp_path / "net.xml",
        """<net>
        <edge id=":j_0"><lane id=":j_0_0"/></edge>
        <edge id="a"><lane id="a_0"/></edge>
        <edge id="b"><lane id="b_0"/></edge>
        <junction id="j"/>
        <connection from="a" to="b"/>
        <tlLogic id="j"><phase duration="30" state="G"/></tlLogic>
        </net>""",
    )
    taz_path = _write(
        tmp_path / "taz.xml",
        """<additional><taz id="z1"><tazSource id="a"/></taz>
        <taz id="z2"><tazSink id="b"/></taz></additional>""",
    )
    route_path = _write(
        tmp_path / "routes.xml",
        """<routes><trip id="x" depart="1" from="a" to="b"
        fromTaz="z1" trip_toTaz="z2"/></routes>""",
    )

    network, edges = _network_inventory(network_path)
    taz, taz_ids = _taz_inventory(taz_path)
    demand = _route_inventory(route_path, edge_ids=edges, taz_ids=taz_ids)

    assert network["noninternal_edge_count"] == 2
    assert network["traffic_light_count"] == 1
    assert network["traffic_light_phase_count"] == 1
    assert taz["taz_count"] == 2
    assert demand["trip_count"] == 1
    assert demand["missing_edge_anchor_count"] == 0
    assert demand["missing_taz_anchor_count"] == 0


def test_route_inventory_reports_missing_source_anchor(tmp_path: Path) -> None:
    route_path = _write(
        tmp_path / "routes.xml",
        """<routes><trip id="x" depart="1" from="missing" to="b"
        fromTaz="z1" trip_toTaz="z2"/></routes>""",
    )

    demand = _route_inventory(
        route_path,
        edge_ids={"a", "b"},
        taz_ids={"z1", "z2"},
    )

    assert demand["missing_edge_anchor_count"] == 1


def test_microscopic_config_is_strict_and_keeps_passive_vtype(tmp_path: Path) -> None:
    additional = _write(
        tmp_path / "additional.xml",
        """<additional><vType id="DEFAULT_VEHTYPE"/>
        <edgeData id="dump" file="large.xml"/></additional>""",
    )
    micro_additional = tmp_path / "micro.add.xml"
    audit = _write_microscopic_additional(additional, micro_additional)
    config = tmp_path / "micro.sumocfg"
    _write_microscopic_config(
        destination=config,
        begin_sec=1.2,
        end_sec=90001.1,
        seed=5057,
    )

    additional_root = ET.parse(micro_additional).getroot()
    assert additional_root.find("vType") is not None
    assert additional_root.find("edgeData") is None
    assert audit["demand_or_network_elements_removed"] == 0
    root = ET.parse(config).getroot()
    assert root.find("./processing/ignore-route-errors").attrib["value"] == "false"
    assert root.find("./processing/time-to-teleport").attrib["value"] == "-1"
    assert root.find("./processing/collision.check-junctions").attrib["value"] == "true"
    assert root.find("./random_number/seed").attrib["value"] == "5057"


def test_native_config_inventory_exposes_claim_boundary(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "meso.sumocfg",
        """<configuration>
        <input><net-file value="n"/><route-files value="r"/>
        <additional-files value="a"/></input>
        <time><begin value="1"/><end value="2"/><step-length value="2"/></time>
        <processing><ignore-route-errors value="true"/>
        <collision.action value="none"/><time-to-teleport value="360"/>
        <max-depart-delay value="1200"/><random-depart-offset value="1200"/>
        </processing>
        <routing><device.rerouting.probability value="0.5"/>
        <device.rerouting.period value="360"/></routing>
        <mesoscopic><mesosim value="true"/></mesoscopic>
        <random_number><seed value="2136"/></random_number>
        </configuration>""",
    )

    inventory = _native_config_inventory(config)

    assert inventory["mesosim"] == "true"
    assert inventory["collision_action"] == "none"
    assert inventory["time_to_teleport_sec"] == "360"
