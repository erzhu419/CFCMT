from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.package_dlr_badhersfeld_sumo import (
    PASSIVE_ADDITIONAL_FILE,
    SOURCE_ADDITIONAL_FILES,
    _dynamic_routing_audit,
    _route_audit,
    _write_canonical_config,
    canonical_config_inputs,
)


def _source_config(*, additional_files: tuple[str, ...] = SOURCE_ADDITIONAL_FILES) -> str:
    return f"""<?xml version="1.0"?>
<configuration>
  <input>
    <net-file value="../osm/osm_edited.net.xml.gz"/>
    <route-files value="../demand/osm_activitygen_present_no_prt.rou.xml.gz"/>
    <additional-files value="{','.join(additional_files)}"/>
  </input>
  <time><begin value="0"/><step-length value="1"/><end value="86400"/></time>
  <processing><ignore-route-errors value="true"/></processing>
  <routing><device.rerouting.probability value="1"/></routing>
  <taxi_device>
    <device.taxi.dispatch-algorithm value="greedy"/>
    <device.taxi.dispatch-algorithm.output value="dispatch.xml"/>
    <device.taxi.idle-algorithm.output value="idle.xml"/>
  </taxi_device>
  <gui_only><gui-settings-file value="view.xml"/></gui_only>
</configuration>"""


def test_canonical_config_preserves_all_active_inputs_and_source_options(
    tmp_path: Path,
) -> None:
    source = tmp_path / "present.sumocfg"
    destination = tmp_path / "cfcmt_present.sumocfg"
    source.write_text(_source_config(), encoding="utf-8")

    inputs = canonical_config_inputs(source_config=source, scenario="present")
    removed = _write_canonical_config(
        source_config=source,
        destination=destination,
        inputs=inputs,
    )

    root = ET.parse(destination).getroot()
    retained = tuple(
        root.find(".//additional-files").attrib["value"].split(",")
    )
    assert retained == tuple(
        value for value in SOURCE_ADDITIONAL_FILES if value != PASSIVE_ADDITIONAL_FILE
    )
    assert "../osm/gtfs_publictransport.rou.xml" in retained
    assert "../osm/osm_parking_rerouters.add.xml" in retained
    assert "../osm/calib.add.xml" in retained
    assert root.find(".//ignore-route-errors").attrib["value"] == "true"
    assert root.find(".//device.rerouting.probability").attrib["value"] == "1"
    assert root.find(".//device.taxi.dispatch-algorithm") is not None
    assert root.find(".//gui_only") is None
    assert set(removed) == {
        "gui_only",
        "device.taxi.dispatch-algorithm.output",
        "device.taxi.idle-algorithm.output",
    }


def test_canonical_config_rejects_additional_input_reordering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "present.sumocfg"
    reordered = list(SOURCE_ADDITIONAL_FILES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    source.write_text(
        _source_config(additional_files=tuple(reordered)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="additional input order changed"):
        canonical_config_inputs(source_config=source, scenario="present")


def _write_dynamic_skeleton_case(tmp_path: Path, *, middle_edge: str) -> Path:
    (tmp_path / "net.xml").write_text(
        """<net>
  <edge id="a" from="n0" to="n1"/><edge id="-a" from="n1" to="n0"/>
  <edge id="-902659463" from="n2" to="n3"/>
  <edge id="other" from="n4" to="n5"/>
</net>""",
        encoding="utf-8",
    )
    vehicles = "".join(
        f'<vehicle id="lkw_{index}" depart="{index}">'
        f'<route edges="a {middle_edge} -a"/></vehicle>'
        for index in range(1, 30)
    )
    (tmp_path / "osm_activitygen.lkw.rou.xml").write_text(
        f"<routes>{vehicles}</routes>", encoding="utf-8"
    )
    config = tmp_path / "run.sumocfg"
    config.write_text(
        """<configuration><input><net-file value="net.xml"/>
  <route-files value="osm_activitygen.lkw.rou.xml"/></input>
  <processing><ignore-route-errors value="true"/></processing>
  <routing><device.rerouting.probability value="1"/>
    <device.rerouting.period value="300"/>
    <device.rerouting.pre-period value="300"/></routing>
</configuration>""",
        encoding="utf-8",
    )
    return config


def test_route_audit_classifies_only_published_lkw_reroute_skeletons(
    tmp_path: Path,
) -> None:
    config = _write_dynamic_skeleton_case(
        tmp_path, middle_edge="-902659463"
    )

    audit = _route_audit(config)

    assert audit["disconnected_route_count"] == 29
    assert audit["source_declared_dynamic_reroute_skeleton_count"] == 29
    assert audit["unclassified_disconnected_route_count"] == 0
    assert _dynamic_routing_audit(config)["rerouting_probability"] == "1"


def test_route_audit_rejects_unclassified_disconnected_lkw_routes(
    tmp_path: Path,
) -> None:
    config = _write_dynamic_skeleton_case(tmp_path, middle_edge="other")

    with pytest.raises(ValueError, match="unclassified"):
        _route_audit(config)
