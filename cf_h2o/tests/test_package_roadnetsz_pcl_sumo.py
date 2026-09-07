from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data import package_roadnetsz_pcl_sumo as package


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_acquisition(root: Path) -> None:
    net = root / package.NET_RELATIVE
    net.parent.mkdir(parents=True, exist_ok=True)
    net.write_text(
        """<net version="1.0">
<edge id="a" from="n0" to="n1"/><edge id="b" from="n1" to="n2"/>
<tlLogic id="j" type="static" programID="0"><phase duration="30" state="G"/></tlLogic>
</net>
""",
        encoding="utf-8",
    )
    trips = root / package.TRIPS_RELATIVE
    trips.write_text(
        "<routes>"
        + "".join(
            f'<trip id="{index}" depart="{index * 3600}" from="a" to="b"/>'
            for index in range(28)
        )
        + "</routes>\n",
        encoding="utf-8",
    )
    component_xml = {
        package.NODE_RELATIVE: (
            '<nodes><node id="j" type="traffic_light"/></nodes>\n'
        ),
        package.EDGE_RELATIVE: "<edges/>\n",
        package.CONNECTION_RELATIVE: "<connections/>\n",
        package.TLS_RELATIVE: (
            '<additional><tlLogic id="j" type="static" programID="0">'
            '<phase duration="30" state="G"/></tlLogic></additional>\n'
        ),
        package.TYPE_RELATIVE: "<types/>\n",
    }
    for relative, content in component_xml.items():
        path = root / relative
        path.write_text(content, encoding="utf-8")
    files = {}
    for relative in (
        package.NET_RELATIVE,
        package.NODE_RELATIVE,
        package.EDGE_RELATIVE,
        package.CONNECTION_RELATIVE,
        package.TLS_RELATIVE,
        package.TYPE_RELATIVE,
        package.TRIPS_RELATIVE,
    ):
        path = root / relative
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (root / "acquisition_manifest.json").write_text(
        json.dumps(
            {
                "protocol": package.ACQUISITION_PROTOCOL,
                "commit": package.COMMIT,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_pcl_package_routes_three_fixed_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquisition"
    _write_acquisition(acquisition)
    duarouter = tmp_path / "duarouter"
    duarouter.write_text("runtime\n", encoding="utf-8")
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("runtime\n", encoding="utf-8")

    def fake_netconvert(
        *,
        netconvert,
        node_path,
        edge_path,
        connection_path,
        tls_path,
        type_path,
        net_path,
    ):
        del netconvert, node_path, edge_path, connection_path, tls_path, type_path
        net_path.write_text(
            """<net version="1.20">
<edge id="a" from="n0" to="n1"/><edge id="b" from="n1" to="n2"/>
<tlLogic id="j" type="static" programID="0"><phase duration="30" state="G"/></tlLogic>
</net>
""",
            encoding="utf-8",
        )
        return {"command": ["netconvert", str(net_path)], "returncode": 0}

    def fake_duarouter(*, duarouter, net_path, trips_path, route_path):
        del duarouter, net_path
        source = ET.parse(trips_path).getroot()
        root = ET.Element("routes")
        for trip in source:
            vehicle = ET.SubElement(
                root,
                "vehicle",
                id=trip.attrib["id"],
                depart=trip.attrib["depart"],
            )
            ET.SubElement(vehicle, "route", edges="a b")
        ET.ElementTree(root).write(route_path, encoding="utf-8")
        return {"command": ["duarouter"], "returncode": 0}

    monkeypatch.setattr(package, "_run_netconvert", fake_netconvert)
    monkeypatch.setattr(package, "_run_duarouter", fake_duarouter)
    output = tmp_path / "package"
    payload = package.package_pcl(
        acquisition_root=acquisition,
        output_root=output,
        netconvert=netconvert,
        duarouter=duarouter,
    )

    assert payload["protocol"] == package.PROTOCOL
    assert list(payload["networks"]) == [row[0] for row in package.WINDOWS]
    assert all(row["vehicle_count"] == 1 for row in payload["networks"].values())
    assert all(
        row["source_window"]["routing_retention"] == 1.0
        for row in payload["networks"].values()
    )
    assert all(
        (output / row["sumocfg"]).is_file()
        for row in payload["networks"].values()
    )
    assert ".staging-" not in json.dumps(payload, sort_keys=True)
    assert payload["selection"]["all_source_trips_covered"] is True
    assert payload["selection"]["source_trip_count"] == 28
    tls = next(iter(payload["networks"].values()))["tls_semantic_audit"]
    assert tls["declared_tls_node_count"] == 1
    assert tls["rebuilt_tls_id_count"] == 1
    assert tls["explicit_phase_count_mismatch_count"] == 0
    assert tls["auto_generated_minor_green_signal_count"] == 0
    build = next(iter(payload["networks"].values()))["network_build"]
    assert build["auto_generated_tls_phase_layout"] == "incoming"
