import gzip
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.data.package_dlr_brunswick_sumo import (
    BUILD_DEPENDENCY_PROTOCOL,
    PYPROJ_VERSION,
    PYPROJ_WHEEL_SHA256,
    _activate_build_dependency,
    _apply_sumo122_connection_compatibility,
    _gtfs_vehicle_inventory,
    _mapped_trip_inventory,
    _road_trip_inventory,
    _tls_inventory,
    _write_full_config,
)


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_brunswick_gzip_demand_and_network_audits(tmp_path: Path) -> None:
    geotrips = tmp_path / "geotrips.xml.gz"
    _write_gzip(
        geotrips,
        """<routes>
        <trip id="a" type="car" depart="10" fromLonLat="1,2" toLonLat="3,4"/>
        <trip id="b" type="bus" depart="20" fromLonLat="1,2" toLonLat="3,4"/>
        </routes>""",
    )
    source = _road_trip_inventory(geotrips)
    assert source["trip_count"] == 2
    assert source["minimum_depart_sec"] == 10.0
    assert source["maximum_depart_sec"] == 20.0
    assert source["vehicle_types"] == {"bus": 1, "car": 1}

    net = tmp_path / "net.xml.gz"
    _write_gzip(
        net,
        """<net>
        <edge id="e0" from="j0" to="j1"><lane id="e0_0" index="0"/></edge>
        <edge id="e1" from="j1" to="j2"><lane id="e1_0" index="0"/></edge>
        <junction id="j0"/><junction id="j1"/><junction id="j2"/>
        <tlLogic id="j1" programID="0"><phase duration="30" state="G"/></tlLogic>
        </net>""",
    )
    mapped_path = tmp_path / "mapped.xml"
    mapped_path.write_text(
        """<routes>
        <trip id="a" depart="10" from="e0" to="e1"/>
        <trip id="b" depart="20" fromJunction="j0" toJunction="j2"/>
        </routes>""",
        encoding="utf-8",
    )
    mapped = _mapped_trip_inventory(mapped_path, net)
    assert mapped["trip_count"] == 2
    assert mapped["anchor_modes"] == {"edge": 1, "junction": 1}
    assert mapped["missing_anchor_count"] == 0
    assert _tls_inventory(net) == {
        "controlled_intersection_count": 1,
        "program_count": 1,
        "phase_count": 1,
    }


def test_brunswick_gtfs_and_full_config_audits(tmp_path: Path) -> None:
    vehicles = tmp_path / "gtfs.add.xml"
    vehicles.write_text(
        """<additional>
        <vehicle id="pt0" depart="100"><route edges="e0 e1"/></vehicle>
        <vehicle id="pt1" depart="1:19:46:50"><route edges="e0 e1"/></vehicle>
        </additional>""",
        encoding="utf-8",
    )
    assert _gtfs_vehicle_inventory(vehicles) == {
        "vehicle_count": 2,
        "unique_id_count": 2,
        "duplicate_id_count": 0,
        "minimum_depart_sec": 100.0,
        "maximum_depart_sec": 157_610.0,
    }

    source = tmp_path / "source.sumocfg"
    source.write_text(
        """<configuration>
        <input><net-file value="net.xml"/></input>
        <output><tripinfo-output value="trip.xml"/></output>
        <time><begin value="99000"/></time>
        <report><log value="run.log"/></report>
        </configuration>""",
        encoding="utf-8",
    )
    destination = tmp_path / "full.sumocfg"
    assert _write_full_config(
        source=source, destination=destination, begin_sec=74670.0
    ) == ["output", "log"]
    root = ET.parse(destination).getroot()
    assert root.find(".//begin").attrib["value"] == "74670.00"
    assert root.find("output") is None
    assert root.find(".//log") is None


def test_brunswick_sumo122_connection_compatibility_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "patch.con.xml"
    path.write_text(
        """<connections>
        <connection from="431412124" fromLane="0" to="299910664" toLane="1"/>
        <connection from="431412124" fromLane="1" to="299910664" toLane="2"/>
        <connection from="other" fromLane="0" to="kept" toLane="0"/>
        </connections>""",
        encoding="utf-8",
    )
    audit = _apply_sumo122_connection_compatibility(path)
    assert audit["replacement_count"] == 2
    assert audit["sumo122_target_edge"] == "299910664#0"
    root = ET.parse(path).getroot()
    assert [
        connection.attrib["to"] for connection in root.findall("connection")
    ] == ["299910664#0", "299910664#0", "kept"]


def test_brunswick_build_dependency_contract(tmp_path: Path, monkeypatch) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (tmp_path / "dependency_manifest.json").write_text(
        json.dumps(
            {
                "protocol": BUILD_DEPENDENCY_PROTOCOL,
                "package": "pyproj",
                "version": PYPROJ_VERSION,
                "wheel": "pyproj.whl",
                "wheel_sha256": PYPROJ_WHEEL_SHA256,
                "import_verified": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", "prior")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="3.7.1\n", stderr=""
        ),
    )
    audit = _activate_build_dependency(tmp_path)
    assert audit["version"] == "3.7.1"
    assert audit["import_verified_for_build"] is True
