import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data import acquire_figshare_xuancheng as acquisition
from scripts.data import package_figshare_xuancheng_sumo as package


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _acquisition(tmp_path: Path) -> Path:
    root = tmp_path / "acquisition"
    root.mkdir()
    (root / "xuancheng.net.xml").write_text(
        '<net version="1.16">'
        '<edge id="r0" from="i0" to="i1"><lane id="r0_0" index="0" length="10"/></edge>'
        '<edge id="r1" from="i1" to="i2"><lane id="r1_0" index="0" length="10"/></edge>'
        '<edge id="r2" from="i2" to="i3"><lane id="r2_0" index="0" length="10"/></edge>'
        '<junction id="i0" type="dead_end"/>'
        '<junction id="i1" type="priority"/>'
        '<junction id="i2" type="traffic_light"/>'
        '<junction id="i3" type="dead_end"/>'
        '<connection from="r0" to="r1" fromLane="0" toLane="0"/>'
        '<connection from="r1" to="r2" fromLane="0" toLane="0" tl="i2"/>'
        '<tlLogic id="i2" programID="0"><phase duration="30" state="G"/></tlLogic>'
        '</net>\n',
        encoding="utf-8",
    )
    intersections = []
    for index in range(4):
        roads = []
        if index > 0:
            roads.append(f"r{index - 1}")
        if index < 3:
            roads.append(f"r{index}")
        intersections.append(
            {
                "id": f"i{index}",
                "virtual": True,
                "point": {"x": index * 10, "y": 0},
                "roads": roads,
                "roadLinks": (
                    [{"startRoad": roads[0], "endRoad": roads[1]}]
                    if len(roads) == 2
                    else []
                ),
                "trafficLight": {"lightphases": []},
            }
        )
    roadnet = {
        "intersections": intersections,
        "roads": [
            {
                "id": f"r{index}",
                "startIntersection": f"i{index}",
                "endIntersection": f"i{index + 1}",
                "points": [
                    {"x": index * 10, "y": 0},
                    {"x": (index + 1) * 10, "y": 0},
                ],
                "lanes": [{"width": 3.2, "maxSpeed": 10}],
            }
            for index in range(3)
        ],
    }
    (root / "roadnet_xuancheng250319.json").write_text(
        json.dumps(roadnet), encoding="utf-8"
    )
    vehicle = {
        "length": 5,
        "width": 1.8,
        "maxPosAcc": 2.6,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.6,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 27.78,
        "headwayTime": 2,
    }
    flow = [
        {
            "vehicle": vehicle,
            "interval": 1,
            "startTime": 0,
            "endTime": 0,
            "route": ["r0", "r2"],
        },
        {
            "vehicle": vehicle,
            "interval": 1,
            "startTime": 86_400,
            "endTime": 86_400,
            "route": ["r0", "r1", "r2"],
        },
    ]
    (root / acquisition.CANDIDATE_DAY).write_text(
        json.dumps(flow), encoding="utf-8"
    )
    (root / "config_xuancheng_test.json").write_text("{}\n", encoding="utf-8")
    files = {}
    for path in root.iterdir():
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "md5": _md5(path),
        }
    (root / "acquisition_manifest.json").write_text(
        json.dumps(
            {
                "protocol": acquisition.PROTOCOL,
                "article_id": acquisition.ARTICLE_ID,
                "article_version": acquisition.ARTICLE_VERSION,
                "coverage": "candidate_screen",
                "candidate_day": acquisition.CANDIDATE_DAY,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_package_completes_anchors_and_preserves_native_net(tmp_path: Path) -> None:
    source = _acquisition(tmp_path)
    output = tmp_path / "package"

    payload = package.package_xuancheng(
        acquisition_root=source,
        output_root=output,
    )

    scenario = payload["networks"]["xuancheng_20230403"]
    assert scenario["vehicle_count"] == 2
    assert scenario["demand_audit"]["all_anchor_edges_preserved_in_order"] is True
    assert scenario["demand_audit"]["unique_inserted_edge_count"] == 1
    assert payload["global_completed_anchor_pair_count"] == 1
    assert (output / "xuancheng.net.xml").read_bytes() == (
        source / "xuancheng.net.xml"
    ).read_bytes()
    route_root = ET.parse(
        output / "xuancheng_20230403/xuancheng_20230403.rou.xml"
    ).getroot()
    assert len(route_root.findall("vehicle")) == 2
    assert {row.attrib["edges"] for row in route_root.findall("route")} == {
        "r0 r1 r2"
    }
    config = ET.parse(
        output / "xuancheng_20230403/xuancheng_20230403.sumocfg"
    ).getroot()
    assert config.find(".//end").attrib["value"] == "86401"
