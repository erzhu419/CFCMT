import json
from pathlib import Path

from scripts.data import acquire_figshare_xuancheng as acquisition
from scripts.data import audit_figshare_xuancheng as audit


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "xuancheng.net.xml").write_text(
        '<net version="1.16">'
        '<edge id="a" from="i0" to="i1"><lane id="a_0" index="0"/></edge>'
        '<edge id="b" from="i1" to="i2"><lane id="b_0" index="0"/></edge>'
        '<junction id="i0" type="dead_end"/>'
        '<junction id="i1" type="traffic_light"/>'
        '<junction id="i2" type="dead_end"/>'
        '<connection from="a" to="b" fromLane="0" toLane="0" tl="i1"/>'
        '<tlLogic id="i1" programID="0"><phase duration="30" state="G"/>'
        '<phase duration="3" state="y"/></tlLogic>'
        '</net>\n',
        encoding="utf-8",
    )
    roadnet = {
        "roads": [{"id": "a"}, {"id": "b"}],
        "intersections": [
            {
                "id": "i1",
                "virtual": False,
                "roadLinks": [{"startRoad": "a", "endRoad": "b"}],
                "trafficLight": {"lightphases": [{}, {}]},
            }
        ],
    }
    (root / "roadnet_xuancheng250319.json").write_text(
        json.dumps(roadnet), encoding="utf-8"
    )
    flow = [
        {
            "vehicle": {"length": 5},
            "interval": 86_399,
            "startTime": 0,
            "endTime": 86_399,
            "route": ["a", "b"],
        }
    ]
    (root / acquisition.CANDIDATE_DAY).write_text(
        json.dumps(flow), encoding="utf-8"
    )
    (root / "config_xuancheng_test.json").write_text("{}\n", encoding="utf-8")
    files = {}
    for path in root.iterdir():
        if path.name == "acquisition_manifest.json":
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "md5": "test",
        }
    manifest = {
        "protocol": acquisition.PROTOCOL,
        "article_id": acquisition.ARTICLE_ID,
        "article_version": acquisition.ARTICLE_VERSION,
        "coverage": "candidate_screen",
        "candidate_day": acquisition.CANDIDATE_DAY,
        "files": files,
    }
    (root / "acquisition_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_candidate_audit_accepts_exact_native_topology(tmp_path: Path) -> None:
    payload = audit.audit_candidate(_candidate(tmp_path))

    assert payload["status"] == "FAIL"
    assert payload["checks"]["road_id_exact_match"] is True
    assert payload["checks"]["all_flow_route_pairs_connected"] is True
    assert payload["network"]["sumo_tls_count"] == 1
    assert payload["demand"]["expanded_vehicle_count"] == 2
    assert payload["checks"]["full_day_time_coverage"] is False


def test_candidate_audit_reports_disconnected_route(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    net_path = root / "xuancheng.net.xml"
    net_path.write_text(
        net_path.read_text(encoding="utf-8").replace(
            '<connection from="a" to="b" fromLane="0" toLane="0" tl="i1"/>',
            "",
        ),
        encoding="utf-8",
    )
    manifest_path = root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["xuancheng.net.xml"]["size_bytes"] = net_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = audit.audit_candidate(root)

    assert payload["status"] == "FAIL"
    assert payload["checks"]["all_flow_route_pairs_connected"] is False
    assert payload["demand"]["missing_route_connection_pair_count"] == 1
