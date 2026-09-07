import json
from pathlib import Path

import pytest

from scripts.data import convert_libsignal_external_to_sumo as converter
from scripts.data.convert_libsignal_external_to_sumo import (
    _la_connection_key,
    _manifest_path,
    _monotone_virtual_lane_pairs,
    _nanchang_connection_plan,
    _nanchang_explicit_departures,
    _parse_nanchang_flows,
    _parse_nanchang_roadnet,
    _remap_cityflow_tls_states,
    _validate_edge_routes,
    convert_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = (
    PROJECT_ROOT
    / "H2Oplus/downloads/traffic_signal_libsignal/repo/data/raw_data"
)


def test_cityflow_v10_remap_downgrades_netconvert_minor_conflict(
    tmp_path: Path,
) -> None:
    net_path = tmp_path / "network.net.xml"
    net_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <connection from="a" to="b" fromLane="0" toLane="0" tl="j" linkIndex="0" state="O"/>
  <connection from="c" to="d" fromLane="0" toLane="0" tl="j" linkIndex="1" state="o"/>
  <tlLogic id="j" type="static" programID="0" offset="0">
    <phase duration="20" state="GG"/>
  </tlLogic>
</net>
""",
        encoding="utf-8",
    )
    key_major = ("a", "b", 0, 0)
    key_minor = ("c", "d", 0, 0)
    audit = _remap_cityflow_tls_states(
        net_path,
        {
            "j": {
                "provisional_indices": {key_major: 0, key_minor: 1},
                "permissive_keys": set(),
                "phases": [
                    {
                        "duration": 20.0,
                        "active_keys": {key_major, key_minor},
                    }
                ],
                "append_intergreen": False,
            }
        },
        yield_to_netconvert_priority=True,
    )

    phase = converter.ET.parse(net_path).getroot().find("tlLogic/phase")
    assert phase is not None
    assert phase.attrib["state"] == "Gg"
    assert audit["protocol"].endswith("merge-conflict-yield-v4")
    assert audit["conflict_yield_phase_link_count"] == 1


def test_conversion_manifest_paths_are_relocation_safe(tmp_path: Path) -> None:
    artifact_root = tmp_path / ".full_networks.staging-123"
    generated = artifact_root / "la_1x4/la_1x4.sumocfg"
    external = tmp_path / "runtime/bin/netconvert"

    assert _manifest_path(generated, artifact_root=artifact_root) == (
        "la_1x4/la_1x4.sumocfg"
    )
    assert ".staging-" not in _manifest_path(
        generated,
        artifact_root=artifact_root,
    )
    assert _manifest_path(external, artifact_root=artifact_root) == str(
        external.resolve()
    )


def test_full_conversion_main_path_writes_relocation_safe_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_netconvert(
        netconvert: Path,
        *,
        node_file: Path,
        edge_file: Path,
        output_file: Path,
        connection_file: Path | None = None,
        tls_file: Path | None = None,
        artifact_root: Path | None = None,
    ) -> dict:
        if connection_file is None or tls_file is None:
            output_file.write_text("<net/>\n", encoding="utf-8")
        else:
            connections = converter.ET.parse(connection_file).getroot()
            tls = converter.ET.parse(tls_file).getroot()
            net = converter.ET.Element("net")
            controlled = [
                row for row in connections.findall("connection") if row.get("tl")
            ]
            counts: dict[str, int] = {}
            for row in controlled:
                counts[str(row.get("tl"))] = counts.get(str(row.get("tl")), 0) + 1
            for row in connections.findall("connection"):
                attributes = dict(row.attrib)
                if attributes.get("tl"):
                    attributes["linkIndex"] = str(
                        counts[attributes["tl"]]
                        - 1
                        - int(attributes["linkIndex"])
                    )
                converter.ET.SubElement(net, "connection", attributes)
            for logic in tls.findall("tlLogic"):
                net.append(converter.ET.fromstring(converter.ET.tostring(logic)))
            converter._write_xml(output_file, net)
        generated = [node_file, edge_file, output_file, connection_file, tls_file]
        return {
            "command": [
                str(netconvert),
                *[
                    _manifest_path(path, artifact_root=artifact_root)
                    for path in generated
                    if path is not None and artifact_root is not None
                ],
            ],
            "returncode": 0,
            "stdout_tail": "Success.\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(converter, "_run_netconvert", fake_netconvert)
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("test runtime\n", encoding="utf-8")
    output_root = tmp_path / "full_networks"
    payload = convert_all(
        la_roadnet=RAW_ROOT / "LA_1x4/roadnet_LA.json",
        la_flow=RAW_ROOT / "LA_1x4/LA.json",
        nanchang_roadnet=RAW_ROOT / "pb_nanchang/roadnet_nanchang.txt",
        nanchang_flow=RAW_ROOT / "pb_nanchang/flow_nanchang.txt",
        output_root=output_root,
        netconvert=netconvert,
    )

    manifest = json.loads(
        (output_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == payload
    assert ".staging-" not in json.dumps(manifest, sort_keys=True)
    assert manifest["networks"]["la_1x4"]["source_lane_link_count"] == 708
    assert manifest["networks"]["la_1x4"]["connection_count"] == 401
    assert manifest["networks"]["la_1x4"]["duplicate_lane_link_count"] == 70
    virtual_reduction = manifest["networks"]["la_1x4"][
        "virtual_lane_link_reduction"
    ]
    assert virtual_reduction == {
        "protocol": "cityflow-virtual-cartesian-lane-link-monotone-rank-v1",
        "reduced_road_link_count": 25,
        "source_unique_lane_pair_count": 333,
        "retained_unique_lane_pair_count": 96,
        "removed_unique_lane_pair_count": 237,
        "post_reduction_crossing_road_link_count": 0,
        "source_lane_coverage_failures": 0,
        "target_lane_coverage_failures": 0,
    }
    tls_audit = manifest["networks"]["la_1x4"]["tls_semantic_audit"]
    assert tls_audit["pre_remap_phase_link_symmetric_difference"] > 0
    assert tls_audit["post_remap_phase_link_symmetric_difference"] == 0
    assert (
        manifest["networks"]["nanchang_full"]["connection_audit"][
            "connection_count"
        ]
        == 13390
    )
    nanchang = manifest["networks"]["nanchang_full"]
    assert nanchang["demand_encoding"] == (
        "explicit_globally_departure_sorted_vehicles"
    )
    assert nanchang["vehicle_count"] == 126669
    assert nanchang["source_flow_begin_nondecreasing"] is False
    nanchang_routes = output_root / "nanchang_full/nanchang_full.rou.xml"
    departures = [
        float(element.attrib["depart"])
        for _, element in converter.ET.iterparse(nanchang_routes, events=("end",))
        if element.tag == "vehicle"
    ]
    assert len(departures) == 126669
    assert departures == sorted(departures)
    for network in manifest["networks"].values():
        assert (output_root / network["sumocfg"]).is_file()


def test_full_nanchang_parser_preserves_topology_and_demand() -> None:
    roadnet = _parse_nanchang_roadnet(
        RAW_ROOT / "pb_nanchang/roadnet_nanchang.txt"
    )
    flows = _parse_nanchang_flows(
        RAW_ROOT / "pb_nanchang/flow_nanchang.txt"
    )
    endpoints = {
        edge_id: (edge["from"], edge["to"])
        for edge_id, edge in roadnet["edges"].items()
    }
    audit = _validate_edge_routes(endpoints, (flow["route"] for flow in flows))
    estimated_vehicles = sum(
        int((flow["end"] - flow["begin"]) // flow["period"]) + 1
        for flow in flows
    )

    assert len(roadnet["intersections"]) == 2048
    assert roadnet["undirected_road_count"] == 3012
    assert len(roadnet["edges"]) == 6024
    assert len(roadnet["signal_rows"]) == 859
    assert len(flows) == 9786
    assert estimated_vehicles == 126669
    assert audit["missing_edge_route_count"] == 0
    assert audit["disconnected_route_count"] == 0


def test_full_nanchang_departures_preserve_inclusive_demand_and_sorting() -> None:
    flows = _parse_nanchang_flows(
        RAW_ROOT / "pb_nanchang/flow_nanchang.txt"
    )
    departures = _nanchang_explicit_departures(flows)

    assert len(departures) == 126669
    assert departures == sorted(departures)
    assert departures[0][0] >= 0.0
    assert departures[-1][0] <= max(float(flow["end"]) for flow in flows)


def test_full_nanchang_connection_plan_preserves_lane_turns_and_routes() -> None:
    roadnet = _parse_nanchang_roadnet(
        RAW_ROOT / "pb_nanchang/roadnet_nanchang.txt"
    )
    flows = _parse_nanchang_flows(
        RAW_ROOT / "pb_nanchang/flow_nanchang.txt"
    )
    connections, audit = _nanchang_connection_plan(
        roadnet=roadnet,
        flows=flows,
    )

    assert len(connections) == 13390
    assert audit["connection_count"] == 13390
    assert audit["controlled_connection_count"] > 0
    assert audit["movement_connection_counts"] == {
        "left": 4091,
        "straight": 5208,
        "right": 4091,
    }
    assert audit["route_transition_count"] == 186350
    assert audit["missing_route_edge_pair_count"] == 0
    assert len(
        {
            (
                row["from"],
                row["to"],
                row["fromLane"],
                row["toLane"],
            )
            for row in connections
        }
    ) == len(connections)


def test_full_la_routes_are_topologically_contiguous() -> None:
    roadnet = json.loads(
        (RAW_ROOT / "LA_1x4/roadnet_LA.json").read_text(encoding="utf-8")
    )
    flows = json.loads(
        (RAW_ROOT / "LA_1x4/LA.json").read_text(encoding="utf-8")
    )
    endpoints = {
        str(road["id"]): (
            str(road["startIntersection"]),
            str(road["endIntersection"]),
        )
        for road in roadnet["roads"]
    }
    audit = _validate_edge_routes(
        endpoints,
        ([str(edge) for edge in flow["route"]] for flow in flows),
    )

    assert len(roadnet["intersections"]) == 30
    assert len(roadnet["roads"]) == 52
    assert len(flows) == 2203
    assert audit["missing_edge_route_count"] == 0
    assert audit["disconnected_route_count"] == 0


def test_full_la_lane_links_have_exact_duplicate_connections() -> None:
    roadnet = json.loads(
        (RAW_ROOT / "LA_1x4/roadnet_LA.json").read_text(encoding="utf-8")
    )
    roads = {str(road["id"]): road for road in roadnet["roads"]}
    source_count = 0
    unique_count = 0
    duplicate_count = 0
    for intersection in roadnet["intersections"]:
        seen: dict[tuple[str, str, int, int], tuple[tuple[float, float], ...]] = {}
        for road_link in intersection.get("roadLinks", ()):
            for lane_link in road_link.get("laneLinks", ()):
                source_count += 1
                key = _la_connection_key(
                    road_link=road_link,
                    lane_link=lane_link,
                    road_by_id=roads,
                )
                geometry = tuple(
                    (float(point["x"]), float(point["y"]))
                    for point in lane_link.get("points", ())
                )
                if key in seen:
                    assert seen[key] == geometry
                    duplicate_count += 1
                else:
                    seen[key] = geometry
                    unique_count += 1

    assert source_count == 708
    assert unique_count == 638
    assert duplicate_count == 70


def test_virtual_cartesian_lane_links_reduce_to_covering_monotone_map() -> None:
    selected, reduced = _monotone_virtual_lane_pairs(
        {(source, target) for source in range(5) for target in range(6)}
    )

    assert reduced is True
    assert selected == {
        (0, 0),
        (1, 1),
        (2, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    }
    assert {source for source, _ in selected} == set(range(5))
    assert {target for _, target in selected} == set(range(6))


def test_virtual_noncartesian_crossing_lane_links_fail_closed() -> None:
    with pytest.raises(
        ValueError,
        match="crossing virtual lane links are not a Cartesian lane fan-out",
    ):
        _monotone_virtual_lane_pairs({(0, 0), (0, 1), (1, 0)})


def test_virtual_non_crossing_lane_links_are_preserved() -> None:
    source = {(0, 0), (1, 1), (2, 1), (2, 2)}

    selected, reduced = _monotone_virtual_lane_pairs(source)

    assert reduced is False
    assert selected == source
