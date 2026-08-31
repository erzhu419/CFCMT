from __future__ import annotations

import gzip
from pathlib import Path

from scripts.data.build_chicago_for_hire_network import (
    ANCHORS_PER_AREA,
    EXPECTED_SUMO_VERSION,
    _stream_network_graph,
    lane_allows_passenger,
    largest_strongly_connected_component,
    netconvert_command,
    point_in_geometry,
    select_anchor_rows,
)


def test_largest_strong_component_is_deterministic() -> None:
    adjacency = {
        "a": ["b"],
        "b": ["a", "c"],
        "c": ["d"],
        "d": ["c"],
        "isolated": [],
    }

    assert largest_strongly_connected_component(adjacency) == {"a", "b"}


def test_polygon_and_multipolygon_holes_are_respected() -> None:
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        ],
    }
    multipolygon = {"type": "MultiPolygon", "coordinates": [polygon["coordinates"]]}

    assert point_in_geometry((2, 2), polygon)
    assert not point_in_geometry((5, 5), polygon)
    assert point_in_geometry((2, 2), multipolygon)
    assert not point_in_geometry((20, 20), multipolygon)


def test_capacity_first_farthest_anchor_selection_is_stable() -> None:
    candidates = [
        {
            "edge_id": f"edge-{index:02d}",
            "x": float(index),
            "y": 0.0,
            "capacity_score": 100.0 if index == 8 else 10.0,
        }
        for index in range(12)
    ]

    selected = select_anchor_rows(candidates, 4)

    assert selected[0]["edge_id"] == "edge-08"
    assert len({row["edge_id"] for row in selected}) == 4
    assert {row["edge_id"] for row in selected[1:]} == {
        "edge-00",
        "edge-04",
        "edge-11",
    }


def test_netconvert_command_freezes_osm_signal_options() -> None:
    command = netconvert_command(
        netconvert=Path("/runtime/bin/netconvert"),
        osm=Path("/data/Chicago.osm.gz"),
        output=Path("/out/chicago.net.xml.gz"),
        typemap=Path("/runtime/osmNetconvert.typ.xml"),
    )

    assert command[0] == "/runtime/bin/netconvert"
    assert "--geometry.remove" in command
    assert "--ramps.guess" in command
    assert "--junctions.join" in command
    assert "--tls.guess-signals" in command
    assert "--tls.discard-simple" in command
    assert "--tls.join" in command
    assert command[command.index("--tls.default-type") + 1] == "actuated"
    assert EXPECTED_SUMO_VERSION == "1.22.0"
    assert ANCHORS_PER_AREA == 32


def test_streaming_network_graph_preserves_passenger_connections(tmp_path: Path) -> None:
    network = tmp_path / "tiny.net.xml.gz"
    xml = """<net>
    <location netOffset="-1,-2" convBoundary="0,0,10,10" origBoundary="0,0,1,1" projParameter="+proj=longlat +datum=WGS84 +no_defs"/>
    <edge id="a" from="n0" to="n1"><lane id="a_0" speed="10" length="20" shape="0,0 10,0"/></edge>
    <edge id="b" from="n1" to="n0"><lane id="b_0" speed="10" length="20" shape="10,0 0,0"/></edge>
    <edge id="bus"><lane id="bus_0" allow="bus" speed="10" length="20" shape="0,1 10,1"/></edge>
    <junction id="n0" type="traffic_light" x="0" y="0" incLanes="b_0" intLanes=""/>
    <junction id="n1" type="priority" x="10" y="0" incLanes="a_0" intLanes=""/>
    <tlLogic id="n0" type="actuated" programID="0" offset="0"><phase duration="30" state="Gr"/></tlLogic>
    <connection from="a" to="b" fromLane="0" toLane="0"/>
    <connection from="b" to="a" fromLane="0" toLane="0"/>
    </net>"""
    with gzip.open(network, "wt", encoding="utf-8") as handle:
        handle.write(xml)

    inventory, eligible, adjacency = _stream_network_graph(network)

    assert inventory["traffic_light_count"] == 1
    assert inventory["connection_count"] == 2
    assert set(eligible) == {"a", "b"}
    assert adjacency == {"a": ("b",), "b": ("a",)}
    assert eligible["a"]["x"] == 5.0
    assert lane_allows_passenger({})
    assert not lane_allows_passenger({"allow": "bus"})
