from __future__ import annotations

from pathlib import Path

from scripts.data.build_chicago_for_hire_network import (
    ANCHORS_PER_AREA,
    EXPECTED_SUMO_VERSION,
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
