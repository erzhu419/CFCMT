import math

from scripts.data.cityflow_anchor_router import (
    CityFlowLengthRouter,
    anchors_are_subsequence,
    cityflow_average_road_lengths,
)


def _roadnet() -> dict:
    intersections = []
    for index in range(5):
        outgoing = []
        incoming = []
        if index > 0:
            incoming.append(f"r{index - 1}")
        if index < 4:
            outgoing.append(f"r{index}")
        intersections.append(
            {
                "id": f"i{index}",
                "virtual": True,
                "point": {"x": index * 10, "y": 0},
                "roads": [*incoming, *outgoing],
                "roadLinks": (
                    [{"startRoad": incoming[0], "endRoad": outgoing[0]}]
                    if incoming and outgoing
                    else []
                ),
                "trafficLight": {"lightphases": []},
            }
        )
    roads = [
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
        for index in range(4)
    ]
    return {"intersections": intersections, "roads": roads}


def test_average_length_ports_cityflow_intersection_trimming() -> None:
    roadnet = {
        "intersections": [
            {
                "id": "a",
                "virtual": False,
                "width": 10,
                "roads": ["r"],
                "roadLinks": [],
            },
            {
                "id": "b",
                "virtual": False,
                "width": 10,
                "roads": ["r"],
                "roadLinks": [],
            },
        ],
        "roads": [
            {
                "id": "r",
                "startIntersection": "a",
                "endIntersection": "b",
                "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                "lanes": [
                    {"width": 3.0, "maxSpeed": 10},
                    {"width": 3.0, "maxSpeed": 10},
                ],
            }
        ],
    }

    lengths = cityflow_average_road_lengths(roadnet)

    assert math.isclose(lengths["r"], 80.0)


def test_length_router_completes_anchor_sequence() -> None:
    router = CityFlowLengthRouter(_roadnet())
    completions = router.shortest_paths({("r0", "r3")})

    expanded = router.expand(("r0", "r3"), completions)

    assert expanded == ("r0", "r1", "r2", "r3")
    assert anchors_are_subsequence(("r0", "r3"), expanded)
    assert all(pair in router.links for pair in zip(expanded, expanded[1:]))


def test_length_router_preserves_direct_and_duplicate_anchors() -> None:
    router = CityFlowLengthRouter(_roadnet())

    expanded = router.expand(("r0", "r0", "r1"), {})

    assert expanded == ("r0", "r1")
