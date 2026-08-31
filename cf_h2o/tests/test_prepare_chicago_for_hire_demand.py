from __future__ import annotations

from scripts.data.prepare_chicago_for_hire_demand import (
    BIN_SECONDS,
    CENTROID_ANCHORS,
    feasible_anchor_pairs,
    resolve_endpoint,
    stratified_departure,
)


def _areas():
    square = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    return [
        {"area": 1, "name": "ONE", "geometry": square, "bounds": (0, 0, 1, 1)},
        {
            "area": 2,
            "name": "TWO",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
            },
            "bounds": (1, 0, 2, 1),
        },
    ]


def _anchors():
    return {
        area: [
            {
                "edge_id": f"a{area}-{index:02d}",
                "lon": area - 0.9 + index / 1000,
                "lat": 0.5,
            }
            for index in range(32)
        ]
        for area in (1, 2)
    }


def test_centroid_precedes_released_area_and_records_mismatch() -> None:
    endpoint = resolve_endpoint(
        area_value="2",
        latitude_value="0.5",
        longitude_value="0.5",
        areas=_areas(),
        anchors=_anchors(),
    )

    assert endpoint is not None
    assert endpoint["area"] == 1
    assert endpoint["mode"] == "centroid"
    assert endpoint["released_area_mismatch"]
    assert len(endpoint["anchors"]) == CENTROID_ANCHORS


def test_area_fallback_and_publicly_unlocatable_endpoint_are_distinct() -> None:
    fallback = resolve_endpoint(
        area_value="2",
        latitude_value="",
        longitude_value="",
        areas=_areas(),
        anchors=_anchors(),
    )
    missing = resolve_endpoint(
        area_value="",
        latitude_value="",
        longitude_value="",
        areas=_areas(),
        anchors=_anchors(),
    )

    assert fallback is not None
    assert fallback["mode"] == "community_area_fallback"
    assert len(fallback["anchors"]) == 32
    assert missing is None


def test_pairing_skips_same_edge_and_is_deterministic() -> None:
    origin = {"key": ("a",), "anchors": ("e1", "e2")}
    destination = {"key": ("b",), "anchors": ("e1", "e3")}

    assert feasible_anchor_pairs(origin, destination) == (
        ("e1", "e3"),
        ("e2", "e1"),
        ("e2", "e3"),
    )


def test_departures_are_evenly_stratified_inside_the_privacy_bin() -> None:
    values = [
        stratified_departure(
            bin_start_sec=3600,
            rank=rank,
            count=3,
            day_offset_sec=86_400,
        )
        for rank in range(3)
    ]

    assert values == [90_150.0, 90_450.0, 90_750.0]
    assert values[-1] < 86_400 + 3600 + BIN_SECONDS
