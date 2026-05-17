from __future__ import annotations

import torch

from cf_h2o.encoders.local_graph_encoder import LocalGraphEncoder
from cf_h2o.graph.local_neighborhood import (
    LOCAL_GRAPH_KEYS,
    LocalNeighborhoodExtractor,
    flatten_local_neighborhood,
)
from cf_h2o.graph.route_graph import RouteGraph


def _snapshot():
    return {
        "sim_time": 120.0,
        "ego_bus_id": "ego",
        "all_buses": [
            {"bus_id": "back", "line_id": "7X", "pos": 50.0, "speed": 7.0, "load": 8, "direction": 1},
            {"bus_id": "ego", "line_id": "7X", "pos": 110.0, "route_length": 300.0, "speed": 9.0, "load": 20, "direction": 1},
            {"bus_id": "front", "line_id": "7X", "pos": 160.0, "speed": 8.0, "load": 16, "direction": 1},
            {"bus_id": "other-line", "line_id": "102S", "pos": 115.0, "speed": 2.0, "load": 1, "direction": 1},
        ],
        "all_stations": [
            {"station_id": "s0", "line_id": "7X", "pos": 0.0, "route_length": 300.0, "waiting_count": 2},
            {"station_id": "s1", "line_id": "7X", "pos": 100.0, "route_length": 300.0, "waiting_count": 3},
            {"station_id": "s2", "line_id": "7X", "pos": 200.0, "route_length": 300.0, "waiting_count": 4},
            {"station_id": "s3", "line_id": "7X", "pos": 300.0, "route_length": 300.0, "waiting_count": 5},
            {"station_id": "o0", "line_id": "102S", "pos": 0.0, "waiting_count": 9},
            {"station_id": "o1", "line_id": "102S", "pos": 100.0, "waiting_count": 10},
        ],
    }


def test_route_graph_from_snapshot_neighbors_are_deterministic():
    graph = RouteGraph.from_snapshot(_snapshot())

    assert graph.lines == ["102S", "7X"]
    assert graph.get_station_neighbors("s1", radius=1) == ["s0", "s1", "s2"]
    segment_ids = [segment["segment_id"] for segment in graph.segments if segment["line_id"] == "7X"]
    assert segment_ids == ["7X:s0->s1", "7X:s1->s2", "7X:s2->s3"]
    assert graph.get_segment_neighbors("7X:s1->s2", radius=1) == segment_ids

    data = graph.to_pyg_data()
    if isinstance(data, dict):
        assert data["station_ids"] == ["o0", "o1", "s0", "s1", "s2", "s3"]
        assert data["segment_ids"][:1] == ["102S:o0->o1"]
    else:
        assert data.num_nodes == 6


def test_local_neighborhood_is_deterministic_and_uses_masks():
    extractor = LocalNeighborhoodExtractor(max_front_back=2, segment_radius=1)
    local_a = extractor.extract(_snapshot())
    local_b = extractor.extract(_snapshot())

    assert list(local_a.keys()) == list(LOCAL_GRAPH_KEYS)
    assert local_a == local_b
    assert local_a["ego"]["bus_id"] == "ego"
    assert [bus["bus_id"] for bus in local_a["front_buses"]] == ["front", ""]
    assert [bus["mask"] for bus in local_a["front_buses"]] == [1.0, 0.0]
    assert [bus["bus_id"] for bus in local_a["back_buses"]] == ["back", ""]
    assert local_a["current_station"]["station_id"] == "s1"
    assert local_a["next_station"]["station_id"] == "s2"
    assert [segment["segment_id"] for segment in local_a["segments"]] == [
        "7X:s0->s1",
        "7X:s1->s2",
        "7X:s2->s3",
    ]

    missing = extractor.extract({"all_buses": [], "all_stations": []}, ego_bus_id="missing")
    assert missing["ego"]["mask"] == 0.0
    assert all(bus["mask"] == 0.0 for bus in missing["front_buses"])
    assert all(segment["mask"] == 0.0 for segment in missing["segments"])


def test_local_neighborhood_features_do_not_use_future_fields():
    sentinel = 123_456_789.0
    snapshot = _snapshot()
    snapshot["snapshot_t1"] = {"all_buses": [{"pos": sentinel}], "all_stations": [{"waiting_count": sentinel}]}
    snapshot["next_observations"] = [sentinel]
    snapshot["all_buses"][1]["future_pos"] = sentinel
    snapshot["all_buses"][1]["next_speed"] = sentinel
    snapshot["all_stations"][1]["future_waiting_count"] = sentinel
    snapshot["all_stations"][2]["next_waiting_count"] = sentinel

    extractor = LocalNeighborhoodExtractor(max_front_back=1, segment_radius=1)
    local_graph = extractor.extract(snapshot)
    features = flatten_local_neighborhood(local_graph)

    assert local_graph["current_station"]["waiting_count"] == 3.0
    assert local_graph["next_station"]["waiting_count"] == 4.0
    assert max(abs(value) for value in features) < 1000.0


def test_local_graph_encoder_output_shape_and_tensor_input():
    extractor = LocalNeighborhoodExtractor(max_front_back=1, segment_radius=1)
    local_graph = extractor.extract(_snapshot())
    feature_dim = extractor.feature_dim()
    encoder = LocalGraphEncoder({"feature_dim": feature_dim}, hidden_dim=16, out_dim=7)

    out = encoder([local_graph, local_graph])
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()

    tensor_out = encoder(torch.zeros(3, feature_dim))
    assert tensor_out.shape == (3, 7)
