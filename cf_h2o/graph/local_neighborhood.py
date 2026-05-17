"""Local single-agent neighborhood extraction for bus control."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from cf_h2o.graph.route_graph import RouteGraph, _coerce_float, normalize_bus, normalize_station


LOCAL_GRAPH_KEYS = (
    "ego",
    "front_buses",
    "back_buses",
    "current_station",
    "next_station",
    "segments",
    "global_context",
)

_DISTANCE_SCALE_M = 10_000.0
_SPEED_SCALE_MPS = 30.0
_LOAD_SCALE = 100.0
_WAITING_SCALE = 100.0
_COUNT_SCALE = 100.0
_TIME_SCALE_SECONDS = 86_400.0


def _route_scale(route_length_m: Any) -> float:
    route_length = _coerce_float(route_length_m, 0.0)
    return route_length if route_length > 1e-6 else _DISTANCE_SCALE_M


def _position_feature(position_m: Any, route_length_m: Any) -> float:
    return _coerce_float(position_m, 0.0) / _route_scale(route_length_m)


def _distance_feature(distance_m: Any, route_length_m: Any) -> float:
    return _coerce_float(distance_m, 0.0) / _route_scale(route_length_m)


def _speed_feature(speed_mps: Any) -> float:
    return _coerce_float(speed_mps, 0.0) / _SPEED_SCALE_MPS


def _load_feature(load: Any) -> float:
    return _coerce_float(load, 0.0) / _LOAD_SCALE


def _waiting_feature(waiting_count: Any) -> float:
    return _coerce_float(waiting_count, 0.0) / _WAITING_SCALE


def _count_feature(count: Any) -> float:
    return _coerce_float(count, 0.0) / _COUNT_SCALE


def _empty_bus() -> dict[str, Any]:
    return {
        "bus_id": "",
        "line_id": "",
        "position_m": 0.0,
        "route_length_m": None,
        "speed_mps": 0.0,
        "load": 0.0,
        "direction": 0,
        "road_id": "",
        "distance_to_ego_m": 0.0,
        "ahead_distance_m": 0.0,
        "exists": False,
        "mask": 0.0,
    }


def _empty_station() -> dict[str, Any]:
    return {
        "station_id": "",
        "station_name": "",
        "line_id": "",
        "position_m": 0.0,
        "route_length_m": None,
        "waiting_count": 0.0,
        "distance_to_ego_m": 0.0,
        "exists": False,
        "mask": 0.0,
    }


def _empty_segment() -> dict[str, Any]:
    return {
        "segment_id": "",
        "line_id": "",
        "from_station_id": "",
        "to_station_id": "",
        "from_node_id": "",
        "to_node_id": "",
        "start_m": 0.0,
        "end_m": 0.0,
        "center_m": 0.0,
        "length_m": 0.0,
        "distance_to_ego_m": 0.0,
        "exists": False,
        "mask": 0.0,
    }


def _with_bus_distances(bus: dict[str, Any], ego: dict[str, Any]) -> dict[str, Any]:
    result = dict(bus)
    signed_distance = result["position_m"] - ego["position_m"]
    direction = ego["direction"] if ego["direction"] != 0 else 1
    result["distance_to_ego_m"] = signed_distance
    result["ahead_distance_m"] = signed_distance * direction
    return result


def _with_station_distance(station: dict[str, Any], ego: dict[str, Any]) -> dict[str, Any]:
    result = dict(station)
    result["distance_to_ego_m"] = result["position_m"] - ego["position_m"]
    if result["route_length_m"] is None:
        result["route_length_m"] = ego.get("route_length_m")
    return result


def _with_segment_distance(segment: dict[str, Any], ego: dict[str, Any]) -> dict[str, Any]:
    result = dict(segment)
    result["distance_to_ego_m"] = result["center_m"] - ego["position_m"]
    return result


def _pad_list(items: Sequence[dict[str, Any]], length: int, pad_factory) -> list[dict[str, Any]]:
    length = max(0, int(length))
    padded = [dict(item) for item in items[:length]]
    while len(padded) < length:
        padded.append(pad_factory())
    return padded


class LocalNeighborhoodExtractor:
    """Extract a deterministic, padded local graph from one present snapshot."""

    def __init__(self, max_front_back: int = 1, station_radius: int = 1, segment_radius: int = 1):
        self.max_front_back = max(0, int(max_front_back))
        self.station_radius = max(0, int(station_radius))
        self.segment_radius = max(0, int(segment_radius))

    def extract(self, snapshot: Mapping[str, Any], ego_bus_id: str | None = None) -> dict[str, Any]:
        """Return deterministic local context using only present-time fields."""

        raw_buses = snapshot.get("all_buses") or snapshot.get("buses") or []
        raw_stations = snapshot.get("all_stations") or snapshot.get("stations") or []
        buses = [normalize_bus(bus, idx) for idx, bus in enumerate(raw_buses)]
        stations = [normalize_station(station, idx) for idx, station in enumerate(raw_stations)]
        buses.sort(key=lambda item: (item["line_id"], item["position_m"], item["bus_id"]))
        stations.sort(key=lambda item: (item["line_id"], item["position_m"], item["station_id"]))

        requested_ego_id = ego_bus_id if ego_bus_id is not None else snapshot.get("ego_bus_id")
        if requested_ego_id is None and buses:
            requested_ego_id = buses[0]["bus_id"]

        ego = next((bus for bus in buses if bus["bus_id"] == str(requested_ego_id)), None)
        ego = dict(ego) if ego is not None else _empty_bus()
        ego["distance_to_ego_m"] = 0.0
        ego["ahead_distance_m"] = 0.0

        same_line_buses = []
        if ego["exists"]:
            same_line_buses = [
                _with_bus_distances(bus, ego)
                for bus in buses
                if bus["line_id"] == ego["line_id"] and bus["bus_id"] != ego["bus_id"]
            ]
        front_buses = sorted(
            [bus for bus in same_line_buses if bus["ahead_distance_m"] > 0.0],
            key=lambda item: (item["ahead_distance_m"], item["bus_id"]),
        )
        back_buses = sorted(
            [bus for bus in same_line_buses if bus["ahead_distance_m"] < 0.0],
            key=lambda item: (abs(item["ahead_distance_m"]), item["bus_id"]),
        )

        line_stations = []
        if ego["exists"]:
            line_stations = [
                _with_station_distance(station, ego)
                for station in stations
                if station["line_id"] == ego["line_id"]
            ]

        direction = ego["direction"] if ego["direction"] != 0 else 1
        current_candidates = [station for station in line_stations if station["distance_to_ego_m"] * direction <= 0.0]
        next_candidates = [station for station in line_stations if station["distance_to_ego_m"] * direction > 0.0]
        current_station = (
            max(current_candidates, key=lambda item: item["distance_to_ego_m"] * direction)
            if current_candidates
            else _empty_station()
        )
        next_station = (
            min(next_candidates, key=lambda item: item["distance_to_ego_m"] * direction)
            if next_candidates
            else _empty_station()
        )

        route_graph = RouteGraph.from_snapshot(snapshot)
        segments = []
        if ego["exists"]:
            current_segment = route_graph.find_segment_at(ego["line_id"], ego["position_m"])
            if current_segment is not None:
                segment_ids = route_graph.get_segment_neighbors(current_segment["segment_id"], self.segment_radius)
                segment_by_id = {segment["segment_id"]: segment for segment in route_graph.segments}
                segments = [_with_segment_distance(segment_by_id[segment_id], ego) for segment_id in segment_ids]
        segment_count = self.segment_radius * 2 + 1

        global_context = {
            "sim_time": _coerce_float(snapshot.get("sim_time", snapshot.get("time", 0.0)), 0.0),
            "bus_count": float(len(buses)),
            "station_count": float(len(stations)),
            "line_count": float(len(route_graph.lines)),
        }

        return {
            "ego": ego,
            "front_buses": _pad_list(front_buses, self.max_front_back, _empty_bus),
            "back_buses": _pad_list(back_buses, self.max_front_back, _empty_bus),
            "current_station": current_station,
            "next_station": next_station,
            "segments": _pad_list(segments, segment_count, _empty_segment),
            "global_context": global_context,
        }

    def feature_dim(self) -> int:
        return local_graph_feature_dim(self)

    def feature_names(self) -> list[str]:
        return local_graph_feature_names(self)


def _bus_features(prefix: str, bus: Mapping[str, Any], names: list[str], values: list[float]) -> None:
    route_length = bus.get("route_length_m")
    fields = [
        ("mask", _coerce_float(bus.get("mask"), 0.0)),
        ("position", _position_feature(bus.get("position_m"), route_length)),
        ("distance_to_ego", _distance_feature(bus.get("distance_to_ego_m"), route_length)),
        ("ahead_distance", _distance_feature(bus.get("ahead_distance_m"), route_length)),
        ("speed", _speed_feature(bus.get("speed_mps"))),
        ("load", _load_feature(bus.get("load"))),
        ("direction", _coerce_float(bus.get("direction"), 0.0)),
    ]
    for name, value in fields:
        names.append(f"{prefix}.{name}")
        values.append(float(value))


def _station_features(prefix: str, station: Mapping[str, Any], names: list[str], values: list[float]) -> None:
    route_length = station.get("route_length_m")
    fields = [
        ("mask", _coerce_float(station.get("mask"), 0.0)),
        ("position", _position_feature(station.get("position_m"), route_length)),
        ("distance_to_ego", _distance_feature(station.get("distance_to_ego_m"), route_length)),
        ("waiting_count", _waiting_feature(station.get("waiting_count"))),
    ]
    for name, value in fields:
        names.append(f"{prefix}.{name}")
        values.append(float(value))


def _segment_features(prefix: str, segment: Mapping[str, Any], names: list[str], values: list[float]) -> None:
    scale = max(_coerce_float(segment.get("end_m"), 0.0), _DISTANCE_SCALE_M)
    fields = [
        ("mask", _coerce_float(segment.get("mask"), 0.0)),
        ("start", _coerce_float(segment.get("start_m"), 0.0) / scale),
        ("end", _coerce_float(segment.get("end_m"), 0.0) / scale),
        ("center", _coerce_float(segment.get("center_m"), 0.0) / scale),
        ("length", _coerce_float(segment.get("length_m"), 0.0) / scale),
        ("distance_to_ego", _coerce_float(segment.get("distance_to_ego_m"), 0.0) / scale),
    ]
    for name, value in fields:
        names.append(f"{prefix}.{name}")
        values.append(float(value))


def _global_features(global_context: Mapping[str, Any], names: list[str], values: list[float]) -> None:
    fields = [
        ("sim_time", _coerce_float(global_context.get("sim_time"), 0.0) / _TIME_SCALE_SECONDS),
        ("bus_count", _count_feature(global_context.get("bus_count"))),
        ("station_count", _count_feature(global_context.get("station_count"))),
        ("line_count", _count_feature(global_context.get("line_count"))),
    ]
    for name, value in fields:
        names.append(f"global_context.{name}")
        values.append(float(value))


def flatten_local_neighborhood(local_graph: Mapping[str, Any], *, return_names: bool = False):
    """Flatten a local graph into numeric policy-safe features.

    Identifiers such as bus_id, line_id, source/domain labels, and target/future
    fields are intentionally excluded from the vector.
    """

    names: list[str] = []
    values: list[float] = []
    _bus_features("ego", local_graph["ego"], names, values)
    for idx, bus in enumerate(local_graph["front_buses"]):
        _bus_features(f"front_buses.{idx}", bus, names, values)
    for idx, bus in enumerate(local_graph["back_buses"]):
        _bus_features(f"back_buses.{idx}", bus, names, values)
    _station_features("current_station", local_graph["current_station"], names, values)
    _station_features("next_station", local_graph["next_station"], names, values)
    for idx, segment in enumerate(local_graph["segments"]):
        _segment_features(f"segments.{idx}", segment, names, values)
    _global_features(local_graph["global_context"], names, values)

    if return_names:
        return values, names
    return values


def local_graph_feature_names(extractor: LocalNeighborhoodExtractor | None = None) -> list[str]:
    extractor = extractor or LocalNeighborhoodExtractor()
    values, names = flatten_local_neighborhood(extractor.extract({"all_buses": [], "all_stations": []}), return_names=True)
    if len(values) != len(names):
        raise RuntimeError("feature name/value length mismatch")
    return names


def local_graph_feature_dim(extractor: LocalNeighborhoodExtractor | None = None) -> int:
    return len(local_graph_feature_names(extractor))


def local_graph_to_tensor(local_graph_batch: Any, *, device=None, dtype=None) -> torch.Tensor:
    """Convert local graph input variants to a dense feature tensor."""

    dtype = dtype or torch.float32
    if torch.is_tensor(local_graph_batch):
        features = local_graph_batch
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features.to(device=device, dtype=dtype)

    if isinstance(local_graph_batch, Mapping) and "features" in local_graph_batch:
        features = local_graph_batch["features"]
        if not torch.is_tensor(features):
            features = torch.as_tensor(features, dtype=dtype, device=device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features.to(device=device, dtype=dtype)

    if isinstance(local_graph_batch, Mapping) and all(key in local_graph_batch for key in LOCAL_GRAPH_KEYS):
        features = torch.as_tensor(flatten_local_neighborhood(local_graph_batch), dtype=dtype, device=device)
        return features.unsqueeze(0)

    if isinstance(local_graph_batch, Sequence):
        rows = [flatten_local_neighborhood(item) for item in local_graph_batch]
        return torch.as_tensor(rows, dtype=dtype, device=device)

    raise TypeError(f"Unsupported local graph batch type: {type(local_graph_batch)!r}")

