"""Static route graph construction from H2O+ bus snapshots."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_direction(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else -1
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"backward", "reverse", "down", "-"}:
            return -1
        if value in {"forward", "up", "+"}:
            return 1
    return 1 if _coerce_float(value, 1.0) >= 0.0 else -1


def _optional_positive_float(value: Any) -> Optional[float]:
    result = _coerce_float(value, 0.0)
    return result if result > 0.0 else None


def normalize_bus(bus: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize known present-time bus fields without reading future targets."""

    bus_id = _first_present(bus, ("bus_id", "id", "trip_id", "vehicle_id"), f"bus_{index}")
    line_id = _first_present(bus, ("line_id", "route_id", "belong_line_id_s"), "_default") or "_default"
    route_length = _optional_positive_float(_first_present(bus, ("route_length", "route_length_m", "total_route_length")))

    return {
        "bus_id": str(bus_id),
        "line_id": str(line_id),
        "position_m": _coerce_float(
            _first_present(bus, ("pos", "position_m", "absolute_distance", "distance_n", "distance")),
            0.0,
        ),
        "route_length_m": route_length,
        "speed_mps": _coerce_float(_first_present(bus, ("speed", "current_speed", "velocity")), 0.0),
        "load": _coerce_float(_first_present(bus, ("load", "current_load", "current_load_n", "passenger_count")), 0.0),
        "direction": _coerce_direction(_first_present(bus, ("direction", "direction_n"), 1)),
        "road_id": str(_first_present(bus, ("road_id", "edge_id", "lane_id"), "")),
        "exists": True,
        "mask": 1.0,
    }


def normalize_station(station: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize known present-time station fields without reading future targets."""

    station_id = _first_present(station, ("station_id", "stop_id", "id", "station_name"), f"station_{index}")
    line_id = _first_present(station, ("line_id", "route_id", "belong_line_id_s"), "_default") or "_default"
    route_length = _optional_positive_float(_first_present(station, ("route_length", "route_length_m", "total_route_length")))

    return {
        "station_id": str(station_id),
        "station_name": str(_first_present(station, ("station_name", "name"), station_id)),
        "line_id": str(line_id),
        "position_m": _coerce_float(
            _first_present(station, ("pos", "position_m", "absolute_distance", "distance_n", "distance")),
            0.0,
        ),
        "route_length_m": route_length,
        "waiting_count": _coerce_float(
            _first_present(
                station,
                (
                    "waiting_count",
                    "passenger_num",
                    "passenger_num_n",
                    "wait_passenger_num_n",
                    "waiting_passenger_num",
                    "waiting",
                ),
                0.0,
            ),
            0.0,
        ),
        "exists": True,
        "mask": 1.0,
    }


class RouteGraph:
    """Line-aware station/segment topology inferred from a present snapshot."""

    def __init__(self, stations: list[dict[str, Any]], segments: list[dict[str, Any]], lines: list[str]):
        self.stations = list(stations)
        self.segments = list(segments)
        self.lines = list(lines)

        self._station_by_node_id = {station["node_id"]: station for station in self.stations}
        self._station_alias: dict[str, str] = {}
        for station in self.stations:
            self._station_alias.setdefault(station["station_id"], station["node_id"])
            self._station_alias[station["node_id"]] = station["node_id"]

        self._segment_by_id = {segment["segment_id"]: segment for segment in self.segments}
        self._station_order_by_line: dict[str, list[str]] = defaultdict(list)
        self._segment_order_by_line: dict[str, list[str]] = defaultdict(list)
        for station in self.stations:
            self._station_order_by_line[station["line_id"]].append(station["node_id"])
        for segment in self.segments:
            self._segment_order_by_line[segment["line_id"]].append(segment["segment_id"])

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "RouteGraph":
        """Build a route graph from present-time station records in a snapshot."""

        raw_stations = snapshot.get("all_stations") or snapshot.get("stations") or []
        stations = [normalize_station(station, idx) for idx, station in enumerate(raw_stations)]
        stations.sort(key=lambda item: (item["line_id"], item["position_m"], item["station_id"]))

        enriched_stations: list[dict[str, Any]] = []
        stations_by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for station in stations:
            node_id = f"{station['line_id']}:{station['station_id']}"
            enriched = {
                **station,
                "node_id": node_id,
                "index_in_line": len(stations_by_line[station["line_id"]]),
            }
            enriched_stations.append(enriched)
            stations_by_line[station["line_id"]].append(enriched)

        segments: list[dict[str, Any]] = []
        for line_id in sorted(stations_by_line):
            line_stations = stations_by_line[line_id]
            for idx, (start, end) in enumerate(zip(line_stations[:-1], line_stations[1:])):
                start_m = min(start["position_m"], end["position_m"])
                end_m = max(start["position_m"], end["position_m"])
                segment_id = f"{line_id}:{start['station_id']}->{end['station_id']}"
                segments.append(
                    {
                        "segment_id": segment_id,
                        "line_id": line_id,
                        "from_station_id": start["station_id"],
                        "to_station_id": end["station_id"],
                        "from_node_id": start["node_id"],
                        "to_node_id": end["node_id"],
                        "start_m": start_m,
                        "end_m": end_m,
                        "center_m": 0.5 * (start_m + end_m),
                        "length_m": max(0.0, end_m - start_m),
                        "index_in_line": idx,
                        "exists": True,
                        "mask": 1.0,
                    }
                )

        lines = sorted(stations_by_line)
        return cls(enriched_stations, segments, lines)

    def get_station_neighbors(self, station_id: str, radius: int = 1) -> list[str]:
        """Return neighboring station ids in route order, including the query."""

        node_id = self._station_alias.get(str(station_id))
        if node_id is None:
            return []
        station = self._station_by_node_id[node_id]
        order = self._station_order_by_line[station["line_id"]]
        idx = order.index(node_id)
        radius = max(0, int(radius))
        neighbor_nodes = order[max(0, idx - radius) : idx + radius + 1]
        return [self._station_by_node_id[item]["station_id"] for item in neighbor_nodes]

    def get_segment_neighbors(self, segment_id: str, radius: int = 1) -> list[str]:
        """Return neighboring segment ids in route order, including the query."""

        segment = self._segment_by_id.get(str(segment_id))
        if segment is None:
            return []
        order = self._segment_order_by_line[segment["line_id"]]
        idx = order.index(segment["segment_id"])
        radius = max(0, int(radius))
        return order[max(0, idx - radius) : idx + radius + 1]

    def find_segment_at(self, line_id: str, position_m: float) -> Optional[dict[str, Any]]:
        """Find the containing segment, falling back to the nearest segment."""

        segment_ids = self._segment_order_by_line.get(str(line_id), [])
        if not segment_ids:
            return None
        position_m = _coerce_float(position_m, 0.0)
        segments = [self._segment_by_id[item] for item in segment_ids]
        for segment in segments:
            if segment["start_m"] <= position_m <= segment["end_m"]:
                return segment
        return min(segments, key=lambda item: abs(item["center_m"] - position_m))

    def to_pyg_data(self) -> object:
        """Return torch_geometric Data when available, otherwise a plain dict."""

        station_ids = [station["station_id"] for station in self.stations]
        node_ids = [station["node_id"] for station in self.stations]
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        edge_pairs: list[tuple[int, int]] = []
        for segment in self.segments:
            src = node_index[segment["from_node_id"]]
            dst = node_index[segment["to_node_id"]]
            edge_pairs.append((src, dst))
            edge_pairs.append((dst, src))

        if not edge_pairs:
            edge_index = [[], []]
        else:
            edge_index = [[src for src, _ in edge_pairs], [dst for _, dst in edge_pairs]]

        try:
            import torch
            from torch_geometric.data import Data
        except Exception:
            return {
                "station_ids": station_ids,
                "node_ids": node_ids,
                "segment_ids": [segment["segment_id"] for segment in self.segments],
                "edge_index": edge_index,
                "lines": list(self.lines),
            }

        edge_tensor = torch.tensor(edge_index, dtype=torch.long)
        return Data(edge_index=edge_tensor, num_nodes=len(self.stations), station_ids=station_ids, node_ids=node_ids)

