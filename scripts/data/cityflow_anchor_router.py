"""Deterministic CityFlow-compatible length routing between flow anchors."""

from __future__ import annotations

from collections import defaultdict
import heapq
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


CITYFLOW_ROUTER_PROTOCOL = "cityflow-length-dijkstra-anchor-completion-v1"
CITYFLOW_REFERENCE_COMMIT = "81ee0f47659ca66177a71f81676691c58ee89184"
NATIVE_SUMO_ROUTER_PROTOCOL = (
    "native-sumo-length-dijkstra-cityflow-anchor-completion-v2"
)


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 0.0:
        raise ValueError("CityFlow road geometry contains a zero-length segment")
    return vector[0] / length, vector[1] / length


def _add(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    return left[0] + right[0], left[1] + right[1]


def _sub(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    return left[0] - right[0], left[1] - right[1]


def _scale(vector: tuple[float, float], factor: float) -> tuple[float, float]:
    return vector[0] * factor, vector[1] * factor


def cityflow_average_road_lengths(roadnet: Mapping[str, Any]) -> dict[str, float]:
    """Port CityFlow ``Road::initLanesPoints`` and ``averageLength``."""
    intersections = {
        str(row["id"]): dict(row) for row in roadnet["intersections"]
    }
    lengths: dict[str, float] = {}
    for road in roadnet["roads"]:
        road_id = str(road["id"])
        points = [
            (float(point["x"]), float(point["y"]))
            for point in road["points"]
        ]
        if len(points) < 2:
            raise ValueError(f"CityFlow road has fewer than two points: {road_id}")
        start = intersections[str(road["startIntersection"])]
        end = intersections[str(road["endIntersection"])]
        if not bool(start.get("virtual", False)):
            direction = _unit(_sub(points[1], points[0]))
            points[0] = _add(points[0], _scale(direction, float(start["width"])))
        if not bool(end.get("virtual", False)):
            direction = _unit(_sub(points[-1], points[-2]))
            points[-1] = _sub(points[-1], _scale(direction, float(end["width"])))

        lane_lengths: list[float] = []
        cumulative_width = 0.0
        for lane in road["lanes"]:
            width = float(lane["width"])
            offset = cumulative_width + width / 2.0
            lane_points: list[tuple[float, float]] = []
            for index, point in enumerate(points):
                if index == 0:
                    direction = _unit(_sub(points[1], points[0]))
                elif index == len(points) - 1:
                    direction = _unit(_sub(points[-1], points[-2]))
                else:
                    ahead = _unit(_sub(points[index + 1], points[index]))
                    behind = _unit(_sub(points[index], points[index - 1]))
                    direction = _unit(_add(ahead, behind))
                # CityFlow uses ``-direction.normal()`` = (dy, -dx).
                normal = (direction[1], -direction[0])
                lane_points.append(_add(point, _scale(normal, offset)))
            lane_length = sum(
                math.hypot(
                    lane_points[index + 1][0] - lane_points[index][0],
                    lane_points[index + 1][1] - lane_points[index][1],
                )
                for index in range(len(lane_points) - 1)
            )
            lane_lengths.append(lane_length)
            cumulative_width += width
        if not lane_lengths or min(lane_lengths) <= 0.0:
            raise ValueError(f"CityFlow road has invalid lane geometry: {road_id}")
        lengths[road_id] = sum(lane_lengths) / len(lane_lengths)
    return lengths


class LengthAnchorRouter:
    """Complete sparse road anchors on a fixed directed length graph."""

    def __init__(
        self,
        *,
        lengths: Mapping[str, float],
        links: Iterable[tuple[str, str]],
        adjacency: Mapping[str, Sequence[str]],
        road_order: Mapping[str, int],
    ) -> None:
        self.lengths = {str(key): float(value) for key, value in lengths.items()}
        self.links = {(str(start), str(end)) for start, end in links}
        self.adjacency = {
            str(key): tuple(str(value) for value in values)
            for key, values in adjacency.items()
        }
        self.road_order = {str(key): int(value) for key, value in road_order.items()}

    def shortest_paths(
        self, pairs: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        targets_by_start: dict[str, set[str]] = defaultdict(set)
        for start, end in set(pairs):
            if start not in self.lengths or end not in self.lengths:
                raise ValueError(f"route anchor is absent: {(start, end)}")
            if start != end and (start, end) not in self.links:
                targets_by_start[start].add(end)
        completed: dict[tuple[str, str], tuple[str, ...]] = {}
        for start, targets in targets_by_start.items():
            distance = {start: 0.0}
            predecessor: dict[str, str] = {}
            queue: list[tuple[float, int, str]] = [
                (0.0, self.road_order[start], start)
            ]
            remaining = set(targets)
            while queue and remaining:
                current_distance, _, current = heapq.heappop(queue)
                if current_distance != distance.get(current):
                    continue
                remaining.discard(current)
                for adjacent in self.adjacency[current]:
                    candidate = current_distance + self.lengths[adjacent]
                    if candidate < distance.get(adjacent, math.inf):
                        distance[adjacent] = candidate
                        predecessor[adjacent] = current
                        heapq.heappush(
                            queue,
                            (candidate, self.road_order[adjacent], adjacent),
                        )
            if remaining:
                raise ValueError(
                    f"route anchors are unreachable from {start}: "
                    f"{sorted(remaining)[:10]}"
                )
            for end in targets:
                reverse_path = [end]
                while reverse_path[-1] != start:
                    reverse_path.append(predecessor[reverse_path[-1]])
                completed[(start, end)] = tuple(reversed(reverse_path))
        return completed

    def expand(
        self,
        anchors: Sequence[str],
        completions: Mapping[tuple[str, str], Sequence[str]],
    ) -> tuple[str, ...]:
        if not anchors:
            raise ValueError("CityFlow anchor route is empty")
        expanded = [str(anchors[0])]
        for raw_end in anchors[1:]:
            start = expanded[-1]
            end = str(raw_end)
            if start == end:
                continue
            if (start, end) in self.links:
                expanded.append(end)
                continue
            path = tuple(completions[(start, end)])
            if path[0] != start or path[-1] != end:
                raise ValueError(f"invalid completed anchor path: {(start, end)}")
            expanded.extend(path[1:])
        return tuple(expanded)


class CityFlowLengthRouter(LengthAnchorRouter):
    """Reproduce CityFlow's default anchor routing on a roadnet JSON object."""

    def __init__(self, roadnet: Mapping[str, Any]) -> None:
        roads = [dict(row) for row in roadnet["roads"]]
        intersections = {
            str(row["id"]): dict(row) for row in roadnet["intersections"]
        }
        links = {
            (str(link["startRoad"]), str(link["endRoad"]))
            for intersection in intersections.values()
            for link in intersection.get("roadLinks", [])
        }
        adjacency: dict[str, tuple[str, ...]] = {}
        for road in roads:
            road_id = str(road["id"])
            end = intersections[str(road["endIntersection"])]
            adjacency[road_id] = tuple(
                str(candidate)
                for candidate in end.get("roads", [])
                if (road_id, str(candidate)) in links
            )
        super().__init__(
            lengths=cityflow_average_road_lengths(roadnet),
            links=links,
            adjacency=adjacency,
            road_order={str(row["id"]): index for index, row in enumerate(roads)},
        )


class NativeSumoLengthRouter(LengthAnchorRouter):
    """Complete CityFlow anchors on the released native SUMO graph."""

    def __init__(self, net_path: Path) -> None:
        root = ET.parse(net_path).getroot()
        edges = [
            edge for edge in root.findall("edge") if not edge.attrib.get("function")
        ]
        road_order = {
            str(edge.attrib["id"]): index for index, edge in enumerate(edges)
        }
        lengths = {}
        for edge in edges:
            lanes = edge.findall("lane")
            if not lanes:
                raise ValueError(f"native SUMO edge has no lanes: {edge.attrib['id']}")
            lengths[str(edge.attrib["id"])] = sum(
                float(lane.attrib["length"]) for lane in lanes
            ) / len(lanes)
        ordered_links: list[tuple[str, str]] = []
        seen_links: set[tuple[str, str]] = set()
        for connection in root.findall("connection"):
            pair = (
                str(connection.attrib.get("from", "")),
                str(connection.attrib.get("to", "")),
            )
            if (
                pair[0] in lengths
                and pair[1] in lengths
                and pair not in seen_links
            ):
                ordered_links.append(pair)
                seen_links.add(pair)
        adjacency_lists: dict[str, list[str]] = {
            edge_id: [] for edge_id in lengths
        }
        for start, end in ordered_links:
            adjacency_lists[start].append(end)
        super().__init__(
            lengths=lengths,
            links=seen_links,
            adjacency=adjacency_lists,
            road_order=road_order,
        )


def anchors_are_subsequence(
    anchors: Sequence[str], expanded: Sequence[str]
) -> bool:
    cursor = 0
    for anchor in anchors:
        while cursor < len(expanded) and expanded[cursor] != anchor:
            cursor += 1
        if cursor == len(expanded):
            return False
        cursor += 1
    return True
