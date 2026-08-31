"""Streaming readers for static SUMO demand spread across config inputs."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import IO, Iterator, Mapping, Sequence
from xml.etree import ElementTree as ET


DEMAND_TAGS = frozenset({"vehicle", "trip", "flow"})
DEMAND_ROOT_TAGS = frozenset({"additional", "routes"})


def parse_sumo_time_seconds(value: str | float | int) -> float:
    """Parse SUMO numeric or human-readable D:H:M:S time values."""

    text = str(value).strip()
    if not text:
        raise ValueError("empty SUMO time value")
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 4 or any(not part for part in parts):
        raise ValueError(f"invalid SUMO time value: {text!r}")
    values = [float(part) for part in parts]
    multipliers = {
        2: (60.0, 1.0),
        3: (3_600.0, 60.0, 1.0),
        4: (86_400.0, 3_600.0, 60.0, 1.0),
    }[len(values)]
    return sum(value * multiplier for value, multiplier in zip(values, multipliers))


def open_xml_binary(path: Path) -> IO[bytes]:
    """Open plain or gzip-compressed SUMO XML without materializing it."""

    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def parse_xml(path: Path) -> ET.ElementTree:
    with open_xml_binary(path) as handle:
        return ET.parse(handle)


def iterparse_xml(
    path: Path, *, events: Sequence[str] = ("end",)
) -> Iterator[tuple[str, ET.Element]]:
    with open_xml_binary(path) as handle:
        yield from ET.iterparse(handle, events=tuple(events))


@dataclass(frozen=True)
class StaticDemandRecord:
    kind: str
    attributes: Mapping[str, str]
    inline_edges: tuple[str, ...]


@dataclass(frozen=True)
class StaticRouteCatalog:
    routes: Mapping[str, tuple[str, ...]]
    distributions: Mapping[
        str, tuple[tuple[tuple[str, ...], float], ...]
    ]


def resolve_sumo_input_paths(
    sumocfg: Path, tags: Sequence[str]
) -> tuple[Path, ...]:
    sumocfg = Path(sumocfg).resolve()
    root = parse_xml(sumocfg).getroot()
    paths: list[Path] = []
    for tag in tags:
        for element in root.findall(f".//{tag}"):
            for value in str(element.attrib.get("value", "")).split(","):
                value = value.strip()
                if value:
                    path = (sumocfg.parent / value).resolve()
                    if path not in paths:
                        paths.append(path)
    return tuple(paths)


def static_demand_input_paths(sumocfg: Path) -> tuple[Path, ...]:
    return resolve_sumo_input_paths(
        sumocfg, ("route-files", "additional-files")
    )


def load_static_route_catalog(paths: Sequence[Path]) -> StaticRouteCatalog:
    routes: dict[str, tuple[str, ...]] = {}
    raw_distributions: dict[
        str, list[tuple[str | None, tuple[str, ...], float]]
    ] = {}
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        active_id: str | None = None
        active_members: list[tuple[str | None, tuple[str, ...], float]] = []
        for event, element in iterparse_xml(
            path, events=("start", "end")
        ):
            if event == "start" and element.tag == "routeDistribution":
                active_id = str(element.attrib.get("id", ""))
                if not active_id or active_id in raw_distributions:
                    raise ValueError(
                        f"invalid route distribution in {path}: {active_id}"
                    )
                active_members = []
                route_ids = str(element.attrib.get("routes", "")).split()
                probabilities = str(
                    element.attrib.get("probabilities", "")
                ).split()
                if probabilities and len(probabilities) != len(route_ids):
                    raise ValueError(
                        f"route distribution probabilities differ in {path}: "
                        f"{active_id}"
                    )
                for index, route_id in enumerate(route_ids):
                    probability = (
                        float(probabilities[index]) if probabilities else 1.0
                    )
                    active_members.append((route_id, (), probability))
            elif event == "end" and element.tag == "route":
                edges = tuple(str(element.attrib.get("edges", "")).split())
                if active_id is not None:
                    reference = element.attrib.get("refId")
                    if reference or edges:
                        active_members.append(
                            (
                                str(reference) if reference else None,
                                edges,
                                float(element.attrib.get("probability", 1.0)),
                            )
                        )
                else:
                    route_id = element.attrib.get("id")
                    if route_id and edges:
                        previous = routes.get(str(route_id))
                        if previous is not None and previous != edges:
                            raise ValueError(
                                f"conflicting route definition in {path}: "
                                f"{route_id}"
                            )
                        routes[str(route_id)] = edges
            elif event == "end" and element.tag == "routeDistribution":
                if active_id is None or not active_members:
                    raise ValueError(f"empty route distribution in {path}")
                raw_distributions[active_id] = list(active_members)
                active_id = None
                active_members = []
            if event == "end":
                element.clear()

    distributions: dict[str, tuple[tuple[tuple[str, ...], float], ...]] = {}
    for distribution_id, members in raw_distributions.items():
        resolved: list[tuple[tuple[str, ...], float]] = []
        for route_id, inline_edges, probability in members:
            edges = inline_edges if inline_edges else routes.get(str(route_id), ())
            if not edges or probability < 0.0:
                raise ValueError(
                    "unresolved route distribution member: "
                    f"{distribution_id}/{route_id}"
                )
            resolved.append((edges, float(probability)))
        total_probability = sum(probability for _, probability in resolved)
        if total_probability <= 0.0:
            raise ValueError(
                f"route distribution has no positive mass: {distribution_id}"
            )
        distributions[distribution_id] = tuple(
            (edges, probability / total_probability)
            for edges, probability in resolved
            if probability > 0.0
        )
    return StaticRouteCatalog(routes=routes, distributions=distributions)


def iter_static_demand_records(
    paths: Sequence[Path],
) -> Iterator[StaticDemandRecord]:
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        active_kind: str | None = None
        active_attributes: dict[str, str] = {}
        inline_edges: tuple[str, ...] = ()
        ancestors: list[str] = []
        for event, element in iterparse_xml(
            path, events=("start", "end")
        ):
            if event == "start":
                parent = ancestors[-1] if ancestors else None
                ancestors.append(str(element.tag))
                if (
                    element.tag in DEMAND_TAGS
                    and parent in DEMAND_ROOT_TAGS
                ):
                    if active_kind is not None:
                        raise ValueError(f"nested SUMO demand element in {path}")
                    active_kind = str(element.tag)
                    active_attributes = dict(element.attrib)
                    inline_edges = ()
            elif (
                event == "end"
                and element.tag == "route"
                and active_kind is not None
                and "edges" in element.attrib
                and "id" not in element.attrib
            ):
                inline_edges = tuple(str(element.attrib["edges"]).split())
            elif event == "end" and element.tag == active_kind:
                yield StaticDemandRecord(
                    kind=str(active_kind),
                    attributes=active_attributes,
                    inline_edges=inline_edges,
                )
                active_kind = None
                active_attributes = {}
                inline_edges = ()
            if event == "end":
                element.clear()
                if not ancestors or ancestors[-1] != element.tag:
                    raise ValueError(f"invalid SUMO XML nesting in {path}")
                ancestors.pop()


def triggered_vehicle_departure_times(
    paths: Sequence[Path],
) -> Mapping[str, float]:
    """Map person-triggered vehicle IDs to the owning plan's departure."""

    departures: dict[str, float] = {}
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        ancestors: list[str] = []
        active_person_departure: float | None = None
        for event, element in iterparse_xml(
            path, events=("start", "end")
        ):
            if event == "start":
                parent = ancestors[-1] if ancestors else None
                ancestors.append(str(element.tag))
                if element.tag == "person" and parent in DEMAND_ROOT_TAGS:
                    depart = str(element.attrib.get("depart", ""))
                    try:
                        active_person_departure = parse_sumo_time_seconds(depart)
                    except ValueError:
                        active_person_departure = None
            elif (
                element.tag == "ride"
                and active_person_departure is not None
                and "person" in ancestors[:-1]
            ):
                for line in str(element.attrib.get("lines", "")).split():
                    previous = departures.get(line)
                    departures[line] = (
                        active_person_departure
                        if previous is None
                        else min(previous, active_person_departure)
                    )
            elif element.tag == "person" and active_person_departure is not None:
                active_person_departure = None
            if event == "end":
                element.clear()
                if not ancestors or ancestors[-1] != element.tag:
                    raise ValueError(f"invalid SUMO XML nesting in {path}")
                ancestors.pop()
    return departures


def demand_route_mixture(
    record: StaticDemandRecord,
    catalog: StaticRouteCatalog,
) -> tuple[tuple[tuple[str, ...], float], ...]:
    route_id = str(record.attributes.get("route", ""))
    if route_id in catalog.routes:
        return ((catalog.routes[route_id], 1.0),)
    if route_id in catalog.distributions:
        return catalog.distributions[route_id]
    if route_id:
        raise ValueError(f"unknown static SUMO route reference: {route_id}")
    if record.inline_edges:
        return ((record.inline_edges, 1.0),)
    origin = str(record.attributes.get("from", ""))
    via = tuple(str(record.attributes.get("via", "")).split())
    destination = str(record.attributes.get("to", ""))
    edges = tuple(value for value in (origin, *via, destination) if value)
    return ((edges, 1.0),) if edges else ()


def flow_expected_count(attributes: Mapping[str, str]) -> float:
    begin = parse_sumo_time_seconds(attributes.get("begin", 0.0))
    end = parse_sumo_time_seconds(attributes.get("end", begin))
    duration = max(end - begin, 0.0)
    if "number" in attributes:
        return max(float(attributes["number"]), 0.0)
    if "vehsPerHour" in attributes:
        return max(float(attributes["vehsPerHour"]), 0.0) * duration / 3600.0
    if "period" in attributes:
        period = parse_sumo_time_seconds(attributes["period"])
        return duration / period if period > 0.0 else 0.0
    if "probability" in attributes:
        return max(float(attributes["probability"]), 0.0) * duration
    return 1.0 if duration > 0.0 else 0.0
