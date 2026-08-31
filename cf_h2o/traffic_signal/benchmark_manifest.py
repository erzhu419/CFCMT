"""Validated scenario manifests for cross-city traffic-signal experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from cf_h2o.traffic_signal.sumo_static_inputs import (
    iterparse_xml,
    parse_xml,
    static_demand_input_paths,
)


@dataclass(frozen=True)
class TrafficSignalScenarioSpec:
    scenario: str
    city_group: str
    suite: str
    sumocfg: Path
    provenance: str
    tls_count: int
    demand_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "city_group": self.city_group,
            "suite": self.suite,
            "sumocfg": str(self.sumocfg),
            "provenance": self.provenance,
            "tls_count": int(self.tls_count),
            "demand_count": int(self.demand_count),
        }


@dataclass(frozen=True)
class TrafficSignalBenchmarkManifest:
    path: Path
    version: int
    scenarios: tuple[TrafficSignalScenarioSpec, ...]
    protocol: str = "legacy-v1"

    @property
    def sumocfgs(self) -> dict[str, Path]:
        return {item.scenario: item.sumocfg for item in self.scenarios}

    @property
    def city_groups(self) -> dict[str, str]:
        return {item.scenario: item.city_group for item in self.scenarios}

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "version": int(self.version),
            "protocol": self.protocol,
            "scenario_count": len(self.scenarios),
            "city_group_count": len(set(self.city_groups.values())),
            "scenarios": [item.to_dict() for item in self.scenarios],
        }


def load_traffic_signal_manifest(path: Path) -> TrafficSignalBenchmarkManifest:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version not in {1, 2}:
        raise ValueError(f"unsupported traffic-signal manifest version: {version}")
    protocol = str(payload.get("protocol", "legacy-v1")).strip()
    if version >= 2 and not protocol:
        raise ValueError("traffic-signal manifest version 2 requires a protocol")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("traffic-signal manifest requires a non-empty scenarios list")

    specs = []
    seen = set()
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise TypeError("manifest scenario entries must be objects")
        scenario = str(raw.get("scenario", "")).strip()
        city_group = str(raw.get("city_group", "")).strip()
        suite = str(raw.get("suite", "")).strip()
        provenance = str(raw.get("provenance", "")).strip()
        if not scenario or not city_group or not suite or not provenance:
            raise ValueError("each manifest scenario requires scenario, city_group, suite, and provenance")
        if scenario in seen:
            raise ValueError(f"duplicate manifest scenario: {scenario}")
        seen.add(scenario)
        raw_sumocfg = str(raw.get("sumocfg", ""))
        expanded_sumocfg = os.path.expandvars(raw_sumocfg)
        if "$" in expanded_sumocfg:
            raise ValueError(
                "traffic-signal manifest contains an unresolved environment "
                f"variable: {raw_sumocfg}"
            )
        sumocfg_path = Path(expanded_sumocfg)
        sumocfg = (
            sumocfg_path
            if sumocfg_path.is_absolute()
            else path.parent / sumocfg_path
        ).resolve()
        tls_count, demand_count = validate_sumo_scenario(sumocfg)
        specs.append(
            TrafficSignalScenarioSpec(
                scenario=scenario,
                city_group=city_group,
                suite=suite,
                sumocfg=sumocfg,
                provenance=provenance,
                tls_count=tls_count,
                demand_count=demand_count,
            )
        )
    return TrafficSignalBenchmarkManifest(
        path=path,
        version=version,
        scenarios=tuple(specs),
        protocol=protocol,
    )


def validate_sumo_scenario(sumocfg: Path) -> tuple[int, int]:
    """Require a loadable network, at least one TLS, and non-empty demand."""

    sumocfg = Path(sumocfg)
    if not sumocfg.is_file():
        raise FileNotFoundError(f"manifest SUMO configuration is missing: {sumocfg}")
    root = parse_xml(sumocfg).getroot()
    net_paths = _referenced_paths(root, sumocfg, "net-file")
    route_paths = static_demand_input_paths(sumocfg)
    if len(net_paths) != 1:
        raise ValueError(f"{sumocfg} must reference exactly one net-file")
    if not route_paths:
        raise ValueError(f"{sumocfg} does not reference route demand")
    for input_path in (*net_paths, *route_paths):
        if not input_path.is_file():
            raise FileNotFoundError(f"SUMO input referenced by {sumocfg} is missing: {input_path}")

    tls_count = sum(
        1
        for _, element in iterparse_xml(net_paths[0], events=("end",))
        if element.tag == "tlLogic"
    )
    demand_tags = {"vehicle", "trip", "flow", "person", "personFlow"}
    demand_count = 0
    for route_path in route_paths:
        for _, element in iterparse_xml(route_path, events=("end",)):
            if element.tag in demand_tags:
                demand_count += 1
            element.clear()
    if tls_count <= 0:
        raise ValueError(f"SUMO network has no traffic lights: {net_paths[0]}")
    if demand_count <= 0:
        raise ValueError(f"SUMO route files contain no demand elements: {route_paths}")
    return int(tls_count), int(demand_count)


def _referenced_paths(root: ET.Element, sumocfg: Path, tag: str) -> tuple[Path, ...]:
    result = []
    for item in root.findall(f".//{tag}"):
        for value in str(item.attrib.get("value", "")).split(","):
            value = value.strip()
            if value:
                result.append((sumocfg.parent / value).resolve())
    return tuple(result)
