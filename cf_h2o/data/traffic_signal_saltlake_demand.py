"""Build deterministic Salt Lake City SUMO demand from RESCO detector counts.

The upstream RESCO converter samples departures, rounds fractional movements
stochastically, and assigns UUID vehicle identifiers.  Those choices are useful
for online training but make a frozen cross-network benchmark difficult to
audit.  This module instead builds a robust weekday profile and uses exact,
deterministic integer apportionment for every five-minute detector interval.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence


INTERVAL_SEC = 300
SLOTS_PER_DAY = 24 * 60 * 60 // INTERVAL_SEC
TURN_DIRECTIONS = ("L", "T", "R")

# The scraper omitted its CSV header.  These layouts are the detector-column
# order documented by RESCO's csv_to_flo.py after its unused Total columns are
# removed.
SIGNAL_LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "7142": {
        "E": ("L", "T", "R"),
        "W": ("L", "T", "R"),
        "N": ("L", "T", "TR"),
        "S": ("L", "T", "TR"),
    },
    "7241": {
        "E": ("L", "T", "R"),
        "W": ("L", "T", "R"),
        "N": ("L", "T", "R"),
        "S": ("L", "T", "R"),
    },
    "7242": {
        "E": ("L", "T", "TR"),
        "W": ("L", "T", "TR"),
        "N": ("L", "T", "R"),
        "S": ("L", "T", "R"),
    },
    "7243": {
        "E": ("L", "T", "TR"),
        "W": ("T", "TR"),
        "N": ("L", "TR"),
        "S": ("L", "TR"),
    },
}


@dataclass(frozen=True)
class SaltLakeMapSpec:
    source_name: str
    scenario_name: str
    signals: tuple[str, str]
    network_name: str
    route_template_name: str
    license_name: str = "LICENSE"


MAP_SPECS: dict[str, SaltLakeMapSpec] = {
    "saltlake2_400sX200w": SaltLakeMapSpec(
        source_name="saltlake2_400sX200w",
        scenario_name="saltlake_400s_200w_q1_weekday_peak",
        signals=("7241", "7242"),
        network_name="saltlake2_400sX200w.net.xml",
        route_template_name="saltlake2_400sX200w.flo.xml",
    ),
    "saltlake2_stateXuniversity": SaltLakeMapSpec(
        source_name="saltlake2_stateXuniversity",
        scenario_name="saltlake_state_university_q1_weekday_peak",
        signals=("7243", "7142"),
        network_name="saltlake2_stateXuniversity.net.xml",
        route_template_name="saltlake2_stateXuniversity.flo.xml",
    ),
}


MovementKey = tuple[str, str, str]


@dataclass(frozen=True)
class DailyDetectorCounts:
    signal_id: str
    day: date
    rows: tuple[dict[tuple[str, str], int], ...]


@dataclass(frozen=True)
class RobustMapProfile:
    spec: SaltLakeMapSpec
    counts: Mapping[tuple[int, str, str, str], float]
    included_dates: tuple[date, ...]
    excluded_dates: Mapping[str, str]
    input_hashes: Mapping[str, str]
    statistic: str

    def count(self, slot: int, signal_id: str, direction: str, movement: str) -> float:
        return float(self.counts[(slot, signal_id, direction, movement)])


@dataclass(frozen=True)
class RouteDemandBuild:
    by_slot: tuple[dict[str, int], ...]
    expected_external_by_slot: tuple[int, ...]
    generated_by_slot: tuple[int, ...]
    fallback_allocations: int
    restricted_destination_reallocations: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_unit_interval(key: str) -> float:
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return value / float(1 << 64)


def _round_half_up(value: float) -> int:
    if value < 0:
        raise ValueError(f"demand must be non-negative, got {value}")
    return int(math.floor(value + 0.5))


def _flatten_layout(signal_id: str) -> tuple[tuple[str, str], ...]:
    try:
        layout = SIGNAL_LAYOUTS[signal_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Salt Lake detector signal {signal_id!r}") from exc
    return tuple((direction, movement) for direction in ("E", "W", "N", "S") for movement in layout[direction])


def read_daily_detector_counts(path: Path, signal_id: str, expected_day: date) -> DailyDetectorCounts:
    """Read and strictly validate one headerless detector-count CSV."""

    columns = _flatten_layout(signal_id)
    rows: list[dict[tuple[str, str], int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for slot, raw in enumerate(csv.reader(handle)):
            if len(raw) != len(columns) + 1:
                raise ValueError(
                    f"{path}: row {slot + 1} has {len(raw) - 1} demand columns; expected {len(columns)}"
                )
            try:
                timestamp = datetime.fromisoformat(raw[0])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid timestamp at row {slot + 1}: {raw[0]!r}") from exc
            expected_timestamp = datetime.combine(expected_day, time()) + timedelta(seconds=slot * INTERVAL_SEC)
            if timestamp != expected_timestamp:
                raise ValueError(
                    f"{path}: row {slot + 1} timestamp {timestamp.isoformat(' ')}; "
                    f"expected {expected_timestamp.isoformat(' ')}"
                )
            values: dict[tuple[str, str], int] = {}
            for column, raw_value in zip(columns, raw[1:]):
                try:
                    value = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{path}: non-integer demand {raw_value!r} at row {slot + 1}") from exc
                if value < 0:
                    raise ValueError(f"{path}: negative demand {value} at row {slot + 1}")
                values[column] = value
            rows.append(values)
    if len(rows) != SLOTS_PER_DAY:
        raise ValueError(f"{path}: contains {len(rows)} rows; expected {SLOTS_PER_DAY}")
    return DailyDetectorCounts(signal_id=signal_id, day=expected_day, rows=tuple(rows))


def _iter_dates(start_day: date, end_day: date, weekdays_only: bool) -> Iterable[date]:
    if end_day < start_day:
        raise ValueError("end_day must be on or after start_day")
    current = start_day
    while current <= end_day:
        if not weekdays_only or current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def build_robust_map_profile(
    source_root: Path,
    spec: SaltLakeMapSpec,
    *,
    start_day: date,
    end_day: date,
    weekdays_only: bool = True,
    statistic: str = "median",
    min_complete_days: int = 20,
) -> RobustMapProfile:
    """Aggregate complete daily files into a coordinate-wise robust profile."""

    if statistic != "median":
        raise ValueError(f"unsupported profile statistic {statistic!r}; expected 'median'")
    map_root = source_root / spec.source_name
    complete: dict[date, dict[str, DailyDetectorCounts]] = {}
    excluded: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for day in _iter_dates(start_day, end_day, weekdays_only):
        day_counts: dict[str, DailyDetectorCounts] = {}
        failures: list[str] = []
        day_paths: list[Path] = []
        for signal_id in spec.signals:
            path = map_root / signal_id / f"{day.isoformat()}.csv"
            try:
                counts = read_daily_detector_counts(path, signal_id, day)
            except (FileNotFoundError, ValueError) as exc:
                failures.append(f"{signal_id}: {exc}")
                continue
            day_counts[signal_id] = counts
            day_paths.append(path)
        if failures:
            excluded[day.isoformat()] = "; ".join(failures)
            continue
        complete[day] = day_counts
        for path in day_paths:
            relative = path.relative_to(source_root).as_posix()
            input_hashes[relative] = _sha256_file(path)

    if len(complete) < min_complete_days:
        raise ValueError(
            f"{spec.source_name}: only {len(complete)} complete profile days; "
            f"requires at least {min_complete_days}"
        )

    profile_counts: dict[tuple[int, str, str, str], float] = {}
    ordered_days = tuple(sorted(complete))
    for slot in range(SLOTS_PER_DAY):
        for signal_id in spec.signals:
            for direction, movement in _flatten_layout(signal_id):
                samples = [complete[day][signal_id].rows[slot][(direction, movement)] for day in ordered_days]
                profile_counts[(slot, signal_id, direction, movement)] = float(statistics.median(samples))
    return RobustMapProfile(
        spec=spec,
        counts=profile_counts,
        included_dates=ordered_days,
        excluded_dates=excluded,
        input_hashes=dict(sorted(input_hashes.items())),
        statistic=statistic,
    )


def _detector_role(spec: SaltLakeMapSpec, signal_id: str, direction: str) -> str:
    left_signal, right_signal = spec.signals
    if signal_id == left_signal:
        return "internal_destination" if direction == "W" else "external_origin"
    if signal_id == right_signal:
        return "internal_destination" if direction == "E" else "external_origin"
    raise ValueError(f"signal {signal_id!r} is not part of {spec.source_name}")


def _expanded_turn_weights(profile: RobustMapProfile, slot: int, signal_id: str, direction: str) -> dict[str, float]:
    weights = {turn: 0.0 for turn in TURN_DIRECTIONS}
    for movement in SIGNAL_LAYOUTS[signal_id][direction]:
        count = profile.count(slot, signal_id, direction, movement)
        if movement == "TR":
            weights["T"] += count / 2.0
            weights["R"] += count / 2.0
        else:
            weights[movement] += count
    return weights


def external_arrivals_by_hour(profile: RobustMapProfile) -> tuple[int, ...]:
    totals: list[int] = []
    for hour in range(24):
        total = 0
        for slot in range(hour * 12, (hour + 1) * 12):
            for signal_id in profile.spec.signals:
                for direction, movement in _flatten_layout(signal_id):
                    if _detector_role(profile.spec, signal_id, direction) == "external_origin":
                        total += _round_half_up(profile.count(slot, signal_id, direction, movement))
        totals.append(total)
    return tuple(totals)


def select_pooled_peak_hour(profiles: Sequence[RobustMapProfile]) -> tuple[int, tuple[int, ...]]:
    """Select the earliest aligned hour with maximum pooled external arrivals."""

    if not profiles:
        raise ValueError("at least one profile is required")
    pooled = tuple(sum(external_arrivals_by_hour(profile)[hour] for profile in profiles) for hour in range(24))
    peak_hour = min(hour for hour, value in enumerate(pooled) if value == max(pooled))
    return peak_hour, pooled


def deterministic_apportion(total: int, weights: Mapping[str, float], *, key: str) -> dict[str, int]:
    """Hamilton-apportion an integer total with stable, hash-based tie breaks."""

    if total < 0:
        raise ValueError("total must be non-negative")
    labels = tuple(sorted(weights))
    if not labels:
        if total:
            raise ValueError("cannot apportion positive demand without destinations")
        return {}
    positive = {label: max(0.0, float(weights[label])) for label in labels}
    denominator = sum(positive.values())
    if denominator <= 0.0:
        positive = {label: 1.0 for label in labels}
        denominator = float(len(labels))
    quotas = {label: total * positive[label] / denominator for label in labels}
    allocated = {label: int(math.floor(quotas[label])) for label in labels}
    remaining = total - sum(allocated.values())
    order = sorted(
        labels,
        key=lambda label: (
            -(quotas[label] - allocated[label]),
            hashlib.sha256(f"{key}|{label}".encode("utf-8")).hexdigest(),
        ),
    )
    for label in order[:remaining]:
        allocated[label] += 1
    if sum(allocated.values()) != total:
        raise AssertionError("deterministic apportionment failed to conserve demand")
    return allocated


def parse_route_template(path: Path) -> tuple[ET.ElementTree, dict[str, ET.Element]]:
    tree = ET.parse(path)
    routes = {
        element.attrib["id"]: element
        for element in tree.getroot()
        if element.tag == "route" and "id" in element.attrib
    }
    if not routes:
        raise ValueError(f"{path}: route template contains no routes")
    existing_vehicles = [element for element in tree.getroot() if element.tag == "vehicle"]
    if existing_vehicles:
        raise ValueError(f"{path}: route template unexpectedly contains {len(existing_vehicles)} vehicles")
    return tree, routes


def _destination_weights(
    profile: RobustMapProfile,
    *,
    slot: int,
    origin_signal: str,
    selected_slots: range,
) -> tuple[dict[str, float], bool]:
    left_signal, right_signal = profile.spec.signals
    if origin_signal == left_signal:
        destination_signal, destination_direction = right_signal, "E"
    elif origin_signal == right_signal:
        destination_signal, destination_direction = left_signal, "W"
    else:
        raise ValueError(f"unknown origin signal {origin_signal!r}")
    weights = _expanded_turn_weights(profile, slot, destination_signal, destination_direction)
    if sum(weights.values()) > 0.0:
        return weights, False
    fallback = {turn: 0.0 for turn in TURN_DIRECTIONS}
    for fallback_slot in selected_slots:
        slot_weights = _expanded_turn_weights(profile, fallback_slot, destination_signal, destination_direction)
        for turn in TURN_DIRECTIONS:
            fallback[turn] += slot_weights[turn]
    return fallback, True


def build_route_demand(
    profile: RobustMapProfile,
    route_ids: Iterable[str],
    *,
    start_hour: int,
    duration_sec: int = 3600,
    allocation_seed: int = 20260808,
) -> RouteDemandBuild:
    if not 0 <= start_hour <= 23:
        raise ValueError("start_hour must be in [0, 23]")
    if duration_sec <= 0 or duration_sec % INTERVAL_SEC:
        raise ValueError(f"duration_sec must be a positive multiple of {INTERVAL_SEC}")
    start_slot = start_hour * 3600 // INTERVAL_SEC
    n_slots = duration_sec // INTERVAL_SEC
    selected_slots = range(start_slot, start_slot + n_slots)
    if selected_slots.stop > SLOTS_PER_DAY:
        raise ValueError("selected demand window extends past midnight")

    available_routes = set(route_ids)
    by_slot: list[dict[str, int]] = []
    expected_by_slot: list[int] = []
    generated_by_slot: list[int] = []
    fallback_allocations = 0
    restricted_reallocations = 0

    for local_slot, profile_slot in enumerate(selected_slots):
        demands: dict[str, int] = {}
        expected = 0
        for signal_id in profile.spec.signals:
            for direction, raw_movement in _flatten_layout(signal_id):
                if _detector_role(profile.spec, signal_id, direction) != "external_origin":
                    continue
                count = _round_half_up(profile.count(profile_slot, signal_id, direction, raw_movement))
                expected += count
                if raw_movement == "TR":
                    origin_counts = deterministic_apportion(
                        count,
                        {"T": 0.5, "R": 0.5},
                        key=f"{allocation_seed}|{profile.spec.source_name}|{profile_slot}|{signal_id}|{direction}|TR",
                    )
                else:
                    origin_counts = {raw_movement: count}

                for movement, movement_count in origin_counts.items():
                    if movement_count <= 0:
                        continue
                    direct_route = f"{signal_id}_{direction}{movement}"
                    if direct_route in available_routes:
                        demands[direct_route] = demands.get(direct_route, 0) + movement_count
                        continue

                    crossing_routes = {
                        turn: f"{signal_id}_{direction}{movement}{turn}"
                        for turn in TURN_DIRECTIONS
                        if f"{signal_id}_{direction}{movement}{turn}" in available_routes
                    }
                    if not crossing_routes:
                        raise ValueError(
                            f"no direct or crossing route for {signal_id} {direction}{movement} "
                            f"in {profile.spec.source_name}"
                        )
                    weights, used_fallback = _destination_weights(
                        profile,
                        slot=profile_slot,
                        origin_signal=signal_id,
                        selected_slots=selected_slots,
                    )
                    fallback_allocations += int(used_fallback)
                    unavailable_positive = any(
                        weights[turn] > 0.0 and turn not in crossing_routes for turn in TURN_DIRECTIONS
                    )
                    restricted_reallocations += int(unavailable_positive)
                    allowed_weights = {turn: weights[turn] for turn in crossing_routes}
                    allocation = deterministic_apportion(
                        movement_count,
                        allowed_weights,
                        key=(
                            f"{allocation_seed}|{profile.spec.source_name}|{profile_slot}|"
                            f"{signal_id}|{direction}{movement}|destination"
                        ),
                    )
                    for turn, allocated in allocation.items():
                        route_id = crossing_routes[turn]
                        demands[route_id] = demands.get(route_id, 0) + allocated

        generated = sum(demands.values())
        if generated != expected:
            raise AssertionError(
                f"{profile.spec.source_name} slot {profile_slot}: generated {generated} vehicles; expected {expected}"
            )
        by_slot.append(dict(sorted(demands.items())))
        expected_by_slot.append(expected)
        generated_by_slot.append(generated)

    return RouteDemandBuild(
        by_slot=tuple(by_slot),
        expected_external_by_slot=tuple(expected_by_slot),
        generated_by_slot=tuple(generated_by_slot),
        fallback_allocations=fallback_allocations,
        restricted_destination_reallocations=restricted_reallocations,
    )


def _build_route_xml(
    template_tree: ET.ElementTree,
    route_demand: RouteDemandBuild,
    *,
    scenario_name: str,
    departure_seed: int,
) -> ET.ElementTree:
    template_root = template_tree.getroot()
    root = ET.Element(template_root.tag, dict(template_root.attrib))
    for element in template_root:
        if element.tag in {"vType", "route"}:
            root.append(copy.deepcopy(element))

    vehicles: list[tuple[float, str, ET.Element]] = []
    for local_slot, route_counts in enumerate(route_demand.by_slot):
        interval_start = local_slot * INTERVAL_SEC
        for route_id, count in route_counts.items():
            if count <= 0:
                continue
            phase = _stable_unit_interval(f"{departure_seed}|{scenario_name}|{local_slot}|{route_id}")
            for index in range(count):
                depart = interval_start + ((index + phase) / count) * INTERVAL_SEC
                vehicle_id = f"veh_{scenario_name}_{local_slot:03d}_{route_id}_{index:04d}"
                vehicle = ET.Element(
                    "vehicle",
                    {
                        "id": vehicle_id,
                        "depart": f"{depart:.3f}",
                        "type": "Car",
                        "route": route_id,
                        "departLane": "free",
                        "departSpeed": "max",
                    },
                )
                vehicles.append((depart, vehicle_id, vehicle))
    for _, _, vehicle in sorted(vehicles, key=lambda item: (item[0], item[1])):
        root.append(vehicle)
    ET.indent(root, space="    ")
    return ET.ElementTree(root)


def _write_sumocfg(path: Path, *, network_name: str, route_name: str, duration_sec: int) -> None:
    root = ET.Element(
        "configuration",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "https://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        },
    )
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", {"value": network_name})
    ET.SubElement(inputs, "route-files", {"value": route_name})
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", {"value": "0"})
    ET.SubElement(timing, "end", {"value": str(duration_sec)})
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "time-to-teleport", {"value": "-1"})
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_profile_csv(path: Path, profile: RobustMapProfile, *, start_hour: int, duration_sec: int) -> None:
    start_slot = start_hour * 12
    stop_slot = start_slot + duration_sec // INTERVAL_SEC
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "slot_index",
                "clock_time",
                "signal_id",
                "detector_role",
                "direction",
                "movement",
                f"{profile.statistic}_count",
                "selected_window",
            ]
        )
        for slot in range(SLOTS_PER_DAY):
            clock = str(timedelta(seconds=slot * INTERVAL_SEC))
            for signal_id in profile.spec.signals:
                for direction, movement in _flatten_layout(signal_id):
                    writer.writerow(
                        [
                            slot,
                            clock,
                            signal_id,
                            _detector_role(profile.spec, signal_id, direction),
                            direction,
                            movement,
                            f"{profile.count(slot, signal_id, direction, movement):.6f}",
                            int(start_slot <= slot < stop_slot),
                        ]
                    )


def build_saltlake_benchmark(
    source_root: Path,
    output_root: Path,
    *,
    map_names: Sequence[str] = tuple(MAP_SPECS),
    start_day: date = date(2023, 1, 1),
    end_day: date = date(2023, 3, 31),
    weekdays_only: bool = True,
    statistic: str = "median",
    start_hour: int | None = None,
    duration_sec: int = 3600,
    allocation_seed: int = 20260808,
    departure_seed: int = 20260808,
) -> dict[str, object]:
    """Build all requested maps and return an auditable summary."""

    specs: list[SaltLakeMapSpec] = []
    for name in map_names:
        try:
            specs.append(MAP_SPECS[name])
        except KeyError as exc:
            raise ValueError(f"unsupported Salt Lake map {name!r}") from exc
    profiles = [
        build_robust_map_profile(
            source_root,
            spec,
            start_day=start_day,
            end_day=end_day,
            weekdays_only=weekdays_only,
            statistic=statistic,
        )
        for spec in specs
    ]
    pooled_peak_hour, pooled_hourly_arrivals = select_pooled_peak_hour(profiles)
    selected_hour = pooled_peak_hour if start_hour is None else start_hour
    if not 0 <= selected_hour <= 23:
        raise ValueError("start_hour must be in [0, 23]")

    output_root.mkdir(parents=True, exist_ok=True)
    generator_path = Path(__file__).resolve()
    scenario_summaries: list[dict[str, object]] = []
    for profile in profiles:
        spec = profile.spec
        source_map_root = source_root / spec.source_name
        template_path = source_map_root / spec.route_template_name
        network_path = source_map_root / spec.network_name
        license_path = source_map_root / spec.license_name
        template_tree, routes = parse_route_template(template_path)
        route_demand = build_route_demand(
            profile,
            routes,
            start_hour=selected_hour,
            duration_sec=duration_sec,
            allocation_seed=allocation_seed,
        )

        scenario_root = output_root / spec.scenario_name
        scenario_root.mkdir(parents=True, exist_ok=True)
        output_network = scenario_root / f"{spec.scenario_name}.net.xml"
        output_route = scenario_root / f"{spec.scenario_name}.rou.xml"
        output_sumocfg = scenario_root / f"{spec.scenario_name}.sumocfg"
        output_profile = scenario_root / "movement_profile.csv"
        output_provenance = scenario_root / "provenance.json"
        shutil.copy2(network_path, output_network)
        if license_path.exists():
            shutil.copy2(license_path, scenario_root / spec.license_name)
        route_tree = _build_route_xml(
            template_tree,
            route_demand,
            scenario_name=spec.scenario_name,
            departure_seed=departure_seed,
        )
        route_tree.write(output_route, encoding="utf-8", xml_declaration=True)
        _write_sumocfg(
            output_sumocfg,
            network_name=output_network.name,
            route_name=output_route.name,
            duration_sec=duration_sec,
        )
        _write_profile_csv(output_profile, profile, start_hour=selected_hour, duration_sec=duration_sec)

        hourly_arrivals = external_arrivals_by_hour(profile)
        provenance: dict[str, object] = {
            "schema_version": 1,
            "construction_protocol": "q1-weekday-coordinate-median-pooled-peak-deterministic-apportionment-v1",
            "source_map": spec.source_name,
            "scenario": spec.scenario_name,
            "source_suite": "RESCO benchmark Salt Lake City detector-count environments",
            "source_license": "GNU General Public License v3",
            "profile": {
                "start_date": start_day.isoformat(),
                "end_date": end_day.isoformat(),
                "weekdays_only": weekdays_only,
                "statistic": statistic,
                "included_dates": [day.isoformat() for day in profile.included_dates],
                "excluded_dates": dict(profile.excluded_dates),
            },
            "window": {
                "selection": "pooled_peak_hour" if start_hour is None else "fixed_hour",
                "source_start_hour": selected_hour,
                "source_start_sec": selected_hour * 3600,
                "duration_sec": duration_sec,
                "output_start_sec": 0,
                "pooled_peak_hour": pooled_peak_hour,
                "map_external_arrivals_by_hour": list(hourly_arrivals),
                "pooled_external_arrivals_by_hour": list(pooled_hourly_arrivals),
            },
            "demand_audit": {
                "expected_external_by_interval": list(route_demand.expected_external_by_slot),
                "generated_by_interval": list(route_demand.generated_by_slot),
                "expected_external_total": sum(route_demand.expected_external_by_slot),
                "generated_vehicle_total": sum(route_demand.generated_by_slot),
                "interval_conservation": (
                    route_demand.expected_external_by_slot == route_demand.generated_by_slot
                ),
                "zero_destination_fallback_allocations": route_demand.fallback_allocations,
                "restricted_destination_reallocations": route_demand.restricted_destination_reallocations,
            },
            "seeds": {
                "integer_allocation": allocation_seed,
                "departure_phase": departure_seed,
            },
            "input_files": dict(profile.input_hashes),
            "hashes": {
                "generator_source_sha256": _sha256_file(generator_path),
                "route_template_sha256": _sha256_file(template_path),
                "source_network_sha256": _sha256_file(network_path),
                "output_network_sha256": _sha256_file(output_network),
                "output_route_sha256": _sha256_file(output_route),
                "output_sumocfg_sha256": _sha256_file(output_sumocfg),
                "movement_profile_sha256": _sha256_file(output_profile),
            },
        }
        output_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scenario_summaries.append(
            {
                "scenario": spec.scenario_name,
                "sumocfg": str(output_sumocfg),
                "included_days": len(profile.included_dates),
                "vehicles": sum(route_demand.generated_by_slot),
                "interval_conservation": True,
                "route_sha256": provenance["hashes"]["output_route_sha256"],  # type: ignore[index]
                "provenance": str(output_provenance),
            }
        )

    summary: dict[str, object] = {
        "status": "PASS",
        "protocol": "q1-weekday-coordinate-median-pooled-peak-deterministic-apportionment-v1",
        "selected_hour": selected_hour,
        "pooled_peak_hour": pooled_peak_hour,
        "scenarios": scenario_summaries,
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maps", nargs="+", choices=tuple(MAP_SPECS), default=list(MAP_SPECS))
    parser.add_argument("--start-date", type=_parse_date, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=date(2023, 3, 31))
    parser.add_argument("--weekdays-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-hour", type=int, default=None, help="omit to select the pooled external-arrival peak")
    parser.add_argument("--duration-sec", type=int, default=3600)
    parser.add_argument("--allocation-seed", type=int, default=20260808)
    parser.add_argument("--departure-seed", type=int, default=20260808)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_saltlake_benchmark(
        args.source_root,
        args.output_root,
        map_names=args.maps,
        start_day=args.start_date,
        end_day=args.end_date,
        weekdays_only=args.weekdays_only,
        start_hour=args.start_hour,
        duration_sec=args.duration_sec,
        allocation_seed=args.allocation_seed,
        departure_seed=args.departure_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
