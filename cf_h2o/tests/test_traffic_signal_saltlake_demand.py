from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path

from cf_h2o.data.traffic_signal_saltlake_demand import (
    INTERVAL_SEC,
    MAP_SPECS,
    SIGNAL_LAYOUTS,
    SLOTS_PER_DAY,
    RobustMapProfile,
    _build_route_xml,
    _flatten_layout,
    build_robust_map_profile,
    build_route_demand,
    deterministic_apportion,
    read_daily_detector_counts,
    select_pooled_peak_hour,
)


def _write_detector_day(path: Path, signal_id: str, day: date, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _flatten_layout(signal_id)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for slot in range(SLOTS_PER_DAY):
            timestamp = datetime.combine(day, time()) + timedelta(seconds=slot * INTERVAL_SEC)
            writer.writerow([timestamp.isoformat(" ", timespec="minutes"), *([value] * len(columns))])


def _zero_profile() -> RobustMapProfile:
    spec = MAP_SPECS["saltlake2_400sX200w"]
    counts = {}
    for slot in range(SLOTS_PER_DAY):
        for signal_id in spec.signals:
            for direction, movements in SIGNAL_LAYOUTS[signal_id].items():
                for movement in movements:
                    counts[(slot, signal_id, direction, movement)] = 0.0
    return RobustMapProfile(
        spec=spec,
        counts=counts,
        included_dates=(date(2023, 1, 2),),
        excluded_dates={},
        input_hashes={},
        statistic="median",
    )


def test_read_daily_detector_counts_validates_layout_and_timestamps(tmp_path: Path) -> None:
    day = date(2023, 1, 2)
    path = tmp_path / "7241" / f"{day.isoformat()}.csv"
    _write_detector_day(path, "7241", day, 7)

    result = read_daily_detector_counts(path, "7241", day)

    assert len(result.rows) == SLOTS_PER_DAY
    assert result.rows[0][("E", "L")] == 7
    assert result.rows[-1][("S", "R")] == 7


def test_robust_profile_uses_complete_day_coordinate_median(tmp_path: Path) -> None:
    spec = MAP_SPECS["saltlake2_400sX200w"]
    days = (date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4))
    for value, day in zip((1, 9, 5), days):
        for signal_id in spec.signals:
            _write_detector_day(tmp_path / spec.source_name / signal_id / f"{day.isoformat()}.csv", signal_id, day, value)

    profile = build_robust_map_profile(
        tmp_path,
        spec,
        start_day=days[0],
        end_day=days[-1],
        min_complete_days=3,
    )

    assert profile.included_dates == days
    assert profile.count(0, "7241", "E", "L") == 5.0
    assert len(profile.input_hashes) == 6


def test_deterministic_apportion_is_conservative_and_repeatable() -> None:
    weights = {"L": 1.0, "T": 2.0, "R": 1.0}
    first = deterministic_apportion(11, weights, key="fixed")
    second = deterministic_apportion(11, weights, key="fixed")

    assert first == second
    assert sum(first.values()) == 11
    assert first["T"] >= first["L"]
    assert first["T"] >= first["R"]


def test_route_demand_conserves_external_counts_and_respects_route_support() -> None:
    profile = _zero_profile()
    mutable_counts = dict(profile.counts)
    mutable_counts[(0, "7241", "S", "L")] = 7.0
    mutable_counts[(0, "7241", "S", "T")] = 3.0
    mutable_counts[(0, "7242", "E", "L")] = 1.0
    mutable_counts[(0, "7242", "E", "T")] = 1.0
    mutable_counts[(0, "7242", "E", "TR")] = 4.0
    profile = RobustMapProfile(
        spec=profile.spec,
        counts=mutable_counts,
        included_dates=profile.included_dates,
        excluded_dates={},
        input_hashes={},
        statistic="median",
    )
    routes = {"7241_ST", "7241_SLL", "7241_SLT"}

    result = build_route_demand(profile, routes, start_hour=0, duration_sec=300)

    assert result.expected_external_by_slot == (10,)
    assert result.generated_by_slot == (10,)
    assert sum(result.by_slot[0].values()) == 10
    assert set(result.by_slot[0]) <= routes
    assert result.by_slot[0]["7241_ST"] == 3
    assert result.restricted_destination_reallocations == 1


def test_route_xml_has_stable_ids_sorted_departures() -> None:
    profile = _zero_profile()
    mutable_counts = dict(profile.counts)
    mutable_counts[(0, "7241", "S", "T")] = 4.0
    profile = RobustMapProfile(
        spec=profile.spec,
        counts=mutable_counts,
        included_dates=profile.included_dates,
        excluded_dates={},
        input_hashes={},
        statistic="median",
    )
    demand = build_route_demand(profile, {"7241_ST"}, start_hour=0, duration_sec=300)

    import xml.etree.ElementTree as ET

    root = ET.Element("routes")
    ET.SubElement(root, "vType", {"id": "Car"})
    ET.SubElement(root, "route", {"id": "7241_ST", "edges": "a b"})
    first = _build_route_xml(ET.ElementTree(root), demand, scenario_name="test", departure_seed=17)
    second = _build_route_xml(ET.ElementTree(root), demand, scenario_name="test", departure_seed=17)
    first_vehicles = [element.attrib for element in first.getroot() if element.tag == "vehicle"]
    second_vehicles = [element.attrib for element in second.getroot() if element.tag == "vehicle"]

    assert first_vehicles == second_vehicles
    assert len(first_vehicles) == 4
    assert [float(vehicle["depart"]) for vehicle in first_vehicles] == sorted(
        float(vehicle["depart"]) for vehicle in first_vehicles
    )
    assert len({vehicle["id"] for vehicle in first_vehicles}) == 4


def test_pooled_peak_selection_uses_external_counts_only() -> None:
    profile = _zero_profile()
    mutable_counts = dict(profile.counts)
    mutable_counts[(2 * 12, "7241", "S", "T")] = 10.0
    mutable_counts[(9 * 12, "7241", "W", "T")] = 1000.0
    profile = RobustMapProfile(
        spec=profile.spec,
        counts=mutable_counts,
        included_dates=profile.included_dates,
        excluded_dates={},
        input_hashes={},
        statistic="median",
    )

    peak, totals = select_pooled_peak_hour([profile])

    assert peak == 2
    assert totals[2] == 10
    assert totals[9] == 0
