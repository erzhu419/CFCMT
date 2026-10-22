"""Regressions for source accounting and assets consumed by the v7 bridge."""

import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from experiments.bus_cross_city_native_v1.asset_adapter import (
    HALIFAX_APC,
    build_city_assets,
    select_services,
    write_weekday_timetables,
)


def _bundle(root: Path) -> Path:
    root.mkdir()
    entries = []
    for direction, trips in ((0, 30), (1, 10)):
        key = f"1_D{direction}"
        directory = root / "data" / key
        directory.mkdir(parents=True)
        names = [f"source{direction}_{index}" for index in range(3)]
        pd.DataFrame(dict(stop_name=names, gtfs_stop_id=[10, 11, 12])).to_excel(
            directory / "stop_news.xlsx", index=False)
        route = pd.DataFrame(dict(start_stop=names[:-1], end_stop=names[1:], distance=[100, 200]))
        for hour in range(24):
            route[f"{hour:02d}:00:00"] = [hour + 1.0, hour + 2.0]
        route.to_excel(directory / "route_news.xlsx", index=False)
        pd.DataFrame(dict(launch_time=[6 * 3600, 7 * 3600, 7 * 3600, 8 * 3600, 11 * 3600],
                          direction=[direction] * 5)).to_excel(directory / "time_table.xlsx", index=False)
        od = []
        for hour in range(7, 11):
            for index, name in enumerate(names):
                values = {destination: 0.0 for destination in names}
                values[names[2]] = [10.0, 5.0, 0.0][index]
                od.append(dict(time_period=f"{hour:02d}:00:00", stop_name=name, **values))
        pd.DataFrame(od).to_excel(directory / "passenger_OD.xlsx", index=False)
        entries.append(dict(line_key=key, route_id="1", direction=direction, pattern_index=0,
                            pattern_trips=trips, pattern_share=1.0, demand_source=HALIFAX_APC))
    # Proxy ridership is not eligible, even if its metadata would sort first.
    entries.insert(0, dict(entries[0], line_key="0_D0", route_id="0", demand_source="schedule_headway_proxy_od"))
    manifest = root / "gtfs_city_manifest.json"
    manifest.write_text(json.dumps(dict(city="HalifaxTransit", lines=entries)))
    return manifest


def _gtfs_fixture(root: Path) -> Path:
    root.mkdir()
    trips, times = [], []
    for direction in (0, 1):
        for suffix, service, departure in (("w7", "W", "07:00:00"), ("w8", "W", "08:00:00"),
                                            ("s7", "S", "07:00:00")):
            trip = f"d{direction}_{suffix}"
            trips.append(dict(route_id="1", service_id=service, trip_id=trip, direction_id=direction))
            times.extend(dict(trip_id=trip, stop_id=stop, stop_sequence=index + 1,
                              departure_time=departure, arrival_time=departure)
                         for index, stop in enumerate((10, 11, 12)))
    pd.DataFrame(trips).to_csv(root / "trips.txt", index=False)
    pd.DataFrame(times).to_csv(root / "stop_times.txt", index=False)
    week = dict(monday=1, tuesday=1, wednesday=1, thursday=1, friday=1, saturday=0, sunday=0)
    sunday = dict(monday=0, tuesday=0, wednesday=0, thursday=0, friday=0, saturday=0, sunday=1)
    pd.DataFrame([dict(service_id="W", start_date="20260518", end_date="20260524", **week),
                  dict(service_id="S", start_date="20260518", end_date="20260524", **sunday)]).to_csv(
        root / "calendar.txt", index=False)
    pd.DataFrame([dict(service_id="W", date="20260518", exception_type=2),
                  dict(service_id="S", date="20260518", exception_type=1)]).to_csv(
        root / "calendar_dates.txt", index=False)
    return root


def test_subset_uses_provenance_and_is_independent_of_manifest_order(tmp_path):
    path = _bundle(tmp_path / "bundle")
    first = select_services(path, max_services=2)
    manifest = json.loads(path.read_text())
    manifest["lines"].reverse()
    path.write_text(json.dumps(manifest))
    second = select_services(path, max_services=2)
    assert first["selected_services"] == second["selected_services"]
    assert [line["line_key"] for line in first["selected_services"]] == ["1_D0", "1_D1"]
    assert [line["window_departures"] for line in first["selected_services"]] == [3, 3]


def test_route_apc_totals_are_conserved_across_directions(tmp_path):
    path = _bundle(tmp_path / "bundle")
    selection = select_services(path, max_services=2)
    directions = selection["selected_services"]
    # The upstream OD table repeats the same route's 60 passenger total in
    # both directions. It must become 45+15, not 60+60.
    assert [line["route_direction_demand_share"] for line in directions] == [.75, .25]
    assert [line["expected_window_passengers"] for line in directions] == [45., 15.]
    assert sum(line["expected_window_passengers"] for line in directions) == 60.


@pytest.mark.skipif(shutil.which("netconvert") is None or shutil.which("sumo") is None,
                   reason="requires the existing SUMO/netconvert binaries")
def test_bridge_assets_keep_terminal_people_and_relative_clock(tmp_path):
    source = _bundle(tmp_path / "bundle")
    out = tmp_path / "native"
    timetable_path = tmp_path / "weekday_timetables.json"
    write_weekday_timetables(source, _gtfs_fixture(tmp_path / "gtfs"), timetable_path, max_services=2)
    manifest = build_city_assets(source, out, weekday_timetables=timetable_path, max_services=2, demand_seed=72101)
    assert manifest["line_ids"] == ["100X", "101S"]
    assert manifest["station_max_by_line"] == [2, 2]
    assert manifest["line_id_to_canonical_index"] == {"100X": 0, "101S": 1}
    assert manifest["scheduled_service_date"] == "2026-05-19"
    assert {trip["gtfs_trip_id"] for line in manifest["lines"] for trip in line["trip_schedule"]} == {
        "d0_w7", "d0_w8", "d1_w7", "d1_w8"}
    objects = out / "online_control/initialize_obj"
    lines = ET.parse(objects / "save_obj_line.add.xml").getroot()
    for line in lines:
        stops = line.attrib["stop_id_l"].split()
        assert len(stops) == 3  # Original generator omitted the final destination.
        periods = next(iter(line.find("edge_average_speed").attrib.values())).split()
        assert list(map(float, periods[:2])) == [8., 9.]  # Source hours 07/08, native periods 0/1.
        assert len(periods) == 7
    departures = [float(bus.get("start_time_n")) for bus in
                  ET.parse(objects / "save_obj_bus.add.xml").getroot()]
    assert sorted(departures) == [0., 0., 3600., 3600.]
    people = list(ET.parse(out / "routes/persons.rou.xml").getroot().findall("person"))
    passenger_objects = list(ET.parse(objects / "save_obj_passenger.add.xml").getroot())
    assert len(people) == len(passenger_objects) == manifest["person_count"] > 0
    assert {person.get("id") for person in people} == {person.get("passenger_id_s") for person in passenger_objects}
    for person in people:
        ride = person.find("ride")
        assert ride.get("busStop") in {"100X003", "101S003"}
        assert 0 <= float(person.get("depart")) < 14400
    bus_routes = ET.parse(out / "routes/buses.rou.xml").getroot().findall("vehicle")
    assert all(len(bus.findall("stop")) == 3 for bus in bus_routes)
    assert (out / "network_data/net.net.xml").is_file()
    network_logics = ET.parse(out / "network_data/net.net.xml").getroot().findall("tlLogic")
    assert len(network_logics) == 2
    assert all([phase.get("state") for phase in logic] == ["G"] for logic in network_logics)
    assert not ET.parse(out / "network_data/stops.add.xml").getroot().findall("tlLogic")
    # The duplicate-program bug passed netconvert but failed SUMO at reset.
    # Launch the generated fixture to exercise the same combined asset load.
    startup = subprocess.run(["sumo", "-c", str(out / "online_control/simulation.sumocfg"),
                              "--end", "10", "--no-step-log", "true",
                              "--duration-log.disable", "true"],
                             capture_output=True, text=True)
    assert startup.returncode == 0, startup.stdout + startup.stderr


def test_more_than_twelve_services_cannot_alias_to_last_actor_identity(tmp_path):
    source = _bundle(tmp_path / "bundle")
    with pytest.raises(ValueError, match="at most 12"):
        select_services(source, max_services=13)


def test_weekday_calendar_removes_mutually_exclusive_services_not_just_same_times(tmp_path):
    source = _bundle(tmp_path / "bundle")
    result = write_weekday_timetables(source, _gtfs_fixture(tmp_path / "gtfs"),
                                     tmp_path / "weekday.json", max_services=2)
    assert result["service_date"] == "2026-05-19"  # Feed start is a holiday Monday.
    assert result["active_service_ids"] == ["W"]
    assert [line["all_feed_window_departures"] for line in result["selected_services"]] == [3, 3]
    assert [[trip["launch_time"] for trip in line["trips"]] for line in result["selected_services"]] == [
        [25200, 28800], [25200, 28800]]
    assert all(trip["gtfs_service_id"] == "W" for line in result["selected_services"] for trip in line["trips"])


def test_genuine_same_day_departures_are_not_silently_merged(tmp_path):
    source = _bundle(tmp_path / "bundle")
    gtfs = _gtfs_fixture(tmp_path / "gtfs")
    trips = pd.read_csv(gtfs / "trips.txt")
    trips.loc[len(trips)] = [1, "W", "second_actual_0700", 0]
    trips.to_csv(gtfs / "trips.txt", index=False)
    times = pd.read_csv(gtfs / "stop_times.txt")
    extra = times[times.trip_id == "d0_w7"].copy()
    extra["trip_id"] = "second_actual_0700"
    pd.concat([times, extra]).to_csv(gtfs / "stop_times.txt", index=False)
    with pytest.raises(ValueError, match="genuine same-day simultaneous"):
        write_weekday_timetables(source, gtfs, tmp_path / "weekday.json", max_services=2)
