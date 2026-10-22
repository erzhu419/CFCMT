"""Build a preselected GTFS/APC corridor subset for the frozen transit-v7 bridge.

The Halifax input supplies observed route/hour demand totals, not observed OD
destinations or a road traffic network.  OD/person realizations and separate
bus corridors below are generated inputs and remain labelled as such.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


LINE_FILES = ("stop_news.xlsx", "route_news.xlsx", "time_table.xlsx", "passenger_OD.xlsx")
HALIFAX_APC = "halifax_route_level_half_hour_apc_proxy_od"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _line_directory(manifest: Path, line: dict[str, Any]) -> Path:
    # The source manifests contain workstation-specific absolute output paths.
    # Both the city bundle and the compact server upload use this directory.
    return manifest.parent / "data" / str(line["line_key"])


def _departures(timetable: pd.DataFrame, start_hour: int, service_hours: int) -> np.ndarray:
    values = pd.to_numeric(timetable["launch_time"], errors="coerce").to_numpy(float)
    start = start_hour * 3600
    return np.sort(values[np.isfinite(values) & (values >= start) &
                          (values < start + service_hours * 3600)]) - start


def _period_hours(passenger_od: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(passenger_od["time_period"].astype(str).str.slice(0, 2), errors="coerce")


def select_services(city_manifest: str | Path, *, start_hour: int = 7,
                    service_hours: int = 4, max_services: int = 12) -> dict[str, Any]:
    """Select by input availability, then fixed IDs; no policy results are read."""
    manifest_path = Path(city_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not 1 <= max_services <= 12:
        raise ValueError("the frozen actor supports at most 12 directional services")
    if not 0 <= start_hour < 24 or not 1 <= service_hours <= 5 or start_hour + service_hours > 24:
        raise ValueError("the frozen passenger-rate table supports a window of 1..5 same-day hours")
    ordered = sorted(manifest["lines"], key=lambda line: (
        str(line["route_id"]), int(line["direction"]), int(line["pattern_index"]), str(line["line_key"])))
    direction_trips: dict[tuple[str, int], float] = {}
    for line in manifest["lines"]:
        key = (str(line["route_id"]), int(line["direction"]))
        direction_trips[key] = direction_trips.get(key, 0.0) + float(line["pattern_trips"])
    selected, excluded = [], []
    for line in ordered:
        key = str(line["line_key"])
        if line.get("demand_source") != HALIFAX_APC:
            excluded.append(dict(line_key=key, reason="no observed Halifax route/hour APC totals"))
            continue
        directory = _line_directory(manifest_path, line)
        missing = [name for name in LINE_FILES if not (directory / name).is_file()]
        if missing:
            excluded.append(dict(line_key=key, reason="missing route inputs", missing=missing))
            continue
        timetable = pd.read_excel(directory / "time_table.xlsx")
        departures = _departures(timetable, start_hour, service_hours)
        if len(departures) < 2:
            excluded.append(dict(line_key=key, reason="fewer than two window departures"))
            continue
        od = pd.read_excel(directory / "passenger_OD.xlsx")
        hours = _period_hours(od)
        window = od.loc[(hours >= start_hour) & (hours < start_hour + service_hours)]
        total = float(window.drop(columns=["time_period", "stop_name"]).sum().sum())
        if total <= 0:
            excluded.append(dict(line_key=key, reason="no positive APC demand in the fixed window"))
            continue
        route_id = str(line["route_id"])
        route_trips = sum(value for (route, _), value in direction_trips.items() if route == route_id)
        share = float(line.get("route_direction_demand_share",
                               direction_trips[(route_id, int(line["direction"]))] / route_trips))
        selected.append(dict(line, directory=str(directory), window_departures=len(departures),
                             source_od_expected_window_passengers=total,
                             route_direction_demand_share=share,
                             expected_window_passengers=total * share))
        if len(selected) == max_services:
            break
    if not selected:
        raise ValueError("no directional service has the required fixed-window inputs")
    return dict(schema="bus-cross-city-native-selection-v1", city=manifest["city"],
                original_manifest=manifest.get("original_manifest", str(manifest_path)),
                start_hour=start_hour, service_hours=service_hours,
                max_services=max_services,
                original_service_count=manifest.get("original_service_count", len(manifest["lines"])),
                selection_rule="APC provenance, all four input tables, >=2 window departures, "
                               "positive window demand; sort (route_id string, direction, pattern_index, line_key)",
                selected_services=selected, excluded_before_subset_filled=excluded)


def write_selection(city_manifest: str | Path, out_dir: str | Path, **kwargs) -> dict[str, Any]:
    selection = select_services(city_manifest, **kwargs)
    out = Path(out_dir).resolve()
    _write_json(out / "selection_summary.json", selection)
    _write_json(out / "selected_manifest.json", dict(
        city=selection["city"], original_manifest=selection["original_manifest"],
        selection_rule=selection["selection_rule"], start_hour=selection["start_hour"],
        service_hours=selection["service_hours"], original_service_count=selection["original_service_count"],
        lines=selection["selected_services"]))
    return selection


def _read_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_excel(directory / name) for name in LINE_FILES)


def _hourly_speeds(route: pd.DataFrame, start_hour: int, episode_seconds: int) -> list[list[float]]:
    period_count = math.ceil(episode_seconds / 3600)
    result = []
    for _, row in route.iterrows():
        values = [float(row[f"{min(start_hour + period, 23):02d}:00:00"])
                  for period in range(period_count)]
        if any(not math.isfinite(speed) or speed <= 0 for speed in values):
            raise ValueError("selected route has a nonpositive/nonfinite scheduled segment speed")
        result.append(values)
    return result


def _native_id(index: int, direction: int) -> str:
    return f"{100 + index}{'S' if direction else 'X'}"


def write_weekday_timetables(city_manifest: str | Path, gtfs_dir: str | Path,
                            out_path: str | Path, *, start_hour: int = 7,
                            service_hours: int = 4, max_services: int = 12) -> dict[str, Any]:
    """Recover one actual weekday's trips for the already selected services.

    Original all-feed XLSX timetables discarded trip/calendar identity and
    combined different days. This small output is staged instead of full GTFS.
    """
    selection = select_services(city_manifest, start_hour=start_hour,
                                service_hours=service_hours, max_services=max_services)
    gtfs = Path(gtfs_dir).resolve()
    calendar = pd.read_csv(gtfs / "calendar.txt", dtype=str)
    exceptions = pd.read_csv(gtfs / "calendar_dates.txt", dtype=str)
    first_day = dt.datetime.strptime(calendar["start_date"].min(), "%Y%m%d").date()
    last_day = dt.datetime.strptime(calendar["end_date"].max(), "%Y%m%d").date()
    candidate = first_day
    while candidate <= last_day:
        day = candidate.strftime("%Y%m%d")
        weekday = candidate.strftime("%A").lower()
        regularly_active = set(calendar.loc[(calendar.start_date <= day) & (calendar.end_date >= day) &
                                           (calendar[weekday] == "1"), "service_id"])
        date_exceptions = exceptions.loc[exceptions.date == day]
        removed = set(date_exceptions.loc[date_exceptions.exception_type == "2", "service_id"])
        if candidate.weekday() < 5 and regularly_active and not (regularly_active & removed):
            active = regularly_active | set(date_exceptions.loc[
                date_exceptions.exception_type == "1", "service_id"])
            break
        candidate += dt.timedelta(days=1)
    else:
        raise ValueError("no nonholiday weekday exists in the supplied GTFS calendar")

    trips = pd.read_csv(gtfs / "trips.txt", dtype=str)
    stop_times = pd.read_csv(gtfs / "stop_times.txt", dtype=str,
                            usecols=["trip_id", "stop_id", "stop_sequence", "departure_time", "arrival_time"])
    stop_times["_sequence"] = pd.to_numeric(stop_times["stop_sequence"])
    stop_times = stop_times.sort_values(["trip_id", "_sequence"])
    trip_stops = stop_times.groupby("trip_id").agg(
        signature=("stop_id", tuple), departure=("departure_time", "first"), arrival=("arrival_time", "first"))
    trips = trips.join(trip_stops, on="trip_id")
    def seconds(value):
        hours, minutes, seconds = map(int, str(value).strip().split(":"))
        return hours * 3600 + minutes * 60 + seconds
    trips["launch_time"] = [seconds(departure if isinstance(departure, str) and departure.strip() else arrival)
                             for departure, arrival in zip(trips.departure, trips.arrival)]
    records = []
    for line in selection["selected_services"]:
        stops = pd.read_excel(Path(line["directory"]) / "stop_news.xlsx")
        signature = tuple(stops["gtfs_stop_id"].astype(str))
        matched = trips.loc[(trips.route_id == str(line["route_id"])) &
                            (trips.direction_id == str(line["direction"])) &
                            trips.signature.map(lambda values: values == signature)]
        matched_window = matched.loc[(matched.launch_time >= start_hour * 3600) &
                                     (matched.launch_time < (start_hour + service_hours) * 3600)]
        chosen = matched_window.loc[matched_window.service_id.isin(active)].sort_values(["launch_time", "trip_id"])
        if len(chosen) < 2:
            raise ValueError(f"fixed subset {line['line_key']} has fewer than two trips on {candidate}; do not reselect")
        if chosen.launch_time.duplicated().any():
            raise ValueError(f"{line['line_key']} has genuine same-day simultaneous departures; do not deduplicate")
        records.append(dict(line_key=line["line_key"], route_id=str(line["route_id"]),
                            direction=int(line["direction"]), stop_signature=list(signature),
                            all_feed_window_departures=len(matched_window),
                            trips=[dict(gtfs_trip_id=row.trip_id, gtfs_service_id=row.service_id,
                                        launch_time=int(row.launch_time), departure_time=row.departure.strip())
                                   for row in chosen.itertuples()]))
    result = dict(schema="bus-cross-city-native-weekday-timetables-v1", service_date=candidate.isoformat(),
                  calendar_rule="earliest weekday in feed validity with no exception removing an otherwise active weekday service",
                  active_service_ids=sorted(active), start_hour=start_hour, service_hours=service_hours,
                  gtfs_source_directory=str(gtfs), original_manifest=selection["original_manifest"],
                  calendar_rows=calendar.to_dict("records"), date_exception_rows=date_exceptions.to_dict("records"),
                  selected_services=records)
    _write_json(Path(out_path), result)
    return result


def build_city_assets(city_manifest: str | Path, out_dir: str | Path, *,
                      weekday_timetables: str | Path,
                      start_hour: int = 7, service_hours: int = 4,
                      episode_seconds: int = 24000, max_services: int = 12,
                      demand_seed: int = 72101, netconvert: str = "netconvert") -> dict[str, Any]:
    """Generate native bridge assets without touching the original city bundle."""
    if episode_seconds <= service_hours * 3600:
        raise ValueError("episode_seconds must include time after the service window")
    source_path, out = Path(city_manifest).resolve(), Path(out_dir).resolve()
    selection = write_selection(source_path, out, start_hour=start_hour,
                                service_hours=service_hours, max_services=max_services)
    weekday = json.loads(Path(weekday_timetables).read_text(encoding="utf-8"))
    if weekday["start_hour"] != start_hour or weekday["service_hours"] != service_hours:
        raise ValueError("weekday timetable and fixed service window disagree")
    weekday_lines = {line["line_key"]: line for line in weekday["selected_services"]}
    if set(weekday_lines) != {line["line_key"] for line in selection["selected_services"]}:
        raise ValueError("weekday timetable does not describe the fixed selected subset")
    nodes, edges, connections = (ET.Element(name) for name in ("nodes", "edges", "connections"))
    additional, sorted_edges = ET.Element("additional"), ET.Element("buslines")
    object_roots = {kind: ET.Element("objects") for kind in
                    ("lane", "stop", "signal", "line", "bus", "passenger")}
    buses, persons = ET.Element("routes"), ET.Element("routes")
    ET.SubElement(buses, "vType", id="Bus", vClass="bus", length="12", personCapacity="50",
                  maxSpeed="25", boardingDuration="2.5", loadingDuration="1.5", speedFactor="1")
    ET.SubElement(persons, "vType", id="Passenger", vClass="pedestrian")
    rng = np.random.default_rng(demand_seed)
    line_records, bus_records, person_records = [], [], []
    n_periods = math.ceil(episode_seconds / 3600)

    for index, source in enumerate(selection["selected_services"]):
        stops, route, timetable, od = _read_tables(_line_directory(source_path, source))
        names = stops["stop_name"].astype(str).tolist()
        if len(names) < 2 or len(route) != len(names) - 1:
            raise ValueError(f"{source['line_key']}: route must connect every stop including the terminus")
        if route["start_stop"].astype(str).tolist() != names[:-1] or \
                route["end_stop"].astype(str).tolist() != names[1:]:
            raise ValueError(f"{source['line_key']}: route and stop order disagree")
        lengths = [30.0] + pd.to_numeric(route["distance"], errors="coerce").astype(float).tolist()
        if any(not math.isfinite(length) or length < 30.0 for length in lengths):
            raise ValueError(f"{source['line_key']}: a segment cannot contain a 25 m bus stop")
        segment_speeds = _hourly_speeds(route, start_hour, episode_seconds)
        speeds = [segment_speeds[0]] + segment_speeds
        native = _native_id(index, int(source["direction"]))
        stop_ids = [f"{native}{stop + 1:03d}" for stop in range(len(names))]
        edge_ids = [f"E{native}{stop + 1:03d}" for stop in range(len(names))]
        exit_edge, signal_id = f"E{native}EXIT", f"TLS{native}"
        all_edges = edge_ids + [exit_edge]
        all_lengths, all_speeds = lengths + [30.0], speeds + [[8.0] * n_periods]
        dated_line = weekday_lines[source["line_key"]]
        if dated_line["stop_signature"] != stops["gtfs_stop_id"].astype(str).tolist():
            raise ValueError("weekday trip stop signature and city stop table disagree")
        dated_trips = sorted(dated_line["trips"], key=lambda trip: (trip["launch_time"], trip["gtfs_trip_id"]))
        departures = np.asarray([trip["launch_time"] - start_hour * 3600 for trip in dated_trips], dtype=float)
        if len(departures) < 2 or np.any(np.diff(departures) <= 0):
            raise ValueError("weekday timetable must contain >=2 strictly ordered actual service departures")
        if np.any(departures < 0) or np.any(departures >= service_hours * 3600):
            raise ValueError("weekday timetable contains a trip outside the fixed window")
        y, x = float(index * 100), 0.0
        ET.SubElement(nodes, "node", id=f"N{native}000", x="0", y=str(y), type="priority")
        for edge_index, (edge_id, length, periods) in enumerate(zip(all_edges, all_lengths, all_speeds)):
            x += length
            node_id = f"N{native}{edge_index + 1:03d}"
            # Frozen Bus.bus_activate requires a next TLS. It is beyond every
            # passenger destination and always green, so it adds no signal delay.
            node_type = "traffic_light" if edge_index == len(edge_ids) - 1 else "priority"
            node_attrs = dict(id=node_id, x=f"{x:.6f}", y=str(y), type=node_type)
            if node_type == "traffic_light":
                node_attrs["tl"] = signal_id
            ET.SubElement(nodes, "node", **node_attrs)
            ET.SubElement(edges, "edge", id=edge_id, **{
                "from": f"N{native}{edge_index:03d}", "to": node_id,
                "length": f"{length:.6f}", "speed": f"{float(np.median(periods[:service_hours])):.6f}",
                "numLanes": "1", "allow": "bus pedestrian", "priority": "1"})
            ET.SubElement(object_roots["lane"], "lane_obj", lane_id_s=f"{edge_id}_0",
                          length_n=f"{length:.6f}", from_junction_s=f"N{native}{edge_index:03d}",
                          to_junction_s=node_id)
            if edge_index:
                attrs = {"from": all_edges[edge_index - 1], "to": edge_id,
                         "fromLane": "0", "toLane": "0"}
                if edge_id == exit_edge:
                    attrs.update(tl=signal_id, linkIndex="0")
                ET.SubElement(connections, "connection", **attrs)
        for stop, edge, length in zip(stop_ids, edge_ids, lengths):
            ET.SubElement(additional, "busStop", id=stop, lane=f"{edge}_0", lines=native,
                          startPos=f"{length - 30:.6f}", endPos=f"{length - 5:.6f}", personCapacity="120")
            ET.SubElement(object_roots["stop"], "stop_obj", stop_id_s=stop,
                          at_edge_s=edge, at_lane_s=f"{edge}_0", service_line_l=native)

        signal = ET.SubElement(object_roots["signal"], "signal_obj", signal_id_s=signal_id)
        incoming_lane, outgoing_lane = f"{edge_ids[-1]}_0", f"{exit_edge}_0"
        ET.SubElement(signal, "in_lane", **{incoming_lane: incoming_lane})
        ET.SubElement(signal, "out_lane", **{outgoing_lane: outgoing_lane})
        ET.SubElement(signal, "detector")
        ET.SubElement(signal, "connection", c_0=f"{incoming_lane} {outgoing_lane} 0 0")
        ET.SubElement(signal, "phase", p_0=f"{incoming_lane};{outgoing_lane};0;0")

        line = ET.SubElement(object_roots["line"], "line_obj", line_id_s=native,
                             edge_id_l=" ".join(all_edges), stop_id_l=" ".join(stop_ids), signal_id_l=signal_id)
        ET.SubElement(line, "edge_length", **{edge: f"{length:.6f}" for edge, length in zip(all_edges, all_lengths)})
        ET.SubElement(line, "edge_average_speed", **{
            edge: " ".join(f"{value:.6f}" for value in periods) for edge, periods in zip(all_edges, all_speeds)})
        ET.SubElement(line, "distance_between_stop", **{f"d_{stop}": f"{length:.6f}"
                      for stop, length in zip(stop_ids, lengths)})
        ET.SubElement(line, "distance_between_signal", **{f"d_{signal_id}": f"{sum(lengths):.6f}"})
        ET.SubElement(line, "edge_between_stop", **{f"e_{stop}": edge for stop, edge in zip(stop_ids, edge_ids)})
        ET.SubElement(line, "edge_between_signal", **{f"e_{signal_id}": " ".join(edge_ids)})
        edge_map = ET.SubElement(sorted_edges, "busline", id=native)
        for edge, stop, length in zip(edge_ids, stop_ids, lengths):
            ET.SubElement(edge_map, "element", id=edge, type="s", length=f"{length:.6f}",
                          dedicated_flag="1", stop_id=stop)
        ET.SubElement(edge_map, "element", id=exit_edge, type="w", length="30", dedicated_flag="1",
                      impacted_edge="", impacted_length="0")

        trip_schedule = []
        for departure_index, (departure, original_trip) in enumerate(zip(departures, dated_trips)):
            bus_id = f"{native}_{departure_index + 1:06d}"
            bus_records.append(dict(id=bus_id, line_id=native, depart=float(departure), edges=all_edges, stops=stop_ids))
            ET.SubElement(object_roots["bus"], "bus_obj", bus_id_s=bus_id,
                          start_time_n=f"{float(departure):.6f}", belong_line_id_s=native)
            trip_schedule.append(dict(native_bus_id=bus_id, **original_trip))

        by_name = {name: position for position, name in enumerate(names)}
        hourly = _period_hours(od)
        od_window = od.loc[(hourly >= start_hour) & (hourly < start_hour + service_hours)]
        generated = 0
        for _, row in od_window.iterrows():
            origin = by_name[str(row["stop_name"])]
            hour = int(str(row["time_period"])[:2]) - start_hour
            for destination, destination_name in enumerate(names):
                expected = float(row[destination_name]) * float(source["route_direction_demand_share"])
                if not math.isfinite(expected) or expected < 0:
                    raise ValueError("OD demand must be finite and nonnegative")
                if expected > 0 and destination <= origin:
                    raise ValueError("OD table contains a non-downstream passenger trip")
                count = int(rng.poisson(expected))
                for departure in sorted(hour * 3600 + rng.uniform(0, 3600, count)):
                    generated += 1
                    person_records.append(dict(id=f"P{native}_{generated:07d}", line_id=native,
                        depart=float(departure), origin_stop=stop_ids[origin], dest_stop=stop_ids[destination],
                        origin_edge=edge_ids[origin], dest_edge=edge_ids[destination],
                        depart_pos=lengths[origin] - 30.0))
        line_records.append(dict(source_line_key=source["line_key"], native_line_id=native, canonical_line_index=index,
                                 station_max=len(stop_ids) - 1, stop_ids=stop_ids,
                                 original_stop_names=names, original_gtfs_stop_ids=stops["gtfs_stop_id"].astype(str).tolist(),
                                 departures_seconds=departures.tolist(), departure_count=len(departures),
                                 trip_schedule=trip_schedule,
                                 person_count=generated, route_id=str(source["route_id"]), direction=int(source["direction"]),
                                 demand_source=source["demand_source"], pattern_share=float(source["pattern_share"]),
                                 route_direction_demand_share=float(source["route_direction_demand_share"]),
                                 expected_window_passengers=float(source["expected_window_passengers"]),
                                 source_od_expected_window_passengers=float(source["source_od_expected_window_passengers"])))

    for record in sorted(bus_records, key=lambda record: (record["depart"], record["id"])):
        vehicle = ET.SubElement(buses, "vehicle", id=record["id"], type="Bus",
                                depart=f"{record['depart']:.6f}", line=record["line_id"])
        ET.SubElement(vehicle, "route", edges=" ".join(record["edges"]))
        for stop in record["stops"]:
            ET.SubElement(vehicle, "stop", busStop=stop, duration="4.5", line=record["line_id"])
    for record in sorted(person_records, key=lambda record: (record["depart"], record["id"])):
        person = ET.SubElement(persons, "person", id=record["id"], type="Passenger",
                               depart=f"{record['depart']:.6f}", departPos=f"{record['depart_pos']:.6f}")
        ET.SubElement(person, "walk", edges=record["origin_edge"], busStop=record["origin_stop"])
        ET.SubElement(person, "ride", **{"from": record["origin_edge"], "to": record["dest_edge"],
                      "busStop": record["dest_stop"], "lines": record["line_id"]})
        ET.SubElement(object_roots["passenger"], "passenger_obj", passenger_id_s=record["id"],
                      start_time_n=f"{record['depart']:.6f}", start_stop_id_s=record["origin_stop"],
                      transfer_stop_id_s="", end_stop_id_s=record["dest_stop"],
                      start_edge_id_s=record["origin_edge"], transfer_edge_id_s="", end_edge_id_s=record["dest_edge"])

    for name, tree in (("nodes.nod.xml", nodes), ("edges.edg.xml", edges),
                       ("connections.con.xml", connections), ("stops.add.xml", additional),
                       ("a_sorted_busline_edge.xml", sorted_edges)):
        _write_xml(out / "network_data" / name, tree)
    for kind, tree in object_roots.items():
        _write_xml(out / "online_control" / "initialize_obj" / f"save_obj_{kind}.add.xml", tree)
    _write_xml(out / "routes" / "buses.rou.xml", buses)
    _write_xml(out / "routes" / "persons.rou.xml", persons)
    cmd = [netconvert, "--node-files", str(out / "network_data/nodes.nod.xml"),
           "--edge-files", str(out / "network_data/edges.edg.xml"), "--connection-files",
           str(out / "network_data/connections.con.xml"), "--output-file", str(out / "network_data/net.net.xml"),
           "--no-internal-links", "true", "--no-turnarounds", "true"]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (out / "netconvert.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"netconvert failed; see {out / 'netconvert.log'}")
    # netconvert already writes program 0 for each traffic-light node. Make
    # that unique network program constant green instead of redefining it in
    # the additional file, which SUMO rejects as a duplicate (TLS, programID).
    net_path = out / "network_data/net.net.xml"
    network = ET.parse(net_path)
    for logic in network.getroot().findall("tlLogic"):
        for phase in list(logic):
            logic.remove(phase)
        ET.SubElement(logic, "phase", duration="60", state="G")
    ET.indent(network, space="  ")
    network.write(net_path, encoding="utf-8", xml_declaration=True)
    cfg = ET.Element("configuration")
    input_xml = ET.SubElement(cfg, "input")
    ET.SubElement(input_xml, "net-file", value="../network_data/net.net.xml")
    ET.SubElement(input_xml, "additional-files", value="../network_data/stops.add.xml")
    ET.SubElement(input_xml, "route-files", value="../routes/buses.rou.xml,../routes/persons.rou.xml")
    time_xml = ET.SubElement(cfg, "time")
    ET.SubElement(time_xml, "begin", value="0")
    ET.SubElement(time_xml, "end", value=str(episode_seconds))
    ET.SubElement(time_xml, "step-length", value="1")
    processing = ET.SubElement(cfg, "processing")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    _write_xml(out / "online_control/simulation.sumocfg", cfg)
    asset_manifest = dict(schema="bus-cross-city-native-assets-v1", city=selection["city"],
        sumo_dir="online_control", edge_xml="network_data/a_sorted_busline_edge.xml",
        sumo_cfg="online_control/simulation.sumocfg",
        passenger_obj_xml="online_control/initialize_obj/save_obj_passenger.add.xml",
        schedule_xml="online_control/initialize_obj/save_obj_bus.add.xml", episode_seconds=episode_seconds,
        start_hour=start_hour, service_hours=service_hours, passenger_arrival_end_seconds=service_hours * 3600,
        demand_seed=demand_seed, line_ids=[line["native_line_id"] for line in line_records],
        scheduled_service_date=weekday["service_date"], calendar_rule=weekday["calendar_rule"],
        station_max_by_line=[line["station_max"] for line in line_records],
        line_id_to_canonical_index={line["native_line_id"]: line["canonical_line_index"] for line in line_records},
        departure_count=len(bus_records), person_count=len(person_records), lines=line_records,
        source_metadata=dict(original_manifest=selection["original_manifest"], original_service_count=selection["original_service_count"],
                             selection_rule=selection["selection_rule"], input_manifest=str(source_path),
                             weekday_timetables=str(Path(weekday_timetables).resolve())),
        generated_inputs=dict(network="separate bus corridors preserving scheduled segment lengths/speeds; no urban traffic",
            terminal_stop="all source stops including the last destination; no policy holding at termini",
            exit_tls="always green beyond the last passenger destination; frozen bus object compatibility",
            demand="observed route/hour APC totals; direction weights from full-feed pattern trip counts, "
                   "existing within-direction pattern shares and generated stop/OD profile; Poisson person realizations",
            episode_tail="no new departures or passengers after the service window; all remaining passenger time is accrued"))
    _write_json(out / "asset_manifest.json", asset_manifest)
    return asset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-hour", type=int, default=7)
    parser.add_argument("--service-hours", type=int, default=4)
    parser.add_argument("--episode-seconds", type=int, default=24000)
    parser.add_argument("--max-services", type=int, default=12)
    parser.add_argument("--demand-seed", type=int, default=72101)
    parser.add_argument("--netconvert", default="netconvert")
    parser.add_argument("--weekday-timetables", type=Path)
    parser.add_argument("--write-weekday-timetables", type=Path)
    parser.add_argument("--gtfs-dir", type=Path)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    options = dict(start_hour=args.start_hour, service_hours=args.service_hours, max_services=args.max_services)
    if args.write_weekday_timetables:
        if args.gtfs_dir is None:
            parser.error("--gtfs-dir is required to write weekday timetables")
        result = write_weekday_timetables(args.city_manifest, args.gtfs_dir, args.write_weekday_timetables, **options)
        print(json.dumps(dict(service_date=result["service_date"],
                              departures=sum(len(line["trips"]) for line in result["selected_services"]))))
    elif args.select_only:
        result = write_selection(args.city_manifest, args.out, **options)
        print(json.dumps(dict(selected_services=len(result["selected_services"]),
                              selected_manifest=str(args.out / "selected_manifest.json"),
                              line_keys=[line["line_key"] for line in result["selected_services"]])))
    else:
        if args.weekday_timetables is None:
            parser.error("--weekday-timetables is required to build native service assets")
        result = build_city_assets(args.city_manifest, args.out, episode_seconds=args.episode_seconds,
                                  weekday_timetables=args.weekday_timetables,
                                  demand_seed=args.demand_seed, netconvert=args.netconvert, **options)
        print(json.dumps(dict(asset_manifest=str(args.out / "asset_manifest.json"),
                              services=len(result["lines"]), buses=result["departure_count"], persons=result["person_count"])))


if __name__ == "__main__":
    main()
