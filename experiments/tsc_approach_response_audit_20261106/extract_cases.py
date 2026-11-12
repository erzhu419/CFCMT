"""Reconstruct preaction focal queues and paired stop-line events from saved native records."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def load(path):
    return json.loads(path.read_text())


def speed_limits(cfg):
    net = cfg.parent / ET.parse(cfg).find("./input/net-file").get("value")
    return {lane.get("id"): float(lane.get("speed")) for lane in ET.parse(net).iter("lane")}


def trace(states, links):
    sequences = {"".join(state[i] for state in states) for i in links}
    if len(sequences) != 1:
        raise ValueError("Movement links have different signals")
    return sequences.pop()


def movement(row, side, starts, observations, limits, first_priority):
    movement_id = row[side + "_id"]
    catalog = observations["high"]["catalog"][movement_id]
    lane, = catalog["incoming_lanes"]
    length = observations["high"]["lane_geometry"][lane]["length_m"]
    limit = limits[lane]
    t0 = row["time_sec"]
    entries = {}
    out = dict(speed_limit_mps=limit, length_m=length, first_priority_offset_s=first_priority)
    for arm, raw in observations.items():
        receipt = row[arm]
        if raw["catalog"][movement_id] != catalog or raw["lane_geometry"][lane]["length_m"] != length:
            raise ValueError("Lane geometry or movement identity differs between arms")
        actual = trace([f["applied_states"][row["tls_id"]] for f in raw["frames"][1:]], catalog["link_indices"])
        nominal = trace(raw["nominal"], catalog["link_indices"])
        if (actual, nominal) != (receipt[side + "_actual"], receipt[side + "_nominal"]):
            raise ValueError("Focal signal trace differs from native receipt")
        events = [e for e in raw["events"] if e["kind"] == "stopline_entry" and e["movement_id"] == movement_id]
        offsets = {e["vehicle_id"]: int(e["time_sec"] - t0) for e in events}
        if (len(events) != len(offsets) or len(events) != receipt[side + "_entries"]
                or any(not 1 <= t <= 10 for t in offsets.values())):
            raise ValueError("Focal stop-line entries differ from native receipt/window")
        entries[arm] = offsets
        out.update({arm + "_entries": len(events), arm + "_actual": actual, arm + "_nominal": nominal})

    start = starts["high"]["vehicles"]
    near = sorted((length - v["lane_position"], vid) for vid, v in start.items()
                  if v["lane_id"] == lane and length - v["lane_position"] <= 10 * limit)
    if len(near) != row[side + "_near"]:
        raise ValueError(f"{side} reconstructed {len(near)} vs frozen {row[side + '_near']} near vehicles")
    frame = observations["high"]["frames"][first_priority - 1] if first_priority else None
    vehicles = []
    for rank, (distance, vid) in enumerate(near, 1):
        v = start[vid]
        index = v["route_index"]
        if index < 0 or v["route"][index] != catalog["incoming_edge"]:
            raise ValueError("Physical start route index does not reference focal incoming edge")
        matches = v["route"][index:index + 2] == [catalog["incoming_edge"], catalog["outgoing_edge"]]
        before_green_distance = None
        if frame is None:
            where = "no_priority_window"
        elif vid in entries["high"] and entries["high"][vid] < first_priority:
            where = "crossed"
        elif vid not in frame["vehicles"]:
            where = "unobserved"
        elif frame["vehicles"][vid]["lane_id"] == lane:
            where = "inlet"
            before_green_distance = length - frame["vehicles"][vid]["position_m"]
        else:
            where = "other"
        vehicles.append(dict(rank=rank, distance_m=distance, speed_mps=v["speed"],
                             high_entry_s=entries["high"].get(vid), low_entry_s=entries["low"].get(vid),
                             route_matches_movement=matches, pre_green_high_location=where,
                             pre_green_high_distance_m=before_green_distance))
    near_ids = {vid for _, vid in near}
    outside = []
    for vid in sorted((entries["high"].keys() | entries["low"].keys()) - near_ids):
        if vid not in start:
            source = "new_after_t0"
        elif start[vid]["lane_id"] == lane:
            source = "same_inlet_far"
        else:
            source = "other_lane_t0"
        outside.append(dict(source=source, high_entry_s=entries["high"].get(vid),
                            low_entry_s=entries["low"].get(vid)))
    out.update(vehicles=vehicles, outside_entries=outside)
    return out


def extract_case(city, directory, row, limits):
    starts, observations = {}, {}
    for arm in ("high", "low"):
        folder = directory / arm
        starts[arm] = load(folder / "physical_start_server_only.json")
        raw = load(folder / "observations_server_only.json")
        if (row[arm]["status"] != "COMPLETE" or starts[arm]["time_sec"] != row["time_sec"]
                or [f["time_sec"] for f in raw["frames"]] != list(range(row["time_sec"], row["time_sec"] + 11))):
            raise ValueError("Native checkpoint/window is incomplete or belongs to a different case")
        observations[arm] = raw
    if starts["high"] != starts["low"]:
        raise ValueError("Physical replay start differs between arms")
    exposure = {}
    for arm in ("high", "low"):
        both = [i for i, (r, s) in enumerate(zip(row[arm]["right_actual"], row[arm]["straight_actual"]), 1)
                if (r, s) == ("g", "G")]
        if arm == "high":
            first_priority = min(both, default=None)
        exposure[arm] = len(both)
        if exposure[arm] != row[arm]["exposure_s"]:
            raise ValueError("Realized priority exposure differs from native receipt")
    movements = {s: movement(row, s, starts, observations, limits, first_priority) for s in ("right", "straight")}
    return dict(city=city, seed=row["seed"], time_sec=row["time_sec"], tls_id=row["tls_id"],
                point=[row["right_near"], row["straight_near"]], exposure=exposure, movements=movements)


def extract(cohorts):
    cases, missing = [], []
    for city, directory, cfg, expected in cohorts:
        roster = load(directory / "pairs.json")
        if len(roster) != expected or load(directory / "result.json")["status"] != "COMPLETE":
            raise ValueError(f"Incomplete original measurement for {city}")
        limits = speed_limits(cfg)
        for i, row in enumerate(roster, 1):
            try:
                cases.append(extract_case(city, directory / f"case_{i:02d}", row, limits))
            except (ValueError, FileNotFoundError, KeyError) as exc:
                missing.append(dict(city=city, seed=row["seed"], time_sec=row["time_sec"], error=str(exc)))
    return dict(cases=cases, missing=missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hangzhou", "hangzhou-cfg", "jinan", "jinan-cfg", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    data = extract([("hangzhou", args.hangzhou, args.hangzhou_cfg, 14),
                    ("jinan", args.jinan, args.jinan_cfg, 28)])
    args.output.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
    print(json.dumps(dict(measured=len(data["cases"]), missing=data["missing"])))
    raise SystemExit(2 if data["missing"] else 0)
