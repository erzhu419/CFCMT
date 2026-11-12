"""Project the repaired development archive to decision inputs and historical right-turn labels."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def read(path):
    return json.loads(path.read_text())


def lane_map(cfg):
    net_file = ET.parse(cfg).find("input/net-file").get("value")
    network = ET.parse(cfg.parent / net_file)
    lanes = {(edge.get("id"), lane.get("index")): lane
             for edge in network.findall("edge") for lane in edge.findall("lane")}
    connections = defaultdict(dict)
    directions = defaultdict(set)
    for c in network.findall("connection"):
        if c.get("tl") is None:
            continue
        lane = lanes[c.get("from"), c.get("fromLane")].get("id")
        index = int(c.get("linkIndex"))
        previous = connections[c.get("tl")].setdefault(index, lane)
        if previous != lane:
            raise ValueError("This network does not have a unique inlet for this signal index")
        directions[lane].add(c.get("dir"))
    geometry = {v.get("id"): (float(v.get("length")), float(v.get("speed"))) for v in lanes.values()}
    return connections, directions, geometry


def extract(root):
    if read(root / "arms/result.json")["status"] != "PASS_DEV_ARMS":
        raise ValueError("Expected a complete repaired high-arm development archive")
    roster = read(root / "census/roster.json")
    cases = {}
    networks = {}
    for role, city in (("target", "jinan"), ("source", "hangzhou")):
        networks[city] = lane_map(Path(roster[role + "_cfg"]))
        for case in roster[role + "_cases"]:
            cases[city, case["seed"], tuple(case["point"])] = case

    compact = []
    for i, r in enumerate(read(root / "arms/observations.json")["rows"], 1):
        if r["group"] == "source_native":
            continue
        city = r["city"]
        case = cases.pop((city, r["seed"], tuple(r["point"])))
        if r["applied_accel_mps2"] != [2, 2] or any(
            r[k] != case[k] for k in ("time_sec", "tls_id", "right_id", "straight_id", "high_candidate")
        ):
            raise ValueError("Measurement differs from its sealed start")
        tls = case["tls_id"]
        start = read(Path(case["start_file"]))
        states = start["candidate_states"]
        hi, lo = case["high_candidate"], case["low_candidate"]
        pp = states.index(start["pp_actions"][tls])
        if (len(states), case["high_exposure_s"], case["low_exposure_s"]) != (8, 4, 0):
            raise ValueError("Decision lacks the sealed high/low action contrast")
        links, directions, geometry = networks[city]
        halted = Counter(v["lane_id"] for v in start["vehicles"].values() if v["speed"] < 0.1)

        def green_lanes(index):
            return {links[tls][j] for j, color in enumerate(states[index]) if color in "Gg"}

        # The raw catalog supplies static movement identity. Only native stop-line events enter k.
        raw = read(root / "arms" / f"arm_{i:02d}" / "observations_server_only.json")
        lane, = raw["catalog"][case["right_id"]]["incoming_lanes"]
        green_hi, green_lo = green_lanes(hi), green_lanes(lo)
        if directions[lane] != {"r"} or lane not in green_hi & green_lo:
            raise ValueError("Both candidate phases must serve the same dedicated right inlet")
        length, speed_limit = geometry[lane]
        near = {vid: v for vid, v in start["vehicles"].items()
                if v["lane_id"] == lane and length - v["lane_position"] <= 10 * speed_limit}
        if len(near) != case["right_near"] or len(near) != start["supply"]["right_near"]:
            raise ValueError("Right vehicle cohort differs from the preaction census")
        stopped = {vid for vid, v in near.items() if v["speed"] < 0.1}
        entered = {e["vehicle_id"] for e in raw["events"] if e["kind"] == "stopline_entry"
                   and e["movement_id"] == case["right_id"] and 0 < e["time_sec"] - case["time_sec"] <= 10}
        if len(entered) != r["right_entries"]:
            raise ValueError("Right-turn entries disagree with the archived native measurement")
        compact.append(dict(city=city, seed=r["seed"], point=r["point"], arm_index=i, time_sec=case["time_sec"],
                            tls_id=tls, right_id=case["right_id"], straight_id=case["straight_id"],
                            hi=hi, lo=lo, pp=pp, b_hi=sum(halted[l] for l in green_hi),
                            b_lo=sum(halted[l] for l in green_lo), n=len(stopped), k=len(stopped & entered),
                            near_right=len(near), min_near_speed_mps=min(v["speed"] for v in near.values()),
                            max_near_speed_mps=max(v["speed"] for v in near.values())))
    if cases or len(compact) != 36:
        raise ValueError("Incomplete development panel")
    return compact


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("development_root", type=Path)
    print(json.dumps(extract(args.parse_args().development_root), separators=(",", ":")))
