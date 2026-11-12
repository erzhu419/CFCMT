"""Compare released CityFlow movement availability with the SUMO scenarios used here."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import sumolib

from source_static import actual_priority, stable_phases

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def compare(roadnet, network):
    junctions = {j["id"]: j for j in json.loads(roadnet.read_text())["intersections"] if not j["virtual"]}
    programs = {logic.get("id"): logic for logic in ET.parse(network).findall("tlLogic")
                if logic.get("programID") == "0"}
    net = sumolib.net.readNet(str(network), withInternal=True, withFoes=True)
    connections = defaultdict(list)
    for edge in net.getEdges():
        for lane in edge.getLanes():
            for c in lane.getOutgoing():
                if c.getTLSID():
                    connections[c.getTLSID()].append(c)
    assert set(programs) == set(junctions) == set(connections)
    counts, right_signals = Counter(), Counter()
    for tls, junction in junctions.items():
        links = junction["roadLinks"]
        physical = [(link["startRoad"], link["endRoad"]) for link in links]
        rights = {i for i, link in enumerate(links) if link["type"] == "turn_right"}
        phases = junction["trafficLight"]["lightphases"]
        stage0 = set(phases[0]["availableRoadLinks"])
        active = [set(p["availableRoadLinks"]) for p in phases[1:]]
        assert phases[0]["time"] == 5 and len(active) == 8
        assert rights == stage0 == set.intersection(*active)
        counts["raw_phase0_right_turns"] += len(rights)
        raw_stages = [frozenset(physical[i] for i in stage) for stage in active]
        cons = connections[tls]
        by_move = defaultdict(list)
        for c in cons:
            by_move[c.getFrom().getID(), c.getTo().getID()].append(c)
        assert set(physical) == set(by_move) and len(set(physical)) == len(physical)
        states = [s for _, _, s, _ in stable_phases(programs[tls], max(c.getTLLinkIndex() for c in cons) + 1)]
        assert len(states) == 8
        native_stages = [frozenset(m for m, cs in by_move.items()
                                   if any(s[c.getTLLinkIndex()] in "Gg" for c in cs)) for s in states]
        counts["stable_stages"] += len(states)
        counts["active_movement_sets_present_in_both"] += sum((Counter(native_stages) & Counter(raw_stages)).values())
        counts["stable_stages_aligned_in_original_order"] += sum(a == b for a, b in zip(raw_stages, native_stages))
        for i in rights:
            own = by_move[physical[i]]
            assert all(c.getDirection() == "r" for c in own)
            characters = {s[c.getTLLinkIndex()] for c in own for s in states}
            right_signals["".join(sorted(characters))] += 1
            straight_merges = [(c, p) for c in own for p in cons if p.getDirection() == "s"
                               and c.getToLane() == p.getToLane()
                               and {s[p.getTLLinkIndex()] for s in states} >= {"r", "G"}]
            if straight_merges:
                counts["right_turns_with_same_lane_straight_signal_toggle"] += 1
            # Does an existing physical right-of-way relation point to a through
            # connection merging into the same lane, with its G varying across actions?
            matching = [(c, p) for c in own for p in actual_priority(c, cons)
                        if p.getDirection() == "s" and c.getToLane() == p.getToLane()
                        and {s[p.getTLLinkIndex()] for s in states} >= {"r", "G"}]
            if matching:
                counts["right_turns_with_same_lane_straight_priority_toggle"] += 1
    return dict(roadnet=str(roadnet.relative_to(PROJECT)), network=str(network.relative_to(PROJECT)),
                tls=len(junctions), **counts, right_movement_signal_sets=dict(right_signals))


def compare_departures(raw_path, sumo_routes):
    flows = json.loads(raw_path.read_text())
    assert all(row["startTime"] == row["endTime"] for row in flows)
    raw = Counter((float(row["startTime"]), tuple(row["route"])) for row in flows)
    vehicles = ET.parse(sumo_routes).findall("vehicle")
    exported = Counter((float(v.get("depart")), tuple(v.find("route").get("edges").split()))
                       for v in vehicles)
    return dict(raw_vehicles=len(flows), sumo_vehicles=len(vehicles),
                matching_departure_time_and_road_sequence=sum((raw & exported).values()))


if __name__ == "__main__":
    hz = PROJECT / "H2Oplus/downloads/traffic_signal_libsignal/repo/data/raw_data/hangzhou_4x4_gudang_18041610_1h"
    jinan = PROJECT / "cf_h2o/results/cluster/tsc_v53r49_external_network_repair_20260810/conversion_v9_authoritative/full_networks_v9/jinan_3x4_real"
    report = dict(
        hangzhou=compare(hz / "roadnet_4X4.json", hz / "hangzhou_4x4_gudang_18041610_1h.net.xml"),
        jinan=compare(PROJECT / "H2Oplus/downloads/traffic_signal_external/colight/data/Jinan/3_4/roadnet_3_4.json",
                      jinan / "jinan_3x4_real.net.xml"),
        hangzhou_demand=compare_departures(hz / (hz.name + ".json"), hz / (hz.name + ".rou.xml")),
        convention="CityFlow provides active road links. The project's Jinan v9 converter interprets the short first stage as a persistent permissive set and maps its active lane links to g.",
        scope="Compare stable actions by physical incoming and outgoing road pair, and Hangzhou scheduled departures by time and road sequence. No realized arrivals, yielding or control outcomes.",
        new_packages=0, new_simulations=0, new_fits=0)
    (HERE / "cityflow_programs.json").write_text(json.dumps(report, indent=2) + "\n")
