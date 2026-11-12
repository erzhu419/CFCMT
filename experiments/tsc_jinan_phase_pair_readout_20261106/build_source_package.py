"""Build and statically inspect the separate Hangzhou CityFlow-aligned source."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

import sumolib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tsc_source_mechanism_selection_20261106"))
from source_static import actual_priority, movement, stable_phases
from scripts.data.convert_libsignal_external_to_sumo import convert_cityflow


def input_file(cfg, name):
    return (cfg.parent / ET.parse(cfg).find(f"./input/{name}").get("value")).resolve()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def span(values):
    return [min(values), max(values)] if values else None


def tls_pairs(connections, states, right_character):
    movements, destinations = defaultdict(list), defaultdict(set)
    for c in connections:
        movements[movement(c)].append(c)
        destinations[c.getFromLane().getID()].add(c.getTo().getID())
    pairs, right_count, wrong_right = [], 0, 0
    for right_id, rights in sorted(movements.items()):
        if {c.getDirection() for c in rights} != {"r"}:
            continue
        right_count += 1
        right_links = sorted({c.getTLLinkIndex() for c in rights})
        right_lanes = sorted({c.getFromLane().getID() for c in rights})
        if any(s[i] != right_character for s in states for i in right_links):
            wrong_right += 1
            continue
        if any(len(destinations[lane]) != 1 for lane in right_lanes):
            continue
        same_lane = defaultdict(set)
        for c in rights:
            for straight in actual_priority(c, connections):
                if straight.getDirection() == "s" and straight.getToLane() == c.getToLane():
                    same_lane[movement(straight)].add((c.getTLLinkIndex(), straight.getTLLinkIndex()))
        for straight_id, physical_pairs in sorted(same_lane.items()):
            straights = movements[straight_id]
            straight_lanes = sorted({c.getFromLane().getID() for c in straights})
            straight_links = sorted({c.getTLLinkIndex() for c in straights})
            if any(len(destinations[lane]) != 1 for lane in straight_lanes):
                continue
            if not all(any(all(s[i] == value for i in straight_links) for s in states) for value in "Gr"):
                continue
            pairs.append(dict(right_id=right_id, straight_id=straight_id,
                right_links=right_links, straight_links=straight_links, right_lanes=right_lanes,
                straight_lanes=straight_lanes, same_lane_link_pairs=sorted(physical_pairs)))
    return pairs, right_count, wrong_right


def inspect(cfg, right_character):
    path = input_file(cfg, "net-file")
    xml = ET.parse(path)
    net = sumolib.net.readNet(str(path), withInternal=True, withFoes=True)
    controlled = defaultdict(list)
    for edge in net.getEdges():
        for lane in edge.getLanes():
            for c in lane.getOutgoing():
                if c.getTLSID():
                    controlled[c.getTLSID()].append(c)
    identities, geometry, stages, pairs, counts = {}, {}, {}, {}, {}
    durations = defaultdict(set)
    rights = wrong_rights = 0
    for program in xml.findall("tlLogic"):
        tls = program.get("id")
        if tls in counts:
            raise ValueError(f"Expected a single signal program at {tls} in {path}")
        cons = controlled[tls]
        phases = list(stable_phases(program, 1 + max(c.getTLLinkIndex() for c in cons)))
        states = [state for _, _, state, _ in phases]
        counts[tls] = len(states)
        for phase in program.findall("phase"):
            state = phase.get("state")
            kind = "green" if state in states else "transition/" + "".join(sorted(set(state)))
            durations[kind].add(float(phase.get("duration")))
        durations["static_minimum_green"].update(minimum for _, _, _, minimum in phases)
        identities[tls] = {(c.getFromLane().getID(), c.getToLane().getID(), c.getDirection())
                           for c in cons}
        stages[tls] = [frozenset((c.getFromLane().getID(), c.getToLane().getID())
                                 for c in cons if state[c.getTLLinkIndex()] in "Gg")
                       for state in states]
        for c in cons:
            lane = c.getFromLane()
            geometry[lane.getID()] = (lane.getLength(), lane.getSpeed())
        pairs[tls], n, wrong = tls_pairs(cons, states, right_character)
        rights += n
        wrong_rights += wrong
    summary = dict(net=str(path), tls=len(counts), stable_phase_counts=sorted(set(counts.values())),
                   stable_right_character=right_character, right_movements=rights,
                   right_movements_with_other_signal=wrong_rights, directed_pair_counts={t:len(p) for t,p in sorted(pairs.items())},
                   incoming_length_m=span([v[0] for v in geometry.values()]),
                   incoming_speed_mps=span([v[1] for v in geometry.values()]),
                   phase_seconds={key: sorted(values) for key, values in sorted(durations.items())})
    return summary, identities, geometry, stages, pairs


def route_counts(flow, route):
    rows = json.loads(flow.read_text())
    if any(row["startTime"] != row["endTime"] for row in rows):
        raise ValueError("Expected single-departure records in original Hangzhou flow")
    raw = Counter((float(v["startTime"]), tuple(v["route"])) for v in rows)
    source = Counter((float(v.get("depart")), tuple(v.find("route").get("edges").split()))
                     for v in ET.parse(route).findall("vehicle"))
    return dict(raw=sum(raw.values()), bound=sum(source.values()), matched=sum((raw & source).values()))


def build(args):
    version = subprocess.check_output([str(args.netconvert), "--version"], text=True).splitlines()[0]
    if not re.search(r"\b1\.22\.0\b", version):
        raise ValueError(f"Frozen source pilot requires netconvert 1.22.0, got {version}")
    out = args.output
    tag = "hangzhou_aligned"
    conversion = convert_cityflow(roadnet_path=args.roadnet, flow_path=args.flow, output_dir=out,
                                  netconvert=args.netconvert, city="hangzhou", network=tag, prefix=tag)
    source = args.source_sumocfg
    bound_route = input_file(source, "route-files")
    cfg = out / f"{tag}.sumocfg"
    config = ET.parse(source)
    config.find("./input/net-file").set("value", f"{tag}.net.xml")
    config.find("./input/route-files").set("value", str(bound_route))
    config.write(cfg, encoding="utf-8", xml_declaration=True)
    (out / f"{tag}.rou.xml").unlink()  # Generated departure defaults differ from the bound original file.
    original, old_ids, old_geo, old_stages, old_pairs = inspect(source, "G")
    aligned, ids, geo, stages, pairs = inspect(cfg, "g")
    target = inspect(args.target_sumocfg, "g")[0]
    demand = route_counts(args.flow, input_file(cfg, "route-files"))
    common = sorted(geo.keys() & old_geo.keys())
    audit = conversion["tls_semantic_audit"]
    checks = dict(original_demand=demand == {"raw":2983, "bound":2983, "matched":2983},
                  physical_lane_connections=old_ids == ids,
                  same_inlet_lane_ids=old_geo.keys() == geo.keys(),
                  active_lane_connections_in_phase_order=stages == old_stages,
                  raw_remap_exact=audit["post_remap_phase_link_symmetric_difference"] == 0,
                  eight_actions_at_all_16=aligned["tls"] == 16 and aligned["stable_phase_counts"] == [8],
                  all_stable_rights_g=aligned["right_movements"] == 64 and aligned["right_movements_with_other_signal"] == 0,
                  directed_switchable_pair_at_every_tls=bool(pairs) and all(pairs.values()),
                  target_has_mapped_pairs=all(target["directed_pair_counts"].values()) and target["tls"] == 12)
    def physical_ids(by_tls):
        return {(p["right_id"], p["straight_id"]) for ps in by_tls.values() for p in ps}
    old, new = physical_ids(old_pairs), physical_ids(pairs)
    result = dict(status="PASS_STATIC_ADMISSION" if all(checks.values()) else "HOLD_STATIC_ADMISSION",
        checks=checks, original=original, aligned=aligned, target=target, demand=demand,
        netconvert_version=version, source_sumocfg=str(source), roadnet=str(args.roadnet), flow=str(args.flow),
        bound_original_vehicle_file=str(bound_route), conversion_tls_semantics=audit["protocol"],
        route_audit=conversion["route_audit"], source_pair_change={"retained":len(old & new),
            "lost":sorted(old - new), "gained":sorted(new - old)},
        original_to_aligned_inlet_changes=dict(common_lanes=len(common),
            length_difference_m=span([geo[l][0]-old_geo[l][0] for l in common]),
            speed_difference_mps=span([geo[l][1]-old_geo[l][1] for l in common])),
        new_episodes=0, new_fits=0)
    write(out / "movement_pairs.json", {"protocol":"right-straight-same-lane-pairs-v1", "tls":pairs})
    write(out / "static_admission.json", result)
    print(json.dumps({"status":result["status"], "failed":[key for key, passed in checks.items() if not passed]}))
    return all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("source-sumocfg", "roadnet", "flow", "target-sumocfg", "netconvert", "output"):
        parser.add_argument("--" + opt, required=True, type=lambda v: Path(v).resolve())
    sys.exit(0 if build(parser.parse_args()) else 2)
