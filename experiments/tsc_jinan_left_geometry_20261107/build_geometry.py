"""Compile the prior Jinan candidate with the frozen opposed-left CityFlow curves."""
import argparse
from itertools import accumulate
import json
from math import dist
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

from audit_static import compile_view


def adapted(points, offset, start, end):
    translated = [(x + offset[0], y + offset[1]) for x, y in points]
    distances = list(accumulate((dist(a, b) for a, b in zip(points, points[1:])), initial=0.0))
    length = distances[-1]
    begin_shift = [a - b for a, b in zip(start, translated[0])]
    end_shift = [a - b for a, b in zip(end, translated[-1])]
    target = []
    for p, distance in zip(translated, distances):
        fraction = distance / length
        target.append([x + (1 - fraction) * d0 + fraction * d1
                       for x, d0, d1 in zip(p, begin_shift, end_shift)])
    return target


def make_curves(net_path, source):
    net = ET.parse(net_path).getroot()
    offset = tuple(map(float, net.find("location").attrib["netOffset"].split(",")))
    junctions = {j["id"]: j for j in source["intersections"]}
    widths = {r["id"]: len(r["lanes"]) for r in source["roads"]}
    view = compile_view(net_path)
    pairs = sorted((a, b) for (a, b, _, _), result in view["contexts"].items()
                   if result == {"both_protected"} and view["dirs"][a][0] == view["dirs"][b][0] == "l")
    keys = sorted({key for pair in pairs for key in pair})
    if len(pairs) != 24 or len(keys) != 48 or len({k[0] for k in keys}) != 12:
        raise ValueError("The parent does not have the frozen 24 pairs of opposed protected left turns")
    curves = {}
    for key in keys:
        tls, inlet, outlet = key
        _, origin, dest = view["dirs"][key]
        from_lane = int(inlet.rsplit("_", 1)[1])
        to_lane = int(outlet.rsplit("_", 1)[1])
        src_pair = (widths[origin] - 1 - from_lane, widths[dest] - 1 - to_lane)
        junction = junctions[tls]
        (index, movement), = ((i, m) for i, m in enumerate(junction["roadLinks"])
                              if m["startRoad"] == origin and m["endRoad"] == dest and m["type"] == "turn_left")
        lane_link, = (v for v in movement["laneLinks"] if (v["startLaneIndex"], v["endLaneIndex"]) == src_pair)
        xy = [(p["x"], p["y"]) for p in lane_link["points"]]
        start = view["exterior"][inlet][0][-1]
        end = view["exterior"][outlet][0][0]
        active = [i for i, phase in enumerate(junction["trafficLight"]["lightphases"])
                  if index in phase["availableRoadLinks"]]
        curves[key] = dict(key=key, source_roadlink=index, source_lane_pair=src_pair,
                           source_phase_indices=active, source_points=xy, target_points=adapted(xy, offset, start, end))
    for a, b in pairs:
        if not set(curves[a]["source_phase_indices"]).intersection(curves[b]["source_phase_indices"]):
            raise ValueError(f"Source does not concurrently serve the selected turn pair {a}, {b}")
    return dict(net_offset=offset, pairs=pairs, curves=list(curves.values()))


def build(priority, roadnet, netconvert, output):
    parent = priority.parent
    receipt = json.loads((parent / "build.json").read_text())
    if priority != Path(receipt["candidate_config"]) or receipt["exit_code"] != 0:
        raise ValueError("Expected the compiled priority candidate")
    original = Path(receipt["original_config"])
    manifest = original.parent.parent / "conversion_manifest.json"
    source = json.loads(manifest.read_text())["source_files"]["jinan_roadnet"]
    if Path(source["path"]).resolve() != roadnet:
        raise ValueError("Roadnet differs from the input bound by the v9 conversion manifest")
    cfg = ET.parse(priority)
    net_file = cfg.find("input/net-file").attrib["value"]
    intended = make_curves(parent / net_file, json.loads(roadnet.read_text()))
    physical = ET.parse(parent / "physical.con.xml")
    targets = {(r["key"][1], r["key"][2]): r for r in intended["curves"]}
    applied = 0
    dx, dy = intended["net_offset"]
    for c in physical.findall("connection"):
        key = (c.attrib["from"] + "_" + c.attrib["fromLane"], c.attrib["to"] + "_" + c.attrib["toLane"])
        if key in targets:
            # netconvert normalizes these plain coordinates together with the input nodes and edges.
            c.set("shape", " ".join(f"{x - dx:.8f},{y - dy:.8f}" for x, y in targets[key]["target_points"]))
            applied += 1
    if applied != len(targets) or applied != 48:
        raise ValueError("Could not place all 48 curves in the frozen physical connections")
    output.mkdir(parents=True, exist_ok=False)
    (output / "intended_curves.json").write_text(json.dumps(intended, separators=(",", ":")) + "\n")
    connection_file = output / "geometry.con.xml"
    physical.write(connection_file, encoding="utf-8", xml_declaration=True)
    # Carry the parent's exact compiler, TLS, geometry options, and original nodes/edges.
    cmd = list(receipt["command"])
    if Path(cmd[0]).resolve() != netconvert or "1.22.0" not in receipt["netconvert_version"]:
        raise ValueError("Expected the same SUMO 1.22.0 netconvert as the priority candidate")
    cmd[cmd.index("--connection-files") + 1] = str(connection_file)
    cmd[cmd.index("--output-file") + 1] = str(output / net_file)
    with (output / "netconvert.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    passed = result.returncode == 0 and (output / net_file).is_file()
    candidate_cfg = output / priority.name
    if passed:
        cfg.write(candidate_cfg, encoding="utf-8", xml_declaration=True)
    summary = dict(status="BUILT_PENDING_STATIC_ADMISSION" if passed else "HOLD_BUILD",
                   priority_config=str(priority), original_config=str(original), roadnet=str(roadnet),
                   source_manifest=str(manifest), candidate_config=str(candidate_cfg), candidate_net=str(output / net_file),
                   intended_curves=str(output / "intended_curves.json"), edited_connections=applied,
                   opposed_left_pairs=len(intended["pairs"]), net_offset=intended["net_offset"],
                   netconvert_version=receipt["netconvert_version"], command=cmd, network_compile_invocations=1,
                   exit_code=result.returncode, log=str(output / "netconvert.log"))
    (output / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("priority", "roadnet", "netconvert", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if build(args.priority.resolve(), args.roadnet.resolve(),
                               args.netconvert.resolve(), args.output.resolve()) else 2)
