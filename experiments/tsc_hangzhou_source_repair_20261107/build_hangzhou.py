"""Bind the Hangzhou source geometry and compile one candidate if its source supports the edit."""
import argparse
import json
from math import dist
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

from audit_geometry import separation
from audit_static import compile_view
from build_candidate import signal_inputs
from build_geometry import adapted


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def bounds(values):
    return [round(min(values), 6), round(max(values), 6)]


def bind_curves(view, source, offset):
    pairs = sorted((a, b) for (a, b, ca, cb), relation in view["contexts"].items()
                   if relation == {"both_protected"} and ca == cb == "G"
                   and view["dirs"][a][0] == view["dirs"][b][0] == "l")
    keys = sorted({link for pair in pairs for link in pair})
    if len(pairs) != 32 or len(keys) != 64 or len({a[0] for a, _ in pairs}) != 16:
        raise ValueError("Expected 32 distinct protected left pairs over all 16 source junctions")
    roads = {r["id"]: r for r in source["roads"]}
    junctions = {j["id"]: j for j in source["intersections"]}
    curves, endpoint_corrections = {}, []
    for k in keys:
        tls, inlet, outlet = k
        _, approach, leaving = view["dirs"][k]
        junction = junctions[tls]
        (index, turn), = ((i, t) for i, t in enumerate(junction["roadLinks"])
                          if (t["startRoad"], t["endRoad"], t["type"]) == (approach, leaving, "turn_left"))
        a = len(roads[approach]["lanes"]) - 1 - int(inlet.rsplit("_", 1)[1])
        b = len(roads[leaving]["lanes"]) - 1 - int(outlet.rsplit("_", 1)[1])
        lane, = (l for l in turn["laneLinks"] if (l["startLaneIndex"], l["endLaneIndex"]) == (a, b))
        points = [(p["x"], p["y"]) for p in lane["points"]]
        start, end = view["exterior"][inlet][0][-1], view["exterior"][outlet][0][0]
        phases = [i for i, p in enumerate(junction["trafficLight"]["lightphases"])
                  if index in p["availableRoadLinks"]]
        widths = [roads[approach]["lanes"][a]["width"], roads[leaving]["lanes"][b]["width"]]
        target = adapted(points, offset, start, end)
        endpoint_corrections.extend(dist(native, (p[0]+offset[0], p[1]+offset[1]))
                                    for native, p in [(start, points[0]), (end, points[-1])])
        curves[k] = dict(key=k, source_roadlink=index, source_lane_indices=[a, b], source_phases=phases,
                         source_points=points, target_points=target, endpoint_widths_m=widths)
    rows = []
    for a, b in pairs:
        left, right = curves[a], curves[b]
        common = sorted(set(left["source_phases"]) & set(right["source_phases"]))
        source_gap = separation(left["source_points"], right["source_points"])
        width = (max(left["endpoint_widths_m"]) + max(right["endpoint_widths_m"])) / 2
        rows.append(dict(tls=a[0], links=[a,b], common_source_phase_indices=common,
                         source_gap_m=source_gap, source_envelope_margin_m=source_gap-width,
                         intended_gap_m=separation(left["target_points"], right["target_points"])))
    failed = [r for r in rows if not r["common_source_phase_indices"] or r["source_envelope_margin_m"] <= 0]
    gate = dict(status="HOLD_SOURCE_GEOMETRY" if failed else "PASS_SOURCE_GEOMETRY", junctions=16,
                target_lane_curves=len(curves), target_lane_pairs=len(rows), unadmitted_pairs=failed,
                source_joint_service_pairs=sum(bool(r["common_source_phase_indices"]) for r in rows),
                source_centerline_gap_range_m=bounds([r["source_gap_m"] for r in rows]),
                source_envelope_margin_range_m=bounds([r["source_envelope_margin_m"] for r in rows]),
                intended_gap_range_m=bounds([r["intended_gap_m"] for r in rows]),
                corrected_endpoint_displacements_m=bounds(endpoint_corrections),
                native_net_offset=offset, network_compiles_before_gate=0)
    return curves, rows, gate


def build(cfg_path, netconvert, out):
    parent = cfg_path.parent
    receipt = json.loads((parent / "static_admission.json").read_text())
    if receipt["status"] != "PASS_STATIC_ADMISSION":
        raise ValueError("Expected the archived aligned source package")
    cfg = ET.parse(cfg_path)
    original_net = (parent / cfg.find("input/net-file").attrib["value"]).resolve(strict=True)
    route = (parent / cfg.find("input/route-files").attrib["value"]).resolve(strict=True)
    if original_net != Path(receipt["aligned"]["net"]) or route != Path(receipt["bound_original_vehicle_file"]):
        raise ValueError("Runtime inputs differ from the archived aligned package")
    source_path = Path(receipt["roadnet"])
    native = ET.parse(original_net)
    offset = tuple(map(float, native.find("location").attrib["netOffset"].split(",")))
    view = compile_view(original_net)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(original_config=str(cfg_path), candidate_config=str(out/cfg_path.name),
                   roadnet=str(source_path), native_net=str(original_net), bound_route=str(route),
                   network_compile_invocations=0)
    try:
        curves, rows, gate = bind_curves(view, json.loads(source_path.read_text()), offset)
    except ValueError as exc:
        curves, rows = {}, []
        gate = dict(status="HOLD_SOURCE_GEOMETRY", error=str(exc), network_compiles_before_gate=0)
    gate.update(original_config=str(cfg_path), roadnet=str(source_path))
    save(out / "source_gate.json", gate)
    if gate["status"] != "PASS_SOURCE_GEOMETRY":
        summary["status"] = gate["status"]
        save(out / "build.json", summary)
        return False
    save(out / "intended_curves.json", dict(curves=list(curves.values()), pairs=rows))
    controls, conn, scope = signal_inputs(native)
    targets = {(key[1],key[2]): c["target_points"] for key,c in curves.items()}
    placed = set()
    for c in conn.findall("connection"):
        a = c.attrib["from"] + "_" + c.attrib["fromLane"]
        b = c.attrib["to"] + "_" + c.attrib["toLane"]
        if (a,b) in targets:
            c.set("shape", " ".join(f"{x-offset[0]:.8f},{y-offset[1]:.8f}" for x,y in targets[a,b]))
            placed.add((a,b))
    if placed != targets.keys():
        raise ValueError("Incomplete target mapping to netconvert physical connections")
    tll, con = out / "final.tll.xml", out / "geometry.con.xml"
    for root,path in [(controls,tll),(conn,con)]:
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    nod, edg = (parent/(cfg_path.stem + "." + suffix + ".xml") for suffix in ("nod","edg"))
    version = subprocess.check_output([str(netconvert),"--version"], text=True).splitlines()[0]
    if "Version 1.22.0" not in version:
        raise ValueError(f"Unexpected compiler: {version}")
    net_out = out / original_net.name
    cmd = [str(netconvert),"--node-files",str(nod),"--edge-files",str(edg),
           "--connection-files",str(con),"--tllogic-files",str(tll),"--no-turnarounds","true",
           "--junctions.corner-detail","5","--output-file",str(net_out)]
    with (out/"netconvert.log").open("w") as log:
        res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    passed = res.returncode == 0 and net_out.is_file()
    if passed:
        cfg.find("input/net-file").set("value",net_out.name)
        cfg.find("input/route-files").set("value",str(route))
        cfg.write(out/cfg_path.name,encoding="utf-8",xml_declaration=True)
    summary.update(status="BUILT_PENDING_STATIC_ADMISSION" if passed else "HOLD_BUILD",
                   netconvert_version=version, candidate_net=str(net_out), edited_connections=len(placed),
                   explicit_edge_prohibitions=scope, command=cmd, network_compile_invocations=1,
                   exit_code=res.returncode, log=str(out/"netconvert.log"))
    save(out/"build.json",summary)
    return passed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source","netconvert","output"):
        p.add_argument("--"+arg, type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if build(args.source.resolve(),args.netconvert.resolve(),args.output.resolve()) else 2)
