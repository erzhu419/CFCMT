"""Compile the final native Jinan signals with the yielding movements they specify."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET


LANES = ("from", "to", "fromLane", "toLane")


def signal_inputs(net):
    tls = ET.Element("tlLogics")
    connections = ET.Element("connections")
    controlled = defaultdict(list)
    for link in net.findall("connection"):
        if link.get("from").startswith(":"):
            continue
        lanes = {k: link.attrib[k] for k in LANES}
        ET.SubElement(connections, "connection", lanes)
        if "tl" in link.attrib:
            controlled[link.attrib["tl"]].append(link)
    relations = {}
    for logic in net.findall("tlLogic"):
        name = logic.attrib["id"]
        tls.append(logic)
        phases = [phase.attrib["state"] for phase in logic.findall("phase")]
        junction = net.find(f"junction[@id='{name}']")
        index = {via: i for i, via in enumerate(junction.attrib["intLanes"].split())}
        requests = {int(r.attrib["index"]): r.attrib for r in junction.findall("request")}
        for yielding, major in permutations(controlled[name], 2):
            y = index[yielding.attrib["via"]]
            m = index[major.attrib["via"]]
            yi, mi = (int(c.attrib["linkIndex"]) for c in (yielding, major))
            if not any(s[yi] == "g" and s[mi] == "G" for s in phases):
                continue
            if requests[y]["foes"][-1 - m] != "1" or requests[m]["foes"][-1 - y] != "1":
                continue
            pair = tuple((c.attrib["from"], c.attrib["to"]) for c in (yielding, major))
            if pair[::-1] in relations:
                raise ValueError(f"Incompatible g/G priorities for {pair}")
            key = (name, yielding.attrib["dir"], major.attrib["dir"])
            relations[pair] = key
    for links in controlled.values():
        for link in links:
            ET.SubElement(tls, "connection", {k: link.attrib[k] for k in (*LANES, "tl", "linkIndex")})
    for yielding, major in sorted(relations):
        ET.SubElement(connections, "prohibition", prohibited="->".join(yielding),
                      prohibitor="->".join(major))
    scope = Counter(f"{minor}_yields_to_{major}" for _, minor, major in relations.values())
    return tls, connections, {"movement_relations": len(relations), "by_direction": dict(scope),
                              "signals": len(controlled),
                              "physical_controlled_connections": sum(map(len, controlled.values()))}


def build(source, netconvert, output):
    config = ET.parse(source)
    net_ref = config.find("./input/net-file")
    route_ref = config.find("./input/route-files")
    native = (source.parent / net_ref.attrib["value"]).resolve()
    route = (source.parent / route_ref.attrib["value"]).resolve(strict=True)
    nodes, edges = ((source.parent / (source.stem + "." + ext + ".xml")).resolve(strict=True)
                    for ext in ("nod", "edg"))
    version = subprocess.run([str(netconvert), "--version"], check=True, capture_output=True,
                             text=True).stdout.splitlines()[0]
    if "Version 1.22.0" not in version:
        raise ValueError(f"Expected SUMO netconvert 1.22.0; got {version}")
    tls, connections, scope = signal_inputs(ET.parse(native))
    output.mkdir(parents=True, exist_ok=False)
    signal_file = output / "final.tll.xml"
    connection_file = output / "physical.con.xml"
    for root, path in ((tls, signal_file), (connections, connection_file)):
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    candidate_net = output / native.name
    candidate_cfg = output / source.name
    cmd = [str(netconvert), "--node-files", str(nodes), "--edge-files", str(edges),
           "--connection-files", str(connection_file), "--tllogic-files", str(signal_file),
           "--no-turnarounds", "true", "--junctions.corner-detail", "5",
           "--output-file", str(candidate_net)]
    log = output / "netconvert.log"
    with log.open("w") as f:
        completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    receipt = dict(original_config=str(source), original_net=str(native), bound_route=str(route),
                   candidate_config=str(candidate_cfg), candidate_net=str(candidate_net),
                   explicit_edge_prohibitions=scope, netconvert_version=version, command=cmd,
                   network_compile_invocations=1, exit_code=completed.returncode, log=str(log))
    if completed.returncode == 0 and candidate_net.is_file():
        net_ref.set("value", candidate_net.name)
        route_ref.set("value", str(route))
        config.write(candidate_cfg, encoding="utf-8", xml_declaration=True)
        receipt["status"] = "BUILT_PENDING_STATIC_ADMISSION"
    else:
        receipt["status"] = "HOLD_BUILD"
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt["status"] == "BUILT_PENDING_STATIC_ADMISSION"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "netconvert", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if build(args.sumocfg.resolve(), args.netconvert.resolve(), args.output.resolve()) else 2)
