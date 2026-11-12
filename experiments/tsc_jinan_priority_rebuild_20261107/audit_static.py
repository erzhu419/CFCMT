"""Check the compiled Jinan signal/exposure contract and connected physical right-of-way."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from math import dist, hypot
from pathlib import Path
from xml.etree import ElementTree as ET


def config(path):
    root = ET.parse(path).getroot()
    network = (path.parent / root.find("input/net-file").attrib.pop("value")).resolve()
    route_paths = root.find("input/route-files").attrib.pop("value").split(",")
    routes = [(path.parent / name).resolve() for name in route_paths]
    options = [(section.tag, option.tag, sorted(option.attrib.items()))
               for section in root for option in section]
    return network, routes, sorted(options)


def points(shape):
    return tuple(tuple(map(float, p.split(","))) for p in shape.split())


def lane_id(c, side):
    return c.get(side) + "_" + c.get(side + "Lane")


def link_key(c):
    return c.get("tl"), lane_id(c, "from"), lane_id(c, "to")


def compile_view(path):
    net = ET.parse(path).getroot()
    junctions = {j.get("id"): j for j in net.findall("junction")}
    lanes = {l.get("id"): l for l in net.findall("edge/lane")}
    links = {link_key(c): c for c in net.findall("connection") if c.get("tl")}
    internal = {(lane_id(c, "from"), lane_id(c, "to")): c for c in net.findall("connection")
                if c.get("from").startswith(":")}
    exterior = {l.get("id"): (points(l.get("shape")), float(l.get("length")), float(l.get("speed")),
                              l.get("width", "3.20"), l.get("allow"), l.get("disallow"))
                for e in net.findall("edge") if e.get("function") != "internal" for l in e.findall("lane")}
    controls = {}
    phases = {}
    paths, request, signal_index, dir_to = {}, {}, {}, {}
    contexts, traces = defaultdict(set), defaultdict(set)
    for program in net.findall("tlLogic"):
        tls, pid = program.get("id"), program.get("programID")
        phase_rows = program.findall("phase")
        controls[tls, pid] = (program.get("type"), float(program.get("offset", 0)),
                              [({k: v for k, v in p.attrib.items() if k not in ("duration", "state")},
                                float(p.get("duration"))) for p in phase_rows])
        for key, c in links.items():
            if key[0] != tls:
                continue
            index = int(c.get("linkIndex"))
            phases[tls, pid, key] = tuple(p.get("state")[index] for p in phase_rows)
    for key, c in links.items():
        tls, _, outgoing = key
        jj = junctions[tls]
        via = c.get("via")
        chain = []
        while via:
            chain.append(via)
            via = internal[via, outgoing].get("via")
        ordered = jj.get("intLanes").split()
        req_index, = (ordered.index(ident) for ident in chain if ident in ordered)
        r = jj.find(f"request[@index='{req_index}']")
        request[key] = (req_index, r.get("foes")[::-1], r.get("response")[::-1])
        signal_index[key] = int(c.get("linkIndex"))
        dir_to[key] = (c.get("dir"), c.get("from"), c.get("to"))
        poly = tuple(p for ident in chain for p in points(lanes[ident].get("shape")))
        paths[key] = (chain, poly)
    for (tls, pid) in controls:
        tl_links = sorted(k for k in links if k[0] == tls)
        for a, b in combinations(tl_links, 2):
            ia, fa, ra = request[a]
            ib, fb, rb = request[b]
            for i, (ca, cb) in enumerate(zip(phases[tls,pid,a], phases[tls,pid,b])):
                if ca not in "gG" or cb not in "gG" or (fa[ib], fb[ia]) == ("0", "0"):
                    continue
                pair = a, b, ca, cb
                # Approaches that merely offer multiple outgoing lanes share a single
                # stream. Two distinct incoming lanes can carry independent vehicles.
                if a[1] == b[1]:
                    continue
                if fa[ib] != fb[ia]:
                    verdict = "asymmetric_foes"
                elif ca == cb == "G":
                    verdict = "both_protected"
                elif ca == cb == "g":
                    verdict = "resolved" if ra[ib] != rb[ia] else "unresolved_permissive"
                else:
                    minor_major = (ra[ib], rb[ia]) if ca == "g" else (rb[ia], ra[ib])
                    verdict = "resolved" if minor_major == ("1", "0") else "wrong_or_ambiguous_yield"
                contexts[pair].add(verdict)
                traces[pair].add((pid, i))
    return dict(exterior=exterior, paths=paths, phases=phases, controls=controls, dirs=dir_to,
                requests=request, signal_index=signal_index, contexts=contexts, traces=traces)


def point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx+dy*dy) if (dx or dy) else 0
    t = max(0, min(1, t))
    return hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def same_path(a, b):
    # A new waiting point can split the same polyline; SUMO emits centimetre coordinates.
    if a == b:
        return True
    if max(dist(a[0], b[0]), dist(a[-1], b[-1])) > 0.02:
        return False
    for source, target in ((a,b), (b,a)):
        for p in source:
            if min(point_segment_distance(p, x, y) for x,y in zip(target, target[1:])) > 0.02:
                return False
    return True


def right_straight(v):
    movements, lanes = defaultdict(set), {}
    for ctx, verdict in v["contexts"].items():
        a, b, ca, cb = ctx
        if v["dirs"][b][0] == "r":
            a,b,ca,cb = b,a,cb,ca
        da, db = v["dirs"][a], v["dirs"][b]
        if (da[0],db[0],ca,cb) == ("r","s","g","G") and da[2] == db[2]:
            key = a[0], da[1], db[1], da[2]
            lanes[a,b] = verdict == {"resolved"}
            movements[key].add(lanes[a,b])
    return movements, lanes


def summarize(v):
    move, rows = right_straight(v)
    classes = Counter()
    examples = {}
    for ctx, verdicts in sorted(v["contexts"].items()):
        a,b,ca,cb = ctx
        for verdict in verdicts:
            label = ":".join([verdict, *sorted([v["dirs"][a][0]+ca, v["dirs"][b][0]+cb])])
            classes[label] += 1
            if label not in examples or (a[0] == "intersection_4_3" and examples[label]["tls"] != a[0]):
                examples[label] = dict(tls=a[0], lanes=[a[1:], b[1:]], tls_indices=[v["signal_index"][k] for k in (a,b)],
                                       signal=[ca,cb], phase_offsets=sorted(v["traces"][ctx]),
                                       response_bits=[v["requests"][a][2][v["requests"][b][0]],
                                                      v["requests"][b][2][v["requests"][a][0]]])
    return dict(physical_movements=len(v["paths"]), controlled_junctions=len(set(k[0] for k in v["paths"])),
                right_straight_movement_pairs=len(move), right_straight_all_lanes_correct=sum(ok == {True} for ok in move.values()),
                right_straight_lane_pairs=len(rows), right_straight_lanes_correct=sum(rows.values()), independent_cogreen_foes=dict(classes),
                unresolved_examples={k:r for k,r in examples.items() if not k.startswith("resolved:")})


def audit(original, candidate):
    net_a, routes_a, options_a = config(original)
    net_b, routes_b, options_b = config(candidate)
    a,b = compile_view(net_a),compile_view(net_b)
    before,after = summarize(a),summarize(b)
    moves_a,rows_a = right_straight(a)
    moves_b,rows_b = right_straight(b)
    common = a["paths"].keys() & b["paths"].keys()
    differences = {}
    for component in ("exterior","phases","controls","dirs"):
        left, right = a[component], b[component]
        bad = [k for k in left.keys() | right.keys() if left.get(k) != right.get(k)]
        if bad:
            differences[component] = dict(count=len(bad), examples=sorted(bad)[:2])
    changed_shape = sorted(k for k in common if not same_path(a["paths"][k][1], b["paths"][k][1]))
    if changed_shape:
        differences["internal_physical_shape"] = dict(count=len(changed_shape),examples=changed_shape[:2])
    invalid = {ctx for ctx,verdict in b["contexts"].items() if verdict != {"resolved"}}
    baseline_invalid = {ctx for ctx,verdict in a["contexts"].items() if verdict != {"resolved"}}
    added = b["contexts"].keys() - a["contexts"].keys()
    demand_same = len(routes_a) == len(routes_b) and all(
        x.is_file() and y.is_file() and (x == y or x.read_bytes() == y.read_bytes())
        for x,y in zip(routes_a,routes_b))
    if (len(moves_a), len(rows_a), sum(rows_a.values())) != (48,432,216):
        raise ValueError("Original package differs from the archived 48-movement/432-link case")
    criteria = dict(physical_lanes_movements_phases_unchanged=not differences and a["paths"].keys()==b["paths"].keys(),
                    config_and_demand_preserved=options_a==options_b and demand_same,
                    original_48_movements_and_conflicting_lanes_bound=moves_a.keys()==moves_b.keys() and rows_a.keys()==rows_b.keys(),
                    all_required_right_straight_lanes_yield=all(rows_b.values()),
                    no_unresolved_cogreen_foes=not invalid, no_new_cogreen_foes=not added)
    return dict(status="PASS_STATIC_PRIORITY" if all(criteria.values()) else "HOLD_STATIC_PRIORITY",
                original_sumocfg=str(original), candidate_sumocfg=str(candidate), original_net=str(net_a), candidate_net=str(net_b),
                criteria=criteria, failed=[k for k,x in criteria.items() if not x], original=before, candidate=after,
                physical_or_phase_differences=differences, route_content_unchanged=demand_same, config_options_unchanged=options_a==options_b,
                changed_via_lane_names_or_splits=sum(a["paths"][k][0]!=b["paths"][k][0] for k in common),
                changed_internal_segment_count=sum(len(a["paths"][k][0])!=len(b["paths"][k][0]) for k in common),
                remapped_tls_indices=sum(a["signal_index"][k]!=b["signal_index"][k] for k in common),
                remapped_request_indices=sum(a["requests"][k][0]!=b["requests"][k][0] for k in common),
                newly_conflicting_cogreen_contexts=len(added), new_conflict_examples=sorted(added)[:2],
                unresolved_also_present_in_original=len(invalid & baseline_invalid), newly_unresolved=len(invalid-baseline_invalid),
                definitions="Each co-green context counts one pair of distinct incoming lane streams and signal aspect pair across programs/phases. Right/straight movement counts group lane pairs by TLS, source edges and common outbound edge. Phase offsets in examples are (program ID, zero-based phase index).",
                simulations=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "candidate", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    args = p.parse_args()
    result = audit(args.original.resolve(), args.candidate.resolve())
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], failed=result["failed"])))
