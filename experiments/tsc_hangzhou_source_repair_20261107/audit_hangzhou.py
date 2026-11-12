"""Verify the Hangzhou candidate and rebind the archived source measurement registry."""
import argparse
from collections import Counter, defaultdict
import json
from math import dist
from pathlib import Path
from xml.etree import ElementTree as ET

from audit_geometry import adapted_curve, limits, separation
from audit_static import compile_view, config, right_straight, same_path, summarize


def source_pairs(view, roadnet):
    source = json.loads(roadnet.read_text())
    roads = {r["id"]: r for r in source["roads"]}
    junctions = {j["id"]: j for j in source["intersections"]}
    pairs = [(a, b) for (a,b,ca,cb), v in view["contexts"].items()
             if ca == cb == "G" and v == {"both_protected"}
             and view["dirs"][a][0] == view["dirs"][b][0] == "l"]
    bindings = {}
    for key in sorted({key for pair in pairs for key in pair}):
        tls, inlet, outlet = key
        _, approach, exit_road = view["dirs"][key]
        # _cityflow_connection_key reverses both source and target lane indices.
        i = len(roads[approach]["lanes"]) - 1 - int(inlet.rsplit("_", 1)[1])
        j = len(roads[exit_road]["lanes"]) - 1 - int(outlet.rsplit("_", 1)[1])
        ix, movement = next((ix,m) for ix,m in enumerate(junctions[tls]["roadLinks"])
                            if (m["type"],m["startRoad"],m["endRoad"]) == ("turn_left",approach,exit_road))
        lane, = [s for s in movement["laneLinks"] if (s["startLaneIndex"],s["endLaneIndex"]) == (i,j)]
        active = {idx for idx,p in enumerate(junctions[tls]["trafficLight"]["lightphases"])
                  if ix in p["availableRoadLinks"]}
        width = max(roads[approach]["lanes"][i]["width"], roads[exit_road]["lanes"][j]["width"])
        bindings[key] = ([(pt["x"], pt["y"]) for pt in lane["points"]], active, width)
    if (len(pairs), len(bindings), len({a[0] for a,b in pairs})) != (32,64,16):
        raise ValueError("Expected 32 opposed left-turn pairs in the parent source network")
    return pairs, bindings


def registry_row(view, tls, right, straight):
    movements = defaultdict(list)
    for key, (direction, origin, dest) in view["dirs"].items():
        if key[0] == tls:
            movements["|".join([tls,origin,dest])].append(key)
    rs, ss = movements[right], movements[straight]
    idx = view["signal_index"]
    same_lane = sorted([idx[a],idx[b]] for a in rs for b in ss if a[2] == b[2])
    return dict(right_id=right, straight_id=straight, right_lanes=sorted({k[1] for k in rs}),
                straight_lanes=sorted({k[1] for k in ss}), right_links=sorted(idx[k] for k in rs),
                straight_links=sorted(idx[k] for k in ss), same_lane_link_pairs=same_lane)


def audit(prior, candidate, roadnet, registry):
    pa, ra, ca = config(prior)
    pb, rb, cb = config(candidate)
    old, new = compile_view(pa), compile_view(pb)
    pairs, bindings = source_pairs(old, roadnet)
    widths = {l.get("id"): float(l.get("width", "3.2")) for l in ET.parse(pb).findall("edge/lane")}
    ends, mismatched, gaps = [], [], []
    for k,(poly,_,_) in bindings.items():
        if k not in new["paths"]:
            mismatched.append(k)
            continue
        actual = new["paths"][k][1]
        start, end = old["exterior"][k[1]][0][-1], old["exterior"][k[2]][0][0]
        ends.append(max(dist(actual[0], start), dist(actual[-1], end)))
        if not same_path(actual, adapted_curve(poly, start, end)):
            mismatched.append(k)
    for a,b in pairs:
        if a not in new["paths"] or b not in new["paths"]:
            continue
        aa, bb = bindings[a], bindings[b]
        gap = separation(new["paths"][a][1], new["paths"][b][1])
        width = sum(max(widths[l] for l in new["paths"][k][0]) for k in (a,b))/2
        src_gap = separation(aa[0],bb[0])
        gaps.append(dict(tls=a[0],links=[new["signal_index"][k] for k in (a,b)],joint_source=bool(aa[1]&bb[1]),
                         source_m=src_gap,source_margin_m=src_gap-(aa[2]+bb[2])/2,compiled_m=gap,margin_m=gap-width))
    outside = sorted(k for k in (old["paths"].keys() & new["paths"].keys())-bindings.keys()
                     if not same_path(old["paths"][k][1],new["paths"][k][1]))
    before, after = summarize(old), summarize(new)
    moves_a, rows_a = right_straight(old)
    moves_b, rows_b = right_straight(new)
    if (len(moves_a),len(rows_a),sum(rows_a.values())) != (64,576,288):
        raise ValueError("Unexpected parent Hangzhou right/straight network")
    archived = json.loads(registry.read_text())
    bound = dict(protocol=archived["protocol"],tls={})
    old_rows = []
    for tls, rows in archived["tls"].items():
        old_rows.extend(registry_row(old,tls,r["right_id"],r["straight_id"]) == r for r in rows)
        bound["tls"][tls] = [registry_row(new,tls,r["right_id"],r["straight_id"]) for r in rows]
    added = new["contexts"].keys()-old["contexts"].keys()
    unresolved = {k for k,v in new["contexts"].items() if v!={"resolved"}}
    differences = {c:sum(old[c].get(k)!=new[c].get(k) for k in old[c].keys()|new[c].keys())
                   for c in ("exterior","controls","phases","dirs")}
    criteria = dict(external_lanes_directions_phases=not any(differences.values()) and old["paths"].keys()==new["paths"].keys(),
                    demand_and_options=ra==rb and all(p.is_file() for p in ra) and ca==cb,
                    registered_source_pairs_bound=len(old_rows)==32 and all(old_rows)
                        and all(r["same_lane_link_pairs"] for rows in bound["tls"].values() for r in rows),
                    source_support=len(gaps)==32 and all(g["joint_source"] and g["source_margin_m"]>0 for g in gaps),
                    adapted_target_centerlines=len(ends)==64 and not mismatched and max(ends)<=0.02,
                    target_width_clearance=len(gaps)==32 and min(g["margin_m"] for g in gaps)>0,
                    unchanged_off_target=not outside,
                    all_64_right_straight=moves_a.keys()==moves_b.keys() and rows_a.keys()==rows_b.keys() and all(rows_b.values()),
                    no_new_independent_cogreen_foes=not added, no_unresolved_independent_cogreen_foes=not unresolved)
    counts = Counter()
    for a,b,ca_,cb_ in added:
        counts["/".join(sorted([new["dirs"][a][0]+ca_,new["dirs"][b][0]+cb_]))]+=1
    report = dict(status="PASS_STATIC_PRIORITY" if all(criteria.values()) else "HOLD_STATIC_HANGZHOU",
                  prior_sumocfg=str(prior),candidate_sumocfg=str(candidate),roadnet=str(roadnet),source_registry=str(registry),
                  criteria=criteria,failed=[k for k,v in criteria.items() if not v],original=before,candidate=after,
                  source_registered_rows=sum(old_rows),target_curves=len(bindings),target_pairs=len(pairs),
                  endpoint_gap_m=limits(ends),source_separation_m=limits([g["source_m"] for g in gaps]),
                  compiled_separation_m=limits([g["compiled_m"] for g in gaps]),compiled_margin_m=limits([g["margin_m"] for g in gaps]),
                  physical_or_phase_differences=differences,off_target_changes=len(outside),off_target_examples=outside[:2],
                  mismatched_curves=len(mismatched),mismatched_examples=mismatched[:2],new_conflicts=len(added),
                  new_conflict_classes=counts,new_conflict_examples=sorted(added)[:2],unresolved_conflicts=len(unresolved),simulations=0)
    return report, bound


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("prior","candidate","roadnet","registry","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result,bound=audit(args.prior.resolve(),args.candidate.resolve(),args.roadnet.resolve(),args.registry.resolve())
    if result["status"]=="PASS_STATIC_PRIORITY":
        path=args.output.resolve().parent/"rebound_registry.json"
        path.write_text(json.dumps(bound,indent=2)+"\n")
        result["registry"]=str(path)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],failed=result["failed"])),flush=True)
