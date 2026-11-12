"""Admit the 24-pair geometric edit against the physical movements of the priority build."""
import argparse
from collections import Counter
from itertools import accumulate
import json
from math import dist
from pathlib import Path
from xml.etree import ElementTree as ET

from audit_static import compile_view, config, point_segment_distance, right_straight, same_path, summarize


def cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def separation(left, right):
    gap = float("inf")
    for a, b in zip(left, left[1:]):
        for c, d in zip(right, right[1:]):
            if cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0:
                return 0.0
            gap = min(gap, point_segment_distance(a,c,d), point_segment_distance(b,c,d),
                      point_segment_distance(c,a,b), point_segment_distance(d,a,b))
    return gap


def target_data(prior, source):
    # Bind the previous unresolved *physical* pairs first, so a missing candidate foe
    # (the desired result) cannot silently remove a case from the denominator.
    pairs = [(a,b) for (a,b,ca,cb), v in prior["contexts"].items()
             if v == {"both_protected"} and ca == cb == "G"
             and prior["dirs"][a][0] == prior["dirs"][b][0] == "l"]
    roads = {r["id"]: r for r in source["roads"]}
    junctions = {i["id"]: i for i in source["intersections"]}
    bindings = {}
    for key in sorted({k for pair in pairs for k in pair}):
        tls, inlet, outlet = key
        _, approach, exit_road = prior["dirs"][key]
        turns = junctions[tls]["roadLinks"]
        index, = [i for i, turn in enumerate(turns)
                  if (turn["startRoad"],turn["endRoad"],turn["type"]) == (approach, exit_road, "turn_left")]
        start = len(roads[approach]["lanes"])-1-int(inlet.rsplit("_",1)[1])
        end = len(roads[exit_road]["lanes"])-1-int(outlet.rsplit("_",1)[1])
        link, = [l for l in turns[index]["laneLinks"] if (l["startLaneIndex"],l["endLaneIndex"])==(start,end)]
        bindings[key] = (index, [(p["x"], p["y"]) for p in link["points"]])
    if len(pairs) != 24 or len(bindings) != 48 or len({a[0] for a,b in pairs}) != 12:
        raise ValueError("Prior candidate does not contain the frozen 24 opposed-left pairs")
    for a,b in pairs:
        indices={bindings[a][0],bindings[b][0]}
        if not any(indices.issubset(phase["availableRoadLinks"])
                   for phase in junctions[a[0]]["trafficLight"]["lightphases"]):
            raise ValueError(f"Source has no common phase for {a} and {b}")
    return pairs, {k:v[1] for k,v in bindings.items()}


def adapted_curve(poly, start, end):
    # A rigid translation cancels in the endpoint corrections. Work relative to the
    # translated start so this calculation does not depend on the builder's offset convention.
    arc = list(accumulate((dist(a,b) for a,b in zip(poly,poly[1:])), initial=0.0))
    return [tuple(start[j]+p[j]-poly[0][j]+s/arc[-1]*(end[j]-start[j]-poly[-1][j]+poly[0][j])
                  for j in range(2)) for p,s in zip(poly,arc)]


def limits(numbers):
    return [round(min(numbers),5),round(max(numbers),5)] if numbers else None


def audit(prior_cfg, candidate_cfg, roadnet):
    net_a, routes_a, options_a = config(prior_cfg)
    net_b, routes_b, options_b = config(candidate_cfg)
    old, new = compile_view(net_a), compile_view(net_b)
    pairs, source = target_data(old, json.loads(roadnet.read_text()))
    lane_width = {l.get("id"):float(l.get("width","3.2")) for l in ET.parse(net_b).findall("edge/lane")}
    widths = {k: max(lane_width[l] for l in chain) for k,(chain,_) in new["paths"].items()}
    missing, bad_curve = [], []
    ends, gaps = [], []
    for key, original_curve in source.items():
        if key not in new["paths"]:
            missing.append(key)
            continue
        _, inlet, outlet = key
        start, end = old["exterior"][inlet][0][-1], old["exterior"][outlet][0][0]
        actual = new["paths"][key][1]
        ends.append(max(dist(start,actual[0]),dist(end,actual[-1])))
        if not same_path(actual,adapted_curve(original_curve,start,end)):
            bad_curve.append(key)
    for a,b in pairs:
        if a not in new["paths"] or b not in new["paths"]:
            continue
        gap = separation(new["paths"][a][1],new["paths"][b][1])
        width_sum = (widths[a]+widths[b])/2
        gaps.append(dict(junction=a[0], links=[new["signal_index"][k] for k in (a,b)],
                         source_m=round(separation(source[a],source[b]),5), compiled_m=round(gap,5),
                         envelope_margin_m=round(gap-width_sum,5)))

    common = old["paths"].keys() & new["paths"].keys()
    off_target = sorted(k for k in common - source.keys() if not same_path(old["paths"][k][1],new["paths"][k][1]))
    differences = {}
    for item in ("exterior","controls","phases","dirs"):
        a,b = old[item],new[item]
        bad = sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
        if bad:
            differences[item] = dict(count=len(bad),examples=bad[:2])
    moves_a, links_a = right_straight(old)
    moves_b, links_b = right_straight(new)
    if (len(moves_a),len(links_a),sum(links_a.values())) != (48,432,432):
        raise ValueError("Expected the archived priority candidate with 48/48 right-straight combinations")
    added = new["contexts"].keys()-old["contexts"].keys()
    unresolved = {k for k,v in new["contexts"].items() if v!={"resolved"}}
    # Report one representative per relation type, with the physical space and
    # request evidence for it. Full keys for an investigation stay on the node.
    classes = Counter()
    examples = {}
    for ctx in sorted(added|unresolved, key=lambda c:(c[0][0]!="intersection_4_3",c)):
        a,b,ca,cb = ctx
        label = (("new" if ctx in added else "existing"), *sorted([new["dirs"][a][0]+ca,new["dirs"][b][0]+cb]),
                 *sorted(new["contexts"][ctx]))
        name = "/".join(label)
        classes[name] += 1
        if name not in examples:
            ia,fa,ra = new["requests"][a]
            ib,fb,rb = new["requests"][b]
            sep = separation(new["paths"][a][1],new["paths"][b][1])
            examples[name] = dict(tls=a[0], physical_lane_pairs=[a[1:],b[1:]], signals=[ca,cb],
                                  tls_links=[new["signal_index"][k] for k in (a,b)], request_indices=[ia,ib],
                                  phases=sorted(new["traces"][ctx]), foes=[fa[ib],fb[ia]],
                                  response=[ra[ib],rb[ia]], centerline_separation_m=round(sep,5),
                                  lane_width_envelope_margin_m=round(sep-(widths[a]+widths[b])/2,5))
    criteria = dict(
        routes_and_configuration=routes_a==routes_b and all(p.is_file() for p in routes_a) and options_a==options_b,
        physical_connections_and_all_phases=not differences and old["paths"].keys()==new["paths"].keys(),
        unedited_internal_centerlines_preserved=not off_target,
        all_target_curves_match_adapted_source=not missing and not bad_curve,
        all_target_endpoints_continuous=len(ends)==48 and max(ends)<=0.02,
        target_lane_envelopes_clear=len(gaps)==24 and all(g["envelope_margin_m"]>0 for g in gaps),
        right_straight_48_preserved=moves_a.keys()==moves_b.keys() and links_a.keys()==links_b.keys() and all(links_b.values()),
        no_new_cogreen_conflicts=not added, all_cogreen_conflicts_resolved=not unresolved)
    return dict(status="PASS_STATIC_PRIORITY" if all(criteria.values()) else "HOLD_STATIC_GEOMETRY",
                prior_sumocfg=str(prior_cfg),candidate_sumocfg=str(candidate_cfg),source_roadnet=str(roadnet),
                criteria=criteria,failed=[k for k,v in criteria.items() if not v],
                before=summarize(old),candidate=summarize(new),target_pairs=len(pairs),target_curves=len(source),
                missing_curves=missing,curves_mismatching_source=bad_curve,endpoint_gap_range_m=limits(ends),
                pair_separations=gaps,source_distance_range_m=limits([x["source_m"] for x in gaps]),
                compiled_distance_range_m=limits([x["compiled_m"] for x in gaps]),
                envelope_margin_range_m=limits([x["envelope_margin_m"] for x in gaps]),
                physical_differences=differences,off_target_curve_changes=len(off_target),off_target_examples=off_target[:2],
                new_cogreen_conflicts=len(added),unresolved_cogreen_conflicts=len(unresolved),relation_counts=classes,
                relation_examples=examples,simulations=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("prior","candidate","roadnet","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.prior.resolve(),args.candidate.resolve(),args.roadnet.resolve())
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],failed=result["failed"])))
