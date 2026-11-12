"""Describe archived approaches and pair response differences at fixed common counts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median


MOVES = ("straight", "right")
ARMS = ("high", "low")


def average(values):
    items = [v for v in values if v is not None]
    return mean(items) if items else None


def rank_stats(rows):
    hi = [v["high_entry_s"] for v in rows if v["high_entry_s"] is not None]
    lo = [v["low_entry_s"] for v in rows if v["low_entry_s"] is not None]
    inlet = [v["pre_green_high_distance_m"] for v in rows if v["pre_green_high_location"] == "inlet"]
    return dict(vehicles=len(rows), t0_distance_m=average(v["distance_m"] for v in rows),
                t0_speed_mps=average(v["speed_mps"] for v in rows),
                t0_distance_median_m=median(v["distance_m"] for v in rows) if rows else None,
                t0_speed_median_mps=median(v["speed_mps"] for v in rows) if rows else None,
                pre_green_locations=dict(Counter(v["pre_green_high_location"] for v in rows)),
                pre_green_inlet_distance_m=average(inlet),
                high=len(hi), low=len(lo), crossed_both=sum(v["high_entry_s"] is not None
                    and v["low_entry_s"] is not None for v in rows),
                high_offset_s_histogram=dict(sorted(Counter(hi).items())),
                low_offset_s_histogram=dict(sorted(Counter(lo).items())))


def move_profile(cases, side, detailed=False):
    movements = [c["movements"][side] for c in cases]
    ranks = defaultdict(list)
    outside = defaultdict(Counter)
    for m in movements:
        for v in m["vehicles"]:
            ranks[v["rank"]].append(v)
        for v in m["outside_entries"]:
            for a in ARMS:
                outside[v["source"]][a] += v[a+"_entry_s"] is not None
    total = {a: sum(m[a+"_entries"] for m in movements) for a in ARMS}
    ranked = {rank: rank_stats(vs) for rank, vs in sorted(ranks.items())}
    for a in ARMS:
        if total[a] != sum(profile[a] for profile in ranked.values()) + sum(s[a] for s in outside.values()):
            raise ValueError(f"Crossing counts do not reconcile for {side}, {a}")
    view = dict(cases=len(cases), near_vehicles=sum(map(len, ranks.values())), high=total["high"],
                low=total["low"], delta_per_case=(total["high"]-total["low"]) / len(cases),
                near_t0_distance_m_per_case=mean(mean(v["distance_m"] for v in m["vehicles"]) for m in movements),
                near_t0_speed_mps_per_case=mean(mean(v["speed_mps"] for v in m["vehicles"]) for m in movements),
                outside_crossings={s: dict(c) for s, c in outside.items()})
    if detailed:
        view["ranks"] = ranked
    else:
        view["rank_counts"] = [[k,r["vehicles"],r["high"],r["low"]] for k,r in ranked.items()]
        for k in (1,2):
            if k in ranked:
                v=ranked[k]
                view[f"rank{k}_distance_speed_pre_green_distance"] = [
                    v["t0_distance_m"],v["t0_speed_mps"],v["pre_green_inlet_distance_m"]]
    return view


def kinematics(cases, rank):
    vs = [v for c in cases for v in c["movements"]["straight"]["vehicles"] if v["rank"] == rank]
    if not vs:
        return None
    r = rank_stats(vs)
    return dict(n=len(vs), distance_m=r["t0_distance_m"], speed_mps=r["t0_speed_mps"],
                pre_green_distance_m=r["pre_green_inlet_distance_m"],
                pre_green_on_inlet=r["pre_green_locations"].get("inlet",0),
                high_cross_rate=r["high"]/len(vs), low_cross_rate=r["low"]/len(vs))


def stratified_groups(targets, classify):
    groups = defaultdict(lambda: defaultdict(list))
    for c in targets:
        groups[tuple(c["point"])][classify(c)].append(c)
    point_rows = []
    for p, gs in sorted(groups.items()):
        point_rows.append(dict(point=p, groups={g:dict(cases=len(rows),first=kinematics(rows,1),
                               second=kinematics(rows,2)) for g,rows in gs.items()}))
    matched = {}
    for key in ("first","second"):
        comparable = [r for r in point_rows if len(r["groups"])==2 and
                      all(v[key] is not None for v in r["groups"].values())]
        stat = {}
        for g in sorted({g for r in comparable for g in r["groups"]}):
            observations = [r["groups"][g][key] for r in comparable]
            stat[g] = dict(case_n=sum(o["n"] for o in observations),
                **{k: average(o[k] for o in observations) for k in
                ("distance_m","speed_mps","pre_green_distance_m","high_cross_rate","low_cross_rate")})
        matched[key]=dict(common_points=[r["point"] for r in comparable],equal_point=stat)
    return dict(by_point=point_rows, within_point=matched)


def summarize(data, plan):
    points = sorted(tuple(r["point"]) for r in plan["source_models"]["points"])
    cities = {"hangzhou": [], "jinan": []}
    for c in data["cases"]:
        cities[c["city"]].append(c)
    counts = {city:Counter((c["seed"],tuple(c["point"])) for c in cs) for city, cs in cities.items()}
    expected = {"hangzhou":(32061101,32061102),"jinan":tuple(plan["seeds"])}
    missing = {city:[(s,p,counts[city][s,p]) for s in seeds for p in points if counts[city][s,p]!=1]
               for city,seeds in expected.items()}
    if data["missing"] or any(missing.values()) or sum(len(rows) for rows in cities.values()) != 42:
        return dict(status="INCOMPLETE", extract_missing=data["missing"], case_denominators=missing)
    by_city, per_point, per_seed = {}, [], []
    for city, rows in cities.items():
        for c in rows:
            for idx, side in enumerate(("right","straight")):
                if len(c["movements"][side]["vehicles"]) != c["point"][idx]:
                    raise ValueError("Reconstructed t0 cohort differs from the locked count")
        by_city[city]={side:move_profile(rows,side,True) for side in MOVES}
        for seed in expected[city]:
            subset=[c for c in rows if c["seed"]==seed]
            per_seed.append(dict(city=city,seed=seed,**{s:move_profile(subset,s) for s in MOVES}))
    for p in points:
        per_point.append(dict(point=p,**{city:{s:move_profile(
            [c for c in rows if tuple(c["point"])==p],s) for s in MOVES} for city,rows in cities.items()}))
    source_arms = {p:{a:mean(c["movements"]["straight"][a+"_entries"] for c in cities["hangzhou"]
                            if tuple(c["point"])==p) for a in ARMS} for p in points}
    frozen = {tuple(p["point"]):p["hangzhou_matched"][1] for p in plan["source_models"]["points"]}
    if any(source_arms[p]["high"]-source_arms[p]["low"] != frozen[p] for p in points):
        raise ValueError("Archived source entries disagree with frozen straight response")
    errors=[]
    for c in cities["jinan"]:
        m=c["movements"]["straight"]
        hz=source_arms[tuple(c["point"])]
        high_deficit=hz["high"]-m["high_entries"]
        low_excess=m["low_entries"]-hz["low"]
        err=high_deficit+low_excess
        errors.append(dict(seed=c["seed"],point=c["point"],time_sec=c["time_sec"],
                           error=err,high_deficit=high_deficit,low_excess=low_excess))
    by_checkpoint={(r["seed"],r["time_sec"]):r for r in errors}
    def err(c):
        return by_checkpoint[c["seed"],c["time_sec"]]
    yellow=[]
    for city,cs in cities.items():
        groups=defaultdict(list)
        for c in cs:
            m=c["movements"]["straight"]
            groups[(m["high_actual"],m["low_actual"])].append(c)
        for (hi,lo),rows in sorted(groups.items()):
            profile=move_profile(rows,"straight")
            counts=dict(Counter((err(c)["error"] for c in rows))) if city=="jinan" else None
            yellow.append(dict(city=city,high_signal=hi,low_signal=lo,straight=profile,target_error_counts=counts))
    return dict(status="COMPLETE",missing=data["missing"],cities=by_city,per_point=per_point,per_seed=per_seed,
                straight_target_errors=errors,straight_actual_signal_strata=yellow,
                error_groups=stratified_groups(cities["jinan"],lambda c:
                    "over" if err(c)["error"]>0 else "under" if err(c)["error"]<0 else "exact"),
                high_arm_groups=stratified_groups(cities["jinan"],lambda c:
                    "high_below_source" if err(c)["high_deficit"]>0 else "high_above_source"
                    if err(c)["high_deficit"]<0 else "high_matches_source"),
                weighting="All seven points are equally represented within each seed. City per-case means average seeds and points equally. Rank counts are conditional on a vehicle of that rank, and pre-green inlet distance on it remaining on the inlet. Within-point group contrasts average only the common represented points equally. Distances are meters to the stop line and speeds m/s; sampled entry offsets run from 1 to 10 seconds.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract",type=Path)
    parser.add_argument("frozen_plan",type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(json.loads(args.extract.read_text()),json.loads(args.frozen_plan.read_text())),
                     separators=(",",":"),allow_nan=False))
