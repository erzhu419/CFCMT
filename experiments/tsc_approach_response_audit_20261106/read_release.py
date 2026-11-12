"""Trace ranks one and two of each preaction straight inlet through the high arm."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from extract_cases import load, speed_limits


def cases(directory, cfg):
    limits = speed_limits(cfg)
    for i, row in enumerate(load(directory / "pairs.json"), 1):
        folder = directory / f"case_{i:02d}" / "high"
        start = load(folder / "physical_start_server_only.json")["vehicles"]
        raw = load(folder / "observations_server_only.json")
        movement = row["straight_id"]
        lane, = raw["catalog"][movement]["incoming_lanes"]
        length = raw["lane_geometry"][lane]["length_m"]
        near = sorted((length-v["lane_position"], vid) for vid, v in start.items()
                      if v["lane_id"] == lane and length-v["lane_position"] <= 10*limits[lane])
        if len(near) != row["straight_near"]:
            raise ValueError("Rank reconstruction differs from frozen near count")
        if len(near) < 2:
            continue
        entry = {e["vehicle_id"]: int(e["time_sec"]-row["time_sec"]) for e in raw["events"]
                 if e["kind"] == "stopline_entry" and e["movement_id"] == movement}
        hi = row["high"]
        green = [i for i, (r,s) in enumerate(zip(hi["right_actual"],hi["straight_actual"]),1)
                 if r == "g" and s == "G"]
        if green != [7,8,9,10] or [f["offset_sec"] for f in raw["frames"]] != list(range(11)):
            raise ValueError("Saved high priority window or one-second clock differs")
        ranks = []
        for distance, vid in near[:2]:
            positions = []
            for f in raw["frames"][6:]:
                v = f["vehicles"].get(vid)
                positions.append(length-v["position_m"] if v and v["lane_id"]==lane else None)
            ranks.append(dict(distances=positions, entry=entry.get(vid), t0_speed=start[vid]["speed"]))
        yield dict(seed=row["seed"], time_sec=row["time_sec"], point=[row["right_near"],row["straight_near"]],
                   ranks=ranks, both_stopped=all(r["t0_speed"]==0 for r in ranks))


def observed(values):
    return dict(n=len(values), median_m=median(values) if values else None)


def rank_summary(rows, index):
    ranks = [r["ranks"][index] for r in rows]
    at_frame, displacements, cumulative = [], [], []
    for j in range(5):
        at_frame.append(observed([v["distances"][j] for v in ranks if v["distances"][j] is not None]))
        if j:
            comparable = [v["distances"][j-1]-v["distances"][j] for v in ranks
                          if None not in v["distances"][j-1:j+1]]
            displacements.append(observed(comparable))
            cumulative.append(observed([v["distances"][0]-v["distances"][j] for v in ranks
                                        if None not in v["distances"][:j+1]]))
    distances = [r["distances"][-1] for r in ranks if r["entry"] is None and r["distances"][-1] is not None]
    return dict(n=len(rows),entry_histogram=dict(Counter("none" if r["entry"] is None else str(r["entry"])
                    for r in ranks)),inlet_distance_by_frame=at_frame,one_second_same_inlet_displacement=displacements,
                cumulative_same_inlet_displacement_since_green=cumulative,
                not_entered_and_still_on_inlet_at_t10=dict(**observed(distances),min_m=min(distances,default=None),
                                                            max_m=max(distances,default=None)))


def summarize(directory,cfg):
    rows = list(cases(directory,cfg))
    groups = []
    for stopped in (False,True):
        for entered in (False,True):
            members=[r for r in rows if (not stopped or r["both_stopped"]) and
                     (r["ranks"][1]["entry"] is not None)==entered]
            groups.append(dict(subset="both_t0_stopped" if stopped else "all",second_entered=entered,n=len(members),
                               first=rank_summary(members,0),second=rank_summary(members,1)))
    late = [dict(seed=r["seed"],time_sec=r["time_sec"],point=r["point"],both_t0_stopped=r["both_stopped"],
                 rank1_entry_s=r["ranks"][0]["entry"],rank2_remaining_m=r["ranks"][1]["distances"][-1])
            for r in rows if r["ranks"][1]["entry"] is None]
    return dict(n_cases_with_second=len(rows),groups=groups,rank2_not_entered=late)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("hangzhou","hangzhou-cfg","jinan","jinan-cfg"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    result=dict(offset_sec=[6,7,8,9,10],unit="meters from vehicle front to stop line, measured on original inlet",
                displacement="Positive longitudinal progress between successive samples still on the inlet in both frames. Distances after entry are excluded.",
                hangzhou=summarize(args.hangzhou,args.hangzhou_cfg),jinan=summarize(args.jinan,args.jinan_cfg))
    print(json.dumps(result,separators=(",",":")))
