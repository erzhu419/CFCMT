"""Summarize simultaneous near-lane supply in the two completed PP trajectories."""
from collections import Counter
import json
from pathlib import Path
from statistics import median


def summarize_supply(pilot, target):
    source = {}
    all_points = []
    for seed in (31061101, 31061102):
        path = pilot / f"pp_{seed}"
        receipt = json.loads((path / "result.json").read_text())
        if receipt["status"] != "COMPLETE":
            raise ValueError(f"Incomplete PP trajectory for seed {seed}")
        states = json.loads((path / "states.json").read_text())
        points = [(p["right_near"], p["straight_near"]) for s in states if s["status"] == "MEASURED"
                  for p in s["pairs"] if p["contrast"] and p["right_near"] and p["straight_near"]]
        if len(points) != receipt["supply"]["supplied_switchable_group_pairs"]:
            raise ValueError("Supplied movement-pair count differs from run receipt")
        source[seed] = dict(scheduled_states=len(states), state_status=receipt["supply"]["state_status"],
                            joint_states=receipt["supply"]["supplied_switchable_states"],
                            point_counts=[[r, t, n] for (r, t), n in sorted(Counter(points).items())])
        all_points.extend(points)
    target_points = [(p["own_near"], p["priority_near"]) for p in target]
    source_support = set(all_points)
    shared = [r for r in target if (r["own_near"], r["priority_near"]) in source_support]
    def executed(r):
        return (r["high"]["right_actual"] == r["low"]["right_actual"]
                and r["high"]["exposure_s"] == r["priority_exposure_s_high"]
                and r["low"]["exposure_s"] == r["priority_exposure_s_low"])
    def tradeoff(r):
        return executed(r) and r["delta_right_entries"] < 0 and r["delta_straight_entries"] > 0
    def spans(points):
        return {name: [min(v), median(v), max(v)] for name, v in zip(("right", "straight"), zip(*points))}
    result = dict(source_by_seed=source, source_pairs=len(all_points), target_pairs=len(target_points),
                  source_min_median_max=spans(all_points), target_min_median_max=spans(target_points),
                  point_fields=["right_near", "straight_near", "number_of_state_movement_pairs"],
                  target_point_counts=[[r,t,n] for (r,t),n in sorted(Counter(target_points).items())],
                  target_exact_supply_matches=len(shared), executed_target_exact_matches=sum(map(executed, shared)),
                  target_tradeoffs_in_exact_matches=sum(map(tradeoff, shared)),
                  target_tradeoffs_total=sum(map(tradeoff, target)),
                  interpretation="Exact near-count matches describe these two panels. Matching the two counts alone does not identify a shared state or an effect of transfer.")
    if all_points:
        high = max(s for _, s in all_points)
        result["target_straight_above_source_max"] = sum(s > high for _,s in target_points)
        result["target_seeds_with_straight_above_source_max"] = sorted(
            {row["seed"] for row in target if row["priority_near"] > high})
    return result
