"""Read the frozen 14 Hangzhou pairs and the common-support Jinan observations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys


EXPERIMENTS = Path(__file__).resolve().parents[1]
JINAN = EXPERIMENTS / "tsc_jinan_phase_pair_readout_20261106"
ROSTER = EXPERIMENTS / "tsc_hangzhou_full_hour_supply_20261106" / "response_roster.json"
sys.path.insert(0, str(JINAN))
from summarize import summarize as jinan_summary


IDENTITY = ("seed", "time_sec", "tls_id", "right_id", "straight_id", "right_near",
            "straight_near", "high_candidate", "low_candidate")


def deltas(rows):
    return {move: mean(r["delta_" + move] for r in rows) for move in ("right", "straight")}


def sign(count):
    return "lower" if count < 0 else "higher" if count > 0 else "tie"


def measured_pair(row):
    exposures = []
    for arm in ("high", "low"):
        obs = row[arm]
        if any(len(obs[k]) != 10 for k in ("right_actual", "straight_actual")):
            raise ValueError("Expected ten measured one-second signals in both arms")
        exposure = sum(r == "g" and s == "G" for r, s in zip(obs["right_actual"], obs["straight_actual"]))
        if "exposure_s" in obs and obs["exposure_s"] != exposure:
            raise ValueError("Reported priority exposure differs from the measured signals")
        if any(type(obs[k]) is not int or obs[k] < 0 for k in ("right_entries", "straight_entries")):
            raise ValueError("Stop-line entries must be measured nonnegative counts")
        exposures.append(exposure)
    hi, lo = row["high"], row["low"]
    dr = hi["right_entries"] - lo["right_entries"]
    ds = hi["straight_entries"] - lo["straight_entries"]
    same_right = hi["right_actual"] == lo["right_actual"]
    executed = same_right and exposures == [4, 0]
    return dict(seed=row["seed"], near_counts=[row["right_near"], row["straight_near"]],
                exposure_high_low_s=exposures, same_right=same_right, realized_4_0=executed,
                delta_right=dr, delta_straight=ds, joint_sign=f"right_{sign(dr)}_straight_{sign(ds)}",
                qualifying_tradeoff=executed and dr < 0 < ds)


def profile(rows):
    return dict(measured_pairs=len(rows), realized_4_0=sum(r["realized_4_0"] for r in rows),
                qualifying_tradeoffs=sum(r["qualifying_tradeoff"] for r in rows),
                exposure_counts=[dict(high=h, low=l, n=n) for (h, l), n in sorted(
                    Counter(tuple(r["exposure_high_low_s"]) for r in rows).items())],
                joint_sign_counts=dict(sorted(Counter(r["joint_sign"] for r in rows).items())),
                realized_4_0_joint_sign_counts=dict(sorted(Counter(
                    r["joint_sign"] for r in rows if r["realized_4_0"]).items())))


def jinan_reference(target, points):
    subset = [r for r in target if (r["own_near"], r["priority_near"]) in points]
    legacy = jinan_summary(subset)
    views = []
    by_point = defaultdict(list)
    for r in subset:
        view = measured_pair(dict(r, right_near=r["own_near"], straight_near=r["priority_near"]))
        views.append(dict(view, case_id=r["case_id"], right_id=r["own_movement"]))
        by_point[tuple(view["near_counts"])].append(view)
    per_point = [dict(near_counts=list(p), n=len(records), mean_delta=deltas(records))
                 for p, records in sorted(by_point.items())]
    return dict(**profile(views), states=sum(s["measured_states"] for s in legacy["per_seed"]),
                qualifying_seeds=legacy["qualifying_seeds"], cases=views,
                matched_target_seeds=len(legacy["per_seed"]), equal_seed_state_movement_mean=legacy["equal_seed_mean"],
                per_point=per_point, equal_point_mean_delta={
                    move: mean(p["mean_delta"][move] for p in per_point) for move in ("right", "straight")},
                aggregation="Primary Jinan means weight movements within states, states within seeds, then seeds equally. Equal-point means separately average observed pairs within count points, then weight points equally.")


def summarize(source, roster, target):
    expected = {(r["seed"], r["time_sec"]): r for r in roster["cases"]}
    points = {tuple(p) for p in roster["count_pairs"]}
    cases, seen = [], set()
    for row in source:
        key = row["seed"], row["time_sec"]
        if key in seen or key not in expected or any(row[k] != expected[key][k] for k in IDENTITY):
            raise ValueError(f"Result differs from frozen Hangzhou roster or is duplicated: {key}")
        seen.add(key)
        cases.append(dict(measured_pair(row), time_sec=row["time_sec"], tls_id=row["tls_id"]))
    cases.sort(key=lambda r: (r["seed"], r["near_counts"]))
    missing = [dict(seed=s, time_sec=t) for s, t in sorted(expected.keys() - seen)]
    seeds = sorted({seed for seed, _ in expected})
    per_seed = []
    for seed in seeds:
        measured = [r for r in cases if r["seed"] == seed]
        complete = {tuple(r["near_counts"]) for r in measured} == points
        per_seed.append(dict(seed=seed, **profile(measured),
                             equal_point_mean_delta=deltas(measured) if complete else None))
    qualifying = [r["seed"] for r in per_seed if r["qualifying_tradeoffs"]]
    if missing:
        decision = "INCOMPLETE_MATCHED_RESPONSE"
    elif qualifying == seeds:
        decision = "PASS_REPEATED_LOCAL_RESPONSE"
    else:
        decision = "CLOSE_LOCAL_REPLICATION"
    return dict(status=decision, missing=missing, planned_pairs=len(expected),
                hangzhou=dict(**profile(cases), qualifying_seeds=qualifying, per_seed=per_seed,
                              equal_seed_equal_point_mean_delta=None if missing else {
                                  move: mean(r["equal_point_mean_delta"][move] for r in per_seed)
                                  for move in ("right", "straight")}, cases=cases),
                jinan_common=jinan_reference(target, points))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pairs", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.source_pairs.read_text()), json.loads(ROSTER.read_text()),
                       json.loads((JINAN / "pairs.json").read_text()))
    print(json.dumps(result, indent=2))
