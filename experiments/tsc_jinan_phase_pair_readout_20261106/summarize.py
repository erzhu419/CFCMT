"""Aggregate locked Jinan action pairs with equal weight at each sampling level."""
from collections import Counter, defaultdict
from statistics import mean


FIELDS = ("right_high", "right_low", "right_delta", "straight_high", "straight_low", "straight_delta")


def average(rows):
    return {key: mean(row[key] for row in rows) for key in FIELDS} if rows else None


def sign(delta):
    return "higher" if delta > 0 else "lower" if delta < 0 else "tie"


def summarize(rows):
    by_seed = defaultdict(lambda: defaultdict(list))
    signed_counts, executed_counts, exposure_counts = Counter(), Counter(), Counter()
    qualified, contrary, missing, incomplete_execution = [], [], [], []
    arm_mismatches = Counter()
    matching_right = 0
    for row in rows:
        state = by_seed[row["seed"]][row["case_id"]]
        pair = {field: row[field] for field in (
            "case_id", "seed", "own_movement", "priority_movement", "high_candidate", "low_candidate")}
        if row["missing"]:
            missing.append({**pair, "missing": row["missing"]})
            continue
        hi, lo = row["high"], row["low"]
        deltas = {}
        for movement in ("right", "straight"):
            key = movement + "_entries"
            deltas[movement + "_high"] = hi[key]
            deltas[movement + "_low"] = lo[key]
            deltas[movement + "_delta"] = hi[key] - lo[key]
        state.append(deltas)
        joint_sign = (sign(deltas["right_delta"]), sign(deltas["straight_delta"]))
        signed_counts[joint_sign] += 1
        for arm in ("high", "low"):
            actual = row[arm]
            if actual["right_actual"] != row["own_sequence_both"]:
                arm_mismatches[arm + "_right"] += 1
            if actual["straight_actual"] != row["priority_sequence_" + arm]:
                arm_mismatches[arm + "_straight"] += 1
        same_right = hi["right_actual"] == lo["right_actual"]
        matching_right += int(same_right)
        exposure_counts[(hi["exposure_s"], lo["exposure_s"])] += 1
        executed = (same_right and hi["exposure_s"] == row["priority_exposure_s_high"]
                    and lo["exposure_s"] == row["priority_exposure_s_low"])
        if executed:
            executed_counts[joint_sign] += 1
            if joint_sign == ("lower", "higher"):
                qualified.append(pair)
        else:
            incomplete_execution.append(pair)
        if joint_sign == ("higher", "lower"):
            contrary.append({**pair, "planned_contrast_executed": executed})

    per_seed = []
    for seed, groups in sorted(by_seed.items()):
        state_means = {case_id: average(values) for case_id, values in sorted(groups.items())}
        measured = [m for m in state_means.values() if m is not None]
        per_seed.append(dict(seed=seed, measured_pairs=sum(map(len, groups.values())),
                             measured_states=len(measured), state_means=state_means,
                             mean=average(measured)))
    qualifying_seeds = sorted({r["seed"] for r in qualified})
    if len(qualifying_seeds) >= 2:
        gate = "PASS_TARGET_READOUT"
    elif missing:
        gate = "INCOMPLETE_TARGET_READOUT"
    else:
        gate = "HOLD_TARGET_READOUT"

    def signs(counts):
        return {right: {straight: counts[right, straight] for straight in ("lower", "tie", "higher")}
                for right in ("lower", "tie", "higher")}

    return dict(
        status=gate, planned_pairs=len(rows), measured_pairs=sum(signed_counts.values()),
        missing_pairs=missing, per_seed=per_seed,
        equal_seed_mean=average([s["mean"] for s in per_seed if s["mean"] is not None]),
        aggregation="Measured movements within each case, measured cases within each seed, then measured seeds equally. Missing outcomes remain absent.",
        matching_actual_right_pairs=matching_right, executed_pairs=sum(executed_counts.values()),
        actual_exposure_counts=[dict(high_s=h, low_s=l, pairs=n) for (h,l),n in sorted(exposure_counts.items())],
        arm_timeline_mismatches={arm+"_"+move: arm_mismatches[arm+"_"+move]
                                 for arm in ("high", "low") for move in ("right", "straight")},
        missed_planned_contrast=incomplete_execution,
        signs_all_right_by_straight=signs(signed_counts), signs_executed_right_by_straight=signs(executed_counts),
        qualifying_pairs=qualified, qualifying_seeds=qualifying_seeds, contrary_pairs=contrary)
