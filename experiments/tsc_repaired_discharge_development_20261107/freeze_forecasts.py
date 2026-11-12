"""Freeze repaired-network discharge forecasts from the complete sealed development panel."""
import argparse
import json
from pathlib import Path

from score_fresh import CANDIDATE, MATCHED_TARGET, predictions


CONTROLS = ["T", "T_pool", MATCHED_TARGET, "(T+A_pool)/2", "(T+H)/2"]
GROUPS = ("target_native", "source_aligned", "source_native")
IDENTITY = ("time_sec", "tls_id", "right_id", "straight_id", "straight_target_ids")


def freeze(plan, roster, arms):
    points = [tuple(p) for p in plan["points_right_straight"]]
    target = plan["target_development_seeds"]
    source = plan["source_seeds"]
    reserved = plan["later_independent_prediction"]["reserved_seeds"]
    if (len(set(points)) != 6 or len(set(target)) != 4 or len(set(source)) != 2
            or set(reserved) & (set(target) | set(source))):
        raise ValueError("Unexpected development points or seeds")
    frozen = {}
    for group, seeds in zip(GROUPS, (target, source, source)):
        cases = roster["target_cases" if group == "target_native" else "source_cases"]
        if [(r["seed"], tuple(r["point"])) for r in cases] != [(s, p) for s in seeds for p in points]:
            raise ValueError(f"Incomplete or unordered frozen roster for {group}")
        for case in cases:
            frozen[group, case["seed"], tuple(case["point"])] = case

    events = {}
    for row in arms:
        key = row["group"], row["seed"], tuple(row["point"])
        case = frozen[key]
        if key in events or any(row[k] != case[k] for k in IDENTITY):
            raise ValueError(f"Repeated or mismatched development arm {key}")
        crossed = row["second_crossed"]
        entry = row["second"]["entry_s"]
        if (type(crossed) is not bool or crossed != (entry is not None)
                or (entry is not None and not 1 <= entry <= 10)):
            raise ValueError(f"Invalid native second-car entry indicator at {key}")
        events[key] = crossed
    if len(events) != 48 or events.keys() != frozen.keys():
        raise ValueError("Incomplete development-arm panel")

    totals = {g: sum(value for (group, _, _), value in events.items() if group == g) for g in GROUPS}
    counts = []
    for point in points:
        k_t, k_a, k_h = (sum(events[g, s, point] for s in seeds)
                         for g, seeds in zip(GROUPS, (target, source, source)))
        counts.append([*point, len(target), k_t, len(source), k_a, k_h])
    output = {
        "status": "FORECASTS_FROZEN",
        "target_cfg": plan["target_cfg"], "source_cfg": plan["source_cfg"],
        "target_seeds": reserved,
        "development_counts": {
            "columns": ["right_near", "straight_near", "jinan_n", "jinan_native_entries",
                        "hangzhou_n", "hangzhou_at_2.0_entries", "hangzhou_at_2.6_entries"],
            "rows": counts},
        "frozen_predictions": {"candidate": CANDIDATE, "controls": CONTROLS},
        "outcome_point_order": points,
        "per_seed": [
            {"seed": seed, "group": group, "second_entries": [int(events[group, seed, p]) for p in points]}
            for group, seeds in zip(GROUPS, (target, source, source)) for seed in seeds],
    }
    probabilities = predictions(output)
    output["point_forecasts"] = [
        {"point": p, **{name: float(value) for name, value in probabilities[p].items()}} for p in points]
    output["case_pooled_rates"] = {}
    for g, seeds in zip(GROUPS, (target, source, source)):
        n = len(seeds) * len(points)
        output["case_pooled_rates"][g] = {"crossed": totals[g], "n": n, "rate": totals[g] / n}
    output["controls_identical_to_candidate"] = [
        ctrl for ctrl in CONTROLS if all(f[CANDIDATE] == f[ctrl] for f in probabilities.values())]
    output["strict_superiority_precluded_by_forecast_equality"] = bool(output["controls_identical_to_candidate"])
    output["independent_labels_used"] = 0
    return output


def read(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "roster", "measurement", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = read(args.measurement)
    if receipt["status"] != "PASS_DEV_ARMS":
        raise ValueError("Forecast freeze requires a completed development run")
    observations = read(args.measurement.with_name("observations.json"))
    result = freeze(read(args.plan), read(args.roster), observations["rows"])
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "controls_identical_to_candidate": result["controls_identical_to_candidate"]}))
