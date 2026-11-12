"""Aggregate a single admitted Hangzhou PP hour without moving its epoch rows."""
import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path


def switchable(row):
    return row["eight_eligible"] and (row["gap_s"] or 0) > 0


def target_exposure(row):
    return row["high_exposure_s"] == 4 and row["low_exposure_s"] == 0


def outside_anchors(row):
    return all(not (a <= row["time_sec"] <= a + 60) for a in (600, 1800))


def count_pairs(rows):
    points = Counter((r["right_near"], r["straight_near"]) for r in rows)
    return dict(
        pair_ticks=sum(points.values()),
        both_empty=points[0, 0],
        right_only=sum(n for (r, s), n in points.items() if r > 0 and s == 0),
        straight_only=sum(n for (r, s), n in points.items() if r == 0 and s > 0),
        both_supplied=sum(n for (r, s), n in points.items() if r > 0 and s > 0),
        joint_histogram=[[r, s, n] for (r, s), n in sorted(points.items())],
    )


CONDITIONS = {
    "straight_ge8": lambda r, s: r > 0 and s >= 8,
    "straight_8to11": lambda r, s: r > 0 and 8 <= s <= 11,
    "straight_gt11": lambda r, s: r > 0 and s > 11,
    "right_ge2": lambda r, s: r >= 2 and s > 0,
    "right_ge2_straight_ge8": lambda r, s: r >= 2 and s >= 8,
}


def visits(by_pair, condition):
    lengths = Counter()
    n_eligible = n_switchable = n_four_zero = n_outside = 0
    with_eligible = with_switchable = with_four_zero = with_outside = 0
    tls_with_supply, tls_with_contrast = set(), set()
    for (tls, _, _), ordered in by_pair.items():
        for present, group in groupby(ordered, key=lambda r: condition(r["right_near"], r["straight_near"])):
            if not present:
                continue
            run = list(group)
            eligible = sum(r["eight_eligible"] for r in run)
            usable = [r for r in run if switchable(r)]
            four_zero = sum(target_exposure(r) for r in usable)
            outside = sum(outside_anchors(r) for r in usable)
            lengths[len(run)] += 1
            n_eligible += eligible
            n_switchable += len(usable)
            n_four_zero += four_zero
            n_outside += outside
            with_eligible += bool(eligible)
            with_switchable += bool(usable)
            with_four_zero += bool(four_zero)
            with_outside += bool(outside)
            tls_with_supply.add(tls)
            if usable:
                tls_with_contrast.add(tls)
    ticks = sum(k * n for k, n in lengths.items())
    return dict(physical_visits=sum(lengths.values()), physical_pair_ticks=ticks,
                physical_tls=len(tls_with_supply), eight_eligible_pair_ticks=n_eligible,
                switchable_pair_ticks=n_switchable, switchable_tls=len(tls_with_contrast),
                target_exposure_pair_ticks=n_four_zero, switchable_pair_ticks_outside_pilot_windows=n_outside,
                visits_with_eight_eligible_tick=with_eligible, visits_with_switchable_tick=with_switchable,
                visits_with_target_exposure_tick=with_four_zero, visits_with_switchable_tick_outside_pilot_windows=with_outside,
                sampled_seconds_per_visit_histogram=[[10 * k, n] for k, n in sorted(lengths.items())])


def summarize(directory):
    receipt = json.loads((directory / "result.json").read_text())
    if receipt["status"] != "COMPLETE":
        raise ValueError(f"Full-hour summary requires a completed trajectory, got {receipt['status']}")
    with (directory / "rows.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    by_pair, by_tls_tick = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["seed"] != receipt["seed"]:
            raise ValueError("Observation seed differs from trajectory receipt")
        by_pair[row["tls_id"], row["right_id"], row["straight_id"]].append(row)
        by_tls_tick[row["tls_id"], row["time_sec"]].append(row)
    ticks = list(range(0, 3600, 10))
    for pair, ordered in by_pair.items():
        if [row["time_sec"] for row in ordered] != ticks:
            raise ValueError(f"Incomplete or repeated epoch for {pair}")
    if len(by_pair) != 32 or len(by_tls_tick) != 16 * len(ticks):
        raise ValueError("Unexpected pair or signal census")
    eligible_tls = 0
    for pair_rows in by_tls_tick.values():
        flags = {r["eight_eligible"] for r in pair_rows}
        if len(pair_rows) != 2 or len(flags) != 1:
            raise ValueError("Expected two registry pairs sharing one signal's eligibility")
        eligible_tls += flags.pop()
    usable = [r for r in rows if switchable(r)]
    exposures = Counter((r["high_exposure_s"], r["low_exposure_s"]) for r in usable)
    return dict(seed=receipt["seed"], tls_epochs=len(by_tls_tick), eight_eligible_tls_epochs=eligible_tls,
                ineligible_tls_epochs=len(by_tls_tick) - eligible_tls,
                all_pairs=count_pairs(rows), eight_eligible_pairs=count_pairs(r for r in rows if r["eight_eligible"]),
                switchable_pairs=count_pairs(usable), gap4_pairs=count_pairs(r for r in usable if r["gap_s"] == 4),
                target_exposure_pairs=count_pairs(r for r in usable if target_exposure(r)),
                switchable_gap_s_histogram=sorted(Counter(r["gap_s"] for r in usable).items()),
                high_low_exposure_histogram=[[hi, lo, n] for (hi, lo), n in sorted(exposures.items())],
                heavy_visits={name: visits(by_pair, predicate) for name, predicate in CONDITIONS.items()},
                histogram_fields=["right_near", "straight_near", "pair_ticks"],
                visit_semantics="A visit comprises consecutive physically qualifying 10-second samples on the same directed pair. Action ineligibility alone does not split visits. Sampled seconds = ticks x 10, not an observed continuous queue duration or a count of unique vehicles. Pilot windows are inclusive [600,660] and [1800,1860].")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().seed_directory), separators=(",", ":")))
