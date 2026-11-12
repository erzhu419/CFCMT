"""Read the four preaction censuses on the node and select the locked Jinan roster."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def select_seed(directory, seed, points):
    receipt = json.loads((directory / "result.json").read_text())
    if receipt["status"] != "COMPLETE" or receipt["seed"] != seed or receipt["final_time_sec"] != 3600:
        raise ValueError("Missing a complete one-hour receipt for this seed")
    if any(receipt["safety"][k] for k in ("raw_collision_events", "starting_teleports", "ending_teleports")):
        raise ValueError("Unsafe trajectory")
    if (receipt["new_alternative_branches"], receipt["new_fits"]) != (0, 0):
        raise ValueError("Census exceeded the preaction budget")

    seen = Counter()
    tls_flags = defaultdict(list)
    last_supplied, visit_starts = {}, {}
    totals = Counter()
    physical, eight, options = Counter(), Counter(), {point: [] for point in points}
    with (directory / "rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            pair = row["tls_id"], row["right_id"], row["straight_id"]
            t = row["time_sec"]
            if row["seed"] != seed or t != 10 * seen[pair]:
                raise ValueError(f"Missing, duplicate, or mismatched epoch for {pair} at {t}")
            seen[pair] += 1
            eligible = row["eight_eligible"]
            tls_flags[pair[0], t].append(eligible)
            point = row["right_near"], row["straight_near"]
            totals["pair_ticks"] += 1
            if min(point) == 0:
                totals["both_empty" if max(point) == 0 else
                       "right_only" if point[0] else "straight_only"] += 1
                continue
            totals["both_supplied"] += 1
            if last_supplied.get(pair) != t - 10:
                visit_starts[pair] = t
            last_supplied[pair] = t
            contrast = eligible and (row["high_exposure_s"], row["low_exposure_s"]) == (4, 0)
            totals["both_supplied_high4_low0"] += int(contrast)
            if point not in options:
                continue
            physical[point] += 1
            eight[point] += int(eligible)
            if contrast:
                options[point].append((t, pair, visit_starts[pair], row))
    if (len(seen) != 24 or set(seen.values()) != {360}
            or len(tls_flags) != 4320 or any(len(f) != 2 or f[0] != f[1] for f in tls_flags.values())):
        raise ValueError("Expected 360 epochs for each of 24 pairs on 12 signals")
    eligible_tls = sum(flags[0] for flags in tls_flags.values())
    if (receipt["movement_rows"], receipt["observed_tls_states"], receipt["eight_eligible_tls_states"]) != (
            totals["pair_ticks"], len(tls_flags), eligible_tls):
        raise ValueError("Hour receipt and measured opportunity denominators differ")
    cases, point_rows, times, visits = [], [], set(), set()
    for point in sorted(points):
        available = options[point]
        selected = None
        for t, pair, start, row in sorted(available, key=lambda x: x[:2]):
            if t in times or (pair, start) in visits:
                continue
            selected = dict(seed=seed, time_sec=t, tls_id=pair[0], right_id=pair[1],
                            straight_id=pair[2], right_near=point[0], straight_near=point[1],
                            high_candidate=row["high_candidate"], low_candidate=row["low_candidate"],
                            right_sequence=row["right_sequence"],
                            high_straight_sequence=row["high_straight_sequence"],
                            low_straight_sequence=row["low_straight_sequence"],
                            both_supplied_visit_start_sec=start)
            cases.append(selected)
            times.add(t)
            visits.add((pair, start))
            break
        point_rows.append(dict(point=point, physical_pair_ticks=physical[point],
                               eight_eligible_pair_ticks=eight[point], high4_low0_pair_ticks=len(available),
                               visits_with_opportunity=len({(pair, start) for _, pair, start, _ in available}),
                               selected=selected is not None))
    return (dict(seed=seed, status="COMPLETE", tls_epochs=len(tls_flags), eight_eligible_tls_epochs=eligible_tls,
                 pair_denominators={k: totals[k] for k in ("pair_ticks", "both_empty", "right_only", "straight_only",
                                                           "both_supplied", "both_supplied_high4_low0")},
                 points=point_rows, missing_points=[p["point"] for p in point_rows if not p["selected"]]), cases)


def select(root, plan):
    points = sorted(tuple(p["point"]) for p in plan["source_models"]["points"])
    seeds = plan["seeds"]
    if len(set(points)) != 7 or len(points) != 7 or len(set(seeds)) != 4 or len(seeds) != 4:
        raise ValueError("Expected the seven fixed common points and four fresh seeds")
    summaries, candidates = [], []
    for seed in seeds:
        try:
            record, cases = select_seed(root / f"pp_{seed}", seed, points)
        except (FileNotFoundError, ValueError) as exc:
            summaries.append(dict(seed=seed, status="INCOMPLETE", reason=str(exc)))
            continue
        summaries.append(record)
        candidates.extend(cases)
    coverage = []
    for point in points:
        available = [r["seed"] for r in candidates if (r["right_near"], r["straight_near"]) == point]
        coverage.append(dict(point=point, selected_seeds=available, n_seeds=len(available)))
    if any(r["status"] != "COMPLETE" for r in summaries):
        status = "INCOMPLETE_FRESH_TARGET_CENSUS"
    elif min(r["n_seeds"] for r in coverage) < 2:
        status = "HOLD_FRESH_TARGET_COVERAGE"
    else:
        status = "PASS_FRESH_TARGET_OPPORTUNITY"
    return dict(status=status, seeds=seeds, count_pairs=points, per_seed=summaries, point_coverage=coverage,
                cases=candidates if status == "PASS_FRESH_TARGET_OPPORTUNITY" else [],
                selection=plan["selection"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census_root", type=Path)
    parser.add_argument("frozen_plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(args.census_root, json.loads(args.frozen_plan.read_text())), separators=(",", ":")))
