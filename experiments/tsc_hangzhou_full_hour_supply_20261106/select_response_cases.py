"""Select preaction checkpoints for a separate, common-support Hangzhou readout."""
import argparse
import json
from pathlib import Path


def select(directory, points):
    receipt = json.loads((directory / "result.json").read_text())
    if receipt["status"] != "COMPLETE":
        raise ValueError("Selection requires a complete PP hour")
    candidates = {point: [] for point in points}
    last_seen, visit_start = {}, {}
    with (directory / "rows.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            point = row["right_near"], row["straight_near"]
            if min(point) == 0:
                continue
            pair = (row["tls_id"], row["right_id"], row["straight_id"])
            t = row["time_sec"]
            if last_seen.get(pair) != t - 10:
                visit_start[pair] = t
            last_seen[pair] = t
            if (point in candidates and row["eight_eligible"]
                    and (row["high_exposure_s"], row["low_exposure_s"]) == (4, 0)):
                candidates[point].append((t, pair, visit_start[pair], row))
    times, visits, cases = set(), set(), []
    for point, available in sorted(candidates.items()):
        for t, pair, start, row in sorted(available, key=lambda item: item[:2]):
            if t in times or (pair, start) in visits:
                continue
            times.add(t)
            visits.add((pair, start))
            cases.append(dict(seed=receipt["seed"], time_sec=t, tls_id=pair[0],
                              right_id=pair[1], straight_id=pair[2], right_near=point[0],
                              straight_near=point[1], high_candidate=row["high_candidate"],
                              low_candidate=row["low_candidate"], both_supplied_visit_start_sec=start))
            break
        else:
            raise ValueError(f"No distinct eligible visit for requested point {point}")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_directory", type=Path)
    parser.add_argument("common_points", help="JSON array of repeated exact 4/0 count pairs")
    args = parser.parse_args()
    points = [tuple(point) for point in json.loads(args.common_points)]
    print(json.dumps(select(args.seed_directory, points), separators=(",", ":")))
