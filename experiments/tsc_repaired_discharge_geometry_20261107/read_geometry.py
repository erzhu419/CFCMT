"""Read original second-car positions at decision, priority onset and second nine."""
import argparse
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def profile(rows):
    fields = ("d0_m", "pregreen_m", "green_to9_m", "d9_m")
    return dict(n=len(rows), crossed=sum(r["entry_s"] is not None for r in rows),
                means={f: mean(r[f] for r in rows) for f in fields},
                ranges={f: [min(r[f] for r in rows), max(r[f] for r in rows)] for f in fields})


def summarize(cases):
    result = {}
    for group in ("source_aligned", "target_native"):
        rows = [r for r in cases if r["group"] == group]
        result[group] = dict(
            overall=profile(rows),
            by_seed=[dict(seed=s, **profile([r for r in rows if r["seed"] == s]))
                     for s in sorted({r["seed"] for r in rows})],
            by_point=[dict(point=p, **profile([r for r in rows if tuple(r["point"]) == p]))
                      for p in sorted({tuple(r["point"]) for r in rows})])
    return result


def extract(root):
    arms = root / "arms"
    if read(arms / "result.json")["status"] != "PASS_DEV_ARMS":
        raise ValueError("Expected the completed repaired-network development run")
    roster = read(root / "census/roster.json")
    selected = {}
    for role, group in (("target", "target_native"), ("source", "source_aligned")):
        for case in roster[role + "_cases"]:
            selected[group, case["seed"], tuple(case["point"])] = case

    cases = []
    for index, row in enumerate(read(arms / "observations.json")["rows"], 1):
        group = row["group"]
        if group == "source_native":
            continue
        case = selected.pop((group, row["seed"], tuple(row["point"])))
        if row["applied_accel_mps2"] != [2.0, 2.0] or any(
            row[k] != case[k] for k in ("time_sec", "tls_id", "straight_id", "straight_lane", "straight_target_ids")
        ):
            raise ValueError("Arm differs from the sealed same-acceleration case")
        folder = arms / f"arm_{index:02d}"
        raw = read(folder / "observations_server_only.json")
        frames = raw["frames"]
        lane = case["straight_lane"]
        length = raw["lane_geometry"][lane]["length_m"]
        first, second = case["straight_target_ids"]
        signals = read(folder / "arm.json")
        priority = [t for t, chars in enumerate(zip(signals["right_actual"], signals["straight_actual"]), 1)
                    if chars == ("g", "G")]
        if priority != row["priority_seconds"] or not priority:
            raise ValueError("Actual priority window differs from the completed observation")
        base = priority[0] - 1
        if priority != list(range(base + 1, 11)):
            raise ValueError("Cannot treat onset through second nine as a single green window")
        if [f["time_sec"] - case["time_sec"] for f in frames] != list(range(11)):
            raise ValueError("Recorded frames differ from the one-second observation clock")

        def distance(vehicle, t):
            sample = frames[t]["vehicles"][vehicle]
            if sample["lane_id"] != lane:
                raise ValueError(f"{vehicle} has left the original inlet at second {t}")
            return length - sample["position_m"]

        d0, d_green, d9 = (distance(second, t) for t in (0, base, 9))
        starts = read(folder / "physical_start_server_only.json")["vehicles"]
        if distance(first, 0) != length - starts[first]["lane_position"] or d0 != length - starts[second]["lane_position"]:
            raise ValueError("Frame zero differs from the archived preaction positions")
        entries = [int(e["time_sec"] - case["time_sec"]) for e in raw["events"]
                   if (e["kind"], e["movement_id"], e["vehicle_id"]) == ("stopline_entry", case["straight_id"], second)]
        endpoint = row["second"]
        if entries != ([] if endpoint["entry_s"] is None else [endpoint["entry_s"]]):
            raise ValueError("Native second-car event disagrees with the sealed development label")
        remaining = None if entries else distance(second, 10)
        if remaining != endpoint["remaining_t10_m"] or d_green - d9 != endpoint["pregreen_to_t9_m"]:
            raise ValueError("Trajectory disagrees with the original readout")
        cases.append(dict(arm=index, group=group, seed=row["seed"], point=row["point"], time_sec=row["time_sec"],
                          second_id=second, first_d0_m=distance(first, 0), d0_m=d0, green_base_s=base,
                          pregreen_m=d0-d_green, d_green_m=d_green, green_to9_m=d_green-d9, d9_m=d9,
                          entry_s=endpoint["entry_s"], remaining_t10_m=remaining))
    if selected or len(cases) != 36:
        raise ValueError("Incomplete sealed case panel")
    return dict(source=str(root), units="metres from vehicle front to original inlet stop line",
                summary=summarize(cases), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.development_root), separators=(",", ":"), allow_nan=False))
