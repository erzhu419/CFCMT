"""Score the frozen vehicles using native stop-line events and 1-second position samples."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def load(path):
    return json.loads(path.read_text())


def vehicle_result(frames, entries, vid, lane, length, green_base):
    distances = []
    for f in frames:
        v = f["vehicles"].get(vid)
        distances.append(length - v["position_m"] if v and v["lane_id"] == lane else None)
    entry = entries.get(vid)
    if entry is None and distances[10] is None:
        raise ValueError("Non-crossing frozen vehicle has no observed endpoint on its original inlet")

    def displacement(begin):
        if begin is not None and begin <= 9 and all(x is not None for x in distances[begin:10]):
            return distances[begin] - distances[9]
        return None

    return dict(entry_s=entry, remaining_t10_m=distances[10] if entry is None else None,
                t0_to_t9_m=displacement(0), pregreen_to_t9_m=displacement(green_base))


def arm_data(path, case, targets, lane, length):
    arm = load(path / "arm.json")
    raw = load(path / "observations_server_only.json")
    frames = raw["frames"]
    if (arm["status"] != "COMPLETE" or arm["arm"] != "high" or arm["seed"] != case["seed"]
            or arm["time_sec"] != case["time_sec"] or arm["candidate_index"] != case["high_candidate"]
            or [f["time_sec"] for f in frames] != list(range(case["time_sec"], case["time_sec"] + 11))
            or raw["catalog"][case["straight_id"]]["incoming_lanes"] != [lane]
            or raw["lane_geometry"][lane]["length_m"] != length):
        raise ValueError("Arm does not match the frozen movement, state or horizon")
    segments = [e for e in raw["events"] if e["kind"] == "stopline_entry" and e["movement_id"] == case["straight_id"]]
    entries = {e["vehicle_id"]: int(e["time_sec"] - case["time_sec"]) for e in segments}
    if (len(entries) != len(segments) or len(entries) != arm["straight_entries"]
            or any(not 1 <= t <= 10 for t in entries.values())):
        raise ValueError("Native straight stop-line events disagree with the completed arm")
    priority = [i for i, chars in enumerate(zip(arm["right_actual"], arm["straight_actual"]), 1) if chars == ("g", "G")]
    green_base = priority[0] - 1 if priority else None
    safety = {e: sum(arm[w][e] for w in ("prefix_safety","safety"))
              for e in ("raw_collision_events","starting_teleports","ending_teleports")}
    if any(safety.values()):
        raise ValueError("Unsafe prefix or action arm")
    metrics = dict(priority_seconds=priority, right_signal=arm["right_actual"], straight_signal=arm["straight_actual"],
                   clearance_extension_steps=arm["clearance_extension_steps"], straight_entries=len(entries),
                   right_entries=arm["right_entries"], safety=safety)
    for label, vid in zip(("first", "second"), targets):
        metrics[label] = vehicle_result(frames, entries, vid, lane, length, green_base)
    return metrics, raw


def read_pair(original_dir, swapped_dir, case):
    intervention = load(swapped_dir / "intervention_server_only.json")
    ids = intervention["target_ids"]
    lane, length = intervention["lane_id"], intervention["lane_length_m"]
    start = load(original_dir / "physical_start_server_only.json")["vehicles"]
    ordered = sorted((length - v["lane_position"], k) for k, v in start.items() if v["lane_id"] == lane)
    if ids != [vid for _, vid in ordered[:2]] or any(start[vid]["speed"] != 0 for vid in ids):
        raise ValueError("Intervention target IDs differ from the two frozen stopped vehicles")
    native, original = arm_data(original_dir, case, ids, lane, length)
    swap, changed = arm_data(swapped_dir, case, ids, lane, length)
    if original["nominal"] != changed["nominal"]:
        raise ValueError("Nominal high action differs from archived request")
    deviations = {}
    for t, (a,b) in enumerate(zip(original["frames"][1:], changed["frames"][1:]), 1):
        before, after = a["applied_states"], b["applied_states"]
        if set(before) != set(after):
            raise ValueError("Recorded signal IDs differ between arms")
        for tls in before:
            if before[tls] != after[tls]:
                deviations.setdefault(tls, []).append(t)
    return dict(city=case["city"], seed=case["seed"], time_sec=case["time_sec"], tls_id=case["tls_id"],
                original=native, swapped=swap, actual_signal_differences=deviations,
                native_accel_mps2=intervention["native_accel_mps2"], swapped_accel_mps2=intervention["swapped_accel_mps2"],
                unaffected_vehicles=intervention["unaffected_vehicles"])


def profile(rows):
    outcomes = Counter()
    progress = []
    for r in rows:
        old, new = (r[k]["second"] for k in ("original","swapped"))
        crossed = old["entry_s"] is not None, new["entry_s"] is not None
        outcomes[{(False,True):"gained",(True,False):"lost",(True,True):"both_crossed",
                  (False,False):"neither_crossed"}[crossed]] += 1
        if old["t0_to_t9_m"] is not None and new["t0_to_t9_m"] is not None:
            progress.append(new["t0_to_t9_m"] - old["t0_to_t9_m"])
    result = dict(measured=len(rows), **{k: outcomes[k] for k in ("gained","lost","both_crossed","neither_crossed")},
                  progress_t0_to_t9_paired_n=len(progress), progress_t0_to_t9_paired_mean_m=mean(progress) if progress else None,
                  progress_t0_to_t9_paired_median_m=median(progress) if progress else None,
                  actual_signal_changed_pairs=sum(bool(r["actual_signal_differences"]) for r in rows))
    for arm in ("original","swapped"):
        second = [r[arm]["second"] for r in rows]
        gaps = [m["remaining_t10_m"] for m in second if m["remaining_t10_m"] is not None]
        result[arm] = dict(second_entries=sum(m["entry_s"] is not None for m in second),
                           observed_entry_seconds=dict(Counter(m["entry_s"] for m in second if m["entry_s"] is not None)),
                           noncrossers_with_gap=len(gaps), median_gap_m=median(gaps) if gaps else None,
                           first_entries=sum(r[arm]["first"]["entry_s"] is not None for r in rows))
    return result


def summarize(plan, folder):
    rows = load(folder / "pairs.json")
    receipt = load(folder / "result.json")

    def key(r):
        return r["city"], r["seed"], r["time_sec"], r["tls_id"], tuple(r["point"])

    expected = [key(r) for r in plan["roster"]]
    if [key(r) for r in rows] != expected[:len(rows)]:
        raise ValueError("Results are not an ordered subset of the frozen roster")
    complete = receipt["status"] == "COMPLETE" and len(rows) == len(expected) == 24
    result = dict(status="COMPLETE_ACCEL_SWAP" if complete else "INCOMPLETE_ACCEL_SWAP",
                  measured=len(rows), planned=len(expected), missing=plan["roster"][len(rows):],
                  t9_measurement="Forward metres on the same inlet from t0 to t+9; pregreen_to_t9 starts at the last frame before the actual g/G onset. A vehicle leaving the inlet before t+9 returns null.",
                  per_city={}, per_seed=[])
    for city in ("hangzhou","jinan"):
        group = [r for r in rows if r["city"] == city]
        result["per_city"][city] = profile(group) if group else None
        for seed in sorted({r["seed"] for r in plan["roster"] if r["city"]==city}):
            subset = [r for r in group if r["seed"] == seed]
            result["per_seed"].append(dict(city=city, seed=seed, scheduled=sum(k[:2] == (city, seed) for k in expected),
                                           observed=profile(subset) if subset else None))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(load(args.plan), args.output_dir), separators=(",",":")))
