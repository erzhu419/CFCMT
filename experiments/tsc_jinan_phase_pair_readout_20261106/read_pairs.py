"""Read frozen movement pairs from native observations, returning small paired records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def movement_trace(states, links):
    sequences = {"".join(state[index] for state in states) for index in links}
    if len(sequences) != 1:
        raise ValueError("Expected a uniform signal over this frozen movement's physical links")
    return sequences.pop()


def read_branch(directory, group, candidate, tls, movements):
    name = "first_pp" if candidate == group["reference_index"] else f"action_{candidate}"
    receipt = load(directory / name / "result.json")
    if receipt["candidate_index"] != candidate:
        raise ValueError(f"{name} belongs to a different candidate")
    obs = load(directory / name / "observations_server_only.json")
    frames = obs["frames"]
    if [f["offset_sec"] for f in frames] != list(range(11)):
        raise ValueError("Incomplete 10-second observation window")
    t0 = frames[0]["time_sec"]
    if any(f["time_sec"] != t0 + i for i, f in enumerate(frames)):
        raise ValueError("Inconsistent native observation time")
    actual = [f["applied_states"][tls] for f in frames[1:]]
    nominal = obs["preaction"]["nominal_states"][tls]
    counted = Counter(e["movement_id"] for e in obs["events"] if e["kind"] == "stopline_entry"
                      and t0 < e["time_sec"] <= t0 + 10)
    # This lookup also covers the reused PP branch whose v1 receipt recorded HOLD.
    # collection_v2's complete group contains the accepted observation-derived labels.
    branch, = (b for b in group["branches"] if b["candidate_index"] == candidate)
    out = {}
    for movement in movements:
        links = obs["catalog"][movement]["link_indices"]
        count = counted[movement]
        if count != branch["labels"][group["movement_ids"].index(movement)][0]:
            raise ValueError(f"Event count and saved stop-line label disagree for {movement}")
        out[movement] = (count, movement_trace(actual, links), movement_trace(nominal, links))
    return t0, out


def read_pairs(roster):
    root = Path(roster["collection"])
    cases = defaultdict(list)
    for row in roster["rows"]:
        cases[row["case_id"]].append(row)
    rows = []
    for case_id, selected in cases.items():
        directory = root / f"seed_{selected[0]['seed']}" / case_id
        try:
            group = load(directory / "group_data_server_only.json")
            if group["case_id"] != case_id or not group["complete"]:
                raise ValueError(f"Incomplete or mismatched group for {case_id}")
        except FileNotFoundError as exc:
            rows.extend(dict(r, missing=[str(exc.filename)]) for r in selected)
            continue
        needed = defaultdict(set)
        for r in selected:
            for arm in ("high", "low"):
                needed[r[f"{arm}_candidate"]].update((r["own_movement"], r["priority_movement"]))
        observations, absent = {}, {}
        for candidate, movements in needed.items():
            try:
                observations[candidate] = read_branch(directory, group, candidate, selected[0]["tls_id"], movements)
            except FileNotFoundError as exc:
                absent[candidate] = str(exc.filename)
        for r in selected:
            row = dict(r, missing=[])
            for arm in ("high", "low"):
                candidate = r[f"{arm}_candidate"]
                if candidate in absent:
                    row["missing"].append(absent[candidate])
                    continue
                t0, response = observations[candidate]
                right, straight = response[r["own_movement"]], response[r["priority_movement"]]
                if (right[2], straight[2]) != (r["own_sequence_both"], r[f"priority_sequence_{arm}"]):
                    raise ValueError("Branch nominal signals disagree with the prespecified pair")
                row[arm] = dict(checkpoint_sec=t0, right_entries=right[0], straight_entries=straight[0],
                                right_actual=right[1], straight_actual=straight[1],
                                exposure_s=sum(a == "g" and b == "G" for a, b in zip(right[1], straight[1])))
            if not row["missing"]:
                if row["high"]["checkpoint_sec"] != row["low"]["checkpoint_sec"]:
                    raise ValueError("Frozen pair refers to different native times")
                row["delta_right_entries"] = row["high"]["right_entries"] - row["low"]["right_entries"]
                row["delta_straight_entries"] = row["high"]["straight_entries"] - row["low"]["straight_entries"]
                row["delta_exposure_s"] = row["high"]["exposure_s"] - row["low"]["exposure_s"]
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    print(json.dumps(read_pairs(load(parser.parse_args().roster)), separators=(",", ":")))
