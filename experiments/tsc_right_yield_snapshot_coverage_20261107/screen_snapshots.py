"""Check the archived PP decisions without reading labels or advancing a traffic simulation.

Uses lane_map and choice from the previous right-yield screen, and the development
archive's source_pilot and nominal signal-plan helpers on PYTHONPATH.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from coverage_summary import summarize
from read_inputs import lane_map, read
from rebuild_requests import action_contrast, load_programs, rebuild_requests, validate_focal
from screen_rules import POINTS, SEEDS, choice


def movement_key(row):
    return row["tls_id"], row["right_id"], row["straight_id"]


def census_at_saved_times(cases):
    times_by_file = defaultdict(set)
    for case in cases:
        times_by_file[Path(case["start_file"]).with_name("rows.jsonl")].add(case["time_sec"])
    recorded = {}
    for path, times in times_by_file.items():
        with path.open() as stream:
            for line in stream:
                rec = json.loads(line)
                if rec["time_sec"] in times:
                    recorded[(rec["seed"], rec["time_sec"], *movement_key(rec))] = rec
    return recorded


def lane_supply(start, geometry):
    halted = Counter()
    near = defaultdict(list)
    for vid, vehicle in start["vehicles"].items():
        lane, speed = vehicle["lane_id"], vehicle["speed"]
        halted[lane] += speed < 0.1
        if lane in geometry:
            length, limit = geometry[lane]
            distance = length - vehicle["lane_position"]
            if distance <= 10 * limit:
                near[lane].append((distance, vid, speed))
    return halted, near


def inspect_city(city, cases, registry, cfg, audit, blocked):
    if ({(c["seed"], tuple(c["point"])) for c in cases} != {(s, p) for s in SEEDS[city] for p in POINTS}
            or len({(c["seed"], c["time_sec"]) for c in cases}) != len(cases)
            or len(cases) != len(SEEDS[city]) * len(POINTS)):
        raise ValueError(f"{city}: expected the sealed six distinct epochs per development seed")
    recorded = census_at_saved_times(cases)
    programs = load_programs(cfg)
    links, directions, geometry = lane_map(cfg)
    registry_count = 0
    for tls, pairs in registry.items():
        for p in pairs:
            right, = p["right_lanes"]
            straight, = p["straight_lanes"]
            if (directions[right] != {"r"} or directions[straight] != {"s"} or not p["same_lane_link_pairs"]
                    or {links[tls][i] for i in p["right_links"]} != {right}
                    or {links[tls][i] for i in p["straight_links"]} != {straight}):
                raise ValueError(f"{tls}: registry does not bind dedicated right and straight inlets")
            registry_count += 1
    if registry_count != {"hangzhou": 32, "jinan": 24}[city]:
        raise ValueError(f"{city}: incomplete repaired registry")

    rows = []
    for case in cases:
        start = read(Path(case["start_file"]))
        if start["time_sec"] != case["time_sec"]:
            raise ValueError("Start time differs from the sealed roster")
        halted, near = lane_supply(start, geometry)
        for tls, pairs in registry.items():
            clock = start["executors"].get(tls)
            states = plans = None
            if clock is not None:
                states, plans = rebuild_requests(programs[tls], tls, clock)
            pp_state = start["pp_actions"].get(tls)
            pp = states.index(pp_state) if states is not None and pp_state is not None else None
            for pair in pairs:
                right, = pair["right_lanes"]
                straight, = pair["straight_lanes"]
                leaders = sorted(near[straight])[:2]
                stopped = len(leaders) == 2 and all(v[2] == 0.0 for v in leaders)
                n = sum(v[2] < 0.1 for v in near[right])
                point = len(near[right]), len(near[straight])
                focal = (tls, pair["right_id"], pair["straight_id"]) == movement_key(case)
                eight = None if states is None else len(states) == 8
                contrast = action_contrast(plans, pair) if eight else None
                four_zero = (None if not eight else contrast is not None and
                             (contrast["high_exposure_s"], contrast["low_exposure_s"]) == (4, 0))
                row = dict(city=city, seed=case["seed"], time_sec=case["time_sec"], tls_id=tls,
                           right_id=pair["right_id"], straight_id=pair["straight_id"], point=point,
                           focal=focal, n=n, stopped_right=n>0, both_straight_stopped=stopped,
                           allowed_point=point in POINTS, eight=eight, contrast=four_zero, pp=pp,
                           pp_on_pair=None, sensitive=None)
                if four_zero:
                    for label, k in (("hi", "high_candidate"), ("lo", "low_candidate")):
                        idx = contrast[k]
                        greens = {links[tls][j] for j, ch in enumerate(states[idx]) if ch in "Gg"}
                        if right not in greens:
                            raise ValueError("Expected the permissive right inlet in both requested phases")
                        row[label] = idx
                        row["b_" + label] = sum(halted[lane] for lane in greens)
                    row["pp_on_pair"] = None if pp is None else pp in (row["hi"], row["lo"])
                    row["sensitive"] = choice(row, 0)[0] != choice(row, 1)[0]

                # Nonfocal junctions can be in a minimum-green or clearance interval
                # never represented by the original 36 nominated (fully open) examples.
                rec = recorded[(case["seed"], case["time_sec"], *movement_key(row))]
                if point != (rec["right_near"], rec["straight_near"]):
                    raise ValueError(f"Vehicle supply differs from archived census: {movement_key(row)}")
                if eight is not None:
                    if (eight != rec["eight_eligible"] or (contrast is None) != (rec["high_candidate"] is None)
                            or (contrast and any(rec[k] != v for k, v in contrast.items()))):
                        raise ValueError(f"Nominal requests differ from archived census: {movement_key(row)}")
                    audit["matched_preaction_census_rows"] += 1
                if "straight_target_ids" in rec:
                    ids = [v[1] for v in leaders] if stopped else []
                    if rec["straight_target_ids"] != ids:
                        raise ValueError("Archived straight leader identities differ")
                if focal:
                    if point != tuple(case["point"]) or not stopped or [v[1] for v in leaders] != case["straight_target_ids"]:
                        raise ValueError("Sealed focal vehicle cohort differs")
                    if states is not None:
                        validate_focal(states, plans, case, start, pair)
                        audit["matched_focal_requests_and_vehicles"] += 1
                if n and ((city == "hangzhou" and not stopped) or (city == "jinan" and stopped)):
                    detail = {k: row[k] for k in ("seed", "time_sec", "tls_id", "right_id", "straight_id",
                                                   "point", "focal", "n", "allowed_point", "eight")}
                    detail["straight_leader_speeds_mps"] = [v[2] for v in leaders]
                    if clock is not None:
                        timing = next(t for t in programs[tls] if t.state == clock["current_green_state"])
                        detail.update(mode=clock["mode"], feasible_requests=len(states),
                                      elapsed_green_s=clock["green_elapsed_sec"], min_green_s=timing.min_green_sec)
                    blocked.append(detail)
                rows.append(row)
    if len(recorded) != len(rows):
        raise ValueError("Snapshot registry and native census cover different movements")
    return rows


def run(root):
    roster = read(root / "census/roster.json")
    if roster["status"] != "PASS_DEV_ROSTER":
        raise ValueError("Expected sealed development snapshots")
    rows, checks, blocked = [], {}, {}
    for city, role in (("hangzhou", "source"), ("jinan", "target")):
        audit = Counter()
        blocked[city] = []
        rows.extend(inspect_city(city, roster[role+"_cases"], read(Path(roster[role+"_registry"]))["tls"],
                                 Path(roster[role+"_cfg"]), audit, blocked[city]))
        checks[city] = dict(audit)
    return dict(**summarize(rows), source_root=str(root), preaction_checks=checks,
                hangzhou_stopped_right_without_straight_pair=blocked["hangzhou"],
                jinan_joint_queue_states=blocked["jinan"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_root", type=Path)
    print(json.dumps(run(parser.parse_args().development_root), separators=(",", ":")))
