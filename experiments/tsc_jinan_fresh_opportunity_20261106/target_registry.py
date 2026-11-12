"""Admit the original Jinan v9 network for the frozen preaction census."""
import argparse
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET


def build(cfg, roster, helper_dir, output):
    # Use the very same dedicated-lane and physical-priority reader as the source.
    sys.path.insert(0, str(helper_dir))
    from build_source_package import input_file, inspect, write

    summary, _, _, _, pairs = inspect(cfg, "g")
    controls = ET.parse(summary["net"]).findall("tlLogic")
    uniform = True
    for program in controls:
        for phase in program.findall("phase"):
            state = phase.get("state")
            if "y" in state.lower() or not any(c in "Gg" for c in state):
                continue
            for row in pairs[program.get("id")]:
                for movement in ("right_links", "straight_links"):
                    uniform &= len({state[index] for index in row[movement]}) == 1
    registered = {(tls, row["right_id"], row["straight_id"]) for tls, rows in pairs.items() for row in rows}
    previous = [(r["tls_id"], r["own_movement"], r["priority_movement"]) for r in roster["rows"]]
    unmatched = sorted(set(previous) - registered)
    roads = input_file(cfg, "route-files")
    checks = dict(
        admitted_layout=set(pairs) == {f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 4)}
            and summary["stable_phase_counts"] == [8] and all(len(rows) == 2 for rows in pairs.values()),
        yielding_rights=summary["right_movements"] == 48 and summary["right_movements_with_other_signal"] == 0,
        unique_priority_straight=len({(tls, right) for tls, right, _ in registered}) == len(registered) == 24,
        dedicated_single_inlets=all(len(p["right_lanes"]) == len(p["straight_lanes"]) == 1
                                    for rows in pairs.values() for p in rows),
        same_lane_yielding=all(bool(p["same_lane_link_pairs"]) for rows in pairs.values() for p in rows),
        uniform_movement_signals=uniform,
        historical_preaction_pairs=roster["status"] == "ROSTER_LOCKED_PRE_RESPONSE"
            and len(previous) == 15 and not unmatched,
        bound_demand_present=roads.is_file(),
    )
    passed = all(checks.values())
    receipt = dict(status="PASS_STATIC_ADMISSION" if passed else "HOLD_STATIC_ADMISSION",
                   sumocfg=str(cfg), net=summary["net"], route_file=str(roads),
                   tls=len(pairs), directed_pairs=len(registered), pairs_per_tls=summary["directed_pair_counts"],
                   historical_rows=len(previous), historical_distinct_pairs=len(set(previous)),
                   unmatched_historical_pairs=unmatched, checks=checks, simulations=0, fits=0)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "static_admission.json", receipt)
    if passed:
        write(output / "movement_pairs.json", {"protocol": "right-straight-same-lane-pairs-v1", "tls": pairs})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sumocfg", "prior-roster", "helper-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.sumocfg.resolve(), json.loads(args.prior_roster.read_text()),
                    args.helper_dir.resolve(), args.output)
    print(json.dumps(receipt, separators=(",", ":")))
    sys.exit(0 if receipt["status"] == "PASS_STATIC_ADMISSION" else 2)
