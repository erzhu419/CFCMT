#!/usr/bin/env python3
"""Summarize the nine v2 replay receipts without fetching per-second traces."""
import argparse
from collections import Counter
import json
from pathlib import Path


def expand_modes(runs):
    return [r["mode"] for r in runs for _ in range(r["start_sec"], r["end_sec"] + 1)]


def summarize(root):
    spec = json.loads((root / "protocol.json").read_text())
    cases = []
    for index, expected in enumerate(spec["cases"]):
        case = json.loads((root / f"case_{index:02d}.json").read_text())
        if (case["group_id"] != expected["group_id"] or case["status"] != "PASS"
                or not all(all(r["checks"].values()) for r in case["receipts"].values())):
            raise ValueError(f"case {index}: missing or invalid fixed pair")
        # v2 logged the native fraction with an incorrect percent suffix.
        for window in case["windows"]:
            for arm in window["arms"].values():
                arm["outgoing_occupancy_mean_fraction"] = arm.pop("outgoing_occupancy_mean_pct")
        modes = {arm: expand_modes(runs) for arm, runs in case["phases"].items()}
        if any(len(values) != 450 for values in modes.values()):
            raise ValueError("phase timeline incomplete")
        case["mechanism_summary"] = {
            "same_focal_mode_every_second": modes["pp"] == modes["selected"],
            "focal_mode_seconds": {arm: dict(Counter(values)) for arm, values in modes.items()},
            "first_green": {arm: next(r for r in runs if r["mode"] == "green")
                            for arm, runs in case["phases"].items()},
            "focal_delta_vehicle_sec": sum(w["focal_delta_vehicle_sec"] for w in case["windows"]),
            "other_tls_delta_vehicle_sec": sum(w["other_tls_delta_vehicle_sec"] for w in case["windows"]),
            "fixed_window_cumulative_sign_reversal": any(
                w["cumulative_delta_vehicle_sec"] * case["delta_vehicle_sec"] < 0
                for w in case["windows"][:-1]),
        }
        cases.append(case)
    return {
        "status": "COMPLETE_PURPOSIVE_MECHANISM_DIAGNOSTIC", "cases": cases,
        "valid_native_trajectories": 2 * len(cases), "extra_aborted_pp_prefixes": 1,
        "new_fits": 0, "endpoint_sec": 450,
        "sign_convention": "selected minus PP halted vehicle-seconds; positive means harm",
        "interpretation": "Nine outcome-selected cases; not a representative cohort or a horizon selection exercise.",
        "raw_units_note": "v2 raw occupancy_pct fields contain fractions; final summary keys corrected, values unchanged.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    for case in result["cases"]:
        m = case["mechanism_summary"]
        print(case["case_index"], [w["delta_vehicle_sec"] for w in case["windows"]],
              "total", case["delta_vehicle_sec"], "local", m["focal_delta_vehicle_sec"],
              "other", m["other_tls_delta_vehicle_sec"], "same_modes", m["same_focal_mode_every_second"],
              "reversal", m["fixed_window_cumulative_sign_reversal"])
