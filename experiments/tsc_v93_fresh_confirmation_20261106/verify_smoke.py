#!/usr/bin/env python3
"""Compare the eight historical smoke cells with the recorded V93 outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("frozen_selector", "max_pressure", "phase_pressure", "exact_generalized_pressure")
PAIRS = (("los_angeles", "la_1x4", 1020485), ("jinan", "jinan_3x4_real", 5622039))
METRICS = (
    "mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
    "system_vehicle_hours", "completion_ratio", "collision_events", "collision_event_steps",
    "collision_incidents", "starting_teleports", "ending_teleports",
)


def identity(row):
    return row["method"], row["city"], row["scenario"], int(row["seed"])


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(smoke, reference, out):
    expected = {(method, city, scenario, seed) for method in METHODS for city, scenario, seed in PAIRS}
    actual_rows = read_csv(smoke)
    reference_rows = [row for row in read_csv(reference) if identity(row) in expected]
    actual_counts = Counter(identity(row) for row in actual_rows)
    reference_counts = Counter(identity(row) for row in reference_rows)
    missing = sorted(expected - actual_counts.keys())
    extra = sorted(actual_counts.keys() - expected)
    duplicates = sorted(key for key, count in actual_counts.items() if count > 1)
    reference_missing = sorted(expected - reference_counts.keys())
    reference_duplicates = sorted(key for key, count in reference_counts.items() if count > 1)
    actual = {identity(row): row for row in actual_rows}
    recorded = {identity(row): row for row in reference_rows}
    differences, invalid = [], []
    maxima = {metric: 0.0 for metric in METRICS}
    compared_values = 0
    for key in sorted(expected & actual.keys() & recorded.keys()):
        for metric in METRICS:
            try:
                a, b = float(actual[key][metric]), float(recorded[key][metric])
                if not (math.isfinite(a) and math.isfinite(b)):
                    raise ValueError("nonfinite value")
            except (KeyError, ValueError) as error:
                invalid.append(dict(identity=key, metric=metric, reason=str(error)))
                continue
            compared_values += 1
            maxima[metric] = max(maxima[metric], abs(a - b))
            if a != b:
                differences.append(dict(identity=key, metric=metric, smoke_value=a,
                                        recorded_value=b, difference=a - b))
    passed = not any((missing, extra, duplicates, reference_missing, reference_duplicates,
                      differences, invalid)) and compared_values == 80
    receipt = dict(
        status="PASS" if passed else "FAIL", smoke_file=str(smoke), reference_file=str(reference),
        expected_identity_count=8, observed_identity_count=len(actual_rows), metric_count=10,
        compared_value_count=compared_values, comparison="Exact equality of parsed finite values; no tolerance",
        missing_identities=missing, extra_identities=extra, duplicate_identities=duplicates,
        reference_missing_identities=reference_missing, reference_duplicate_identities=reference_duplicates,
        max_abs_difference_by_metric=maxima, differences=differences, invalid_metrics=invalid,
        scope="Eight historical replay cells only; no fresh-seed outcomes or source-data benefit claim")
    Path(out).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, default=HERE / "smoke/rollouts.csv")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "paper/Gemini_V93_TITS/source_data/seven_controller_rollouts.csv")
    parser.add_argument("--out", type=Path, default=HERE / "smoke_equivalence.json")
    args = parser.parse_args()
    receipt = verify(args.smoke, args.reference, args.out)
    print(json.dumps({key: receipt[key] for key in ("status", "observed_identity_count", "compared_value_count")}))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
