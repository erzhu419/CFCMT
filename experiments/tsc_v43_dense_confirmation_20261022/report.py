"""Paired fresh-seed confirmation of the two unchanged V43 controllers."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

NAME = "cfcmt-v43-dense-fresh10-20261022"
POLICIES = ("cfcmt_selected_mpc", "h2oplus_style_dense_residual_mpc")
METRICS = ("mean_tripinfo_waiting_time", "mean_completed_tripinfo_waiting_time",
    "system_vehicle_hours", "completion_ratio")
COUNTS = ("tripinfo_count", "tripinfo_completed_count", "tripinfo_unfinished_count",
    "starting_teleports", "ending_teleports", "collision_incidents")
CITIES = {"los_angeles": ["la_1x4"], "jinan": ["jinan_3x4_real",
    "jinan_3x4_real_2000", "jinan_3x4_real_2500"]}


def identity(row):
    return row["city"], row["scenario"], row["seed"], row["policy"]


def summarize(records, protocol):
    seeds, cities = protocol["seeds"], protocol["city_scenarios"]
    if (protocol["name"] != NAME or cities != CITIES or tuple(protocol["policies"]) != POLICIES
            or len(seeds) != len(set(seeds)) or len(seeds) != 10
            or protocol["bootstrap"]["replicates"] != 20000
            or protocol["bootstrap"]["seed"] != 20261022):
        raise ValueError("The frozen two-policy, four-scenario, ten-seed protocol changed")
    expected = {(city, scenario, seed, policy) for city, scenarios in cities.items()
        for scenario in scenarios for seed in seeds for policy in POLICIES}
    rows = {}
    for row in records:
        key, m = identity(row), row["metrics"]
        if key in rows or key not in expected:
            raise ValueError(f"Duplicate or unregistered rollout identity: {key}")
        if row["evaluation_horizon_sec"] != 3600 or row["sumo_version"] != "1.22.0":
            raise ValueError(f"Changed evaluation horizon or SUMO version: {key}")
        if not all(math.isfinite(float(m[k])) for k in (*METRICS, *COUNTS)):
            raise ValueError(f"Nonfinite evaluation metric: {key}")
        if (any(m[k] < 0 or int(m[k]) != m[k] for k in COUNTS)
                or m["tripinfo_completed_count"] + m["tripinfo_unfinished_count"] != m["tripinfo_count"]
                or not 0 <= m["completion_ratio"] <= 1):
            raise ValueError(f"Vehicle population accounting changed: {key}")
        if m["starting_teleports"] != 0 or m["ending_teleports"] != 0:
            raise ValueError(f"Rollout contains teleports: {key}")
        rows[key] = row
    report = dict(protocol=NAME, complete=len(rows) == len(expected), ready=len(rows),
        expected=len(expected), missing=[list(key) for key in sorted(expected - rows.keys())], records=records)
    if not report["complete"]:
        return report

    # Every array retains the same ten seed blocks and metric columns.
    by_city = {city: {policy: np.asarray([[np.mean([rows[(city, scenario, seed, policy)]["metrics"][metric]
        for scenario in scenarios]) for metric in METRICS] for seed in seeds], dtype=float)
        for policy in POLICIES} for city, scenarios in cities.items()}
    macro = {policy: np.mean([values[policy] for values in by_city.values()], axis=0)
        for policy in POLICIES}

    def describe(values):
        means = {p: v.mean(axis=0) for p, v in values.items()}
        return dict(means={p: dict(zip(METRICS, v.tolist())) for p, v in means.items()},
            differences=dict(zip(METRICS, (means[POLICIES[0]] - means[POLICIES[1]]).tolist())),
            better_seed_count=int(np.sum(values[POLICIES[0]][:, 0] < values[POLICIES[1]][:, 0])))

    c, d = (macro[p][:, 0] for p in POLICIES)
    # One shared draw indexes both policies, both cities and all four scenarios.
    rng = np.random.Generator(np.random.PCG64(protocol["bootstrap"]["seed"]))
    indices = rng.integers(0, len(seeds), size=(protocol["bootstrap"]["replicates"], len(seeds)))
    draw_c, draw_d = c[indices].mean(axis=1), d[indices].mean(axis=1)
    ci = np.quantile(draw_c - draw_d, [.025, .975]).tolist()
    relative_ci = np.quantile(100 * (draw_c - draw_d) / draw_d, [.025, .975]).tolist()
    primary = dict(cfcmt_wait_sec=float(c.mean()), dense_wait_sec=float(d.mean()),
        difference_sec=float((c - d).mean()), relative_difference_percent=float(100 * (c.mean() - d.mean()) / d.mean()),
        ci95_difference_sec=ci, ci95_relative_difference_percent=relative_ci)
    report.update(macro=primary, secondary=describe(macro),
        by_city={city: describe(values) for city, values in by_city.items()},
        by_scenario={scenario: describe({policy: np.asarray([[rows[(city, scenario, seed, policy)]["metrics"][metric]
            for metric in METRICS] for seed in seeds], dtype=float) for policy in POLICIES})
            for city, scenarios in cities.items() for scenario in scenarios},
        by_seed=[dict(seed=seed, cfcmt_wait_sec=float(c[i]), dense_wait_sec=float(d[i]),
            difference_sec=float(c[i] - d[i]), relative_difference_percent=float(100 * (c[i] - d[i]) / d[i]),
            city_differences_sec={city: float(v[POLICIES[0]][i, 0] - v[POLICIES[1]][i, 0]) for city, v in by_city.items()})
            for i, seed in enumerate(seeds)], better_seed_count=int(np.sum(c < d)),
        advantage_confirmed=bool(primary["difference_sec"] < 0 and ci[1] < 0),
        bootstrap=dict(protocol["bootstrap"], generator="PCG64", unit="complete paired simulator-seed block",
            interval="two-sided percentile 95%", independent_seed_blocks=len(seeds)),
        collision_incidents={policy: sum(row["metrics"]["collision_incidents"] for row in records if row["policy"] == policy)
            for policy in POLICIES},
        weighting="Equal scenarios within city, then equal cities, then equal fresh seeds; no old-result pooling.",
        interpretation="Frozen-controller comparison on ten new simulator seeds in two fixed city geometries; intervals concern seed variation, not a city population.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    result = summarize(json.loads(args.input.read_text())["records"], protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("complete", "ready", "expected")}))


if __name__ == "__main__":
    main()
