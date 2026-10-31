#!/usr/bin/env python3
"""Export V93's complete frozen cohort; copy registered statistics, never refit.

Run with python3 from any directory. Inputs are existing repository-relative
V93 learned and V92 comparator shard JSONs, plus the frozen config and audit.
All *_pct values are percentages (original relative fractions multiplied by 100).
Completion ratios are fractions. City means weight scenarios equally; macro raw
means weight cities equally. Registered macro deltas instead average the two
city-relative waiting changes per seed, then average seeds; their copied CIs do
not describe a ratio of raw macro means. All 56 frozen seeds are retained.
Source row numbers are one-based positions within a shard's ``rows`` array.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "cf_h2o/results/cluster"
V93 = RESULTS / "tsc_v93_source_city_components_20260831"
V92 = RESULTS / "tsc_v92_topology_context_repair_20260831"
CONFIG = ROOT / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
METRICS = ("mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
           "system_vehicle_hours", "completion_ratio", "collision_incidents",
           "starting_teleports", "ending_teleports")
SUMMARY_METRICS = ("mean_waiting_sec", "mean_duration_sec", "mean_time_loss_sec",
                   "mean_system_vehicle_hours", "mean_completion_ratio_fraction")
CITY_SCENARIOS = {"jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
                  "los_angeles": ("la_1x4",)}
METHODS = ("frozen_selector", "target_only", "equal_all_sources", "phase_pressure", "h2oplus")
LABELS = {"frozen_selector": "V93 selected source-city mixture with target-anchor fallback",
          "target_only": "V93 legacy near-target-only anchor with matched city guard",
          "equal_all_sources": "V93 equal mixture of seven source-city components",
          "phase_pressure": "PhasePressure",
          "h2oplus": "H2Oplus-style dense residual MPC"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(suffix, rows):
    path = OUT / f"source_selector_confirmation_{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


def main():
    config, audit = read(CONFIG), read(V93 / "disjoint_confirmation_v1/audit.json")
    seeds = config["confirmation_seeds"]
    assert len(seeds) == len(set(seeds)) == 56 and audit["passed_integrity_audit"]
    rows = []
    for base, baseline in ((V93 / "disjoint_confirmation_v1", None),
                           (V92 / "phase_pressure_v1", "phase_pressure"),
                           (V92 / "h2oplus_v1", "h2oplus")):
        shards = sorted((base / "_shards").glob("shard_*.json"))
        assert len(shards) == 6
        for path in shards:
            shard = read(path)
            assert shard["passed"]
            for position, item in enumerate(shard["rows"], 1):
                if item["seed"] not in seeds:
                    continue
                method = baseline or item["candidate_key"]
                assert method in METHODS
                row = dict(method=method, model_label=LABELS[method], city=item["city"],
                           scenario=item["scenario"], seed=item["seed"])
                row.update({metric: item["metrics"][metric] for metric in METRICS})
                row.update(candidate_key=item.get("candidate_key", ""), source_policy=item.get("policy", ""),
                           source_city_weights=json.dumps(item.get("source_city_weights", {}), sort_keys=True),
                           guard_config=json.dumps(item.get("guard_config"), sort_keys=True),
                           source_file=str(path.relative_to(ROOT)), source_row_1based=position,
                           result_path=item["result_path"], tripinfo_path=item["tripinfo_path"])
                rows.append(row)
    expected = {(method, scenario, seed) for method in METHODS for scenarios in CITY_SCENARIOS.values()
                for scenario in scenarios for seed in seeds}
    indexed = {(r["method"], r["scenario"], r["seed"]): r for r in rows}
    assert len(rows) == len(indexed) == 1120 and set(indexed) == expected
    assert audit["matrix_size"] == len(rows)
    seed_order = {seed: i for i, seed in enumerate(seeds)}
    rows.sort(key=lambda r: (METHODS.index(r["method"]), r["scenario"], seed_order[r["seed"]]))
    write("rollouts", rows)

    def value(method, scope, seed, metric):
        if scope == "macro":
            return mean(value(method, city, seed, metric) for city in CITY_SCENARIOS)
        scenarios = CITY_SCENARIOS.get(scope, (scope,))
        return mean(indexed[method, scenario, seed][metric] for scenario in scenarios)

    scopes = [("scenario", scenario) for scenarios in CITY_SCENARIOS.values() for scenario in scenarios]
    scopes += [("city", city) for city in CITY_SCENARIOS] + [("macro", "macro")]
    summary = []
    for scope_type, scope in scopes:
        scenario_count = 4 if scope == "macro" else len(CITY_SCENARIOS.get(scope, (scope,)))
        for method in METHODS:
            row = dict(scope_type=scope_type, scope=scope, method=method, seed_count=len(seeds),
                       scenario_count=scenario_count, rollout_count=len(seeds) * scenario_count)
            row.update({column: mean(value(method, scope, seed, metric) for seed in seeds)
                        for column, metric in zip(SUMMARY_METRICS, METRICS[:5])})
            summary.append(row)
    write("summary", summary)

    comparisons, paired = [], []
    for comparator, comparison in audit["statistics"]["comparators"].items():
        for scope in (*CITY_SCENARIOS, "macro"):
            stats = comparison["macro"] if scope == "macro" else comparison["by_city"][scope]
            row = dict(scope=scope, method="frozen_selector", comparator=comparator, seed_count=stats["n"])
            for key in ("mean_relative_delta", "median_relative_delta", "std_relative_delta"):
                row[f"registered_{key}_pct"] = 100 * stats[key]
            row.update(registered_bootstrap_ci95_low_pct=100 * stats["bootstrap_mean_ci95"][0],
                       registered_bootstrap_ci95_high_pct=100 * stats["bootstrap_mean_ci95"][1])
            row.update({key: stats[key] for key in ("improved_seed_count", "tied_seed_count", "worsened_seed_count",
                                                   "wilcoxon_p_two_sided", "wilcoxon_p_holm", "exact_tie")})
            comparisons.append(row)
            for i, (seed, delta) in enumerate(zip(seeds, stats["relative_deltas"])):
                if scope == "macro":
                    calculated = mean((value("frozen_selector", city, seed, METRICS[0]) /
                                       value(comparator, city, seed, METRICS[0]) - 1) for city in CITY_SCENARIOS)
                else:
                    calculated = value("frozen_selector", scope, seed, METRICS[0]) / value(comparator, scope, seed, METRICS[0]) - 1
                assert math.isclose(calculated, delta, abs_tol=1e-12)
                paired.append(dict(scope=scope, method="frozen_selector", comparator=comparator,
                                   seed_order_1based=i + 1, seed=seed, registered_relative_delta_pct=100 * delta,
                                   selector_mean_waiting_sec=value("frozen_selector", scope, seed, METRICS[0]),
                                   comparator_mean_waiting_sec=value(comparator, scope, seed, METRICS[0])))
    assert len(comparisons) == 12 and len(paired) == 672
    write("comparisons", comparisons)
    write("seed", paired)


if __name__ == "__main__":
    main()
