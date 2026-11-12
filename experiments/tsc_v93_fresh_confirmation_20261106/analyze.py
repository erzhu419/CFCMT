#!/usr/bin/env python3
"""Analyze the complete, separately reported fresh V93 four-controller panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
METRICS = (
    "mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
    "system_vehicle_hours", "completion_ratio", "collision_events",
    "collision_event_steps", "collision_incidents", "starting_teleports", "ending_teleports",
)
INCIDENTS = METRICS[5:]
SELECTOR = "frozen_selector"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows, protocol):
    cities = protocol["city_scenarios"]
    expected = {(method, scenario, seed) for method in protocol["methods"]
                for scenarios in cities.values() for scenario in scenarios
                for seed in protocol["confirmation_seeds"]}
    indexed = {(r["method"], r["scenario"], int(r["seed"])): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Complete unique method/scenario/seed matrix required; no replacement seeds")
    for row in rows:
        if row["scenario"] not in cities[row["city"]]:
            raise ValueError("City/scenario mapping differs from the protocol")
        if not all(math.isfinite(float(row[key])) for key in METRICS):
            raise ValueError("Every rollout needs all ten finite outcome metrics")
    return indexed


def value(indexed, cities, method, scope, seed, metric):
    if scope == "macro":
        return mean(value(indexed, cities, method, city, seed, metric) for city in cities)
    return mean(float(indexed[method, scenario, seed][metric])
                for scenario in cities.get(scope, [scope]))


def paired_effects(indexed, protocol, comparator, scope):
    cities = protocol["city_scenarios"]
    result = []
    for seed in protocol["confirmation_seeds"]:
        selected_cities = list(cities) if scope == "macro" else [scope]
        changes = []
        for city in selected_cities:
            a = value(indexed, cities, SELECTOR, city, seed, METRICS[0])
            b = value(indexed, cities, comparator, city, seed, METRICS[0])
            if b <= 0:
                raise ValueError("Relative waiting estimand requires positive comparator waiting")
            changes.append((a - b) / b)
        result.append(mean(changes))
    return np.asarray(result)


def holm_adjust(comparisons):
    running = 0.0
    for rank, row in enumerate(sorted(comparisons, key=lambda r: (r["wilcoxon_p_two_sided"], r["comparator"], r["scope"]))):
        running = max(running, min(1.0, row["wilcoxon_p_two_sided"] * (len(comparisons) - rank)))
        row["wilcoxon_p_holm9"] = running
        row["waiting_superiority_confirmed"] = bool(
            row["mean_relative_delta_pct"] < 0
            and row["bootstrap_ci95_high_pct"] < 0 and running < 0.05)


def build(records, protocol, out_dir):
    rows = read_csv(records)
    indexed = index_rows(rows, protocol)
    seeds, cities = protocol["confirmation_seeds"], protocol["city_scenarios"]
    scopes = [*cities, "macro"]
    comparators = [m for m in protocol["methods"] if m != SELECTOR]
    comparisons, paired, summary = [], [], []
    for scope_index, scope in enumerate(scopes):
        rng = np.random.default_rng(protocol["bootstrap"]["seed_base"] + scope_index)
        samples = rng.integers(0, len(seeds), size=(protocol["bootstrap"]["replicates"], len(seeds)))
        for comparator in comparators:
            effects = paired_effects(indexed, protocol, comparator, scope)
            ci = np.quantile(effects[samples].mean(axis=1), [0.025, 0.975])
            p = 1.0 if np.all(effects == 0) else float(wilcoxon(
                effects, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
            a = mean(value(indexed, cities, SELECTOR, scope, seed, METRICS[0]) for seed in seeds)
            b = mean(value(indexed, cities, comparator, scope, seed, METRICS[0]) for seed in seeds)
            comparisons.append(dict(
                scope=scope, method=SELECTOR, comparator=comparator, seed_count=len(seeds),
                mean_relative_delta_pct=100 * float(effects.mean()),
                bootstrap_ci95_low_pct=100 * float(ci[0]), bootstrap_ci95_high_pct=100 * float(ci[1]),
                improved_seed_count=int(np.sum(effects < 0)), tied_seed_count=int(np.sum(effects == 0)),
                worsened_seed_count=int(np.sum(effects > 0)), wilcoxon_p_two_sided=p,
                mean_waiting_delta_sec=a-b, ratio_of_scope_means_delta_pct=100 * (a/b-1)))
            paired.extend(dict(scope=scope, comparator=comparator, seed=seed,
                               relative_waiting_delta_pct=100 * float(effect))
                          for seed, effect in zip(seeds, effects))
    holm_adjust(comparisons)
    for scope in [*(s for group in cities.values() for s in group), *scopes]:
        for method in protocol["methods"]:
            record = dict(scope=scope, method=method, seed_count=len(seeds))
            record.update({metric: mean(value(indexed, cities, method, scope, seed, metric)
                                        for seed in seeds) for metric in METRICS[:5]})
            summary.append(record)
    incidents = {method: {metric: sum(float(r[metric]) for r in rows if r["method"] == method)
                          for metric in INCIDENTS} for method in protocol["methods"]}
    exact_gp = {}
    for city, scenarios in cities.items():
        pairs = [(indexed[SELECTOR, s, seed], indexed["exact_generalized_pressure", s, seed])
                 for s in scenarios for seed in seeds]
        exact_gp[city] = dict(run_count=len(pairs), all_ten_metrics_exact_ties=sum(
            all(float(a[k]) == float(b[k]) for k in METRICS) for a, b in pairs))
    report = dict(
        protocol=protocol["name"], status="COMPLETE", rollout_count=len(rows), seed_count=len(seeds),
        comparison_count=len(comparisons), multiplicity="Holm9: three comparators by three scopes",
        estimand=protocol["primary_estimand"], cohort="Fresh panel only; historical runs are not pooled",
        macro_comparisons=[r for r in comparisons if r["scope"] == "macro"],
        incident_totals=incidents, exact_gp_outcome_ties=exact_gp,
        interpretation="A non-confirmed comparison is not evidence of zero benefit. Report observed means and intervals; this panel does not isolate source-data attribution.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "comparisons.csv", comparisons)
    write_csv(out_dir / "paired_seed.csv", paired)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.records, json.loads(args.protocol.read_text()), args.out_dir)
    print(json.dumps({key: report[key] for key in ("status", "rollout_count", "comparison_count")}))


if __name__ == "__main__":
    main()
