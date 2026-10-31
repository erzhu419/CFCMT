#!/usr/bin/env python3
"""Append the complete MP supplement to V93's frozen exported evidence.

Original CSV cells and their registered intervals remain unchanged. The new
MP comparison uses the same city-wise paired relative effect, with 20,000
seed-block bootstrap replicates and seeds 20261030, 20261031 and 20261032 for
Los Angeles, Jinan and their equal-city mean. The expanded 15-test Holm family
is recorded separately from the original 12-test correction.
"""

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
ROOT = HERE.parents[1]
DATA = ROOT / "paper/GPT_TITS_paper/source_data"
SPEC = ROOT / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
CITY_SCENARIOS = {
    "los_angeles": ("la_1x4",),
    "jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
}
SCOPES = (*CITY_SCENARIOS, "macro")
OLD_METHODS = ("frozen_selector", "target_only", "equal_all_sources", "phase_pressure", "h2oplus")
METRICS = ("mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
           "system_vehicle_hours", "completion_ratio", "collision_incidents",
           "starting_teleports", "ending_teleports")
SUMMARY_METRICS = ("mean_waiting_sec", "mean_duration_sec", "mean_time_loss_sec",
                   "mean_system_vehicle_hours", "mean_completion_ratio_fraction")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261030
SHARD_PROTOCOL = "tsc-v93-max-pressure-supplement-shard-v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows: list[dict], seeds: list[int], methods: tuple[str, ...]) -> dict:
    expected = {(method, scenario, seed) for method in methods
                for scenarios in CITY_SCENARIOS.values() for scenario in scenarios for seed in seeds}
    indexed = {(row["method"], row["scenario"], int(row["seed"])): row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != expected:
        raise ValueError("The complete method/scenario/seed matrix is required; no seed may be replaced")
    for row in rows:
        if row["scenario"] not in CITY_SCENARIOS[row["city"]]:
            raise ValueError("City and scenario disagree")
        if not all(math.isfinite(float(row[key])) for key in METRICS):
            raise ValueError("Nonfinite rollout metrics cannot enter the comparison")
    return indexed


def load_mp_rows(shards_root: Path, seeds: list[int]) -> list[dict]:
    paths = sorted(shards_root.glob("shard_*.json"))
    rows, shard_indices = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        if not (payload.get("protocol") == SHARD_PROTOCOL and payload.get("passed") is True
                and payload.get("source_policy") == "max_pressure"
                and payload.get("task_count") == len(payload.get("rows", []))
                and payload.get("shard_count") == len(paths)):
            raise ValueError(f"Incomplete or mismatched MP shard: {path}")
        shard_indices.append(payload["shard_index"])
        for position, item in enumerate(payload["rows"], 1):
            if not (item.get("source_policy") == "max_pressure"
                    and item.get("runtime_policy") == "max_pressure"
                    and item.get("policy") == "max_pressure_v93_supplement"
                    and item.get("evaluation_horizon_sec") == 3600
                    and item.get("control_interval_sec") == 10 and item.get("warmup_sec") == 60):
                raise ValueError(f"MP identity or original execution timing changed: {path}:{position}")
            row = dict(method="max_pressure", model_label="MaxPressure", city=item["city"],
                       scenario=item["scenario"], seed=item["seed"])
            row.update({metric: item["metrics"][metric] for metric in METRICS})
            row.update(candidate_key="", source_policy=item["policy"], source_city_weights="{}",
                       guard_config="null", source_file=str(path), source_row_1based=position,
                       result_path=item["result_path"], tripinfo_path=item["tripinfo_path"])
            rows.append(row)
    if sorted(shard_indices) != list(range(len(paths))) or not paths:
        raise ValueError("Every MP shard must be present exactly once")
    index_rows(rows, seeds, ("max_pressure",))
    return rows


def value(indexed: dict, method: str, scope: str, seed: int, metric: str) -> float:
    if scope == "macro":
        return mean(value(indexed, method, city, seed, metric) for city in CITY_SCENARIOS)
    scenarios = CITY_SCENARIOS.get(scope, (scope,))
    return mean(float(indexed[method, scenario, seed][metric]) for scenario in scenarios)


def relative_deltas(indexed: dict, seeds: list[int]) -> dict[str, np.ndarray]:
    deltas = {}
    for city in CITY_SCENARIOS:
        deltas[city] = np.array([
            (value(indexed, "frozen_selector", city, seed, METRICS[0])
             - value(indexed, "max_pressure", city, seed, METRICS[0]))
            / max(value(indexed, "max_pressure", city, seed, METRICS[0]), 1e-9)
            for seed in seeds
        ])
    deltas["macro"] = np.mean(list(deltas.values()), axis=0)
    return deltas


def paired_summary(values: np.ndarray, bootstrap_seed: int) -> dict:
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    interval = np.quantile(np.mean(values[samples], axis=1), [0.025, 0.975])
    exact_tie = bool(np.allclose(values, 0.0, rtol=0.0, atol=1e-15))
    p_value = 1.0 if exact_tie else float(wilcoxon(
        values, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
    return dict(seed_count=len(values), registered_mean_relative_delta_pct=100 * float(np.mean(values)),
                registered_median_relative_delta_pct=100 * float(np.median(values)),
                registered_std_relative_delta_pct=100 * float(np.std(values, ddof=1)),
                registered_bootstrap_ci95_low_pct=100 * float(interval[0]),
                registered_bootstrap_ci95_high_pct=100 * float(interval[1]),
                improved_seed_count=int(np.sum(values < 0)),
                tied_seed_count=int(np.sum(np.isclose(values, 0.0, atol=1e-15))),
                worsened_seed_count=int(np.sum(values > 0)), wilcoxon_p_two_sided=p_value,
                wilcoxon_p_holm="", exact_tie=exact_tie)


def holm_adjust(p_values: dict[tuple, float]) -> dict:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, p_values[key] * (len(ordered) - rank)))
        adjusted[key] = running
    return adjusted


def build(shards_root: Path, source_data: Path, out_dir: Path, analysis_out: Path,
          confirmation_spec: Path = SPEC) -> dict:
    config = json.loads(confirmation_spec.read_text())
    seeds = config["confirmation_seeds"]
    if len(seeds) != len(set(seeds)) or len(seeds) != 56:
        raise ValueError("The original frozen 56-seed cohort is required")
    original = {suffix: read_csv(source_data / f"source_selector_confirmation_{suffix}.csv")
                for suffix in ("rollouts", "summary", "comparisons", "seed")}
    index_rows(original["rollouts"], seeds, OLD_METHODS)
    mp_rows = load_mp_rows(shards_root, seeds)
    order = {seed: i for i, seed in enumerate(seeds)}
    mp_rows.sort(key=lambda row: (row["scenario"], order[int(row["seed"])]))
    rollouts = original["rollouts"] + mp_rows
    indexed = index_rows(rollouts, seeds, (*OLD_METHODS, "max_pressure"))

    summary = list(original["summary"])
    scopes = [("scenario", scenario) for scenarios in CITY_SCENARIOS.values() for scenario in scenarios]
    scopes += [("city", city) for city in CITY_SCENARIOS] + [("macro", "macro")]
    for scope_type, scope in scopes:
        count = 4 if scope == "macro" else len(CITY_SCENARIOS.get(scope, (scope,)))
        row = dict(scope_type=scope_type, scope=scope, method="max_pressure", seed_count=len(seeds),
                   scenario_count=count, rollout_count=len(seeds) * count)
        row.update({column: mean(value(indexed, "max_pressure", scope, seed, metric) for seed in seeds)
                    for column, metric in zip(SUMMARY_METRICS, METRICS[:5])})
        summary.append(row)

    deltas = relative_deltas(indexed, seeds)
    comparisons, paired = list(original["comparisons"]), list(original["seed"])
    if len(comparisons) != 12 or len(paired) != 672:
        raise ValueError("The original four-comparator export is incomplete")
    for scope_index, scope in enumerate(SCOPES):
        comparisons.append(dict(scope=scope, method="frozen_selector", comparator="max_pressure",
                                **paired_summary(deltas[scope], BOOTSTRAP_SEED + scope_index)))
        for index, seed in enumerate(seeds):
            paired.append(dict(scope=scope, method="frozen_selector", comparator="max_pressure",
                               seed_order_1based=index + 1, seed=seed,
                               registered_relative_delta_pct=100 * float(deltas[scope][index]),
                               selector_mean_waiting_sec=value(indexed, "frozen_selector", scope, seed, METRICS[0]),
                               comparator_mean_waiting_sec=value(indexed, "max_pressure", scope, seed, METRICS[0])))
    adjusted = holm_adjust({(row["comparator"], row["scope"]): float(row["wilcoxon_p_two_sided"])
                            for row in comparisons})
    for row in comparisons:
        row["wilcoxon_p_holm_original12"] = row["wilcoxon_p_holm"]
        row["wilcoxon_p_holm_expanded15"] = adjusted[row["comparator"], row["scope"]]

    outputs = dict(rollouts=rollouts, summary=summary, comparisons=comparisons, seed=paired)
    for suffix, rows in outputs.items():
        write_csv(out_dir / f"source_selector_with_mp_{suffix}.csv", rows)
    mp_comparisons = {row["scope"]: row for row in comparisons if row["comparator"] == "max_pressure"}
    report = dict(protocol="tsc-v93-max-pressure-supplement-report-v1", complete=True,
                  original_rollout_count=len(original["rollouts"]), added_mp_rollout_count=len(mp_rows),
                  total_rollout_count=len(rollouts), seed_count=len(seeds), seeds=seeds,
                  estimand="Mean seed-paired relative change within each city, then equal-city mean",
                  bootstrap_replicates=BOOTSTRAP_REPLICATES,
                  bootstrap_seed_by_scope={scope: BOOTSTRAP_SEED + i for i, scope in enumerate(SCOPES)},
                  multiplicity="Holm correction over all 15 city/equal-city comparisons against five comparators",
                  legacy_multiplicity="Original wilcoxon_p_holm and wilcoxon_p_holm_original12 columns retain the 12-test family; blank for MP",
                  max_pressure_comparisons=mp_comparisons,
                  max_pressure_summary=[row for row in summary if row["method"] == "max_pressure"],
                  incident_totals={method: {metric: sum(int(float(row[metric])) for row in rollouts
                                                          if row["method"] == method)
                                             for metric in METRICS[5:]}
                                   for method in ("frozen_selector", "max_pressure")})
    analysis_out.parent.mkdir(parents=True, exist_ok=True)
    analysis_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=DATA)
    parser.add_argument("--analysis-out", type=Path, default=HERE / "analysis.json")
    args = parser.parse_args()
    report = build(args.shards_root, args.source_data, args.out_dir, args.analysis_out)
    print(json.dumps({key: report[key] for key in
                      ("complete", "added_mp_rollout_count", "total_rollout_count", "max_pressure_comparisons")}))


if __name__ == "__main__":
    main()
