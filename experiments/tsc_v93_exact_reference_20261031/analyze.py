#!/usr/bin/env python3
"""Add the complete exact-reference arm to the frozen V93 exported evidence.

Existing effects, paired observations and intervals are retained verbatim.
The added comparator uses the existing city-wise ratio estimand and 20,000
paired-seed bootstrap resamples. Holm correction expands from 15 to 18 tests;
the previous adjusted p-values remain in their historical columns.
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
DATA = ROOT / "paper/Gemini_V93_TITS/source_data"
SPEC = ROOT / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
CITY_SCENARIOS = {
    "los_angeles": ("la_1x4",),
    "jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
}
SCOPES = (*CITY_SCENARIOS, "macro")
ORIGINAL_METHODS = (
    "frozen_selector", "target_only", "equal_all_sources", "h2oplus",
    "phase_pressure", "max_pressure",
)
REFERENCE = "exact_generalized_pressure"
SHARD_PROTOCOL = "tsc-v93-exact-reference-supplement-shard-v1"
POLICY = "exact_generalized_pressure_v93_supplement"
REFERENCE_KEY = "gp_d0_o0p02_s0"
METRICS = (
    "mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
    "system_vehicle_hours", "completion_ratio", "collision_incidents",
    "starting_teleports", "ending_teleports",
)
SUMMARY_METRICS = (
    "mean_waiting_sec", "mean_duration_sec", "mean_time_loss_sec",
    "mean_system_vehicle_hours", "mean_completion_ratio_fraction",
)
INCIDENT_FIELDS = (
    "collision_events", "collision_event_steps", "collision_incidents",
    "starting_teleports", "ending_teleports",
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261030


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    # Preserve original field order; append new-arm and counter fields afterwards.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def identity(row: dict) -> tuple:
    return row["method"], row["scenario"], int(row["seed"])


def index_rows(rows: list[dict], seeds: list[int], methods: tuple[str, ...]) -> dict:
    expected = {(method, scenario, seed) for method in methods
                for scenarios in CITY_SCENARIOS.values() for scenario in scenarios
                for seed in seeds}
    indexed = {identity(row): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("The complete method/scenario/seed matrix is required; no replacement seeds")
    for row in rows:
        if row["scenario"] not in CITY_SCENARIOS[row["city"]]:
            raise ValueError("City/scenario mapping changed")
        if not all(math.isfinite(float(row[key])) for key in METRICS):
            raise ValueError("Nonfinite rollout metric")
    return indexed


def value(indexed: dict, method: str, scope: str, seed: int, metric: str) -> float:
    if scope == "macro":
        return mean(value(indexed, method, city, seed, metric) for city in CITY_SCENARIOS)
    return mean(float(indexed[method, scenario, seed][metric])
                for scenario in CITY_SCENARIOS.get(scope, (scope,)))


def load_reference_rows(shards_root: Path, seeds: list[int]) -> list[dict]:
    paths = sorted(shards_root.glob("shard_*.json"))
    rows, indices = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        if not (payload.get("protocol") == SHARD_PROTOCOL and payload.get("passed") is True
                and payload.get("source_policy") == "selected_source_prior"
                and payload.get("task_count") == len(payload.get("rows", []))
                and payload.get("shard_count") == len(paths)):
            raise ValueError(f"Incomplete or mismatched exact-reference shard: {path}")
        indices.append(payload["shard_index"])
        for position, item in enumerate(payload["rows"], 1):
            if not (item.get("policy") == POLICY
                    and item.get("source_policy") == item.get("runtime_policy") == "selected_source_prior"
                    and item.get("reference_key") == REFERENCE_KEY
                    and item.get("evaluation_horizon_sec") == 3600
                    and item.get("control_interval_sec") == 10 and item.get("warmup_sec") == 60):
                raise ValueError(f"Reference identity or frozen execution timing changed: {path}:{position}")
            row = dict(method=REFERENCE, model_label="Exact generalized-pressure reference",
                       city=item["city"], scenario=item["scenario"], seed=item["seed"])
            row.update({key: item["metrics"][key] for key in METRICS})
            row.update(candidate_key="", source_policy=item["source_policy"],
                       source_city_weights="{}", guard_config="null", source_file=str(path),
                       source_row_1based=position, result_path=item["result_path"],
                       tripinfo_path=item["tripinfo_path"])
            row["collision_events"] = item["metrics"]["collision_events"]
            if "collision_event_steps" in item["metrics"]:
                row["collision_event_steps"] = item["metrics"]["collision_event_steps"]
            row.update(reference_key=item["reference_key"],
                       reference_decision_requests=item["reference_decision_requests"],
                       reference_post_warmup_decision_requests=item["reference_post_warmup_decision_requests"])
            rows.append(row)
    if not paths or sorted(indices) != list(range(len(paths))):
        raise ValueError("Every reference shard must be present exactly once")
    index_rows(rows, seeds, (REFERENCE,))
    return rows


def paired_summary(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    ci = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
    exact_tie = bool(np.allclose(values, 0, rtol=0, atol=1e-15))
    p = 1.0 if exact_tie else float(wilcoxon(
        values, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
    return dict(seed_count=len(values), registered_mean_relative_delta_pct=100 * float(values.mean()),
                registered_median_relative_delta_pct=100 * float(np.median(values)),
                registered_std_relative_delta_pct=100 * float(np.std(values, ddof=1)),
                registered_bootstrap_ci95_low_pct=100 * float(ci[0]),
                registered_bootstrap_ci95_high_pct=100 * float(ci[1]),
                improved_seed_count=int(np.sum(values < 0)),
                tied_seed_count=int(np.sum(np.isclose(values, 0, rtol=0, atol=1e-15))),
                worsened_seed_count=int(np.sum(values > 0)), wilcoxon_p_two_sided=p,
                wilcoxon_p_holm="", wilcoxon_p_holm_original12="",
                wilcoxon_p_holm_expanded15="", exact_tie=exact_tie)


def holm_adjust(rows: list[dict]) -> dict:
    pvalues = {(row["comparator"], row["scope"]): float(row["wilcoxon_p_two_sided"])
               for row in rows}
    if len(pvalues) != 18:
        raise ValueError("The expanded six-comparator by three-scope family must contain 18 tests")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, pvalues[key] * (len(ordered) - rank)))
        adjusted[key] = running
    return adjusted


def outcome_decomposition(indexed: dict, audit: dict, seeds: list[int]) -> dict:
    """Separate exact matching, numerical closeness, and city attribution."""
    metrics = (*METRICS[:5], *INCIDENT_FIELDS)
    close_atol = 1e-12
    cities = {}
    for city, scenarios in CITY_SCENARIOS.items():
        pairs = [(indexed["frozen_selector", scenario, seed], indexed[REFERENCE, scenario, seed])
                 for scenario in scenarios for seed in seeds]
        run_differences = np.array([float(a[METRICS[0]]) - float(b[METRICS[0]]) for a, b in pairs])
        city_differences = np.array([
            value(indexed, "frozen_selector", city, seed, METRICS[0])
            - value(indexed, REFERENCE, city, seed, METRICS[0]) for seed in seeds])
        comparable = [key for key in metrics if all(key in a and key in b for a, b in pairs)]
        cities[city] = dict(
            paired_run_count=len(pairs),
            waiting_exact_tie_run_count=int(np.sum(run_differences == 0)),
            waiting_close_run_count=int(np.sum(np.abs(run_differences) <= close_atol)),
            selector_lower_waiting_run_count=int(np.sum(run_differences < 0)),
            selector_higher_waiting_run_count=int(np.sum(run_differences > 0)),
            city_mean_waiting_exact_tie_seed_count=int(np.sum(city_differences == 0)),
            city_mean_waiting_close_seed_count=int(np.sum(np.abs(city_differences) <= close_atol)),
            selector_mean_waiting_sec=mean(value(indexed, "frozen_selector", city, seed, METRICS[0])
                                          for seed in seeds),
            reference_mean_waiting_sec=mean(value(indexed, REFERENCE, city, seed, METRICS[0])
                                           for seed in seeds),
            max_abs_paired_run_metric_difference={
                key: max(abs(float(a[key]) - float(b[key])) for a, b in pairs) for key in comparable},
            max_abs_paired_city_mean_metric_difference={
                key: max(abs(value(indexed, "frozen_selector", city, seed, key)
                             - value(indexed, REFERENCE, city, seed, key)) for seed in seeds)
                for key in comparable},
            all_compared_metrics_exact_tie_run_count=sum(
                all(float(a[key]) == float(b[key]) for key in comparable) for a, b in pairs),
        )
    scenario, seed = "jinan_3x4_real", 5622039
    a, b = indexed["frozen_selector", scenario, seed], indexed[REFERENCE, scenario, seed]
    counters = audit["frozen_selector", scenario, seed]
    comparable = [key for key in metrics if key in a and key in b]
    return dict(
        difference_direction="CFCMT minus exact generalized-pressure reference",
        exact_ties="Strict equality of parsed numeric values; not a tolerance-based test",
        close_ties=f"Absolute difference <= {close_atol}; zero relative tolerance; includes exact ties",
        by_city=cities,
        unique_jinan_override_case=dict(
            scenario=scenario, seed=seed, recorded_request_time_sec=1880,
            traffic_light_id="intersection_2_2",
            accepted_nonreference_switch_requests=int(counters["guard_executed_switch_overrides"]),
            effective_current_green_holds=int(counters["guard_executed_stay_overrides"]),
            selector_metrics={key: float(a[key]) for key in comparable},
            exact_reference_metrics={key: float(b[key]) for key in comparable},
            metric_differences={key: float(a[key]) - float(b[key]) for key in comparable},
            waiting_exact_tie=float(a[METRICS[0]]) == float(b[METRICS[0]]),
            waiting_close=abs(float(a[METRICS[0]]) - float(b[METRICS[0]])) <= close_atol,
        ),
    )


def build(shards_root: Path, source_data: Path, out_dir: Path,
          confirmation_spec: Path = SPEC) -> dict:
    seeds = json.loads(confirmation_spec.read_text())["confirmation_seeds"]
    if len(seeds) != 56 or len(set(seeds)) != 56:
        raise ValueError("The frozen 56-seed cohort is required")
    old = {name: read_csv(source_data / f"confirmation_{name}.csv")
           for name in ("rollouts", "summary", "comparisons", "seed")}
    index_rows(old["rollouts"], seeds, ORIGINAL_METHODS)
    if len(old["summary"]) != 42 or len(old["comparisons"]) != 15 or len(old["seed"]) != 840:
        raise ValueError("Original six-controller statistical exports are incomplete")

    # Existing result.json-derived counters are joined, not inferred from traces.
    counters = read_csv(source_data / "execution_rollouts.csv")
    audit = {identity(row): row for row in counters}
    if len(audit) != len(counters) or set(audit) != {identity(row) for row in old["rollouts"]}:
        raise ValueError("Existing per-run execution/incident export is incomplete")
    rollouts = []
    for original in old["rollouts"]:
        row = dict(original)
        counter = audit[identity(row)]
        if row["city"] != counter["city"] or any(
            int(float(row[key])) != int(counter[key]) for key in METRICS[5:]
        ):
            raise ValueError(f"Existing incident export disagrees with rollout: {identity(row)}")
        row.update({key: counter[key] for key in INCIDENT_FIELDS})
        rollouts.append(row)
    added = load_reference_rows(shards_root, seeds)
    order = {seed: i for i, seed in enumerate(seeds)}
    added.sort(key=lambda row: (row["scenario"], order[int(row["seed"])]))
    rollouts += added
    indexed = index_rows(rollouts, seeds, (*ORIGINAL_METHODS, REFERENCE))

    summary, comparisons, paired = [dict(row) for row in old["summary"]], [], list(old["seed"])
    scopes = [("scenario", scenario) for scenarios in CITY_SCENARIOS.values() for scenario in scenarios]
    scopes += [("city", city) for city in CITY_SCENARIOS] + [("macro", "macro")]
    for kind, scope in scopes:
        count = 4 if scope == "macro" else len(CITY_SCENARIOS.get(scope, (scope,)))
        row = dict(scope_type=kind, scope=scope, method=REFERENCE, seed_count=len(seeds),
                   scenario_count=count, rollout_count=len(seeds) * count)
        row.update({column: mean(value(indexed, REFERENCE, scope, seed, metric) for seed in seeds)
                    for column, metric in zip(SUMMARY_METRICS, METRICS[:5])})
        summary.append(row)
    comparisons = [dict(row) for row in old["comparisons"]]
    deltas = {}
    for city in CITY_SCENARIOS:
        reference = np.array([value(indexed, REFERENCE, city, seed, METRICS[0]) for seed in seeds])
        if np.any(reference <= 0):
            raise ValueError("The relative waiting estimand requires positive reference city means")
        selected = np.array([value(indexed, "frozen_selector", city, seed, METRICS[0]) for seed in seeds])
        deltas[city] = (selected - reference) / reference
    deltas["macro"] = np.mean(list(deltas.values()), axis=0)
    for i, scope in enumerate(SCOPES):
        comparisons.append(dict(scope=scope, method="frozen_selector", comparator=REFERENCE,
                                **paired_summary(deltas[scope], BOOTSTRAP_SEED + i)))
        for j, seed in enumerate(seeds):
            paired.append(dict(scope=scope, method="frozen_selector", comparator=REFERENCE,
                               seed_order_1based=j + 1, seed=seed,
                               registered_relative_delta_pct=100 * float(deltas[scope][j]),
                               selector_mean_waiting_sec=value(indexed, "frozen_selector", scope, seed, METRICS[0]),
                               comparator_mean_waiting_sec=value(indexed, REFERENCE, scope, seed, METRICS[0])))
    adjusted = holm_adjust(comparisons)
    for row in comparisons:
        row["wilcoxon_p_holm_expanded18"] = adjusted[row["comparator"], row["scope"]]

    new_comparisons = {row["scope"]: row for row in comparisons if row["comparator"] == REFERENCE}
    gates = {}
    for scope, row in new_comparisons.items():
        low, high = [row[f"registered_bootstrap_ci95_{bound}_pct"] for bound in ("low", "high")]
        corrected = row["wilcoxon_p_holm_expanded18"]
        conclusion = "positive_incremental_benefit_not_established"
        if high < 0 and corrected < 0.05:
            conclusion = "supports_incremental_benefit_on_reused_panel"
        elif low > 0 and corrected < 0.05:
            conclusion = "supports_incremental_harm_on_reused_panel"
        gates[scope] = dict(conclusion=conclusion, ci_below_zero=high < 0,
                            ci_above_zero=low > 0, holm18_below_005=corrected < 0.05)
    report = dict(protocol="tsc-v93-exact-reference-supplement-report-v1", complete=True,
                  original_rollout_count=len(old["rollouts"]), added_reference_rollout_count=len(added),
                  total_rollout_count=len(rollouts), seed_count=len(seeds), seeds=seeds,
                  estimand="Within-city seed-paired relative waiting change, then equal-city average",
                  evidence_scope="Reused benchmark panel; no wholly independent seed confirmation",
                  bootstrap_replicates=BOOTSTRAP_REPLICATES,
                  bootstrap_seed_by_scope={scope: BOOTSTRAP_SEED + i for i, scope in enumerate(SCOPES)},
                  multiplicity="Holm correction over 18 comparator/scope pairs (six comparators, three scopes)",
                  historical_columns="Original intervals, raw p-values, and Holm12/Holm15 columns retained",
                  reference_comparisons=new_comparisons, incremental_benefit_assessment=gates,
                  outcome_decomposition=outcome_decomposition(indexed, audit, seeds),
                  contribution_interpretation={
                      "full_pipeline_beyond_reference": gates["macro"]["conclusion"],
                      "los_angeles": "The selected target anchor has zero extra source-component weights. "
                                     "A benefit here would not establish a selected cross-city source benefit.",
                      "jinan": "The Atlanta component and guard are evaluated together. The archived policy "
                               "accepted one non-reference switch request; its paired outcome is reported separately.",
                      "source_data_attribution": "This comparison does not isolate the contribution of source data "
                                                 "from model capacity, target adaptation, guard, or coordination.",
                  },
                  reference_summary=[row for row in summary if row["method"] == REFERENCE],
                  incident_totals={method: {key: sum(int(float(row[key])) for row in rollouts
                                                       if row["method"] == method)
                                            for key in INCIDENT_FIELDS if all(key in row for row in rollouts)}
                                   for method in (*ORIGINAL_METHODS, REFERENCE)})
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in dict(rollouts=rollouts, summary=summary, comparisons=comparisons, seed=paired).items():
        write_csv(out_dir / f"seven_controller_{name}.csv", rows)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--source-data", default=DATA, type=Path)
    parser.add_argument("--out-dir", default=HERE / "analysis", type=Path)
    args = parser.parse_args()
    report = build(args.shards_root, args.source_data, args.out_dir)
    print(json.dumps({key: report[key] for key in (
        "complete", "total_rollout_count", "reference_comparisons", "incremental_benefit_assessment")}))


if __name__ == "__main__":
    main()
