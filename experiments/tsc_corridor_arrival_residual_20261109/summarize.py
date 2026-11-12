"""Reproduce the old analytical benchmark and extract compact descriptive findings."""
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CONDITIONS = (
    "overall", "entry_queue:false", "entry_queue:true",
    "normalized_elapsed_delay:<=0", "normalized_elapsed_delay:(0,0.25]", "normalized_elapsed_delay:>0.25",
    "past_10s_speed_ratio:<=0.5", "past_10s_speed_ratio:(0.5,0.9]", "past_10s_speed_ratio:(0.9,1.1]",
    "past_10s_speed_ratio:>1.1", "past_10s_speed_ratio:missing",
    "nominal_PP_protected_seconds:0", "nominal_PP_protected_seconds:1-9", "nominal_PP_protected_seconds:10",
)


def compact(cell):
    n = cell["residual_n"]
    return dict(link_traversals=cell["unique_vehicles"], observed_straight=n,
                median_residual_s=cell["residual_median"], mean_residual_s=cell["residual_mean"],
                early_fraction=cell["early_beyond_1s"] / n if n else None,
                late_fraction=cell["late_beyond_1s"] / n if n else None)


def summarize():
    result = json.loads((HERE / "diagnostic_result.json").read_text())
    original = json.loads((HERE.parent / "tsc_corridor_arrival_transfer_20261107/development_result.json").read_text())
    findings = dict(status="SHARED_PHYSICAL_PATTERN_FOUND", analysis_date="2026-11-09",
                    hours=len(result["seed_results"]), new_simulations=0, original_gate="HOLD_DEVELOPMENT",
                    inference="Descriptive existing PP data; no learned source correction or control benefit tested.",
                    identity_unit="One vehicle-link release (edge,vid,entry_s), not a distinct vehicle across the network.", cities={})
    for city in ("hangzhou", "jinan"):
        records = [r for r in result["seed_results"] if r["city"] == city]
        old = {r["seed"]: r for r in original["cities"][city]["by_seed"]}
        blocks = {}
        for part in ("dev", "adapt", "donor"):
            panel = [r for r in records if r["part"] == part]
            scope = "original_common_scored" if part == "dev" else "all_observable_risk"
            by_seed = []
            for r in panel:
                p = r["scopes"][scope]
                a, h = p["analytic_count_mae"], p["history_count_mae"]
                if part == "dev" and (abs(a - old[r["seed"]]["mae"]["analytic"]) > 1e-12
                        or p["edge_epochs"] != old[r["seed"]]["counts"]["scored_edge_epochs"]):
                    raise ValueError(f"Original benchmark or denominator changed: {city}/{r['seed']}")
                near = r["scopes"]["all_observable_risk"]["nearest_eta10_current_straight"]
                low, normal = near["past_10s_speed_ratio:(0.5,0.9]"], near["past_10s_speed_ratio:(0.9,1.1]"]
                by_seed.append(dict(seed=r["seed"], block=r["block"], analytic_mae=a, history_mae=h,
                                    low_speed_median_s=low["residual_median"], normal_speed_median_s=normal["residual_median"]))
            a, h = mean(r["analytic_mae"] for r in by_seed), mean(r["history_mae"] for r in by_seed)
            near = result["city_summaries"][city][part]["all_observable_risk"]["nearest_eta10_current_straight"]
            blocks[part] = dict(seeds=len(panel), equal_seed_analytic_mae=a, equal_seed_history_mae=h,
                               relative_mae_change_percent=100 * (h / a - 1),
                               history_better_seeds=sum(r["history_mae"] < r["analytic_mae"] for r in by_seed),
                               low_speed_later_than_normal_seeds=sum(r["low_speed_median_s"] > r["normal_speed_median_s"] for r in by_seed),
                               conditions={name: compact(near[name]) for name in CONDITIONS if name in near}, by_seed=by_seed)
        counts = result["city_summaries"][city]["dev"]["original_common_scored"]["snapshot_weighted"]["overall"]
        errors = counts["false_positive"] + counts["false_negative"]
        findings["cities"][city] = dict(parts=blocks, dev_error_decomposition={
            **{key: counts[key] for key in ("false_positive", "false_negative", "timing_fp", "timing_fn", "lanechange_fn", "competing_fp", "censored_fp")},
            "timing_fraction_of_individual_errors": (counts["timing_fp"] + counts["timing_fn"]) / errors})
    findings["original_benchmark_reproduced_for_all_12_seeds"] = True
    (HERE / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False) + "\n")
    print(json.dumps({city: findings["cities"][city]["parts"]["dev"] | {"conditions": "see findings.json", "by_seed": "see findings.json"} for city in findings["cities"]}, indent=2))


if __name__ == "__main__":
    summarize()
