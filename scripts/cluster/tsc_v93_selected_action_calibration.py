"""Pure selected-action calibration analysis; no model fitting or simulation."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean


ALPHA, BETA, QUANTILE = 0.9, 0.5, 0.9
OOF_SEEDS = (5057, 6067)
VALIDATION_SEEDS = (8171,)
SCENARIOS = ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500")
POLICIES = ("gp", "old_guard", "new_guard", "raw")
CALIBRATION_ROLE = "two_seed_oof_selected_action_error_calibration"


def _validate(records, seeds):
    if {int(row["seed"]) for row in records} != set(seeds):
        raise ValueError(f"Expected only simulator seeds {seeds}")
    groups = [row["group_id"] for row in records]
    if len(groups) != len(set(groups)):
        raise ValueError("Expected one final-argmin record per unique action group")
    strata = {(int(row["seed"]), row["scenario"]) for row in records}
    if strata != {(seed, scenario) for seed in seeds for scenario in SCENARIOS}:
        raise ValueError("Each seed must retain all three scenario strata")
    for row in records:
        if not all(math.isfinite(float(row[key])) for key in (
            "score", "original_uncertainty", "gp_cost", "selected_cost", "normalization_range")):
            raise ValueError("Selected-action records require finite scores and labels")
        if row["normalization_range"] <= 0 or row["original_uncertainty"] < 0:
            raise ValueError("Invalid normalization range or original uncertainty")
        if type(row["is_reference"]) is not bool or type(row["pressure_eligible"]) is not bool:
            raise ValueError("Reference and pressure eligibility fields must be booleans")


def _candidate(row):
    return not row["is_reference"] and row["pressure_eligible"]


def calibrate(oof_records):
    """Freeze one q from the two OOF seeds, before opening historical validation."""
    _validate(oof_records, OOF_SEEDS)
    strata = {(seed, scenario): [] for seed in OOF_SEEDS for scenario in SCENARIOS}
    for row in oof_records:
        if _candidate(row):
            true_contrast = (row["selected_cost"] - row["gp_cost"]) / row["normalization_range"]
            strata[int(row["seed"]), row["scenario"]].append(abs(true_contrast - row["score"]))
    weighted = sorted((error, 1 / len(errors))
                      for errors in strata.values() for error in errors)
    q = None
    if weighted:
        threshold = QUANTILE * sum(weight for _, weight in weighted)
        cumulative = 0.0
        for error, weight in weighted:
            cumulative += weight
            if cumulative >= threshold:
                q = float(error)
                break
        if q is None:
            q = float(weighted[-1][0])
    summaries = [dict(seed=seed, scenario=scenario, eligible_group_count=len(errors))
                 for (seed, scenario), errors in strata.items()]
    return dict(
        role=CALIBRATION_ROLE, calibration_seeds=list(OOF_SEEDS), alpha=ALPHA, beta=BETA,
        quantile=QUANTILE, q=q, oof_group_count=len(oof_records),
        eligible_group_count=len(weighted), eligible_stratum_count=sum(bool(v) for v in strata.values()),
        strata=summaries, empty_strata=[r for r in summaries if not r["eligible_group_count"]],
        error="abs((selected_cost-gp_cost)/normalization_range - final_argmin_score)",
        weighting="Each nonempty seed/scenario eligible stratum has total weight one; its groups have equal weights",
        quantile_definition="Weighted inverse empirical CDF at 0.90; no interpolation or quantile search",
        application="Replace the entire old uncertainty with q; release only if score + 0.5*q < 0 and the original pressure condition passes",
        assessment="Applying this q to its calibration OOF records is calibration-apparent, not independent verification",
        coverage="No 90-percent coverage guarantee; beta remains 0.5 and action groups within a seed are dependent")


def _released(row, policy, q):
    if policy == "gp" or row["is_reference"]:
        return False
    if policy == "raw":
        return True
    if not row["pressure_eligible"]:
        return False
    uncertainty = row["original_uncertainty"] if policy == "old_guard" else q
    return uncertainty is not None and row["score"] + BETA * uncertainty < 0


def _equal_seed_scenario_mean(records, values):
    grouped = defaultdict(lambda: defaultdict(list))
    for row, value in zip(records, values):
        grouped[int(row["seed"])][row["scenario"]].append(value)
    return mean(mean(mean(v) for v in scenarios.values()) for scenarios in grouped.values())


def _summarize(records, q):
    policies = {}
    for policy in POLICIES:
        releases = [_released(row, policy, q) for row in records]
        deltas = [row["selected_cost"] - row["gp_cost"] if released else 0.0
                  for row, released in zip(records, releases)]
        released_deltas = [d for d, released in zip(deltas, releases) if released]
        harms = sum(d > 0 for d in released_deltas)
        policies[policy] = dict(
            released_action_count=len(released_deltas), benefit_count=sum(d < 0 for d in released_deltas),
            harm_count=harms, tie_count=sum(d == 0 for d in released_deltas),
            false_release_rate=harms / len(released_deltas) if released_deltas else None,
            raw_mean_delta=_equal_seed_scenario_mean(records, deltas),
            raw_mean_cost=_equal_seed_scenario_mean(
                records, [row["gp_cost"] + d for row, d in zip(records, deltas)]),
            released_action_raw_mean_delta=mean(released_deltas) if released_deltas else None,
            cumulative_harmful_cost=sum(max(d, 0.0) for d in released_deltas),
            cumulative_beneficial_cost_reduction=sum(max(-d, 0.0) for d in released_deltas))
    return dict(
        group_count=len(records), seed_count=len({row["seed"] for row in records}),
        eligibility=dict(reference_agreements=sum(row["is_reference"] for row in records),
                         nonreference_final_choices=sum(not row["is_reference"] for row in records),
                         pressure_eligible_nonreference=sum(_candidate(row) for row in records),
                         pressure_rejected_nonreference=sum(not row["is_reference"] and not row["pressure_eligible"] for row in records)),
        policies=policies,
        new_minus_old_raw_mean_delta=policies["new_guard"]["raw_mean_delta"] - policies["old_guard"]["raw_mean_delta"])


def _assessment(records, q, role):
    summary = _summarize(records, q)
    summary.update(
        role=role,
        by_seed={str(seed): _summarize([r for r in records if int(r["seed"]) == seed], q)
                 for seed in sorted({int(r["seed"]) for r in records})},
        by_scenario={scenario: _summarize([r for r in records if r["scenario"] == scenario], q)
                     for scenario in SCENARIOS},
        by_seed_scenario=[dict(seed=seed, scenario=scenario, **_summarize(
            [r for r in records if int(r["seed"]) == seed and r["scenario"] == scenario], q))
            for seed in sorted({int(r["seed"]) for r in records}) for scenario in SCENARIOS])
    return summary


def analyze_with_calibration(calibration, oof_records, validation_records):
    """Apply an already saved scalar; validation labels never enter q estimation."""
    _validate(oof_records, OOF_SEEDS)
    _validate(validation_records, VALIDATION_SEEDS)
    if (calibration["role"] != CALIBRATION_ROLE or calibration["calibration_seeds"] != list(OOF_SEEDS)
            or (calibration["alpha"], calibration["beta"], calibration["quantile"]) != (ALPHA, BETA, QUANTILE)):
        raise ValueError("The saved calibration must use the fixed two-seed OOF design")
    q = calibration["q"]
    if q is not None and (not math.isfinite(q) or q < 0):
        raise ValueError("Saved q must be a nonnegative finite scalar or unavailable")
    oof = _assessment(oof_records, q, "calibration_apparent_oof_assessment")
    validation = _assessment(validation_records, q, "historically_exposed_seed8171_not_used_for_fit_or_calibration")
    new = validation["policies"]["new_guard"]
    if new["released_action_count"] == 0:
        decision = "HOLD_NO_EFFECT"
    elif new["raw_mean_delta"] >= 0 or validation["new_minus_old_raw_mean_delta"] >= 0:
        decision = "HOLD_NO_NET_BENEFIT"
    else:
        decision = "HOLD_REQUIRES_FRESH_MULTI_SEED_VALIDATION"
    return dict(
        calibration=calibration, oof_calibration_apparent=oof, historical_validation=validation,
        decision=decision, closed_loop_expansion_authorized=False,
        aggregation="Raw label deltas average groups within scenario/seed, then scenarios equally within seed, then seeds equally; release counts and cumulative costs are descriptive unweighted sums",
        false_release_definition="Selected raw cost strictly exceeds GP cost among released non-reference actions; ties are reported separately",
        raw_policy_definition="Final argmin without confidence or pressure filtering; other guarded policies preserve the original pressure condition",
        inference="No confidence intervals from the single historical validation seed; OOF calibration-apparent results and historical8171 are not fresh independent confirmation")
