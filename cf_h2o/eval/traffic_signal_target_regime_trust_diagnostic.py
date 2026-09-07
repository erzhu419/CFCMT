"""Seed-blocked diagnostic for a passive-history target-regime trust layer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.target_regime_trust import (
    FEATURE_NAME,
    fit_target_regime_trust,
)


RESULT_PROTOCOL = "tsc-v79r75-target-regime-trust-diagnostic-result-v1"


def _validate_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
) -> dict[tuple[int, str], dict[str, Any]]:
    expected = {
        (int(seed), str(scenario)) for seed in seeds for scenario in scenarios
    }
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (int(row["seed"]), str(row["scenario"]))
        feature = float(row["phase_mean_queue_per_lane"])
        delta = float(row["relative_delta"])
        if key in indexed or not np.isfinite(feature) or not np.isfinite(delta):
            raise ValueError("target regime diagnostic matrix is duplicated or non-finite")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("target regime diagnostic matrix is incomplete")
    return indexed


def _bootstrap_seed_means(
    seed_means: Mapping[int, float], *, replicates: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(
        [float(seed_means[key]) for key in sorted(seed_means)], dtype=float
    )
    if values.size < 2 or int(replicates) <= 0:
        raise ValueError("target regime bootstrap specification is invalid")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    draws = values[indices].mean(axis=1)
    return {
        "protocol": "paired-seed-equal-scenario-bootstrap-v1",
        "unit": "simulator_seed_mean_across_target_scenarios",
        "replicates": int(replicates),
        "seed": int(seed),
        "observed": float(np.mean(values)),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    delta_key: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    seed_means = {
        int(seed): float(
            np.mean(
                [float(row[delta_key]) for row in rows if int(row["seed"]) == seed]
            )
        )
        for seed in seeds
    }
    values = np.asarray(list(seed_means.values()), dtype=float)
    return {
        "mean_relative_delta": float(np.mean(values)),
        "seed_delta_std": float(np.std(values)),
        "worst_seed_relative_delta": float(np.max(values)),
        "improved_seed_fraction": float(np.mean(values < 0.0)),
        "seed_mean_relative_deltas": {
            str(key): float(value) for key, value in sorted(seed_means.items())
        },
        "bootstrap": _bootstrap_seed_means(
            seed_means,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        ),
    }


def run_cross_fitted_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    l2: float,
    bootstrap: Mapping[str, Any],
    advancement_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate history-to-new-seed trust without using held-out-day features."""

    seed_order = tuple(int(value) for value in seeds)
    scenario_order = tuple(str(value) for value in scenarios)
    indexed = _validate_matrix(rows, seeds=seed_order, scenarios=scenario_order)
    fold_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for heldout_seed in seed_order:
        training_seeds = tuple(seed for seed in seed_order if seed != heldout_seed)
        training_rows = [
            indexed[(seed, scenario)]
            for seed in training_seeds
            for scenario in scenario_order
        ]
        model = fit_target_regime_trust(
            [float(row["phase_mean_queue_per_lane"]) for row in training_rows],
            [float(row["relative_delta"]) for row in training_rows],
            l2=float(l2),
        )
        fold_decisions = []
        for scenario in scenario_order:
            # The deployment feature is an average of prior days only. The held-out
            # day's full-horizon queue statistic is deliberately not read here.
            historical_feature = float(
                np.mean(
                    [
                        float(indexed[(seed, scenario)]["phase_mean_queue_per_lane"])
                        for seed in training_seeds
                    ]
                )
            )
            predicted_delta = model.predict_relative_delta(historical_feature)
            trust = predicted_delta < 0.0
            observed_delta = float(indexed[(heldout_seed, scenario)]["relative_delta"])
            row = {
                "seed": heldout_seed,
                "scenario": scenario,
                "feature_name": FEATURE_NAME,
                "historical_phase_mean_queue_per_lane": historical_feature,
                "history_source_seeds": list(training_seeds),
                "heldout_phase_feature_used": False,
                "predicted_relative_delta": predicted_delta,
                "trust_local_controller": bool(trust),
                "observed_cfcmt_relative_delta": observed_delta,
                "gated_relative_delta": observed_delta if trust else 0.0,
            }
            fold_rows.append(row)
            fold_decisions.append(bool(trust))
        folds.append(
            {
                "heldout_seed": heldout_seed,
                "training_seeds": list(training_seeds),
                "training_row_count": len(training_rows),
                "model": model.to_dict(),
                "scenario_trust_decisions": {
                    scenario: decision
                    for scenario, decision in zip(scenario_order, fold_decisions)
                },
            }
        )

    always_rows = [
        {
            "seed": int(row["seed"]),
            "scenario": str(row["scenario"]),
            "always_cfcmt_relative_delta": float(row["relative_delta"]),
        }
        for row in rows
    ]
    always_cfcmt = _policy_summary(
        always_rows,
        seeds=seed_order,
        delta_key="always_cfcmt_relative_delta",
        bootstrap=bootstrap,
    )
    gated = _policy_summary(
        fold_rows,
        seeds=seed_order,
        delta_key="gated_relative_delta",
        bootstrap=bootstrap,
    )
    selection_rate = float(
        np.mean([bool(row["trust_local_controller"]) for row in fold_rows])
    )
    coefficient_negative = all(
        float(fold["model"]["coefficient"]) < 0.0 for fold in folds
    )
    heldout_excluded = all(
        int(row["seed"]) not in set(row["history_source_seeds"])
        and row["heldout_phase_feature_used"] is False
        for row in fold_rows
    )
    correction = (
        float(always_cfcmt["mean_relative_delta"])
        - float(gated["mean_relative_delta"])
    )
    gates = {
        "heldout_seed_excluded_from_history": heldout_excluded,
        "single_declared_feature_only": all(
            row["feature_name"] == FEATURE_NAME for row in fold_rows
        ),
        "negative_mechanistic_slope_in_every_fold": coefficient_negative,
        "nontrivial_abstention": selection_rate
        >= float(advancement_rule["minimum_selection_rate"])
        and selection_rate <= float(advancement_rule["maximum_selection_rate"]),
        "mean_delta_nonpositive": float(gated["mean_relative_delta"])
        <= float(advancement_rule["maximum_mean_relative_delta"]),
        "bootstrap_upper_within_safety_limit": float(gated["bootstrap"]["ci95"][1])
        <= float(advancement_rule["maximum_bootstrap_95pct_upper_relative_delta"]),
        "worst_seed_within_safety_limit": float(gated["worst_seed_relative_delta"])
        <= float(advancement_rule["maximum_seed_mean_relative_regression"]),
        "minimum_improved_seed_fraction": float(gated["improved_seed_fraction"])
        >= float(advancement_rule["minimum_improved_seed_fraction"]),
        "minimum_correction_vs_always_cfcmt": correction
        >= float(advancement_rule["minimum_mean_correction_vs_always_cfcmt"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "classification": (
            "disclosed_post_v78_target_simulator_closed_loop_adaptation_"
            "diagnostic"
        ),
        "feature_names": [FEATURE_NAME],
        "ridge_l2": float(l2),
        "fold_protocol": "leave_one_simulator_seed_out_history_to_new_day-v1",
        "folds": folds,
        "rows": fold_rows,
        "always_phase_pressure": {"mean_relative_delta": 0.0},
        "always_cfcmt": always_cfcmt,
        "cross_fitted_regime_gate": gated,
        "selection_rate": selection_rate,
        "mean_correction_vs_always_cfcmt": correction,
        "advancement_gate": gates,
    }
