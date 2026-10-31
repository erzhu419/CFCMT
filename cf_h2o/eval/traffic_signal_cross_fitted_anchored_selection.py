"""Nested city-level selection for the v40 cross-fitted anchored policy."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)


CANDIDATE_SCOPES = ("constant", "all")
MINIMUM_OOF_RELATIVE_IMPROVEMENTS = (0.0, 0.01, 0.02, 0.05)
MAXIMUM_OOF_FOLD_REGRESSIONS = (0.0, 0.02, 0.05, 0.10)
STANDARD_ERROR_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
FOLD_COUNT = 5
MIN_RELATIVE_IMPROVEMENT = 0.02
MAX_ABSOLUTE_CITY_REGRESSION = 0.01
MIN_TRAINING_CITY_IMPROVEMENTS = 4
MIN_GLOBAL_CITY_IMPROVEMENTS = 5
NUMERICAL_TOLERANCE = 1e-12
ALWAYS_ANCHOR_SELECTOR = "always_anchor"
SEED_BLOCKED_SELECTION_PROTOCOL = (
    "target-seed-blocked-risk-adjusted-anchored-selection-v2"
)


def _number_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def selector_key(
    scope: str,
    minimum_improvement: float,
    maximum_fold_regression: float,
    standard_error_multiplier: float,
) -> str:
    return (
        f"scope_{scope}"
        f"_tau_{_number_key(minimum_improvement)}"
        f"_rho_{_number_key(maximum_fold_regression)}"
        f"_lambda_{_number_key(standard_error_multiplier)}"
    )


def selector_specs() -> dict[str, dict[str, float | str]]:
    return {
        selector_key(scope, tau, rho, multiplier): {
            "scope": scope,
            "minimum_improvement": float(tau),
            "maximum_fold_regression": float(rho),
            "standard_error_multiplier": float(multiplier),
        }
        for scope in CANDIDATE_SCOPES
        for tau in MINIMUM_OOF_RELATIVE_IMPROVEMENTS
        for rho in MAXIMUM_OOF_FOLD_REGRESSIONS
        for multiplier in STANDARD_ERROR_MULTIPLIERS
    }


SELECTOR_SPECS = selector_specs()
SELECTOR_KEYS = tuple(SELECTOR_SPECS)


def _validated_target_oof(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    expected_fold_count: int = FOLD_COUNT,
) -> dict[str, dict[str, Any]]:
    expected_fold_count = int(expected_fold_count)
    if expected_fold_count < 2:
        raise ValueError("OOF selection requires at least two folds")
    if set(candidates) != set(CANDIDATE_KEYS):
        raise ValueError("v40 OOF candidate set changed")
    result = {}
    for candidate in CANDIDATE_KEYS:
        row = candidates[candidate]
        folds = np.asarray(row.get("fold_regrets", ()), dtype=float)
        mean_alpha = float(row.get("mean_alpha", float("nan")))
        if folds.shape != (expected_fold_count,) or not np.all(
            np.isfinite(folds)
        ):
            raise ValueError(f"invalid OOF fold regrets for {candidate}")
        if np.any(folds < 0.0):
            raise ValueError(f"negative v40 fold regret for {candidate}")
        if not math.isfinite(mean_alpha) or not 0.0 <= mean_alpha <= 1.0:
            raise ValueError(f"invalid v40 mean alpha for {candidate}")
        result[candidate] = {
            "fold_regrets": folds,
            "mean_alpha": mean_alpha,
        }
    return result


def select_target_candidate(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    scope: str,
    minimum_improvement: float,
    maximum_fold_regression: float,
    standard_error_multiplier: float,
    expected_fold_count: int = FOLD_COUNT,
) -> dict[str, Any]:
    expected_fold_count = int(expected_fold_count)
    rows = _validated_target_oof(
        candidates, expected_fold_count=expected_fold_count
    )
    if scope not in CANDIDATE_SCOPES:
        raise ValueError(f"unknown v40 candidate scope: {scope}")
    anchor_folds = rows[ANCHOR_KEY]["fold_regrets"]
    anchor_mean = float(np.mean(anchor_folds))
    if anchor_mean <= NUMERICAL_TOLERANCE:
        raise ValueError("v40 anchor OOF mean must be positive")
    allowed = (
        tuple(key for key in CANDIDATE_KEYS if key.startswith("constant_alpha_"))
        if scope == "constant"
        else CANDIDATE_KEYS
    )
    summaries: dict[str, dict[str, Any]] = {}
    admissible = []
    for candidate in allowed:
        folds = rows[candidate]["fold_regrets"]
        differences = folds - anchor_folds
        mean_difference = float(np.mean(differences))
        standard_error = float(
            np.std(differences, ddof=1) / np.sqrt(expected_fold_count)
        )
        relative_improvement = float(-mean_difference / anchor_mean)
        maximum_regression = float(np.max(np.maximum(differences, 0.0)))
        upper_delta = float(
            mean_difference + float(standard_error_multiplier) * standard_error
        )
        summary = {
            "candidate": candidate,
            "oof_anchor_mean_regret": anchor_mean,
            "oof_candidate_mean_regret": float(np.mean(folds)),
            "oof_mean_regret_difference": mean_difference,
            "oof_relative_improvement": relative_improvement,
            "oof_difference_standard_error": standard_error,
            "oof_risk_upper_difference": upper_delta,
            "maximum_oof_fold_regression": maximum_regression,
            "mean_alpha": rows[candidate]["mean_alpha"],
            "fold_differences": differences.tolist(),
        }
        summary["admissible"] = bool(
            candidate != ANCHOR_KEY
            and relative_improvement + NUMERICAL_TOLERANCE
            >= float(minimum_improvement)
            and maximum_regression
            <= float(maximum_fold_regression) + NUMERICAL_TOLERANCE
            and upper_delta <= NUMERICAL_TOLERANCE
        )
        summaries[candidate] = summary
        if summary["admissible"]:
            admissible.append(summary)
    if admissible:
        selected = min(
            admissible,
            key=lambda row: (
                row["oof_risk_upper_difference"],
                row["oof_candidate_mean_regret"],
                row["mean_alpha"],
                row["candidate"],
            ),
        )
    else:
        selected = summaries[ANCHOR_KEY]
    return {
        "protocol": (
            "target-b60-five-fold-risk-adjusted-anchored-selection-v1"
            if expected_fold_count == FOLD_COUNT
            else SEED_BLOCKED_SELECTION_PROTOCOL
        ),
        "fold_count": expected_fold_count,
        "scope": scope,
        "minimum_improvement": float(minimum_improvement),
        "maximum_fold_regression": float(maximum_fold_regression),
        "standard_error_multiplier": float(standard_error_multiplier),
        "selected_candidate": selected["candidate"],
        "selected_summary": selected,
        "admissible_candidates": sorted(
            row["candidate"] for row in admissible
        ),
        "candidate_summaries": summaries,
    }


def _validate_development_inputs(
    *,
    oof_by_target: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evaluation_by_target: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_city_groups: Mapping[str, str],
) -> tuple[str, ...]:
    targets = tuple(oof_by_target)
    if not targets or set(targets) != set(evaluation_by_target):
        raise ValueError("v40 OOF and evaluation target sets differ")
    if set(targets) != set(target_city_groups):
        raise ValueError("v40 target city mapping differs from result targets")
    if set(target_city_groups.values()) != set(DEVELOPMENT_CITY_GROUPS):
        raise ValueError("v40 development city set changed")
    for target in targets:
        _validated_target_oof(oof_by_target[target])
        evaluation = evaluation_by_target[target]
        if set(evaluation) != set(CANDIDATE_KEYS):
            raise ValueError(f"v40 evaluation candidate set changed for {target}")
        for candidate in CANDIDATE_KEYS:
            regret = float(evaluation[candidate]["regret"])
            mean_alpha = float(evaluation[candidate]["mean_alpha"])
            if not math.isfinite(regret) or regret < 0.0:
                raise ValueError(f"invalid evaluation regret for {target}/{candidate}")
            if not math.isfinite(mean_alpha) or not 0.0 <= mean_alpha <= 1.0:
                raise ValueError(f"invalid evaluation alpha for {target}/{candidate}")
    return targets


def apply_selector(
    *,
    selector: str,
    oof_by_target: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evaluation_by_target: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_city_groups: Mapping[str, str],
) -> dict[str, Any]:
    targets = _validate_development_inputs(
        oof_by_target=oof_by_target,
        evaluation_by_target=evaluation_by_target,
        target_city_groups=target_city_groups,
    )
    if selector != ALWAYS_ANCHOR_SELECTOR and selector not in SELECTOR_SPECS:
        raise ValueError(f"unknown v40 selector: {selector}")
    decisions = {}
    for target in targets:
        if selector == ALWAYS_ANCHOR_SELECTOR:
            decision = {
                "selected_candidate": ANCHOR_KEY,
                "selected_summary": {"mean_alpha": 0.0},
                "admissible_candidates": [],
            }
        else:
            decision = select_target_candidate(
                oof_by_target[target], **SELECTOR_SPECS[selector]
            )
        selected = decision["selected_candidate"]
        decisions[target] = {
            "target": target,
            "city_group": target_city_groups[target],
            "selected_candidate": selected,
            "evaluation_regret": float(
                evaluation_by_target[target][selected]["regret"]
            ),
            "anchor_evaluation_regret": float(
                evaluation_by_target[target][ANCHOR_KEY]["regret"]
            ),
            "mean_alpha": float(
                evaluation_by_target[target][selected]["mean_alpha"]
            ),
            "target_selection": decision,
        }
    city_rows = {}
    for city in DEVELOPMENT_CITY_GROUPS:
        city_targets = [
            target for target in targets if target_city_groups[target] == city
        ]
        city_rows[city] = {
            "target_count": len(city_targets),
            "anchor_regret": float(
                np.mean(
                    [decisions[target]["anchor_evaluation_regret"] for target in city_targets]
                )
            ),
            "selected_regret": float(
                np.mean(
                    [decisions[target]["evaluation_regret"] for target in city_targets]
                )
            ),
            "mean_alpha": float(
                np.mean([decisions[target]["mean_alpha"] for target in city_targets])
            ),
            "selected_non_anchor_target_count": sum(
                decisions[target]["selected_candidate"] != ANCHOR_KEY
                for target in city_targets
            ),
        }
        city_rows[city]["improvement"] = (
            city_rows[city]["anchor_regret"] - city_rows[city]["selected_regret"]
        )
    return {
        "protocol": "apply-v40-target-selector-to-city-macro-evaluation-v1",
        "selector": selector,
        "target_decisions": decisions,
        "city_rows": city_rows,
    }


def _selector_training_summary(
    application: Mapping[str, Any],
    *,
    training_cities: Sequence[str],
) -> dict[str, Any]:
    cities = tuple(training_cities)
    rows = application["city_rows"]
    anchor = np.asarray([rows[city]["anchor_regret"] for city in cities], dtype=float)
    selected = np.asarray([rows[city]["selected_regret"] for city in cities], dtype=float)
    improvements = anchor - selected
    anchor_macro = float(np.mean(anchor))
    selected_macro = float(np.mean(selected))
    relative = (
        (anchor_macro - selected_macro) / anchor_macro
        if anchor_macro > NUMERICAL_TOLERANCE
        else 0.0
    )
    return {
        "selector": application["selector"],
        "training_cities": list(cities),
        "anchor_macro_regret": anchor_macro,
        "selected_macro_regret": selected_macro,
        "macro_relative_improvement": relative,
        "improved_city_count": int(np.sum(improvements > NUMERICAL_TOLERANCE)),
        "maximum_absolute_city_regression": float(
            np.max(np.maximum(-improvements, 0.0))
        ),
        "worst_city_regret": float(np.max(selected)),
        "mean_selected_alpha": float(
            np.mean([rows[city]["mean_alpha"] for city in cities])
        ),
        "city_improvements": {
            city: float(value)
            for city, value in zip(cities, improvements.tolist(), strict=True)
        },
    }


def select_meta_selector(
    *,
    oof_by_target: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evaluation_by_target: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_city_groups: Mapping[str, str],
    training_cities: Sequence[str],
    minimum_improved_cities: int,
) -> dict[str, Any]:
    cities = tuple(str(value) for value in training_cities)
    if not cities or len(cities) != len(set(cities)):
        raise ValueError("v40 training cities must be nonempty and unique")
    if not set(cities) <= set(DEVELOPMENT_CITY_GROUPS):
        raise ValueError("v40 training cities are outside the development set")
    applications = {
        selector: apply_selector(
            selector=selector,
            oof_by_target=oof_by_target,
            evaluation_by_target=evaluation_by_target,
            target_city_groups=target_city_groups,
        )
        for selector in (ALWAYS_ANCHOR_SELECTOR, *SELECTOR_KEYS)
    }
    summaries = {
        selector: _selector_training_summary(
            application, training_cities=cities
        )
        for selector, application in applications.items()
    }
    admissible = [
        row
        for selector, row in summaries.items()
        if selector != ALWAYS_ANCHOR_SELECTOR
        and row["macro_relative_improvement"] + NUMERICAL_TOLERANCE
        >= MIN_RELATIVE_IMPROVEMENT
        and row["improved_city_count"] >= int(minimum_improved_cities)
        and row["maximum_absolute_city_regression"]
        <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE
    ]
    if admissible:
        selected = min(
            admissible,
            key=lambda row: (
                row["selected_macro_regret"],
                row["worst_city_regret"],
                row["mean_selected_alpha"],
                row["selector"],
            ),
        )
    else:
        selected = summaries[ALWAYS_ANCHOR_SELECTOR]
    selected_key = selected["selector"]
    return {
        "protocol": "nested-city-macro-v40-selector-selection-v1",
        "training_cities": list(cities),
        "minimum_improved_cities": int(minimum_improved_cities),
        "selected_selector": selected_key,
        "selected_summary": selected,
        "selected_application": applications[selected_key],
        "admissible_selectors": sorted(row["selector"] for row in admissible),
        "selector_summaries": summaries,
    }


def summarize_cross_fitted_anchored_development(
    *,
    oof_by_target: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evaluation_by_target: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_city_groups: Mapping[str, str],
) -> dict[str, Any]:
    _validate_development_inputs(
        oof_by_target=oof_by_target,
        evaluation_by_target=evaluation_by_target,
        target_city_groups=target_city_groups,
    )
    loco_rows = []
    for heldout in DEVELOPMENT_CITY_GROUPS:
        training = tuple(city for city in DEVELOPMENT_CITY_GROUPS if city != heldout)
        selection = select_meta_selector(
            oof_by_target=oof_by_target,
            evaluation_by_target=evaluation_by_target,
            target_city_groups=target_city_groups,
            training_cities=training,
            minimum_improved_cities=MIN_TRAINING_CITY_IMPROVEMENTS,
        )
        city = selection["selected_application"]["city_rows"][heldout]
        loco_rows.append(
            {
                "heldout_city": heldout,
                "selected_selector": selection["selected_selector"],
                "anchor_regret": city["anchor_regret"],
                "selected_regret": city["selected_regret"],
                "improvement": city["improvement"],
                "selected_non_anchor_target_count": city[
                    "selected_non_anchor_target_count"
                ],
                "selection": selection,
            }
        )
    anchor_macro = float(np.mean([row["anchor_regret"] for row in loco_rows]))
    selected_macro = float(np.mean([row["selected_regret"] for row in loco_rows]))
    improvements = np.asarray([row["improvement"] for row in loco_rows], dtype=float)
    relative = (
        (anchor_macro - selected_macro) / anchor_macro
        if anchor_macro > NUMERICAL_TOLERANCE
        else 0.0
    )
    loco_gate = {
        "macro_relative_improvement_at_least_0_02": (
            relative + NUMERICAL_TOLERANCE >= MIN_RELATIVE_IMPROVEMENT
        ),
        "at_least_five_heldout_cities_improve": int(
            np.sum(improvements > NUMERICAL_TOLERANCE)
        )
        >= MIN_GLOBAL_CITY_IMPROVEMENTS,
        "maximum_absolute_city_regression_at_most_0_01": float(
            np.max(np.maximum(-improvements, 0.0))
        )
        <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE,
    }
    loco_gate["passed"] = all(loco_gate.values())
    global_selection = select_meta_selector(
        oof_by_target=oof_by_target,
        evaluation_by_target=evaluation_by_target,
        target_city_groups=target_city_groups,
        training_cities=DEVELOPMENT_CITY_GROUPS,
        minimum_improved_cities=MIN_GLOBAL_CITY_IMPROVEMENTS,
    )
    global_summary = global_selection["selected_summary"]
    global_gate = {
        "selected_non_anchor_selector": (
            global_selection["selected_selector"] != ALWAYS_ANCHOR_SELECTOR
        ),
        "macro_relative_improvement_at_least_0_02": (
            global_summary["macro_relative_improvement"] + NUMERICAL_TOLERANCE
            >= MIN_RELATIVE_IMPROVEMENT
        ),
        "at_least_five_cities_improve": (
            global_summary["improved_city_count"]
            >= MIN_GLOBAL_CITY_IMPROVEMENTS
        ),
        "maximum_absolute_city_regression_at_most_0_01": (
            global_summary["maximum_absolute_city_regression"]
            <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE
        ),
    }
    global_gate["passed"] = all(global_gate.values())
    passed = bool(loco_gate["passed"] and global_gate["passed"])
    return {
        "protocol": "seven-city-nested-cross-fitted-anchored-development-v1",
        "passed": passed,
        "decision": (
            "freeze_cross_fitted_anchored_selector_for_external_confirmation"
            if passed
            else "reject_v40_cross_fitted_anchored_selector"
        ),
        "loco": {
            "anchor_macro_regret": anchor_macro,
            "selected_macro_regret": selected_macro,
            "macro_relative_improvement": relative,
            "macro_relative_improvement_pct": 100.0 * relative,
            "improved_city_count": int(
                np.sum(improvements > NUMERICAL_TOLERANCE)
            ),
            "maximum_absolute_city_regression": float(
                np.max(np.maximum(-improvements, 0.0))
            ),
            "rows": loco_rows,
            "gate": loco_gate,
        },
        "global_selection": global_selection,
        "global_gate": global_gate,
    }
