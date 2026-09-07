"""Frozen selector for the v34 target-simulator pairwise adaptation curve."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    RIGID_FAMILY,
    TARGETS,
)


TARGET_ONLY_FAMILY = "causal_target_only"
TARGET_GROUP_BUDGETS = (0, 8, 16, 32, 60)
TARGET_ADAPTATION_GROUPS = {0: 0, 8: 4, 16: 9, 32: 19, 60: 36}
CURVE_FAMILIES = (
    RIGID_FAMILY,
    GROUP_NORMALIZED_RIGID_FAMILY,
    TARGET_ONLY_FAMILY,
    PAIRWISE_PREFERENCE_FAMILY,
)
PRIMARY_BUDGET = 60
MACRO_IMPROVEMENT_THRESHOLD_PCT = 10.0
MIN_IMPROVED_TARGETS = 4
MAX_ABSOLUTE_REGRESSION = 0.05
MAX_SPEARMAN_CORRELATION = -0.8
NUMERICAL_TOLERANCE = 1e-12


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=float)
    position = 0
    while position < array.size:
        end = position + 1
        while end < array.size and array[order[end]] == array[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1)
        position = end
    return ranks


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    x_rank = _average_ranks(values_x)
    y_rank = _average_ranks(values_y)
    if np.std(x_rank) <= 0.0 or np.std(y_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _validated_regrets(
    regrets: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[int, dict[str, dict[str, float]]]:
    if set(regrets) != set(TARGET_GROUP_BUDGETS):
        raise ValueError("target-budget set differs from the frozen curve")
    validated: dict[int, dict[str, dict[str, float]]] = {}
    for budget in TARGET_GROUP_BUDGETS:
        family_rows = regrets[budget]
        if set(family_rows) != set(CURVE_FAMILIES):
            raise ValueError(f"family set changed at budget {budget}")
        validated[budget] = {}
        for family in CURVE_FAMILIES:
            target_rows = family_rows[family]
            if set(target_rows) != set(TARGETS):
                raise ValueError(f"target set changed for {family} at budget {budget}")
            validated[budget][family] = {}
            for target in TARGETS:
                value = float(target_rows[target])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"invalid regret for budget={budget}/{family}/{target}: {value}"
                    )
                validated[budget][family][target] = value
    return validated


def summarize_pairwise_target_curve(
    regrets: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Apply the preregistered same-subset efficacy and trend gates."""

    values = _validated_regrets(regrets)
    budget_rows: dict[int, dict[str, Any]] = {}
    pairwise_macros = []
    for budget in TARGET_GROUP_BUDGETS:
        family_macros = {
            family: float(np.mean(list(values[budget][family].values())))
            for family in CURVE_FAMILIES
        }
        rigid = values[budget][RIGID_FAMILY]
        pairwise = values[budget][PAIRWISE_PREFERENCE_FAMILY]
        rigid_macro = family_macros[RIGID_FAMILY]
        pairwise_macro = family_macros[PAIRWISE_PREFERENCE_FAMILY]
        if rigid_macro <= NUMERICAL_TOLERANCE:
            raise ValueError(f"rigid macro regret is zero at budget {budget}")
        improvements = {
            target: rigid[target] - pairwise[target] for target in TARGETS
        }
        budget_rows[budget] = {
            "target_group_budget": budget,
            "family_macro_regrets": family_macros,
            "relative_pairwise_improvement_vs_rigid_pct": (
                100.0 * (rigid_macro - pairwise_macro) / rigid_macro
            ),
            "pairwise_cities_improved": sum(
                value > NUMERICAL_TOLERANCE for value in improvements.values()
            ),
            "pairwise_max_absolute_regression": max(
                max(-value, 0.0) for value in improvements.values()
            ),
            "pairwise_target_improvements_vs_rigid": improvements,
        }
        pairwise_macros.append(pairwise_macro)

    primary = budget_rows[PRIMARY_BUDGET]
    spearman = _spearman(TARGET_GROUP_BUDGETS, pairwise_macros)
    gate = {
        "budget_60_macro_improvement_at_least_10_pct": (
            primary["relative_pairwise_improvement_vs_rigid_pct"]
            + NUMERICAL_TOLERANCE
            >= MACRO_IMPROVEMENT_THRESHOLD_PCT
        ),
        "budget_60_at_least_four_cities_improved": (
            primary["pairwise_cities_improved"] >= MIN_IMPROVED_TARGETS
        ),
        "budget_60_max_regression_at_most_0_05": (
            primary["pairwise_max_absolute_regression"]
            <= MAX_ABSOLUTE_REGRESSION + NUMERICAL_TOLERANCE
        ),
        "pairwise_budget_60_better_than_zero_shot": (
            pairwise_macros[-1] + NUMERICAL_TOLERANCE < pairwise_macros[0]
        ),
        "pairwise_budget_regret_spearman_at_most_minus_0_8": (
            spearman <= MAX_SPEARMAN_CORRELATION + NUMERICAL_TOLERANCE
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "frozen-pairwise-target-simulator-curve-selector-v1",
        "passed": gate["passed"],
        "decision": (
            "promote_pairwise_target_adaptation"
            if gate["passed"]
            else "reject_pooled_pairwise_target_adaptation"
        ),
        "primary_budget": PRIMARY_BUDGET,
        "budgets": list(TARGET_GROUP_BUDGETS),
        "pairwise_macro_regret_curve": dict(
            zip(TARGET_GROUP_BUDGETS, pairwise_macros, strict=True)
        ),
        "pairwise_budget_regret_spearman": spearman,
        "budget_rows": budget_rows,
        "stage_gate": gate,
    }
