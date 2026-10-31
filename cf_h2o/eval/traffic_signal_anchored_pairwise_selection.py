"""Frozen city-level selector for anchored pairwise TSC development."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


DEVELOPMENT_CITY_GROUPS = (
    "resco_synthetic",
    "cologne",
    "ingolstadt",
    "atlanta",
    "hangzhou",
    "new_york",
    "salt_lake_city",
)
CONSTANT_ALPHAS = tuple(round(index / 10.0, 1) for index in range(11))
DISAGREEMENT_CAPS = (0.25, 0.5, 1.0, 2.0, 4.0)
CONFIDENCE_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)
MIN_RELATIVE_IMPROVEMENT = 0.02
MAX_ABSOLUTE_CITY_REGRESSION = 0.01
MIN_TRAINING_CITY_IMPROVEMENTS = 4
MIN_GLOBAL_CITY_IMPROVEMENTS = 5
NUMERICAL_TOLERANCE = 1e-12


def _number_key(prefix: str, value: float) -> str:
    return f"{prefix}{float(value):g}".replace(".", "p")


def constant_candidate_key(alpha: float) -> str:
    return _number_key("constant_alpha_", alpha)


def local_candidate_key(cap: float, confidence: float) -> str:
    return (
        _number_key("local_cap_", cap)
        + "_"
        + _number_key("z_", confidence)
    )


def candidate_keys() -> tuple[str, ...]:
    return tuple(constant_candidate_key(value) for value in CONSTANT_ALPHAS) + tuple(
        local_candidate_key(cap, confidence)
        for cap in DISAGREEMENT_CAPS
        for confidence in CONFIDENCE_MULTIPLIERS
    )


ANCHOR_KEY = constant_candidate_key(0.0)
PAIRWISE_KEY = constant_candidate_key(1.0)
CANDIDATE_KEYS = candidate_keys()


def _validated_city_rows(
    city_rows: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    if set(city_rows) != set(DEVELOPMENT_CITY_GROUPS):
        raise ValueError("anchored development city-group set changed")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for city in DEVELOPMENT_CITY_GROUPS:
        candidates = city_rows[city]
        if set(candidates) != set(CANDIDATE_KEYS):
            raise ValueError(f"anchored candidate set changed for {city}")
        result[city] = {}
        for candidate in CANDIDATE_KEYS:
            regret = float(candidates[candidate]["regret"])
            mean_alpha = float(candidates[candidate]["mean_alpha"])
            if not math.isfinite(regret) or regret < 0.0:
                raise ValueError(f"invalid regret for {city}/{candidate}")
            if not math.isfinite(mean_alpha) or not 0.0 <= mean_alpha <= 1.0:
                raise ValueError(f"invalid mean alpha for {city}/{candidate}")
            result[city][candidate] = {
                "regret": regret,
                "mean_alpha": mean_alpha,
            }
    return result


def _candidate_training_summary(
    rows: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    candidate: str,
    cities: Sequence[str],
) -> dict[str, Any]:
    anchor_values = np.asarray(
        [rows[city][ANCHOR_KEY]["regret"] for city in cities], dtype=float
    )
    candidate_values = np.asarray(
        [rows[city][candidate]["regret"] for city in cities], dtype=float
    )
    anchor_macro = float(np.mean(anchor_values))
    candidate_macro = float(np.mean(candidate_values))
    relative_improvement = (
        (anchor_macro - candidate_macro) / anchor_macro
        if anchor_macro > NUMERICAL_TOLERANCE
        else 0.0
    )
    improvements = anchor_values - candidate_values
    return {
        "candidate": candidate,
        "anchor_macro_regret": anchor_macro,
        "candidate_macro_regret": candidate_macro,
        "macro_relative_improvement": relative_improvement,
        "improved_city_count": int(np.sum(improvements > NUMERICAL_TOLERANCE)),
        "maximum_absolute_city_regression": float(
            np.max(np.maximum(-improvements, 0.0))
        ),
        "worst_city_regret": float(np.max(candidate_values)),
        "mean_correction_weight": float(
            np.mean([rows[city][candidate]["mean_alpha"] for city in cities])
        ),
        "city_improvements": {
            city: float(value)
            for city, value in zip(cities, improvements.tolist(), strict=True)
        },
    }


def select_anchored_candidate(
    city_rows: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    training_cities: Sequence[str],
    minimum_improved_cities: int,
) -> dict[str, Any]:
    rows = _validated_city_rows(city_rows)
    cities = tuple(str(value) for value in training_cities)
    if not cities or len(cities) != len(set(cities)):
        raise ValueError("training cities must be nonempty and unique")
    if not set(cities) <= set(DEVELOPMENT_CITY_GROUPS):
        raise ValueError("training cities are outside the development set")
    summaries = {
        candidate: _candidate_training_summary(
            rows,
            candidate=candidate,
            cities=cities,
        )
        for candidate in CANDIDATE_KEYS
    }
    admissible = [
        summary
        for candidate, summary in summaries.items()
        if candidate != ANCHOR_KEY
        and summary["macro_relative_improvement"] + NUMERICAL_TOLERANCE
        >= MIN_RELATIVE_IMPROVEMENT
        and summary["improved_city_count"] >= int(minimum_improved_cities)
        and summary["maximum_absolute_city_regression"]
        <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE
    ]
    if admissible:
        selected = min(
            admissible,
            key=lambda summary: (
                summary["candidate_macro_regret"],
                summary["worst_city_regret"],
                summary["mean_correction_weight"],
                summary["candidate"],
            ),
        )
    else:
        selected = summaries[ANCHOR_KEY]
    return {
        "protocol": "anchored-pairwise-city-macro-selection-v1",
        "training_cities": list(cities),
        "minimum_improved_cities": int(minimum_improved_cities),
        "selected_candidate": selected["candidate"],
        "selected_summary": selected,
        "admissible_candidates": sorted(
            summary["candidate"] for summary in admissible
        ),
        "candidate_summaries": summaries,
    }


def summarize_anchored_development(
    city_rows: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    rows = _validated_city_rows(city_rows)
    loco_rows = []
    for heldout in DEVELOPMENT_CITY_GROUPS:
        training = tuple(city for city in DEVELOPMENT_CITY_GROUPS if city != heldout)
        selection = select_anchored_candidate(
            rows,
            training_cities=training,
            minimum_improved_cities=MIN_TRAINING_CITY_IMPROVEMENTS,
        )
        selected = selection["selected_candidate"]
        anchor_regret = rows[heldout][ANCHOR_KEY]["regret"]
        selected_regret = rows[heldout][selected]["regret"]
        loco_rows.append(
            {
                "heldout_city": heldout,
                "selected_candidate": selected,
                "anchor_regret": anchor_regret,
                "selected_regret": selected_regret,
                "improvement": anchor_regret - selected_regret,
                "selection": selection,
            }
        )
    loco_anchor_macro = float(np.mean([row["anchor_regret"] for row in loco_rows]))
    loco_selected_macro = float(
        np.mean([row["selected_regret"] for row in loco_rows])
    )
    loco_improvements = np.asarray(
        [row["improvement"] for row in loco_rows], dtype=float
    )
    loco_relative_improvement = (
        (loco_anchor_macro - loco_selected_macro) / loco_anchor_macro
        if loco_anchor_macro > NUMERICAL_TOLERANCE
        else 0.0
    )
    loco_gate = {
        "macro_relative_improvement_at_least_0_02": (
            loco_relative_improvement + NUMERICAL_TOLERANCE
            >= MIN_RELATIVE_IMPROVEMENT
        ),
        "at_least_five_heldout_cities_improve": int(
            np.sum(loco_improvements > NUMERICAL_TOLERANCE)
        )
        >= MIN_GLOBAL_CITY_IMPROVEMENTS,
        "maximum_absolute_city_regression_at_most_0_01": float(
            np.max(np.maximum(-loco_improvements, 0.0))
        )
        <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE,
    }
    loco_gate["passed"] = all(loco_gate.values())

    global_selection = select_anchored_candidate(
        rows,
        training_cities=DEVELOPMENT_CITY_GROUPS,
        minimum_improved_cities=MIN_GLOBAL_CITY_IMPROVEMENTS,
    )
    selected_summary = global_selection["selected_summary"]
    global_gate = {
        "selected_non_anchor_candidate": (
            global_selection["selected_candidate"] != ANCHOR_KEY
        ),
        "macro_relative_improvement_at_least_0_02": (
            selected_summary["macro_relative_improvement"] + NUMERICAL_TOLERANCE
            >= MIN_RELATIVE_IMPROVEMENT
        ),
        "at_least_five_cities_improve": (
            selected_summary["improved_city_count"]
            >= MIN_GLOBAL_CITY_IMPROVEMENTS
        ),
        "maximum_absolute_city_regression_at_most_0_01": (
            selected_summary["maximum_absolute_city_regression"]
            <= MAX_ABSOLUTE_CITY_REGRESSION + NUMERICAL_TOLERANCE
        ),
    }
    global_gate["passed"] = all(global_gate.values())
    passed = bool(loco_gate["passed"] and global_gate["passed"])
    return {
        "protocol": "seven-city-loco-anchored-pairwise-development-v1",
        "passed": passed,
        "decision": (
            "freeze_anchored_pairwise_for_external_confirmation"
            if passed
            else "reject_v39_anchored_pairwise"
        ),
        "loco": {
            "anchor_macro_regret": loco_anchor_macro,
            "selected_macro_regret": loco_selected_macro,
            "macro_relative_improvement": loco_relative_improvement,
            "macro_relative_improvement_pct": 100.0 * loco_relative_improvement,
            "improved_city_count": int(
                np.sum(loco_improvements > NUMERICAL_TOLERANCE)
            ),
            "maximum_absolute_city_regression": float(
                np.max(np.maximum(-loco_improvements, 0.0))
            ),
            "rows": loco_rows,
            "gate": loco_gate,
        },
        "global_selection": global_selection,
        "global_gate": global_gate,
    }
