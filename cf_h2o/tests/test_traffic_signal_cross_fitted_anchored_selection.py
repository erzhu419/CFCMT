from __future__ import annotations

from pathlib import Path

from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
    PAIRWISE_KEY,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_development import (
    _load_protocol,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    SEED_BLOCKED_SELECTION_PROTOCOL,
    select_target_candidate,
    summarize_cross_fitted_anchored_development,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _target_rows(*, pairwise_regret: float):
    rows = {
        candidate: {
            "fold_regrets": [0.31] * 5,
            "mean_alpha": 0.5,
        }
        for candidate in CANDIDATE_KEYS
    }
    rows[ANCHOR_KEY] = {"fold_regrets": [0.30] * 5, "mean_alpha": 0.0}
    rows[PAIRWISE_KEY] = {
        "fold_regrets": [pairwise_regret] * 5,
        "mean_alpha": 1.0,
    }
    return rows


def _evaluation_rows(*, pairwise_regret: float):
    rows = {
        candidate: {"regret": 0.31, "mean_alpha": 0.5}
        for candidate in CANDIDATE_KEYS
    }
    rows[ANCHOR_KEY] = {"regret": 0.30, "mean_alpha": 0.0}
    rows[PAIRWISE_KEY] = {"regret": pairwise_regret, "mean_alpha": 1.0}
    return rows


def test_target_selector_uses_stable_oof_pairwise_gain() -> None:
    decision = select_target_candidate(
        _target_rows(pairwise_regret=0.20),
        scope="constant",
        minimum_improvement=0.05,
        maximum_fold_regression=0.0,
        standard_error_multiplier=2.0,
    )

    assert decision["selected_candidate"] == PAIRWISE_KEY
    assert decision["selected_summary"]["oof_risk_upper_difference"] < 0.0


def test_target_selector_supports_explicit_seed_blocked_fold_count() -> None:
    rows = _target_rows(pairwise_regret=0.20)
    two_fold_rows = {
        candidate: {
            **row,
            "fold_regrets": row["fold_regrets"][:2],
        }
        for candidate, row in rows.items()
    }

    decision = select_target_candidate(
        two_fold_rows,
        scope="constant",
        minimum_improvement=0.05,
        maximum_fold_regression=0.0,
        standard_error_multiplier=2.0,
        expected_fold_count=2,
    )

    assert decision["selected_candidate"] == PAIRWISE_KEY
    assert decision["fold_count"] == 2
    assert decision["protocol"] == SEED_BLOCKED_SELECTION_PROTOCOL


def test_v40_protocol_freezes_four_by_four_by_four_by_two_grid() -> None:
    payload = _load_protocol(
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v23_cross_fitted_anchored_selector.json"
    )

    assert len(payload["development"]["evaluation_targets"]) == 14
    assert payload["development"]["cross_fit_folds"] == 5


def test_nested_selector_can_gate_one_reversed_city() -> None:
    oof = {}
    evaluation = {}
    city_groups = {}
    for city in DEVELOPMENT_CITY_GROUPS:
        target = f"target_{city}"
        reversed_city = city == "salt_lake_city"
        oof[target] = _target_rows(
            pairwise_regret=0.40 if reversed_city else 0.20
        )
        evaluation[target] = _evaluation_rows(
            pairwise_regret=0.40 if reversed_city else 0.20
        )
        city_groups[target] = city

    summary = summarize_cross_fitted_anchored_development(
        oof_by_target=oof,
        evaluation_by_target=evaluation,
        target_city_groups=city_groups,
    )

    assert summary["passed"] is True
    assert summary["loco"]["improved_city_count"] == 6
    assert summary["loco"]["maximum_absolute_city_regression"] == 0.0
