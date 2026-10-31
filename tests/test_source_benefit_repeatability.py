from copy import deepcopy

import pytest

from cf_h2o.eval.traffic_signal_multicity_source_compatibility_inventory import (
    RESULT_PROTOCOL as V144_TARGET_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_source_benefit_repeatability import (
    aggregate_source_benefit_repeatability,
)


SEEDS = ("1", "2", "3")


def _summary(values: dict[str, float]) -> dict:
    return {
        "mean": sum(values.values()) / len(values),
        "seed_values": values,
    }


def _rows() -> list[dict]:
    rows = []
    for target in EXPECTED_CITY_GROUPS:
        sources = [city for city in EXPECTED_CITY_GROUPS if city != target]
        target_values = {seed: 1.0 for seed in SEEDS}
        source_arms = {}
        placebo_arms = {}
        for index, source in enumerate(sources):
            if index == 0:
                effects = {"1": -0.3, "2": -0.2, "3": -0.1}
            else:
                effects = {
                    "1": 0.01 * index,
                    "2": 0.02 * index,
                    "3": 0.03 * index,
                }
            source_values = {
                seed: target_values[seed] + effects[seed] for seed in SEEDS
            }
            source_arms[source] = _summary(source_values)
            placebo_arms[source] = _summary(
                {seed: source_values[seed] + 0.05 for seed in SEEDS}
            )
        rows.append(
            {
                "protocol": V144_TARGET_PROTOCOL,
                "target_city": target,
                "target_budget": 25,
                "source_city_groups": sources,
                "information_budget": {
                    "evaluation_groups_used_for_fit_or_selection": False
                },
                "arm_summaries": {
                    "target_only_causal": _summary(target_values)
                },
                "source_arm_summaries": source_arms,
                "source_placebo_arm_summaries": placebo_arms,
            }
        )
    return rows


def test_v150a_recovers_repeatable_source_without_heldout_seed_leakage() -> None:
    result = aggregate_source_benefit_repeatability(_rows())
    summary = result["repeatability_summary"]
    forced = summary["forced_source_leave_one_seed_out"]
    assert forced["effect_vs_target_only"]["mean"] == pytest.approx(-0.2)
    assert forced["effect_vs_target_only"]["improving_fraction"] == 1.0
    assert forced["effect_vs_matched_placebo"]["mean"] == pytest.approx(-0.05)
    assert forced["improving_city_count_vs_target_only"] == 7
    assert result["information_boundary"]["deployable_target_gate"] is False
    for target in result["target_diagnostics"].values():
        for fold in target["forced_source_leave_one_seed_out"]:
            assert fold["heldout_seed"] not in fold["training_seeds"]


def test_v150a_source_null_aware_selector_can_refuse_transfer() -> None:
    rows = _rows()
    target = EXPECTED_CITY_GROUPS[0]
    for source in rows[0]["source_arm_summaries"]:
        rows[0]["source_arm_summaries"][source]["seed_values"] = {
            seed: 1.1 for seed in SEEDS
        }
    result = aggregate_source_benefit_repeatability(rows)
    folds = result["target_diagnostics"][target][
        "source_null_aware_leave_one_seed_out"
    ]
    assert all(fold["source_admitted"] is False for fold in folds)
    assert all(fold["heldout_effect_vs_target_only"] == 0.0 for fold in folds)


def test_v150a_rejects_evaluation_selection_or_incomplete_seed_values() -> None:
    rows = _rows()
    changed = deepcopy(rows)
    changed[0]["information_budget"][
        "evaluation_groups_used_for_fit_or_selection"
    ] = True
    with pytest.raises(ValueError, match="input boundary changed"):
        aggregate_source_benefit_repeatability(changed)

    changed = deepcopy(rows)
    source = next(iter(changed[0]["source_arm_summaries"]))
    del changed[0]["source_arm_summaries"][source]["seed_values"]["3"]
    with pytest.raises(ValueError, match="seed-value coverage changed"):
        aggregate_source_benefit_repeatability(changed)
