import json
from pathlib import Path

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import TARGET_BUDGETS
from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting import (
    RESULT_PROTOCOL as BUDGET_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting_aggregate import (
    aggregate_budget_results,
    simultaneous_max_t_intervals,
)


ADAPTATION_SEEDS = [1, 2, 3, 4, 5]
EVALUATION_SEEDS = list(range(101, 118))
SOURCES = [f"source_{index}" for index in range(7)]


def _budget_result(budget: int) -> dict:
    offset = TARGET_BUDGETS.index(budget) * 0.0001
    source_target = [-0.01 - offset + (index - 8) * 0.00001 for index in range(17)]
    source_placebo = [-0.005 - offset + (index - 8) * 0.00001 for index in range(17)]
    return {
        "protocol": BUDGET_RESULT_PROTOCOL,
        "budget": budget,
        "primary_budget": 25,
        "adaptation_seeds": ADAPTATION_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "source_group_order": SOURCES,
        "inputs": {"fit": "a", "source": "b", "selector": "c"},
        "information_budget": {
            "union_target_group_count": budget,
            "evaluation_seed_labels_used_for_fit_weighting_or_selection": False,
            "adaptation_group_ids": [f"group-{index}" for index in range(budget)],
        },
        "paired_effects": {
            "source_weighted_minus_target_only": {
                "mean": float(np.mean(source_target)),
                "seed_values": source_target,
            },
            "source_weighted_minus_matched_placebo": {
                "mean": float(np.mean(source_placebo)),
                "seed_values": source_placebo,
            },
        },
        "development_gate": {
            "adaptation_source_authorized": True,
            "selector_relative_passed": budget == 25,
            "primary_budget_passed": budget == 25,
        },
        "effective_source_null_weight": 0.3,
        "effective_source_weights": {source: 0.1 for source in SOURCES},
        "arm_summaries": {
            "target_only": {"mean": 0.02},
            "causal_source_weighted": {"mean": 0.01},
        },
    }


def _write_results(tmp_path: Path) -> list[Path]:
    paths = []
    for budget in TARGET_BUDGETS:
        path = tmp_path / f"B{budget}.json"
        path.write_text(json.dumps(_budget_result(budget)), encoding="utf-8")
        paths.append(path)
    return paths


def test_simultaneous_max_t_uses_shared_seed_matrix() -> None:
    values = np.asarray(
        [[float(seed + budget) for budget in range(3)] for seed in range(8)]
    )
    result = simultaneous_max_t_intervals(values, replicates=500, seed=7)
    assert result["resampling_unit"] == "shared_evaluation_seed_across_all_budgets"
    assert result["critical_value"] > 0.0
    assert len(result["upper_95_simultaneous"]) == 3
    assert np.all(
        np.asarray(result["lower_95_simultaneous"])
        <= np.mean(values, axis=0)
    )


def test_aggregate_preserves_primary_budget_and_absolute_columns(tmp_path: Path) -> None:
    result = aggregate_budget_results(_write_results(tmp_path))
    assert result["development_decision"]["primary_budget_passed"] is True
    assert result["development_decision"]["budget_selection_after_evaluation_permitted"] is False
    assert [row["budget"] for row in result["curve"]] == list(TARGET_BUDGETS)
    assert result["curve"][0]["target_only_minus_phase_pressure"] == pytest.approx(0.02)
    assert result["curve"][0]["source_weighted_minus_phase_pressure"] == pytest.approx(0.01)
    assert (
        result["curve"][0]["source_weighted_minus_target_only"][
            "upper_95_simultaneous"
        ]
        < 0.0
    )


def test_aggregate_rejects_non_nested_budget_groups(tmp_path: Path) -> None:
    paths = _write_results(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["information_budget"]["adaptation_group_ids"][0] = "not-in-b25"
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not nested"):
        aggregate_budget_results(paths)


def test_aggregate_rejects_input_identity_drift(tmp_path: Path) -> None:
    paths = _write_results(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["inputs"]["selector"] = "changed"
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="input identity changed"):
        aggregate_budget_results(paths)
