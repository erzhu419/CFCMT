import pytest

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    RESULT_PROTOCOL,
    TARGET_BUDGETS,
    aggregate_results,
)


def _result(
    city: str,
    budget: int,
    *,
    admitted: bool,
    target_effect: float,
    placebo_effect: float,
) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): rigid[str(seed)] + target_effect for seed in SOURCE_SEEDS
    }
    placebo = {
        str(seed): source[str(seed)] - placebo_effect for seed in SOURCE_SEEDS
    }
    selected = [f"group-{index:03d}" for index in range(budget)]
    reserve = [f"group-{index:03d}" for index in range(100)]
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": budget,
        "target_scenarios": [city],
        "selection_audit": {"selected_group_ids": selected},
        "evaluation_reserve_audit": {"selected_group_ids": reserve},
        "method": {
            "selector": {
                "candidate": "source|spillback",
                "admitted": admitted,
                "crossfitted_selection_audit": {"passed": admitted},
            }
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {"seed_values": placebo},
        },
    }


def _curve(primary_admitted: int) -> list[dict]:
    rows = []
    for budget in TARGET_BUDGETS:
        admitted_count = 1 if budget < 100 else primary_admitted
        for index, city in enumerate(EXPECTED_CITY_GROUPS):
            admitted = index < admitted_count
            rows.append(
                _result(
                    city,
                    budget,
                    admitted=admitted,
                    target_effect=-0.02 if admitted else 0.0,
                    placebo_effect=-0.01 if admitted else 0.0,
                )
            )
    return rows


def test_budget_curve_passes_when_b100_identifies_two_sources() -> None:
    result = aggregate_results(_curve(primary_admitted=2))

    assert result["development_gate"]["passed"] is True
    assert result["budget_results"]["100"]["source_admission_count"] == 2
    assert result["curve_diagnostics"]["admission_count_nondecreasing"] is True


def test_budget_curve_rejects_when_b100_identifies_only_one_source() -> None:
    result = aggregate_results(_curve(primary_admitted=1))

    assert result["development_gate"]["passed"] is False
    assert (
        result["development_gate"]["checks"][
            "at_least_two_cities_admit_crossfitted_source"
        ]
        is False
    )


def test_budget_curve_rejects_non_nested_target_groups() -> None:
    rows = _curve(primary_admitted=2)
    row = next(
        value
        for value in rows
        if value["target_city"] == EXPECTED_CITY_GROUPS[0]
        and value["target_budget"] == 50
    )
    row["selection_audit"]["selected_group_ids"][0] = "not-in-b25"

    with pytest.raises(ValueError, match="not nested"):
        aggregate_results(rows)
