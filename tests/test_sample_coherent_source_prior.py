import numpy as np

from cf_h2o.eval.traffic_signal_sample_coherent_source_prior import (
    RESULT_PROTOCOL,
    aggregate_results,
    prior_strength_for_budget,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    TARGET_BUDGETS,
)
from tests.test_source_identifiability_budget_curve import _curve


def test_prior_strength_decreases_with_target_group_count() -> None:
    assert np.isclose(prior_strength_for_budget(25), 0.10)
    assert np.isclose(prior_strength_for_budget(50), 0.05)
    assert np.isclose(prior_strength_for_budget(100), 0.025)


def test_sample_coherent_aggregate_preserves_budget_gate() -> None:
    rows = _curve(primary_admitted=2)
    for row in rows:
        budget = row["target_budget"]
        strength = prior_strength_for_budget(budget)
        row["protocol"] = RESULT_PROTOCOL
        row["method"]["source_prior_strength"] = strength
        row["method"]["prior_schedule"] = {
            "protocol": "inverse-target-group-count-v1",
            "effective_strength": strength,
        }

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_prior_strength_by_budget"] == {
        str(budget): prior_strength_for_budget(budget)
        for budget in TARGET_BUDGETS
    }
