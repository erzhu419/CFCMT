from cf_h2o.eval.traffic_signal_external_heldout_evaluation import (
    _equal_scenario_city_summary,
    _paired_bootstrap,
)


def _scenario(regrets_a, regrets_b):
    rows = []
    for index, (a, b) in enumerate(zip(regrets_a, regrets_b, strict=True)):
        rows.append(
            {
                "group_id": f"g{index}",
                "policies": {
                    "policy": {"normalized_action_regret": a},
                    "reference": {"normalized_action_regret": b},
                },
            }
        )
    return {
        "summaries": {
            "policy": {
                "mean_normalized_action_regret": sum(regrets_a) / len(regrets_a),
                "optimal_action_rate": 0.5,
            },
            "reference": {
                "mean_normalized_action_regret": sum(regrets_b) / len(regrets_b),
                "optimal_action_rate": 0.25,
            },
        },
        "group_rows": rows,
    }


def test_equal_scenario_summary_does_not_weight_by_group_count():
    results = {
        "short": _scenario([0.0], [0.5]),
        "long": _scenario([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    }
    summary = _equal_scenario_city_summary(results)
    assert summary["policy"]["mean_normalized_action_regret"] == 0.5
    assert summary["reference"]["mean_normalized_action_regret"] == 0.75


def test_paired_bootstrap_positive_means_policy_improves():
    results = {
        "a": _scenario([0.0, 0.1, 0.2], [0.5, 0.6, 0.7]),
        "b": _scenario([0.2, 0.3], [0.4, 0.5]),
    }
    inference = _paired_bootstrap(
        results,
        policy="policy",
        reference="reference",
        seed=17,
    )
    assert inference["observed_improvement"] > 0.0
    assert inference["ci95"][0] > 0.0
