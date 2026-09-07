from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (
    summarize_quantile_candidate,
)


def test_quantile_candidate_uses_all_action_groups_as_deployed_denominator() -> None:
    all_records = []
    predictions = []
    for seed in (1, 2):
        for index in range(10):
            all_records.append(
                {
                    "group_id": f"s{seed}-{index}",
                    "simulator_seed": seed,
                    "scenario": "x",
                }
            )
        predictions.append(
            {
                "group_id": f"s{seed}-0",
                "simulator_seed": seed,
                "scenario": "x",
                "actual_group_normalized_delta": -1.0,
                "predicted_delta": -1.0,
                "fold_thresholds": {"0.1": -0.5},
            }
        )
    result = summarize_quantile_candidate(
        predictions,
        all_records=all_records,
        seeds=(1, 2),
        scenarios=("x",),
        acceptance_quantile=0.1,
        selection_rule={
            "minimum_retained_groups_per_city": 2,
            "minimum_retained_seed_fraction": 1.0,
            "maximum_equal_seed_scenario_mean_delta": -0.05,
            "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
            "maximum_worst_seed_mean_delta": 0.0,
            "maximum_harmful_group_fraction": 0.45,
        },
        bootstrap={"replicates": 100, "seed": 3},
    )

    assert result["mean_delta"] == -0.1
    assert result["retained_group_count"] == 2
    assert result["feasible"] is True
