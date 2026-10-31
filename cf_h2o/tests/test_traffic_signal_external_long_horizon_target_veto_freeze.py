from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    summarize_veto_gate_candidate,
)


def test_veto_gate_never_accepts_an_action_not_proposed_by_short_model() -> None:
    records = []
    for seed in (1, 2):
        for index in range(8):
            records.append(
                {
                    "simulator_seed": seed,
                    "scenario": "s",
                    "short_proposed": index < 7,
                    "target_predicted_delta": -1.0,
                    "target_uncertainty": 0.0,
                    "target_context_trust": 1.0,
                    "actual_group_normalized_delta": -0.5,
                }
            )
    summary = summarize_veto_gate_candidate(
        records,
        seeds=(1, 2),
        scenarios=("s",),
        risk_multiplier=1.0,
        min_context_trust=0.5,
        selection_rule={
            "minimum_retained_groups_per_city": 1,
            "minimum_retained_seed_fraction": 1.0,
            "maximum_equal_seed_scenario_mean_delta": -0.001,
            "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
            "maximum_worst_seed_mean_delta": 0.01,
            "maximum_harmful_group_fraction": 0.45,
        },
        bootstrap={"replicates": 100, "seed": 7},
    )

    assert summary["short_proposal_count"] == 14
    assert summary["retained_group_count"] == 14
    assert summary["feasible"] is True


def test_veto_gate_rejects_confidently_harmful_predictions() -> None:
    records = [
        {
            "simulator_seed": seed,
            "scenario": "s",
            "short_proposed": True,
            "target_predicted_delta": 0.1,
            "target_uncertainty": 0.0,
            "target_context_trust": 1.0,
            "actual_group_normalized_delta": -0.5,
        }
        for seed in (1, 2)
    ]
    summary = summarize_veto_gate_candidate(
        records,
        seeds=(1, 2),
        scenarios=("s",),
        risk_multiplier=0.0,
        min_context_trust=0.0,
        selection_rule={
            "minimum_retained_groups_per_city": 1,
            "minimum_retained_seed_fraction": 0.5,
            "maximum_equal_seed_scenario_mean_delta": 0.0,
            "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
            "maximum_worst_seed_mean_delta": 0.01,
            "maximum_harmful_group_fraction": 0.45,
        },
        bootstrap={"replicates": 100, "seed": 7},
    )

    assert summary["retained_group_count"] == 0
    assert summary["feasible"] is False
