from __future__ import annotations

from cf_h2o.eval.traffic_signal_distributional_action_state_diagnostic import (
    select_distributional_gate,
    summarize_distributional_gate,
)


def _record(
    *,
    seed: int,
    group: str,
    delta: float,
    probability: float,
) -> dict[str, object]:
    return {
        "model_key": "dist_test",
        "model_family": "distributional_action_state_latent",
        "simulator_seed": seed,
        "group_id": group,
        "selected_differs": True,
        "predicted_delta": -0.2,
        "selected_uncertainty": 0.1,
        "selected_context_trust": 0.8,
        "selected_benefit_probability_raw": probability,
        "selected_benefit_probability_calibrated": probability,
        "actual_group_normalized_delta": delta,
    }


def _selection() -> dict[str, object]:
    return {
        "gate_candidates": [
            {
                "benefit_probability_kind": "calibrated",
                "minimum_benefit_probability": 0.6,
                "risk_multiplier": 0.0,
                "minimum_context_trust": 0.5,
            }
        ],
        "selection_rule": {
            "minimum_retained_groups_per_city": 2,
            "minimum_retained_seed_fraction": 1.0,
            "maximum_equal_seed_scenario_mean_delta": -0.01,
            "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
            "maximum_worst_seed_mean_delta": 0.0,
            "maximum_harmful_group_fraction": 0.42,
        },
        "bootstrap": {"replicates": 1000, "seed": 7},
    }


def test_probability_gate_excludes_low_probability_harmful_action() -> None:
    records = [
        _record(seed=1, group="g1", delta=-0.2, probability=0.8),
        _record(seed=2, group="g2", delta=-0.1, probability=0.7),
        _record(seed=1, group="g3", delta=0.5, probability=0.4),
    ]
    selection = _selection()

    summary = summarize_distributional_gate(
        records,
        seeds=(1, 2),
        candidate=selection["gate_candidates"][0],
        selection_rule=selection["selection_rule"],
        bootstrap=selection["bootstrap"],
    )

    assert summary["retained_group_count"] == 2
    assert summary["harmful_group_fraction"] == 0.0
    assert summary["mean_delta"] < 0.0
    assert summary["feasible"] is True


def test_distributional_selector_returns_only_feasible_candidate() -> None:
    records = [
        _record(seed=1, group="g1", delta=-0.2, probability=0.8),
        _record(seed=2, group="g2", delta=-0.1, probability=0.7),
    ]

    selected = select_distributional_gate(
        records,
        model_specs=(
            {
                "key": "dist_test",
                "family": "distributional_action_state_latent",
            },
        ),
        seeds=(1, 2),
        selection=_selection(),
    )

    assert selected["feasible_candidate_count"] == 1
    assert selected["selected"]["model_key"] == "dist_test"
