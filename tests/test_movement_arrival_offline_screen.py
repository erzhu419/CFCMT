import numpy as np

from cf_h2o.eval.traffic_signal_movement_arrival_offline_screen import (
    leave_one_seed_out_selection,
    score_arrival_rule_grid,
)


def _synthetic_rows(arrival_is_helpful: bool) -> list[dict[str, object]]:
    groups = []
    base = []
    arrival = []
    target = []
    for seed in (11, 22, 33):
        for group_index in range(2):
            group = f"tiny:seed{seed}:{group_index}:tls0"
            groups.extend([group, group])
            base.extend([2.0, 1.0])
            arrival.extend([0.0, 2.0])
            target.extend([2.0, 1.0 if arrival_is_helpful else 3.0])
    return score_arrival_rule_grid(
        base_scores=np.asarray(base),
        arrival_scores=np.asarray(arrival),
        target=np.asarray(target),
        groups=groups,
        coefficients=(0.0, 1.0),
    )


def test_seed_blocked_selector_chooses_helpful_arrival_rule() -> None:
    result = leave_one_seed_out_selection(
        _synthetic_rows(True),
        coefficients=(0.0, 1.0),
        minimum_training_improvement=0.002,
    )

    assert result["nonzero_selected_seed_count"] == 3
    assert result["changed_group_fraction"] == 1.0
    assert result["equal_seed_mean_normalized_delta"] < 0.0
    assert result["paired_seed_bootstrap"]["ci95"][1] < 0.0


def test_seed_blocked_selector_falls_back_when_arrival_rule_hurts() -> None:
    result = leave_one_seed_out_selection(
        _synthetic_rows(False),
        coefficients=(0.0, 1.0),
        minimum_training_improvement=0.002,
    )

    assert result["nonzero_selected_seed_count"] == 0
    assert result["changed_group_count"] == 0
    assert result["equal_seed_mean_normalized_delta"] == 0.0
