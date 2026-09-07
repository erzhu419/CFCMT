import pytest

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    select_target_pairwise,
    summarize_deployment_gate,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import TARGETS


def _calibration(gain: float, count: int = 24) -> list[dict]:
    return [
        {
            "group_id": f"g{index}",
            "snapshot_time_sec": float(index),
            "paired_gain": gain,
        }
        for index in range(count)
    ]


def test_target_gate_selects_pairwise_on_stable_gain() -> None:
    decision = select_target_pairwise(_calibration(0.04))
    assert decision["selected_family"] == PAIRWISE_PREFERENCE_FAMILY
    assert decision["gate"]["passed"] is True


def test_target_gate_falls_back_on_temporal_instability() -> None:
    rows = _calibration(0.08)
    for row in rows[-6:]:
        row["paired_gain"] = -0.08
    decision = select_target_pairwise(rows)
    assert decision["selected_family"] == GROUP_NORMALIZED_RIGID_FAMILY
    assert decision["gate"][
        "worst_temporal_block_gain_at_least_minus_0_05"
    ] is False


def test_target_gate_rejects_duplicate_groups() -> None:
    rows = _calibration(0.04)
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="unique"):
        select_target_pairwise(rows)


def test_deployment_summary_passes_broad_safe_selection() -> None:
    decisions = {
        target: select_target_pairwise(_calibration(0.04)) for target in TARGETS
    }
    regrets = {
        GROUP_NORMALIZED_RIGID_FAMILY: {target: 0.30 for target in TARGETS},
        PAIRWISE_PREFERENCE_FAMILY: {target: 0.24 for target in TARGETS},
    }
    result = summarize_deployment_gate(
        decisions=decisions,
        evaluation_regrets=regrets,
    )
    assert result["passed"] is True
    assert result["selected_pairwise_target_count"] == 6


def test_deployment_summary_rejects_unjustified_pairwise_selection() -> None:
    decisions = {
        target: select_target_pairwise(_calibration(-0.04)) for target in TARGETS
    }
    decisions[TARGETS[0]]["selected_family"] = PAIRWISE_PREFERENCE_FAMILY
    regrets = {
        GROUP_NORMALIZED_RIGID_FAMILY: {target: 0.30 for target in TARGETS},
        PAIRWISE_PREFERENCE_FAMILY: {target: 0.24 for target in TARGETS},
    }
    with pytest.raises(ValueError, match="without passing gate"):
        summarize_deployment_gate(
            decisions=decisions,
            evaluation_regrets=regrets,
        )
