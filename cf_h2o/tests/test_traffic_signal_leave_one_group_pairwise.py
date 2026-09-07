import pytest

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_leave_one_group_pairwise import (
    select_leave_one_group_pairwise,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)


def _rows(gain: float) -> list[dict]:
    return [
        {
            "group_id": f"x:seed{(2027, 3037, 4047)[index % 3]}:{index}:tls",
            "simulator_seed": (2027, 3037, 4047)[index % 3],
            "training_group_count": 59,
            "paired_gain": gain,
        }
        for index in range(60)
    ]


def test_logo_gate_selects_stable_pairwise_gain() -> None:
    decision = select_leave_one_group_pairwise(_rows(0.04))
    assert decision["selected_family"] == PAIRWISE_PREFERENCE_FAMILY
    assert decision["gate"]["passed"] is True


def test_logo_gate_falls_back_on_one_seed_regression() -> None:
    rows = _rows(0.04)
    for row in rows:
        if row["simulator_seed"] == 4047:
            row["paired_gain"] = -0.04
    decision = select_leave_one_group_pairwise(rows)
    assert decision["selected_family"] == GROUP_NORMALIZED_RIGID_FAMILY
    assert decision["gate"]["worst_seed_gain_at_least_minus_0_02"] is False


def test_logo_gate_requires_fifty_nine_training_groups() -> None:
    rows = _rows(0.04)
    rows[0]["training_group_count"] = 58
    with pytest.raises(ValueError, match="59 target groups"):
        select_leave_one_group_pairwise(rows)


def test_logo_gate_rejects_duplicate_group() -> None:
    rows = _rows(0.04)
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="exactly one LOGO"):
        select_leave_one_group_pairwise(rows)
