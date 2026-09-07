import pytest

from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import (
    assign_seed_stratified_folds,
    select_cross_fitted_pairwise,
)
from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)


def _records() -> list[dict]:
    counts = {2027: 18, 3037: 18, 4047: 24}
    rows = []
    for seed, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "group_id": f"x:seed{seed}:{index}:tls{index % 3}",
                    "simulator_seed": seed,
                    "snapshot_time_sec": float(index * 10),
                    "tls_id": f"tls{index % 3}",
                }
            )
    return rows


def _oof(gain: float) -> list[dict]:
    fold = assign_seed_stratified_folds(_records())
    by_id = {row["group_id"]: row for row in _records()}
    return [
        {
            **by_id[group_id],
            "fold_index": fold_index,
            "paired_gain": gain,
        }
        for group_id, fold_index in sorted(fold["assignments"].items())
    ]


def test_seed_stratified_folds_are_exactly_balanced_and_deterministic() -> None:
    first = assign_seed_stratified_folds(_records())
    second = assign_seed_stratified_folds(list(reversed(_records())))
    assert first["assignments"] == second["assignments"]
    assert first["fold_group_counts"] == [12, 12, 12, 12, 12]


def test_seed_stratified_folds_support_b100_protocol() -> None:
    records = []
    for seed in (2027, 3037):
        for index in range(50):
            records.append(
                {
                    "group_id": f"x:seed{seed}:{index}:tls{index % 4}",
                    "simulator_seed": seed,
                    "snapshot_time_sec": float(index * 10),
                    "tls_id": f"tls{index % 4}",
                }
            )
    result = assign_seed_stratified_folds(
        records,
        expected_group_count=100,
        fold_count=5,
    )

    assert result["protocol"].endswith("equal-fold-v2")
    assert result["fold_group_counts"] == [20, 20, 20, 20, 20]
    assert result["seed_fold_counts"] == {
        2027: [10, 10, 10, 10, 10],
        3037: [10, 10, 10, 10, 10],
    }


def test_cross_fitted_gate_selects_stable_pairwise_gain() -> None:
    decision = select_cross_fitted_pairwise(_oof(0.04))
    assert decision["selected_family"] == PAIRWISE_PREFERENCE_FAMILY
    assert decision["gate"]["passed"] is True


def test_cross_fitted_gate_falls_back_on_seed_regression() -> None:
    rows = _oof(0.04)
    for row in rows:
        if row["simulator_seed"] == 4047:
            row["paired_gain"] = -0.04
    decision = select_cross_fitted_pairwise(rows)
    assert decision["selected_family"] == GROUP_NORMALIZED_RIGID_FAMILY
    assert decision["gate"]["worst_seed_gain_at_least_minus_0_02"] is False


def test_cross_fitted_gate_rejects_duplicate_oof_group() -> None:
    rows = _oof(0.04)
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="exactly one OOF"):
        select_cross_fitted_pairwise(rows)
