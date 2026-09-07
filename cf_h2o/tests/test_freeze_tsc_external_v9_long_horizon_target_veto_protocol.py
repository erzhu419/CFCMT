from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (
    CLOSED_LOOP_COOLDOWN_INTERVALS,
    COLLECTION_SEEDS,
    closed_loop_candidates,
    veto_gate_candidates,
)


def test_v65_veto_grid_is_unique_and_monotone() -> None:
    rows = veto_gate_candidates()

    assert len(rows) == 20
    assert len({row["key"] for row in rows}) == 20
    assert {row["risk_multiplier"] for row in rows} == {0.0, 0.25, 0.5, 1.0, 2.0}
    assert {row["min_context_trust"] for row in rows} == {0.0, 0.25, 0.5, 0.75}


def test_v65_collection_and_closed_loop_partitions_are_fixed() -> None:
    rows = closed_loop_candidates()

    assert COLLECTION_SEEDS == (80314, 88625, 27178, 77524, 63775, 50945)
    assert tuple(row["cooldown_intervals"] for row in rows) == (
        CLOSED_LOOP_COOLDOWN_INTERVALS
    )
    assert [row["key"] for row in rows] == [
        "cfcmt_target_veto_cd120",
        "cfcmt_target_veto_cd300",
        "cfcmt_target_veto_cd450",
    ]
    assert all(row["max_simultaneous_overrides"] == 1 for row in rows)
