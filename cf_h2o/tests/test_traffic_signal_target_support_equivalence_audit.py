from cf_h2o.eval.traffic_signal_target_support_equivalence_audit import (
    _compare_records,
)


def _record(group: str, support: float) -> dict:
    return {
        "group_id": group,
        "scenario": "s",
        "learned_differs": True,
        "predicted_delta": -0.1,
        "uncertainty": 0.01,
        "relative_rule_gap": 0.02,
        "actual_delta": -0.03,
        "local_action_support": support,
        "context_trust": min(0.8, support),
    }


def test_equivalence_comparison_accepts_roundoff_only() -> None:
    result = _compare_records(
        [_record("g1", 0.6), _record("g2", 0.7)],
        [_record("g2", 0.7 + 1e-15), _record("g1", 0.6 - 1e-15)],
    )

    assert result["passed"] is True
    assert result["action_group_count"] == 2


def test_equivalence_comparison_rejects_gate_relevant_drift() -> None:
    result = _compare_records([_record("g1", 0.6)], [_record("g1", 0.59)])

    assert result["passed"] is False
