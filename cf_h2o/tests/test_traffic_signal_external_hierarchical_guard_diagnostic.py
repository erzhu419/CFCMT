import numpy as np

from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (
    CONFORMAL_ONLY,
    HIERARCHICAL,
    PRESSURE_ONLY,
    _decision_summary,
    hierarchical_variant_rows,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
    hierarchical_guard_decision_v3,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.target_action_support import (
    HierarchicalTargetActionSupport,
    TargetActionSupport,
)


def _record(group: str, *, support: float, actual: float = -0.4) -> dict:
    return {
        "scenario": "network",
        "group_id": group,
        "learned_differs": True,
        "predicted_delta": -0.3,
        "uncertainty": 0.05,
        "context_trust": support,
        "local_action_support": support,
        "relative_rule_gap": 0.0,
        "actual_delta": actual,
        "candidate_features": {"total_veh": 4.0},
    }


def test_hierarchical_guard_intersects_pressure_and_conformal_support() -> None:
    pressure = [_record("a", support=0.9), _record("b", support=0.9)]
    conformal = [_record("a", support=0.9), _record("b", support=0.2)]
    rows = hierarchical_variant_rows(
        pressure_records=pressure,
        conformal_records=conformal,
        regularizer=PriorRegularizationConfig(
            enabled=True,
            blend_weight=2.0,
            risk_multiplier=0.0,
            min_context_trust=0.5,
        ),
        guard=ContrastGuardConfig(
            enabled=True,
            risk_multiplier=0.0,
            min_context_trust=0.5,
            margin=0.0,
            max_relative_rule_gap=0.0,
        ),
    )

    assert [row["selected"] for row in rows[PRESSURE_ONLY]] == [True, True]
    assert [row["selected"] for row in rows[CONFORMAL_ONLY]] == [True, False]
    assert [row["selected"] for row in rows[HIERARCHICAL]] == [True, False]


def test_disabled_conformal_guard_is_identity_filter() -> None:
    records = [_record("a", support=0.9)]
    rows = hierarchical_variant_rows(
        pressure_records=records,
        conformal_records=records,
        regularizer=PriorRegularizationConfig(
            enabled=True,
            blend_weight=2.0,
            risk_multiplier=0.0,
            min_context_trust=0.5,
        ),
        guard=ContrastGuardConfig(enabled=False),
    )

    assert rows[CONFORMAL_ONLY][0]["selected"] is False
    assert rows[HIERARCHICAL][0]["selected"] is True


def test_decision_summary_weights_scenarios_equally() -> None:
    rows = [
        {"scenario": "small", "selected": True, "actual_delta": -1.0, "raw_actual_delta": -1.0},
        *(
            {"scenario": "large", "selected": False, "actual_delta": 0.0, "raw_actual_delta": 1.0}
            for _ in range(9)
        ),
    ]

    summary = _decision_summary(rows)

    assert np.isclose(summary["mean_oof_delta_vs_pressure"], -0.5)
    assert summary["accepted_overrides"] == 1
    assert summary["accepted_harmful_actions"] == 0


def test_shared_hierarchical_kernel_requires_both_enabled_gates() -> None:
    regularizer = PriorRegularizationConfig(
        enabled=True,
        blend_weight=2.0,
        risk_multiplier=0.0,
        min_context_trust=0.5,
    )
    guard = ContrastGuardConfig(
        enabled=True,
        risk_multiplier=0.0,
        min_context_trust=0.5,
        margin=0.0,
        max_relative_rule_gap=0.0,
    )

    accepted = hierarchical_guard_decision_v3(
        learned_differs=True,
        predicted_delta=-0.3,
        uncertainty=0.1,
        relative_rule_gap=0.0,
        pressure_context_trust=0.9,
        conformal_context_trust=0.9,
        total_vehicles=3.0,
        regularizer=regularizer,
        guard=guard,
    )
    rejected = hierarchical_guard_decision_v3(
        learned_differs=True,
        predicted_delta=-0.3,
        uncertainty=0.1,
        relative_rule_gap=0.0,
        pressure_context_trust=0.9,
        conformal_context_trust=0.2,
        total_vehicles=3.0,
        regularizer=regularizer,
        guard=guard,
    )

    assert accepted["eligible"] is True
    assert rejected["eligible"] is False
    assert rejected["rejection"] == "conformal_support"


def test_hierarchical_support_preserves_component_scores() -> None:
    dataset = MechanismDataset(
        feature_names=("total_veh",),
        features=np.asarray([[2.0], [8.0]]),
        context_names=(),
        context=np.empty((2, 0)),
        priors={},
        targets={},
        domains=np.asarray(["target", "target"]),
        metadata={},
    )
    pressure = TargetActionSupport.fit(dataset, feature_names=("total_veh",))
    conformal = TargetActionSupport.fit_validated_records(
        [{"candidate_features": {"total_veh": 2.0}}],
        feature_names=("total_veh",),
    )
    support = HierarchicalTargetActionSupport(
        pressure_support=pressure,
        conformal_support=conformal,
        conformal_guard_enabled=True,
    )

    components = support.component_support(dataset)

    assert np.allclose(
        support.support(dataset),
        np.minimum(components["pressure"], components["conformal"]),
    )
    assert support.diagnostics()["combination"] == "minimum_component_support"
