from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_diagnostic import (
    ANALYSIS_PROTOCOL,
    DIAGNOSTIC_SPECS,
    POLICIES,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_diagnostic_aggregate import (
    _validate_deployment_audit,
)


def test_diagnostic_matrix_is_explicitly_post_hoc_and_uses_frozen_sources():
    assert POLICIES == (
        "selected_source_prior",
        "cfcmt_selected_spatial_only",
        "cfcmt_selected_direct",
        "rigid_anchor_direct",
        "simulator_only_direct",
        "h2oplus_style_dense_residual_direct",
    )
    assert ANALYSIS_PROTOCOL.endswith("deployment-mechanism-diagnostic-v1")
    assert RESULT_PROTOCOL.endswith("deployment-diagnostic-v1")
    assert DIAGNOSTIC_SPECS["cfcmt_selected_spatial_only"].source_policy == METHOD_POLICY
    assert DIAGNOSTIC_SPECS["cfcmt_selected_spatial_only"].coordination_mode == "spatial_only"
    assert DIAGNOSTIC_SPECS["cfcmt_selected_direct"].coordination_mode == "direct"
    for spec in DIAGNOSTIC_SPECS.values():
        audit = spec.to_dict()
        assert audit["analysis_status"] == (
            "post_hoc_explanatory_not_preregistered_confirmation"
        )
        assert audit["frozen_models_reused_without_refit"] is True
        assert audit["frozen_evaluation_scenarios_and_seeds_reused"] is True


def _audit_metrics(
    *, mode: str, proposed: int, accepted: int, cooldown: int, coordination: int
):
    return {
        "residual_deployment": {
            "mode": mode,
            "cooldown_intervals": 0,
            "coordination": "none" if mode == "direct" else "greedy_priority_independent_set",
            "conflict_graph_applied": mode != "direct",
        },
        "guard_audit": {
            "decisions": 20,
            "prior_agreements": 20 - proposed,
            "proposed_overrides": proposed,
            "accepted_overrides": accepted,
            "rejected_cooldown": cooldown,
            "rejected_coordination": coordination,
        },
    }


def test_direct_diagnostic_audit_requires_every_proposal_to_execute():
    valid = _validate_deployment_audit(
        policy="cfcmt_selected_direct",
        metrics=_audit_metrics(
            mode="direct", proposed=8, accepted=8, cooldown=0, coordination=0
        ),
    )
    assert valid["accepted_overrides"] == 8

    with pytest.raises(ValueError, match="direct execution invariant"):
        _validate_deployment_audit(
            policy="cfcmt_selected_direct",
            metrics=_audit_metrics(
                mode="direct", proposed=8, accepted=7, cooldown=0, coordination=1
            ),
        )


def test_spatial_only_audit_partitions_all_proposals_without_cooldown():
    valid = _validate_deployment_audit(
        policy="cfcmt_selected_spatial_only",
        metrics=_audit_metrics(
            mode="spatial_only",
            proposed=8,
            accepted=5,
            cooldown=0,
            coordination=3,
        ),
    )
    assert valid["rejected_coordination"] == 3

    with pytest.raises(ValueError, match="spatial-only execution invariant"):
        _validate_deployment_audit(
            policy="cfcmt_selected_spatial_only",
            metrics=_audit_metrics(
                mode="spatial_only",
                proposed=8,
                accepted=5,
                cooldown=1,
                coordination=2,
            ),
        )
