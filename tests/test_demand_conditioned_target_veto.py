from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (
    combined_feature_matrix,
    combined_feature_names,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
)
from cf_h2o.traffic_signal.demand_conditioned_target_veto import (
    DemandConditionedTargetVeto,
    DemandConditionedVetoConfig,
    VETO_PROTOCOL,
)
from cf_h2o.traffic_signal.demand_schedule_context import (
    DEMAND_SCHEDULE_FEATURE_NAMES,
    DEMAND_SCHEDULE_PROTOCOL,
    DemandScheduleProfile,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


def _profile() -> DemandScheduleProfile:
    return DemandScheduleProfile(
        protocol=DEMAND_SCHEDULE_PROTOCOL,
        horizon_sec=3600,
        tls_count=1,
        scheduled_vehicle_count=10.0,
        input_sha256="a" * 64,
        feature_names=DEMAND_SCHEDULE_FEATURE_NAMES,
        features=tuple(0.1 * index for index in range(len(DEMAND_SCHEDULE_FEATURE_NAMES))),
    )


def _dataset() -> MechanismDataset:
    features = np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=float)
    return MechanismDataset(
        feature_names=("x", "unused"),
        features=features,
        context_names=("context",),
        context=np.zeros((2, 1)),
        priors={"cost": np.zeros(2)},
        targets={"cost": np.zeros(2)},
        domains=np.asarray(["target", "target"]),
        metadata={
            "action_group_ids": ["g", "g"],
            "is_reference": [True, False],
        },
    )


def test_demand_conditioned_veto_only_accepts_existing_candidate() -> None:
    profile = _profile()
    names = ("x", *profile.feature_names)
    model = ProposalConditionalRegressor(feature_names=names)
    demand = np.repeat(profile.vector()[None, :], 4, axis=0)
    x = np.concatenate(
        [np.asarray([[0.0], [1.0], [2.0], [3.0]]), demand], axis=1
    )
    model.fit(x, np.asarray([0.2, -0.2, -0.4, -0.6]))
    veto = DemandConditionedTargetVeto(
        model=model,
        config=DemandConditionedVetoConfig(
            enabled=True, acceptance_quantile=0.5, score_threshold=1.0
        ),
        candidate_feature_names=("x",),
        demand_profile=profile,
    )

    result = veto.evaluate(_dataset(), candidate_index=1, reference_index=0)

    assert result["protocol"] == VETO_PROTOCOL
    assert result["eligible"] is True
    assert result["demand_profile_input_sha256"] == "a" * 64
    assert veto.diagnostics()["action_originator"] is False


def test_demand_conditioned_veto_rejects_cross_group_indices() -> None:
    profile = _profile()
    model = ProposalConditionalRegressor(
        feature_names=("x", *profile.feature_names)
    )
    demand = np.repeat(profile.vector()[None, :], 2, axis=0)
    model.fit(np.concatenate([np.asarray([[0.0], [1.0]]), demand], axis=1), np.asarray([0.0, 1.0]))
    veto = DemandConditionedTargetVeto(
        model=model,
        config=DemandConditionedVetoConfig(
            enabled=False, acceptance_quantile=0.5, score_threshold=0.0
        ),
        candidate_feature_names=("x",),
        demand_profile=profile,
    )
    dataset = _dataset()
    dataset.metadata["action_group_ids"] = ["left", "right"]

    try:
        veto.evaluate(dataset, candidate_index=1, reference_index=0)
    except ValueError as error:
        assert "candidate/reference" in str(error)
    else:
        raise AssertionError("cross-group candidate was not rejected")


def test_training_matrix_uses_profile_without_scenario_id() -> None:
    profile = _profile()
    record = {
        "scenario": "anonymous_target",
        "candidate_features": {
            name: float(index) for index, name in enumerate(TARGET_SUPPORT_FEATURES)
        },
    }

    matrix = combined_feature_matrix(
        [record], profiles={"anonymous_target": profile}
    )

    assert matrix.shape == (1, 25)
    assert "scenario" not in combined_feature_names()
    np.testing.assert_allclose(matrix[0, 16:], profile.vector())
