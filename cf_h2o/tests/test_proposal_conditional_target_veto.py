import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
    ProposalConditionalRegressorConfig,
    ProposalConditionalTargetVeto,
    ProposalConditionalVetoConfig,
)


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("total_q", "delta_service_pressure"),
        features=np.asarray([[10.0, 0.0], [10.0, -2.0], [10.0, 2.0]]),
        context_names=("demand",),
        context=np.ones((3, 1)),
        priors={"interval_cost": np.zeros(3)},
        targets={"interval_cost": np.asarray([0.0, -1.0, 1.0])},
        domains=np.asarray(["city"] * 3),
        metadata={
            "action_group_ids": ["g", "g", "g"],
            "is_reference": [True, False, False],
        },
    )


def test_proposal_conditional_veto_never_redirects_and_applies_threshold() -> None:
    dataset = _dataset()
    model = ProposalConditionalRegressor(
        feature_names=("total_q", "delta_service_pressure"),
        config=ProposalConditionalRegressorConfig(
            max_iter=40,
            max_leaf_nodes=3,
            min_samples_leaf=1,
            l2_regularization=0.0,
        ),
    )
    model.fit(dataset.features, np.asarray([0.0, -1.0, 1.0]))
    veto = ProposalConditionalTargetVeto(
        model=model,
        config=ProposalConditionalVetoConfig(
            enabled=True,
            acceptance_quantile=0.125,
            score_threshold=0.0,
        ),
    )

    accepted = veto.evaluate(dataset, candidate_index=1, reference_index=0)
    rejected = veto.evaluate(dataset, candidate_index=2, reference_index=0)

    assert accepted["eligible"] is True
    assert rejected["eligible"] is False
    assert rejected["rejection"] == "proposal_conditional_score"
    assert veto.diagnostics()["action_originator"] is False


def test_disabled_proposal_conditional_veto_always_abstains() -> None:
    dataset = _dataset()
    model = ProposalConditionalRegressor(feature_names=("total_q",))
    model.fit(dataset.features[:, :1], np.asarray([0.0, -1.0, 1.0]))
    veto = ProposalConditionalTargetVeto(
        model=model,
        config=ProposalConditionalVetoConfig(
            enabled=False,
            acceptance_quantile=0.125,
            score_threshold=0.0,
        ),
    )

    result = veto.evaluate(dataset, candidate_index=1, reference_index=0)

    assert result["eligible"] is False
    assert result["rejection"] == "disabled"
