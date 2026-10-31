import numpy as np

from cf_h2o.traffic_signal.action_ranker import PairwisePreferenceConfig
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.source_gated_pairwise import (
    SourceGatedPairwiseActionRegressor,
    SourceGatedPairwiseConfig,
)


def _dataset() -> MechanismDataset:
    rows = []
    targets = []
    domains = []
    groups = []
    references = []
    contexts = []
    for domain_index, domain in enumerate(("a", "b", "c", "d")):
        for group_index in range(8):
            state = float(group_index + domain_index)
            for action_index, action in enumerate((0.0, 1.0, 2.0)):
                rows.append([state, action])
                targets.append((action - 1.5) ** 2 + 0.01 * state)
                domains.append(domain)
                groups.append(f"{domain}:{group_index}")
                references.append(action_index == 0)
                contexts.append([float(domain_index)])
    return MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray(rows, dtype=float),
        context_names=("demand",),
        context=np.asarray(contexts, dtype=float),
        priors={"interval_cost": np.zeros(len(rows))},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(domains),
        metadata={"action_group_ids": groups, "is_reference": references},
    )


def test_source_gated_pairwise_uses_complete_source_city_oof_and_zeroes_reference():
    dataset = _dataset()
    model = SourceGatedPairwiseActionRegressor(
        config=SourceGatedPairwiseConfig(
            minimum_disagreement_groups=1,
            fit_workers=2,
            pairwise_config=PairwisePreferenceConfig(
                max_iter=20,
                max_leaf_nodes=7,
                min_samples_leaf=2,
                l2_regularization=1.0,
            ),
        )
    )

    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]

    assert diagnostics["source"]["source_domain_count"] == 4
    assert diagnostics["source"]["source_city_oof_fold_count"] == 4
    assert diagnostics["source"]["target_labels_used"] is False
    reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    assert np.allclose(prediction["mean"][reference], 0.0)
    assert np.allclose(prediction["uncertainty"][reference], 0.0)
