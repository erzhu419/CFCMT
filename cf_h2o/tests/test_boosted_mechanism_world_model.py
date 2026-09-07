import numpy as np

from cf_h2o.traffic_signal.boosted_mechanism_world_model import (
    BoostedCausalTransferWorldModel,
    BoostedFitConfig,
    BoostedMechanismWorldModel,
)
from cf_h2o.traffic_signal.mechanism_world_model import (
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
)


DEFINITIONS = (
    MechanismDefinition(
        name="cost",
        target_name="interval_cost",
        parent_variants={
            "minimal": ("state", "action"),
            "physical": ("state", "action", "interaction"),
        },
        clip_min=-20.0,
        clip_max=20.0,
    ),
)


def _grouped_dataset():
    features = []
    contexts = []
    targets = []
    domains = []
    groups = []
    references = []
    for domain_idx in range(4):
        for group_idx in range(12):
            state = (group_idx - 5.0) / 3.0
            for action in (0.0, 1.0):
                features.append([state, action, state * action])
                contexts.append([float(domain_idx), float(domain_idx % 2)])
                targets.append((action - float(state > 0.0)) ** 2 + 0.05 * domain_idx * action)
                domains.append(f"d{domain_idx}")
                groups.append(f"d{domain_idx}:g{group_idx}")
                references.append(action == 0.0)
    target = np.asarray(targets, dtype=float)
    return MechanismDataset(
        feature_names=("state", "action", "interaction"),
        features=np.asarray(features, dtype=float),
        context_names=("domain_scale", "domain_parity"),
        context=np.asarray(contexts, dtype=float),
        priors={"interval_cost": np.zeros_like(target)},
        targets={"interval_cost": target},
        domains=np.asarray(domains),
        metadata={"action_group_ids": groups, "is_reference": references},
    )


def test_boosted_causal_and_dense_models_share_output_contract():
    dataset = _grouped_dataset()
    config = BoostedFitConfig(max_iter=30, min_samples_leaf=5)
    for causal in (True, False):
        model = BoostedMechanismWorldModel(DEFINITIONS, causal=causal, config=config)
        diagnostics = model.fit(dataset)
        prediction = model.predict(dataset)["cost"]
        assert diagnostics["cost"]["estimator"] == "HistGradientBoostingRegressor"
        assert np.isfinite(prediction["mean"]).all()
        assert np.isfinite(prediction["uncertainty"]).all()
        assert prediction["mean"].shape == (dataset.size,)


def test_boosted_transfer_uses_cross_fitted_adaptation_targets():
    dataset = _grouped_dataset()
    model = BoostedCausalTransferWorldModel(
        DEFINITIONS,
        boosted_config=BoostedFitConfig(max_iter=20, min_samples_leaf=5),
        adaptation_config=MechanismFitConfig(
            latent_ranks=(0, 1),
            adaptation_shrinkages=(0.0, 1.0),
            action_ranking_targets=("interval_cost",),
        ),
    )
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["cost"]
    assert set(diagnostics["cost"]) == {"global", "adaptation", "enabled"}
    assert np.isfinite(prediction["mean"]).all()
    assert np.all(prediction["uncertainty"] >= 0.0)


def test_disabled_adaptation_is_exact_global_bypass():
    dataset = _grouped_dataset()
    zeros = np.zeros(dataset.size, dtype=float)
    no_residual = MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors={"interval_cost": zeros},
        targets={"interval_cost": zeros},
        domains=dataset.domains,
        metadata=dataset.metadata,
    )
    model = BoostedCausalTransferWorldModel(
        DEFINITIONS,
        boosted_config=BoostedFitConfig(max_iter=10, min_samples_leaf=5),
        adaptation_config=MechanismFitConfig(action_ranking_targets=("interval_cost",)),
    )
    model.fit(no_residual)
    assert not model.adaptation_model.components["cost"].enabled
    full = model.predict(no_residual)["cost"]
    global_only = model.global_model.predict(no_residual)["cost"]
    assert np.allclose(full["mean"], global_only["mean"])
    assert np.allclose(full["uncertainty"], global_only["uncertainty"])
