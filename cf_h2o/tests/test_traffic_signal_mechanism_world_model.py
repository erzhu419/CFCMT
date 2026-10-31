import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import (
    DenseResidualWorldModel,
    FixedParentResidualWorldModel,
    InvariantMechanismWorldModel,
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
    PriorMechanismWorldModel,
    _grouped_action_regret,
    _leave_one_domain_score,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range


def _dataset(*, target_context=0.0, include_target=False):
    rng = np.random.default_rng(7)
    domains = []
    features = []
    contexts = []
    targets = []
    priors = []
    domain_values = [-1.0, 0.0, 1.0]
    if include_target:
        domain_values.append(target_context)
    for domain_idx, context_value in enumerate(domain_values):
        x = rng.normal(size=(90, 4))
        prior = 3.0 + 0.4 * x[:, 0]
        residual = 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.55 * context_value * x[:, 0]
        # Nuisance correlation changes sign across domains and should not be a
        # permitted parent of the invariant mechanism.
        x[:, 3] = (1.0 if domain_idx % 2 == 0 else -1.0) * residual + rng.normal(scale=0.15, size=x.shape[0])
        features.append(x)
        contexts.append(np.repeat([[context_value, context_value**2]], x.shape[0], axis=0))
        domains.extend([f"d{domain_idx}"] * x.shape[0])
        priors.append(prior)
        targets.append(np.maximum(prior + residual + rng.normal(scale=0.05, size=x.shape[0]), 0.0))
    return MechanismDataset(
        feature_names=("upstream_q", "downstream_q", "occupancy", "nuisance"),
        features=np.vstack(features),
        context_names=("network_scale", "network_scale_sq"),
        context=np.vstack(contexts),
        priors={"next_queue": np.concatenate(priors)},
        targets={"next_queue": np.concatenate(targets)},
        domains=np.asarray(domains),
    )


DEFINITION = MechanismDefinition(
    name="queue",
    target_name="next_queue",
    parent_variants={
        "minimal": ("upstream_q",),
        "mechanism": ("upstream_q", "downstream_q"),
    },
    clip_min=0.0,
)


def test_mechanism_model_selects_and_predicts_with_auditable_outputs():
    dataset = _dataset()
    model = InvariantMechanismWorldModel(
        [DEFINITION],
        MechanismFitConfig(latent_ranks=(0, 1), adaptation_shrinkages=(0.0, 0.5, 1.0)),
    )
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["queue"]

    assert diagnostics["queue"]["variant_name"] in {"prior_only", "minimal", "mechanism"}
    assert "nuisance" not in diagnostics["queue"]["parent_names"]
    assert prediction["mean"].shape == (dataset.size,)
    assert prediction["uncertainty"].shape == (dataset.size,)
    assert np.all(prediction["uncertainty"] >= 0.0)
    model_mae = np.mean(np.abs(prediction["mean"] - dataset.targets["next_queue"]))
    prior_mae = np.mean(np.abs(dataset.priors["next_queue"] - dataset.targets["next_queue"]))
    assert model_mae < 0.35 * prior_mae


def test_dense_baseline_uses_same_outputs_and_prediction_contract():
    dataset = _dataset()
    model = DenseResidualWorldModel([DEFINITION])
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["queue"]

    assert diagnostics["queue"]["feature_count"] == len(dataset.feature_names) + len(dataset.context_names)
    assert prediction["mean"].shape == (dataset.size,)
    assert np.allclose(prediction["latent_residual"], 0.0)


def test_prior_baseline_has_the_same_prediction_contract():
    dataset = _dataset()
    model = PriorMechanismWorldModel([DEFINITION])
    model.fit(dataset)
    prediction = model.predict(dataset)["queue"]

    assert np.allclose(prediction["mean"], dataset.priors["next_queue"])
    assert np.all(prediction["uncertainty"] > 0.0)


def test_fixed_parent_model_supports_a_single_target_domain():
    dataset = _dataset()
    target_only = dataset.subset(dataset.domains == "d0")
    model = FixedParentResidualWorldModel(
        [DEFINITION],
        parent_variant="mechanism",
    )
    diagnostics = model.fit(target_only)
    prediction = model.predict(target_only)["queue"]

    assert diagnostics["queue"]["domain_count"] == 1
    assert diagnostics["queue"]["parent_names"] == [
        "upstream_q",
        "downstream_q",
    ]
    assert diagnostics["queue"]["latent_rank"] == 0
    assert prediction["mean"].shape == (target_only.size,)
    model_mae = np.mean(
        np.abs(prediction["mean"] - target_only.targets["next_queue"])
    )
    prior_mae = np.mean(
        np.abs(
            target_only.priors["next_queue"]
            - target_only.targets["next_queue"]
        )
    )
    assert model_mae < prior_mae


def test_forbidden_identity_parent_is_rejected():
    definition = MechanismDefinition(
        name="bad",
        target_name="next_queue",
        parent_variants={"bad": ("tls_id",)},
    )
    with pytest.raises(ValueError):
        InvariantMechanismWorldModel([definition])


def test_two_domain_fit_validates_each_cross_domain_direction():
    dataset = _dataset()
    keep = np.isin(dataset.domains, ["d0", "d1"])
    two_domain = dataset.subset(keep)
    score = _leave_one_domain_score(
        two_domain,
        DEFINITION,
        variant_name="minimal",
        parent_names=("upstream_q",),
        latent_rank=0,
        adaptation_shrinkage=0.0,
        config=MechanismFitConfig(latent_ranks=(0,)),
    )

    assert set(score["domain_errors"]) == {"d0", "d1"}
    assert np.isfinite(score["score"])


def test_grouped_action_regret_matches_naive_stable_tie_semantics():
    groups = np.asarray(["b", "a", "b", "c", "a", "c", "b"])
    reference = np.asarray([True, True, False, False, False, True, False])
    actual = np.asarray([2.0, 4.0, 1.0, 7.0, 2.0, 3.0, 5.0])
    prediction = np.asarray([0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0])
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.zeros((groups.size, 1), dtype=float),
        context_names=("context",),
        context=np.zeros((groups.size, 1), dtype=float),
        priors={"next_queue": np.zeros(groups.size, dtype=float)},
        targets={"next_queue": actual},
        domains=np.asarray(["source"] * groups.size),
        metadata={
            "action_group_ids": groups,
            "is_reference": reference,
        },
    )
    naive = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        selected = int(rows[int(np.argmin(prediction[rows]))])
        values = actual[rows]
        naive.append((actual[selected] - np.min(values)) / action_group_range(values))

    assert np.isclose(
        _grouped_action_regret(dataset, prediction, "next_queue"),
        np.mean(naive),
        rtol=0.0,
        atol=1e-12,
    )
