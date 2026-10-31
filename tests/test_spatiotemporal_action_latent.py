from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (
    _fit_latent_oof_fold,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    PHASE_PAIR_SCOPE,
    TLS_TIME_PHASE_PAIR_SCOPE,
    SpatiotemporalActionLatentConfig,
    TargetSpatiotemporalActionLatent,
)


def _dataset(*, seeds: tuple[int, ...], tls_values: tuple[str, ...] = ("A", "B")) -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    candidate_states = []
    row_tls = []
    row_times = []
    contexts = []
    domains = []
    for seed in seeds:
        for tls in tls_values:
            for time_sec in (100.0, 700.0):
                group = f"test:seed{seed}:{int(time_sec)}:{tls}"
                for action in range(3):
                    features.append([float(action)])
                    if action == 0:
                        value = 0.0
                    elif action == 1:
                        value = -2.0 if tls == "A" and time_sec < 600.0 else 1.0
                    else:
                        value = -2.0 if tls == "B" and time_sec >= 600.0 else 1.0
                    targets.append(value)
                    groups.append(group)
                    references.append(action == 0)
                    candidate_states.append(f"phase{action}")
                    row_tls.append(tls)
                    row_times.append(time_sec)
                    contexts.append([0.0])
                    domains.append("target")
    return MechanismDataset(
        feature_names=("action",),
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.asarray(contexts, dtype=float),
        priors={"interval_cost": np.zeros(len(features), dtype=float)},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(domains),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": candidate_states,
            "row_tls": row_tls,
            "row_times": row_times,
        },
    )


def test_tls_time_latent_generalizes_across_heldout_seed() -> None:
    model = TargetSpatiotemporalActionLatent(
        SpatiotemporalActionLatentConfig(
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            time_bin_sec=600,
            shrinkage=0.0,
        )
    )
    diagnostics = model.fit(_dataset(seeds=(11, 22)))

    validation = _dataset(seeds=(33,))
    prediction = model.predict(validation)["control_cost"]
    groups = np.asarray(validation.metadata["action_group_ids"])
    selected_states = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        selected = int(rows[np.argmin(prediction["mean"][rows])])
        selected_states.append(validation.metadata["candidate_states"][selected])

    assert diagnostics["uses_target_labels_at_fit"] is True
    assert diagnostics["uses_future_outcomes_at_prediction"] is False
    assert selected_states.count("phase1") == 1
    assert selected_states.count("phase2") == 1
    assert selected_states.count("phase0") == 2
    assert np.allclose(
        prediction["mean"][np.asarray(validation.metadata["is_reference"])], 0.0
    )


def test_unseen_tls_falls_back_to_phase_pair_prior() -> None:
    model = TargetSpatiotemporalActionLatent(
        SpatiotemporalActionLatentConfig(
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            time_bin_sec=600,
            shrinkage=10.0,
        )
    )
    model.fit(_dataset(seeds=(11, 22)))

    unseen = _dataset(seeds=(33,), tls_values=("C",))
    prediction = model.predict(unseen)["control_cost"]
    candidate = ~np.asarray(unseen.metadata["is_reference"], dtype=bool)

    assert np.all(prediction["latent_level"][candidate] == 1.0)
    assert np.all(prediction["context_trust"][candidate] == 1.0)
    assert np.isfinite(prediction["mean"]).all()


def test_latent_uses_explicit_multihorizon_target() -> None:
    source = _dataset(seeds=(11, 22))
    target_name = "prefix_mean_cost_30s"
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=source.features,
        context_names=source.context_names,
        context=source.context,
        priors={**source.priors, target_name: np.zeros(source.size)},
        targets={
            **source.targets,
            target_name: -np.asarray(source.targets["interval_cost"], dtype=float),
        },
        domains=source.domains,
        metadata=source.metadata,
    )
    model = TargetSpatiotemporalActionLatent(
        SpatiotemporalActionLatentConfig(
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            time_bin_sec=600,
            shrinkage=0.0,
        ),
        target_name=target_name,
    )

    diagnostics = model.fit(dataset)

    assert diagnostics["target_name"] == target_name
    assert model.diagnostics()["target_name"] == target_name


def test_phase_pair_config_rejects_time_bin() -> None:
    with pytest.raises(ValueError, match="only valid"):
        SpatiotemporalActionLatentConfig(
            scope=PHASE_PAIR_SCOPE,
            time_bin_sec=600,
        )


def test_latent_rejects_missing_online_action_metadata() -> None:
    source = _dataset(seeds=(11,))
    broken = MechanismDataset(
        feature_names=source.feature_names,
        features=source.features,
        context_names=source.context_names,
        context=source.context,
        priors=source.priors,
        targets=source.targets,
        domains=source.domains,
        metadata={
            key: value
            for key, value in source.metadata.items()
            if key != "row_tls"
        },
    )
    model = TargetSpatiotemporalActionLatent(
        SpatiotemporalActionLatentConfig(scope=PHASE_PAIR_SCOPE)
    )

    with pytest.raises(ValueError, match="row-aligned action metadata"):
        model.fit(broken)


def test_latent_oof_fold_keeps_heldout_seed_out_of_fit() -> None:
    fold = _fit_latent_oof_fold(
        heldout_seed=33,
        contrast=_dataset(seeds=(11, 22, 33)),
        model_specs=[
            {
                "key": "tls_time",
                "family": "target_spatiotemporal_action_latent",
                "config": {
                    "scope": TLS_TIME_PHASE_PAIR_SCOPE,
                    "shrinkage": 0.0,
                    "time_bin_sec": 600,
                    "uncertainty_quantile": 0.9,
                },
            }
        ],
        scenario="test",
        expected_action_count=3,
    )

    assert fold["heldout_absent_from_training"] is True
    assert fold["training_validation_disjoint"] is True
    assert fold["training_group_count"] == 8
    assert fold["validation_group_count"] == 4
    records = fold["model_results"][0]["records"]
    assert len(records) == 4
    assert {row["simulator_seed"] for row in records} == {33}
    assert {row["selected_latent_level"] for row in records} == {3}
    assert all(row["selected_support_count"] > 0 for row in records)
