from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    TLS_TIME_PHASE_PAIR_SCOPE,
    SpatiotemporalActionLatentConfig,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    COMPACT_STATE_FEATURES,
    StateConditionedLatentConfig,
    StateConditionedSpatiotemporalActionLatent,
)


def _dataset(*, queues: tuple[float, ...], tls: str = "tls0") -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    candidate_states = []
    row_tls = []
    row_times = []
    for group_index, queue in enumerate(queues):
        for action in range(3):
            row = np.zeros(len(COMPACT_STATE_FEATURES), dtype=float)
            row[COMPACT_STATE_FEATURES.index("total_q")] = queue
            row[COMPACT_STATE_FEATURES.index("total_veh")] = queue + 2.0
            row[COMPACT_STATE_FEATURES.index("mean_speed")] = 12.0 - queue / 2.0
            features.append(row)
            if action == 0:
                target = 0.0
            elif action == 1:
                target = -1.0 if queue >= 8.0 else 1.0
            else:
                target = 0.5
            targets.append(target)
            groups.append(f"seed{group_index}:{tls}:{queue}")
            references.append(action == 0)
            candidate_states.append(f"phase{action}")
            row_tls.append(tls)
            row_times.append(300.0)
    return MechanismDataset(
        feature_names=COMPACT_STATE_FEATURES,
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={"interval_cost": np.zeros(len(features), dtype=float)},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(["target"] * len(features)),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": candidate_states,
            "row_tls": row_tls,
            "row_times": row_times,
        },
    )


def _model() -> StateConditionedSpatiotemporalActionLatent:
    return StateConditionedSpatiotemporalActionLatent(
        StateConditionedLatentConfig(
            base=SpatiotemporalActionLatentConfig(
                scope=TLS_TIME_PHASE_PAIR_SCOPE,
                shrinkage=2.0,
                time_bin_sec=300,
            ),
            feature_set="compact_state",
            neighbor_count=2,
            local_shrinkage=0.0,
            distance_bandwidth=1.0,
        )
    )


def test_local_state_changes_action_inside_same_spatiotemporal_cell() -> None:
    model = _model()
    diagnostics = model.fit(_dataset(queues=(1.0, 2.0, 10.0, 11.0)))
    validation = _dataset(queues=(1.5, 10.5))
    prediction = model.predict(validation)["control_cost"]
    groups = np.asarray(validation.metadata["action_group_ids"])
    selected = []
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        selected.append(
            validation.metadata["candidate_states"][
                int(rows[np.argmin(prediction["mean"][rows])])
            ]
        )

    assert diagnostics["uses_target_labels_at_fit"] is True
    assert diagnostics["uses_future_outcomes_at_prediction"] is False
    assert diagnostics["calibration_mode"] == "leave_self_out_within_cell"
    assert diagnostics["training_nonreference_mae"] > 0.0
    assert selected == ["phase0", "phase1"]
    assert np.all(prediction["latent_level"][~np.asarray(validation.metadata["is_reference"])] == 4)


def test_unseen_tls_falls_back_to_base_latent() -> None:
    model = _model()
    model.fit(_dataset(queues=(1.0, 2.0, 10.0, 11.0)))
    unseen = _dataset(queues=(10.5,), tls="tls_new")
    prediction = model.predict(unseen)["control_cost"]
    nonreference = ~np.asarray(unseen.metadata["is_reference"], dtype=bool)

    assert np.all(prediction["latent_level"][nonreference] < 4.0)
    assert np.isfinite(prediction["mean"]).all()


def test_state_conditioned_latent_uses_explicit_target() -> None:
    source = _dataset(queues=(1.0, 2.0, 10.0, 11.0))
    target_name = "prefix_mean_cost_60s"
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
    model = StateConditionedSpatiotemporalActionLatent(
        _model().config,
        target_name=target_name,
    )

    diagnostics = model.fit(dataset)

    assert diagnostics["target_name"] == target_name
    assert model.diagnostics()["target_name"] == target_name


def test_state_conditioned_latent_requires_deterministic_features() -> None:
    source = _dataset(queues=(1.0, 10.0))
    broken = MechanismDataset(
        feature_names=source.feature_names[:-1],
        features=source.features[:, :-1],
        context_names=source.context_names,
        context=source.context,
        priors=source.priors,
        targets=source.targets,
        domains=source.domains,
        metadata=source.metadata,
    )

    with pytest.raises(KeyError, match="lacks features"):
        _model().fit(broken)
