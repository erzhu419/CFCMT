from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.traffic_signal.hierarchical_action_state_residual import (
    HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL,
    HierarchicalActionStateResidual,
    HierarchicalActionStateResidualConfig,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_ACTION_FEATURES,
)


def _dataset(queues: tuple[float, ...], *, tls: str = "tls0") -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    states = []
    tls_rows = []
    times = []
    for group_index, queue in enumerate(queues):
        for action in range(3):
            row = np.zeros(len(STATE_ACTION_FEATURES), dtype=float)
            row[STATE_ACTION_FEATURES.index("total_q")] = queue
            row[STATE_ACTION_FEATURES.index("total_veh")] = queue + 2.0
            row[STATE_ACTION_FEATURES.index("mean_speed")] = 12.0 - queue / 3.0
            row[STATE_ACTION_FEATURES.index("delta_green_q")] = action * queue / 10.0
            row[STATE_ACTION_FEATURES.index("delta_service_pressure")] = (
                action * (queue - 5.0)
            )
            features.append(row)
            targets.append(
                0.0
                if action == 0
                else (
                    (-1.0 if queue >= 8.0 else 1.0)
                    if action == 1
                    else 0.5
                )
            )
            groups.append(f"seed{group_index}:{tls}:{queue}")
            references.append(action == 0)
            states.append(f"phase{action}")
            tls_rows.append(tls)
            times.append(300.0)
    size = len(features)
    return MechanismDataset(
        feature_names=STATE_ACTION_FEATURES,
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.zeros((size, 1), dtype=float),
        priors={"interval_cost": np.zeros(size, dtype=float)},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(["target"] * size),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": states,
            "row_tls": tls_rows,
            "row_times": times,
        },
    )


def _model() -> HierarchicalActionStateResidual:
    return HierarchicalActionStateResidual(
        HierarchicalActionStateResidualConfig(
            base=SpatiotemporalActionLatentConfig(
                scope="tls_time_phase_pair",
                shrinkage=2.0,
                time_bin_sec=300,
            ),
            feature_set="compact_state_action",
            residual_scale=1.0,
            max_iter=60,
            max_leaf_nodes=7,
            min_samples_leaf=2,
            l2_regularization=1.0,
        )
    )


def test_hierarchical_residual_learns_state_response_within_fixed_cell() -> None:
    model = _model()
    training = _dataset(
        tuple(float(value) for value in range(1, 13)) * 3
    )
    diagnostics = model.fit(training)
    validation = _dataset((2.5, 10.5))
    prediction = model.predict(validation)["control_cost"]
    groups = np.asarray(validation.metadata["action_group_ids"], dtype=str)
    states = np.asarray(validation.metadata["candidate_states"], dtype=str)
    selected = []
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        selected.append(str(states[int(rows[np.argmin(prediction["mean"][rows])])]))

    assert diagnostics["protocol"] == HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL
    assert diagnostics["uses_target_labels_at_fit"] is True
    assert diagnostics["uses_future_outcomes_at_prediction"] is False
    assert selected == ["phase0", "phase1"]


def test_hierarchical_residual_unseen_cell_falls_back_to_base() -> None:
    model = _model()
    model.fit(_dataset((1.0, 2.0, 8.0, 9.0, 10.0)))
    unseen = _dataset((6.0,), tls="tls_new")
    prediction = model.predict(unseen)["control_cost"]
    base = model.base.predict(unseen)["control_cost"]
    nonreference = ~np.asarray(unseen.metadata["is_reference"], dtype=bool)

    assert np.allclose(prediction["mean"][nonreference], base["mean"][nonreference])
    assert np.all(prediction["state_residual_known"][nonreference] == 0.0)


def test_hierarchical_residual_requires_frozen_causal_features() -> None:
    source = _dataset((1.0, 9.0))
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
