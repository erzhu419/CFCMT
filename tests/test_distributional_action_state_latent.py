from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.traffic_signal.distributional_action_state_latent import (
    DISTRIBUTIONAL_ACTION_STATE_PROTOCOL,
    DistributionalActionStateConfig,
    DistributionalActionStateLatent,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    COMPACT_STATE_FEATURES,
    StateConditionedLatentConfig,
)


def _dataset(queues: tuple[float, ...], *, tls: str = "tls0") -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    states = []
    tls_rows = []
    times = []
    for index, queue in enumerate(queues):
        for action in range(3):
            row = np.zeros(len(COMPACT_STATE_FEATURES), dtype=float)
            row[COMPACT_STATE_FEATURES.index("total_q")] = queue
            row[COMPACT_STATE_FEATURES.index("total_veh")] = queue + 1.0
            row[COMPACT_STATE_FEATURES.index("mean_speed")] = 12.0 - queue / 3.0
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
            groups.append(f"seed{index}:{tls}:{queue}")
            references.append(action == 0)
            states.append(f"phase{action}")
            tls_rows.append(tls)
            times.append(300.0)
    size = len(features)
    return MechanismDataset(
        feature_names=COMPACT_STATE_FEATURES,
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


def _model() -> DistributionalActionStateLatent:
    return DistributionalActionStateLatent(
        DistributionalActionStateConfig(
            state_model=StateConditionedLatentConfig(
                base=SpatiotemporalActionLatentConfig(
                    scope="tls_time_phase_pair",
                    shrinkage=2.0,
                    time_bin_sec=300,
                ),
                feature_set="compact_state",
                neighbor_count=3,
                local_shrinkage=1.0,
                distance_bandwidth=1.0,
            )
        )
    )


def test_distributional_probability_tracks_local_action_benefit() -> None:
    model = _model()
    diagnostics = model.fit(_dataset((1.0, 2.0, 3.0, 8.0, 9.0, 10.0)))
    validation = _dataset((2.5, 9.5))
    prediction = model.predict(validation)["control_cost"]
    groups = np.asarray(validation.metadata["action_group_ids"], dtype=str)
    states = np.asarray(validation.metadata["candidate_states"], dtype=str)
    phase_one_rows = [
        int(np.flatnonzero((groups == group) & (states == "phase1"))[0])
        for group in dict.fromkeys(groups.tolist())
    ]

    assert diagnostics["protocol"] == DISTRIBUTIONAL_ACTION_STATE_PROTOCOL
    assert diagnostics["calibration_mode"] == (
        "leave_self_out_local_probability_isotonic"
    )
    assert diagnostics["calibration_row_count"] > 0
    assert prediction["benefit_probability_raw"][phase_one_rows[1]] > prediction[
        "benefit_probability_raw"
    ][phase_one_rows[0]]
    assert prediction["benefit_probability_calibrated"][phase_one_rows[1]] > prediction[
        "benefit_probability_calibrated"
    ][phase_one_rows[0]]
    assert all(
        np.isfinite(np.asarray(values, dtype=float)).all()
        for values in prediction.values()
    )


def test_distributional_probability_falls_back_without_local_cell() -> None:
    model = _model()
    model.fit(_dataset((1.0, 2.0, 8.0, 9.0)))
    unseen = _dataset((5.0,), tls="tls_new")
    prediction = model.predict(unseen)["control_cost"]
    nonreference = ~np.asarray(unseen.metadata["is_reference"], dtype=bool)

    assert np.all(prediction["benefit_effective_support"][nonreference] == 0.0)
    assert np.all(prediction["context_trust"][nonreference] == 0.0)


def test_distributional_prior_requires_positive_shape() -> None:
    with pytest.raises(ValueError, match="alpha"):
        DistributionalActionStateConfig(
            state_model=_model().config.state_model,
            benefit_prior_alpha=0.0,
        )
