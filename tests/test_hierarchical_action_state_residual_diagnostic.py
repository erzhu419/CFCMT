from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    HIERARCHICAL_RESIDUAL_MODEL_FAMILY,
    _fit_state_conditioned_oof_fold,
)
from cf_h2o.traffic_signal.hierarchical_action_state_residual import (
    HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_ACTION_FEATURES,
)


def _contrast() -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    states = []
    tls_rows = []
    times = []
    for seed in (11, 22, 33):
        for group_index, queue in enumerate((2.0, 10.0)):
            group = f"jinan:seed{seed}:tls0:g{group_index}"
            for action in range(3):
                row = np.zeros(len(STATE_ACTION_FEATURES), dtype=float)
                row[STATE_ACTION_FEATURES.index("total_q")] = queue
                row[STATE_ACTION_FEATURES.index("total_veh")] = queue + 1.0
                row[STATE_ACTION_FEATURES.index("delta_green_q")] = action * queue
                row[STATE_ACTION_FEATURES.index("delta_service_pressure")] = (
                    action * (queue - 5.0)
                )
                features.append(row)
                targets.append(
                    0.0
                    if action == 0
                    else (-1.0 if action == 1 and queue > 5.0 else 1.0)
                )
                groups.append(group)
                references.append(action == 0)
                states.append(f"phase{action}")
                tls_rows.append("tls0")
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


def test_hierarchical_residual_runs_through_seed_blocked_fold() -> None:
    spec = {
        "key": "hier_test",
        "family": HIERARCHICAL_RESIDUAL_MODEL_FAMILY,
        "config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 2.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": "compact_state_action",
            "residual_scale": 0.5,
            "max_iter": 20,
            "max_leaf_nodes": 5,
            "min_samples_leaf": 2,
            "l2_regularization": 1.0,
        },
    }

    fold = _fit_state_conditioned_oof_fold(
        heldout_seed=33,
        contrast=_contrast(),
        model_specs=(spec,),
        scenario="jinan_3x4_real",
        expected_action_count=3,
    )

    result = fold["model_results"][0]
    assert fold["heldout_absent_from_training"] is True
    assert fold["validation_group_count"] == 2
    assert result["fit_diagnostics"]["protocol"] == (
        HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL
    )
    assert len(result["records"]) == 2
