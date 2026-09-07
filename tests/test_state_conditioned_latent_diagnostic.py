from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    DISTRIBUTIONAL_MODEL_FAMILY,
    HIERARCHICAL_RESIDUAL_MODEL_FAMILY,
    MODEL_FAMILY,
    _build_model,
    _fit_state_conditioned_oof_fold,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.hierarchical_action_state_residual import (
    HierarchicalActionStateResidual,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    COMPACT_STATE_FEATURES,
    STATE_CONDITIONED_PROTOCOL,
)


def _contrast() -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    candidate_states = []
    row_tls = []
    row_times = []
    for seed in (11, 22, 33):
        for group_index, queue in enumerate((2.0, 10.0)):
            group = f"jinan:seed{seed}:tls0:g{group_index}"
            for action in range(3):
                row = np.zeros(len(COMPACT_STATE_FEATURES), dtype=float)
                row[COMPACT_STATE_FEATURES.index("total_q")] = queue
                row[COMPACT_STATE_FEATURES.index("total_veh")] = queue + 1.0
                features.append(row)
                targets.append(
                    0.0
                    if action == 0
                    else (-1.0 if action == 1 and queue > 5.0 else 1.0)
                )
                groups.append(group)
                references.append(action == 0)
                candidate_states.append(f"phase{action}")
                row_tls.append("tls0")
                row_times.append(300.0)
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
            "candidate_states": candidate_states,
            "row_tls": row_tls,
            "row_times": row_times,
        },
    )


def _spec() -> dict[str, object]:
    return {
        "key": "compact_test",
        "family": MODEL_FAMILY,
        "config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 2.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": "compact_state",
            "neighbor_count": 2,
            "local_shrinkage": 1.0,
            "distance_bandwidth": 1.0,
            "uncertainty_quantile": 0.9,
        },
    }


def test_fold_excludes_heldout_seed_and_preserves_action_contract() -> None:
    fold = _fit_state_conditioned_oof_fold(
        heldout_seed=33,
        contrast=_contrast(),
        model_specs=(_spec(),),
        scenario="jinan_3x4_real",
        expected_action_count=3,
    )

    assert fold["training_group_count"] == 4
    assert fold["validation_group_count"] == 2
    assert fold["training_validation_disjoint"] is True
    assert fold["heldout_absent_from_training"] is True
    result = fold["model_results"][0]
    assert result["fit_diagnostics"]["protocol"] == STATE_CONDITIONED_PROTOCOL
    assert len(result["records"]) == 2
    assert {row["heldout_seed"] for row in result["records"]} == {33}
    assert {row["action_count"] for row in result["records"]} == {3}


def test_model_builder_rejects_unfrozen_family() -> None:
    spec = _spec()
    spec["family"] = "dense_ranker"

    with pytest.raises(ValueError, match="unsupported"):
        _build_model(spec)


def test_fold_records_distributional_benefit_probability() -> None:
    state_config = dict(_spec()["config"])
    spec = {
        "key": "distributional_test",
        "family": DISTRIBUTIONAL_MODEL_FAMILY,
        "config": {
            "state_model": state_config,
            "benefit_prior_alpha": 1.0,
            "benefit_prior_beta": 1.0,
        },
    }

    fold = _fit_state_conditioned_oof_fold(
        heldout_seed=33,
        contrast=_contrast(),
        model_specs=(spec,),
        scenario="jinan_3x4_real",
        expected_action_count=3,
    )

    records = fold["model_results"][0]["records"]
    assert len(records) == 2
    assert all(
        0.0 <= row["selected_benefit_probability_calibrated"] <= 1.0
        for row in records
    )
    assert all(row["selected_benefit_effective_support"] > 0.0 for row in records)


def test_model_builder_constructs_hierarchical_residual_family() -> None:
    model = _build_model(
        {
            "key": "hierarchical_test",
            "family": HIERARCHICAL_RESIDUAL_MODEL_FAMILY,
            "config": {
                "base": {
                    "scope": "tls_time_phase_pair",
                    "shrinkage": 20.0,
                    "time_bin_sec": 300,
                    "uncertainty_quantile": 0.9,
                },
                "feature_set": "compact_state_action",
                "residual_scale": 0.5,
            },
        }
    )

    assert isinstance(model, HierarchicalActionStateResidual)
