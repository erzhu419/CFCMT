from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    _argmin_with_reference_tie_break,
    _fit_oof_fold,
    _row_subset,
    select_ranker_gate,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset() -> MechanismDataset:
    features = []
    targets = []
    domains = []
    groups = []
    references = []
    states = []
    contexts = []
    for seed in (11, 22, 33):
        for group_index in range(12):
            total_q = float(group_index % 4)
            group = f"jinan_3x4_real:seed{seed}:tls0:t{group_index}"
            for action, green_q in enumerate((0.0, 1.0, 2.0)):
                features.append([total_q, green_q])
                targets.append((2.0 - green_q) ** 2 + 0.01 * total_q)
                domains.append("jinan")
                groups.append(group)
                references.append(action == 0)
                states.append(f"phase{action}")
                contexts.append([float(seed % 10), total_q])
    return MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray(features, dtype=float),
        context_names=("demand", "queue"),
        context=np.asarray(contexts, dtype=float),
        priors={"interval_cost": np.zeros(len(features), dtype=float)},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(domains),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": states,
            "row_times": list(range(len(features))),
            "scalar_metadata": "kept",
        },
    )


def _model_specs() -> list[dict[str, object]]:
    return [
        {
            "key": "pairwise",
            "family": "antisymmetric_pairwise",
            "config": {
                "learning_rate": 0.1,
                "max_iter": 30,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "l2_regularization": 1.0,
                "uncertainty_quantile": 0.9,
                "random_state": 7,
            },
            "state_feature_names": ["total_q"],
            "action_feature_names": ["green_q"],
        },
        {
            "key": "advantage",
            "family": "group_normalized_advantage",
            "config": {
                "learning_rate": 0.1,
                "max_iter": 30,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "l2_regularization": 1.0,
                "target_clip": 5.0,
                "uncertainty_quantile": 0.9,
                "random_state": 7,
                "candidate_only": False,
                "balance_candidate_signs": False,
                "max_sign_weight_multiplier": 4.0,
                "target_normalization_protocol": "group_action_range_v1",
            },
            "causal_feature_names": ["total_q", "green_q"],
        },
    ]


def test_row_subset_keeps_all_row_aligned_metadata_in_sync() -> None:
    dataset = _dataset()
    mask = np.asarray(
        ["seed11:" in str(group) for group in dataset.metadata["action_group_ids"]]
    )

    subset = _row_subset(dataset, mask)

    assert subset.size == 36
    assert len(subset.metadata["candidate_states"]) == subset.size
    assert len(subset.metadata["row_times"]) == subset.size
    assert set(subset.metadata["candidate_states"]) == {"phase0", "phase1", "phase2"}
    assert subset.metadata["scalar_metadata"] == "kept"


def test_oof_fold_excludes_heldout_seed_and_ranks_all_actions() -> None:
    fold = _fit_oof_fold(
        heldout_seed=11,
        contrast=_dataset(),
        model_specs=_model_specs(),
        scenario="jinan_3x4_real",
        expected_action_count=3,
    )

    assert fold["heldout_absent_from_training"] is True
    assert fold["training_validation_disjoint"] is True
    assert fold["training_group_count"] == 24
    assert fold["validation_group_count"] == 12
    assert len(fold["model_results"]) == 2
    for result in fold["model_results"]:
        assert len(result["records"]) == 12
        assert {row["simulator_seed"] for row in result["records"]} == {11}
        assert {row["action_count"] for row in result["records"]} == {3}
        assert all(row["reference_candidate_state"] == "phase0" for row in result["records"])


def test_score_tie_falls_back_to_reference() -> None:
    rows = np.asarray([0, 1, 2], dtype=int)
    scores = np.asarray([0.0, 0.0, 0.0])

    selected = _argmin_with_reference_tie_break(
        rows,
        scores,
        reference=1,
        candidate_states=np.asarray(["phase0", "phase1", "phase2"]),
    )

    assert selected == 1


def test_gate_selection_uses_deployed_delta_and_seed_level_bootstrap() -> None:
    records = []
    for seed in (11, 22, 33):
        for group_index in range(10):
            records.append(
                {
                    "model_key": "pairwise",
                    "model_family": "antisymmetric_pairwise",
                    "simulator_seed": seed,
                    "selected_differs": True,
                    "predicted_delta": -0.5,
                    "selected_uncertainty": 0.1,
                    "selected_context_trust": 1.0,
                    "actual_group_normalized_delta": -0.2,
                }
            )
    selection = select_ranker_gate(
        records,
        model_specs=[{"key": "pairwise"}],
        seeds=(11, 22, 33),
        selection={
            "gate_candidates": [
                {"risk_multiplier": 1.0, "minimum_context_trust": 0.9}
            ],
            "selection_rule": {
                "minimum_retained_groups_per_city": 20,
                "minimum_retained_seed_fraction": 1.0,
                "maximum_equal_seed_scenario_mean_delta": -0.1,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.0,
                "maximum_harmful_group_fraction": 0.45,
            },
            "bootstrap": {"replicates": 1000, "seed": 9},
        },
    )

    assert selection["decision"] == "freeze_selected_direct_ranker"
    assert selection["feasible_candidate_count"] == 1
    assert selection["selected"]["retained_group_count"] == 30
    assert selection["selected"]["bootstrap"]["ci95"][1] < 0.0
