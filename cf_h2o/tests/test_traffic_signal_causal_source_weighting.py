from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    FrozenCausalSourceMixtureModel,
    evaluate_source_weight_grid_from_offline_groups,
    select_source_city_candidate_from_closed_loop,
    select_source_city_weight_from_offline_folds,
    select_source_city_weight_with_familywise_control,
    select_source_weight_from_closed_loop,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _rows(waiting_by_weight):
    return [
        {
            "seed": seed,
            "source_weight": weight,
            "waiting": waiting,
            "collision_incidents": 0,
            "teleports": 0,
        }
        for seed, seed_values in waiting_by_weight.items()
        for weight, waiting in seed_values.items()
    ]


def test_source_weight_selector_accepts_robust_improvement() -> None:
    result = select_source_weight_from_closed_loop(
        _rows(
            {
                1: {0.0: 100, 0.25: 98, 0.5: 96, 0.75: 97, 1.0: 101},
                2: {0.0: 100, 0.25: 99, 0.5: 95, 0.75: 98, 1.0: 102},
            }
        )
    )

    assert result["selected_source_weight"] == pytest.approx(0.5)
    assert result["selection_reason"] == "robust_nonzero_source_weight"


def test_source_weight_selector_falls_back_when_gain_is_unstable() -> None:
    result = select_source_weight_from_closed_loop(
        _rows(
            {
                1: {0.0: 100, 0.25: 90, 0.5: 88, 0.75: 87, 1.0: 86},
                2: {0.0: 100, 0.25: 102, 0.5: 105, 0.75: 108, 1.0: 112},
            }
        )
    )

    assert result["selected_source_weight"] == pytest.approx(0.0)
    assert result["selection_reason"] == "target_only_fallback"


def test_source_weight_selector_rejects_added_incident() -> None:
    rows = _rows(
        {
            1: {0.0: 100, 0.25: 98, 0.5: 95, 0.75: 96, 1.0: 97},
            2: {0.0: 100, 0.25: 98, 0.5: 95, 0.75: 96, 1.0: 97},
        }
    )
    for row in rows:
        if row["source_weight"] == 0.5 and row["seed"] == 2:
            row["collision_incidents"] = 1

    result = select_source_weight_from_closed_loop(rows)

    assert result["selected_source_weight"] != pytest.approx(0.5)


def test_source_city_mixture_is_convex_and_uses_conservative_trust(
    monkeypatch,
) -> None:
    class Dataset:
        size = 2

    class Model:
        def __init__(self, mean, uncertainty, trust):
            self.values = (
                np.asarray(mean, dtype=float),
                np.asarray(uncertainty, dtype=float),
                np.asarray(trust, dtype=float),
            )

        def predict(self, dataset):
            assert dataset.size == 2
            return self.values

    def adjusted(dataset, prediction, *, objective_mode):
        assert dataset.size == 2
        assert objective_mode == "control_only"
        return (*prediction, {})

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_causal_source_weighting._group_adjusted_scores",
        adjusted,
    )
    model = FrozenCausalSourceMixtureModel(
        source_models={
            "a": Model([8, 14], [4, 6], [0.8, 0.7]),
            "b": Model([12, 6], [2, 8], [0.9, 0.5]),
        },
        target_only_model=Model([10, 10], [1, 1], [1.0, 0.9]),
        source_objective_modes={"a": "control_only", "b": "control_only"},
        target_only_objective_mode="control_only",
        source_city_weights={"a": 0.25, "b": 0.5},
    )

    prediction = model.predict(Dataset())["control_cost"]

    assert np.allclose(prediction["mean"], [10.5, 9.0])
    assert np.allclose(prediction["uncertainty"], [2.25, 5.75])
    assert np.allclose(prediction["context_trust"], [0.8, 0.5])
    assert model.audit.to_dict() == {"prediction_calls": 1, "prediction_rows": 2}


def test_source_city_mixture_rejects_mass_above_one() -> None:
    with pytest.raises(ValueError, match="sum to at most one"):
        FrozenCausalSourceMixtureModel(
            source_models={"a": object(), "b": object()},
            target_only_model=object(),
            source_objective_modes={"a": "x", "b": "x"},
            target_only_objective_mode="x",
            source_city_weights={"a": 0.6, "b": 0.5},
        )


def test_source_city_selector_equal_weights_scenarios_and_selects_robust_gain() -> None:
    rows = []
    for seed, baseline, source in (
        (1, (100.0, 200.0), (96.0, 192.0)),
        (2, (120.0, 180.0), (116.0, 173.0)),
    ):
        for scenario, base_waiting, source_waiting in zip(
            ("a", "b"), baseline, source, strict=True
        ):
            rows.extend(
                [
                    {
                        "city": "test",
                        "scenario": scenario,
                        "seed": seed,
                        "candidate_key": "target_only",
                        "metrics": {
                            "mean_tripinfo_waiting_time": base_waiting,
                            "collision_incidents": 0,
                            "starting_teleports": 0,
                            "ending_teleports": 0,
                        },
                    },
                    {
                        "city": "test",
                        "scenario": scenario,
                        "seed": seed,
                        "candidate_key": "source_x",
                        "metrics": {
                            "mean_tripinfo_waiting_time": source_waiting,
                            "collision_incidents": 0,
                            "starting_teleports": 0,
                            "ending_teleports": 0,
                        },
                    },
                ]
            )

    result = select_source_city_candidate_from_closed_loop(
        rows,
        city="test",
        scenarios=("a", "b"),
        candidate_keys=("target_only", "source_x"),
    )

    assert result["selected_candidate_key"] == "source_x"
    assert result["selection_reason"] == "robust_source_city_component"


def test_source_city_selector_rejects_added_collision() -> None:
    rows = []
    for candidate, waiting, collisions in (
        ("target_only", 100.0, 0),
        ("source_x", 90.0, 1),
    ):
        rows.append(
            {
                "city": "test",
                "scenario": "a",
                "seed": 1,
                "candidate_key": candidate,
                "metrics": {
                    "mean_tripinfo_waiting_time": waiting,
                    "collision_incidents": collisions,
                    "starting_teleports": 0,
                    "ending_teleports": 0,
                },
            }
        )

    result = select_source_city_candidate_from_closed_loop(
        rows,
        city="test",
        scenarios=("a",),
        candidate_keys=("target_only", "source_x"),
    )

    assert result["selected_candidate_key"] == "target_only"


def _offline_selection_rows(source_deltas):
    rows = []
    baseline = (0.20, 0.22, 0.18, 0.21, 0.19)
    for fold, baseline_regret in enumerate(baseline):
        rows.append(
            {
                "fold_index": fold,
                "candidate_key": "target_only",
                "source_city_group": None,
                "source_weight": 0.0,
                "equal_scenario_mean_normalized_action_regret": baseline_regret,
            }
        )
        rows.append(
            {
                "fold_index": fold,
                "candidate_key": "atlanta_w1",
                "source_city_group": "atlanta",
                "source_weight": 1.0,
                "equal_scenario_mean_normalized_action_regret": (
                    baseline_regret + source_deltas[fold]
                ),
            }
        )
    return rows


def test_offline_source_selector_accepts_stable_cross_fitted_gain() -> None:
    result = select_source_city_weight_from_offline_folds(
        _offline_selection_rows((-0.04, -0.03, -0.05, -0.02, -0.04))
    )

    assert result["selected_candidate_key"] == "atlanta_w1"
    assert result["selected_source_city_group"] == "atlanta"
    assert result["selection_reason"] == "offline_cross_fitted_source_benefit"


def test_offline_source_selector_falls_back_on_one_bad_fold() -> None:
    result = select_source_city_weight_from_offline_folds(
        _offline_selection_rows((-0.04, -0.03, -0.05, 0.02, -0.04))
    )

    assert result["selected_candidate_key"] == "target_only"
    assert result["selected_source_weight"] == pytest.approx(0.0)


def test_familywise_source_selector_corrects_all_transfer_candidates() -> None:
    rows = _offline_selection_rows((-0.08, -0.07, -0.09, -0.06, -0.08))
    baseline_by_fold = {
        int(row["fold_index"]): float(
            row["equal_scenario_mean_normalized_action_regret"]
        )
        for row in rows
        if row["candidate_key"] == "target_only"
    }
    for candidate_index in range(1, 20):
        for fold, baseline in baseline_by_fold.items():
            rows.append(
                {
                    "fold_index": fold,
                    "candidate_key": f"source_{candidate_index}_w1",
                    "source_city_group": f"source_{candidate_index}",
                    "source_weight": 1.0,
                    "equal_scenario_mean_normalized_action_regret": baseline,
                }
            )

    result = select_source_city_weight_with_familywise_control(rows)

    assert result["selected_candidate_key"] == "atlanta_w1"
    assert result["nonbaseline_candidate_count"] == 20
    assert result["simultaneous_confidence_multiplier"] > 2.8
    assert result["negative_transfer_fallback"] == "exact_target_only"


def test_familywise_source_selector_keeps_exact_target_only_fallback() -> None:
    result = select_source_city_weight_with_familywise_control(
        _offline_selection_rows((-0.02, 0.0, -0.01, 0.01, -0.02))
    )

    assert result["selected_candidate_key"] == "target_only"
    assert result["selected_source_weight"] == pytest.approx(0.0)


def test_offline_source_evaluation_equal_weights_scenarios() -> None:
    feature_names = ("green_q", "green_down_q", "service_pressure")
    dataset = MechanismDataset(
        feature_names=feature_names,
        features=np.asarray(
            [
                [3.0, 0.0, 3.0],
                [1.0, 0.0, 1.0],
                [3.0, 0.0, 3.0],
                [1.0, 0.0, 1.0],
                [3.0, 0.0, 3.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        context_names=("demand",),
        context=np.ones((6, 1), dtype=float),
        priors={"interval_cost": np.zeros(6, dtype=float)},
        targets={
            "interval_cost": np.asarray([0.0, 10.0, 0.0, 10.0, 10.0, 0.0])
        },
        domains=np.asarray(["target"] * 6),
        metadata={
            "action_group_ids": [
                "large:0",
                "large:0",
                "large:1",
                "large:1",
                "small:0",
                "small:0",
            ]
        },
    )

    class FixedModel:
        def __init__(self, candidate_scores):
            self.candidate_scores = np.asarray(candidate_scores, dtype=float)

        def predict(self, contrast):
            return {
                "control_cost": {
                    "mean": self.candidate_scores,
                    "uncertainty": np.zeros(contrast.size, dtype=float),
                    "context_trust": np.ones(contrast.size, dtype=float),
                }
            }

    result = evaluate_source_weight_grid_from_offline_groups(
        dataset,
        reference_policy="phase_pressure",
        source_model=FixedModel([0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        target_only_model=FixedModel([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        source_objective_mode="control_only",
        target_only_objective_mode="control_only",
        scenarios=("large", "small"),
        weights=(0.0, 1.0),
        contrast_features=feature_names,
    )

    assert result["grid"]["0.0"][
        "equal_scenario_mean_normalized_action_regret"
    ] == pytest.approx(0.5)
    assert result["grid"]["1.0"][
        "equal_scenario_mean_normalized_action_regret"
    ] == pytest.approx(0.0)
