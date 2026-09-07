from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cf_h2o.traffic_signal.target_calibrated_source_gate import (
    ARM_NAMES,
    GATE_BASE_FEATURES,
    SOURCE_AGGREGATE_FEATURES,
    aggregate_source_predictions,
    balanced_group_folds,
    extract_gate_base_features,
    fit_architecture_matched_gate_arms,
    gate_feature_matrix,
    group_conformal_margins,
    permute_source_features_by_group,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    nested_leave_one_seed_gate_evaluation,
    select_identifiable_source_profile,
)


def test_source_aggregation_is_city_permutation_invariant() -> None:
    rows = 9
    predictions = {
        "city_b": (
            np.linspace(-0.2, 0.3, rows),
            np.full(rows, 0.05),
            np.ones(rows),
        ),
        "city_a": (
            np.linspace(-0.1, 0.4, rows),
            np.full(rows, 0.02),
            np.ones(rows),
        ),
    }
    first, scores, order = aggregate_source_predictions(predictions)
    second, _, second_order = aggregate_source_predictions(
        dict(reversed(list(predictions.items())))
    )
    assert order == second_order == ("city_a", "city_b")
    assert first.shape == (rows, len(SOURCE_AGGREGATE_FEATURES))
    assert scores.shape == (2, rows)
    np.testing.assert_allclose(first, second)


def test_group_folds_never_split_action_groups() -> None:
    groups = np.repeat(
        [f"scenario:seed5057:{index}:tls" for index in range(15)], 4
    )
    folds = balanced_group_folds(groups, fold_count=5)
    assert set(folds.tolist()) == set(range(5))
    for group in np.unique(groups):
        assert np.unique(folds[groups == group]).size == 1


def test_architecture_matched_source_arm_beats_neutral_and_placebo() -> None:
    rng = np.random.default_rng(1407)
    group_count = 40
    actions_per_group = 3
    rows = group_count * actions_per_group
    groups = np.repeat(
        [f"scenario:seed5057:{index}:tls" for index in range(group_count)],
        actions_per_group,
    )
    is_reference = np.tile([True, False, False], group_count)
    base = rng.normal(size=(rows, len(GATE_BASE_FEATURES)))
    source = rng.normal(size=(rows, len(SOURCE_AGGREGATE_FEATURES)))
    source[is_reference] = 0.0
    target = 0.05 * base[:, 0] + 0.8 * source[:, 0] - 0.4 * source[:, 1]
    target[is_reference] = 0.0
    placebo = permute_source_features_by_group(
        source,
        groups,
        is_reference,
        base[:, 0],
    )
    matrices = {
        "target_only": gate_feature_matrix(base, source, arm="target_only"),
        "source_aligned": gate_feature_matrix(base, source, arm="source_aligned"),
        "source_placebo": gate_feature_matrix(base, placebo, arm="source_placebo"),
    }
    fitted = fit_architecture_matched_gate_arms(
        feature_matrices=matrices,
        targets=target,
        groups=groups,
        is_reference=is_reference,
    )
    assert set(fitted["models"]) == set(ARM_NAMES)
    assert len({model.alpha for model in fitted["models"].values()}) == 1
    assert len({model.feature_names for model in fitted["models"].values()}) == 1
    source_mse = fitted["arm_diagnostics"]["source_aligned"]["group_mean_mse"]
    target_mse = fitted["arm_diagnostics"]["target_only"]["group_mean_mse"]
    placebo_mse = fitted["arm_diagnostics"]["source_placebo"]["group_mean_mse"]
    assert source_mse < 0.1 * target_mse
    assert source_mse < 0.1 * placebo_mse
    margins = group_conformal_margins(
        actual=target,
        oof_predictions=fitted["oof_predictions"],
        groups=groups,
        is_reference=is_reference,
        quantiles=(0.5, 0.9),
    )
    assert set(margins) == set(ARM_NAMES)
    assert all(np.isfinite(value) for rows in margins.values() for value in rows.values())


def test_extract_base_features_uses_declared_order() -> None:
    feature_names = tuple(reversed(GATE_BASE_FEATURES))
    values = np.arange(3 * len(feature_names), dtype=float).reshape(3, -1)
    dataset = SimpleNamespace(
        feature_names=feature_names,
        features=values,
        size=3,
    )
    extracted = extract_gate_base_features(dataset)
    expected = values[:, [feature_names.index(name) for name in GATE_BASE_FEATURES]]
    np.testing.assert_array_equal(extracted, expected)


def _profile(
    key: str,
    arm: str,
    value: float,
    *,
    matched_target: str | None = None,
    matched_placebo: str | None = None,
) -> dict[str, object]:
    return {
        "profile_key": key,
        "arm": arm,
        "matched_target_profile_key": matched_target,
        "matched_placebo_profile_key": matched_placebo,
        "seed_metrics": {
            str(seed): {
                "group_count": 100,
                "accepted_override_count": 2,
                "harmful_override_count": 0,
                "mean_normalized_delta_vs_phase_pressure": value,
            }
            for seed in range(22)
        },
    }


def test_identifiable_source_gate_requires_target_and_placebo_gain() -> None:
    profiles = {
        "target": _profile("target", "target_only", -0.0010),
        "placebo": _profile("placebo", "source_placebo", -0.0005),
        "source": _profile(
            "source",
            "source_aligned",
            -0.0030,
            matched_target="target",
            matched_placebo="placebo",
        ),
    }
    selection = select_identifiable_source_profile(
        profiles, training_seeds=range(22)
    )
    assert selection["selected_profile_key"] == "source"
    nested = nested_leave_one_seed_gate_evaluation(profiles, range(22))
    assert nested["summary"]["source_transfer_gate_passed"] is True

    profiles["placebo"] = _profile("placebo", "source_placebo", -0.0030)
    rejected = select_identifiable_source_profile(
        profiles, training_seeds=range(22)
    )
    assert rejected["selected_profile_key"] is None
