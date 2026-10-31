import numpy as np

from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    CONDITIONAL_MECHANISM_STATE_FEATURES,
    MECHANISM_ACTION_FEATURES,
    conditional_mechanism_parameter_design,
    equal_city_rms_scale,
    mechanism_parameter_design,
    permute_group_targets,
    ridge_coefficients,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset() -> MechanismDataset:
    required = sorted(
        {
            name
            for values in MECHANISM_ACTION_FEATURES.values()
            for name in values
        }
        | {
            "queue_concentration",
            "mean_occ",
            "mean_speed",
            "current_green_elapsed_norm",
            "graph_focal_to_neighbor_q_ratio",
            "neighbor_phase_observation_ratio",
        }
        | {
            name
            for values in CONDITIONAL_MECHANISM_STATE_FEATURES.values()
            for name in values
        }
    )
    values = np.arange(4 * len(required), dtype=float).reshape(4, -1) / 10.0
    return MechanismDataset(
        feature_names=tuple(required),
        features=values,
        context_names=(),
        context=np.zeros((4, 0)),
        priors={},
        targets={},
        domains=np.asarray(["city"] * 4),
        metadata={
            "action_group_ids": ["g0", "g0", "g1", "g1"],
            "is_reference": [True, False, True, False],
        },
    )


def test_design_is_reference_centered_and_block_partitioned() -> None:
    design = mechanism_parameter_design(_dataset())
    assert design.values.shape == (4, 2 * sum(map(len, MECHANISM_ACTION_FEATURES.values())))
    assert np.all(design.values[[0, 2]] == 0.0)
    observed = np.concatenate(list(design.block_indices.values()))
    assert np.array_equal(np.sort(observed), np.arange(design.values.shape[1]))


def test_conditional_design_adds_only_declared_local_interactions() -> None:
    design = conditional_mechanism_parameter_design(_dataset())
    expected_width = sum(
        len(MECHANISM_ACTION_FEATURES[block]) * (1 + len(conditioners))
        for block, conditioners in CONDITIONAL_MECHANISM_STATE_FEATURES.items()
    )
    assert design.values.shape == (4, expected_width)
    assert np.all(design.values[[0, 2]] == 0.0)
    assert (
        "delta_protected_green_ratio*neighbor_switching_fraction"
        in design.feature_names
    )
    assert (
        "delta_switch_indicator*neighbor_clearance_fraction_mean"
        in design.feature_names
    )


def test_zero_source_strength_is_exact_target_only() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [2.0, -1.0]])
    y = np.asarray([0.0, -1.0, 0.0, 0.5])
    groups = np.asarray(["a", "a", "b", "b"])
    target = ridge_coefficients(x, y, groups, ridge_l2=0.1)
    with_zero_prior = ridge_coefficients(
        x,
        y,
        groups,
        ridge_l2=0.1,
        prior=np.asarray([100.0, -100.0]),
        prior_indices=(0,),
        prior_strength=0.0,
    )
    assert np.array_equal(target, with_zero_prior)


def test_positive_source_strength_moves_only_through_selected_prior() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    y = np.zeros(4)
    groups = np.asarray(["a", "a", "b", "b"])
    fitted = ridge_coefficients(
        x,
        y,
        groups,
        ridge_l2=0.1,
        prior=np.asarray([2.0, 50.0]),
        prior_indices=(0,),
        prior_strength=1.0,
    )
    assert fitted[0] > 0.0
    assert fitted[1] == 0.0


def test_city_scale_and_group_placebo_are_deterministic() -> None:
    scale = equal_city_rms_scale(
        (np.asarray([[0.0, 2.0], [2.0, 2.0]]), np.asarray([[4.0, 0.0]]))
    )
    assert np.all(scale > 0.0)
    values = np.asarray([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = permute_group_targets(values, groups, seed=17)
    second = permute_group_targets(values, groups, seed=17)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == sorted(values.tolist())
    assert not np.array_equal(first, values)


def test_group_placebo_recenters_destination_reference() -> None:
    values = np.asarray([0.0, 2.0, 4.0, 0.0, -1.0, 3.0])
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    references = np.asarray([True, False, False, False, True, False])
    placebo = permute_group_targets(
        values, groups, seed=5, references=references
    )
    assert np.all(placebo[references] == 0.0)
