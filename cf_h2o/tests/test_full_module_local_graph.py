import numpy as np
from types import SimpleNamespace

from cf_h2o.eval.full_module_pipeline_validation import (
    CityTensorData,
    LOCAL_STATIC_FEATURE_NAMES,
    _city_similarity_vector,
    _fit_dense_residual_ensemble,
    _predict_dense_residual_ensemble_y,
    _rigid_first_candidate_name,
    _select_rigid_guard_candidate,
    _select_rigid_first_delta_candidate,
    _select_rigid_first_mechanism_delta_candidate,
    _select_static_local_candidate,
    _source_similarity_weights,
    _static_local_features_np,
)


def test_static_local_features_use_ordered_same_line_neighbors():
    observations = np.zeros((3, 15), dtype=np.float32)
    observations[:, 0] = 0.25
    observations[:, 2] = [0.1, 0.4, 0.9]
    observations[:, 3] = 0.5
    observations[:, 4] = 1.0
    observations[:, 5] = [300.0, 280.0, 260.0]
    observations[:, 6] = [300.0, 320.0, 340.0]
    observations[:, 7] = [10.0, 20.0, 30.0]
    observations[:, 8] = 300.0
    observations[:, 12] = [310.0, 305.0, 300.0]
    observations[:, 13] = [290.0, 295.0, 300.0]
    observations[:, 14] = [5.0, 10.0, 15.0]
    actions = np.zeros((3, 1), dtype=np.float32)

    features = _static_local_features_np(observations, actions, speed_edges=np.array([0.0, 10.0, 20.0]))

    assert features.shape == (3, len(LOCAL_STATIC_FEATURE_NAMES))
    assert np.isfinite(features).all()
    mid = dict(zip(LOCAL_STATIC_FEATURE_NAMES, features[1]))
    assert np.isclose(mid["prev_station_distance"], 0.3)
    assert np.isclose(mid["next_station_distance"], 0.5)
    assert np.isclose(mid["prev_stop_demand"], 10.0)
    assert np.isclose(mid["next_stop_demand"], 30.0)
    assert np.isclose(mid["same_line_near_stop_demand"], 20.0)


def test_static_local_selector_requires_win_rate_and_mean_gain():
    assert _select_static_local_candidate(0.98, 1.0, min_win_rate=2 / 3, min_improvement=0.0)[0] == "deterministic_static_local"
    assert _select_static_local_candidate(1.02, 1.0, min_win_rate=2 / 3, min_improvement=0.0)[0] == "no_local"
    assert _select_static_local_candidate(0.98, 1 / 3, min_win_rate=2 / 3, min_improvement=0.0)[0] == "no_local"


def _mini_city(key: str, offset: float) -> CityTensorData:
    observations = np.zeros((8, 15), dtype=np.float32)
    observations[:, 0] = offset
    observations[:, 2] = np.linspace(0.1, 0.9, 8, dtype=np.float32)
    observations[:, 3] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    observations[:, 4] = 0.5
    observations[:, 5] = 250.0 + offset * 10.0
    observations[:, 6] = 310.0 + offset * 10.0
    observations[:, 7] = np.linspace(2.0, 18.0, 8, dtype=np.float32) + offset
    observations[:, 8] = 300.0
    observations[:, 9] = 90.0
    observations[:, 11] = 25.0
    observations[:, 12] = 280.0
    observations[:, 13] = 330.0
    observations[:, 14] = np.linspace(6.0, 14.0, 8, dtype=np.float32) + offset
    actions = np.full((8, 1), 10.0 + offset, dtype=np.float32)
    sim_y = np.column_stack([observations[:, 5:9], observations[:, 11:15], np.ones(8)]).astype(np.float32)
    return CityTensorData(
        key=key,
        city=key,
        observations=observations,
        actions=actions,
        sim_next_observations=observations.copy(),
        real_next_observations=observations.copy(),
        sim_y=sim_y,
        real_y=sim_y.copy(),
    )


def test_similarity_feature_sets_are_unlabeled_and_sized_by_inputs():
    city = _mini_city("a", 0.0)
    full = _city_similarity_vector(city, max_rows=0, seed=0, feature_set="full")
    obs_only = _city_similarity_vector(city, max_rows=0, seed=0, feature_set="obs_only")
    sim_only = _city_similarity_vector(city, max_rows=0, seed=0, feature_set="sim_only")
    local_only = _city_similarity_vector(city, max_rows=0, seed=0, feature_set="local_only")

    city.real_y[:] = 9999.0
    corrupted = _city_similarity_vector(city, max_rows=0, seed=0, feature_set="full")

    assert full.shape[0] == obs_only.shape[0] + sim_only.shape[0] + local_only.shape[0]
    assert np.allclose(full, corrupted)


def test_uniform_ensemble_weighting_ignores_similarity_distances():
    args = SimpleNamespace(
        rigid_first_ensemble_weighting="uniform",
        selector_similarity_feature_set="full",
        selector_similarity_max_rows=0,
        seed=0,
        rigid_first_similarity_temperature=1.0,
    )
    weights, diagnostics = _source_similarity_weights(
        _mini_city("target", 0.5),
        {"a": _mini_city("a", 0.0), "b": _mini_city("b", 10.0)},
        args,
    )

    assert weights == {"a": 0.5, "b": 0.5}
    assert diagnostics["weighting"] == "uniform"
    assert diagnostics["source_distances"] == {}


def test_dense_residual_ensemble_uses_source_weights():
    source_a = _mini_city("a", 0.0)
    source_b = _mini_city("b", 1.0)
    source_a.real_y = source_a.sim_y + 1.0
    source_b.real_y = source_b.sim_y + 3.0
    target = _mini_city("target", 0.5)

    model = _fit_dense_residual_ensemble(
        {"a": source_a, "b": source_b},
        ridge=1e-6,
        weights={"a": 0.25, "b": 0.75},
    )
    pred = _predict_dense_residual_ensemble_y(model, target)
    residual = pred - target.sim_y

    assert np.isclose(model.weights["a"], 0.25)
    assert np.isclose(model.weights["b"], 0.75)
    assert pred.shape == target.sim_y.shape
    assert np.isfinite(residual).all()


def test_rigid_guard_selector_falls_back_unless_full_is_stable():
    unstable = [
        {
            "candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                "no_local": {"vs_rigid_cfcmt": 0.9},
                "deterministic_static_local": {"vs_rigid_cfcmt": 0.8},
            }
        },
        {
                "candidates": {
                    "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                    "no_local": {"vs_rigid_cfcmt": 1.1},
                    "deterministic_static_local": {"vs_rigid_cfcmt": 1.05},
                }
            },
    ]
    stable = [
        {
            "candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                "no_local": {"vs_rigid_cfcmt": 0.95},
                "deterministic_static_local": {"vs_rigid_cfcmt": 0.85},
            }
        },
        {
            "candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                "no_local": {"vs_rigid_cfcmt": 0.96},
                "deterministic_static_local": {"vs_rigid_cfcmt": 0.86},
            }
        },
    ]

    assert _select_rigid_guard_candidate(unstable, min_win_rate=1.0, min_improvement=0.0)["selected_candidate"] == "rigid_cfcmt"
    assert (
        _select_rigid_guard_candidate(stable, min_win_rate=1.0, min_improvement=0.0)["selected_candidate"]
        == "deterministic_static_local"
    )


def test_rigid_first_delta_selector_requires_stable_gain():
    unstable = [
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                _rigid_first_candidate_name("dense_delta", 0.25): {"vs_rigid_cfcmt": 0.95},
            }
        },
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                _rigid_first_candidate_name("dense_delta", 0.25): {"vs_rigid_cfcmt": 1.02},
            }
        },
    ]
    stable = [
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                _rigid_first_candidate_name("static_local_delta", 0.5): {"vs_rigid_cfcmt": 0.93},
            }
        },
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                _rigid_first_candidate_name("static_local_delta", 0.5): {"vs_rigid_cfcmt": 0.94},
            }
        },
    ]

    assert (
        _select_rigid_first_delta_candidate(unstable, min_win_rate=1.0, min_improvement=0.0)["selected_candidate"]
        == "rigid_cfcmt"
    )
    selected = _select_rigid_first_delta_candidate(stable, min_win_rate=1.0, min_improvement=0.0)
    assert selected["selected_mode"] == "static_local_delta"
    assert np.isclose(selected["selected_alpha"], 0.5)


def test_rigid_first_delta_selector_can_weight_target_similar_sources():
    candidate = _rigid_first_candidate_name("dense_delta", 0.25)
    folds = [
        {
            "target_similarity_weight": 2.8,
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                candidate: {"vs_rigid_cfcmt": 0.9},
            },
        },
        {
            "target_similarity_weight": 0.2,
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                candidate: {"vs_rigid_cfcmt": 1.1},
            },
        },
    ]

    uniform = _select_rigid_first_delta_candidate(folds, min_win_rate=0.8, min_improvement=0.0)
    weighted = _select_rigid_first_delta_candidate(
        folds,
        min_win_rate=0.8,
        min_improvement=0.0,
        min_unweighted_win_rate=0.0,
        weighting="similarity",
    )

    assert uniform["selected_candidate"] == "rigid_cfcmt"
    assert weighted["selected_candidate"] == candidate


def test_rigid_first_delta_selector_requires_dominant_source_for_ensemble():
    ensemble_candidate = "ensemble_" + _rigid_first_candidate_name("dense_delta", 0.25)
    folds = [
        {
            "target_similarity_weight": 2.0,
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                ensemble_candidate: {"vs_rigid_cfcmt": 0.8},
            },
        },
        {
            "target_similarity_weight": 1.0,
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                ensemble_candidate: {"vs_rigid_cfcmt": 1.02},
            },
        },
        {
            "target_similarity_weight": 0.2,
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt": 1.0},
                ensemble_candidate: {"vs_rigid_cfcmt": 0.9},
            },
        },
    ]

    blocked = _select_rigid_first_delta_candidate(
        folds,
        min_win_rate=1.0,
        min_improvement=0.0,
        min_unweighted_win_rate=1.0,
        weighting="similarity",
        ensemble_min_win_rate=0.6,
        ensemble_min_unweighted_win_rate=2 / 3,
        ensemble_min_max_source_weight=0.5,
        ensemble_max_source_weight=0.4,
    )
    selected = _select_rigid_first_delta_candidate(
        folds,
        min_win_rate=1.0,
        min_improvement=0.0,
        min_unweighted_win_rate=1.0,
        weighting="similarity",
        ensemble_min_win_rate=0.6,
        ensemble_min_unweighted_win_rate=2 / 3,
        ensemble_min_max_source_weight=0.5,
        ensemble_max_source_weight=0.6,
    )

    assert blocked["selected_candidate"] == "rigid_cfcmt"
    assert selected["selected_candidate"] == ensemble_candidate


def test_rigid_first_mechanism_selector_can_mix_delta_and_rigid():
    demand_candidate = _rigid_first_candidate_name("static_local_delta", 0.5)
    headway_candidate = _rigid_first_candidate_name("dense_delta", 0.25)
    folds = [
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt_by_mechanism": {"demand": 1.0, "headway": 1.0}},
                demand_candidate: {"vs_rigid_cfcmt_by_mechanism": {"demand": 0.8, "headway": 1.1}},
                headway_candidate: {"vs_rigid_cfcmt_by_mechanism": {"demand": 1.1, "headway": 0.9}},
            }
        },
        {
            "rigid_first_candidates": {
                "rigid_cfcmt": {"vs_rigid_cfcmt_by_mechanism": {"demand": 1.0, "headway": 1.0}},
                demand_candidate: {"vs_rigid_cfcmt_by_mechanism": {"demand": 0.85, "headway": 1.2}},
                headway_candidate: {"vs_rigid_cfcmt_by_mechanism": {"demand": 1.05, "headway": 1.02}},
            }
        },
    ]

    selected = _select_rigid_first_mechanism_delta_candidate(
        folds,
        min_win_rate=1.0,
        min_unweighted_win_rate=1.0,
        min_improvement=0.0,
        weighting="uniform",
    )

    assert selected["selected_by_mechanism"]["demand"]["selected_candidate"] == demand_candidate
    assert selected["selected_by_mechanism"]["headway"]["selected_candidate"] == "rigid_cfcmt"
