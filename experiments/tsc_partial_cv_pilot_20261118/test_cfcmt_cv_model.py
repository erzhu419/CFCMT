"""Unit tests for EndpointAntisymmetricActionRanker and non-confounded arms."""

from __future__ import annotations

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    ACTION_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    ActionGroupRecord,
    BlendedActionRanker,
    CapacityMatchedActionCostRegressor,
    EndpointAntisymmetricActionRanker,
    evaluate_offline_regret,
    permute_source_groups_placebo,
)


def test_exact_antisymmetry_and_nonlinear_endpoint_learning() -> None:
    """Verify antisymmetry and recovery of nonlinear cost F(u) = (u - 5)^2."""
    rng = np.random.default_rng(123)
    groups: list[ActionGroupRecord] = []
    n_s = len(STATE_FEATURE_NAMES)
    n_a = len(ACTION_FEATURE_NAMES)

    for idx in range(120):
        state = rng.normal(0.0, 1.0, size=n_s)
        u_vals = rng.uniform(0.0, 10.0, size=4)
        action_mat = np.zeros((4, n_a), dtype=np.float64)
        action_mat[:, 0] = u_vals
        action_mat[:, 1:] = rng.normal(0.0, 0.2, size=(4, n_a - 1))
        # Nonlinear cost depending on absolute endpoint value, not just difference!
        costs = (u_vals - 5.0) ** 2
        groups.append(
            ActionGroupRecord(
                group_id=f"g_{idx}",
                domain="src_a",
                scenario="scen_a",
                seed=1,
                tls_id="J0",
                time_sec=float(100 + idx * 10),
                candidate_states=("p0", "p1", "p2", "p3"),
                reference_index=0,
                state_vec=state,
                action_mat=action_mat,
                branch_costs=costs,
            )
        )

    train_g = groups[:90]
    test_g = groups[90:]
    ranker = EndpointAntisymmetricActionRanker(
        max_iter=100, min_samples_leaf=10, random_state=7
    ).fit(train_g)

    # Check antisymmetry invariance to candidate permutation
    g0 = test_g[0]
    s_orig = ranker.score_group(g0.state_vec, g0.action_mat)
    perm = [2, 0, 3, 1]
    s_perm = ranker.score_group(g0.state_vec, g0.action_mat[perm])
    assert np.allclose(s_orig[perm], s_perm, atol=1e-12)
    assert abs(float(np.sum(s_orig))) < 1e-12

    metrics = evaluate_offline_regret(ranker, test_g)
    # Random regret on 4 actions is ~0.45; nonlinear endpoint ranker should be < 0.08
    assert metrics["mean_normalized_regret"] < 0.08, metrics

    # Placebo permutation destroys ranking signal
    placebo_ranker = EndpointAntisymmetricActionRanker(
        max_iter=100, min_samples_leaf=10, random_state=7
    ).fit(permute_source_groups_placebo(train_g, rng_seed=99))
    placebo_metrics = evaluate_offline_regret(placebo_ranker, test_g)
    assert placebo_metrics["mean_normalized_regret"] > 0.25, placebo_metrics

    blend = BlendedActionRanker(
        target_ranker=placebo_ranker, source_ranker=ranker, omega=0.8
    )
    blend_metrics = evaluate_offline_regret(blend, test_g)
    assert blend_metrics["mean_normalized_regret"] < 0.12, blend_metrics


def test_action_cost_regressor_can_split_on_original_la_fold_size() -> None:
    groups = []
    for idx, action_count in enumerate((4, 2, 4)):
        actions = np.arange(action_count, dtype=np.float64)
        features = np.zeros((action_count, len(ACTION_FEATURE_NAMES)))
        features[:, 0] = actions
        groups.append(
            ActionGroupRecord(
                group_id=f"g_{idx}",
                domain="target_la",
                scenario="la_1x4",
                seed=5057,
                tls_id="J0",
                time_sec=float(180 + idx * 120),
                candidate_states=tuple(f"p{i}" for i in range(action_count)),
                reference_index=0,
                state_vec=np.zeros(len(STATE_FEATURE_NAMES)),
                action_mat=features,
                branch_costs=actions,
            )
        )
    pair = EndpointAntisymmetricActionRanker(max_iter=10).fit(groups).model_
    direct = CapacityMatchedActionCostRegressor(max_iter=10).fit(groups)
    assert direct.model_.max_leaf_nodes == pair.max_leaf_nodes == 4
    assert direct.model_.min_samples_leaf == 2
    assert np.ptp(direct.score_group(groups[0].state_vec, groups[0].action_mat)) > 1e-5


if __name__ == "__main__":
    test_exact_antisymmetry_and_nonlinear_endpoint_learning()
    print("ALL_CFCMT_CV_MODEL_TESTS_PASSED")
