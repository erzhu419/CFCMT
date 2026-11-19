"""Unit tests for sensor-matched MPLightCvQNetwork."""

from __future__ import annotations

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.mplight_cv_baseline import (
    MPLightCvQNetwork,
    ReplayTransition,
    build_phase_competition_inputs,
)


def test_phase_competition_and_td_convergence() -> None:
    rng = np.random.default_rng(7)
    s_vec = rng.normal(size=18)
    a_mat = rng.normal(size=(4, 30))
    z = build_phase_competition_inputs(s_vec, a_mat)
    assert z.shape == (4, 78)

    # Permutation equivariance across candidate actions
    perm = [2, 0, 3, 1]
    net = MPLightCvQNetwork(lr=3e-3, gamma=0.9, seed=11)
    s1 = net.score_group(s_vec, a_mat)
    s2 = net.score_group(s_vec, a_mat[perm])
    assert np.allclose(s1[perm], s2, atol=1e-10)

    # Verify TD learning picks the highest-reward phase on synthetic bandit transitions
    transitions = []
    for _ in range(128):
        sv = rng.normal(size=18)
        am = rng.normal(size=(3, 30))
        am[1, 0] = 2.5  # phase 1 clearly has highest demand
        am[0, 0] = -1.0
        am[2, 0] = -1.0
        zc = build_phase_competition_inputs(sv, am)
        for act_i, rew in ((0, -3.0), (1, -0.2), (2, -3.2)):
            transitions.append(
                ReplayTransition(
                    z_cur=zc,
                    action_idx=act_i,
                    reward=rew,
                    z_next=zc,
                    done=True,
                )
            )

    for _ in range(80):
        idx = rng.choice(len(transitions), size=64, replace=False)
        net.train_minibatch([transitions[int(i)] for i in idx])

    sv_test = rng.normal(size=18)
    am_test = rng.normal(size=(3, 30))
    am_test[1, 0] = 2.5
    am_test[0, 0] = -1.0
    am_test[2, 0] = -1.0
    scores = net.score_group(sv_test, am_test)
    assert int(np.argmin(scores)) == 1


if __name__ == "__main__":
    test_phase_competition_and_td_convergence()
    print("ALL_MPLIGHT_CV_BASELINE_TESTS_PASSED")
