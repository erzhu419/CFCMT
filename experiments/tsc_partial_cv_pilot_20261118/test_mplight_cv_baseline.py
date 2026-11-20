"""Unit tests for sensor-matched MPLightCvQNetwork."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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
                    next_allowed=(),
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


def test_double_dqn_bootstrap_uses_successor_allowed_actions(monkeypatch) -> None:
    net = MPLightCvQNetwork(gamma=0.9)
    for parameter in net.params.values():
        parameter.fill(0.0)
    monkeypatch.setattr(
        net,
        "predict_q",
        lambda z_mat, *, use_target: np.array(
            [20.0, 2.0, 3.0] if use_target else [5.0, 2.0, 1.0]
        ),
    )
    transition = ReplayTransition(
        z_cur=np.zeros((1, 78)),
        action_idx=0,
        reward=0.0,
        z_next=np.zeros((3, 78)),
        next_allowed=(1, 2),
        done=False,
    )

    # Action 0 is disallowed and action 2 only leads according to the target net.
    loss = net.train_minibatch([transition])

    assert loss == pytest.approx(0.5 * (0.9 * 2.0) ** 2)


def test_rollout_records_singleton_ticks_and_terminal_reward(monkeypatch) -> None:
    from cf_h2o import sumo_runtime
    from cf_h2o.eval import traffic_signal_resco_phase_benchmark as phase_benchmark
    from cf_h2o.traffic_signal import safe_phase_controller
    from experiments.tsc_partial_cv_pilot_20261118 import phase3_tits_suite as suite

    class SimulationClock:
        time_sec = 0

        def getTime(self):
            return self.time_sec

        def step(self):
            self.time_sec += 1

    clock = SimulationClock()
    requests = []
    simulator = SimpleNamespace(simulation=clock, close=lambda: None)
    executor = SimpleNamespace(
        current_green_state="Gr",
        request=requests.append,
        advance=lambda dt: None,
    )
    info = SimpleNamespace(incoming_lanes=("in",))
    lane = SimpleNamespace(shockwave_queue_est=0.0, conservation_queue_est=0.0)

    def observe_step(simulator, dt_sec):
        lane.shockwave_queue_est = clock.time_sec / 10

    def make_features(**kwargs):
        feasible = ("rG",) if clock.time_sec in (60, 70) else ("Gr", "rG")
        state = np.zeros(18)
        state[0] = clock.time_sec / 10
        return feasible, state, np.zeros((len(feasible), 30))

    def allowed_actions(*, feasible, **kwargs):
        return (1,) if clock.time_sec == 80 else tuple(range(len(feasible)))

    monkeypatch.setattr(sumo_runtime, "load_libsumo", lambda: simulator)
    monkeypatch.setattr(phase_benchmark, "_start_sumo", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase_benchmark, "_tls_phase_infos", lambda api: {"tls": info})
    monkeypatch.setattr(
        safe_phase_controller, "build_safe_phase_executors", lambda api, infos: {"tls": executor}
    )
    monkeypatch.setattr(suite, "build_1hop_tls_neighbors", lambda infos: ({"tls": ()}, {"tls": ()}, {}))
    monkeypatch.setattr(
        suite,
        "PartialCvObserver",
        lambda **kwargs: SimpleNamespace(
            observe_step=observe_step, observations={"in": lane}
        ),
    )
    monkeypatch.setattr(suite, "extract_decision_group_features", make_features)
    monkeypatch.setattr(suite, "feasible_indices_under_max_red", allowed_actions)

    transitions = suite.collect_rollout_transitions_worker(
        {
            "sumocfg": "unused.sumocfg",
            "seed": 7,
            "epsilon": 0.0,
            "rng_seed": 8,
            "net": MPLightCvQNetwork(),
        }
    )

    assert len(transitions) == 355
    assert [tr.z_cur[0, 0] for tr in transitions[:4]] == [5.0, 6.0, 7.0, 8.0]
    assert [tr.reward for tr in transitions[:4]] == pytest.approx([-1.2, -1.4, -1.6, -1.8])
    assert [tr.action_idx for tr in transitions[1:4]] == [0, 0, 1]
    assert transitions[2].z_next.shape == (2, 78)
    assert transitions[2].next_allowed == (1,)
    assert sum(tr.done for tr in transitions) == 1
    assert transitions[-1].done
    assert transitions[-1].reward == -10.0
    assert transitions[-1].next_allowed == ()
    assert len(requests) == 360
    assert clock.time_sec == 3600


if __name__ == "__main__":
    test_phase_competition_and_td_convergence()
    print("ALL_MPLIGHT_CV_BASELINE_TESTS_PASSED")
