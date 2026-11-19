"""Unit tests for Phase 2 Residual CFCMT, Freedman-Lane placebo, & Max-Red filter."""

from __future__ import annotations

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    ACTION_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    ActionGroupRecord,
    EndpointAntisymmetricActionRanker,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_model import (
    PhysicsAnchoredActionController,
    ResidualActionRanker,
    ResidualBlendedController,
    ResidualGroupRecord,
    evaluate_residual_regret,
    feasible_indices_under_max_red,
    permute_residual_groups_placebo,
    pressures_to_base_cost_scores,
)


def test_residual_learning_and_freedman_lane_placebo() -> None:
    rng = np.random.default_rng(2026)
    n_s = len(STATE_FEATURE_NAMES)
    n_a = len(ACTION_FEATURE_NAMES)
    groups: list[ResidualGroupRecord] = []

    for idx in range(140):
        s_vec = rng.normal(0.0, 1.0, size=n_s)
        a_mat = rng.normal(0.0, 0.3, size=(4, n_a))
        raw_p = rng.uniform(0.5, 6.0, size=4)
        a_mat[:, 0] = raw_p / np.max(raw_p)
        # Switch indicator in feature 27 creates systematic residual yellow-loss penalty
        sw = np.array([0.0, 1.0, 1.0, 1.0])
        rng.shuffle(sw)
        a_mat[:, 27] = sw

        b_scores = pressures_to_base_cost_scores(raw_p)
        # True cost = base + switching yellow-loss residual + hidden red relief on col 17
        red_age = rng.uniform(0.0, 3.0, size=4)
        a_mat[:, 17] = red_age
        true_cost = b_scores + 0.55 * (sw - np.mean(sw)) - 0.35 * (red_age - np.mean(red_age))

        groups.append(
            ResidualGroupRecord(
                group_id=f"rg_{idx}",
                domain="src_1",
                scenario="scen_1",
                seed=1,
                tls_id="J1",
                time_sec=float(100 + idx * 10),
                candidate_states=("p0", "p1", "p2", "p3"),
                reference_index=int(np.argmax(raw_p)),
                state_vec=s_vec,
                action_mat=a_mat,
                base_scores=b_scores,
                branch_costs=true_cost,
            )
        )

    tr, te = groups[:100], groups[100:]
    base_m = evaluate_residual_regret(None, te)

    ranker = ResidualActionRanker(max_iter=100, l2_regularization=10.0).fit(tr)
    ctrl_eta0 = ResidualBlendedController(
        target_ranker=None, source_ranker=ranker, omega=1.0, eta=0.0
    )
    m_eta0 = evaluate_residual_regret(ctrl_eta0, te)
    assert abs(m_eta0["mean_normalized_regret"] - base_m["mean_normalized_regret"]) < 1e-12

    ctrl = ResidualBlendedController(
        target_ranker=None, source_ranker=ranker, omega=1.0, eta=1.0
    )
    m_real = evaluate_residual_regret(ctrl, te)
    assert m_real["mean_normalized_regret"] < 0.35 * base_m["mean_normalized_regret"], (
        base_m,
        m_real,
    )

    plac_ranker = ResidualActionRanker(max_iter=100, l2_regularization=10.0).fit(
        permute_residual_groups_placebo(tr, rng_seed=77)
    )
    plac_ctrl = ResidualBlendedController(
        target_ranker=None, source_ranker=plac_ranker, omega=1.0, eta=1.0
    )
    m_plac = evaluate_residual_regret(plac_ctrl, te)
    assert m_plac["mean_normalized_regret"] > 2.0 * m_real["mean_normalized_regret"], (
        m_plac,
        m_real,
    )


def test_physics_anchored_and_two_clause_maxred() -> None:
    from types import SimpleNamespace

    # 8-phase intersection with 8 varying lanes (2 green per phase -> multiplex_ratio = 4.0)
    lanes = [f"L{i}" for i in range(8)]
    links = [[(l, f"O{i}", 0)] for i, l in enumerate(lanes)]
    feasible = [
        "".join("G" if j in (2 * (k % 4), 2 * (k % 4) + 1) else "r" for j in range(8))
        for k in range(8)
    ]
    info = SimpleNamespace(incoming_lanes=lanes, controlled_links=links)

    def make_obs(cv_cnt: float, red_s: float, ema300: float, cons_q: float) -> SimpleNamespace:
        return SimpleNamespace(
            cv_veh_count=cv_cnt,
            red_elapsed_sec=red_s,
            ema_arrival_rate_300s=ema300,
            shockwave_queue_est=0.0,
            conservation_queue_est=cons_q,
        )

    # Dead left-turn pocket: 0 CVs, decayed EMA (0.0168 < 0.020), even at 265s red -> NO false trigger!
    obs_dead = {l: make_obs(0.0, 265.0, 0.0168, 5.0) for l in lanes}
    allowed_dead = feasible_indices_under_max_red(
        info=info, feasible=feasible, observer=SimpleNamespace(observations=obs_dead)
    )
    assert allowed_dead == tuple(range(8)), allowed_dead

    # Confirmed waiting CV on L2 at 260s (>= 255s) -> restricts to phases serving L2 (1 and 5)
    obs_cv = dict(obs_dead)
    obs_cv["L2"] = make_obs(1.0, 260.0, 0.015, 3.0)
    allowed_cv = feasible_indices_under_max_red(
        info=info, feasible=feasible, observer=SimpleNamespace(observations=obs_cv)
    )
    assert allowed_cv == (1, 5), allowed_cv

    # PhysicsAnchoredActionController falls back to exact base_scores ordering when total_q <= 0.4
    ag = ActionGroupRecord(
        group_id="g0",
        domain="d",
        scenario="s",
        seed=1,
        tls_id="J",
        time_sec=10.0,
        candidate_states=("p0", "p1"),
        reference_index=0,
        state_vec=np.zeros(len(STATE_FEATURE_NAMES)),
        action_mat=np.vstack([np.zeros(len(ACTION_FEATURE_NAMES)), np.ones(len(ACTION_FEATURE_NAMES))]),
        branch_costs=np.array([1.0, 0.0]),
    )
    rk = EndpointAntisymmetricActionRanker(max_iter=20).fit([ag] * 8)
    ctrl = PhysicsAnchoredActionController(target_ranker=rk, source_ranker=None, omega=0.0, lam=0.80)
    empty_s = np.zeros(len(STATE_FEATURE_NAMES))
    empty_s[14] = 4.0 / 16.0  # 4 lanes, 0 queue -> total_q = 0 <= 0.4
    bs = pressures_to_base_cost_scores(np.array([5.0, 1.0]))
    sc_empty = ctrl.score_group(empty_s, ag.action_mat, bs)
    assert int(np.argmin(sc_empty)) == 0  # follows base_scores (`p0` has higher pressure)

    busy_s = empty_s.copy()
    busy_s[3] = 2.0  # total_q = 8.0 >= 1.2 -> full lam=0.80 learned preference (`p1` has lower branch_cost)
    bs_flat = pressures_to_base_cost_scores(np.array([1.05, 1.00]))
    sc_busy = ctrl.score_group(busy_s, ag.action_mat, bs_flat)
    assert int(np.argmin(sc_busy)) == 1


if __name__ == "__main__":
    test_residual_learning_and_freedman_lane_placebo()
    test_physics_anchored_and_two_clause_maxred()
    print("ALL_PHASE2_RESIDUAL_MODEL_TESTS_PASSED")
