"""Phase 2 Residual CFCMT Model, Phase-Discriminating Features, & Symmetric NEMA Max-Red Cap.

Core Phase 2 mechanisms:
1. Phase-discriminating lane representation (`active_in_lanes`): strips constant permissive
   right-turn slip lanes ('g' active across all phases in 8-phase Chinese grids) while
   preserving `PRIOR_ARRIVAL_RATE = 0.04` and `red_elapsed_sec` anti-starvation cues.
2. Analytical `conservation_gp` base cost anchor `b_g(a)` + Endpoint-Antisymmetric Residual
   Action Ranker learning `r_g(a) = z_g(a) - b_g(a)` (`eta = 0` recovers exact `conservation_gp`).
3. Offline Negative-Transfer Source Admission Gate: admits source domains `m` strictly when
   their offline residual regret on the target calibration set improves over `b_g(a)`.
4. Symmetric Demand-Aware NEMA `Max-Red` Anti-Starvation Cap (`max_red_sec = 110.0`):
   applied identically across `maxred_conservation_gp_10` and all learned arms (`target_only`,
   `source_placebo`, `pool_all`, `zero_shot_cfcmt`, `cfcmt`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    ACTION_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    EndpointAntisymmetricActionRanker,
    normalized_group_costs,
)
from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    GREEN_CHARS,
    PartialCvObserver,
)

BASE_GAMMA0 = 0.65
DEFAULT_MAX_RED_SEC = 110.0
MIN_STARVED_DEMAND_VEH = 1.2


@dataclass(frozen=True)
class ResidualGroupRecord:
    """Counterfactual decision state storing both analytical base score and true branch cost."""

    group_id: str
    domain: str
    scenario: str
    seed: int
    tls_id: str
    time_sec: float
    candidate_states: tuple[str, ...]
    reference_index: int
    state_vec: np.ndarray  # shape (18,)
    action_mat: np.ndarray  # shape (K_g, 30)
    base_scores: np.ndarray  # shape (K_g,), lower = higher conservation_gp pressure
    branch_costs: np.ndarray  # shape (K_g,), 1-hop light-cone delay cost


def compute_exact_cons_gp_raw_pressures(
    *,
    info: Any,
    feasible: Sequence[str],
    observer: PartialCvObserver,
) -> np.ndarray:
    """Compute exact `conservation_gp` pressure for each feasible phase (matching `cv_sensor`)."""
    obs_map = observer.observations
    p_cv = observer.p_cv
    pressures = np.zeros(len(feasible), dtype=np.float64)

    for k, state in enumerate(feasible):
        green_lanes: list[str] = []
        for link_idx, char in enumerate(state):
            if char not in GREEN_CHARS or link_idx >= len(info.controlled_links):
                continue
            lg = info.controlled_links[link_idx]
            if not lg or not lg[0]:
                continue
            in_lane = str(lg[0][0])
            if in_lane not in green_lanes:
                green_lanes.append(in_lane)

        if not green_lanes:
            pressures[k] = 0.0
            continue

        q_green = 0.0
        down_occ_vals: list[float] = []
        for in_lane in green_lanes:
            obs = obs_map[in_lane]
            lane_q = (
                max(obs.shockwave_queue_est, obs.conservation_queue_est)
                + 0.35 * max(0.0, obs.cv_veh_count / p_cv - obs.cv_halted_count / p_cv)
                + 0.025 * min(100.0, obs.cv_occ_pct / p_cv)
            )
            q_green += lane_q
            outs = observer.in_to_out_lanes.get(in_lane, ())
            if outs:
                out_occs = [
                    min(100.0, obs_map[o].cv_occ_pct / p_cv)
                    for o in outs
                    if o in obs_map
                ]
                if out_occs:
                    down_occ_vals.append(float(np.mean(out_occs)))

        o_down = float(np.mean(down_occ_vals)) if down_occ_vals else 0.0
        pressures[k] = q_green - 0.02 * o_down

    return pressures


def pressures_to_base_cost_scores(
    pressures: np.ndarray, gamma0: float = BASE_GAMMA0
) -> np.ndarray:
    """Convert raw `conservation_gp` pressures (higher=better) to zero-mean base costs (lower=better)."""
    p = np.asarray(pressures, dtype=np.float64)
    if p.size <= 1:
        return np.zeros_like(p)
    span = max(float(np.max(p) - np.min(p)), 0.50)
    # Tiny index ramp preserves exact first-max tie-breaking of `select_observed_pressure_phase`
    tie_ramp = 1e-9 * (np.arange(p.size, dtype=np.float64) - 0.5 * (p.size - 1))
    return -float(gamma0) * (p - float(np.mean(p))) / span + tie_ramp


def feasible_indices_under_max_red(
    *,
    info: Any,
    feasible: Sequence[str],
    observer: PartialCvObserver,
    max_red_sec: float | None = None,
    min_demand_veh: float = MIN_STARVED_DEMAND_VEH,
    current_green_state: str | None = None,
) -> tuple[int, ...]:
    """Symmetric NEMA Max-Red (anti-starvation) + Gap-Out (work-conservation) phase filter."""
    n_cand = len(feasible)
    if n_cand <= 1:
        return tuple(range(n_cand))

    in_lanes = tuple(str(l) for l in info.incoming_lanes)
    raw_green_sets: list[set[str]] = []
    for state in feasible:
        gs: set[str] = set()
        for link_idx, char in enumerate(state):
            if char in GREEN_CHARS and link_idx < len(info.controlled_links):
                lg = info.controlled_links[link_idx]
                if lg and lg[0]:
                    gs.add(str(lg[0][0]))
        raw_green_sets.append(gs)

    varying_lanes = [
        l for l in in_lanes if 0 < sum(1 for gs in raw_green_sets if l in gs) < n_cand
    ]
    check_lanes = varying_lanes if varying_lanes else list(in_lanes)
    avg_green = max(
        float(np.mean([sum(1 for l in check_lanes if l in gs) for gs in raw_green_sets])),
        1.0,
    )
    multiplex_ratio = float(len(check_lanes)) / avg_green
    cv_wait_cap = (
        float(max_red_sec)
        if max_red_sec is not None
        else max(115.0, 70.0 * multiplex_ratio - 25.0)
    )
    drought_cap = (
        float(max_red_sec)
        if max_red_sec is not None
        else float(np.clip(75.0 * multiplex_ratio, 140.0, 310.0))
    )
    obs_map = observer.observations

    starved_lane: str | None = None
    worst_urgency = -1.0
    for lane in check_lanes:
        obs = obs_map[lane]
        est_q = max(obs.shockwave_queue_est, obs.conservation_queue_est)
        has_waiting_cv = obs.cv_veh_count >= 1.0 and obs.red_elapsed_sec >= cv_wait_cap
        has_active_drought = (
            obs.ema_arrival_rate_300s >= 0.020
            and obs.red_elapsed_sec >= drought_cap
            and est_q >= max(min_demand_veh, 2.0)
        )
        if has_waiting_cv or has_active_drought:
            urgency = obs.red_elapsed_sec + 5.0 * est_q + 20.0 * obs.cv_veh_count
            if urgency > worst_urgency:
                worst_urgency = urgency
                starved_lane = lane

    if starved_lane is None:
        return tuple(range(n_cand))

    relieving = tuple(
        idx for idx, gs in enumerate(raw_green_sets) if starved_lane in gs
    )
    return relieving if relieving else tuple(range(n_cand))


def extract_phase2_features(
    *,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
    infos_all: Mapping[str, Any],
    up_neighbors: Sequence[str],
    down_neighbors: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Extract `(feasible_states, state_vec[18], action_mat[K, 30], base_scores[K])` strictly from CV obs."""
    feasible = tuple(str(s) for s in executor.feasible_states_now())
    obs_map = observer.observations
    p_cv = observer.p_cv

    in_lanes = tuple(str(l) for l in info.incoming_lanes)
    raw_green_sets: list[set[str]] = []
    for cand_state in feasible:
        gs: set[str] = set()
        for link_idx, char in enumerate(cand_state):
            if char in GREEN_CHARS and link_idx < len(info.controlled_links):
                lg = info.controlled_links[link_idx]
                if lg and lg[0]:
                    gs.add(str(lg[0][0]))
        raw_green_sets.append(gs)

    if len(feasible) >= 2:
        varying = [
            l
            for l in in_lanes
            if 0 < sum(1 for gs in raw_green_sets if l in gs) < len(feasible)
        ]
        active_in_lanes = tuple(varying) if varying else in_lanes
    else:
        active_in_lanes = in_lanes

    n_in = max(len(active_in_lanes), 1)
    in_obs = [obs_map[l] for l in active_in_lanes]

    up_qs: list[float] = []
    for utls in up_neighbors:
        for l in infos_all[utls].incoming_lanes:
            if l in obs_map:
                up_qs.append(obs_map[l].conservation_queue_est)
    down_qs: list[float] = []
    for dtls in down_neighbors:
        for l in infos_all[dtls].incoming_lanes:
            if l in obs_map:
                down_qs.append(obs_map[l].conservation_queue_est)

    state_vec = np.array(
        [
            sum(o.cv_halted_count for o in in_obs) / (p_cv * n_in),
            sum(o.cv_veh_count for o in in_obs) / (p_cv * n_in),
            sum(o.shockwave_queue_est for o in in_obs) / n_in,
            sum(o.conservation_queue_est for o in in_obs) / n_in,
            max((o.shockwave_queue_est for o in in_obs), default=0.0),
            max((o.conservation_queue_est for o in in_obs), default=0.0),
            max((o.red_elapsed_sec for o in in_obs), default=0.0) / 60.0,
            float(np.mean([o.red_elapsed_sec for o in in_obs])) / 60.0,
            float(executor.green_elapsed_sec) / 60.0,
            float(np.mean([o.ema_arrival_rate_60s for o in in_obs])),
            float(np.mean([o.ema_arrival_rate_300s for o in in_obs])),
            max((o.cv_wait_sec_max for o in in_obs), default=0.0) / 60.0,
            sum(1.0 for o in in_obs if o.cv_veh_count > 0.0) / n_in,
            len(feasible) / 8.0,
            n_in / 16.0,
            float(np.mean(up_qs)) if up_qs else 0.0,
            float(np.mean(down_qs)) if down_qs else 0.0,
            p_cv,
        ],
        dtype=np.float64,
    )

    cur_green = str(executor.current_green_state)
    total_flow_all = max(sum(o.ema_arrival_rate_300s for o in in_obs), 1e-3)

    cand_meta: list[dict[str, Any]] = []
    for cand_state, green_set in zip(feasible, raw_green_sets):
        g_obs = [obs_map[l] for l in active_in_lanes if l in green_set]
        r_obs = [obs_map[l] for l in active_in_lanes if l not in green_set]

        down_occs_raw: list[float] = []
        down_occs_scaled: list[float] = []
        down_sws: list[float] = []
        for l in active_in_lanes:
            if l not in green_set:
                continue
            outs = observer.in_to_out_lanes.get(l, ())
            if outs:
                l_raw, l_scaled, l_sw = [], [], []
                for out_l in outs:
                    if out_l in obs_map:
                        o_out = obs_map[out_l]
                        l_raw.append(o_out.cv_occ_pct)
                        l_scaled.append(min(100.0, o_out.cv_occ_pct / p_cv))
                        l_sw.append(o_out.shockwave_queue_est)
                if l_raw:
                    down_occs_raw.append(float(np.mean(l_raw)))
                    down_occs_scaled.append(float(np.mean(l_scaled)))
                    down_sws.append(float(np.mean(l_sw)))

        o_down_raw = float(np.mean(down_occs_raw)) if down_occs_raw else 0.0
        o_down_scaled = float(np.mean(down_occs_scaled)) if down_occs_scaled else 0.0
        down_occ_frac = o_down_scaled / 100.0
        down_sw_mean = float(np.mean(down_sws)) if down_sws else 0.0

        obs_gp_q = sum(
            o.cv_halted_count + 0.35 * o.cv_veh_count + 0.025 * o.cv_occ_pct
            for o in g_obs
        )
        shock_gp_q = sum(
            o.shockwave_queue_est
            + 0.35 * max(0.0, o.cv_veh_count / p_cv - o.cv_halted_count / p_cv)
            + 0.025 * min(100.0, o.cv_occ_pct / p_cv)
            for o in g_obs
        )
        cons_gp_q = sum(
            max(o.shockwave_queue_est, o.conservation_queue_est)
            + 0.35 * max(0.0, o.cv_veh_count / p_cv - o.cv_halted_count / p_cv)
            + 0.025 * min(100.0, o.cv_occ_pct / p_cv)
            for o in g_obs
        )
        obs_gp_score = (obs_gp_q - 0.02 * o_down_raw) if g_obs else 0.0
        shock_gp_score = (shock_gp_q - 0.02 * o_down_scaled) if g_obs else 0.0
        cons_gp_score = (cons_gp_q - 0.02 * o_down_scaled) if g_obs else 0.0
        hidden_wait_load = sum(
            o.cv_wait_sec_sum / p_cv
            + 0.5 * (1.0 - p_cv) * o.ema_arrival_rate_300s * (o.red_elapsed_sec ** 2)
            for o in g_obs
        )

        cand_meta.append(
            {
                "cand_state": cand_state,
                "g_obs": g_obs,
                "r_obs": r_obs,
                "down_occ_frac": down_occ_frac,
                "down_sw_mean": down_sw_mean,
                "obs_gp_score": obs_gp_score,
                "shock_gp_score": shock_gp_score,
                "cons_gp_score": cons_gp_score,
                "hidden_wait_load": hidden_wait_load,
            }
        )

    max_cons_gp = max(max((m["cons_gp_score"] for m in cand_meta), default=0.0), 0.25)
    max_shock_gp = max(max((m["shock_gp_score"] for m in cand_meta), default=0.0), 0.25)
    max_obs_gp = max(max((m["obs_gp_score"] for m in cand_meta), default=0.0), 0.10)
    sum_hidden_wait = max(sum(m["hidden_wait_load"] for m in cand_meta), 1.0)

    action_rows: list[np.ndarray] = []
    for m in cand_meta:
        cand_state = m["cand_state"]
        g_obs = m["g_obs"]
        r_obs = m["r_obs"]
        n_g = max(len(g_obs), 1)
        n_r = max(len(r_obs), 1)
        down_occ_frac = m["down_occ_frac"]
        down_sw_mean = m["down_sw_mean"]

        is_switch = 1.0 if cand_state != cur_green else 0.0
        clearance_sec = float(executor.projected_clearance_sec(cand_state))
        eff_green_sec = max(0.0, 10.0 - clearance_sec)
        est_green_queue = sum(
            max(o.shockwave_queue_est, o.conservation_queue_est) for o in g_obs
        )
        serviceable_10s = min(
            est_green_queue,
            len(g_obs) * 0.45 * eff_green_sec * max(0.1, 1.0 - 0.7 * down_occ_frac),
        )

        row = np.array(
            [
                float(np.clip(m["cons_gp_score"] / max_cons_gp, -0.5, 1.0)),
                float(np.clip(m["shock_gp_score"] / max_shock_gp, -0.5, 1.0)),
                float(np.clip(m["obs_gp_score"] / max_obs_gp, -0.5, 1.0)),
                m["hidden_wait_load"] / sum_hidden_wait,
                sum(o.ema_arrival_rate_300s for o in g_obs) / total_flow_all,
                len(g_obs) / n_in,
                sum(o.cv_halted_count for o in g_obs) / (p_cv * n_g),
                sum(o.cv_veh_count for o in g_obs) / (p_cv * n_g),
                sum(o.cv_slow_count for o in g_obs) / (p_cv * n_g),
                sum(o.shockwave_queue_est for o in g_obs) / n_g,
                max((o.shockwave_queue_est for o in g_obs), default=0.0),
                sum(o.conservation_queue_est for o in g_obs) / n_g,
                max((o.conservation_queue_est for o in g_obs), default=0.0),
                max(
                    (
                        min(
                            1.5,
                            max(o.cv_max_halt_dist_m, o.conservation_queue_est * 7.5)
                            / max(o.lane_length_m, 10.0),
                        )
                        for o in g_obs
                    ),
                    default=0.0,
                ),
                (
                    sum(o.cv_wait_sec_sum for o in g_obs)
                    / max(sum(o.cv_veh_count for o in g_obs), 1.0)
                )
                / 60.0,
                max((o.cv_wait_sec_max for o in g_obs), default=0.0) / 60.0,
                (sum(o.red_elapsed_sec for o in g_obs) / n_g) / 60.0,
                max((o.red_elapsed_sec for o in g_obs), default=0.0) / 60.0,
                sum(
                    (1.0 - p_cv) * o.ema_arrival_rate_300s * o.red_elapsed_sec
                    for o in g_obs
                )
                / n_g,
                sum(
                    0.5
                    * (1.0 - p_cv)
                    * o.ema_arrival_rate_300s
                    * (o.red_elapsed_sec ** 2)
                    for o in g_obs
                )
                / (60.0 * n_g),
                float(
                    np.mean(
                        [o.cv_mean_speed_mps / max(o.speed_limit_mps, 1.0) for o in g_obs]
                    )
                )
                if g_obs
                else 1.0,
                serviceable_10s / n_g,
                sum(o.conservation_queue_est for o in r_obs) / n_r,
                max((o.red_elapsed_sec for o in r_obs), default=0.0) / 60.0,
                max(
                    (
                        0.5
                        * (1.0 - p_cv)
                        * o.ema_arrival_rate_300s
                        * (o.red_elapsed_sec ** 2)
                        for o in r_obs
                    ),
                    default=0.0,
                )
                / 60.0,
                down_occ_frac,
                down_sw_mean,
                is_switch,
                clearance_sec / 10.0,
                (float(executor.green_elapsed_sec) / 60.0) * (1.0 - is_switch),
            ],
            dtype=np.float64,
        )
        action_rows.append(row)

    exact_pressures = compute_exact_cons_gp_raw_pressures(
        info=info, feasible=feasible, observer=observer
    )
    base_scores = pressures_to_base_cost_scores(exact_pressures, gamma0=BASE_GAMMA0)
    return feasible, state_vec, np.vstack(action_rows), base_scores


class ResidualActionRanker:
    """Identical-architecture endpoint-antisymmetric residual ranker over `b_g(a)`."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.04,
        max_iter: int = 110,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 20.0,
        scale_floor: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.scale_floor = float(scale_floor)
        self.random_state = int(random_state)
        self.model_: HistGradientBoostingRegressor | None = None

    def fit(
        self,
        groups: Sequence[ResidualGroupRecord],
        *,
        domain_weights: Mapping[str, float] | None = None,
    ) -> "ResidualActionRanker":
        dom_counts: dict[str, int] = {}
        for g in groups:
            if len(g.candidate_states) >= 2:
                dom_counts[g.domain] = dom_counts.get(g.domain, 0) + 1
        if not dom_counts:
            raise ValueError("No multi-action groups provided for ResidualActionRanker")

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        w_rows: list[float] = []

        for g in groups:
            k_actions = len(g.candidate_states)
            if k_actions < 2:
                continue
            z, span = normalized_group_costs(g.branch_costs, self.scale_floor)
            res = z - g.base_scores
            dom_total = dom_counts[g.domain]
            dom_mass = (
                float(domain_weights[g.domain])
                if domain_weights is not None and g.domain in domain_weights
                else 1.0 / len(dom_counts)
            )
            n_unordered = k_actions * (k_actions - 1) // 2
            importance = float(np.clip(span / self.scale_floor, 0.25, 4.0))
            pair_w = (dom_mass / max(dom_total * n_unordered, 1)) * importance

            for i in range(k_actions):
                phi_i = g.action_mat[i]
                for j in range(i + 1, k_actions):
                    phi_j = g.action_mat[j]
                    diff = phi_i - phi_j
                    mid = 0.5 * (phi_i + phi_j)
                    target = float(res[i] - res[j])
                    x_rows.append(np.concatenate([g.state_vec, diff, mid]))
                    y_rows.append(target)
                    w_rows.append(pair_w)

                    x_rows.append(np.concatenate([g.state_vec, -diff, mid]))
                    y_rows.append(-target)
                    w_rows.append(pair_w)

        x_arr = np.vstack(x_rows)
        y_arr = np.asarray(y_rows, dtype=np.float64)
        w_arr = np.asarray(w_rows, dtype=np.float64)
        w_arr = w_arr / max(float(np.mean(w_arr)), 1e-12)

        n_rows = int(x_arr.shape[0])
        eff_min_leaf = max(8, min(self.min_samples_leaf, max(n_rows // 6, 8)))
        eff_max_leaves = max(3, min(self.max_leaf_nodes, max(n_rows // eff_min_leaf, 3)))

        reg = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=eff_max_leaves,
            min_samples_leaf=eff_min_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )
        reg.fit(x_arr, y_arr, sample_weight=w_arr)
        self.model_ = reg
        return self

    def predict_residual(
        self, state_vec: np.ndarray, action_mat: np.ndarray
    ) -> np.ndarray:
        """Return antisymmetric zero-sum residual cost adjustments."""
        if self.model_ is None:
            raise RuntimeError("ResidualActionRanker is not fitted")
        k_actions = action_mat.shape[0]
        if k_actions <= 1:
            return np.zeros(k_actions, dtype=np.float64)

        fwd_rows: list[np.ndarray] = []
        rev_rows: list[np.ndarray] = []
        pairs: list[tuple[int, int]] = []
        for i in range(k_actions):
            phi_i = action_mat[i]
            for j in range(i + 1, k_actions):
                phi_j = action_mat[j]
                diff = phi_i - phi_j
                mid = 0.5 * (phi_i + phi_j)
                fwd_rows.append(np.concatenate([state_vec, diff, mid]))
                rev_rows.append(np.concatenate([state_vec, -diff, mid]))
                pairs.append((i, j))

        batch = np.vstack(fwd_rows + rev_rows)
        preds = self.model_.predict(batch)
        n_pairs = len(pairs)
        deltas = 0.5 * (preds[:n_pairs] - preds[n_pairs:])

        res = np.zeros(k_actions, dtype=np.float64)
        for (i, j), d_ij in zip(pairs, deltas):
            res[i] += d_ij
            res[j] -= d_ij
        res /= float(k_actions - 1)
        return res


class ResidualBlendedController:
    """Evaluates S(s, a) = base_scores(a) + eta * [(1 - omega)*Delta_target + omega*Delta_source]."""

    def __init__(
        self,
        *,
        target_ranker: ResidualActionRanker | None,
        source_ranker: ResidualActionRanker | None,
        omega: float,
        eta: float,
    ) -> None:
        self.target_ranker = target_ranker
        self.source_ranker = source_ranker
        self.omega = float(np.clip(omega, 0.0, 1.0))
        self.eta = float(max(0.0, eta))

    def score_group(
        self,
        state_vec: np.ndarray,
        action_mat: np.ndarray,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        if self.eta <= 1e-9:
            return base_scores.copy()
        if self.omega <= 0.0 or self.source_ranker is None:
            assert self.target_ranker is not None
            delta = self.target_ranker.predict_residual(state_vec, action_mat)
        elif self.omega >= 1.0 or self.target_ranker is None:
            assert self.source_ranker is not None
            delta = self.source_ranker.predict_residual(state_vec, action_mat)
        else:
            dt = self.target_ranker.predict_residual(state_vec, action_mat)
            ds = self.source_ranker.predict_residual(state_vec, action_mat)
            delta = (1.0 - self.omega) * dt + self.omega * ds
        return base_scores + self.eta * delta


def _unit_span_zero_mean(v: np.ndarray) -> np.ndarray:
    c = np.asarray(v, dtype=np.float64) - float(np.mean(v))
    sp = float(np.ptp(c))
    if sp <= 1e-8:
        return np.zeros_like(c)
    return c / sp


def project_invariant_group(
    state_vec: np.ndarray, action_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gate green-continuation strictly by active green queue (>= 1.0 veh/lane) so low-demand
    source domains never induce empty-phase postponement in closed loop.
    """
    sv = np.asarray(state_vec, dtype=np.float64).copy()
    am = np.asarray(action_mat, dtype=np.float64).copy()
    q_green_mean = np.maximum(am[:, 9], am[:, 11])
    has_q = np.clip((q_green_mean - 1.0) / 2.5, 0.0, 1.0)
    is_keep = 1.0 - am[:, 27]
    am[:, 27] = -has_q * is_keep
    am[:, 28] = am[:, 28] * (1.0 - has_q * is_keep)
    am[:, 29] = am[:, 29] * has_q
    return sv, am


class PhysicsAnchoredActionController:
    """Confidence-preserving convex action ranker (`(1 - omega) * f_target + omega * f_source`
    with natural score magnitudes, defaulting to `base_scores` only when `total_q <= 0.4`).
    """

    def __init__(
        self,
        *,
        target_ranker: EndpointAntisymmetricActionRanker | None,
        source_ranker: EndpointAntisymmetricActionRanker | None,
        omega: float,
        lam: float = 1.0,
    ) -> None:
        self.target_ranker = target_ranker
        self.source_ranker = source_ranker
        self.omega = float(np.clip(omega, 0.0, 1.0))
        self.lam = float(np.clip(lam, 0.0, 1.0))

    def score_group(
        self,
        state_vec: np.ndarray,
        action_mat: np.ndarray,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        b_unit = np.asarray(base_scores, dtype=np.float64) / BASE_GAMMA0
        n_in = max(float(state_vec[14]) * 16.0, 1.0)
        total_q = max(float(state_vec[2]), float(state_vec[3])) * n_in
        if total_q <= 0.4 or self.lam <= 1e-9:
            return b_unit.copy()
        if self.omega <= 0.0 or self.source_ranker is None:
            assert self.target_ranker is not None
            f_arm = self.target_ranker.score_group(state_vec, action_mat)
        elif self.omega >= 1.0 or self.target_ranker is None:
            assert self.source_ranker is not None
            f_arm = self.source_ranker.score_group(state_vec, action_mat)
        else:
            ft = self.target_ranker.score_group(state_vec, action_mat)
            fs = self.source_ranker.score_group(state_vec, action_mat)
            f_arm = (1.0 - self.omega) * ft + self.omega * fs
        if self.lam >= 1.0 - 1e-9:
            return f_arm
        return (1.0 - self.lam) * b_unit + self.lam * f_arm


def evaluate_residual_regret(
    controller: Any | None,
    groups: Sequence[ResidualGroupRecord],
    scale_floor: float = 0.15,
) -> dict[str, float]:
    """Compute offline counterfactual regret; `controller=None` evaluates exact `conservation_gp`."""
    norm_regs: list[float] = []
    phys_regs: list[float] = []
    hits = 0
    for g in groups:
        if len(g.candidate_states) < 2:
            continue
        scores = (
            g.base_scores
            if controller is None
            else controller.score_group(g.state_vec, g.action_mat, g.base_scores)
        )
        chosen = int(np.argmin(scores))
        min_c = float(np.min(g.branch_costs))
        span = max(float(np.max(g.branch_costs) - min_c), scale_floor)
        pr = float(g.branch_costs[chosen] - min_c)
        phys_regs.append(pr)
        norm_regs.append(pr / span)
        if abs(pr) <= 1e-9:
            hits += 1
    n = max(len(norm_regs), 1)
    return {
        "group_count": len(norm_regs),
        "mean_normalized_regret": float(np.mean(norm_regs)) if norm_regs else 0.0,
        "mean_physical_regret": float(np.mean(phys_regs)) if phys_regs else 0.0,
        "top1_accuracy": float(hits / n),
    }


def permute_residual_groups_placebo(
    source_groups: Sequence[ResidualGroupRecord], rng_seed: int = 20261119
) -> list[ResidualGroupRecord]:
    """Permute true branch_costs within each group so `z_perm - base_scores` has zero true signal."""
    rng = np.random.default_rng(rng_seed)
    out: list[ResidualGroupRecord] = []
    for g in source_groups:
        # Permute the true residual `(z - b)` directly around `b` so placebo cannot accidentally inherit `-b`!!
        z, span = normalized_group_costs(g.branch_costs, 0.15)
        res = z - g.base_scores
        perm_res = res.copy()
        rng.shuffle(perm_res)
        # Reconstruct synthetic branch_costs whose normalized z equals `base_scores + perm_res`
        synth_z = g.base_scores + perm_res
        synth_costs = synth_z * max(span, 0.15) + float(np.mean(g.branch_costs))
        out.append(
            ResidualGroupRecord(
                group_id=g.group_id,
                domain=g.domain,
                scenario=g.scenario,
                seed=g.seed,
                tls_id=g.tls_id,
                time_sec=g.time_sec,
                candidate_states=g.candidate_states,
                reference_index=g.reference_index,
                state_vec=g.state_vec,
                action_mat=g.action_mat,
                base_scores=g.base_scores,
                branch_costs=synth_costs,
            )
        )
    return out
