"""Portable feature extraction and architecture-matched CFCMT rankers for partial-CV TSC.

Architectural non-leakage & fairness invariants:
1. Feature extraction (`extract_decision_group_features`) does NOT accept `sumo_api`
   as an argument. It reads strictly from `PartialCvObserver`, `SafePhaseExecutor`,
   `TlsPhaseInfo`, and static 1-hop `neighbor_tls` topology. Ground-truth unequipped
   vehicle leakage is impossible by function signature.
2. All learned arms (`zero_shot_cfcmt_b0`, `pool_all_bfull`, `target_only`, `source_placebo`,
   `cfcmt`) use the EXACT SAME `EndpointAntisymmetricActionRanker` class, 78-dimensional
   endpoint-preserving pairwise representation `[x_g, phi_i - phi_j, 0.5*(phi_i + phi_j)]`,
   label scale floor, and HistGradientBoostingRegressor hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    GREEN_CHARS,
    PartialCvObserver,
)

ACTIVE_CHARS = frozenset("GgsO")

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "total_cv_halted_scaled_per_lane",
    "total_cv_veh_scaled_per_lane",
    "total_shockwave_q_per_lane",
    "total_cons_q_per_lane",
    "max_shockwave_q",
    "max_cons_q",
    "max_red_elapsed_norm",
    "mean_red_elapsed_norm",
    "current_green_elapsed_norm",
    "mean_ema_rate_60",
    "mean_ema_rate_300",
    "max_cv_wait_norm",
    "frac_lanes_with_cv_now",
    "candidate_count_norm",
    "lane_count_norm",
    "neighbor_up_cons_q_mean",
    "neighbor_down_cons_q_mean",
    "p_cv",
)

ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "cons_gp_rel_to_best",
    "shock_gp_rel_to_best",
    "obs_gp_rel_to_best",
    "hidden_wait_share",
    "arrival_flow_share",
    "green_lane_ratio",
    "green_cv_halted_per_lane",
    "green_cv_veh_per_lane",
    "green_cv_slow_per_lane",
    "green_shockwave_q_per_lane",
    "green_shockwave_q_max",
    "green_cons_q_per_lane",
    "green_cons_q_max",
    "green_spillback_frac_max",
    "green_cv_wait_mean_norm",
    "green_cv_wait_max_norm",
    "green_red_age_mean_norm",
    "green_red_age_max_norm",
    "green_hidden_arrivals_per_lane",
    "green_quad_red_wait_per_lane",
    "green_mean_speed_ratio",
    "serviceable_discharge_10s_per_lane",
    "red_cons_q_per_lane",
    "red_age_max_norm",
    "red_quad_wait_max_norm",
    "down_occ_scaled_mean",
    "down_shockwave_q_mean",
    "switch_indicator",
    "clearance_sec_norm",
    "green_hold_age_if_stay",
)


@dataclass(frozen=True)
class ActionGroupRecord:
    """One matched counterfactual decision state with K_g >= 2 feasible phases."""

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
    branch_costs: np.ndarray  # shape (K_g,), 1-hop light-cone waiting cost


def build_1hop_tls_neighbors(
    infos: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Construct 1-hop upstream/downstream TLS neighbors and 1-hop causal light-cone lanes."""
    lane_to_controlling_tls: dict[str, set[str]] = {}
    tls_out_lanes: dict[str, set[str]] = {}
    for tls_id, info in infos.items():
        tid = str(tls_id)
        for lane in info.incoming_lanes:
            lane_to_controlling_tls.setdefault(str(lane), set()).add(tid)
        outs: set[str] = set()
        for link_group in info.controlled_links:
            if link_group and link_group[0] and len(link_group[0]) >= 2 and link_group[0][1]:
                outs.add(str(link_group[0][1]))
        tls_out_lanes[tid] = outs

    up_neighbors: dict[str, set[str]] = {str(t): set() for t in infos}
    down_neighbors: dict[str, set[str]] = {str(t): set() for t in infos}

    edge_to_controlling_tls: dict[str, set[str]] = {}
    for lane, tids in lane_to_controlling_tls.items():
        edge = lane.rsplit("_", 1)[0]
        edge_to_controlling_tls.setdefault(edge, set()).update(tids)

    for tid, outs in tls_out_lanes.items():
        for out_lane in outs:
            out_edge = out_lane.rsplit("_", 1)[0]
            for down_tid in edge_to_controlling_tls.get(out_edge, ()):
                if down_tid != tid:
                    down_neighbors[tid].add(down_tid)
                    up_neighbors[down_tid].add(tid)

    light_cone_lanes: dict[str, tuple[str, ...]] = {}
    for tls_id, info in infos.items():
        tid = str(tls_id)
        cone = set(str(l) for l in info.incoming_lanes)
        cone.update(tls_out_lanes[tid])
        for down_tid in down_neighbors[tid]:
            cone.update(str(l) for l in infos[down_tid].incoming_lanes)
        light_cone_lanes[tid] = tuple(sorted(cone))

    return (
        {k: tuple(sorted(v)) for k, v in up_neighbors.items()},
        {k: tuple(sorted(v)) for k, v in down_neighbors.items()},
        light_cone_lanes,
    )


def extract_decision_group_features(
    *,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
    infos_all: Mapping[str, Any],
    up_neighbors: Sequence[str],
    down_neighbors: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Extract (feasible_states, state_vec[18], action_mat[K, 30]) purely from CV observations."""
    feasible = tuple(str(s) for s in executor.feasible_states_now())
    obs_map = observer.observations
    p_cv = observer.p_cv

    in_lanes = tuple(str(l) for l in info.incoming_lanes)
    n_in = max(len(in_lanes), 1)
    in_obs = [obs_map[l] for l in in_lanes]

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
    for cand_state in feasible:
        green_set: set[str] = set()
        for link_idx, char in enumerate(cand_state):
            if char in GREEN_CHARS and link_idx < len(info.controlled_links):
                lg = info.controlled_links[link_idx]
                if lg and lg[0]:
                    green_set.add(str(lg[0][0]))
        g_obs = [obs_map[l] for l in in_lanes if l in green_set]
        r_obs = [obs_map[l] for l in in_lanes if l not in green_set]

        down_occs_raw: list[float] = []
        down_occs_scaled: list[float] = []
        down_sws: list[float] = []
        for l in in_lanes:
            if l not in green_set:
                continue
            outs = observer.in_to_out_lanes.get(l, ())
            if outs:
                l_raw = []
                l_scaled = []
                l_sw = []
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

    return feasible, state_vec, np.vstack(action_rows)


def normalized_group_costs(
    branch_costs: np.ndarray, scale_floor: float = 0.15
) -> tuple[np.ndarray, float]:
    """Soft-floored group range normalization so trivial noise groups stay small."""
    costs = np.asarray(branch_costs, dtype=np.float64)
    span = float(np.max(costs) - np.min(costs))
    scale = max(span, float(scale_floor))
    return (costs - float(np.mean(costs))) / scale, span


class EndpointAntisymmetricActionRanker:
    """Identical-architecture pairwise action-cost predictor for all experimental arms.

    Pair feature vector u_ij = [x_g, phi_i - phi_j, 0.5*(phi_i + phi_j)] in R^78.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 120,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 10.0,
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
        groups: Sequence[ActionGroupRecord],
        *,
        domain_weights: Mapping[str, float] | None = None,
    ) -> "EndpointAntisymmetricActionRanker":
        if not groups:
            raise ValueError("Cannot fit EndpointAntisymmetricActionRanker on empty groups")

        domain_counts: dict[str, int] = {}
        for g in groups:
            if len(g.candidate_states) >= 2:
                domain_counts[g.domain] = domain_counts.get(g.domain, 0) + 1
        if not domain_counts:
            raise ValueError("No multi-action groups provided for fitting")

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        w_rows: list[float] = []

        for g in groups:
            k_actions = len(g.candidate_states)
            if k_actions < 2:
                continue
            z, span = normalized_group_costs(g.branch_costs, self.scale_floor)
            dom_total = domain_counts[g.domain]
            dom_mass = (
                float(domain_weights[g.domain])
                if domain_weights is not None and g.domain in domain_weights
                else 1.0 / len(domain_counts)
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
                    target = float(z[i] - z[j])
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

        n_groups = sum(domain_counts.values())
        n_rows = int(x_arr.shape[0])
        eff_min_leaf = max(6, min(self.min_samples_leaf, max(n_rows // 6, 6)))
        group_leaf_cap = 4 if n_groups <= 12 else self.max_leaf_nodes
        eff_max_leaves = max(2, min(group_leaf_cap, max(n_rows // eff_min_leaf, 2)))

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

    def score_group(self, state_vec: np.ndarray, action_mat: np.ndarray) -> np.ndarray:
        """Return antisymmetric Borda cost scores (lower = better phase)."""
        if self.model_ is None:
            raise RuntimeError("Ranker is not fitted")
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

        scores = np.zeros(k_actions, dtype=np.float64)
        for (i, j), d_ij in zip(pairs, deltas):
            scores[i] += d_ij
            scores[j] -= d_ij
        scores /= float(k_actions - 1)
        tie_ref = action_mat[:, 0] - float(np.mean(action_mat[:, 0]))
        return scores - 1e-7 * tie_ref


class BlendedActionRanker:
    """Convex score-space combination (1 - omega)*target_scale*target_ranker + omega*source_ranker."""

    def __init__(
        self,
        *,
        target_ranker: EndpointAntisymmetricActionRanker | None,
        source_ranker: EndpointAntisymmetricActionRanker | None,
        omega: float,
        target_scale: float = 1.0,
    ) -> None:
        self.target_ranker = target_ranker
        self.source_ranker = source_ranker
        self.omega = float(np.clip(omega, 0.0, 1.0))
        self.target_scale = float(target_scale)

    def score_group(self, state_vec: np.ndarray, action_mat: np.ndarray) -> np.ndarray:
        if self.omega <= 0.0 or self.source_ranker is None:
            assert self.target_ranker is not None
            return self.target_ranker.score_group(state_vec, action_mat)
        if self.omega >= 1.0 or self.target_ranker is None:
            assert self.source_ranker is not None
            return self.source_ranker.score_group(state_vec, action_mat)
        s_t = self.target_scale * self.target_ranker.score_group(state_vec, action_mat)
        s_s = self.source_ranker.score_group(state_vec, action_mat)
        return (1.0 - self.omega) * s_t + self.omega * s_s


class SimplexBlendedActionRanker:
    """CFCMT simplex source-mixture + target combination:
    S(s, a) = (1 - omega) * S_target(s, a) + omega * sum_m alpha_m * S_{source, m}(s, a).
    """

    def __init__(
        self,
        *,
        target_ranker: EndpointAntisymmetricActionRanker | None,
        source_rankers: Mapping[str, EndpointAntisymmetricActionRanker],
        alpha_weights: Mapping[str, float],
        omega: float,
    ) -> None:
        self.target_ranker = target_ranker
        self.omega = float(np.clip(omega, 0.0, 1.0))
        active = {
            k: float(w)
            for k, w in alpha_weights.items()
            if float(w) > 1e-9 and k in source_rankers
        }
        tot = sum(active.values())
        self.active_sources: tuple[tuple[EndpointAntisymmetricActionRanker, float], ...] = (
            tuple((source_rankers[k], w / tot) for k, w in sorted(active.items()))
            if tot > 0.0
            else ()
        )

    def score_group(self, state_vec: np.ndarray, action_mat: np.ndarray) -> np.ndarray:
        if self.omega <= 0.0 or not self.active_sources:
            assert self.target_ranker is not None
            return self.target_ranker.score_group(state_vec, action_mat)
        s_src = np.zeros(action_mat.shape[0], dtype=np.float64)
        for ranker, w in self.active_sources:
            s_src += w * ranker.score_group(state_vec, action_mat)
        if self.omega >= 1.0 or self.target_ranker is None:
            return s_src
        s_tgt = self.target_ranker.score_group(state_vec, action_mat)
        return (1.0 - self.omega) * s_tgt + self.omega * s_src


def evaluate_offline_regret(
    ranker: Any, groups: Sequence[ActionGroupRecord], scale_floor: float = 0.15
) -> dict[str, float]:
    """Compute mean normalized and physical regret on held-out counterfactual groups."""
    norm_regrets: list[float] = []
    phys_regrets: list[float] = []
    top1_hits = 0
    for g in groups:
        if len(g.candidate_states) < 2:
            continue
        scores = ranker.score_group(g.state_vec, g.action_mat)
        chosen = int(np.argmin(scores))
        min_cost = float(np.min(g.branch_costs))
        span = max(float(np.max(g.branch_costs) - min_cost), scale_floor)
        phys_reg = float(g.branch_costs[chosen] - min_cost)
        phys_regrets.append(phys_reg)
        norm_regrets.append(phys_reg / span)
        if abs(phys_reg) <= 1e-9:
            top1_hits += 1
    n = max(len(norm_regrets), 1)
    return {
        "group_count": len(norm_regrets),
        "mean_normalized_regret": float(np.mean(norm_regrets)) if norm_regrets else 0.0,
        "mean_physical_regret": float(np.mean(phys_regrets)) if phys_regrets else 0.0,
        "top1_accuracy": float(top1_hits / n),
    }


def permute_source_groups_placebo(
    source_groups: Sequence[ActionGroupRecord], rng_seed: int = 20261118
) -> list[ActionGroupRecord]:
    """Within-group label permutation control preserving exact feature & cost marginals."""
    rng = np.random.default_rng(rng_seed)
    permuted: list[ActionGroupRecord] = []
    for g in source_groups:
        perm_costs = g.branch_costs.copy()
        rng.shuffle(perm_costs)
        permuted.append(
            ActionGroupRecord(
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
                branch_costs=perm_costs,
            )
        )
    return permuted
