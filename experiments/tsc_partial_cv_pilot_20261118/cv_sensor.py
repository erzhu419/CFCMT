"""Realistic Connected-Vehicle (CV) partial-observability sensor layer for SUMO TSC.

Guarantees:
1. Strict vehicle-level persistence: each vehicle ID is deterministically assigned
   equipped status via blake2b(seed, veh_id) < p_cv, with zero SUMO RNG calls.
2. Strict non-leakage: unequipped vehicles are never queried for speed, position,
   halting status, or occupancy.
3. Causal 1-Hz temporal state: tracks ONLY past/current CV observations and the
   controller's own signal state (elapsed red/green, rolling CV arrival rate,
   red-phase shockwave tail distance, and conservation queue estimate).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

GREEN_CHARS = frozenset("Gg")
JAM_SPACING_M = 7.5
SATURATION_FLOW_VEH_PER_SEC = 0.45
PRIOR_ARRIVAL_RATE_VEH_PER_SEC = 0.04


def is_equipped_cv(seed: int, veh_id: str, p_cv: float) -> bool:
    """Deterministic, trip-persistent CV equipment assignment."""
    if p_cv >= 1.0:
        return True
    if p_cv <= 0.0:
        return False
    digest = hashlib.blake2b(
        f"{int(seed)}:cv:{veh_id}".encode("utf-8"), digest_size=8
    ).digest()
    u64 = int.from_bytes(digest, byteorder="big", signed=False)
    return (u64 / 18446744073709551616.0) < float(p_cv)


@dataclass(frozen=True)
class LaneCvObservation:
    """Causal per-lane observation summary at the current decision epoch."""

    lane_id: str
    lane_length_m: float
    speed_limit_mps: float
    # Instantaneous CV measurements (unscaled raw CV counts + scaled proxies)
    cv_veh_count: float
    cv_halted_count: float
    cv_slow_count: float
    cv_mean_speed_mps: float
    cv_occ_pct: float
    cv_max_halt_dist_m: float
    cv_max_slow_dist_m: float
    cv_wait_sec_sum: float
    cv_wait_sec_max: float
    # Causal temporal & signal-timing state
    is_green_now: bool
    red_elapsed_sec: float
    green_elapsed_sec: float
    sec_since_last_cv: float
    red_window_max_halt_dist_m: float
    ema_arrival_rate_60s: float
    ema_arrival_rate_300s: float
    shockwave_queue_est: float
    conservation_queue_est: float


class PartialCvObserver:
    """1-Hz causal partial-CV observer across all signal-controlled and outgoing lanes."""

    def __init__(
        self,
        *,
        sumo_api: Any,
        infos: Mapping[str, Any],
        seed: int,
        p_cv: float,
    ) -> None:
        if not (0.0 < float(p_cv) <= 1.0):
            raise ValueError(f"p_cv must be in (0, 1], got {p_cv}")
        self.seed = int(seed)
        self.p_cv = float(p_cv)
        self._equipped_cache: dict[str, bool] = {}

        incoming_set: set[str] = set()
        outgoing_set: set[str] = set()
        self.tls_lane_green_links: dict[str, list[tuple[str, int]]] = {}
        self.in_to_out_lanes: dict[str, tuple[str, ...]] = {}
        in_to_out_lists: dict[str, list[str]] = {}

        for tls_id, info in infos.items():
            for link_idx, link_group in enumerate(info.controlled_links):
                if not link_group or not link_group[0]:
                    continue
                in_lane = str(link_group[0][0])
                out_lane = str(link_group[0][1])
                incoming_set.add(in_lane)
                if out_lane:
                    outgoing_set.add(out_lane)
                    in_to_out_lists.setdefault(in_lane, [])
                    if out_lane not in in_to_out_lists[in_lane]:
                        in_to_out_lists[in_lane].append(out_lane)
                self.tls_lane_green_links.setdefault(in_lane, []).append(
                    (str(tls_id), int(link_idx))
                )

        self.incoming_lanes: tuple[str, ...] = tuple(sorted(incoming_set))
        self.tracked_lanes: tuple[str, ...] = tuple(sorted(incoming_set | outgoing_set))
        self.in_to_out_lanes = {
            lane: tuple(outs) for lane, outs in in_to_out_lists.items()
        }

        self.lane_length: dict[str, float] = {}
        self.lane_speed_limit: dict[str, float] = {}
        for lane in self.tracked_lanes:
            try:
                length = max(float(sumo_api.lane.getLength(lane)), 5.0)
            except Exception:
                length = 200.0
            try:
                vmax = max(float(sumo_api.lane.getMaxSpeed(lane)), 1.0)
            except Exception:
                vmax = 11.11
            self.lane_length[lane] = length
            self.lane_speed_limit[lane] = vmax

        # Causal temporal state per tracked lane
        self._seen_cvs_on_lane: dict[str, set[str]] = {
            lane: set() for lane in self.tracked_lanes
        }
        self._red_elapsed: dict[str, float] = {lane: 0.0 for lane in self.tracked_lanes}
        self._green_elapsed: dict[str, float] = {lane: 0.0 for lane in self.tracked_lanes}
        self._sec_since_cv: dict[str, float] = {lane: 60.0 for lane in self.tracked_lanes}
        self._red_max_halt_dist: dict[str, float] = {
            lane: 0.0 for lane in self.tracked_lanes
        }
        self._ema_rate_60: dict[str, float] = {
            lane: PRIOR_ARRIVAL_RATE_VEH_PER_SEC for lane in self.tracked_lanes
        }
        self._ema_rate_300: dict[str, float] = {
            lane: PRIOR_ARRIVAL_RATE_VEH_PER_SEC for lane in self.tracked_lanes
        }
        self._cons_queue: dict[str, float] = {lane: 0.0 for lane in self.tracked_lanes}
        self._cv_halt_wait_sec: dict[str, float] = {}
        self._latest_obs: dict[str, LaneCvObservation] = {}

    def is_cv(self, veh_id: str) -> bool:
        res = self._equipped_cache.get(veh_id)
        if res is None:
            res = is_equipped_cv(self.seed, veh_id, self.p_cv)
            self._equipped_cache[veh_id] = res
        return res

    def observe_step(self, sumo_api: Any, dt_sec: float = 1.0) -> None:
        """Update 1-Hz causal observation state using ONLY equipped CVs + TLS state."""
        dt = max(float(dt_sec), 1e-3)
        alpha_60 = 1.0 - float(np.exp(-dt / 60.0))
        alpha_300 = 1.0 - float(np.exp(-dt / 300.0))

        tls_states: dict[str, str] = {}
        for lane_links in self.tls_lane_green_links.values():
            for tls_id, _ in lane_links:
                if tls_id not in tls_states:
                    tls_states[tls_id] = str(
                        sumo_api.trafficlight.getRedYellowGreenState(tls_id)
                    )

        for lane in self.tracked_lanes:
            length = self.lane_length[lane]
            vmax = self.lane_speed_limit[lane]

            # Determine if lane currently has any green link
            links = self.tls_lane_green_links.get(lane, ())
            if links:
                is_green = any(
                    link_idx < len(tls_states[tls_id])
                    and tls_states[tls_id][link_idx] in GREEN_CHARS
                    for tls_id, link_idx in links
                )
            else:
                is_green = True

            if is_green:
                self._green_elapsed[lane] += dt
                self._red_elapsed[lane] = 0.0
                # Discharge wave moves shockwave tail forward during green
                self._red_max_halt_dist[lane] = max(
                    0.0,
                    self._red_max_halt_dist[lane]
                    - SATURATION_FLOW_VEH_PER_SEC * JAM_SPACING_M * dt,
                )
            else:
                self._red_elapsed[lane] += dt
                self._green_elapsed[lane] = 0.0

            # Filter strictly to equipped CVs on this lane
            raw_vids: Sequence[str] = sumo_api.lane.getLastStepVehicleIDs(lane)
            cv_ids = [vid for vid in raw_vids if self.is_cv(vid)]

            seen_set = self._seen_cvs_on_lane[lane]
            new_cv_count = 0
            for vid in cv_ids:
                if vid not in seen_set:
                    seen_set.add(vid)
                    new_cv_count += 1
            # Prevent unbounded set growth while keeping active IDs
            if len(seen_set) > 2048:
                seen_set.intersection_update(cv_ids)

            inst_arrival_rate = (new_cv_count / self.p_cv) / dt
            self._ema_rate_60[lane] = (
                (1.0 - alpha_60) * self._ema_rate_60[lane]
                + alpha_60 * inst_arrival_rate
            )
            self._ema_rate_300[lane] = (
                (1.0 - alpha_300) * self._ema_rate_300[lane]
                + alpha_300 * inst_arrival_rate
            )

            cv_veh = float(len(cv_ids))
            cv_halted = 0.0
            cv_slow = 0.0
            speed_sum = 0.0
            max_halt_dist = 0.0
            max_slow_dist = 0.0
            wait_sum = 0.0
            wait_max = 0.0
            free_flow_near_stopline = False

            for vid in cv_ids:
                spd = float(sumo_api.vehicle.getSpeed(vid))
                pos = float(sumo_api.vehicle.getLanePosition(vid))
                dist_to_stop = max(0.0, length - pos)
                speed_sum += spd
                if spd < 0.1:
                    cv_halted += 1.0
                    if dist_to_stop > max_halt_dist:
                        max_halt_dist = dist_to_stop
                    w = self._cv_halt_wait_sec.get(vid, 0.0) + dt
                    self._cv_halt_wait_sec[vid] = w
                elif spd > 2.5:
                    w = 0.0
                    self._cv_halt_wait_sec[vid] = 0.0
                else:
                    w = self._cv_halt_wait_sec.get(vid, 0.0)
                wait_sum += w
                if w > wait_max:
                    wait_max = w
                if spd < 3.0:
                    cv_slow += 1.0
                    if dist_to_stop > max_slow_dist:
                        max_slow_dist = dist_to_stop
                elif is_green and dist_to_stop < 20.0 and spd > 0.45 * vmax:
                    free_flow_near_stopline = True

            if cv_veh > 0.0:
                mean_speed = speed_sum / cv_veh
                self._sec_since_cv[lane] = 0.0
            else:
                mean_speed = vmax
                self._sec_since_cv[lane] += dt

            # Each CV occupies ~7.5m effective cell (5m vehicle + 2.5m gap)
            cv_occ_pct = min(100.0, 100.0 * (cv_veh * 5.0) / max(length, 5.0))

            if max_halt_dist > self._red_max_halt_dist[lane]:
                self._red_max_halt_dist[lane] = max_halt_dist
            if free_flow_near_stopline and cv_halted == 0.0:
                self._red_max_halt_dist[lane] = 0.0

            # Instantaneous + short-memory shockwave queue estimate (Comert & Cetin)
            shockwave_q = max(
                cv_halted / self.p_cv,
                max_halt_dist / JAM_SPACING_M,
                self._red_max_halt_dist[lane] / JAM_SPACING_M,
            )

            # Causal arrival-service conservation queue estimate
            cons_q = (
                self._cons_queue[lane]
                + self._ema_rate_300[lane] * dt
                - (SATURATION_FLOW_VEH_PER_SEC * dt if is_green else 0.0)
            )
            cons_q = max(0.0, cons_q)
            if free_flow_near_stopline and cv_halted == 0.0:
                cons_q = min(cons_q, cv_slow / self.p_cv)
            if max_halt_dist > 0.0:
                cons_q = max(cons_q, max_halt_dist / JAM_SPACING_M)
            cons_q = min(cons_q, length / JAM_SPACING_M)
            self._cons_queue[lane] = cons_q

            self._latest_obs[lane] = LaneCvObservation(
                lane_id=lane,
                lane_length_m=length,
                speed_limit_mps=vmax,
                cv_veh_count=cv_veh,
                cv_halted_count=cv_halted,
                cv_slow_count=cv_slow,
                cv_mean_speed_mps=mean_speed,
                cv_occ_pct=cv_occ_pct,
                cv_max_halt_dist_m=max_halt_dist,
                cv_max_slow_dist_m=max_slow_dist,
                cv_wait_sec_sum=wait_sum,
                cv_wait_sec_max=wait_max,
                is_green_now=is_green,
                red_elapsed_sec=self._red_elapsed[lane],
                green_elapsed_sec=self._green_elapsed[lane],
                sec_since_last_cv=self._sec_since_cv[lane],
                red_window_max_halt_dist_m=self._red_max_halt_dist[lane],
                ema_arrival_rate_60s=self._ema_rate_60[lane],
                ema_arrival_rate_300s=self._ema_rate_300[lane],
                shockwave_queue_est=shockwave_q,
                conservation_queue_est=cons_q,
            )

    @property
    def observations(self) -> Mapping[str, LaneCvObservation]:
        return self._latest_obs


def select_observed_pressure_phase(
    *,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
    mode: str,
) -> str:
    """Evaluate pressure baselines strictly from partial-CV observations.

    Supported modes:
    - 'observed_pp': PhasePressure on raw observed CV queue proxy
    - 'observed_mp': MaxPressure on raw observed CV queue proxy (upstream - downstream)
    - 'observed_gp': Exact GP formula (q_green - 0.02 * o_down) on raw observed CV
    - 'shockwave_gp': Shockwave-reconstructed queue + scaled downstream occupancy GP
    - 'conservation_gp': Hybrid shockwave + conservation queue GP
    """
    feasible = executor.feasible_states_now()
    if len(feasible) == 1:
        return str(feasible[0])

    obs_map = observer.observations
    p_cv = observer.p_cv
    best_state = str(feasible[0])
    best_score = -float("inf")

    for state in feasible:
        green_lanes: list[str] = []
        for link_idx, char in enumerate(state):
            if char not in GREEN_CHARS or link_idx >= len(info.controlled_links):
                continue
            link_group = info.controlled_links[link_idx]
            if not link_group or not link_group[0]:
                continue
            in_lane = str(link_group[0][0])
            if in_lane not in green_lanes:
                green_lanes.append(in_lane)

        if not green_lanes:
            score = 0.0
        else:
            q_green = 0.0
            down_q_vals: list[float] = []
            down_occ_vals: list[float] = []
            for in_lane in green_lanes:
                obs = obs_map[in_lane]
                raw_q_cv = (
                    obs.cv_halted_count
                    + 0.35 * obs.cv_veh_count
                    + 0.025 * obs.cv_occ_pct
                )
                if mode in {"observed_pp", "observed_mp", "observed_gp"}:
                    lane_q = raw_q_cv
                elif mode == "shockwave_gp":
                    lane_q = (
                        obs.shockwave_queue_est
                        + 0.35 * max(0.0, obs.cv_veh_count / p_cv - obs.cv_halted_count / p_cv)
                        + 0.025 * min(100.0, obs.cv_occ_pct / p_cv)
                    )
                elif mode == "conservation_gp":
                    lane_q = (
                        max(obs.shockwave_queue_est, obs.conservation_queue_est)
                        + 0.35 * max(0.0, obs.cv_veh_count / p_cv - obs.cv_halted_count / p_cv)
                        + 0.025 * min(100.0, obs.cv_occ_pct / p_cv)
                    )
                else:
                    raise ValueError(f"Unknown observed pressure mode: {mode!r}")
                q_green += lane_q

                outs = observer.in_to_out_lanes.get(in_lane, ())
                if outs:
                    out_qs = []
                    out_occs = []
                    for out_lane in outs:
                        out_obs = obs_map.get(out_lane)
                        if out_obs is None:
                            continue
                        if mode in {"observed_pp", "observed_mp", "observed_gp"}:
                            out_qs.append(
                                out_obs.cv_halted_count
                                + 0.35 * out_obs.cv_veh_count
                                + 0.025 * out_obs.cv_occ_pct
                            )
                            out_occs.append(out_obs.cv_occ_pct)
                        else:
                            out_qs.append(out_obs.shockwave_queue_est)
                            out_occs.append(min(100.0, out_obs.cv_occ_pct / p_cv))
                    if out_qs:
                        down_q_vals.append(float(np.mean(out_qs)))
                    if out_occs:
                        down_occ_vals.append(float(np.mean(out_occs)))

            if mode == "observed_pp":
                score = q_green
            elif mode == "observed_mp":
                q_down = float(np.sum(down_q_vals)) if down_q_vals else 0.0
                score = q_green - q_down
            elif mode in {"observed_gp", "shockwave_gp", "conservation_gp"}:
                o_down = float(np.mean(down_occ_vals)) if down_occ_vals else 0.0
                score = q_green - 0.02 * o_down
            else:
                raise ValueError(f"Unknown observed pressure mode: {mode!r}")

        # Exact first-maximum tie-breaking rule matching GP/PP in manuscript
        if score > best_score + 1e-12:
            best_score = score
            best_state = str(state)

    return best_state
