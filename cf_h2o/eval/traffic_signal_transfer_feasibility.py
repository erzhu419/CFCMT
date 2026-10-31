"""Phase-0 feasibility test for cross-network traffic signal transfer.

The bus-holding experiments have a strong classical-control ceiling.  This
script tests whether the CFCMT idea has more room in a traffic-signal-control
setting before committing to a full SUMO implementation.

This is intentionally a small analytic queueing benchmark, not a SUMO result.
It keeps the important transfer constraints:

* leave-one-network-out source/target split;
* zero-shot methods see no target next-state labels;
* the few-shot variant only sees passive target transitions under a behavior
  controller, then adapts a mechanism bias;
* model-based policies choose among a small green-split action grid by one-step
  MPC under their predicted next queues.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MOVEMENTS = ("north", "south", "east", "west")
ACTION_GRID_DEFAULT = (0.15, 0.3, 0.5, 0.7, 0.85)


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    arrival_base: tuple[float, float, float, float]
    arrival_amp: tuple[float, float, float, float]
    peak_phase: tuple[float, float, float, float]
    saturation: tuple[float, float, float, float]
    downstream_base: tuple[float, float, float, float]
    downstream_amp: tuple[float, float, float, float]
    discharge_bias: tuple[float, float, float, float]
    spillback_strength: float
    burstiness: float
    lost_time_sensitivity: float


@dataclass(frozen=True)
class SignalBatch:
    t: np.ndarray
    q: np.ndarray
    arrivals: np.ndarray
    downstream: np.ndarray
    saturation: np.ndarray
    summary: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    beta: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.beta


@dataclass(frozen=True)
class MechanismModels:
    queue_models: tuple[RidgeModel, RidgeModel, RidgeModel, RidgeModel]


@dataclass(frozen=True)
class FittedModels:
    dense: RidgeModel
    cfcmt_global: MechanismModels
    cfcmt_by_source: dict[str, MechanismModels]
    source_summaries: dict[str, np.ndarray]


def _network_specs() -> list[NetworkSpec]:
    return [
        NetworkSpec(
            name="grid_balanced",
            arrival_base=(7.0, 6.8, 6.2, 6.1),
            arrival_amp=(4.5, 4.2, 3.8, 3.5),
            peak_phase=(0.30, 0.31, 0.55, 0.56),
            saturation=(17.5, 17.0, 15.8, 15.5),
            downstream_base=(0.28, 0.30, 0.32, 0.31),
            downstream_amp=(0.18, 0.16, 0.20, 0.19),
            discharge_bias=(1.00, 0.97, 0.95, 0.96),
            spillback_strength=0.48,
            burstiness=0.18,
            lost_time_sensitivity=0.13,
        ),
        NetworkSpec(
            name="arterial_peak",
            arrival_base=(9.5, 9.0, 4.2, 4.5),
            arrival_amp=(8.0, 7.5, 2.6, 2.8),
            peak_phase=(0.34, 0.35, 0.66, 0.68),
            saturation=(21.0, 20.5, 13.5, 13.0),
            downstream_base=(0.38, 0.36, 0.24, 0.25),
            downstream_amp=(0.28, 0.27, 0.16, 0.16),
            discharge_bias=(1.10, 1.08, 0.88, 0.86),
            spillback_strength=0.58,
            burstiness=0.30,
            lost_time_sensitivity=0.18,
        ),
        NetworkSpec(
            name="downtown_spillback",
            arrival_base=(6.8, 6.4, 7.5, 7.1),
            arrival_amp=(4.8, 4.5, 5.4, 5.0),
            peak_phase=(0.41, 0.42, 0.47, 0.48),
            saturation=(15.5, 15.0, 14.6, 14.2),
            downstream_base=(0.46, 0.48, 0.58, 0.56),
            downstream_amp=(0.32, 0.31, 0.34, 0.33),
            discharge_bias=(0.92, 0.90, 0.84, 0.82),
            spillback_strength=0.88,
            burstiness=0.38,
            lost_time_sensitivity=0.21,
        ),
        NetworkSpec(
            name="suburban_asymmetric",
            arrival_base=(4.2, 4.0, 5.8, 3.2),
            arrival_amp=(2.4, 2.2, 5.2, 1.8),
            peak_phase=(0.22, 0.24, 0.72, 0.70),
            saturation=(18.5, 17.8, 16.2, 12.5),
            downstream_base=(0.18, 0.20, 0.34, 0.22),
            downstream_amp=(0.12, 0.14, 0.28, 0.15),
            discharge_bias=(1.03, 1.01, 0.92, 0.78),
            spillback_strength=0.40,
            burstiness=0.22,
            lost_time_sensitivity=0.10,
        ),
        NetworkSpec(
            name="event_reversal",
            arrival_base=(5.5, 5.2, 8.4, 8.1),
            arrival_amp=(5.0, 4.8, 8.2, 7.8),
            peak_phase=(0.78, 0.79, 0.27, 0.28),
            saturation=(14.8, 14.6, 18.8, 18.3),
            downstream_base=(0.42, 0.40, 0.36, 0.38),
            downstream_amp=(0.24, 0.25, 0.30, 0.30),
            discharge_bias=(0.82, 0.84, 1.06, 1.03),
            spillback_strength=0.70,
            burstiness=0.46,
            lost_time_sensitivity=0.19,
        ),
    ]


def _summary(spec: NetworkSpec) -> np.ndarray:
    base = np.asarray(spec.arrival_base, dtype=float)
    amp = np.asarray(spec.arrival_amp, dtype=float)
    sat = np.asarray(spec.saturation, dtype=float)
    downstream = np.asarray(spec.downstream_base, dtype=float)
    ns_demand = float(np.mean(base[:2] + 0.5 * amp[:2]))
    ew_demand = float(np.mean(base[2:] + 0.5 * amp[2:]))
    return np.asarray(
        [
            float(np.mean(base + 0.5 * amp)) / 14.0,
            (ns_demand - ew_demand) / 14.0,
            float(np.mean(sat)) / 18.0,
            float(np.std(sat)) / 6.0,
            float(np.mean(downstream)),
            float(spec.spillback_strength),
            float(spec.burstiness),
        ],
        dtype=float,
    )


def _as_state_dict(batch: SignalBatch) -> dict[str, np.ndarray]:
    return {
        "t": batch.t,
        "q": batch.q,
        "arrivals": batch.arrivals,
        "downstream": batch.downstream,
        "saturation": batch.saturation,
        "summary": batch.summary,
    }


def _movement_green(g_ns: float) -> np.ndarray:
    return np.asarray([g_ns, g_ns, 1.0 - g_ns, 1.0 - g_ns], dtype=float)


def _daily_wave(t: np.ndarray, phase: float) -> np.ndarray:
    return 0.5 + 0.5 * np.sin(2.0 * math.pi * (t - phase))


def _sample_batch(spec: NetworkSpec, n: int, rng: np.random.Generator) -> SignalBatch:
    t = rng.uniform(0.0, 1.0, size=n)
    base = np.asarray(spec.arrival_base, dtype=float)
    amp = np.asarray(spec.arrival_amp, dtype=float)
    phase = np.asarray(spec.peak_phase, dtype=float)
    waves = np.stack([_daily_wave(t, float(phase_i)) for phase_i in phase], axis=1)
    deterministic_arrivals = base + amp * np.power(waves, 1.35)
    burst = rng.gamma(shape=2.0, scale=spec.burstiness, size=(n, 4))
    arrivals = np.maximum(0.1, deterministic_arrivals * (0.86 + burst))

    downstream_base = np.asarray(spec.downstream_base, dtype=float)
    downstream_amp = np.asarray(spec.downstream_amp, dtype=float)
    downstream_wave = np.stack([_daily_wave(t, float(phase_i + 0.08)) for phase_i in phase], axis=1)
    downstream_noise = rng.normal(0.0, 0.045 + 0.025 * spec.spillback_strength, size=(n, 4))
    downstream = np.clip(downstream_base + downstream_amp * downstream_wave + downstream_noise, 0.02, 0.96)

    sat = np.repeat(np.asarray(spec.saturation, dtype=float)[None, :], n, axis=0)
    congestion_scale = 1.6 + 1.9 * downstream + 0.08 * arrivals
    queue_noise = rng.gamma(shape=2.4, scale=4.2, size=(n, 4))
    q = np.clip(arrivals * congestion_scale + queue_noise, 0.0, 120.0)
    return SignalBatch(
        t=t,
        q=q,
        arrivals=arrivals,
        downstream=downstream,
        saturation=sat,
        summary=np.repeat(_summary(spec)[None, :], n, axis=0),
    )


def _real_next_queue(spec: NetworkSpec, batch: SignalBatch, g_ns: float) -> np.ndarray:
    green = _movement_green(g_ns)[None, :]
    q = batch.q
    arrivals = batch.arrivals
    downstream = batch.downstream
    sat = np.asarray(spec.saturation, dtype=float)[None, :]
    bias = np.asarray(spec.discharge_bias, dtype=float)[None, :]

    blocked = np.clip(1.0 - spec.spillback_strength * np.power(downstream, 1.25), 0.05, 1.0)
    lost_time = np.clip(1.0 - spec.lost_time_sensitivity * np.abs(g_ns - 0.5) / 0.3, 0.70, 1.0)
    effective_sat = sat * bias * blocked * lost_time

    # Serving a nearly blocked downstream link can return a fraction of flow as
    # local queue.  This captures spillback effects that simple pressure rules
    # often miss without a full network model.
    nominal_service = effective_sat * green * 2.35
    served = np.minimum(q + 0.55 * arrivals, nominal_service)
    returned = np.maximum(0.0, nominal_service - served) * downstream * spec.spillback_strength * 0.10

    turn_coupling = np.zeros_like(q)
    turn_coupling[:, 0] = 0.09 * served[:, 2] * downstream[:, 0]
    turn_coupling[:, 1] = 0.08 * served[:, 3] * downstream[:, 1]
    turn_coupling[:, 2] = 0.10 * served[:, 0] * downstream[:, 2]
    turn_coupling[:, 3] = 0.07 * served[:, 1] * downstream[:, 3]

    return np.clip(q + arrivals + turn_coupling + returned - served, 0.0, 180.0)


def _sim_next_queue(batch: SignalBatch, g_ns: float) -> np.ndarray:
    green = _movement_green(g_ns)[None, :]
    q = batch.q
    arrivals = batch.arrivals
    downstream = batch.downstream
    generic_sat = np.full((1, 4), 16.0, dtype=float)
    blocked = np.clip(1.0 - 0.34 * downstream, 0.22, 1.0)
    lost_time = np.clip(1.0 - 0.10 * np.abs(g_ns - 0.5) / 0.3, 0.82, 1.0)
    service = generic_sat * blocked * green * lost_time * 2.05
    served = np.minimum(q + 0.45 * arrivals, service)
    return np.clip(q + arrivals - served, 0.0, 180.0)


def _cost(batch: SignalBatch, q_next: np.ndarray, g_ns: float) -> np.ndarray:
    downstream = batch.downstream
    total_q = np.sum(q_next, axis=1)
    ns_q = np.sum(q_next[:, :2], axis=1)
    ew_q = np.sum(q_next[:, 2:], axis=1)
    spill = np.sum(q_next * np.power(downstream, 1.4), axis=1)
    storage_violation = np.sum(np.maximum(q_next - (70.0 - 30.0 * downstream), 0.0), axis=1)
    lost_time_penalty = 0.8 * np.abs(g_ns - 0.5)
    return total_q + 0.22 * spill + 0.10 * np.abs(ns_q - ew_q) + 0.35 * storage_violation + lost_time_penalty


def _feature_base(batch: SignalBatch, g_ns: float) -> np.ndarray:
    t_sin = np.sin(2.0 * math.pi * batch.t)[:, None]
    t_cos = np.cos(2.0 * math.pi * batch.t)[:, None]
    g = np.full((batch.q.shape[0], 1), g_ns, dtype=float)
    green = np.repeat(_movement_green(g_ns)[None, :], batch.q.shape[0], axis=0)
    q = batch.q / 80.0
    arrivals = batch.arrivals / 18.0
    saturation = batch.saturation / 20.0
    downstream = batch.downstream
    ns_q = np.sum(q[:, :2], axis=1, keepdims=True)
    ew_q = np.sum(q[:, 2:], axis=1, keepdims=True)
    ns_arr = np.sum(arrivals[:, :2], axis=1, keepdims=True)
    ew_arr = np.sum(arrivals[:, 2:], axis=1, keepdims=True)
    return np.concatenate(
        [
            np.ones((batch.q.shape[0], 1), dtype=float),
            t_sin,
            t_cos,
            q,
            arrivals,
            downstream,
            saturation,
            batch.summary,
            g,
            g * g,
            green,
            q * green,
            arrivals * green,
            downstream * green,
            ns_q - ew_q,
            ns_arr - ew_arr,
        ],
        axis=1,
    )


def _dense_features(batch: SignalBatch, g_ns: float) -> np.ndarray:
    base = _feature_base(batch, g_ns)
    q = batch.q / 80.0
    arrivals = batch.arrivals / 18.0
    downstream = batch.downstream
    cross = np.concatenate(
        [
            q * arrivals,
            q * downstream,
            arrivals * downstream,
            np.square(q),
            np.square(arrivals),
            np.square(downstream),
        ],
        axis=1,
    )
    return np.concatenate([base, cross], axis=1)


def _mechanism_features(batch: SignalBatch, g_ns: float, movement_idx: int) -> np.ndarray:
    n = batch.q.shape[0]
    green = _movement_green(g_ns)
    paired_idx = 1 - movement_idx if movement_idx < 2 else 5 - movement_idx
    same_phase = 0 if movement_idx < 2 else 1
    other_phase_idx = (2, 3) if same_phase == 0 else (0, 1)
    q = batch.q / 80.0
    arrivals = batch.arrivals / 18.0
    saturation = batch.saturation / 20.0
    downstream = batch.downstream
    own_green = np.full((n, 1), green[movement_idx], dtype=float)
    opp_green = np.full((n, 1), 1.0 - green[movement_idx], dtype=float)
    t_sin = np.sin(2.0 * math.pi * batch.t)[:, None]
    t_cos = np.cos(2.0 * math.pi * batch.t)[:, None]
    own_q = q[:, [movement_idx]]
    own_arr = arrivals[:, [movement_idx]]
    own_down = downstream[:, [movement_idx]]
    own_sat = saturation[:, [movement_idx]]
    pair_q = q[:, [paired_idx]]
    pair_arr = arrivals[:, [paired_idx]]
    other_q = np.sum(q[:, other_phase_idx], axis=1, keepdims=True)
    other_arr = np.sum(arrivals[:, other_phase_idx], axis=1, keepdims=True)
    return np.concatenate(
        [
            np.ones((n, 1), dtype=float),
            t_sin,
            t_cos,
            own_q,
            own_arr,
            own_down,
            own_sat,
            pair_q,
            pair_arr,
            other_q,
            other_arr,
            own_green,
            opp_green,
            own_green * own_q,
            own_green * own_arr,
            own_green * own_down,
            own_green * own_sat,
            own_q * own_down,
            own_arr * own_down,
            batch.summary[:, [0, 1, 2, 4, 5, 6]],
        ],
        axis=1,
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, *, l2: float) -> RidgeModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * float(l2)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return RidgeModel(beta=beta)


def _fit_mechanism(rows: list[tuple[SignalBatch, float, np.ndarray]], *, l2: float) -> MechanismModels:
    queue_models: list[RidgeModel] = []
    for movement_idx in range(4):
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for batch, g_ns, residual in rows:
            x_parts.append(_mechanism_features(batch, g_ns, movement_idx))
            y_parts.append(residual[:, movement_idx])
        queue_models.append(_ridge_fit(np.vstack(x_parts), np.concatenate(y_parts), l2=l2))
    return MechanismModels(queue_models=tuple(queue_models))  # type: ignore[arg-type]


def _predict_mechanism(model: MechanismModels, batch: SignalBatch, g_ns: float) -> np.ndarray:
    cols = [
        model.queue_models[movement_idx].predict(_mechanism_features(batch, g_ns, movement_idx))
        for movement_idx in range(4)
    ]
    return np.column_stack(cols)


def _source_weights(target_summary: np.ndarray, source_summaries: dict[str, np.ndarray]) -> dict[str, float]:
    target = np.asarray(target_summary, dtype=float)
    distances = {
        name: float(np.linalg.norm((target - summary) / np.asarray([0.35, 0.45, 0.25, 0.35, 0.30, 0.35, 0.30])))
        for name, summary in source_summaries.items()
    }
    logits = {name: -dist for name, dist in distances.items()}
    max_logit = max(logits.values())
    weights_raw = {name: math.exp(value - max_logit) for name, value in logits.items()}
    total = sum(weights_raw.values())
    return {name: value / total for name, value in weights_raw.items()}


def _build_action_rows(spec: NetworkSpec, batch: SignalBatch, action_grid: tuple[float, ...]) -> list[tuple[SignalBatch, float, np.ndarray]]:
    rows: list[tuple[SignalBatch, float, np.ndarray]] = []
    for g_ns in action_grid:
        real_q = _real_next_queue(spec, batch, g_ns)
        sim_q = _sim_next_queue(batch, g_ns)
        rows.append((batch, g_ns, real_q - sim_q))
    return rows


def _fit_models(
    *,
    train_specs: list[NetworkSpec],
    batches: dict[str, SignalBatch],
    action_grid: tuple[float, ...],
    dense_l2: float,
    mechanism_l2: float,
) -> FittedModels:
    dense_x: list[np.ndarray] = []
    dense_y: list[np.ndarray] = []
    mechanism_rows: list[tuple[SignalBatch, float, np.ndarray]] = []
    cfcmt_by_source: dict[str, MechanismModels] = {}
    source_summaries: dict[str, np.ndarray] = {}
    for spec in train_specs:
        batch = batches[spec.name]
        rows = _build_action_rows(spec, batch, action_grid)
        source_summaries[spec.name] = _summary(spec)
        mechanism_rows.extend(rows)
        cfcmt_by_source[spec.name] = _fit_mechanism(rows, l2=mechanism_l2)
        for row_batch, g_ns, residual in rows:
            dense_x.append(_dense_features(row_batch, g_ns))
            dense_y.append(residual)

    dense = _ridge_fit(np.vstack(dense_x), np.vstack(dense_y), l2=dense_l2)
    cfcmt_global = _fit_mechanism(mechanism_rows, l2=mechanism_l2)
    return FittedModels(
        dense=dense,
        cfcmt_global=cfcmt_global,
        cfcmt_by_source=cfcmt_by_source,
        source_summaries=source_summaries,
    )


def _predict_weighted_cfcmt(models: FittedModels, batch: SignalBatch, g_ns: float) -> np.ndarray:
    weights = _source_weights(batch.summary[0], models.source_summaries)
    pred = np.zeros_like(batch.q)
    for source_name, weight in weights.items():
        pred += weight * _predict_mechanism(models.cfcmt_by_source[source_name], batch, g_ns)
    return pred


def _choose_pressure(batch: SignalBatch, action_grid: tuple[float, ...], *, use_downstream: bool) -> np.ndarray:
    q = batch.q
    arrivals = batch.arrivals
    downstream = batch.downstream
    block = 1.0 - 0.75 * downstream if use_downstream else 1.0
    pressure = (q + 1.15 * arrivals) * block
    ns = np.sum(pressure[:, :2], axis=1)
    ew = np.sum(pressure[:, 2:], axis=1)
    chosen = np.full(batch.q.shape[0], _nearest_action(action_grid, 0.5), dtype=float)
    high = _nearest_action(action_grid, 0.8)
    mid_high = _nearest_action(action_grid, 0.65)
    low = _nearest_action(action_grid, 0.2)
    mid_low = _nearest_action(action_grid, 0.35)
    diff = ns - ew
    chosen[diff > 18.0] = high
    chosen[(diff > 6.0) & (diff <= 18.0)] = mid_high
    chosen[diff < -18.0] = low
    chosen[(diff < -6.0) & (diff >= -18.0)] = mid_low
    return chosen


def _nearest_action(action_grid: tuple[float, ...], value: float) -> float:
    return float(min(action_grid, key=lambda item: abs(item - value)))


def _subset(batch: SignalBatch, idx: np.ndarray) -> SignalBatch:
    return SignalBatch(
        t=batch.t[idx],
        q=batch.q[idx],
        arrivals=batch.arrivals[idx],
        downstream=batch.downstream[idx],
        saturation=batch.saturation[idx],
        summary=batch.summary[idx],
    )


def _predict_next_queue(
    *,
    method: str,
    models: FittedModels,
    batch: SignalBatch,
    g_ns: float,
    fewshot_bias: np.ndarray | None,
) -> np.ndarray:
    sim_q = _sim_next_queue(batch, g_ns)
    if method == "sim_mpc":
        return sim_q
    if method == "h2oplus_dense_mpc":
        residual = models.dense.predict(_dense_features(batch, g_ns))
    elif method == "cfcmt_global_mpc":
        residual = _predict_mechanism(models.cfcmt_global, batch, g_ns)
    elif method == "cfcmt_weighted_mpc":
        residual = _predict_weighted_cfcmt(models, batch, g_ns)
    elif method == "cfcmt_fewshot_bias_mpc":
        residual = _predict_mechanism(models.cfcmt_global, batch, g_ns)
        if method == "cfcmt_fewshot_bias_mpc" and fewshot_bias is not None:
            residual = residual + fewshot_bias[None, :]
    else:
        raise ValueError(f"unknown model policy: {method}")
    return np.clip(sim_q + residual, 0.0, 180.0)


def _choose_model_mpc(
    *,
    method: str,
    models: FittedModels,
    batch: SignalBatch,
    action_grid: tuple[float, ...],
    fewshot_bias: np.ndarray | None,
) -> np.ndarray:
    costs = []
    for g_ns in action_grid:
        pred_q = _predict_next_queue(
            method=method,
            models=models,
            batch=batch,
            g_ns=g_ns,
            fewshot_bias=fewshot_bias,
        )
        costs.append(_cost(batch, pred_q, g_ns))
    cost_matrix = np.column_stack(costs)
    return np.asarray(action_grid, dtype=float)[np.argmin(cost_matrix, axis=1)]


def _oracle_best_actions(spec: NetworkSpec, batch: SignalBatch, action_grid: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    costs = []
    for g_ns in action_grid:
        costs.append(_cost(batch, _real_next_queue(spec, batch, g_ns), g_ns))
    matrix = np.column_stack(costs)
    return np.asarray(action_grid, dtype=float)[np.argmin(matrix, axis=1)], matrix


def _evaluate_actions(
    *,
    spec: NetworkSpec,
    batch: SignalBatch,
    action_grid: tuple[float, ...],
    chosen: np.ndarray,
    best_actions: np.ndarray,
    best_cost_matrix: np.ndarray,
) -> dict[str, float]:
    action_to_col = {float(action): idx for idx, action in enumerate(action_grid)}
    cols = np.asarray([action_to_col[float(action)] for action in chosen], dtype=int)
    chosen_cost = best_cost_matrix[np.arange(batch.q.shape[0]), cols]
    oracle_cost = np.min(best_cost_matrix, axis=1)
    max_pressure_actions = _choose_pressure(batch, action_grid, use_downstream=True)
    mp_cols = np.asarray([action_to_col[float(action)] for action in max_pressure_actions], dtype=int)
    mp_cost = best_cost_matrix[np.arange(batch.q.shape[0]), mp_cols]
    return {
        "mean_cost": float(np.mean(chosen_cost)),
        "mean_regret": float(np.mean(chosen_cost - oracle_cost)),
        "p90_regret": float(np.quantile(chosen_cost - oracle_cost, 0.90)),
        "oracle_action_accuracy": float(np.mean(chosen == best_actions)),
        "win_rate_vs_spillback_pressure": float(np.mean(chosen_cost < mp_cost)),
        "mean_green_ns": float(np.mean(chosen)),
    }


def _model_mae(
    *,
    spec: NetworkSpec,
    batch: SignalBatch,
    action_grid: tuple[float, ...],
    method: str,
    models: FittedModels,
    fewshot_bias: np.ndarray | None,
) -> float:
    errors = []
    for g_ns in action_grid:
        pred_q = _predict_next_queue(
            method=method,
            models=models,
            batch=batch,
            g_ns=g_ns,
            fewshot_bias=fewshot_bias,
        )
        real_q = _real_next_queue(spec, batch, g_ns)
        errors.append(np.abs(pred_q - real_q))
    return float(np.mean(np.concatenate([err.reshape(-1) for err in errors])))


def _fit_fewshot_bias(
    *,
    spec: NetworkSpec,
    models: FittedModels,
    batch: SignalBatch,
    action_grid: tuple[float, ...],
    rng: np.random.Generator,
    fewshot_samples: int,
) -> np.ndarray:
    n = min(int(fewshot_samples), batch.q.shape[0])
    if n <= 0:
        return np.zeros(4, dtype=float)
    idx = rng.choice(batch.q.shape[0], size=n, replace=False)
    few = _subset(batch, idx)
    behavior_actions = _choose_pressure(few, action_grid, use_downstream=True)
    residual_errors = []
    for g_ns in action_grid:
        mask = behavior_actions == g_ns
        if not np.any(mask):
            continue
        few_action = _subset(few, np.where(mask)[0])
        real_residual = _real_next_queue(spec, few_action, g_ns) - _sim_next_queue(few_action, g_ns)
        pred_residual = _predict_mechanism(models.cfcmt_global, few_action, g_ns)
        residual_errors.append(real_residual - pred_residual)
    if not residual_errors:
        return np.zeros(4, dtype=float)
    bias = np.mean(np.vstack(residual_errors), axis=0)
    return np.clip(bias, -20.0, 20.0)


def _run_target(
    *,
    target_spec: NetworkSpec,
    all_specs: list[NetworkSpec],
    batches: dict[str, SignalBatch],
    eval_batch: SignalBatch,
    action_grid: tuple[float, ...],
    rng: np.random.Generator,
    fewshot_samples: int,
    dense_l2: float,
    mechanism_l2: float,
) -> dict[str, Any]:
    train_specs = [spec for spec in all_specs if spec.name != target_spec.name]
    models = _fit_models(
        train_specs=train_specs,
        batches=batches,
        action_grid=action_grid,
        dense_l2=dense_l2,
        mechanism_l2=mechanism_l2,
    )
    best_actions, best_cost_matrix = _oracle_best_actions(target_spec, eval_batch, action_grid)
    fewshot_bias = _fit_fewshot_bias(
        spec=target_spec,
        models=models,
        batch=batches[target_spec.name],
        action_grid=action_grid,
        rng=rng,
        fewshot_samples=fewshot_samples,
    )

    policy_actions: dict[str, np.ndarray] = {
        "fixed_time": np.full(eval_batch.q.shape[0], _nearest_action(action_grid, 0.5), dtype=float),
        "queue_pressure": _choose_pressure(eval_batch, action_grid, use_downstream=False),
        "spillback_pressure": _choose_pressure(eval_batch, action_grid, use_downstream=True),
    }
    for method in (
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_weighted_mpc",
        "cfcmt_fewshot_bias_mpc",
    ):
        policy_actions[method] = _choose_model_mpc(
            method=method,
            models=models,
            batch=eval_batch,
            action_grid=action_grid,
            fewshot_bias=fewshot_bias,
        )
    policy_actions["oracle_mpc"] = best_actions

    policy_metrics = {
        method: _evaluate_actions(
            spec=target_spec,
            batch=eval_batch,
            action_grid=action_grid,
            chosen=chosen,
            best_actions=best_actions,
            best_cost_matrix=best_cost_matrix,
        )
        for method, chosen in policy_actions.items()
    }
    model_metrics = {
        method: {
            "next_queue_mae": _model_mae(
                spec=target_spec,
                batch=eval_batch,
                action_grid=action_grid,
                method=method,
                models=models,
                fewshot_bias=fewshot_bias,
            )
        }
        for method in (
            "sim_mpc",
            "h2oplus_dense_mpc",
            "cfcmt_global_mpc",
            "cfcmt_weighted_mpc",
            "cfcmt_fewshot_bias_mpc",
        )
    }
    return {
        "target": target_spec.name,
        "source_networks": [spec.name for spec in train_specs],
        "target_summary": _summary(target_spec).tolist(),
        "fewshot_samples": int(fewshot_samples),
        "fewshot_bias": fewshot_bias.tolist(),
        "policy_metrics": policy_metrics,
        "model_metrics": model_metrics,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(results[0]["policy_metrics"].keys())
    out: dict[str, Any] = {}
    for method in methods:
        rows = [target["policy_metrics"][method] for target in results]
        out[method] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0].keys()
        }
    model_methods = sorted(results[0]["model_metrics"].keys())
    model_out: dict[str, Any] = {}
    for method in model_methods:
        rows = [target["model_metrics"][method] for target in results]
        model_out[method] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0].keys()
        }
    return {"policy": out, "model": model_out}


def run_experiment(
    *,
    samples_per_network: int = 1400,
    eval_samples_per_network: int = 700,
    fewshot_samples: int = 80,
    seed: int = 7,
    action_grid: tuple[float, ...] = ACTION_GRID_DEFAULT,
    dense_l2: float = 25.0,
    mechanism_l2: float = 8.0,
) -> dict[str, Any]:
    specs = _network_specs()
    train_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1009)
    fewshot_rng = np.random.default_rng(seed + 2017)
    train_batches = {
        spec.name: _sample_batch(spec, samples_per_network, train_rng)
        for spec in specs
    }
    eval_batches = {
        spec.name: _sample_batch(spec, eval_samples_per_network, eval_rng)
        for spec in specs
    }
    targets = [
        _run_target(
            target_spec=target_spec,
            all_specs=specs,
            batches=train_batches,
            eval_batch=eval_batches[target_spec.name],
            action_grid=action_grid,
            rng=fewshot_rng,
            fewshot_samples=fewshot_samples,
            dense_l2=dense_l2,
            mechanism_l2=mechanism_l2,
        )
        for target_spec in specs
    ]
    aggregate = _aggregate(targets)
    return {
        "experiment": "traffic_signal_transfer_feasibility_phase0",
        "setting": {
            "simulator": "analytic_signal_queue",
            "transfer_split": "leave_one_network_out",
            "zero_shot_target_labels": 0,
            "fewshot_target_labels": int(fewshot_samples),
            "samples_per_network": int(samples_per_network),
            "eval_samples_per_network": int(eval_samples_per_network),
            "action_grid_green_ns": list(action_grid),
            "seed": int(seed),
            "dense_l2": float(dense_l2),
            "mechanism_l2": float(mechanism_l2),
        },
        "methods": {
            "fixed_time": "constant balanced green split",
            "queue_pressure": "current queue plus arrival pressure, no downstream occupancy",
            "spillback_pressure": "pressure rule with downstream blocking proxy",
            "sim_mpc": "uncalibrated analytic simulator, one-step MPC",
            "h2oplus_dense_mpc": "dense residual world model trained on source networks",
            "cfcmt_global_mpc": "parent-restricted mechanism residual trained on pooled source networks",
            "cfcmt_weighted_mpc": "source-weighted mechanism residual using target static summaries only",
            "cfcmt_fewshot_bias_mpc": "source-weighted CFCMT plus passive target residual-bias adaptation",
            "oracle_mpc": "true target mechanism for action selection",
        },
        "aggregate": aggregate,
        "targets": targets,
    }


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(result: dict[str, Any], path: Path) -> None:
    policy_rows = [
        {
            "method": method,
            "mean_cost": metrics["mean_cost"],
            "mean_regret": metrics["mean_regret"],
            "p90_regret": metrics["p90_regret"],
            "oracle_action_accuracy": metrics["oracle_action_accuracy"],
            "win_rate_vs_spillback_pressure": metrics["win_rate_vs_spillback_pressure"],
        }
        for method, metrics in sorted(
            result["aggregate"]["policy"].items(),
            key=lambda item: item[1]["mean_cost"],
        )
    ]
    model_rows = [
        {
            "method": method,
            "next_queue_mae": metrics["next_queue_mae"],
        }
        for method, metrics in sorted(
            result["aggregate"]["model"].items(),
            key=lambda item: item[1]["next_queue_mae"],
        )
    ]
    target_rows: list[dict[str, Any]] = []
    for target in result["targets"]:
        best_non_oracle = min(
            (
                (method, metrics)
                for method, metrics in target["policy_metrics"].items()
                if method != "oracle_mpc"
            ),
            key=lambda item: item[1]["mean_cost"],
        )
        target_rows.append(
            {
                "target": target["target"],
                "best_non_oracle": best_non_oracle[0],
                "best_cost": best_non_oracle[1]["mean_cost"],
                "best_regret": best_non_oracle[1]["mean_regret"],
                "spillback_pressure_cost": target["policy_metrics"]["spillback_pressure"]["mean_cost"],
                "cfcmt_weighted_cost": target["policy_metrics"]["cfcmt_weighted_mpc"]["mean_cost"],
                "cfcmt_fewshot_cost": target["policy_metrics"]["cfcmt_fewshot_bias_mpc"]["mean_cost"],
            }
        )

    lines = [
        "# Traffic Signal Transfer Feasibility Phase 0",
        "",
        "This is an analytic queueing benchmark for deciding whether traffic signal control is a better cross-city transfer environment than bus holding. It is not a SUMO experiment.",
        "",
        "Transfer protocol: leave-one-network-out; zero-shot methods use no target next-state labels; the few-shot CFCMT variant uses only passive target transitions under the spillback-pressure behavior controller.",
        "",
        "## Aggregate Policy Metrics",
        "",
        _format_table(
            policy_rows,
            [
                "method",
                "mean_cost",
                "mean_regret",
                "p90_regret",
                "oracle_action_accuracy",
                "win_rate_vs_spillback_pressure",
            ],
        ),
        "",
        "## Aggregate Model Metrics",
        "",
        _format_table(model_rows, ["method", "next_queue_mae"]),
        "",
        "## Target-Level Winners",
        "",
        _format_table(
            target_rows,
            [
                "target",
                "best_non_oracle",
                "best_cost",
                "best_regret",
                "spillback_pressure_cost",
                "cfcmt_weighted_cost",
                "cfcmt_fewshot_cost",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- If CFCMT weighted/few-shot beats spillback-pressure on regret and cost, the TSC direction is worth a SUMO Phase 1 implementation.",
        "- If spillback-pressure remains best, TSC may have the same rule-ceiling problem as bus holding and should not replace the paper's main environment without a stronger task design.",
        "- The few-shot row is not zero-shot: it uses target passive next-state labels and should be reported as target offline adaptation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_action_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if len(values) < 3:
        raise argparse.ArgumentTypeError("action grid must contain at least three values")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("green splits must be inside (0, 1)")
    return tuple(sorted(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-network", type=int, default=1400)
    parser.add_argument("--eval-samples-per-network", type=int, default=700)
    parser.add_argument("--fewshot-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-grid", type=_parse_action_grid, default=ACTION_GRID_DEFAULT)
    parser.add_argument("--dense-l2", type=float, default=25.0)
    parser.add_argument("--mechanism-l2", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/traffic_signal_transfer_feasibility.json"))
    parser.add_argument("--md-out", type=Path, default=Path("cf_h2o/results/traffic_signal_transfer_feasibility.md"))
    args = parser.parse_args(argv)

    result = run_experiment(
        samples_per_network=args.samples_per_network,
        eval_samples_per_network=args.eval_samples_per_network,
        fewshot_samples=args.fewshot_samples,
        seed=args.seed,
        action_grid=args.action_grid,
        dense_l2=args.dense_l2,
        mechanism_l2=args.mechanism_l2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    best = min(
        (
            (method, metrics["mean_cost"])
            for method, metrics in result["aggregate"]["policy"].items()
            if method != "oracle_mpc"
        ),
        key=lambda item: item[1],
    )
    print(f"best_non_oracle={best[0]} mean_cost={best[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
