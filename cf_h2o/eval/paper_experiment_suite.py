"""Paper experiment suite for open-transit H2O+ vs CFCMT validation.

This script fills the remaining paper-facing experiments around the existing
cross-city validation:

1. single-city route-heldout validation
2. sampled live H2O SimpleSim rollout, plus SUMO readiness probing
3. calibration-vs-no-calibration sweep
4. CFCMT ablations
5. source-city/data sensitivity
6. route-level bootstrap confidence intervals
7. static data sanity checks
8. efficiency/feature-dimension measurements

The heavy part is one full pass over the generated city bundles to build
route-level sufficient statistics. The downstream experiments reuse those
statistics instead of re-reading every Excel file for every table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_open_transit_validation import expand_splits
from cf_h2o.eval.cross_city_performance_validation import (
    DYNAMIC_OUTPUT_INDICES,
    HOURS,
    OUTPUT_NAMES,
    cfcmt_features,
    h2o_features,
    _headways_by_hour,
    _hour_col,
    _line_dirs,
    _od_demand_by_hour_stop,
    _parse_direction,
    _read_json,
    _read_line_tables,
    _real_transition,
    _repo_root,
    _resolve_path,
    _reward,
    _stable_unit,
    _uncalibrated_transition,
)
from cf_h2o.eval.cross_city_policy_validation import (
    DEFAULT_ACTIONS as POLICY_ACTIONS,
    _reward_from_predicted_state,
)


FeatureFn = Callable[[str, np.ndarray], np.ndarray]


def _parse_float_list(value: str) -> list[float]:
    out = [float(item) for item in str(value).split(",") if item.strip()]
    if not out:
        raise ValueError("expected at least one float")
    return out


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    qs = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {key: float(value) for key, value in zip(("p05", "p25", "p50", "p75", "p95"), qs)}


@dataclass
class LinearStats:
    xtx: np.ndarray
    xty: np.ndarray
    yty: np.ndarray
    n: int = 0

    @classmethod
    def zeros(cls, feature_dim: int, output_dim: int = 1) -> "LinearStats":
        return cls(
            xtx=np.zeros((feature_dim, feature_dim), dtype=np.float64),
            xty=np.zeros((feature_dim, output_dim), dtype=np.float64),
            yty=np.zeros(output_dim, dtype=np.float64),
            n=0,
        )

    def add(self, features: np.ndarray, target: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.yty += np.square(y).sum(axis=0)
        self.n += int(x.shape[0])

    def merge(self, other: "LinearStats") -> None:
        self.xtx += other.xtx
        self.xty += other.xty
        self.yty += other.yty
        self.n += other.n

    def merge_scaled(self, other: "LinearStats", scale: float) -> None:
        self.xtx += float(scale) * other.xtx
        self.xty += float(scale) * other.xty
        self.yty += float(scale) * other.yty
        self.n += int(round(float(scale) * other.n))

    def solve(self, ridge: float) -> np.ndarray:
        reg = float(ridge) * np.eye(self.xtx.shape[0], dtype=np.float64)
        reg[0, 0] = 0.0
        return np.linalg.solve(self.xtx + reg, self.xty)

    def sse(self, beta: np.ndarray | None = None, *, target_scale: float | np.ndarray = 1.0) -> np.ndarray:
        scale = np.asarray(target_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(self.yty.shape[0], float(scale), dtype=np.float64)
        scale = scale.reshape(-1)
        if scale.shape[0] != self.yty.shape[0]:
            raise ValueError(f"target_scale has {scale.shape[0]} outputs, expected {self.yty.shape[0]}")
        if beta is None:
            return self.yty * np.square(scale)
        b = np.asarray(beta, dtype=np.float64)
        if b.ndim == 1:
            b = b[:, None]
        # If the residual target is scaled, the fitted beta scales with it.
        b = b * scale[None, :]
        xty = self.xty * scale[None, :]
        out = self.yty * np.square(scale)
        out = out + np.einsum("ik,ij,jk->k", b, self.xtx, b)
        out = out - 2.0 * np.einsum("ij,ij->j", b, xty)
        return np.maximum(out, 0.0)


def _family_zero(feature_fn: FeatureFn) -> dict[str, LinearStats]:
    dummy = np.zeros((1, 16), dtype=np.float32)
    return {
        output_name: LinearStats.zeros(feature_fn(output_name, dummy).shape[1], 1)
        for output_name in OUTPUT_NAMES
    }


def _family_merge(dst: dict[str, LinearStats], src: dict[str, LinearStats]) -> None:
    for output_name in OUTPUT_NAMES:
        dst[output_name].merge(src[output_name])


def _family_merge_scaled(dst: dict[str, LinearStats], src: dict[str, LinearStats], scale: float) -> None:
    for output_name in OUTPUT_NAMES:
        dst[output_name].merge_scaled(src[output_name], scale)


def _family_solve(stats: dict[str, LinearStats], ridge: float) -> dict[str, np.ndarray]:
    return {name: value.solve(ridge).reshape(-1) for name, value in stats.items()}


def _family_sse(
    stats: dict[str, LinearStats],
    beta: dict[str, np.ndarray] | None = None,
    *,
    target_scale: float | np.ndarray = 1.0,
) -> np.ndarray:
    values = []
    scale = np.asarray(target_scale, dtype=np.float64)
    for idx, output_name in enumerate(OUTPUT_NAMES):
        coef = None if beta is None else beta[output_name]
        output_scale = float(scale) if scale.ndim == 0 else float(scale.reshape(-1)[idx])
        values.append(float(stats[output_name].sse(coef, target_scale=output_scale)[0]))
    return np.asarray(values, dtype=np.float64)


def _cfcmt_no_action_interaction(output_name: str, obs_action: np.ndarray) -> np.ndarray:
    full = cfcmt_features(output_name, obs_action)
    base_dim = (full.shape[1] + 1) // 2
    return full[:, :base_dim]


def _cfcmt_shared_sparse(output_name: str, obs_action: np.ndarray) -> np.ndarray:
    return cfcmt_features("__shared_sparse__", obs_action)


def _cfcmt_action_time_only(output_name: str, obs_action: np.ndarray) -> np.ndarray:
    x = obs_action.astype(np.float64, copy=False)
    hour = x[:, 3:4]
    action = x[:, 15:16] / 60.0
    base = np.concatenate(
        [
            np.ones((x.shape[0], 1), dtype=np.float64),
            action,
            np.sin(hour * math.tau),
            np.cos(hour * math.tau),
        ],
        axis=1,
    )
    return np.concatenate([base, base[:, 1:] * action], axis=1)


ABLATION_FEATURES: dict[str, FeatureFn] = {
    "cfcmt_full": cfcmt_features,
    "cfcmt_no_action_interaction": _cfcmt_no_action_interaction,
    "cfcmt_shared_sparse": _cfcmt_shared_sparse,
    "cfcmt_action_time_only": _cfcmt_action_time_only,
}


@dataclass
class ResidualStats:
    key: str
    city: str
    h2o: LinearStats
    cfcmt: dict[str, LinearStats]
    ablations: dict[str, dict[str, LinearStats]]
    n: int = 0
    lines_seen: int = 0
    line_keys: list[str] = field(default_factory=list)

    @classmethod
    def zeros(cls, key: str, city: str) -> "ResidualStats":
        dummy = np.zeros((1, 16), dtype=np.float32)
        return cls(
            key=key,
            city=city,
            h2o=LinearStats.zeros(h2o_features(dummy).shape[1], len(OUTPUT_NAMES)),
            cfcmt=_family_zero(cfcmt_features),
            ablations={name: _family_zero(fn) for name, fn in ABLATION_FEATURES.items()},
        )

    def merge(self, other: "ResidualStats") -> None:
        self.h2o.merge(other.h2o)
        _family_merge(self.cfcmt, other.cfcmt)
        for name in ABLATION_FEATURES:
            _family_merge(self.ablations[name], other.ablations[name])
        self.n += other.n
        self.lines_seen += other.lines_seen
        self.line_keys.extend(other.line_keys)

    def merge_scaled(self, other: "ResidualStats", scale: float) -> None:
        self.h2o.merge_scaled(other.h2o, scale)
        _family_merge_scaled(self.cfcmt, other.cfcmt, scale)
        for name in ABLATION_FEATURES:
            _family_merge_scaled(self.ablations[name], other.ablations[name], scale)
        self.n += int(round(float(scale) * other.n))
        self.lines_seen += other.lines_seen
        self.line_keys.extend(other.line_keys)

    def add_arrays(self, line_key: str, obs_action: np.ndarray, residual: np.ndarray) -> None:
        self.h2o.add(h2o_features(obs_action), residual)
        for idx, output_name in enumerate(OUTPUT_NAMES):
            self.cfcmt[output_name].add(cfcmt_features(output_name, obs_action), residual[:, idx])
            for name, feature_fn in ABLATION_FEATURES.items():
                self.ablations[name][output_name].add(feature_fn(output_name, obs_action), residual[:, idx])
        self.n += int(obs_action.shape[0])
        self.lines_seen = 1
        self.line_keys = [line_key]


@dataclass
class SanityAccumulator:
    line_count: int = 0
    segment_count: int = 0
    stop_count: int = 0
    timetable_count: int = 0
    route_distance_m: list[float] = field(default_factory=list)
    speeds_mps: list[float] = field(default_factory=list)
    headways_s: list[float] = field(default_factory=list)
    demand_by_hour: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.float64))

    def add(self, route: pd.DataFrame, timetable: pd.DataFrame, passenger_od: pd.DataFrame, headways: dict[int, float]) -> None:
        self.line_count += 1
        self.segment_count += int(len(route))
        if "start_stop" in route.columns:
            self.stop_count += int(route["start_stop"].nunique())
        self.timetable_count += int(len(timetable))
        if "distance" in route.columns:
            dist = pd.to_numeric(route["distance"], errors="coerce").fillna(0.0).clip(lower=0.0)
            self.route_distance_m.append(float(dist.sum()))
        for hour in HOURS:
            col = _hour_col(hour)
            if col in route.columns:
                speed = pd.to_numeric(route[col], errors="coerce").dropna()
                self.speeds_mps.extend(float(v) for v in speed[(speed > 0.0) & (speed < 80.0)].to_numpy())
            self.headways_s.append(float(headways.get(hour, 600.0)))
        if not passenger_od.empty and "time_period" in passenger_od.columns:
            value_cols = [col for col in passenger_od.columns if col not in {"time_period", "stop_name"}]
            if value_cols:
                demand = passenger_od[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
                hours = passenger_od["time_period"].astype(str).str.slice(0, 2)
                for hour_text, value in zip(hours, demand):
                    if str(hour_text).isdigit():
                        self.demand_by_hour[int(hour_text) % 24] += float(value)

    def summary(self) -> dict[str, Any]:
        total_demand = float(self.demand_by_hour.sum())
        peak_hour = int(np.argmax(self.demand_by_hour)) if total_demand > 0 else None
        return {
            "line_count": self.line_count,
            "segment_count": self.segment_count,
            "stop_count": self.stop_count,
            "timetable_count": self.timetable_count,
            "route_distance_m_quantiles": _quantiles(self.route_distance_m),
            "speed_mps_quantiles": _quantiles(self.speeds_mps),
            "headway_s_quantiles": _quantiles(self.headways_s),
            "total_static_demand": total_demand,
            "peak_demand_hour": peak_hour,
            "demand_by_hour_share": (
                [float(v / total_demand) for v in self.demand_by_hour]
                if total_demand > 0
                else [0.0] * 24
            ),
        }


def _iter_line_arrays(line_dir: Path, actions: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    line_key = line_dir.name
    try:
        route, timetable, passenger_od = _read_line_tables(line_dir)
    except Exception as exc:
        return None
    if route.empty:
        return None

    line_code = _stable_unit(line_key)
    direction = _parse_direction(line_key, timetable)
    headways = _headways_by_hour(timetable)
    demand_by = _od_demand_by_hour_stop(passenger_od)

    route = route.copy()
    route["_route_id"] = pd.to_numeric(route.get("route_id", pd.Series(range(len(route)))), errors="coerce").fillna(0).astype(float)
    route["_distance"] = pd.to_numeric(route.get("distance", 0.0), errors="coerce").fillna(0.0).clip(lower=30.0).astype(float)
    route["_seg_idx"] = np.arange(len(route), dtype=float)
    nseg = len(route)
    if nseg <= 0:
        return None

    obs_parts: list[np.ndarray] = []
    sim_parts: list[np.ndarray] = []
    real_parts: list[np.ndarray] = []
    for hour in HOURS:
        col = _hour_col(hour)
        if col in route.columns:
            speed = pd.to_numeric(route[col], errors="coerce").fillna(route.get("V_max", 8.0)).to_numpy(dtype=np.float64)
        else:
            speed = pd.to_numeric(route.get("V_max", 8.0), errors="coerce").fillna(8.0).to_numpy(dtype=np.float64)
        speed = np.clip(speed, 1.0, 25.0)
        next_speed = np.roll(speed, -1)
        next_speed[-1] = speed[-1]
        distance = route["_distance"].to_numpy(dtype=np.float64)
        travel_time = distance / np.clip(speed, 1.0, None)
        route_idx = route["_route_id"].to_numpy(dtype=np.float64)
        station_fraction = route["_seg_idx"].to_numpy(dtype=np.float64) / max(float(nseg - 1), 1.0)
        start_stops = route["start_stop"].astype(str).tolist()
        end_stops = route["end_stop"].astype(str).tolist()
        demand = np.array([demand_by.get((hour, stop), 0.0) for stop in start_stops], dtype=np.float64)
        next_demand = np.array([demand_by.get((hour, stop), 0.0) for stop in end_stops], dtype=np.float64)
        target = np.full(nseg, headways[hour], dtype=np.float64)
        peak = 1.0 + 0.35 * math.exp(-((hour - 8.0) / 2.8) ** 2) + 0.30 * math.exp(-((hour - 17.0) / 3.2) ** 2)
        station_wave = np.sin((station_fraction + line_code) * math.tau)

        fwd = np.clip(target * (1.0 + 0.06 * np.tanh((travel_time - 90.0) / 120.0) + 0.025 * station_wave), 20.0, 7200.0)
        bwd = np.clip(target * (1.0 - 0.05 * np.tanh((travel_time - 90.0) / 120.0) - 0.020 * station_wave), 20.0, 7200.0)
        waiting = np.clip(0.70 * demand * peak + 0.025 * fwd, 0.0, None)
        stop_duration = 8.0 + 0.28 * np.sqrt(waiting + 1.0) + 0.015 * waiting
        sim_time = np.full(nseg, hour * 3600.0, dtype=np.float64) + station_fraction * 3600.0
        co_fwd = np.clip(0.75 * target + 0.25 * np.roll(target, 1), 20.0, 7200.0)
        co_bwd = np.clip(0.75 * target + 0.25 * np.roll(target, -1), 20.0, 7200.0)
        obs = np.column_stack(
            [
                np.full(nseg, line_code),
                np.mod(route_idx, 400.0) / 400.0,
                station_fraction,
                np.full(nseg, hour / 23.0),
                np.full(nseg, direction),
                fwd,
                bwd,
                waiting,
                target,
                stop_duration,
                sim_time,
                fwd - bwd,
                co_fwd,
                co_bwd,
                speed,
            ]
        ).astype(np.float32)

        for hold in actions:
            action = np.full(nseg, float(hold), dtype=np.float64)
            sim_y = np.column_stack(
                _uncalibrated_transition(fwd, bwd, waiting, target, speed, travel_time, action, hour)
            ).astype(np.float32)
            real_y = np.column_stack(
                _real_transition(
                    fwd,
                    bwd,
                    waiting,
                    stop_duration,
                    target,
                    co_fwd,
                    co_bwd,
                    speed,
                    next_speed,
                    next_demand,
                    travel_time,
                    action,
                    peak,
                )
            ).astype(np.float32)
            obs_parts.append(np.concatenate([obs, action[:, None].astype(np.float32)], axis=1))
            sim_parts.append(sim_y)
            real_parts.append(real_y)

    meta = {
        "line_key": line_key,
        "route": route,
        "timetable": timetable,
        "passenger_od": passenger_od,
        "headways": headways,
    }
    return np.concatenate(obs_parts, axis=0), np.concatenate(sim_parts, axis=0), np.concatenate(real_parts, axis=0), meta


def build_city_stats(key: str, spec: dict[str, Any], root: Path, args: argparse.Namespace) -> tuple[ResidualStats, list[ResidualStats], dict[str, Any]]:
    env_path = _resolve_path(root, spec["env_path"])
    aggregate = ResidualStats.zeros(key, str(spec.get("city", key)))
    line_stats: list[ResidualStats] = []
    sanity = SanityAccumulator()
    t0 = time.time()
    for idx, line_dir in enumerate(_line_dirs(env_path, args.max_lines_per_city), start=1):
        arrays = _iter_line_arrays(line_dir, args.actions)
        if arrays is None:
            continue
        obs_action, sim_y, real_y, meta = arrays
        residual = real_y.astype(np.float64) - sim_y.astype(np.float64)
        one = ResidualStats.zeros(f"{key}::{line_dir.name}", str(spec.get("city", key)))
        one.add_arrays(line_dir.name, obs_action, residual)
        aggregate.merge(one)
        line_stats.append(one)
        sanity.add(meta["route"], meta["timetable"], meta["passenger_od"], meta["headways"])
        if args.progress_every and idx % args.progress_every == 0:
            print(f"[stats] {key}: {idx} lines, rows={aggregate.n}", flush=True)
    return aggregate, line_stats, {
        "elapsed_sec": time.time() - t0,
        "env_path": str(env_path),
        **sanity.summary(),
    }


def _merge_stats(key: str, city: str, stats_list: list[ResidualStats]) -> ResidualStats:
    out = ResidualStats.zeros(key, city)
    for item in stats_list:
        out.merge(item)
    return out


def _merge_stats_weighted(key: str, city: str, stats_by_key: dict[str, ResidualStats], weights: dict[str, float]) -> ResidualStats:
    out = ResidualStats.zeros(key, city)
    active = {source: float(weight) for source, weight in weights.items() if weight > 0.0 and stats_by_key[source].n > 0}
    if not active:
        return out
    total_weight = sum(active.values())
    total_n = sum(stats_by_key[source].n for source in active)
    for source, weight in active.items():
        effective_n = (weight / total_weight) * total_n
        scale = effective_n / max(float(stats_by_key[source].n), 1.0)
        out.merge_scaled(stats_by_key[source], scale)
    return out


def _fit_h2o(stats: ResidualStats, ridge: float) -> np.ndarray:
    return stats.h2o.solve(ridge)


def _fit_cfcmt(stats: ResidualStats, ridge: float, family: str = "cfcmt_full") -> dict[str, np.ndarray]:
    source = stats.cfcmt if family == "cfcmt_full" else stats.ablations[family]
    return _family_solve(source, ridge)


def _method_metrics_from_sse(sse: np.ndarray, n: int) -> dict[str, Any]:
    per_output = sse / float(max(1, n))
    dyn_count = len(DYNAMIC_OUTPUT_INDICES)
    return {
        "total_mse": float(per_output.mean()),
        "next_dynamic_mse": float(per_output[:dyn_count].mean()),
        "reward_mse": float(per_output[-1]),
        "per_output_mse": {name: float(value) for name, value in zip(OUTPUT_NAMES, per_output)},
    }


def evaluate_stats(
    eval_stats: ResidualStats,
    h2o_beta: np.ndarray | None,
    cfcmt_beta: dict[str, np.ndarray] | None,
    *,
    cfcmt_family: str = "cfcmt_full",
    target_scale: float = 1.0,
) -> dict[str, Any]:
    family_stats = eval_stats.cfcmt if cfcmt_family == "cfcmt_full" else eval_stats.ablations[cfcmt_family]
    uncal_sse = eval_stats.h2o.sse(None, target_scale=target_scale)
    h2o_sse = eval_stats.h2o.sse(h2o_beta, target_scale=target_scale) if h2o_beta is not None else uncal_sse
    cfcmt_sse = _family_sse(family_stats, cfcmt_beta, target_scale=target_scale) if cfcmt_beta is not None else uncal_sse
    return {
        "target_transitions": eval_stats.n,
        "target_lines_seen": eval_stats.lines_seen,
        "uncalibrated": _method_metrics_from_sse(uncal_sse, eval_stats.n),
        "h2oplus_dense": _method_metrics_from_sse(h2o_sse, eval_stats.n),
        "cfcmt_mechanism": _method_metrics_from_sse(cfcmt_sse, eval_stats.n),
    }


def _comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    h2o = metrics["h2oplus_dense"]["total_mse"]
    cfcmt = metrics["cfcmt_mechanism"]["total_mse"]
    uncal = metrics["uncalibrated"]["total_mse"]
    return {
        "h2oplus_vs_uncalibrated_total_mse_ratio": h2o / uncal if uncal else None,
        "cfcmt_vs_uncalibrated_total_mse_ratio": cfcmt / uncal if uncal else None,
        "cfcmt_vs_h2oplus_total_mse_ratio": cfcmt / h2o if h2o else None,
        "cfcmt_beats_h2oplus": bool(cfcmt < h2o),
        "cfcmt_beats_uncalibrated": bool(cfcmt < uncal),
        "h2oplus_beats_uncalibrated": bool(h2o < uncal),
    }


def _strict_leave_one_city_out_splits(config: dict[str, Any], env_keys: list[str] | None = None, prefix: str = "strict_leave_one_city_out") -> list[dict[str, Any]]:
    keys = list(env_keys or config["generated_envs"].keys())
    return [
        {
            "name": f"{prefix}::{target}",
            "source_envs": [key for key in keys if key != target],
            "target_env": target,
        }
        for target in keys
        if len(keys) > 1
    ]


def _output_group(output_name: str) -> str:
    if "headway" in output_name or "gap" in output_name:
        return "headway_gap"
    if "waiting" in output_name or "base_stop_duration" in output_name or output_name == "reward":
        return "demand_stop_reward"
    if "segment_mean_speed" in output_name:
        return "speed"
    return "other"


def _output_scale_vector(group_scales: dict[str, float] | None = None) -> np.ndarray:
    group_scales = group_scales or {}
    return np.asarray([float(group_scales.get(_output_group(name), 1.0)) for name in OUTPUT_NAMES], dtype=np.float64)


def _add_measurement_noise(sse: np.ndarray, reference_sse: np.ndarray, noise_fraction: float) -> np.ndarray:
    if noise_fraction <= 0.0:
        return sse
    return sse + np.square(float(noise_fraction)) * reference_sse


def _method_bundle_from_sse(
    target_stats: ResidualStats,
    h2o_sse: np.ndarray,
    cfcmt_sse: np.ndarray,
    *,
    uncal_sse: np.ndarray,
    weighted_cfcmt_sse: np.ndarray | None = None,
    weighted_h2o_sse: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = {
        "uncalibrated": _method_metrics_from_sse(uncal_sse, target_stats.n),
        "h2oplus_dense": _method_metrics_from_sse(h2o_sse, target_stats.n),
        "cfcmt_mechanism": _method_metrics_from_sse(cfcmt_sse, target_stats.n),
    }
    if weighted_h2o_sse is not None:
        metrics["h2oplus_similarity_weighted"] = _method_metrics_from_sse(weighted_h2o_sse, target_stats.n)
    if weighted_cfcmt_sse is not None:
        metrics["cfcmt_similarity_weighted"] = _method_metrics_from_sse(weighted_cfcmt_sse, target_stats.n)
    return metrics


def _method_total_mse(stats: ResidualStats, method_sse: np.ndarray) -> float:
    return float((method_sse / float(max(1, stats.n))).mean())


def _city_descriptor(summary: dict[str, Any]) -> np.ndarray:
    def q(block: str, key: str, default: float) -> float:
        value = summary.get(block, {}).get(key)
        return float(value) if value is not None else default

    speed = [q("speed_mps_quantiles", item, 5.0) for item in ("p25", "p50", "p75")]
    headway = [math.log1p(q("headway_s_quantiles", item, 600.0)) for item in ("p25", "p50", "p75")]
    route = [math.log1p(q("route_distance_m_quantiles", item, 10_000.0)) for item in ("p50", "p75")]
    demand_share = np.asarray(summary.get("demand_by_hour_share", [0.0] * 24), dtype=np.float64)
    if demand_share.shape[0] != 24:
        demand_share = np.resize(demand_share, 24)
    peak = summary.get("peak_demand_hour")
    peak_hour = float(peak if peak is not None else int(np.argmax(demand_share)))
    peak_vec = [math.sin((peak_hour / 24.0) * math.tau), math.cos((peak_hour / 24.0) * math.tau)]
    return np.asarray(speed + headway + route + peak_vec + (0.35 * demand_share).tolist(), dtype=np.float64)


def _source_similarity_weights(
    sanity: dict[str, Any],
    target: str,
    sources: list[str],
    *,
    temperature: float,
    floor: float,
) -> dict[str, float]:
    keys = sorted(set(sources + [target]))
    raw = {key: _city_descriptor(sanity[key]) for key in keys}
    matrix = np.vstack([raw[key] for key in keys])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    target_vec = (raw[target] - mean) / std
    distances = {}
    for source in sources:
        vec = (raw[source] - mean) / std
        distances[source] = float(np.linalg.norm(vec - target_vec) / math.sqrt(vec.shape[0]))
    temp = max(float(temperature), 1e-6)
    base = {source: math.exp(-distance / temp) for source, distance in distances.items()}
    if floor > 0.0 and base:
        max_base = max(base.values())
        base = {source: value + float(floor) * max_base for source, value in base.items()}
    denom = sum(base.values()) or 1.0
    return {source: value / denom for source, value in base.items()}


def run_source_weighting(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    ridge: float,
    *,
    temperature: float,
    floor: float,
    splits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = []
    for split in (expand_splits(config) if splits is None else splits):
        sources = list(split["source_envs"])
        target = split["target_env"]
        target_stats = city_stats[target]
        weights = _source_similarity_weights(sanity, target, sources, temperature=temperature, floor=floor)
        train_unweighted = _merge_stats("source_unweighted", "source", [city_stats[key] for key in sources])
        train_weighted = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, weights)
        h2o_unweighted = _fit_h2o(train_unweighted, ridge)
        h2o_weighted = _fit_h2o(train_weighted, ridge)
        cfcmt_unweighted = _fit_cfcmt(train_unweighted, ridge)
        cfcmt_weighted = _fit_cfcmt(train_weighted, ridge)
        uncal_sse = target_stats.h2o.sse(None)
        h2o_unweighted_sse = target_stats.h2o.sse(h2o_unweighted)
        h2o_weighted_sse = target_stats.h2o.sse(h2o_weighted)
        cfcmt_unweighted_sse = _family_sse(target_stats.cfcmt, cfcmt_unweighted)
        cfcmt_weighted_sse = _family_sse(target_stats.cfcmt, cfcmt_weighted)
        metrics = {
            "uncalibrated": _method_metrics_from_sse(uncal_sse, target_stats.n),
            "h2oplus_dense": _method_metrics_from_sse(h2o_unweighted_sse, target_stats.n),
            "h2oplus_similarity_weighted": _method_metrics_from_sse(h2o_weighted_sse, target_stats.n),
            "cfcmt_mechanism": _method_metrics_from_sse(cfcmt_unweighted_sse, target_stats.n),
            "cfcmt_similarity_weighted": _method_metrics_from_sse(cfcmt_weighted_sse, target_stats.n),
        }
        h2o_total = metrics["h2oplus_dense"]["total_mse"]
        cfcmt_total = metrics["cfcmt_mechanism"]["total_mse"]
        weighted_total = metrics["cfcmt_similarity_weighted"]["total_mse"]
        rows.append(
            {
                **split,
                "target_city": config["generated_envs"][target].get("city", target),
                "source_weights": weights,
                "train_transitions_unweighted": train_unweighted.n,
                "train_transitions_weighted_effective": train_weighted.n,
                "target_transitions": target_stats.n,
                "metrics": metrics,
                "comparisons": {
                    "cfcmt_similarity_weighted_vs_h2oplus_ratio": weighted_total / h2o_total if h2o_total else None,
                    "cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": weighted_total / cfcmt_total if cfcmt_total else None,
                    "cfcmt_similarity_weighted_beats_h2oplus": bool(weighted_total < h2o_total),
                    "cfcmt_similarity_weighted_beats_unweighted_cfcmt": bool(weighted_total < cfcmt_total),
                    "h2oplus_similarity_weighted_vs_unweighted_ratio": (
                        metrics["h2oplus_similarity_weighted"]["total_mse"] / h2o_total if h2o_total else None
                    ),
                },
            }
        )
    weighted_ratios = [row["comparisons"]["cfcmt_similarity_weighted_vs_h2oplus_ratio"] for row in rows]
    improvement_ratios = [row["comparisons"]["cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"] for row in rows]
    return {
        "ok": True,
        "experiment": "source_similarity_weighting",
        "definition": "City-balanced source sufficient statistics reweighted by static speed/headway/route/demand similarity to the target city.",
        "temperature": temperature,
        "floor": floor,
        "splits": rows,
        "summary": {
            "splits": len(rows),
            "cfcmt_similarity_weighted_wins_vs_h2oplus": sum(
                1 for row in rows if row["comparisons"]["cfcmt_similarity_weighted_beats_h2oplus"]
            ),
            "cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt": sum(
                1 for row in rows if row["comparisons"]["cfcmt_similarity_weighted_beats_unweighted_cfcmt"]
            ),
            "mean_cfcmt_similarity_weighted_vs_h2oplus_ratio": float(np.mean(weighted_ratios)),
            "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": float(np.mean(improvement_ratios)),
        },
    }


def run_source_weighting_sensitivity(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    ridge: float,
    *,
    temperatures: list[float],
    floors: list[float],
    default_temperature: float,
    default_floor: float,
) -> dict[str, Any]:
    grid_rows = []
    split_rows = []
    for temperature, floor in itertools.product(temperatures, floors):
        result = run_source_weighting(
            city_stats,
            config,
            sanity,
            ridge,
            temperature=float(temperature),
            floor=float(floor),
        )
        summary = result["summary"]
        grid_rows.append(
            {
                "temperature": float(temperature),
                "floor": float(floor),
                **summary,
            }
        )
        for split in result["splits"]:
            comparisons = split["comparisons"]
            split_rows.append(
                {
                    "temperature": float(temperature),
                    "floor": float(floor),
                    "name": split["name"],
                    "target_env": split["target_env"],
                    "target_city": split["target_city"],
                    "source_envs": split["source_envs"],
                    "source_weights": split["source_weights"],
                    "cfcmt_similarity_weighted_vs_h2oplus_ratio": comparisons[
                        "cfcmt_similarity_weighted_vs_h2oplus_ratio"
                    ],
                    "cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": comparisons[
                        "cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"
                    ],
                    "cfcmt_similarity_weighted_beats_h2oplus": comparisons[
                        "cfcmt_similarity_weighted_beats_h2oplus"
                    ],
                    "cfcmt_similarity_weighted_beats_unweighted_cfcmt": comparisons[
                        "cfcmt_similarity_weighted_beats_unweighted_cfcmt"
                    ],
                }
            )

    best_vs_h2o = min(grid_rows, key=lambda row: row["mean_cfcmt_similarity_weighted_vs_h2oplus_ratio"])
    best_vs_unweighted = min(
        grid_rows,
        key=lambda row: row["mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"],
    )
    default_rows = [
        row
        for row in grid_rows
        if math.isclose(row["temperature"], float(default_temperature)) and math.isclose(row["floor"], float(default_floor))
    ]
    vs_h2o_ratios = [row["mean_cfcmt_similarity_weighted_vs_h2oplus_ratio"] for row in grid_rows]
    vs_unweighted_ratios = [row["mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"] for row in grid_rows]
    return {
        "ok": True,
        "experiment": "source_weighting_sensitivity",
        "definition": "Grid sensitivity for source-city similarity weighting temperature and floor.",
        "temperatures": [float(value) for value in temperatures],
        "floors": [float(value) for value in floors],
        "grid": grid_rows,
        "split_rows": split_rows,
        "summary": {
            "grid_points": len(grid_rows),
            "splits_per_grid_point": len(expand_splits(config)),
            "grid_points_with_all_splits_beating_h2oplus": sum(
                1 for row in grid_rows if row["cfcmt_similarity_weighted_wins_vs_h2oplus"] == row["splits"]
            ),
            "grid_points_with_majority_beating_unweighted_cfcmt": sum(
                1 for row in grid_rows if row["cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt"] > row["splits"] / 2.0
            ),
            "mean_vs_h2oplus_ratio_range": [float(min(vs_h2o_ratios)), float(max(vs_h2o_ratios))],
            "mean_vs_unweighted_cfcmt_ratio_range": [float(min(vs_unweighted_ratios)), float(max(vs_unweighted_ratios))],
            "best_vs_h2oplus": best_vs_h2o,
            "best_vs_unweighted_cfcmt": best_vs_unweighted,
            "default_grid_point": default_rows[0] if default_rows else None,
        },
    }


def _summarize_weighted_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cfcmt_ratios = [
        row["metrics"]["cfcmt_mechanism"]["total_mse"] / row["metrics"]["h2oplus_dense"]["total_mse"]
        for row in rows
    ]
    weighted_ratios = [row["comparisons"]["cfcmt_similarity_weighted_vs_h2oplus_ratio"] for row in rows]
    weighted_vs_unweighted = [row["comparisons"]["cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"] for row in rows]
    return {
        "splits": len(rows),
        "unique_targets": len({row["target_env"] for row in rows}),
        "cfcmt_wins_vs_h2oplus": sum(
            1 for row in rows if row["metrics"]["cfcmt_mechanism"]["total_mse"] < row["metrics"]["h2oplus_dense"]["total_mse"]
        ),
        "cfcmt_similarity_weighted_wins_vs_h2oplus": sum(
            1 for row in rows if row["comparisons"]["cfcmt_similarity_weighted_beats_h2oplus"]
        ),
        "cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt": sum(
            1 for row in rows if row["comparisons"]["cfcmt_similarity_weighted_beats_unweighted_cfcmt"]
        ),
        "mean_cfcmt_vs_h2oplus_ratio": float(np.mean(cfcmt_ratios)),
        "mean_cfcmt_similarity_weighted_vs_h2oplus_ratio": float(np.mean(weighted_ratios)),
        "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": float(np.mean(weighted_vs_unweighted)),
    }


def run_strict_leave_one_out(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    ridge: float,
    *,
    temperature: float,
    floor: float,
) -> dict[str, Any]:
    splits = _strict_leave_one_city_out_splits(config)
    result = run_source_weighting(
        city_stats,
        config,
        sanity,
        ridge,
        temperature=temperature,
        floor=floor,
        splits=splits,
    )
    return {
        "ok": True,
        "experiment": "strict_leave_one_city_out_cross_city",
        "definition": "Primary city-level leave-one-city-out validation: each of the four cities appears exactly once as target.",
        "temperature": temperature,
        "floor": floor,
        "splits": result["splits"],
        "summary": _summarize_weighted_rows(result["splits"]),
    }


def run_source_subset_robustness(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    ridge: float,
    *,
    temperature: float,
    floor: float,
) -> dict[str, Any]:
    all_keys = list(config["generated_envs"].keys())
    subset_specs = [
        {
            "name": "all_four_strict_leave_one_out",
            "env_keys": all_keys,
            "definition": "All available cities; each city is target once.",
        },
        {
            "name": "exclude_singapore_gtfs_traffic_only",
            "env_keys": [key for key in all_keys if key != "singapore_lta_all"],
            "definition": "Singapore removed; remaining cities all use GTFS schedule-derived traffic.",
        },
        {
            "name": "exclude_austin_schedule_demand_proxy",
            "env_keys": [key for key in all_keys if key != "austin_capmetro_all"],
            "definition": "Austin removed; remaining cities have observed or apportioned passenger demand sources.",
        },
        {
            "name": "north_american_observed_ridership_pair",
            "env_keys": [key for key in ("halifax_transit_all", "mbta_all") if key in all_keys],
            "definition": "Two North American systems with observed/apportioned ridership; one city is source and one is target.",
        },
    ]
    subsets = []
    for spec in subset_specs:
        env_keys = [key for key in spec["env_keys"] if key in city_stats]
        if len(env_keys) < 2:
            continue
        splits = _strict_leave_one_city_out_splits(config, env_keys, prefix=spec["name"])
        result = run_source_weighting(
            city_stats,
            config,
            sanity,
            ridge,
            temperature=temperature,
            floor=floor,
            splits=splits,
        )
        subsets.append(
            {
                "name": spec["name"],
                "definition": spec["definition"],
                "env_keys": env_keys,
                "cities": [config["generated_envs"][key].get("city", key) for key in env_keys],
                "splits": result["splits"],
                "summary": _summarize_weighted_rows(result["splits"]),
            }
        )
    return {
        "ok": True,
        "experiment": "source_subset_robustness",
        "definition": "Checks whether cross-city conclusions hold after removing major data-source families.",
        "subsets": subsets,
        "summary": {
            "subsets": len(subsets),
            "subsets_all_splits_weighted_cfcmt_beats_h2oplus": sum(
                1 for subset in subsets if subset["summary"]["cfcmt_similarity_weighted_wins_vs_h2oplus"] == subset["summary"]["splits"]
            ),
            "subsets_all_splits_unweighted_cfcmt_beats_h2oplus": sum(
                1 for subset in subsets if subset["summary"]["cfcmt_wins_vs_h2oplus"] == subset["summary"]["splits"]
            ),
        },
    }


def _robustness_scenarios() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "group_scales": {}, "noise_fraction": 0.0},
        {"name": "headway_gap_bias_125", "group_scales": {"headway_gap": 1.25}, "noise_fraction": 0.0},
        {"name": "demand_stop_reward_bias_125", "group_scales": {"demand_stop_reward": 1.25}, "noise_fraction": 0.0},
        {"name": "speed_bias_150", "group_scales": {"speed": 1.50}, "noise_fraction": 0.0},
        {
            "name": "mixed_mechanism_bias",
            "group_scales": {"headway_gap": 1.15, "demand_stop_reward": 0.85, "speed": 1.25},
            "noise_fraction": 0.0,
        },
        {"name": "measurement_noise_10pct", "group_scales": {}, "noise_fraction": 0.10},
        {"name": "measurement_noise_25pct", "group_scales": {}, "noise_fraction": 0.25},
    ]


def run_generator_robustness(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    ridge: float,
    *,
    temperature: float,
    floor: float,
) -> dict[str, Any]:
    rows = []
    strict_splits = _strict_leave_one_city_out_splits(config)
    for scenario in _robustness_scenarios():
        output_scales = _output_scale_vector(scenario["group_scales"])
        noise_fraction = float(scenario["noise_fraction"])
        for split in strict_splits:
            sources = list(split["source_envs"])
            target = split["target_env"]
            target_stats = city_stats[target]
            weights = _source_similarity_weights(sanity, target, sources, temperature=temperature, floor=floor)
            train_unweighted = _merge_stats("source_unweighted", "source", [city_stats[key] for key in sources])
            train_weighted = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, weights)
            h2o_unweighted = _fit_h2o(train_unweighted, ridge)
            cfcmt_unweighted = _fit_cfcmt(train_unweighted, ridge)
            cfcmt_weighted = _fit_cfcmt(train_weighted, ridge)

            uncal_sse = target_stats.h2o.sse(None, target_scale=output_scales)
            h2o_sse = target_stats.h2o.sse(h2o_unweighted, target_scale=output_scales)
            cfcmt_sse = _family_sse(target_stats.cfcmt, cfcmt_unweighted, target_scale=output_scales)
            weighted_sse = _family_sse(target_stats.cfcmt, cfcmt_weighted, target_scale=output_scales)
            noise_reference_sse = uncal_sse.copy()
            uncal_sse = _add_measurement_noise(uncal_sse, noise_reference_sse, noise_fraction)
            h2o_sse = _add_measurement_noise(h2o_sse, noise_reference_sse, noise_fraction)
            cfcmt_sse = _add_measurement_noise(cfcmt_sse, noise_reference_sse, noise_fraction)
            weighted_sse = _add_measurement_noise(weighted_sse, noise_reference_sse, noise_fraction)
            metrics = _method_bundle_from_sse(
                target_stats,
                h2o_sse,
                cfcmt_sse,
                uncal_sse=uncal_sse,
                weighted_cfcmt_sse=weighted_sse,
            )
            h2o_total = metrics["h2oplus_dense"]["total_mse"]
            cfcmt_total = metrics["cfcmt_mechanism"]["total_mse"]
            weighted_total = metrics["cfcmt_similarity_weighted"]["total_mse"]
            rows.append(
                {
                    **split,
                    "scenario": scenario["name"],
                    "output_group_scales": dict(scenario["group_scales"]),
                    "noise_fraction": noise_fraction,
                    "target_city": config["generated_envs"][target].get("city", target),
                    "metrics": metrics,
                    "comparisons": {
                        "cfcmt_vs_h2oplus_ratio": cfcmt_total / h2o_total if h2o_total else None,
                        "cfcmt_similarity_weighted_vs_h2oplus_ratio": weighted_total / h2o_total if h2o_total else None,
                        "cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": weighted_total / cfcmt_total if cfcmt_total else None,
                        "cfcmt_beats_h2oplus": bool(cfcmt_total < h2o_total),
                        "cfcmt_similarity_weighted_beats_h2oplus": bool(weighted_total < h2o_total),
                        "cfcmt_similarity_weighted_beats_unweighted_cfcmt": bool(weighted_total < cfcmt_total),
                    },
                }
            )

    by_scenario = {}
    for scenario in [item["name"] for item in _robustness_scenarios()]:
        group = [row for row in rows if row["scenario"] == scenario]
        by_scenario[scenario] = {
            "splits": len(group),
            "cfcmt_wins_vs_h2oplus": sum(1 for row in group if row["comparisons"]["cfcmt_beats_h2oplus"]),
            "cfcmt_similarity_weighted_wins_vs_h2oplus": sum(
                1 for row in group if row["comparisons"]["cfcmt_similarity_weighted_beats_h2oplus"]
            ),
            "cfcmt_similarity_weighted_wins_vs_unweighted_cfcmt": sum(
                1 for row in group if row["comparisons"]["cfcmt_similarity_weighted_beats_unweighted_cfcmt"]
            ),
            "mean_cfcmt_vs_h2oplus_ratio": float(np.mean([row["comparisons"]["cfcmt_vs_h2oplus_ratio"] for row in group])),
            "mean_cfcmt_similarity_weighted_vs_h2oplus_ratio": float(
                np.mean([row["comparisons"]["cfcmt_similarity_weighted_vs_h2oplus_ratio"] for row in group])
            ),
            "mean_cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio": float(
                np.mean([row["comparisons"]["cfcmt_similarity_weighted_vs_unweighted_cfcmt_ratio"] for row in group])
            ),
        }
    return {
        "ok": True,
        "experiment": "generator_bias_noise_robustness",
        "definition": "Strict leave-one-city-out evaluation under output-group target perturbations and expected measurement noise.",
        "rows": rows,
        "summary_by_scenario": by_scenario,
        "summary": {
            "scenarios": len(by_scenario),
            "splits_per_scenario": len(strict_splits),
            "scenarios_all_splits_weighted_cfcmt_beats_h2oplus": sum(
                1 for value in by_scenario.values() if value["cfcmt_similarity_weighted_wins_vs_h2oplus"] == value["splits"]
            ),
            "scenarios_all_splits_unweighted_cfcmt_beats_h2oplus": sum(
                1 for value in by_scenario.values() if value["cfcmt_wins_vs_h2oplus"] == value["splits"]
            ),
        },
    }


def run_single_city(city_line_stats: dict[str, list[ResidualStats]], config: dict[str, Any], ridge: float, test_fraction: float, seed: int) -> dict[str, Any]:
    results = []
    for key, lines in city_line_stats.items():
        ordered = sorted(lines, key=lambda item: item.line_keys[0])
        hashes = np.array([_stable_unit(f"{seed}:{item.line_keys[0]}") for item in ordered])
        test_mask = hashes < test_fraction
        if not np.any(test_mask):
            test_mask[int(np.argmin(hashes))] = True
        if np.all(test_mask):
            test_mask[int(np.argmax(hashes))] = False
        train = [item for item, is_test in zip(ordered, test_mask) if not is_test]
        test = [item for item, is_test in zip(ordered, test_mask) if is_test]
        train_stats = _merge_stats(f"{key}::single_train", key, train)
        test_stats = _merge_stats(f"{key}::single_test", key, test)
        h2o_beta = _fit_h2o(train_stats, ridge)
        cfcmt_beta = _fit_cfcmt(train_stats, ridge)
        metrics = evaluate_stats(test_stats, h2o_beta, cfcmt_beta)
        results.append(
            {
                "city_key": key,
                "city": config["generated_envs"][key].get("city", key),
                "protocol": "route-heldout within city",
                "train_lines": len(train),
                "test_lines": len(test),
                "train_transitions": train_stats.n,
                "test_transitions": test_stats.n,
                "metrics": metrics,
                "comparisons": _comparison(metrics),
            }
        )
    wins = sum(1 for item in results if item["comparisons"]["cfcmt_beats_h2oplus"])
    ratios = [item["comparisons"]["cfcmt_vs_h2oplus_total_mse_ratio"] for item in results]
    return {
        "ok": True,
        "experiment": "single_city_route_heldout_dynamics",
        "test_fraction": test_fraction,
        "cities": results,
        "summary": {
            "cities": len(results),
            "cfcmt_wins_vs_h2oplus": wins,
            "h2oplus_wins_vs_cfcmt": len(results) - wins,
            "mean_cfcmt_vs_h2oplus_total_mse_ratio": float(np.mean(ratios)),
        },
    }


def _cross_city_eval(
    city_stats: dict[str, ResidualStats],
    split: dict[str, Any],
    ridge: float,
    *,
    cfcmt_family: str = "cfcmt_full",
    target_scale: float = 1.0,
) -> dict[str, Any]:
    train_stats = _merge_stats("source", "source", [city_stats[key] for key in split["source_envs"]])
    h2o_beta = _fit_h2o(train_stats, ridge)
    cfcmt_beta = _fit_cfcmt(train_stats, ridge, cfcmt_family)
    metrics = evaluate_stats(city_stats[split["target_env"]], h2o_beta, cfcmt_beta, cfcmt_family=cfcmt_family, target_scale=target_scale)
    return {
        **split,
        "train_transitions": train_stats.n,
        "metrics": metrics,
        "comparisons": _comparison(metrics),
    }


def run_calibration_sweep(city_stats: dict[str, ResidualStats], config: dict[str, Any], ridge: float, strengths: list[float]) -> dict[str, Any]:
    split_results = []
    for split in expand_splits(config):
        base = _cross_city_eval(city_stats, split, ridge)
        h2o_base = base["metrics"]["h2oplus_dense"]["total_mse"]
        cfcmt_no_cal = base["metrics"]["cfcmt_mechanism"]["total_mse"]
        sweep = []
        for strength in strengths:
            scale = max(0.0, 1.0 - float(strength))
            h2o_cal = h2o_base * scale * scale
            sweep.append(
                {
                    "calibration_strength": float(strength),
                    "h2oplus_calibrated_total_mse": float(h2o_cal),
                    "cfcmt_uncalibrated_total_mse": float(cfcmt_no_cal),
                    "cfcmt_uncalibrated_beats_h2oplus_calibrated": bool(cfcmt_no_cal < h2o_cal),
                    "cfcmt_uncalibrated_vs_h2oplus_calibrated_ratio": float(cfcmt_no_cal / h2o_cal) if h2o_cal > 0 else None,
                }
            )
        ratio = cfcmt_no_cal / h2o_base if h2o_base else float("nan")
        break_even = 1.0 - math.sqrt(ratio) if np.isfinite(ratio) and ratio >= 0 else None
        split_results.append(
            {
                "name": base["name"],
                "source_envs": base["source_envs"],
                "target_env": base["target_env"],
                "target_city": config["generated_envs"][base["target_env"]].get("city", base["target_env"]),
                "h2oplus_uncalibrated_total_mse": h2o_base,
                "cfcmt_uncalibrated_total_mse": cfcmt_no_cal,
                "break_even_calibration_strength_for_h2oplus": break_even,
                "sweep": sweep,
            }
        )
    return {
        "ok": True,
        "experiment": "calibration_vs_no_calibration_sweep",
        "definition": "calibration_strength shrinks simulator residual by strength; 0 means uncalibrated, 0.2 means 20% residual removal before H2O+ correction",
        "splits": split_results,
        "summary": {
            "mean_break_even_calibration_strength_for_h2oplus": float(np.mean([s["break_even_calibration_strength_for_h2oplus"] for s in split_results])),
            "strengths": strengths,
        },
    }


def run_ablation(city_stats: dict[str, ResidualStats], config: dict[str, Any], ridge: float) -> dict[str, Any]:
    family_results = {}
    for family in ABLATION_FEATURES:
        splits = []
        for split in expand_splits(config):
            out = _cross_city_eval(city_stats, split, ridge, cfcmt_family=family)
            splits.append(out)
        ratios = [item["comparisons"]["cfcmt_vs_h2oplus_total_mse_ratio"] for item in splits]
        family_results[family] = {
            "splits": splits,
            "summary": {
                "cfcmt_wins_vs_h2oplus": sum(1 for item in splits if item["comparisons"]["cfcmt_beats_h2oplus"]),
                "mean_cfcmt_vs_h2oplus_total_mse_ratio": float(np.mean(ratios)),
            },
        }
    return {
        "ok": True,
        "experiment": "cfcmt_mechanism_ablation",
        "families": family_results,
    }


def run_source_sensitivity(city_stats: dict[str, ResidualStats], config: dict[str, Any], ridge: float) -> dict[str, Any]:
    keys = list(config["generated_envs"].keys())
    rows = []
    for target in keys:
        sources = [key for key in keys if key != target]
        for size in range(1, len(sources) + 1):
            for combo in itertools.combinations(sources, size):
                split = {
                    "name": f"source_size_{size}_to_{target}::{'_'.join(combo)}",
                    "source_envs": list(combo),
                    "target_env": target,
                }
                out = _cross_city_eval(city_stats, split, ridge)
                rows.append(
                    {
                        "source_size": size,
                        "source_envs": list(combo),
                        "target_env": target,
                        "target_city": config["generated_envs"][target].get("city", target),
                        "train_transitions": out["train_transitions"],
                        "target_transitions": out["metrics"]["target_transitions"],
                        "comparisons": out["comparisons"],
                        "metrics": out["metrics"],
                    }
                )
    summary_by_size = {}
    for size in sorted({row["source_size"] for row in rows}):
        group = [row for row in rows if row["source_size"] == size]
        summary_by_size[str(size)] = {
            "splits": len(group),
            "cfcmt_wins_vs_h2oplus": sum(1 for row in group if row["comparisons"]["cfcmt_beats_h2oplus"]),
            "mean_cfcmt_vs_h2oplus_total_mse_ratio": float(np.mean([row["comparisons"]["cfcmt_vs_h2oplus_total_mse_ratio"] for row in group])),
        }
    return {
        "ok": True,
        "experiment": "source_city_sensitivity",
        "rows": rows,
        "summary_by_source_size": summary_by_size,
    }


def _line_total_mse(line: ResidualStats, h2o_beta: np.ndarray, cfcmt_beta: dict[str, np.ndarray]) -> tuple[float, float, int]:
    metrics = evaluate_stats(line, h2o_beta, cfcmt_beta)
    return (
        metrics["h2oplus_dense"]["total_mse"],
        metrics["cfcmt_mechanism"]["total_mse"],
        line.n,
    )


def run_bootstrap(city_stats: dict[str, ResidualStats], city_line_stats: dict[str, list[ResidualStats]], config: dict[str, Any], ridge: float, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = []
    for target in config["generated_envs"]:
        sources = [key for key in config["generated_envs"] if key != target]
        train_stats = _merge_stats("source", "source", [city_stats[key] for key in sources])
        h2o_beta = _fit_h2o(train_stats, ridge)
        cfcmt_beta = _fit_cfcmt(train_stats, ridge)
        line_values = [_line_total_mse(line, h2o_beta, cfcmt_beta) for line in city_line_stats[target]]
        h2o = np.asarray([item[0] for item in line_values], dtype=np.float64)
        cfcmt = np.asarray([item[1] for item in line_values], dtype=np.float64)
        weights = np.asarray([item[2] for item in line_values], dtype=np.float64)
        n_lines = len(line_values)
        sampled_diff = []
        sampled_ratio = []
        for _ in range(samples):
            idx = rng.integers(0, n_lines, size=n_lines)
            w = weights[idx]
            h = float(np.sum(h2o[idx] * w) / np.sum(w))
            c = float(np.sum(cfcmt[idx] * w) / np.sum(w))
            sampled_diff.append(c - h)
            sampled_ratio.append(c / h if h > 0 else np.nan)
        empirical_h = float(np.sum(h2o * weights) / np.sum(weights))
        empirical_c = float(np.sum(cfcmt * weights) / np.sum(weights))
        rows.append(
            {
                "target_env": target,
                "target_city": config["generated_envs"][target].get("city", target),
                "source_envs": sources,
                "lines": n_lines,
                "h2oplus_total_mse": empirical_h,
                "cfcmt_total_mse": empirical_c,
                "cfcmt_minus_h2oplus_total_mse": empirical_c - empirical_h,
                "cfcmt_vs_h2oplus_total_mse_ratio": empirical_c / empirical_h if empirical_h > 0 else None,
                "diff_ci95": [float(v) for v in np.quantile(sampled_diff, [0.025, 0.975])],
                "ratio_ci95": [float(v) for v in np.nanquantile(sampled_ratio, [0.025, 0.975])],
                "bootstrap_samples": samples,
            }
        )
    return {
        "ok": True,
        "experiment": "route_level_bootstrap_leave_one_city_out",
        "rows": rows,
        "summary": {
            "targets": len(rows),
            "ci_excludes_zero_in_cfcmt_favor": sum(1 for row in rows if row["diff_ci95"][1] < 0.0),
            "ci_excludes_one_in_cfcmt_favor": sum(1 for row in rows if row["ratio_ci95"][1] < 1.0),
        },
    }


def probe_sumo_readiness(config: dict[str, Any], root: Path) -> dict[str, Any]:
    traci_import = False
    for candidate in [os.environ.get("SUMO_HOME", "") + "/tools" if os.environ.get("SUMO_HOME") else "", "/usr/share/sumo/tools"]:
        if candidate and os.path.isdir(candidate):
            sys.path.insert(0, candidate)
            try:
                import traci  # noqa: F401

                traci_import = True
                break
            except Exception:
                pass
    city_configs = {}
    for key, spec in config["generated_envs"].items():
        env_path = _resolve_path(root, spec["env_path"])
        sumocfg = sorted(str(path) for path in env_path.rglob("*.sumocfg"))
        city_configs[key] = {
            "env_path": str(env_path),
            "sumocfg_count": len(sumocfg),
            "sample_sumocfg": sumocfg[:3],
        }
    return {
        "sumo_binary": shutil.which("sumo"),
        "sumo_gui_binary": shutil.which("sumo-gui"),
        "traci_import_available": traci_import,
        "city_sumocfg": city_configs,
        "can_run_generated_city_sumo_rollout": bool(traci_import and all(item["sumocfg_count"] > 0 for item in city_configs.values())),
    }


def _rollout_obs_vectors(obs: dict[Any, Any]) -> dict[int, np.ndarray]:
    vectors = {}
    for key, value in obs.items():
        if not value:
            continue
        arr = np.asarray(value[-1], dtype=np.float32)
        if arr.shape == (15,):
            vectors[int(key)] = arr
    return vectors


def _predict_action_from_obs(
    obs: np.ndarray,
    method: str,
    h2o_beta: np.ndarray,
    cfcmt_beta: dict[str, np.ndarray],
    actions: list[float],
) -> float:
    if method == "no_hold":
        return 0.0
    if method == "fixed_30":
        return 30.0
    obs_mat = np.repeat(obs[None, :], len(actions), axis=0).astype(np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    obs_action = np.concatenate([obs_mat, action_arr[:, None]], axis=1)
    fwd = obs_mat[:, 5].astype(np.float64)
    bwd = obs_mat[:, 6].astype(np.float64)
    waiting = np.maximum(obs_mat[:, 7].astype(np.float64), 0.0)
    target = np.maximum(obs_mat[:, 8].astype(np.float64), 60.0)
    speed = np.clip(obs_mat[:, 14].astype(np.float64), 1.0, 25.0)
    travel_time = np.clip(target / 4.0, 30.0, 600.0)
    hour = int(max(0.0, float(obs[10])) // 3600) % 24
    sim_y = np.column_stack(_uncalibrated_transition(fwd, bwd, waiting, target, speed, travel_time, action_arr, hour))
    if method == "h2oplus_dense_policy":
        pred_y = sim_y + h2o_features(obs_action) @ h2o_beta
    elif method == "cfcmt_mechanism_policy":
        pred_y = sim_y.copy()
        for idx, output_name in enumerate(OUTPUT_NAMES):
            if output_name == "reward":
                continue
            pred_y[:, idx] += cfcmt_features(output_name, obs_action) @ cfcmt_beta[output_name]
    else:
        pred_y = sim_y
    reward = _reward_from_predicted_state(pred_y, obs_action)
    return float(actions[int(np.argmax(reward))])


def run_sampled_rollout(
    city_stats: dict[str, ResidualStats],
    config: dict[str, Any],
    sanity: dict[str, Any],
    root: Path,
    ridge: float,
    actions: list[float],
    lines_per_city: int,
    max_decisions: int,
    seed: int,
    *,
    source_weight_temperature: float,
    source_weight_floor: float,
) -> dict[str, Any]:
    sys.path.insert(0, str(root / "H2Oplus" / "bus_h2o"))
    from envs.bus_sim_env import BusSimEnv  # noqa: WPS433

    policies = [
        "no_hold",
        "fixed_30",
        "h2oplus_dense_policy",
        "cfcmt_mechanism_policy",
        "cfcmt_similarity_weighted_policy",
    ]
    rows = []
    for target, spec in config["generated_envs"].items():
        sources = [key for key in config["generated_envs"] if key != target]
        train_stats = _merge_stats("source", "source", [city_stats[key] for key in sources])
        source_weights = _source_similarity_weights(
            sanity,
            target,
            sources,
            temperature=source_weight_temperature,
            floor=source_weight_floor,
        )
        train_weighted = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, source_weights)
        h2o_beta = _fit_h2o(train_stats, ridge)
        cfcmt_beta = _fit_cfcmt(train_stats, ridge)
        cfcmt_weighted_beta = _fit_cfcmt(train_weighted, ridge)
        env_path = _resolve_path(root, spec["env_path"])
        line_env_root = env_path / "_line_envs"
        if not line_env_root.exists():
            rows.append(
                {
                    "target_env": target,
                    "target_city": spec.get("city", target),
                    "line_env": None,
                    "policy": None,
                    "source_envs": sources,
                    "decisions": 0,
                    "skipped": True,
                    "skip_reason": f"missing line env directory: {line_env_root}",
                }
            )
            continue
        line_envs = [path for path in sorted(line_env_root.iterdir()) if path.is_dir() and (path / "config.json").exists()]
        if not line_envs:
            rows.append(
                {
                    "target_env": target,
                    "target_city": spec.get("city", target),
                    "line_env": None,
                    "policy": None,
                    "source_envs": sources,
                    "decisions": 0,
                    "skipped": True,
                    "skip_reason": f"no runnable line envs under: {line_env_root}",
                }
            )
            continue
        for line_env in line_envs[:lines_per_city]:
            for policy in policies:
                np.random.seed(seed)
                env = BusSimEnv(path=str(line_env))
                action_dict = {agent: 0.0 for agent in range(env.max_agent_num)}
                env.reset()
                decisions = 0
                reward_sum = 0.0
                headway_abs_sum = 0.0
                hold_sum = 0.0
                done = False
                t0 = time.time()
                while decisions < max_decisions and not done:
                    obs, rew, done = env.step_to_event(action_dict)
                    vectors = _rollout_obs_vectors(obs)
                    if not vectors:
                        continue
                    action_dict = {}
                    for agent, vec in vectors.items():
                        model_cfcmt_beta = cfcmt_weighted_beta if policy == "cfcmt_similarity_weighted_policy" else cfcmt_beta
                        model_policy = "cfcmt_mechanism_policy" if policy == "cfcmt_similarity_weighted_policy" else policy
                        action = _predict_action_from_obs(vec, model_policy, h2o_beta, model_cfcmt_beta, actions)
                        action_dict[agent] = action
                        reward_sum += float(rew.get(agent, 0.0))
                        target_hw = max(float(vec[8]), 60.0)
                        headway_abs_sum += 0.5 * (abs(float(vec[5]) - target_hw) + abs(float(vec[6]) - target_hw))
                        hold_sum += action
                        decisions += 1
                        if decisions >= max_decisions:
                            break
                denom = max(1, decisions)
                rows.append(
                    {
                        "target_env": target,
                        "target_city": spec.get("city", target),
                        "line_env": line_env.name,
                        "policy": policy,
                        "source_envs": sources,
                        "decisions": decisions,
                        "done": bool(done),
                        "elapsed_sec": time.time() - t0,
                        "mean_reward": reward_sum / denom,
                        "mean_headway_abs_error": headway_abs_sum / denom,
                        "mean_hold_seconds": hold_sum / denom,
                    }
                )
    by_policy = {}
    for policy in policies:
        group = [row for row in rows if row.get("policy") == policy and not row.get("skipped")]
        by_policy[policy] = {
            "episodes": len(group),
            "mean_reward": float(np.mean([row["mean_reward"] for row in group])) if group else None,
            "mean_headway_abs_error": float(np.mean([row["mean_headway_abs_error"] for row in group])) if group else None,
            "mean_hold_seconds": float(np.mean([row["mean_hold_seconds"] for row in group])) if group else None,
        }
    return {
        "ok": True,
        "experiment": "sampled_h2o_sim_live_rollout",
        "live_sumo_rollout": False,
        "note": "Generated four-city bundles do not include SUMO .sumocfg files; this uses H2O BusSimEnv live event rollout as the executable rollout substitute.",
        "lines_per_city": lines_per_city,
        "max_decisions_per_episode": max_decisions,
        "rows": rows,
        "summary_by_policy": by_policy,
    }


def run_efficiency(city_stats: dict[str, ResidualStats], config: dict[str, Any], ridge: float) -> dict[str, Any]:
    dummy = np.zeros((1, 16), dtype=np.float32)
    feature_dims = {
        "h2oplus_dense": int(h2o_features(dummy).shape[1]),
        "cfcmt_full_total": int(sum(cfcmt_features(name, dummy).shape[1] for name in OUTPUT_NAMES)),
        "cfcmt_full_by_output": {name: int(cfcmt_features(name, dummy).shape[1]) for name in OUTPUT_NAMES},
    }
    for name, fn in ABLATION_FEATURES.items():
        feature_dims[f"{name}_total"] = int(sum(fn(output, dummy).shape[1] for output in OUTPUT_NAMES))

    fit_rows = []
    for split in expand_splits(config):
        train_stats = _merge_stats("source", "source", [city_stats[key] for key in split["source_envs"]])
        t0 = time.time()
        _fit_h2o(train_stats, ridge)
        h2o_sec = time.time() - t0
        t0 = time.time()
        _fit_cfcmt(train_stats, ridge)
        cfcmt_sec = time.time() - t0
        fit_rows.append(
            {
                "split": split["name"],
                "source_envs": split["source_envs"],
                "target_env": split["target_env"],
                "train_transitions": train_stats.n,
                "h2oplus_solve_sec": h2o_sec,
                "cfcmt_solve_sec": cfcmt_sec,
            }
        )
    return {
        "ok": True,
        "experiment": "efficiency",
        "feature_dims": feature_dims,
        "city_transitions": {key: value.n for key, value in city_stats.items()},
        "fit_rows": fit_rows,
        "summary": {
            "mean_h2oplus_solve_sec": float(np.mean([row["h2oplus_solve_sec"] for row in fit_rows])),
            "mean_cfcmt_solve_sec": float(np.mean([row["cfcmt_solve_sec"] for row in fit_rows])),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    city_stats: dict[str, ResidualStats] = {}
    city_line_stats: dict[str, list[ResidualStats]] = {}
    sanity: dict[str, Any] = {}
    t0 = time.time()
    for key, spec in config["generated_envs"].items():
        print(f"[build] {key}: streaming route-level stats", flush=True)
        aggregate, lines, city_sanity = build_city_stats(key, spec, root, args)
        city_stats[key] = aggregate
        city_line_stats[key] = lines
        sanity[key] = city_sanity
        print(f"[build] {key}: lines={aggregate.lines_seen}, rows={aggregate.n}", flush=True)

    experiments = {
        "single_city": run_single_city(city_line_stats, config, args.ridge, args.single_city_test_fraction, args.seed),
        "strict_leave_one_city_out": run_strict_leave_one_out(
            city_stats,
            config,
            sanity,
            args.ridge,
            temperature=args.source_weight_temperature,
            floor=args.source_weight_floor,
        ),
        "calibration_vs_no_calibration": run_calibration_sweep(city_stats, config, args.ridge, args.calibration_strengths),
        "ablation": run_ablation(city_stats, config, args.ridge),
        "source_sensitivity": run_source_sensitivity(city_stats, config, args.ridge),
        "source_similarity_weighting": run_source_weighting(
            city_stats,
            config,
            sanity,
            args.ridge,
            temperature=args.source_weight_temperature,
            floor=args.source_weight_floor,
        ),
        "source_weighting_sensitivity": run_source_weighting_sensitivity(
            city_stats,
            config,
            sanity,
            args.ridge,
            temperatures=args.source_weight_temperatures,
            floors=args.source_weight_floors,
            default_temperature=args.source_weight_temperature,
            default_floor=args.source_weight_floor,
        ),
        "source_subset_robustness": run_source_subset_robustness(
            city_stats,
            config,
            sanity,
            args.ridge,
            temperature=args.source_weight_temperature,
            floor=args.source_weight_floor,
        ),
        "generator_robustness": run_generator_robustness(
            city_stats,
            config,
            sanity,
            args.ridge,
            temperature=args.source_weight_temperature,
            floor=args.source_weight_floor,
        ),
        "bootstrap": run_bootstrap(city_stats, city_line_stats, config, args.ridge, args.bootstrap_samples, args.seed),
        "data_sanity": {
            "ok": True,
            "experiment": "static_data_sanity",
            "cities": sanity,
        },
        "efficiency": run_efficiency(city_stats, config, args.ridge),
        "sumo_readiness": probe_sumo_readiness(config, root),
    }
    if not args.skip_rollout and args.rollout_lines_per_city > 0:
        experiments["sampled_rollout"] = run_sampled_rollout(
            city_stats,
            config,
            sanity,
            root,
            args.ridge,
            args.policy_actions,
            args.rollout_lines_per_city,
            args.rollout_max_decisions,
            args.seed,
            source_weight_temperature=args.source_weight_temperature,
            source_weight_floor=args.source_weight_floor,
        )

    existing_policy_path = _resolve_path(root, args.existing_policy_result)
    if existing_policy_path.exists():
        experiments["existing_cross_city_policy_validation"] = json.loads(existing_policy_path.read_text(encoding="utf-8"))

    return {
        "ok": True,
        "validation_level": "paper_experiment_suite",
        "config": str(_resolve_path(root, args.config)),
        "elapsed_sec": time.time() - t0,
        "actions_seconds": args.actions,
        "ridge": args.ridge,
        "max_lines_per_city": args.max_lines_per_city,
        "experiments": experiments,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/paper_experiment_suite.json"))
    parser.add_argument("--existing-policy-result", type=Path, default=Path("cf_h2o/results/cross_city_policy_validation.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--actions", type=_parse_float_list, default=[0.0, 30.0])
    parser.add_argument("--policy-actions", type=_parse_float_list, default=list(POLICY_ACTIONS))
    parser.add_argument("--max-lines-per-city", type=int, default=0, help="0 means all lines; >0 for smoke")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--single-city-test-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-strengths", type=_parse_float_list, default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--source-weight-temperature", type=float, default=1.0)
    parser.add_argument("--source-weight-floor", type=float, default=0.05)
    parser.add_argument("--source-weight-temperatures", type=_parse_float_list, default=[0.5, 1.0, 2.0])
    parser.add_argument("--source-weight-floors", type=_parse_float_list, default=[0.0, 0.05, 0.10])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--rollout-lines-per-city", type=int, default=1)
    parser.add_argument("--rollout-max-decisions", type=int, default=120)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Write full JSON to --out but print only a compact summary")
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.quiet:
        compact = {
            "ok": result["ok"],
            "elapsed_sec": result["elapsed_sec"],
            "experiments": sorted(result["experiments"].keys()),
        }
        for key in (
            "single_city",
            "strict_leave_one_city_out",
            "calibration_vs_no_calibration",
            "ablation",
            "source_sensitivity",
            "source_similarity_weighting",
            "source_weighting_sensitivity",
            "source_subset_robustness",
            "generator_robustness",
            "bootstrap",
            "sampled_rollout",
        ):
            value = result["experiments"].get(key)
            if isinstance(value, dict) and "summary" in value:
                compact[key] = value["summary"]
            elif key == "sampled_rollout" and isinstance(value, dict):
                compact[key] = value.get("summary_by_policy")
        print(json.dumps(compact, indent=2))
    else:
        print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
