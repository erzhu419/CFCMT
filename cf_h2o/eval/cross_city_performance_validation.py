"""Cross-city H2O+ vs CFCMT performance validation on open-transit bundles.

This is a static-offline dynamics validation, not a policy rollout benchmark.
It streams every generated line directory in each city bundle, builds synthetic
offline transition targets from the same route/demand/schedule tables used by
H2O+, and compares:

* uncalibrated simulator target
* H2O+ style dense monolithic dynamics-gap correction
* CFCMT style sparse mechanism-wise dynamics-gap correction

The script is intentionally full-route by default. Use ``--max-lines-per-city``
only for development smoke checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_open_transit_validation import expand_splits


OBS_NAMES = [
    "line_id",
    "bus_id",
    "station_id",
    "time_period",
    "direction",
    "forward_headway",
    "backward_headway",
    "waiting_passengers",
    "target_headway",
    "base_stop_duration",
    "sim_time",
    "gap",
    "co_line_forward_headway",
    "co_line_backward_headway",
    "segment_mean_speed",
]

DYNAMIC_OUTPUT_INDICES = [5, 6, 7, 9, 11, 12, 13, 14]
OUTPUT_NAMES = [f"next_{OBS_NAMES[idx]}" for idx in DYNAMIC_OUTPUT_INDICES] + ["reward"]
HOURS = list(range(24))
DEFAULT_ACTIONS = [0.0, 30.0]
REQUIRED_LINE_FILES = {"stop_news.xlsx", "route_news.xlsx", "time_table.xlsx", "passenger_OD.xlsx"}


@dataclass
class RidgeStats:
    xtx: np.ndarray
    xty: np.ndarray
    n: int = 0

    @classmethod
    def zeros(cls, feature_dim: int, output_dim: int = 1) -> "RidgeStats":
        return cls(
            xtx=np.zeros((feature_dim, feature_dim), dtype=np.float64),
            xty=np.zeros((feature_dim, output_dim), dtype=np.float64),
            n=0,
        )

    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += int(x.shape[0])

    def merge(self, other: "RidgeStats") -> None:
        self.xtx += other.xtx
        self.xty += other.xty
        self.n += other.n

    def solve(self, ridge: float) -> np.ndarray:
        reg = float(ridge) * np.eye(self.xtx.shape[0], dtype=np.float64)
        reg[0, 0] = 0.0
        return np.linalg.solve(self.xtx + reg, self.xty)


@dataclass
class CityStats:
    key: str
    city: str
    env_path: Path
    lines_seen: int
    transitions_seen: int
    h2o: RidgeStats
    cfcmt: dict[str, RidgeStats]
    warnings: list[str]


@dataclass
class ModelBundle:
    h2o_beta: np.ndarray
    cfcmt_beta: dict[str, np.ndarray]
    train_transitions: int
    source_envs: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_unit(value: str, modulus: int = 10000) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return float(int(digest[:10], 16) % modulus) / float(modulus)


def _line_dirs(env_path: Path, max_lines: int) -> list[Path]:
    data_dir = env_path / "data"
    line_dirs = [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and REQUIRED_LINE_FILES.issubset({item.name for item in child.iterdir()})
    ]
    if max_lines > 0:
        return line_dirs[:max_lines]
    return line_dirs


def _hour_col(hour: int) -> str:
    return f"{int(hour):02d}:00:00"


def _parse_direction(line_key: str, timetable: pd.DataFrame) -> float:
    match = re.search(r"D(\d+)", line_key)
    if match:
        return float(int(match.group(1)) % 2)
    if "direction" in timetable.columns and len(timetable):
        return float(int(pd.to_numeric(timetable["direction"], errors="coerce").fillna(1).iloc[0]) > 0)
    return 1.0


def _headways_by_hour(timetable: pd.DataFrame) -> dict[int, float]:
    if timetable.empty or "launch_time" not in timetable.columns:
        return {hour: 600.0 for hour in HOURS}
    launch = pd.to_numeric(timetable["launch_time"], errors="coerce").dropna().astype(float)
    launch = launch[(launch >= 0.0) & (launch < 24.0 * 3600.0)]
    if launch.empty:
        return {hour: 600.0 for hour in HOURS}
    global_diffs = np.diff(np.sort(launch.to_numpy()))
    global_diffs = global_diffs[global_diffs > 1.0]
    fallback = float(np.median(global_diffs)) if len(global_diffs) else 600.0
    fallback = float(np.clip(fallback, 120.0, 3600.0))
    result: dict[int, float] = {}
    for hour in HOURS:
        sub = np.sort(launch[(launch >= hour * 3600.0) & (launch < (hour + 1) * 3600.0)].to_numpy())
        if len(sub) >= 2:
            diffs = np.diff(sub)
            diffs = diffs[diffs > 1.0]
            value = float(np.median(diffs)) if len(diffs) else 3600.0 / max(len(sub), 1)
        elif len(sub) == 1:
            value = fallback
        else:
            value = fallback
        result[hour] = float(np.clip(value, 60.0, 7200.0))
    return result


def _od_demand_by_hour_stop(passenger_od: pd.DataFrame) -> dict[tuple[int, str], float]:
    if passenger_od.empty or "time_period" not in passenger_od.columns or "stop_name" not in passenger_od.columns:
        return {}
    value_cols = [col for col in passenger_od.columns if col not in {"time_period", "stop_name"}]
    if not value_cols:
        return {}
    rows = passenger_od[["time_period", "stop_name"]].copy()
    demand = passenger_od[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    hours = rows["time_period"].astype(str).str.slice(0, 2).apply(lambda x: int(x) if x.isdigit() else 0)
    return {
        (int(hour), str(stop)): float(value)
        for hour, stop, value in zip(hours, rows["stop_name"].astype(str), demand)
    }


def _read_line_tables(line_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route = pd.read_excel(line_dir / "route_news.xlsx")
    timetable = pd.read_excel(line_dir / "time_table.xlsx")
    passenger_od = pd.read_excel(line_dir / "passenger_OD.xlsx")
    return route, timetable, passenger_od


def _iter_city_chunks(
    env_path: Path,
    *,
    max_lines: int,
    actions: list[float],
    chunk_rows: int = 65536,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]]:
    """Yield obs/action, uncalibrated target, real target chunks for one city."""

    buffer_obs: list[np.ndarray] = []
    buffer_actions: list[np.ndarray] = []
    buffer_sim: list[np.ndarray] = []
    buffer_real: list[np.ndarray] = []
    line_count = 0
    for line_dir in _line_dirs(env_path, max_lines):
        line_key = line_dir.name
        try:
            route, timetable, passenger_od = _read_line_tables(line_dir)
        except Exception:
            continue
        if route.empty:
            continue
        line_count += 1
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
            continue

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

            for hold in actions:
                action = np.full(nseg, float(hold), dtype=np.float64)
                real_fwd, real_bwd, real_waiting, real_stop, real_gap, real_co_fwd, real_co_bwd, real_speed, real_reward = _real_transition(
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
                sim_fwd, sim_bwd, sim_waiting, sim_stop, sim_gap, sim_co_fwd, sim_co_bwd, sim_speed, sim_reward = _uncalibrated_transition(
                    fwd,
                    bwd,
                    waiting,
                    target,
                    speed,
                    travel_time,
                    action,
                    hour,
                )
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
                sim_y = np.column_stack(
                    [sim_fwd, sim_bwd, sim_waiting, sim_stop, sim_gap, sim_co_fwd, sim_co_bwd, sim_speed, sim_reward]
                ).astype(np.float32)
                real_y = np.column_stack(
                    [real_fwd, real_bwd, real_waiting, real_stop, real_gap, real_co_fwd, real_co_bwd, real_speed, real_reward]
                ).astype(np.float32)
                buffer_obs.append(obs)
                buffer_actions.append(action[:, None].astype(np.float32))
                buffer_sim.append(sim_y)
                buffer_real.append(real_y)
                if sum(item.shape[0] for item in buffer_obs) >= chunk_rows:
                    yield _flush(buffer_obs, buffer_actions, buffer_sim, buffer_real, line_count)
                    buffer_obs, buffer_actions, buffer_sim, buffer_real = [], [], [], []

    if buffer_obs:
        yield _flush(buffer_obs, buffer_actions, buffer_sim, buffer_real, line_count)


def _flush(
    obs: list[np.ndarray],
    actions: list[np.ndarray],
    sim: list[np.ndarray],
    real: list[np.ndarray],
    lines_seen: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    obs_arr = np.concatenate(obs, axis=0)
    action_arr = np.concatenate(actions, axis=0)
    sim_arr = np.concatenate(sim, axis=0)
    real_arr = np.concatenate(real, axis=0)
    return np.concatenate([obs_arr, action_arr], axis=1), sim_arr, real_arr, {"lines_seen": lines_seen, "rows": obs_arr.shape[0]}


def _real_transition(
    fwd: np.ndarray,
    bwd: np.ndarray,
    waiting: np.ndarray,
    stop_duration: np.ndarray,
    target: np.ndarray,
    co_fwd: np.ndarray,
    co_bwd: np.ndarray,
    speed: np.ndarray,
    next_speed: np.ndarray,
    next_demand: np.ndarray,
    travel_time: np.ndarray,
    action: np.ndarray,
    peak: float,
) -> tuple[np.ndarray, ...]:
    speed_delta = (next_speed - speed) / np.maximum(speed, 1.0)
    real_fwd = np.clip(fwd + 0.56 * action - 0.10 * travel_time * speed_delta + 0.012 * waiting, 10.0, 7200.0)
    real_bwd = np.clip(bwd - 0.43 * action + 0.06 * travel_time * speed_delta - 0.006 * waiting, 10.0, 7200.0)
    real_waiting = np.clip(0.58 * next_demand * peak + 0.22 * waiting + 0.018 * real_fwd - 0.055 * action, 0.0, None)
    real_stop = np.clip(7.5 + 0.32 * np.sqrt(real_waiting + 1.0) + 0.017 * real_waiting + 0.05 * action, 1.0, 300.0)
    real_speed = np.clip(next_speed * (1.0 - np.minimum(real_waiting, 240.0) * 0.00045), 0.8, 25.0)
    real_gap = real_fwd - real_bwd
    real_co_fwd = np.clip(0.82 * co_fwd + 0.18 * real_fwd, 10.0, 7200.0)
    real_co_bwd = np.clip(0.82 * co_bwd + 0.18 * real_bwd, 10.0, 7200.0)
    reward = _reward(real_fwd, real_bwd, real_waiting, target, action)
    return real_fwd, real_bwd, real_waiting, real_stop, real_gap, real_co_fwd, real_co_bwd, real_speed, reward


def _uncalibrated_transition(
    fwd: np.ndarray,
    bwd: np.ndarray,
    waiting: np.ndarray,
    target: np.ndarray,
    speed: np.ndarray,
    travel_time: np.ndarray,
    action: np.ndarray,
    hour: int,
) -> tuple[np.ndarray, ...]:
    generic_peak = 1.0 + 0.20 * math.exp(-((hour - 9.0) / 4.0) ** 2)
    generic_speed = 8.0 + 1.5 * math.sin((hour / 24.0) * math.tau)
    sim_speed = np.clip(0.35 * speed + 0.65 * generic_speed, 0.8, 25.0)
    sim_fwd = np.clip(target + 0.28 * action + 0.030 * (travel_time - 90.0), 10.0, 7200.0)
    sim_bwd = np.clip(target - 0.22 * action - 0.018 * (travel_time - 90.0), 10.0, 7200.0)
    sim_waiting = np.clip(0.48 * waiting + 0.025 * target * generic_peak - 0.025 * action, 0.0, None)
    sim_stop = np.clip(7.0 + 0.22 * np.sqrt(sim_waiting + 1.0) + 0.010 * sim_waiting + 0.035 * action, 1.0, 300.0)
    sim_gap = sim_fwd - sim_bwd
    sim_co_fwd = np.clip(0.9 * target + 0.1 * fwd, 10.0, 7200.0)
    sim_co_bwd = np.clip(0.9 * target + 0.1 * bwd, 10.0, 7200.0)
    reward = _reward(sim_fwd, sim_bwd, sim_waiting, target, action)
    return sim_fwd, sim_bwd, sim_waiting, sim_stop, sim_gap, sim_co_fwd, sim_co_bwd, sim_speed, reward


def _reward(fwd: np.ndarray, bwd: np.ndarray, waiting: np.ndarray, target: np.ndarray, action: np.ndarray) -> np.ndarray:
    target = np.maximum(target, 60.0)
    headway_penalty = np.abs(fwd - target) / target + np.abs(bwd - target) / target
    return -(headway_penalty + 0.018 * np.log1p(waiting) + 0.004 * action)


def _scaled_inputs(obs_action: np.ndarray) -> np.ndarray:
    x = obs_action.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0] = x[:, 0]
    out[:, 1] = x[:, 1]
    out[:, 2] = x[:, 2]
    out[:, 3] = x[:, 3]
    out[:, 4] = x[:, 4]
    out[:, 5] = x[:, 5] / 900.0
    out[:, 6] = x[:, 6] / 900.0
    out[:, 7] = np.log1p(np.maximum(x[:, 7], 0.0)) / 5.0
    out[:, 8] = x[:, 8] / 900.0
    out[:, 9] = x[:, 9] / 80.0
    out[:, 10] = x[:, 10] / 86400.0
    out[:, 11] = x[:, 11] / 900.0
    out[:, 12] = x[:, 12] / 900.0
    out[:, 13] = x[:, 13] / 900.0
    out[:, 14] = x[:, 14] / 15.0
    out[:, 15] = x[:, 15] / 60.0
    return np.nan_to_num(out, copy=False)


def h2o_features(obs_action: np.ndarray) -> np.ndarray:
    x = _scaled_inputs(obs_action)
    action = x[:, 15:16]
    selected = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15]]
    pieces = [
        np.ones((x.shape[0], 1), dtype=np.float64),
        x,
        x * x,
        x * action,
    ]
    pairwise = []
    for i in range(selected.shape[1]):
        for j in range(i + 1, selected.shape[1]):
            pairwise.append((selected[:, i] * selected[:, j])[:, None])
    if pairwise:
        pieces.append(np.concatenate(pairwise, axis=1))
    return np.concatenate(pieces, axis=1)


def cfcmt_features(output_name: str, obs_action: np.ndarray) -> np.ndarray:
    x = _scaled_inputs(obs_action)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    hour = x[:, 3:4]
    hour_sin = np.sin(hour * math.tau)
    hour_cos = np.cos(hour * math.tau)
    action = x[:, 15:16]
    if "headway" in output_name or "gap" in output_name:
        cols = [x[:, 5:6], x[:, 6:7], x[:, 8:9], x[:, 11:12], x[:, 14:15], action, hour_sin, hour_cos]
    elif "waiting" in output_name:
        cols = [x[:, 7:8], x[:, 8:9], x[:, 14:15], action, hour_sin, hour_cos, x[:, 2:3]]
    elif "base_stop_duration" in output_name:
        cols = [x[:, 7:8], x[:, 9:10], x[:, 14:15], action, hour_sin, hour_cos]
    elif "segment_mean_speed" in output_name:
        cols = [x[:, 14:15], x[:, 10:11], hour_sin, hour_cos, x[:, 2:3]]
    elif output_name == "reward":
        abs_fwd = np.abs(x[:, 5:6] - x[:, 8:9])
        abs_bwd = np.abs(x[:, 6:7] - x[:, 8:9])
        cols = [abs_fwd, abs_bwd, x[:, 7:8], x[:, 11:12], x[:, 14:15], action, hour_sin, hour_cos]
    else:
        cols = [x[:, 5:6], x[:, 6:7], x[:, 7:8], x[:, 8:9], x[:, 14:15], action, hour_sin, hour_cos]
    base = np.concatenate([ones] + cols, axis=1)
    return np.concatenate([base, base[:, 1:] * action], axis=1)


def empty_stats(key: str, city: str, env_path: Path) -> CityStats:
    h2o_dim = h2o_features(np.zeros((1, 16), dtype=np.float32)).shape[1]
    h2o = RidgeStats.zeros(h2o_dim, len(OUTPUT_NAMES))
    cfcmt = {
        name: RidgeStats.zeros(cfcmt_features(name, np.zeros((1, 16), dtype=np.float32)).shape[1], 1)
        for name in OUTPUT_NAMES
    }
    return CityStats(key=key, city=city, env_path=env_path, lines_seen=0, transitions_seen=0, h2o=h2o, cfcmt=cfcmt, warnings=[])


def accumulate_city_stats(key: str, spec: dict[str, Any], root: Path, args: argparse.Namespace) -> CityStats:
    env_path = _resolve_path(root, spec["env_path"])
    stats = empty_stats(key, str(spec.get("city", key)), env_path)
    t0 = time.time()
    for obs_action, sim_y, real_y, meta in _iter_city_chunks(
        env_path,
        max_lines=args.max_lines_per_city,
        actions=args.actions,
        chunk_rows=args.chunk_rows,
    ):
        residual = real_y.astype(np.float64) - sim_y.astype(np.float64)
        stats.h2o.add(h2o_features(obs_action), residual)
        for idx, output_name in enumerate(OUTPUT_NAMES):
            stats.cfcmt[output_name].add(cfcmt_features(output_name, obs_action), residual[:, idx])
        stats.transitions_seen += int(obs_action.shape[0])
        stats.lines_seen = max(stats.lines_seen, int(meta["lines_seen"]))
    if stats.transitions_seen == 0:
        stats.warnings.append("no generated transitions")
    stats.warnings.append(f"stats_elapsed_sec={time.time() - t0:.1f}")
    return stats


def fit_model(source_envs: list[str], city_stats: dict[str, CityStats], ridge: float) -> ModelBundle:
    h2o_dim = next(iter(city_stats.values())).h2o.xtx.shape[0]
    merged_h2o = RidgeStats.zeros(h2o_dim, len(OUTPUT_NAMES))
    merged_cfcmt = {
        name: RidgeStats.zeros(next(iter(city_stats.values())).cfcmt[name].xtx.shape[0], 1)
        for name in OUTPUT_NAMES
    }
    train_n = 0
    for key in source_envs:
        city = city_stats[key]
        merged_h2o.merge(city.h2o)
        for name in OUTPUT_NAMES:
            merged_cfcmt[name].merge(city.cfcmt[name])
        train_n += city.transitions_seen
    return ModelBundle(
        h2o_beta=merged_h2o.solve(ridge),
        cfcmt_beta={name: stats.solve(ridge).reshape(-1) for name, stats in merged_cfcmt.items()},
        train_transitions=train_n,
        source_envs=source_envs,
    )


def evaluate_model(target_key: str, spec: dict[str, Any], root: Path, model: ModelBundle, args: argparse.Namespace) -> dict[str, Any]:
    env_path = _resolve_path(root, spec["env_path"])
    sse = {
        "uncalibrated": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
        "h2oplus_dense": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
        "cfcmt_mechanism": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
    }
    n = 0
    lines_seen = 0
    for obs_action, sim_y, real_y, meta in _iter_city_chunks(
        env_path,
        max_lines=args.max_lines_per_city,
        actions=args.actions,
        chunk_rows=args.chunk_rows,
    ):
        y = real_y.astype(np.float64)
        sim = sim_y.astype(np.float64)
        h2o_pred = sim + h2o_features(obs_action) @ model.h2o_beta
        cfcmt_pred = sim.copy()
        for idx, output_name in enumerate(OUTPUT_NAMES):
            cfcmt_pred[:, idx] += cfcmt_features(output_name, obs_action) @ model.cfcmt_beta[output_name]
        sse["uncalibrated"] += np.square(sim - y).sum(axis=0)
        sse["h2oplus_dense"] += np.square(h2o_pred - y).sum(axis=0)
        sse["cfcmt_mechanism"] += np.square(cfcmt_pred - y).sum(axis=0)
        n += int(y.shape[0])
        lines_seen = max(lines_seen, int(meta["lines_seen"]))

    if n == 0:
        raise RuntimeError(f"No target transitions generated for {target_key}")
    output = {}
    dyn_count = len(DYNAMIC_OUTPUT_INDICES)
    for method, values in sse.items():
        per_output = values / float(n)
        output[method] = {
            "total_mse": float(per_output.mean()),
            "next_dynamic_mse": float(per_output[:dyn_count].mean()),
            "reward_mse": float(per_output[-1]),
            "per_output_mse": {name: float(value) for name, value in zip(OUTPUT_NAMES, per_output)},
        }
    output["target_transitions"] = n
    output["target_lines_seen"] = lines_seen
    output["target_env"] = target_key
    output["target_city"] = spec.get("city", target_key)
    return output


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("inf")


def _summarize_split(split: dict[str, Any], metrics: dict[str, Any], model: ModelBundle) -> dict[str, Any]:
    uncal = metrics["uncalibrated"]["total_mse"]
    h2o = metrics["h2oplus_dense"]["total_mse"]
    cfcmt = metrics["cfcmt_mechanism"]["total_mse"]
    return {
        **split,
        "ok": True,
        "performance_validation": True,
        "policy_rollout": False,
        "train_transitions": model.train_transitions,
        "target_transitions": metrics["target_transitions"],
        "target_lines_seen": metrics["target_lines_seen"],
        "metrics": metrics,
        "comparisons": {
            "h2oplus_vs_uncalibrated_total_mse_ratio": _ratio(h2o, uncal),
            "cfcmt_vs_uncalibrated_total_mse_ratio": _ratio(cfcmt, uncal),
            "cfcmt_vs_h2oplus_total_mse_ratio": _ratio(cfcmt, h2o),
            "cfcmt_beats_h2oplus": bool(cfcmt < h2o),
            "cfcmt_beats_uncalibrated": bool(cfcmt < uncal),
            "h2oplus_beats_uncalibrated": bool(h2o < uncal),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    generated_envs = config.get("generated_envs", {})
    city_stats: dict[str, CityStats] = {}
    for key, spec in generated_envs.items():
        print(f"[stats] {key}: streaming full city bundle", flush=True)
        city_stats[key] = accumulate_city_stats(key, spec, root, args)
        print(
            f"[stats] {key}: lines={city_stats[key].lines_seen}, "
            f"transitions={city_stats[key].transitions_seen}",
            flush=True,
        )

    split_results = []
    for split in expand_splits(config):
        print(f"[split] {split['name']}: fit source={split['source_envs']} target={split['target_env']}", flush=True)
        model = fit_model(list(split["source_envs"]), city_stats, args.ridge)
        metrics = evaluate_model(split["target_env"], generated_envs[split["target_env"]], root, model, args)
        split_results.append(_summarize_split(split, metrics, model))

    cfcmt_wins = sum(1 for split in split_results if split["comparisons"]["cfcmt_beats_h2oplus"])
    h2o_wins = len(split_results) - cfcmt_wins
    mean_ratio = float(np.mean([split["comparisons"]["cfcmt_vs_h2oplus_total_mse_ratio"] for split in split_results]))
    return {
        "ok": True,
        "validation_level": "cross_city_static_offline_dynamics_performance",
        "performance_validation": True,
        "policy_rollout": False,
        "config": str(_resolve_path(root, args.config)),
        "actions_seconds": args.actions,
        "ridge": args.ridge,
        "max_lines_per_city": args.max_lines_per_city,
        "output_names": OUTPUT_NAMES,
        "cities": {
            key: {
                "city": value.city,
                "env_path": str(value.env_path),
                "lines_seen": value.lines_seen,
                "transitions_seen": value.transitions_seen,
                "warnings": value.warnings,
            }
            for key, value in city_stats.items()
        },
        "splits": split_results,
        "summary": {
            "splits": len(split_results),
            "cfcmt_wins_vs_h2oplus": cfcmt_wins,
            "h2oplus_wins_vs_cfcmt": h2o_wins,
            "mean_cfcmt_vs_h2oplus_total_mse_ratio": mean_ratio,
        },
    }


def _parse_actions(value: str) -> list[float]:
    actions = [float(item) for item in str(value).split(",") if item.strip()]
    if not actions:
        raise ValueError("At least one action must be provided")
    return actions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/cross_city_performance_validation.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--actions", type=_parse_actions, default=list(DEFAULT_ACTIONS))
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--max-lines-per-city", type=int, default=0, help="0 means all lines; use >0 only for smoke checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
