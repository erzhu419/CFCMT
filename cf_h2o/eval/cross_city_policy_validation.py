"""Cross-city policy-level validation for H2O+ vs CFCMT.

This benchmark uses full-route static city bundles and evaluates one-step
lookahead hold policies under headway-stress scenarios. It is policy-level
because each method predicts target-city next-state dynamics, applies the same
policy reward to the predicted state, and selects actions from a candidate hold
set; it is not a live SUMO rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_open_transit_validation import expand_splits
from cf_h2o.eval.cross_city_performance_validation import (
    HOURS,
    OUTPUT_NAMES,
    cfcmt_features,
    empty_stats,
    fit_model,
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
    _stable_unit,
    _uncalibrated_transition,
)


DEFAULT_ACTIONS = [0.0, 15.0, 30.0, 45.0, 60.0]
DEFAULT_HEADWAY_STRESS = [(0.75, 1.25), (0.90, 1.10), (1.0, 1.0), (1.10, 0.90), (1.25, 0.75)]
POLICY_HOLD_COST = 0.0008


def _policy_reward(
    fwd: np.ndarray,
    bwd: np.ndarray,
    waiting: np.ndarray,
    target: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    target = np.maximum(target, 60.0)
    headway_penalty = np.abs(fwd - target) / target + np.abs(bwd - target) / target
    return -(headway_penalty + 0.018 * np.log1p(np.maximum(waiting, 0.0)) + POLICY_HOLD_COST * action)


def _transition_with_policy_reward(values: tuple[np.ndarray, ...], target: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, ...]:
    items = list(values)
    items[-1] = _policy_reward(items[0], items[1], items[2], target, action)
    return tuple(items)


def _reward_from_predicted_state(pred_y: np.ndarray, obs_action: np.ndarray) -> np.ndarray:
    fwd = np.clip(pred_y[:, 0].astype(np.float64), 10.0, 7200.0)
    bwd = np.clip(pred_y[:, 1].astype(np.float64), 10.0, 7200.0)
    waiting = np.clip(pred_y[:, 2].astype(np.float64), 0.0, None)
    target = np.maximum(obs_action[:, 8].astype(np.float64), 60.0)
    action = obs_action[:, 15].astype(np.float64)
    return _policy_reward(fwd, bwd, waiting, target, action)


@dataclass
class PolicyAccumulator:
    actions: list[float]
    n: int = 0
    reward_sum: float = 0.0
    oracle_reward_sum: float = 0.0
    headway_abs_sum: float = 0.0
    gap_abs_sum: float = 0.0
    waiting_sum: float = 0.0
    hold_sum: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)

    def add(self, chosen: np.ndarray, target: np.ndarray, action: np.ndarray, oracle_reward: np.ndarray) -> None:
        self.n += int(chosen.shape[0])
        self.reward_sum += float(chosen[:, -1].sum())
        self.oracle_reward_sum += float(oracle_reward.sum())
        self.headway_abs_sum += float((0.5 * (np.abs(chosen[:, 0] - target) + np.abs(chosen[:, 1] - target))).sum())
        self.gap_abs_sum += float(np.abs(chosen[:, 4]).sum())
        self.waiting_sum += float(chosen[:, 2].sum())
        self.hold_sum += float(action.sum())
        for value, count in zip(*np.unique(action, return_counts=True)):
            key = f"{float(value):.1f}"
            self.action_counts[key] = self.action_counts.get(key, 0) + int(count)

    def summary(self) -> dict[str, Any]:
        denom = max(1, self.n)
        return {
            "n": self.n,
            "mean_reward": self.reward_sum / denom,
            "mean_oracle_reward": self.oracle_reward_sum / denom,
            "mean_regret_to_oracle": (self.oracle_reward_sum - self.reward_sum) / denom,
            "mean_headway_abs_error": self.headway_abs_sum / denom,
            "mean_abs_gap": self.gap_abs_sum / denom,
            "mean_waiting_passengers": self.waiting_sum / denom,
            "mean_hold_seconds": self.hold_sum / denom,
            "action_counts": dict(sorted(self.action_counts.items(), key=lambda item: float(item[0]))),
        }


def _nearest_action_index(actions: list[float], value: float) -> int:
    return min(range(len(actions)), key=lambda idx: abs(float(actions[idx]) - float(value)))


def _candidate_groups(
    env_path: Path,
    *,
    actions: list[float],
    headway_stress: list[tuple[float, float]],
    max_lines: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    """Yield one line-hour group with all candidate action outcomes."""

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

            for stress_idx, (fwd_mult, bwd_mult) in enumerate(headway_stress):
                fwd_s = np.clip(fwd * float(fwd_mult), 10.0, 7200.0)
                bwd_s = np.clip(bwd * float(bwd_mult), 10.0, 7200.0)
                co_fwd_s = np.clip(0.82 * co_fwd + 0.18 * fwd_s, 10.0, 7200.0)
                co_bwd_s = np.clip(0.82 * co_bwd + 0.18 * bwd_s, 10.0, 7200.0)
                base_obs = np.column_stack(
                    [
                        np.full(nseg, line_code),
                        np.mod(route_idx, 400.0) / 400.0,
                        station_fraction,
                        np.full(nseg, hour / 23.0),
                        np.full(nseg, direction),
                        fwd_s,
                        bwd_s,
                        waiting,
                        target,
                        stop_duration,
                        sim_time,
                        fwd_s - bwd_s,
                        co_fwd_s,
                        co_bwd_s,
                        speed,
                    ]
                ).astype(np.float32)

                obs_action_parts = []
                sim_parts = []
                real_parts = []
                for hold in actions:
                    action = np.full(nseg, float(hold), dtype=np.float64)
                    real_parts.append(
                        np.column_stack(
                            _transition_with_policy_reward(
                                _real_transition(
                                    fwd_s,
                                    bwd_s,
                                    waiting,
                                    stop_duration,
                                    target,
                                    co_fwd_s,
                                    co_bwd_s,
                                    speed,
                                    next_speed,
                                    next_demand,
                                    travel_time,
                                    action,
                                    peak,
                                ),
                                target,
                                action,
                            )
                        ).astype(np.float32)
                    )
                    sim_parts.append(
                        np.column_stack(
                            _transition_with_policy_reward(
                                _uncalibrated_transition(
                                    fwd_s,
                                    bwd_s,
                                    waiting,
                                    target,
                                    speed,
                                    travel_time,
                                    action,
                                    hour,
                                ),
                                target,
                                action,
                            )
                        ).astype(np.float32)
                    )
                    obs_action_parts.append(
                        np.concatenate([base_obs, action[:, None].astype(np.float32)], axis=1)
                    )

                yield (
                    np.concatenate(obs_action_parts, axis=0),
                    np.concatenate(sim_parts, axis=0),
                    np.concatenate(real_parts, axis=0),
                    {
                        "line_key": line_key,
                        "hour": hour,
                        "stress_idx": stress_idx,
                        "stress": [float(fwd_mult), float(bwd_mult)],
                        "lines_seen": line_count,
                        "n_segments": nseg,
                        "target_headway": target,
                    },
                )


def accumulate_policy_city_stats(key: str, spec: dict[str, Any], root: Path, args: argparse.Namespace):
    env_path = _resolve_path(root, spec["env_path"])
    stats = empty_stats(key, str(spec.get("city", key)), env_path)
    t0 = time.time()
    for obs_action, sim_y, real_y, meta in _candidate_groups(
        env_path,
        actions=args.actions,
        headway_stress=args.headway_stress,
        max_lines=args.max_lines_per_city,
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


def _method_choices(
    obs_action: np.ndarray,
    sim_y: np.ndarray,
    real_y: np.ndarray,
    n_segments: int,
    actions: list[float],
    model,
) -> dict[str, np.ndarray]:
    action_count = len(actions)
    reward_idx = OUTPUT_NAMES.index("reward")
    h2o_pred_y = sim_y.astype(np.float64) + h2o_features(obs_action) @ model.h2o_beta
    cfcmt_pred_y = sim_y.astype(np.float64).copy()
    for idx, output_name in enumerate(OUTPUT_NAMES):
        if output_name == "reward":
            continue
        cfcmt_pred_y[:, idx] += cfcmt_features(output_name, obs_action) @ model.cfcmt_beta[output_name]
    sim_pred_reward = _reward_from_predicted_state(sim_y, obs_action)
    h2o_pred_reward = _reward_from_predicted_state(h2o_pred_y, obs_action)
    cfcmt_pred_reward = _reward_from_predicted_state(cfcmt_pred_y, obs_action)
    choices = {
        "no_hold": np.full(n_segments, _nearest_action_index(actions, 0.0), dtype=np.int64),
        "fixed_30": np.full(n_segments, _nearest_action_index(actions, 30.0), dtype=np.int64),
        "uncalibrated_policy": sim_pred_reward.reshape(action_count, n_segments).argmax(axis=0).astype(np.int64),
        "h2oplus_dense_policy": h2o_pred_reward.reshape(action_count, n_segments).argmax(axis=0).astype(np.int64),
        "cfcmt_mechanism_policy": cfcmt_pred_reward.reshape(action_count, n_segments).argmax(axis=0).astype(np.int64),
        "oracle_policy": real_y[:, reward_idx].reshape(action_count, n_segments).argmax(axis=0).astype(np.int64),
    }
    return choices


def evaluate_policy(target_key: str, spec: dict[str, Any], root: Path, model, args: argparse.Namespace) -> dict[str, Any]:
    env_path = _resolve_path(root, spec["env_path"])
    actions = [float(value) for value in args.actions]
    accumulators = {
        name: PolicyAccumulator(actions=actions)
        for name in (
            "no_hold",
            "fixed_30",
            "uncalibrated_policy",
            "h2oplus_dense_policy",
            "cfcmt_mechanism_policy",
            "oracle_policy",
        )
    }
    lines_seen = 0
    t0 = time.time()
    for obs_action, sim_y, real_y, meta in _candidate_groups(
        env_path,
        actions=actions,
        headway_stress=args.headway_stress,
        max_lines=args.max_lines_per_city,
    ):
        n_segments = int(meta["n_segments"])
        target = np.asarray(meta["target_headway"], dtype=np.float64)
        oracle_idx = real_y[:, -1].reshape(len(actions), n_segments).argmax(axis=0)
        oracle_flat = oracle_idx * n_segments + np.arange(n_segments)
        oracle_reward = real_y[oracle_flat, -1]
        choices = _method_choices(obs_action, sim_y, real_y, n_segments, actions, model)
        for method, choice in choices.items():
            flat = choice * n_segments + np.arange(n_segments)
            chosen = real_y[flat]
            action_values = np.asarray([actions[int(idx)] for idx in choice], dtype=np.float64)
            accumulators[method].add(chosen, target, action_values, oracle_reward)
        lines_seen = max(lines_seen, int(meta["lines_seen"]))

    metrics = {name: acc.summary() for name, acc in accumulators.items()}
    no_hold_reward = metrics["no_hold"]["mean_reward"]
    h2o_reward = metrics["h2oplus_dense_policy"]["mean_reward"]
    cfcmt_reward = metrics["cfcmt_mechanism_policy"]["mean_reward"]
    oracle_reward = metrics["oracle_policy"]["mean_reward"]
    for method, values in metrics.items():
        values["mean_reward_gain_vs_no_hold"] = values["mean_reward"] - no_hold_reward
        values["oracle_reward_gap"] = oracle_reward - values["mean_reward"]

    return {
        "target_env": target_key,
        "target_city": spec.get("city", target_key),
        "target_lines_seen": lines_seen,
        "elapsed_sec": time.time() - t0,
        "methods": metrics,
        "comparisons": {
            "cfcmt_reward_beats_h2oplus": bool(cfcmt_reward > h2o_reward),
            "cfcmt_reward_gain_vs_h2oplus": cfcmt_reward - h2o_reward,
            "cfcmt_regret_ratio_vs_h2oplus": _safe_ratio(
                metrics["cfcmt_mechanism_policy"]["mean_regret_to_oracle"],
                metrics["h2oplus_dense_policy"]["mean_regret_to_oracle"],
            ),
            "cfcmt_headway_error_ratio_vs_h2oplus": _safe_ratio(
                metrics["cfcmt_mechanism_policy"]["mean_headway_abs_error"],
                metrics["h2oplus_dense_policy"]["mean_headway_abs_error"],
            ),
        },
    }


def _safe_ratio(num: float, den: float) -> float | None:
    return float(num / den) if abs(den) > 1e-12 else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    generated_envs = config.get("generated_envs", {})
    city_stats = {}
    for key, spec in generated_envs.items():
        print(f"[stats] {key}: streaming policy training stats", flush=True)
        city_stats[key] = accumulate_policy_city_stats(key, spec, root, args)
        print(
            f"[stats] {key}: lines={city_stats[key].lines_seen}, "
            f"transitions={city_stats[key].transitions_seen}",
            flush=True,
        )

    split_results = []
    cache: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for split in expand_splits(config):
        source_envs = tuple(split["source_envs"])
        target_env = split["target_env"]
        cache_key = (source_envs, target_env)
        print(f"[split] {split['name']}: source={list(source_envs)} target={target_env}", flush=True)
        model = fit_model(list(source_envs), city_stats, args.ridge)
        if cache_key not in cache:
            cache[cache_key] = evaluate_policy(target_env, generated_envs[target_env], root, model, args)
        eval_result = cache[cache_key]
        split_results.append(
            {
                **split,
                "ok": True,
                "policy_validation": True,
                "live_env_rollout": False,
                "one_step_lookahead": True,
                "train_transitions": model.train_transitions,
                **eval_result,
            }
        )

    eps = 1e-12
    cfcmt_wins_reward = sum(
        1 for split in split_results if split["comparisons"]["cfcmt_reward_gain_vs_h2oplus"] > eps
    )
    h2o_wins_reward = sum(
        1 for split in split_results if split["comparisons"]["cfcmt_reward_gain_vs_h2oplus"] < -eps
    )
    ties_reward = len(split_results) - cfcmt_wins_reward - h2o_wins_reward
    mean_reward_gain = float(np.mean([split["comparisons"]["cfcmt_reward_gain_vs_h2oplus"] for split in split_results]))
    regret_ratios = [
        split["comparisons"]["cfcmt_regret_ratio_vs_h2oplus"]
        for split in split_results
        if split["comparisons"]["cfcmt_regret_ratio_vs_h2oplus"] is not None
    ]
    headway_ratios = [
        split["comparisons"]["cfcmt_headway_error_ratio_vs_h2oplus"]
        for split in split_results
        if split["comparisons"]["cfcmt_headway_error_ratio_vs_h2oplus"] is not None
    ]
    mean_regret_ratio = float(np.mean(regret_ratios)) if regret_ratios else None
    cfcmt_wins_regret = sum(
        1
        for split in split_results
        if split["methods"]["cfcmt_mechanism_policy"]["mean_regret_to_oracle"]
        < split["methods"]["h2oplus_dense_policy"]["mean_regret_to_oracle"] - eps
    )
    h2o_wins_regret = sum(
        1
        for split in split_results
        if split["methods"]["cfcmt_mechanism_policy"]["mean_regret_to_oracle"]
        > split["methods"]["h2oplus_dense_policy"]["mean_regret_to_oracle"] + eps
    )
    cfcmt_wins_headway = sum(
        1
        for split in split_results
        if split["methods"]["cfcmt_mechanism_policy"]["mean_headway_abs_error"]
        < split["methods"]["h2oplus_dense_policy"]["mean_headway_abs_error"] - eps
    )
    h2o_wins_headway = sum(
        1
        for split in split_results
        if split["methods"]["cfcmt_mechanism_policy"]["mean_headway_abs_error"]
        > split["methods"]["h2oplus_dense_policy"]["mean_headway_abs_error"] + eps
    )
    return {
        "ok": True,
        "validation_level": "cross_city_static_offline_headway_stress_policy_lookahead",
        "policy_validation": True,
        "live_env_rollout": False,
        "one_step_lookahead": True,
        "config": str(_resolve_path(root, args.config)),
        "actions_seconds": args.actions,
        "headway_stress": args.headway_stress,
        "policy_hold_cost": POLICY_HOLD_COST,
        "ridge": args.ridge,
        "max_lines_per_city": args.max_lines_per_city,
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
            "cfcmt_reward_wins_vs_h2oplus": cfcmt_wins_reward,
            "h2oplus_reward_wins_vs_cfcmt": h2o_wins_reward,
            "reward_ties": ties_reward,
            "mean_cfcmt_reward_gain_vs_h2oplus": mean_reward_gain,
            "cfcmt_regret_wins_vs_h2oplus": cfcmt_wins_regret,
            "h2oplus_regret_wins_vs_cfcmt": h2o_wins_regret,
            "mean_cfcmt_regret_ratio_vs_h2oplus": mean_regret_ratio,
            "cfcmt_headway_error_wins_vs_h2oplus": cfcmt_wins_headway,
            "h2oplus_headway_error_wins_vs_cfcmt": h2o_wins_headway,
            "mean_cfcmt_headway_error_ratio_vs_h2oplus": float(np.mean(headway_ratios)) if headway_ratios else None,
        },
    }


def _parse_actions(value: str) -> list[float]:
    actions = [float(item) for item in str(value).split(",") if item.strip()]
    if not actions:
        raise ValueError("At least one action must be provided")
    return actions


def _parse_headway_stress(value: str) -> list[tuple[float, float]]:
    text = str(value).strip()
    if not text or text.lower() in {"none", "off"}:
        return [(1.0, 1.0)]
    scenarios = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid headway stress scenario {item!r}; expected fwd_mult:bwd_mult")
        fwd, bwd = item.split(":", 1)
        scenarios.append((float(fwd), float(bwd)))
    if not scenarios:
        raise ValueError("At least one headway stress scenario is required")
    return scenarios


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/cross_city_policy_validation.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--actions", type=_parse_actions, default=list(DEFAULT_ACTIONS))
    parser.add_argument(
        "--headway-stress",
        type=_parse_headway_stress,
        default=list(DEFAULT_HEADWAY_STRESS),
        help="Comma-separated fwd_mult:bwd_mult scenarios, or 'none'.",
    )
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
