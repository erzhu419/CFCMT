"""Libsumo closed-loop holding diagnostic for generated APC/AVL SUMO scenarios.

This is an executable counterfactual-control diagnostic. It requires Stage 2
SUMO scenarios with explicit ``busStop`` dwell points; older scenarios without
stops can still be used for AVL/APC snapshots but are not valid for holding
counterfactuals.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.paper_experiment_suite import (
    POLICY_ACTIONS,
    _bus_linear_reward_from_predicted_state,
    _fit_cfcmt,
    _fit_h2o,
    _merge_stats,
    _merge_stats_weighted,
    _parse_float_list,
    _predict_action_from_obs,
    _repo_root,
    _resolve_path,
    _rollout_feature_obs,
    _source_similarity_weights,
    build_all_city_stats,
)
from cf_h2o.eval.cross_city_performance_validation import (
    OUTPUT_NAMES,
    cfcmt_features,
    h2o_features,
    _read_json,
    _uncalibrated_transition,
)


DEFAULT_STAGE2_REPORT = Path("cf_h2o/results/sumo_apc_avl_sumo_generation_control_smoke.json")
DEFAULT_CONFIG = Path("cf_h2o/config/cross_city_open_transit.json")
DEFAULT_OUT = Path("cf_h2o/results/sumo_policy_rollout_validation_smoke.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/sumo_policy_rollout_validation_smoke.md")


def _vehicle_line_uid(vehicle_id: str) -> str:
    match = re.match(r"veh_(.+)_\d{6}$", vehicle_id)
    return match.group(1) if match else ""


def _vehicle_departure_index(vehicle_id: str) -> int | None:
    match = re.match(r"veh_.+_(\d{6})$", vehicle_id)
    return int(match.group(1)) if match else None


def _stop_index(stop_id: str) -> int:
    match = re.search(r"_(\d{4,6})$", stop_id)
    return int(match.group(1)) if match else 0


def _line_departures(line_map: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        str(line["line_uid"]): sorted(float(value) for value in line.get("departures", []))
        for line in line_map
    }


def _target_headways(line_map: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for line_uid, departures in _line_departures(line_map).items():
        diffs = np.diff(np.asarray(departures, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        out[line_uid] = float(np.median(diffs)) if diffs.size else 600.0
    return out


def _line_segment_counts(line_map: list[dict[str, Any]]) -> dict[str, int]:
    return {str(line["line_uid"]): int(line.get("segments") or len(line.get("edges", [])) or 1) for line in line_map}


def _target_summary_context(line_map: list[dict[str, Any]]) -> dict[str, float]:
    headways: list[float] = []
    departures: list[float] = []
    distances: list[float] = []
    segments: list[float] = []
    for line in line_map:
        deps = sorted(float(value) for value in line.get("departures", []))
        diffs = np.diff(np.asarray(deps, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            headways.append(float(np.median(diffs)))
        departures.append(float(len(deps)))
        distances.append(float(line.get("distance_m") or 0.0))
        segments.append(float(line.get("segments") or len(line.get("edges", [])) or 0.0))
    return {
        "median_headway": float(np.median(headways)) if headways else 600.0,
        "median_departures": float(np.median(departures)) if departures else 0.0,
        "median_distance_m": float(np.median(distances)) if distances else 0.0,
        "median_segments": float(np.median(segments)) if segments else 0.0,
    }


def _segment_speed_by_stop(city: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    with Path(city["segment_map_path"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            line_uid = str(row["line_uid"])
            stop_id = f"bs_{line_uid}_{int(row['segment_idx']):04d}"
            out[stop_id] = float(row.get("speed_mps") or 8.0)
    return out


def _segment_times_by_line(city: dict[str, Any]) -> dict[str, list[float]]:
    rows: dict[str, dict[int, float]] = {}
    with Path(city["segment_map_path"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            line_uid = str(row["line_uid"])
            segment_idx = int(row["segment_idx"])
            speed = max(float(row.get("speed_mps") or 8.0), 0.5)
            length = max(float(row.get("length_m") or 0.0), 1.0)
            rows.setdefault(line_uid, {})[segment_idx] = float(np.clip(length / speed, 1.0, 3600.0))
    out: dict[str, list[float]] = {}
    for line_uid, by_idx in rows.items():
        max_idx = max(by_idx) if by_idx else -1
        out[line_uid] = [float(by_idx.get(idx, 60.0)) for idx in range(max_idx + 1)]
    return out


def _remaining_time_to_stop(
    *,
    line_uid: str,
    route_idx: int,
    station_idx: int,
    segment_times: dict[str, list[float]],
    target_hw: float,
) -> float | None:
    if route_idx > station_idx:
        return None
    times = segment_times.get(line_uid) or []
    if not times:
        return float(np.clip(target_hw * max(station_idx - route_idx + 1, 1) / 10.0, 10.0, 7200.0))
    stop_idx = min(max(station_idx, 0), len(times) - 1)
    start_idx = min(max(route_idx, 0), stop_idx)
    return float(np.clip(sum(times[start_idx : stop_idx + 1]), 10.0, 7200.0))


def _cumulative_time_to_stop(
    *,
    line_uid: str,
    station_idx: int,
    segment_times: dict[str, list[float]],
    target_hw: float,
) -> float:
    return _remaining_time_to_stop(
        line_uid=line_uid,
        route_idx=0,
        station_idx=station_idx,
        segment_times=segment_times,
        target_hw=target_hw,
    ) or float(target_hw)


def _estimate_backward_headway(
    *,
    sumo_api: Any,
    line_uid: str,
    vehicle_id: str,
    stop_id: str,
    now: float,
    target_hw: float,
    active_vehicle_ids: list[str],
    handled: set[tuple[str, str]],
    line_departures: dict[str, list[float]],
    segment_times: dict[str, list[float]],
) -> float:
    current_idx = _vehicle_departure_index(vehicle_id)
    station_idx = _stop_index(stop_id)
    candidates: list[float] = []
    for other_id in active_vehicle_ids:
        if other_id == vehicle_id or (other_id, stop_id) in handled:
            continue
        other_idx = _vehicle_departure_index(other_id)
        if current_idx is not None and other_idx is not None and other_idx <= current_idx:
            continue
        try:
            route_idx = int(sumo_api.vehicle.getRouteIndex(other_id))
        except Exception:
            continue
        remaining = _remaining_time_to_stop(
            line_uid=line_uid,
            route_idx=route_idx,
            station_idx=station_idx,
            segment_times=segment_times,
            target_hw=target_hw,
        )
        if remaining is not None:
            candidates.append(float(remaining))

    departures = line_departures.get(line_uid, [])
    if current_idx is not None and current_idx + 1 < len(departures):
        future_limit = min(len(departures), current_idx + 5)
        from_start = _cumulative_time_to_stop(
            line_uid=line_uid,
            station_idx=station_idx,
            segment_times=segment_times,
            target_hw=target_hw,
        )
        for departure in departures[current_idx + 1 : future_limit]:
            if departure >= now:
                candidates.append(float(departure - now + from_start))

    if not candidates:
        return float(np.clip(target_hw, 10.0, 7200.0))
    return float(np.clip(min(candidates), 10.0, 7200.0))


def _merge_target_lines(city_line_stats: dict[str, list[Any]], target: str, max_lines: int) -> tuple[Any | None, list[str]]:
    if max_lines <= 0:
        return None, []
    selected = city_line_stats.get(target, [])[:max_lines]
    if not selected:
        return None, []
    stats = _merge_stats(f"{target}_fewshot", target, selected)
    line_keys: list[str] = []
    for item in selected:
        line_keys.extend(str(key) for key in getattr(item, "line_keys", []))
    return stats, line_keys


def _merge_with_target_mass(source_stats: Any, target_stats: Any, target_mass: float) -> Any:
    out = _merge_stats("source_target_fewshot", "source_target", [source_stats])
    if target_stats is None or target_stats.n <= 0:
        return out
    scale = max(0.0, float(target_mass)) * float(source_stats.n) / max(float(target_stats.n), 1.0)
    if scale > 0.0:
        out.merge_scaled(target_stats, scale)
    return out


def _build_transfer_models(
    *,
    config: dict[str, Any],
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stats_args = SimpleNamespace(
        max_lines_per_city=int(args.transfer_max_lines_per_city),
        actions=list(args.actions),
        progress_every=int(args.progress_every),
        workers=int(args.workers),
    )
    city_stats, city_line_stats, sanity = build_all_city_stats(config, root, stats_args)
    models: dict[str, dict[str, Any]] = {}
    for target in config["generated_envs"]:
        sources = [key for key in config["generated_envs"] if key != target]
        train_stats = _merge_stats("source", "source", [city_stats[key] for key in sources])
        weights = _source_similarity_weights(
            sanity,
            target,
            sources,
            temperature=float(args.source_weight_temperature),
            floor=float(args.source_weight_floor),
        )
        weighted_stats = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, weights)
        target_stats, target_lines = _merge_target_lines(
            city_line_stats,
            target,
            int(args.target_adaptation_lines),
        )
        source_target_stats = _merge_with_target_mass(
            weighted_stats,
            target_stats,
            float(args.target_adaptation_mass),
        )
        model = {
            "source_envs": sources,
            "source_weights": weights,
            "h2o_beta": _fit_h2o(train_stats, float(args.ridge)),
            "cfcmt_beta": _fit_cfcmt(train_stats, float(args.ridge)),
            "cfcmt_weighted_beta": _fit_cfcmt(weighted_stats, float(args.ridge)),
            "target_adaptation_lines": target_lines,
            "target_adaptation_rows": int(target_stats.n) if target_stats is not None else 0,
        }
        if target_stats is not None and target_stats.n > 0:
            model["cfcmt_target_beta"] = _fit_cfcmt(target_stats, float(args.ridge))
            model["cfcmt_source_target_beta"] = _fit_cfcmt(source_target_stats, float(args.ridge))
        models[target] = model
    return models, sanity


def _event_obs(
    *,
    line_uid: str,
    stop_id: str,
    now: float,
    target_hw: float,
    forward_hw: float,
    backward_hw: float,
    station_count: int,
    speed_mps: float,
) -> np.ndarray:
    station_idx = _stop_index(stop_id)
    station_fraction = station_idx / max(float(station_count - 1), 1.0)
    hour = int(now // 3600) % 24
    obs = np.zeros(15, dtype=np.float32)
    obs[1] = station_fraction
    obs[2] = station_idx
    obs[3] = hour / 23.0
    obs[5] = float(np.clip(forward_hw, 10.0, 7200.0))
    obs[6] = float(np.clip(backward_hw, 10.0, 7200.0))
    obs[7] = max(0.0, 0.02 * target_hw)
    obs[8] = float(max(target_hw, 60.0))
    obs[9] = 8.0
    obs[10] = float(now)
    obs[11] = obs[5] - obs[6]
    obs[12] = obs[8]
    obs[13] = obs[8]
    obs[14] = float(np.clip(speed_mps, 1.0, 25.0))
    return obs


def _choose_extra_hold(
    *,
    policy: str,
    obs: np.ndarray,
    model: dict[str, Any] | None,
    actions: list[float],
    line_uid: str,
    station_count: int,
    sim_start_hour: int,
    mpc_horizon: int,
    mpc_discount: float,
    mpc_action_cost: float,
    policy_context: dict[str, float] | None = None,
) -> float:
    if policy == "no_hold":
        return 0.0
    if policy == "fixed_30":
        return 30.0
    if policy == "daganzo_policy":
        return _daganzo_hold(obs, alpha=0.6)
    if policy == "daganzo_a03_policy":
        return _daganzo_hold(obs, alpha=0.3)
    if policy == "daganzo_a04_policy":
        return _daganzo_hold(obs, alpha=0.4)
    if policy == "daganzo_a05_policy":
        return _daganzo_hold(obs, alpha=0.5)
    if policy == "daganzo_a07_policy":
        return _daganzo_hold(obs, alpha=0.7)
    if policy == "daganzo_a08_policy":
        return _daganzo_hold(obs, alpha=0.8)
    if policy == "target_summary_daganzo_policy":
        context = policy_context or {}
        median_headway = float(context.get("median_headway", 600.0))
        median_departures = float(context.get("median_departures", 0.0))
        alpha = 0.3 if median_headway < 900.0 or median_departures <= 4.0 else 0.8
        return _daganzo_hold(obs, alpha=alpha)
    if policy == "threshold_equalization_policy":
        return _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0)
    if policy == "threshold_fine_policy":
        return _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=1.00, severe_hold=60.0, moderate_hold=45.0, mild_hold=15.0)
    if policy == "threshold_aggressive_policy":
        return _threshold_hold(obs, actions, severe_ratio=0.85, moderate_ratio=1.00, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0)
    if policy == "threshold_soft_policy":
        return _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=45.0, moderate_hold=15.0, mild_hold=0.0)
    if model is None:
        return 0.0
    if policy in {
        "cfcmt_target_fewshot_mpc_threshold_guard_policy",
        "cfcmt_source_target_fewshot_mpc_threshold_guard_policy",
        "cfcmt_target_fewshot_mpc_rule_selector_policy",
        "cfcmt_source_target_fewshot_mpc_rule_selector_policy",
        "cfcmt_target_fewshot_mpc_alpha_selector_policy",
        "cfcmt_source_target_fewshot_mpc_alpha_selector_policy",
        "cfcmt_target_fewshot_mpc_threshold_residual_policy",
        "cfcmt_source_target_fewshot_mpc_threshold_residual_policy",
    }:
        base_policy = (
            "cfcmt_source_target_fewshot_mpc_policy"
            if policy.startswith("cfcmt_source_target_fewshot")
            else "cfcmt_target_fewshot_mpc_policy"
        )
        threshold = _threshold_hold(
            obs,
            actions,
            severe_ratio=0.75,
            moderate_ratio=0.90,
            mild_ratio=None,
            severe_hold=60.0,
            moderate_hold=30.0,
            mild_hold=0.0,
        )
        daganzo = _daganzo_hold(obs, alpha=0.6)
        if policy.endswith("_threshold_guard_policy"):
            model_action = _predict_mpc_action_from_obs(
                obs,
                base_policy,
                model,
                actions,
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
            return float(max(model_action, threshold))
        if policy.endswith("_rule_selector_policy"):
            return _predict_mpc_action_from_obs(
                obs,
                base_policy,
                model,
                _unique_actions([daganzo, threshold]),
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
        if policy.endswith("_alpha_selector_policy"):
            alpha_actions = [_daganzo_hold(obs, alpha=alpha) for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
            alpha_actions.append(threshold)
            return _predict_mpc_action_from_obs(
                obs,
                base_policy,
                model,
                _unique_actions(alpha_actions),
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
        residual_actions = _local_action_candidates(threshold, actions, extra=[daganzo])
        return _predict_mpc_action_from_obs(
            obs,
            base_policy,
            model,
            residual_actions,
            line_uid=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
            horizon=mpc_horizon,
            discount=mpc_discount,
            action_cost=mpc_action_cost,
        )
    if policy in {
        "cfcmt_similarity_weighted_mpc_threshold_guard_policy",
        "cfcmt_similarity_weighted_mpc_threshold_fine_guard_policy",
        "cfcmt_similarity_weighted_mpc_daganzo_guard_policy",
        "cfcmt_similarity_weighted_mpc_daganzo_residual_policy",
        "cfcmt_similarity_weighted_mpc_threshold_residual_policy",
        "cfcmt_similarity_weighted_mpc_rule_selector_policy",
    }:
        if policy == "cfcmt_similarity_weighted_mpc_daganzo_residual_policy":
            base = _daganzo_hold(obs, alpha=0.6)
            residual_actions = _local_action_candidates(base, actions, extra=[_threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0)])
            return _predict_mpc_action_from_obs(
                obs,
                "cfcmt_similarity_weighted_mpc_policy",
                model,
                residual_actions,
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
        if policy == "cfcmt_similarity_weighted_mpc_threshold_residual_policy":
            base = _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0)
            residual_actions = _local_action_candidates(base, actions, extra=[_daganzo_hold(obs, alpha=0.6)])
            return _predict_mpc_action_from_obs(
                obs,
                "cfcmt_similarity_weighted_mpc_policy",
                model,
                residual_actions,
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
        if policy == "cfcmt_similarity_weighted_mpc_rule_selector_policy":
            candidate_actions = [
                _daganzo_hold(obs, alpha=0.6),
                _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0),
            ]
            return _predict_mpc_action_from_obs(
                obs,
                "cfcmt_similarity_weighted_mpc_policy",
                model,
                _unique_actions(candidate_actions),
                line_uid=line_uid,
                station_count=station_count,
                sim_start_hour=sim_start_hour,
                horizon=mpc_horizon,
                discount=mpc_discount,
                action_cost=mpc_action_cost,
            )
        model_action = _predict_mpc_action_from_obs(
            obs,
            "cfcmt_similarity_weighted_mpc_policy",
            model,
            actions,
            line_uid=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
            horizon=mpc_horizon,
            discount=mpc_discount,
            action_cost=mpc_action_cost,
        )
        if policy == "cfcmt_similarity_weighted_mpc_threshold_guard_policy":
            guard = _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=None, severe_hold=60.0, moderate_hold=30.0, mild_hold=0.0)
        elif policy == "cfcmt_similarity_weighted_mpc_threshold_fine_guard_policy":
            guard = _threshold_hold(obs, actions, severe_ratio=0.75, moderate_ratio=0.90, mild_ratio=1.00, severe_hold=60.0, moderate_hold=45.0, mild_hold=15.0)
        else:
            guard = _daganzo_hold(obs, alpha=0.6)
        return float(max(model_action, guard))
    if policy.endswith("_mpc_policy"):
        return _predict_mpc_action_from_obs(
            obs,
            policy,
            model,
            actions,
            line_uid=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
            horizon=mpc_horizon,
            discount=mpc_discount,
            action_cost=mpc_action_cost,
        )
    if policy == "h2oplus_dense_policy":
        return _predict_action_from_obs(
            obs,
            policy,
            model["h2o_beta"],
            model["cfcmt_beta"],
            actions,
            line_key=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
        )
    if policy in {
        "cfcmt_mechanism_policy",
        "cfcmt_similarity_weighted_policy",
        "cfcmt_target_fewshot_policy",
        "cfcmt_source_target_fewshot_policy",
    }:
        beta = _cfcmt_beta_for_family(model, _model_family_from_policy(policy))
        return _predict_action_from_obs(
            obs,
            "cfcmt_mechanism_policy",
            model["h2o_beta"],
            beta,
            actions,
            line_key=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
        )
    return 0.0


def _nearest_action(actions: list[float], value: float) -> float:
    return float(min(actions, key=lambda item: abs(float(item) - float(value))))


def _unique_actions(actions: list[float]) -> list[float]:
    seen: dict[float, float] = {}
    for value in actions:
        clipped = float(np.clip(value, 0.0, 60.0))
        seen[round(clipped, 6)] = clipped
    return [seen[key] for key in sorted(seen)]


def _local_action_candidates(base: float, actions: list[float], *, extra: list[float] | None = None) -> list[float]:
    candidates = [float(base), float(base) - 15.0, float(base) + 15.0]
    candidates.extend(_nearest_action(actions, float(base) + delta) for delta in (-15.0, 0.0, 15.0))
    if extra:
        candidates.extend(extra)
    return _unique_actions(candidates)


def _daganzo_hold(obs: np.ndarray, *, alpha: float) -> float:
    target = max(float(obs[8]), 60.0)
    fwd = float(obs[5])
    return float(np.clip(float(alpha) * (target - fwd), 0.0, 60.0))


def _threshold_hold(
    obs: np.ndarray,
    actions: list[float],
    *,
    severe_ratio: float,
    moderate_ratio: float,
    mild_ratio: float | None,
    severe_hold: float,
    moderate_hold: float,
    mild_hold: float,
) -> float:
    target = max(float(obs[8]), 60.0)
    fwd = float(obs[5])
    if fwd < float(severe_ratio) * target:
        return _nearest_action(actions, severe_hold)
    if fwd < float(moderate_ratio) * target:
        return _nearest_action(actions, moderate_hold)
    if mild_ratio is not None and fwd < float(mild_ratio) * target:
        return _nearest_action(actions, mild_hold)
    return 0.0


def _model_family_from_policy(policy: str) -> str:
    if policy.startswith("h2oplus_dense"):
        return "h2oplus_dense_policy"
    if policy.startswith("cfcmt_source_target_fewshot"):
        return "cfcmt_source_target_fewshot_policy"
    if policy.startswith("cfcmt_target_fewshot"):
        return "cfcmt_target_fewshot_policy"
    if policy.startswith("cfcmt_similarity_weighted"):
        return "cfcmt_similarity_weighted_policy"
    if policy.startswith("cfcmt_mechanism"):
        return "cfcmt_mechanism_policy"
    return policy


def _cfcmt_beta_for_family(model: dict[str, Any], family: str) -> dict[str, np.ndarray]:
    if family == "cfcmt_similarity_weighted_policy":
        return model["cfcmt_weighted_beta"]
    if family == "cfcmt_target_fewshot_policy":
        return model.get("cfcmt_target_beta", model["cfcmt_weighted_beta"])
    if family == "cfcmt_source_target_fewshot_policy":
        return model.get("cfcmt_source_target_beta", model.get("cfcmt_target_beta", model["cfcmt_weighted_beta"]))
    return model["cfcmt_beta"]


def _predict_next_y_batch_from_model(
    obs: np.ndarray,
    *,
    policy: str,
    model: dict[str, Any],
    actions: list[float] | np.ndarray,
    line_uid: str,
    station_count: int,
    sim_start_hour: int,
) -> tuple[np.ndarray, np.ndarray]:
    family = _model_family_from_policy(policy)
    feature_obs = _rollout_feature_obs(
        obs,
        line_key=line_uid,
        station_count=station_count,
        sim_start_hour=sim_start_hour,
    )
    action_arr = np.asarray(actions, dtype=np.float32)
    obs_mat = np.repeat(feature_obs[None, :], len(action_arr), axis=0).astype(np.float32)
    obs_action = np.concatenate([obs_mat, action_arr[:, None]], axis=1)
    fwd = obs_mat[:, 5].astype(np.float64)
    bwd = obs_mat[:, 6].astype(np.float64)
    waiting = np.maximum(obs_mat[:, 7].astype(np.float64), 0.0)
    target = np.maximum(obs_mat[:, 8].astype(np.float64), 60.0)
    speed = np.clip(obs_mat[:, 14].astype(np.float64), 1.0, 25.0)
    travel_time = np.clip(target / 4.0, 30.0, 600.0)
    hour = int(round(float(feature_obs[3]) * 23.0)) % 24
    sim_y = np.column_stack(_uncalibrated_transition(fwd, bwd, waiting, target, speed, travel_time, action_arr, hour))
    if family == "h2oplus_dense_policy":
        pred_y = sim_y + h2o_features(obs_action) @ model["h2o_beta"]
    elif family in {
        "cfcmt_mechanism_policy",
        "cfcmt_similarity_weighted_policy",
        "cfcmt_target_fewshot_policy",
        "cfcmt_source_target_fewshot_policy",
    }:
        beta = _cfcmt_beta_for_family(model, family)
        pred_y = sim_y.copy()
        for idx, output_name in enumerate(OUTPUT_NAMES):
            if output_name == "reward":
                continue
            pred_y[:, idx] += cfcmt_features(output_name, obs_action) @ beta[output_name]
    else:
        pred_y = sim_y
    return pred_y.astype(np.float64), obs_action.astype(np.float64)


def _predict_next_y_from_model(
    obs: np.ndarray,
    *,
    policy: str,
    model: dict[str, Any],
    action: float,
    line_uid: str,
    station_count: int,
    sim_start_hour: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_y, obs_action = _predict_next_y_batch_from_model(
        obs,
        policy=policy,
        model=model,
        actions=[float(action)],
        line_uid=line_uid,
        station_count=station_count,
        sim_start_hour=sim_start_hour,
    )
    return pred_y[0], obs_action[0]


def _advance_obs_from_prediction(
    obs: np.ndarray,
    pred_y: np.ndarray,
    action: float,
    *,
    station_count: int,
) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy()
    out[5] = float(np.clip(pred_y[0], 10.0, 7200.0))
    out[6] = float(np.clip(pred_y[1], 10.0, 7200.0))
    out[7] = float(max(0.0, pred_y[2]))
    out[9] = float(max(0.0, pred_y[3]))
    out[11] = float(pred_y[4])
    out[12] = float(np.clip(pred_y[5], 10.0, 7200.0))
    out[13] = float(np.clip(pred_y[6], 10.0, 7200.0))
    out[14] = float(np.clip(pred_y[7], 1.0, 25.0))
    out[10] = float(out[10] + max(1.0, out[9] + float(action)))
    out[2] = float(min(max(station_count - 1, 0), int(round(float(out[2]))) + 1))
    out[1] = float(out[2] / max(float(station_count - 1), 1.0))
    return out


def _predict_mpc_action_from_obs(
    obs: np.ndarray,
    policy: str,
    model: dict[str, Any],
    actions: list[float],
    *,
    line_uid: str,
    station_count: int,
    sim_start_hour: int,
    horizon: int,
    discount: float,
    action_cost: float,
) -> float:
    horizon = max(1, int(horizon))
    discount = float(np.clip(discount, 0.0, 1.0))
    action_arr = np.asarray(actions, dtype=np.float32)

    def best_score_from(sim_obs: np.ndarray, depth: int) -> tuple[float, float]:
        pred_y, obs_action = _predict_next_y_batch_from_model(
            sim_obs,
            policy=policy,
            model=model,
            actions=action_arr,
            line_uid=line_uid,
            station_count=station_count,
            sim_start_hour=sim_start_hour,
        )
        scores = _bus_linear_reward_from_predicted_state(pred_y, obs_action).astype(np.float64)
        scores -= float(action_cost) * action_arr.astype(np.float64)
        if depth <= 1:
            best_idx = int(np.argmax(scores))
            return float(scores[best_idx]), float(action_arr[best_idx])
        totals = []
        for idx, action in enumerate(action_arr):
            next_obs = _advance_obs_from_prediction(
                sim_obs,
                pred_y[idx],
                float(action),
                station_count=station_count,
            )
            future_score, _future_action = best_score_from(next_obs, depth - 1)
            totals.append(float(scores[idx]) + discount * future_score)
        best_idx = int(np.argmax(np.asarray(totals, dtype=np.float64)))
        return float(totals[best_idx]), float(action_arr[best_idx])

    _score, best_action = best_score_from(np.asarray(obs, dtype=np.float32).copy(), horizon)
    return best_action


def _run_city_policy(
    *,
    sumo_api: Any,
    city_key: str,
    city: dict[str, Any],
    policy: str,
    model: dict[str, Any] | None,
    actions: list[float],
    base_dwell_sec: float,
    max_steps: int,
    max_events: int,
    mpc_horizon: int,
    mpc_discount: float,
    mpc_action_cost: float = 0.05,
    backward_headway_mode: str = "target",
) -> dict[str, Any]:
    line_map = _read_json(Path(city["line_map_path"]))
    target_by_line = _target_headways(line_map)
    station_counts = _line_segment_counts(line_map)
    policy_context = _target_summary_context(line_map)
    speed_by_stop = _segment_speed_by_stop(city)
    line_departures = _line_departures(line_map)
    segment_times = _segment_times_by_line(city)
    sim_start_hour = int(float(city.get("begin", 0.0)) // 3600)
    cmd = [
        "sumo",
        "-c",
        str(city["sumocfg_path"]),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    handled: set[tuple[str, str]] = set()
    last_arrival: dict[tuple[str, str], float] = {}
    first_arrivals = 0
    event_rewards: list[float] = []
    headway_errors: list[float] = []
    holds: list[float] = []
    dwell_values: list[float] = []
    event_records: list[dict[str, Any]] = []
    bunching = 0
    large_gap = 0
    arrived = 0
    started = time.perf_counter()
    sumo_api.start(cmd)
    try:
        for _step in range(max_steps):
            sumo_api.simulationStep()
            now = float(sumo_api.simulation.getTime())
            arrived += len(sumo_api.simulation.getArrivedIDList())
            vehicle_ids = list(sumo_api.vehicle.getIDList())
            vehicles_by_line: dict[str, list[str]] = {}
            for active_vehicle_id in vehicle_ids:
                vehicles_by_line.setdefault(_vehicle_line_uid(active_vehicle_id), []).append(active_vehicle_id)
            for vehicle_id in vehicle_ids:
                stops = sumo_api.vehicle.getStops(vehicle_id, 1)
                if not stops:
                    continue
                stop = stops[0]
                stop_id = str(stop.stoppingPlaceID)
                if not stop_id:
                    continue
                key = (vehicle_id, stop_id)
                if key in handled:
                    continue
                speed = float(sumo_api.vehicle.getSpeed(vehicle_id))
                if speed > 0.1:
                    continue
                line_uid = _vehicle_line_uid(vehicle_id)
                target_hw = target_by_line.get(line_uid, 600.0)
                prev_time = last_arrival.get((line_uid, stop_id))
                has_observed_headway = prev_time is not None
                forward_hw = now - prev_time if prev_time is not None else target_hw
                if backward_headway_mode == "avl":
                    backward_hw = _estimate_backward_headway(
                        sumo_api=sumo_api,
                        line_uid=line_uid,
                        vehicle_id=vehicle_id,
                        stop_id=stop_id,
                        now=now,
                        target_hw=target_hw,
                        active_vehicle_ids=vehicles_by_line.get(line_uid, []),
                        handled=handled,
                        line_departures=line_departures,
                        segment_times=segment_times,
                    )
                else:
                    backward_hw = target_hw
                station_count = station_counts.get(line_uid, 1)
                obs = _event_obs(
                    line_uid=line_uid,
                    stop_id=stop_id,
                    now=now,
                    target_hw=target_hw,
                    forward_hw=forward_hw,
                    backward_hw=backward_hw,
                    station_count=station_count,
                    speed_mps=speed_by_stop.get(stop_id, 8.0),
                )
                hold = _choose_extra_hold(
                    policy=policy,
                    obs=obs,
                    model=model,
                    actions=actions,
                    line_uid=line_uid,
                    station_count=station_count,
                    sim_start_hour=sim_start_hour,
                    mpc_horizon=mpc_horizon,
                    mpc_discount=mpc_discount,
                    mpc_action_cost=mpc_action_cost,
                    policy_context=policy_context,
                )
                dwell = max(0.0, float(base_dwell_sec) + float(hold))
                sumo_api.vehicle.setStopParameter(vehicle_id, 0, "duration", f"{dwell:.3f}")
                handled.add(key)
                last_arrival[(line_uid, stop_id)] = now
                if not has_observed_headway:
                    first_arrivals += 1
                    continue
                abs_error = abs(forward_hw - target_hw)
                headway_errors.append(abs_error)
                holds.append(float(hold))
                dwell_values.append(dwell)
                reward = -abs_error - 0.05 * float(hold)
                event_rewards.append(reward)
                is_bunching = int(forward_hw < 0.5 * target_hw)
                is_large_gap = int(forward_hw > 1.5 * target_hw)
                bunching += is_bunching
                large_gap += is_large_gap
                event_records.append(
                    {
                        "city_key": city_key,
                        "city": city.get("city", city_key),
                        "policy": policy,
                        "line_uid": line_uid,
                        "stop_id": stop_id,
                        "vehicle_id": vehicle_id,
                        "timestamp_sec": now,
                        "target_headway": target_hw,
                        "forward_headway": forward_hw,
                        "backward_headway": backward_hw,
                        "headway_abs_error": abs_error,
                        "reward": reward,
                        "hold_seconds": float(hold),
                        "dwell_seconds": dwell,
                        "bunching": is_bunching,
                        "large_gap": is_large_gap,
                    }
                )
                if len(event_records) >= max_events:
                    break
            if len(event_records) >= max_events:
                break
            if sumo_api.simulation.getMinExpectedNumber() <= 0:
                break
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass
    events = max(1, len(event_records))
    line_rows = []
    for line_uid in sorted({str(row["line_uid"]) for row in event_records}):
        group = [row for row in event_records if str(row["line_uid"]) == line_uid]
        if not group:
            continue
        line_events = len(group)
        line_rows.append(
            {
                "city_key": city_key,
                "city": city.get("city", city_key),
                "policy": policy,
                "line_uid": line_uid,
                "events": line_events,
                "mean_reward": float(np.mean([float(row["reward"]) for row in group])),
                "mean_headway_abs_error": float(np.mean([float(row["headway_abs_error"]) for row in group])),
                "bunching_rate": float(np.mean([float(row["bunching"]) for row in group])),
                "large_gap_rate": float(np.mean([float(row["large_gap"]) for row in group])),
                "mean_hold_seconds": float(np.mean([float(row["hold_seconds"]) for row in group])),
                "mean_dwell_seconds": float(np.mean([float(row["dwell_seconds"]) for row in group])),
            }
        )
    return {
        "city_key": city_key,
        "city": city.get("city", city_key),
        "policy": policy,
        "ok": len(event_records) > 0,
        "events": len(event_records),
        "handled_stop_events": len(handled),
        "warmup_first_arrivals": int(first_arrivals),
        "arrived_vehicles": int(arrived),
        "mean_reward": float(np.mean(event_rewards)) if event_rewards else None,
        "mean_headway_abs_error": float(np.mean(headway_errors)) if headway_errors else None,
        "bunching_rate": float(bunching / events),
        "large_gap_rate": float(large_gap / events),
        "mean_hold_seconds": float(np.mean(holds)) if holds else None,
        "mean_dwell_seconds": float(np.mean(dwell_values)) if dwell_values else None,
        "line_rows": line_rows,
        "elapsed_sec": time.perf_counter() - started,
    }


def _summary_by_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        group = [row for row in rows if row["policy"] == policy and row.get("ok")]
        if not group:
            continue
        out[policy] = {
            "cities": len(group),
            "events": int(sum(int(row["events"]) for row in group)),
            "handled_stop_events": int(sum(int(row.get("handled_stop_events", row["events"])) for row in group)),
            "warmup_first_arrivals": int(sum(int(row.get("warmup_first_arrivals", 0)) for row in group)),
            "mean_reward": float(np.mean([float(row["mean_reward"]) for row in group])),
            "mean_headway_abs_error": float(np.mean([float(row["mean_headway_abs_error"]) for row in group])),
            "bunching_rate": float(np.mean([float(row["bunching_rate"]) for row in group])),
            "large_gap_rate": float(np.mean([float(row["large_gap_rate"]) for row in group])),
            "mean_hold_seconds": float(np.mean([float(row["mean_hold_seconds"]) for row in group])),
            "mean_dwell_seconds": float(np.mean([float(row["mean_dwell_seconds"]) for row in group])),
        }
    return out


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Libsumo Policy Rollout Validation",
        "",
        f"Verdict: **{'PASS' if result['ok'] else 'FAIL'}**",
        "",
        "This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.",
        "",
        "| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, row in result.get("summary_by_policy", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    str(row["cities"]),
                    str(row["events"]),
                    str(row.get("warmup_first_arrivals", 0)),
                    f"{row['mean_reward']:.3f}",
                    f"{row['mean_headway_abs_error']:.3f}",
                    f"{row['bunching_rate']:.3f}",
                    f"{row['mean_hold_seconds']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    stage2 = _read_json(_resolve_path(root, args.stage2_report))
    config = _read_json(_resolve_path(root, args.config))
    cities = stage2.get("cities", {})
    if args.cities:
        cities = {key: value for key, value in cities.items() if key in set(args.cities)}
    invalid = [
        key
        for key, city in cities.items()
        if not city.get("include_bus_stops") or not city.get("additional_path")
    ]
    if invalid:
        raise RuntimeError(
            "SUMO policy rollout requires Stage 2 scenarios generated with explicit bus stops; "
            f"missing for: {', '.join(invalid)}"
        )

    transfer_models: dict[str, dict[str, Any]] = {}
    sanity: dict[str, Any] = {}
    transfer_policies = {
        "h2oplus_dense_policy",
        "h2oplus_dense_mpc_policy",
        "cfcmt_mechanism_policy",
        "cfcmt_mechanism_mpc_policy",
        "cfcmt_similarity_weighted_policy",
        "cfcmt_similarity_weighted_mpc_policy",
        "cfcmt_similarity_weighted_mpc_threshold_guard_policy",
        "cfcmt_similarity_weighted_mpc_threshold_fine_guard_policy",
        "cfcmt_similarity_weighted_mpc_daganzo_guard_policy",
        "cfcmt_similarity_weighted_mpc_daganzo_residual_policy",
        "cfcmt_similarity_weighted_mpc_threshold_residual_policy",
        "cfcmt_similarity_weighted_mpc_rule_selector_policy",
        "cfcmt_target_fewshot_policy",
        "cfcmt_target_fewshot_mpc_policy",
        "cfcmt_target_fewshot_mpc_threshold_guard_policy",
        "cfcmt_target_fewshot_mpc_rule_selector_policy",
        "cfcmt_target_fewshot_mpc_alpha_selector_policy",
        "cfcmt_target_fewshot_mpc_threshold_residual_policy",
        "cfcmt_source_target_fewshot_policy",
        "cfcmt_source_target_fewshot_mpc_policy",
        "cfcmt_source_target_fewshot_mpc_threshold_guard_policy",
        "cfcmt_source_target_fewshot_mpc_rule_selector_policy",
        "cfcmt_source_target_fewshot_mpc_alpha_selector_policy",
        "cfcmt_source_target_fewshot_mpc_threshold_residual_policy",
    }
    requested_policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    if not args.skip_transfer_policies and any(policy in transfer_policies for policy in requested_policies):
        transfer_models, sanity = _build_transfer_models(config=config, root=root, args=args)

    sumo_api = _load_libsumo()
    rows = []
    for city_key, city in cities.items():
        for policy in requested_policies:
            model = transfer_models.get(city_key) if policy in transfer_policies else None
            row = _run_city_policy(
                sumo_api=sumo_api,
                city_key=city_key,
                city=city,
                policy=policy,
                model=model,
                actions=list(args.policy_actions),
                base_dwell_sec=float(city.get("base_dwell_sec", args.base_dwell_sec)),
                max_steps=int(args.max_steps),
                max_events=int(args.max_events_per_city_policy),
                mpc_horizon=int(args.mpc_horizon),
                mpc_discount=float(args.mpc_discount),
                mpc_action_cost=float(args.mpc_action_cost),
                backward_headway_mode=str(args.backward_headway_mode),
            )
            rows.append(row)
            if not args.quiet:
                print(
                    f"[sumo-policy] {city_key} {policy} events={row['events']} reward={row['mean_reward']}",
                    flush=True,
                )
    result = {
        "ok": bool(rows) and all(row.get("ok") for row in rows),
        "experiment": "libsumo_closed_loop_policy_diagnostic",
        "performance_validation": True,
        "calibrated_field_counterfactual": False,
        "warmup_first_arrivals_excluded_from_metrics": True,
        "stage2_report": str(_resolve_path(root, args.stage2_report)),
        "policies": requested_policies,
        "rows": rows,
        "summary_by_policy": _summary_by_policy(rows),
        "model_actions": list(args.actions),
        "policy_actions": list(args.policy_actions),
        "ridge": float(args.ridge),
        "mpc_horizon": int(args.mpc_horizon),
        "mpc_discount": float(args.mpc_discount),
        "mpc_action_cost": float(args.mpc_action_cost),
        "backward_headway_mode": str(args.backward_headway_mode),
        "target_adaptation_lines": int(args.target_adaptation_lines),
        "target_adaptation_mass": float(args.target_adaptation_mass),
        "target_adaptation_rows_by_city": {
            key: int(model.get("target_adaptation_rows", 0))
            for key, model in transfer_models.items()
        },
        "transfer_sanity_available": bool(sanity),
    }
    out = _resolve_path(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_out = _resolve_path(root, args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_markdown(result), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-report", type=Path, default=DEFAULT_STAGE2_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument(
        "--policies",
        default="no_hold,fixed_30,daganzo_policy,threshold_equalization_policy,h2oplus_dense_policy,h2oplus_dense_mpc_policy,cfcmt_mechanism_policy,cfcmt_mechanism_mpc_policy,cfcmt_similarity_weighted_policy,cfcmt_similarity_weighted_mpc_policy",
    )
    parser.add_argument("--policy-actions", type=_parse_float_list, default=list(POLICY_ACTIONS))
    parser.add_argument("--actions", type=_parse_float_list, default=list(POLICY_ACTIONS))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--transfer-max-lines-per-city", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--source-weight-temperature", type=float, default=1.0)
    parser.add_argument("--source-weight-floor", type=float, default=0.05)
    parser.add_argument("--target-adaptation-lines", type=int, default=0)
    parser.add_argument("--target-adaptation-mass", type=float, default=1.0)
    parser.add_argument("--skip-transfer-policies", action="store_true")
    parser.add_argument("--base-dwell-sec", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument("--max-events-per-city-policy", type=int, default=120)
    parser.add_argument("--mpc-horizon", type=int, default=3)
    parser.add_argument("--mpc-discount", type=float, default=0.85)
    parser.add_argument("--mpc-action-cost", type=float, default=0.05)
    parser.add_argument("--backward-headway-mode", choices=["target", "avl"], default="target")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    if result["ok"]:
        print(json.dumps({"ok": True, "summary_by_policy": result["summary_by_policy"]}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
