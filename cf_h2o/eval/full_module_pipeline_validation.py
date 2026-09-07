"""Staged cross-city validation that turns on CF-H2O modules one by one.

The existing paper experiments use deterministic sufficient-statistics ridge
models. This script is intentionally separate: it uses the same full-route
open-transit transition generator, then progressively adds the implemented
CF-H2O modules:

1. H2O+-style dense ridge residual.
2. CFCMT ridge residual with hand-designed mechanism feature maps.
3. Neural causal-factored residual world model with template parents.
4. AutoDAG-learned graph posterior and parent masks.
5. Time-varying latent mechanism factors trained jointly with residuals.
6. Local graph encoder features for the latent factors.
7. Factor-wise trust and the H2O trust bridge.

The output is a diagnostic experiment, not a replacement for the deterministic
paper suite. It reports where the full neural pipeline helps or hurts relative
to the ridge instantiation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.encoders.local_graph_encoder import LocalGraphEncoder
from cf_h2o.eval.cross_city_open_transit_validation import expand_splits
from cf_h2o.eval.cross_city_performance_validation import (
    DYNAMIC_OUTPUT_INDICES,
    OBS_NAMES,
    OUTPUT_NAMES,
    RidgeStats,
    cfcmt_features,
    h2o_features,
    _iter_city_chunks,
    _read_json,
    _repo_root,
    _resolve_path,
    _scaled_inputs,
    _stable_unit,
)
from cf_h2o.graph.auto_dag import AutoDAGDiscoverer
from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.latent.factor_encoder import TimeVaryingFactorEncoder, build_history_windows
from cf_h2o.rl.cf_h2o_trainer import CFH2OTrainer
from cf_h2o.rl.h2o_mcwm_bridge import H2OFactorTrustBridge
from cf_h2o.schemas import GraphPosterior, GraphSpec, MechanismSpec, TransitionBatch
from cf_h2o.trust.factor_trust import FactorTrustEstimator, FactorWiseTrustWeightProvider
from cf_h2o.trust.weight_composer import WeightComposer
from cf_h2o.world_model.causal_factored_residual import (
    CausalFactoredResidualWorldModel,
    _batch_node_values,
    _target_for_spec,
)


ACTION_NAMES = ["holding"]
AUX_OBS_NAMES = [
    "hour_sin",
    "hour_cos",
    "forward_headway_deviation",
    "backward_headway_deviation",
    "abs_forward_headway_deviation",
    "abs_backward_headway_deviation",
    "action_x_gap",
    "action_x_waiting_passengers",
]
ID_PARENT_NAMES = {"line_id@t", "bus_id@t"}
MECHANISM_CHILDREN = {
    "headway": [
        "forward_headway@t1",
        "backward_headway@t1",
        "gap@t1",
        "co_line_forward_headway@t1",
        "co_line_backward_headway@t1",
    ],
    "demand": ["waiting_passengers@t1"],
    "dwell": ["base_stop_duration@t1"],
    "speed": ["segment_mean_speed@t1"],
    "reward": ["reward@t1"],
}
RIGID_FIRST_MECHANISM_OUTPUT_NAMES = {
    "headway": [
        "next_forward_headway",
        "next_backward_headway",
        "next_gap",
        "next_co_line_forward_headway",
        "next_co_line_backward_headway",
    ],
    "demand": ["next_waiting_passengers"],
    "dwell": ["next_base_stop_duration"],
    "speed": ["next_segment_mean_speed"],
    "reward": ["reward"],
}
TEMPLATE_PARENTS = {
    "headway": [
        "forward_headway@t",
        "backward_headway@t",
        "target_headway@t",
        "gap@t",
        "segment_mean_speed@t",
        "time_period@t",
        "holding@t",
    ],
    "demand": [
        "waiting_passengers@t",
        "target_headway@t",
        "segment_mean_speed@t",
        "station_id@t",
        "time_period@t",
        "holding@t",
    ],
    "dwell": [
        "waiting_passengers@t",
        "base_stop_duration@t",
        "segment_mean_speed@t",
        "time_period@t",
        "holding@t",
    ],
    "speed": [
        "segment_mean_speed@t",
        "station_id@t",
        "time_period@t",
        "sim_time@t",
    ],
    "reward": [
        "forward_headway@t",
        "backward_headway@t",
        "waiting_passengers@t",
        "target_headway@t",
        "gap@t",
        "segment_mean_speed@t",
        "holding@t",
    ],
}
LATENT_DIMS = {"headway": 2, "demand": 2, "dwell": 2, "speed": 2, "reward": 1}
LOCAL_MODE_NONE = "none"
LOCAL_MODE_PROXY_ENCODER = "proxy_encoder"
LOCAL_MODE_STATIC = "deterministic_static"
ANCHOR_MODE_SIM = "sim"
ANCHOR_MODE_RIGID_CFCMT = "rigid_cfcmt"
RIGID_FIRST_DELTA_MODES = ["dense_delta", "static_local_delta", "dense_static_local_delta"]
RIGID_FIRST_ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CITY_TENSOR_CACHE_VERSION = 1
LOCAL_STATIC_FEATURE_NAMES = [
    "route_position",
    "prev_station_distance",
    "next_station_distance",
    "prev_stop_demand",
    "current_stop_demand",
    "next_stop_demand",
    "same_line_near_stop_demand",
    "segment_speed_quantile",
    "headway_imbalance",
    "abs_headway_imbalance",
    "co_line_headway_imbalance",
    "local_headway_pressure",
    "demand_per_target_headway",
    "holding_per_target_headway",
    "route_center_distance",
]


@dataclass
class CityTensorData:
    key: str
    city: str
    observations: np.ndarray
    actions: np.ndarray
    sim_next_observations: np.ndarray
    real_next_observations: np.ndarray
    sim_y: np.ndarray
    real_y: np.ndarray

    @property
    def n(self) -> int:
        return int(self.observations.shape[0])

    def take(self, indices: np.ndarray) -> "CityTensorData":
        return CityTensorData(
            key=self.key,
            city=self.city,
            observations=self.observations[indices],
            actions=self.actions[indices],
            sim_next_observations=self.sim_next_observations[indices],
            real_next_observations=self.real_next_observations[indices],
            sim_y=self.sim_y[indices],
            real_y=self.real_y[indices],
        )

    def obs_action(self) -> np.ndarray:
        return np.concatenate([self.observations, self.actions], axis=1)


@dataclass
class Scaler:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    reward_mean: torch.Tensor
    reward_std: torch.Tensor
    local_mean: torch.Tensor
    local_std: torch.Tensor
    local_speed_edges: np.ndarray


@dataclass
class RidgeModelBundle:
    h2o_beta: np.ndarray
    cfcmt_beta: dict[str, np.ndarray]


@dataclass
class RigidFirstDeltaBundle:
    mode: str
    beta: dict[str, np.ndarray]


@dataclass
class RigidFirstDeltaEnsembleBundle:
    mode: str
    models: dict[str, RigidFirstDeltaBundle]
    weights: dict[str, float]


@dataclass
class DenseResidualEnsembleBundle:
    models: dict[str, np.ndarray]
    weights: dict[str, float]


class DummyH2O:
    def __init__(self):
        self._total_steps = 0
        self.external_sim_weight_provider = None
        self.replay_buffer = None

    def train(self, batch_size: int, pretrain_steps: int = 0) -> dict[str, Any]:
        self._total_steps += 1
        return {"dummy_h2o_train_calls": self._total_steps, "batch_size": int(batch_size)}


class SimAnchoredWorldModelView:
    def __init__(self, model: CausalFactoredResidualWorldModel):
        self.model = model
        self.mechanism_specs = model.mechanism_specs

    def predict(self, batch: TransitionBatch, theta_dict: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        return _predict_sim_anchored(self.model, batch, theta_dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _full_next_observations(obs_action: np.ndarray, y: np.ndarray) -> np.ndarray:
    next_obs = obs_action[:, : len(OBS_NAMES)].astype(np.float32, copy=True)
    for out_idx, obs_idx in enumerate(DYNAMIC_OUTPUT_INDICES):
        next_obs[:, obs_idx] = y[:, out_idx].astype(np.float32)
    return next_obs


def _aux_observations(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    hour = observations[:, 3].astype(np.float64)
    action = actions[:, 0].astype(np.float64)
    fwd_dev = observations[:, 5].astype(np.float64) - observations[:, 8].astype(np.float64)
    bwd_dev = observations[:, 6].astype(np.float64) - observations[:, 8].astype(np.float64)
    gap = observations[:, 11].astype(np.float64)
    waiting = observations[:, 7].astype(np.float64)
    return np.column_stack(
        [
            np.sin(hour * math.tau),
            np.cos(hour * math.tau),
            fwd_dev,
            bwd_dev,
            np.abs(fwd_dev),
            np.abs(bwd_dev),
            (action / 60.0) * gap,
            (action / 60.0) * waiting,
        ]
    ).astype(np.float32)


def _speed_quantile(speed: np.ndarray, edges: np.ndarray | None) -> np.ndarray:
    speed = speed.astype(np.float64, copy=False)
    if edges is None or len(edges) < 2:
        order = np.argsort(speed, kind="mergesort")
        ranks = np.empty_like(speed, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, num=len(speed), endpoint=True) if len(speed) else []
        return ranks
    bins = np.searchsorted(edges, speed, side="right") - 1
    return np.clip(bins.astype(np.float64) / float(max(1, len(edges) - 2)), 0.0, 1.0)


def _neighbor_stop_features(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = observations.shape[0]
    station = observations[:, 2].astype(np.float64, copy=False)
    direction = np.round(observations[:, 4].astype(np.float64, copy=False) * 10_000).astype(np.int64)
    waiting = observations[:, 7].astype(np.float64, copy=False)
    line = np.round(observations[:, 0].astype(np.float64, copy=False) * 1_000_000).astype(np.int64)
    time_period = np.round(observations[:, 3].astype(np.float64, copy=False) * 10_000).astype(np.int64)

    prev_distance = np.zeros(n_rows, dtype=np.float64)
    next_distance = np.zeros(n_rows, dtype=np.float64)
    prev_demand = waiting.astype(np.float64, copy=True)
    next_demand = waiting.astype(np.float64, copy=True)
    order = np.lexsort((station, time_period, direction, line))
    sorted_line = line[order]
    sorted_direction = direction[order]
    sorted_time = time_period[order]
    starts = np.flatnonzero(
        np.r_[
            True,
            (sorted_line[1:] != sorted_line[:-1])
            | (sorted_direction[1:] != sorted_direction[:-1])
            | (sorted_time[1:] != sorted_time[:-1]),
        ]
    )
    ends = np.r_[starts[1:], n_rows]
    for start, end in zip(starts, ends):
        idx = order[start:end]
        if idx.size <= 1:
            continue
        station_values = station[idx]
        unique_station, inverse, counts = np.unique(station_values, return_inverse=True, return_counts=True)
        if unique_station.size <= 1:
            continue
        demand_sum = np.bincount(inverse, weights=waiting[idx], minlength=unique_station.size)
        demand_mean = demand_sum / np.maximum(counts.astype(np.float64), 1.0)
        prev_pos = np.maximum(inverse - 1, 0)
        next_pos = np.minimum(inverse + 1, unique_station.size - 1)
        prev_distance[idx] = np.maximum(station_values - unique_station[prev_pos], 0.0)
        next_distance[idx] = np.maximum(unique_station[next_pos] - station_values, 0.0)
        prev_demand[idx] = demand_mean[prev_pos]
        next_demand[idx] = demand_mean[next_pos]
    return prev_distance, next_distance, prev_demand, next_demand


def _static_local_features_np(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    speed_edges: np.ndarray | None = None,
) -> np.ndarray:
    station = observations[:, 2].astype(np.float64, copy=False)
    fwd = observations[:, 5].astype(np.float64, copy=False)
    bwd = observations[:, 6].astype(np.float64, copy=False)
    waiting = observations[:, 7].astype(np.float64, copy=False)
    target = np.maximum(observations[:, 8].astype(np.float64, copy=False), 1.0)
    co_fwd = observations[:, 12].astype(np.float64, copy=False)
    co_bwd = observations[:, 13].astype(np.float64, copy=False)
    speed = observations[:, 14].astype(np.float64, copy=False)
    action = actions[:, 0].astype(np.float64, copy=False)
    prev_distance, next_distance, prev_demand, next_demand = _neighbor_stop_features(observations)
    same_line_near_demand = (prev_demand + waiting + next_demand) / 3.0
    headway_imbalance = (fwd - bwd) / target
    co_line_imbalance = (co_fwd - co_bwd) / target
    local_headway_pressure = (target - np.minimum(fwd, bwd)) / target
    features = np.column_stack(
        [
            station,
            prev_distance,
            next_distance,
            prev_demand,
            waiting,
            next_demand,
            same_line_near_demand,
            _speed_quantile(speed, speed_edges),
            headway_imbalance,
            np.abs(headway_imbalance),
            co_line_imbalance,
            local_headway_pressure,
            waiting / target,
            action / target,
            np.abs(station - 0.5),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _scaled_output_y(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[:, 0] = y[:, 0] / 900.0
    out[:, 1] = y[:, 1] / 900.0
    out[:, 2] = np.log1p(np.maximum(y[:, 2], 0.0)) / 5.0
    out[:, 3] = y[:, 3] / 80.0
    out[:, 4] = y[:, 4] / 900.0
    out[:, 5] = y[:, 5] / 900.0
    out[:, 6] = y[:, 6] / 900.0
    out[:, 7] = y[:, 7] / 15.0
    out[:, 8] = y[:, 8]
    return np.nan_to_num(out, copy=False)


SIMILARITY_FEATURE_SETS = ("full", "obs_sim", "obs_only", "sim_only", "local_only")


def _city_similarity_vector(
    data: CityTensorData,
    *,
    max_rows: int,
    seed: int,
    feature_set: str = "full",
) -> np.ndarray:
    if feature_set not in SIMILARITY_FEATURE_SETS:
        raise ValueError(f"unknown similarity feature_set={feature_set!r}")
    idx = _select_rows(data.n, int(max_rows), seed, f"{data.key}:similarity") if int(max_rows) > 0 else np.arange(data.n, dtype=np.int64)
    sample = data.take(idx)
    scaled_obs_action = _scaled_inputs(sample.obs_action())
    obs_cols = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    sim = _scaled_output_y(sample.sim_y)
    local = _static_local_features_np(sample.observations, sample.actions, speed_edges=None).astype(np.float64)
    local[:, [1, 2]] = np.log1p(np.maximum(local[:, [1, 2]], 0.0))
    local[:, [3, 4, 5, 6]] = np.log1p(np.maximum(local[:, [3, 4, 5, 6]], 0.0)) / 5.0
    local[:, 7:] = np.clip(local[:, 7:], -5.0, 5.0)
    pieces: list[np.ndarray] = []
    if feature_set in {"full", "obs_sim", "obs_only"}:
        pieces.append(scaled_obs_action[:, obs_cols])
    if feature_set in {"full", "obs_sim", "sim_only"}:
        pieces.append(sim)
    if feature_set in {"full", "local_only"}:
        pieces.append(local)
    matrix = np.concatenate(pieces, axis=1).astype(np.float64, copy=False)
    quantiles = np.quantile(matrix, [0.1, 0.5, 0.9], axis=0)
    summary = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0), quantiles.reshape(-1)])
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def _annotate_source_target_similarity(
    folds: list[dict[str, Any]],
    target: CityTensorData,
    city_data: dict[str, CityTensorData],
    args: argparse.Namespace,
) -> None:
    if not folds:
        return
    target_vec = _city_similarity_vector(
        target,
        max_rows=int(args.selector_similarity_max_rows),
        seed=int(args.seed),
        feature_set=str(getattr(args, "selector_similarity_feature_set", "full")),
    )
    distances = []
    for fold in folds:
        heldout_key = str(fold["heldout_source_env"])
        source_vec = _city_similarity_vector(
            city_data[heldout_key],
            max_rows=int(args.selector_similarity_max_rows),
            seed=int(args.seed),
            feature_set=str(getattr(args, "selector_similarity_feature_set", "full")),
        )
        distance = float(np.linalg.norm(source_vec - target_vec) / math.sqrt(max(1, target_vec.shape[0])))
        fold["target_similarity_distance"] = distance
        distances.append(distance)
    positive = [value for value in distances if value > 1e-12]
    scale = float(np.median(positive)) if positive else 1.0
    scale = max(scale * float(args.rigid_first_similarity_temperature), 1e-6)
    weights = np.exp(-np.asarray(distances, dtype=np.float64) / scale)
    mean_weight = float(weights.mean()) if weights.size else 1.0
    weights = weights / max(mean_weight, 1e-12)
    for fold, weight in zip(folds, weights.tolist()):
        fold["target_similarity_weight"] = float(weight)


def _source_similarity_weights(
    target: CityTensorData,
    sources: dict[str, CityTensorData],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not sources:
        return {}, {"weighting": "similarity", "source_distances": {}, "source_weights": {}}
    weighting = str(getattr(args, "rigid_first_ensemble_weighting", "similarity"))
    feature_set = str(getattr(args, "selector_similarity_feature_set", "full"))
    if weighting == "uniform":
        weights = {key: 1.0 / float(len(sources)) for key in sources}
        return weights, {
            "weighting": "uniform",
            "feature_set": feature_set,
            "source_distances": {},
            "source_weights": weights,
        }
    target_vec = _city_similarity_vector(
        target,
        max_rows=int(args.selector_similarity_max_rows),
        seed=int(args.seed),
        feature_set=feature_set,
    )
    distances: dict[str, float] = {}
    for key, data in sources.items():
        source_vec = _city_similarity_vector(
            data,
            max_rows=int(args.selector_similarity_max_rows),
            seed=int(args.seed),
            feature_set=feature_set,
        )
        distances[key] = float(np.linalg.norm(source_vec - target_vec) / math.sqrt(max(1, target_vec.shape[0])))
    positive = [value for value in distances.values() if value > 1e-12]
    scale = float(np.median(positive)) if positive else 1.0
    scale = max(scale * float(args.rigid_first_similarity_temperature), 1e-6)
    raw = {key: math.exp(-distance / scale) for key, distance in distances.items()}
    total = sum(raw.values())
    weights = {key: (value / total if total > 1e-12 else 1.0 / float(len(raw))) for key, value in raw.items()}
    return weights, {
        "weighting": "similarity",
        "feature_set": feature_set,
        "temperature": float(args.rigid_first_similarity_temperature),
        "max_rows": int(args.selector_similarity_max_rows),
        "source_distances": distances,
        "source_weights": weights,
    }


def _extend_observations(observations: np.ndarray, actions: np.ndarray, use_aux: bool) -> np.ndarray:
    if not use_aux:
        return observations.astype(np.float32, copy=False)
    return np.concatenate([observations.astype(np.float32, copy=False), _aux_observations(observations, actions)], axis=1)


def _obs_names(use_aux: bool) -> list[str]:
    return list(OBS_NAMES) + (list(AUX_OBS_NAMES) if use_aux else [])


def _load_city_data(key: str, spec: dict[str, Any], root: Path, args: argparse.Namespace) -> CityTensorData:
    cache_path = _city_tensor_cache_path(key, spec, root, args)
    if cache_path is not None and cache_path.exists() and not bool(args.refresh_city_tensor_cache):
        return _read_city_tensor_cache(cache_path)

    env_path = _resolve_path(root, spec["env_path"])
    obs_items: list[np.ndarray] = []
    action_items: list[np.ndarray] = []
    sim_y_items: list[np.ndarray] = []
    real_y_items: list[np.ndarray] = []
    for obs_action, sim_y, real_y, _meta in _iter_city_chunks(
        env_path,
        max_lines=args.max_lines_per_city or 0,
        actions=args.actions,
        chunk_rows=args.chunk_rows,
    ):
        obs_items.append(obs_action[:, : len(OBS_NAMES)].astype(np.float32, copy=False))
        action_items.append(obs_action[:, len(OBS_NAMES) :].astype(np.float32, copy=False))
        sim_y_items.append(sim_y.astype(np.float32, copy=False))
        real_y_items.append(real_y.astype(np.float32, copy=False))

    if not obs_items:
        raise RuntimeError(f"{key}: no generated transitions")

    observations = np.concatenate(obs_items, axis=0)
    actions = np.concatenate(action_items, axis=0)
    sim_y = np.concatenate(sim_y_items, axis=0)
    real_y = np.concatenate(real_y_items, axis=0)
    obs_action = np.concatenate([observations, actions], axis=1)
    data = CityTensorData(
        key=key,
        city=str(spec.get("city", key)),
        observations=observations,
        actions=actions,
        sim_next_observations=_full_next_observations(obs_action, sim_y),
        real_next_observations=_full_next_observations(obs_action, real_y),
        sim_y=sim_y,
        real_y=real_y,
    )
    if cache_path is not None and not bool(args.no_write_city_tensor_cache):
        _write_city_tensor_cache(cache_path, data)
    return data


def _city_tensor_cache_path(key: str, spec: dict[str, Any], root: Path, args: argparse.Namespace) -> Path | None:
    cache_dir = getattr(args, "city_tensor_cache_dir", None)
    if bool(getattr(args, "no_city_tensor_cache", False)) or cache_dir in {None, ""}:
        return None
    env_path = _resolve_path(root, spec["env_path"])
    payload = {
        "version": CITY_TENSOR_CACHE_VERSION,
        "key": key,
        "city": str(spec.get("city", key)),
        "env_path": str(env_path),
        "max_lines_per_city": int(args.max_lines_per_city or 0),
        "actions": [float(value) for value in args.actions],
        "tag": str(getattr(args, "city_tensor_cache_tag", "")),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return _resolve_path(root, cache_dir) / f"{key}_{digest}.npz"


def _read_city_tensor_cache(path: Path) -> CityTensorData:
    with np.load(path, allow_pickle=False) as data:
        observations = data["observations"].astype(np.float32, copy=False)
        actions = data["actions"].astype(np.float32, copy=False)
        sim_y = data["sim_y"].astype(np.float32, copy=False)
        real_y = data["real_y"].astype(np.float32, copy=False)
        key = str(data["key"].item())
        city = str(data["city"].item())
    obs_action = np.concatenate([observations, actions], axis=1)
    return CityTensorData(
        key=key,
        city=city,
        observations=observations,
        actions=actions,
        sim_next_observations=_full_next_observations(obs_action, sim_y),
        real_next_observations=_full_next_observations(obs_action, real_y),
        sim_y=sim_y,
        real_y=real_y,
    )


def _write_city_tensor_cache(path: Path, data: CityTensorData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            key=np.array(data.key),
            city=np.array(data.city),
            observations=data.observations.astype(np.float32, copy=False),
            actions=data.actions.astype(np.float32, copy=False),
            sim_y=data.sim_y.astype(np.float32, copy=False),
            real_y=data.real_y.astype(np.float32, copy=False),
        )
    tmp.replace(path)


def _select_rows(n_rows: int, max_rows: int, seed: int, label: str) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    unit = int(_stable_unit(label, modulus=1_000_000))
    rng = np.random.default_rng(seed + unit)
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False).astype(np.int64))


def _select_city_subset(data: CityTensorData, max_rows: int, seed: int, label: str) -> CityTensorData:
    return data.take(_select_rows(data.n, max_rows, seed, label))


def _concat_preselected_data(items: Iterable[CityTensorData]) -> CityTensorData:
    selected = list(items)
    if not selected:
        raise ValueError("cannot concatenate an empty city tensor list")
    return CityTensorData(
        key="+".join(item.key for item in selected),
        city="+".join(item.city for item in selected),
        observations=np.concatenate([item.observations for item in selected], axis=0),
        actions=np.concatenate([item.actions for item in selected], axis=0),
        sim_next_observations=np.concatenate([item.sim_next_observations for item in selected], axis=0),
        real_next_observations=np.concatenate([item.real_next_observations for item in selected], axis=0),
        sim_y=np.concatenate([item.sim_y for item in selected], axis=0),
        real_y=np.concatenate([item.real_y for item in selected], axis=0),
    )


def _concat_data(items: Iterable[CityTensorData], max_rows_per_city: int, seed: int, suffix: str) -> CityTensorData:
    selected = []
    for item in items:
        selected.append(_select_city_subset(item, max_rows_per_city, seed, f"{item.key}:{suffix}"))
    return _concat_preselected_data(selected)


def _fit_scaler(source: CityTensorData, device: torch.device, use_aux: bool) -> Scaler:
    obs = _extend_observations(source.observations, source.actions, use_aux)
    sim_next = _extend_observations(source.sim_next_observations, source.actions, use_aux)
    real_next = _extend_observations(source.real_next_observations, source.actions, use_aux)
    obs_like = np.concatenate([obs, sim_next, real_next], axis=0)
    rewards = np.concatenate([source.sim_y[:, -1:], source.real_y[:, -1:]], axis=0)
    speed_edges = np.quantile(source.observations[:, 14].astype(np.float64), np.linspace(0.0, 1.0, 101)).astype(np.float32)
    local = _static_local_features_np(source.observations, source.actions, speed_edges=speed_edges)
    obs_mean = torch.as_tensor(obs_like.mean(axis=0), dtype=torch.float32, device=device)
    obs_std = torch.as_tensor(obs_like.std(axis=0), dtype=torch.float32, device=device).clamp_min(1e-4)
    action_mean = torch.as_tensor(source.actions.mean(axis=0), dtype=torch.float32, device=device)
    action_std = torch.as_tensor(source.actions.std(axis=0), dtype=torch.float32, device=device).clamp_min(1e-4)
    reward_mean = torch.as_tensor(rewards.mean(axis=0), dtype=torch.float32, device=device).reshape(1)
    reward_std = torch.as_tensor(rewards.std(axis=0), dtype=torch.float32, device=device).reshape(1).clamp_min(1e-4)
    local_mean = torch.as_tensor(local.mean(axis=0), dtype=torch.float32, device=device)
    local_std = torch.as_tensor(local.std(axis=0), dtype=torch.float32, device=device).clamp_min(1e-4)
    return Scaler(obs_mean, obs_std, action_mean, action_std, reward_mean, reward_std, local_mean, local_std, speed_edges)


def _to_batch(data: CityTensorData, scaler: Scaler, device: torch.device, use_aux: bool) -> TransitionBatch:
    obs_np = _extend_observations(data.observations, data.actions, use_aux)
    real_next_np = _extend_observations(data.real_next_observations, data.actions, use_aux)
    sim_next_np = _extend_observations(data.sim_next_observations, data.actions, use_aux)
    local_np = _static_local_features_np(data.observations, data.actions, speed_edges=scaler.local_speed_edges)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    act = torch.as_tensor(data.actions, dtype=torch.float32, device=device)
    real_next = torch.as_tensor(real_next_np, dtype=torch.float32, device=device)
    sim_next = torch.as_tensor(sim_next_np, dtype=torch.float32, device=device)
    local = torch.as_tensor(local_np, dtype=torch.float32, device=device)
    real_reward = torch.as_tensor(data.real_y[:, -1], dtype=torch.float32, device=device)
    sim_reward = torch.as_tensor(data.sim_y[:, -1], dtype=torch.float32, device=device)
    norm_obs = (obs - scaler.obs_mean) / scaler.obs_std
    norm_act = (act - scaler.action_mean) / scaler.action_std
    norm_real_next = (real_next - scaler.obs_mean) / scaler.obs_std
    norm_sim_next = (sim_next - scaler.obs_mean) / scaler.obs_std
    norm_local = (local - scaler.local_mean) / scaler.local_std
    norm_real_reward = (real_reward - scaler.reward_mean) / scaler.reward_std
    norm_sim_reward = (sim_reward - scaler.reward_mean) / scaler.reward_std
    return TransitionBatch(
        observations=norm_obs,
        actions=norm_act,
        rewards=norm_real_reward.reshape(-1),
        next_observations=norm_real_next,
        dones=torch.zeros(data.n, dtype=torch.float32, device=device),
        metadata={
            "obs_names": _obs_names(use_aux),
            "action_names": list(ACTION_NAMES),
            "sim_next_observations": norm_sim_next,
            "sim_rewards": norm_sim_reward.reshape(-1),
            "local_static_features": norm_local,
            "local_static_feature_names": list(LOCAL_STATIC_FEATURE_NAMES),
        },
    )


def _sim_batch(data: CityTensorData, scaler: Scaler, device: torch.device, use_aux: bool) -> TransitionBatch:
    batch = _to_batch(data, scaler, device, use_aux)
    return TransitionBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.metadata["sim_rewards"],
        next_observations=batch.metadata["sim_next_observations"],
        dones=batch.dones,
        metadata={"obs_names": _obs_names(use_aux), "action_names": list(ACTION_NAMES)},
    )


def _install_cfcmt_base(
    batch: TransitionBatch,
    data: CityTensorData,
    scaler: Scaler,
    device: torch.device,
    use_aux: bool,
    ridge_model: RidgeModelBundle,
) -> None:
    cfcmt_y = _predict_cfcmt_y(ridge_model, data.obs_action(), data.sim_y).astype(np.float32)
    cfcmt_next = _full_next_observations(data.obs_action(), cfcmt_y).astype(np.float32)
    cfcmt_next_ext = _extend_observations(cfcmt_next, data.actions, use_aux)
    next_tensor = torch.as_tensor(cfcmt_next_ext, dtype=torch.float32, device=device)
    reward_tensor = torch.as_tensor(cfcmt_y[:, -1], dtype=torch.float32, device=device)
    batch.metadata["sim_next_observations"] = (next_tensor - scaler.obs_mean) / scaler.obs_std
    batch.metadata["sim_rewards"] = ((reward_tensor - scaler.reward_mean) / scaler.reward_std).reshape(-1)
    batch.metadata["base_anchor_mode"] = ANCHOR_MODE_RIGID_CFCMT


def _slice_batch(batch: TransitionBatch, start: int, end: int) -> TransitionBatch:
    metadata = {}
    for key, value in batch.metadata.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch.batch_size,):
            metadata[key] = value[start:end]
        else:
            metadata[key] = value
    return TransitionBatch(
        observations=batch.observations[start:end],
        actions=batch.actions[start:end],
        rewards=batch.rewards[start:end],
        next_observations=batch.next_observations[start:end],
        dones=batch.dones[start:end],
        metadata=metadata,
    )


def _template_specs(latent: bool) -> list[MechanismSpec]:
    specs = []
    for name, children in MECHANISM_CHILDREN.items():
        specs.append(
            MechanismSpec(
                name=name,
                child_names=list(children),
                parent_names=list(TEMPLATE_PARENTS[name]),
                latent_dim=LATENT_DIMS[name] if latent else 0,
                output_dim=len(children),
                loss_type="mse",
            )
        )
    return specs


def _template_posterior(batch: TransitionBatch) -> GraphPosterior:
    registry = FeatureRegistry.from_transition_dataset(batch)
    hard_mask = registry.build_temporal_hard_mask()
    edge_probs = torch.where(hard_mask, torch.full_like(hard_mask, 0.02, dtype=torch.float32), torch.zeros_like(hard_mask, dtype=torch.float32))
    idx = registry.node_index
    for name, parents in TEMPLATE_PARENTS.items():
        for child in MECHANISM_CHILDREN[name]:
            if child not in idx:
                continue
            for parent in parents:
                if parent in idx and bool(hard_mask[idx[parent], idx[child]]):
                    edge_probs[idx[parent], idx[child]] = 1.0
    graph = GraphSpec(
        node_names=registry.node_names,
        adjacency=(edge_probs >= 0.8).float(),
        hard_mask=hard_mask,
        edge_probs=edge_probs,
        node_groups=registry.node_groups(),
    )
    return GraphPosterior(
        graphs=[graph],
        log_weights=torch.zeros(1),
        edge_marginals=edge_probs,
        diagnostics={"method": "template", "node_names": registry.node_names, "mechanisms": [spec.__dict__ for spec in _template_specs(True)]},
    )


def _learned_specs(
    posterior: GraphPosterior,
    latent: bool,
    threshold: float,
    top_k: int,
    *,
    allow_id_parents: bool,
) -> list[MechanismSpec]:
    node_names = posterior.graphs[0].node_names if posterior.graphs else list(posterior.diagnostics["node_names"])
    node_index = {name: idx for idx, name in enumerate(node_names)}
    hard_mask = posterior.graphs[0].hard_mask if posterior.graphs else torch.ones_like(posterior.edge_marginals, dtype=torch.bool)
    specs = []
    for name, children in MECHANISM_CHILDREN.items():
        parent_scores: dict[str, float] = {}
        for child in children:
            if child not in node_index:
                continue
            child_idx = node_index[child]
            allowed = torch.where(hard_mask[:, child_idx])[0]
            scored = []
            for parent_idx in allowed.tolist():
                parent = node_names[parent_idx]
                if not parent.endswith("@t"):
                    continue
                if not allow_id_parents and parent in ID_PARENT_NAMES:
                    continue
                if any(marker in parent.lower() for marker in ("domain", "source", "city", "@t1", "reward@t1")):
                    continue
                prob = float(posterior.edge_marginals[parent_idx, child_idx].detach().cpu())
                scored.append((parent, prob))
            for parent, prob in scored:
                if prob >= threshold:
                    parent_scores[parent] = max(parent_scores.get(parent, 0.0), prob)
            if not parent_scores and scored:
                for parent, prob in sorted(scored, key=lambda item: item[1], reverse=True)[: max(1, top_k)]:
                    parent_scores[parent] = max(parent_scores.get(parent, 0.0), prob)
        parents = [parent for parent, _ in sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        if not parents:
            parents = list(TEMPLATE_PARENTS[name])
        specs.append(
            MechanismSpec(
                name=name,
                child_names=list(children),
                parent_names=parents,
                latent_dim=LATENT_DIMS[name] if latent else 0,
                output_dim=len(children),
                loss_type="mse",
            )
        )
    return specs


def _remove_id_parent_edges(posterior: GraphPosterior) -> GraphPosterior:
    if not posterior.graphs:
        return posterior
    node_names = list(posterior.graphs[0].node_names)
    node_index = {name: idx for idx, name in enumerate(node_names)}
    forbidden = [node_index[name] for name in ID_PARENT_NAMES if name in node_index]
    if not forbidden:
        return posterior
    edge_marginals = posterior.edge_marginals.clone()
    edge_marginals[forbidden, :] = 0.0
    graphs = []
    for graph in posterior.graphs:
        adjacency = graph.adjacency.clone()
        edge_probs = graph.edge_probs.clone()
        hard_mask = graph.hard_mask.clone()
        adjacency[forbidden, :] = 0.0
        edge_probs[forbidden, :] = 0.0
        hard_mask[forbidden, :] = False
        graphs.append(
            GraphSpec(
                node_names=list(graph.node_names),
                adjacency=adjacency,
                hard_mask=hard_mask,
                edge_probs=edge_probs,
                node_groups=dict(graph.node_groups),
                graph_type=graph.graph_type,
                version=graph.version,
            )
        )
    diagnostics = dict(posterior.diagnostics)
    diagnostics["forbidden_id_parent_nodes"] = sorted(name for name in ID_PARENT_NAMES if name in node_index)
    return GraphPosterior(graphs=graphs, log_weights=posterior.log_weights, edge_marginals=edge_marginals, diagnostics=diagnostics)


def _fit_ridge_baselines(
    source: CityTensorData,
    target: CityTensorData,
    ridge: float,
    chunk_size: int,
    *,
    return_models: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], RidgeModelBundle]:
    h2o_dim = h2o_features(np.zeros((1, len(OBS_NAMES) + 1), dtype=np.float32)).shape[1]
    h2o_stats = RidgeStats.zeros(h2o_dim, len(OUTPUT_NAMES))
    cfcmt_stats = {
        name: RidgeStats.zeros(cfcmt_features(name, np.zeros((1, len(OBS_NAMES) + 1), dtype=np.float32)).shape[1], 1)
        for name in OUTPUT_NAMES
    }
    residual = source.real_y.astype(np.float64) - source.sim_y.astype(np.float64)
    obs_action = source.obs_action()
    for start in range(0, source.n, chunk_size):
        end = min(source.n, start + chunk_size)
        h2o_stats.add(h2o_features(obs_action[start:end]), residual[start:end])
        for idx, name in enumerate(OUTPUT_NAMES):
            cfcmt_stats[name].add(cfcmt_features(name, obs_action[start:end]), residual[start:end, idx])
    h2o_beta = h2o_stats.solve(ridge)
    cfcmt_beta = {name: stats.solve(ridge).reshape(-1) for name, stats in cfcmt_stats.items()}

    sse = {
        "uncalibrated": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
        "h2oplus_dense_ridge": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
        "cfcmt_ridge_template": np.zeros(len(OUTPUT_NAMES), dtype=np.float64),
    }
    target_obs_action = target.obs_action()
    for start in range(0, target.n, chunk_size):
        end = min(target.n, start + chunk_size)
        real = target.real_y[start:end].astype(np.float64)
        sim = target.sim_y[start:end].astype(np.float64)
        h2o_pred = sim + h2o_features(target_obs_action[start:end]) @ h2o_beta
        cfcmt_pred = sim.copy()
        for idx, name in enumerate(OUTPUT_NAMES):
            cfcmt_pred[:, idx] += cfcmt_features(name, target_obs_action[start:end]) @ cfcmt_beta[name]
        sse["uncalibrated"] += np.square(sim - real).sum(axis=0)
        sse["h2oplus_dense_ridge"] += np.square(h2o_pred - real).sum(axis=0)
        sse["cfcmt_ridge_template"] += np.square(cfcmt_pred - real).sum(axis=0)
    metrics = {name: _metrics_from_sse(value, target.n) for name, value in sse.items()}
    if return_models:
        return metrics, RidgeModelBundle(h2o_beta=h2o_beta, cfcmt_beta=cfcmt_beta)
    return metrics


def _predict_cfcmt_y(model: RidgeModelBundle, obs_action: np.ndarray, sim_y: np.ndarray) -> np.ndarray:
    pred = sim_y.astype(np.float64, copy=True)
    for idx, name in enumerate(OUTPUT_NAMES):
        pred[:, idx] += cfcmt_features(name, obs_action) @ model.cfcmt_beta[name]
    return pred


def _fit_dense_residual_model(source: CityTensorData, *, ridge: float) -> np.ndarray:
    features = h2o_features(source.obs_action())
    stats = RidgeStats.zeros(features.shape[1], len(OUTPUT_NAMES))
    stats.add(features, source.real_y.astype(np.float64) - source.sim_y.astype(np.float64))
    return stats.solve(ridge)


def _fit_dense_residual_ensemble(
    source_by_city: dict[str, CityTensorData],
    *,
    ridge: float,
    weights: dict[str, float],
) -> DenseResidualEnsembleBundle:
    models = {key: _fit_dense_residual_model(data, ridge=ridge) for key, data in source_by_city.items()}
    total = sum(max(float(weights.get(key, 0.0)), 0.0) for key in models)
    if total <= 1e-12:
        normalized = {key: 1.0 / float(max(1, len(models))) for key in models}
    else:
        normalized = {key: max(float(weights.get(key, 0.0)), 0.0) / total for key in models}
    return DenseResidualEnsembleBundle(models=models, weights=normalized)


def _predict_dense_residual_ensemble_y(model: DenseResidualEnsembleBundle, data: CityTensorData) -> np.ndarray:
    features = h2o_features(data.obs_action())
    residual = np.zeros((data.n, len(OUTPUT_NAMES)), dtype=np.float64)
    for key, beta in model.models.items():
        residual += float(model.weights.get(key, 0.0)) * (features @ beta)
    return data.sim_y.astype(np.float64) + residual


def _evaluate_dense_residual_ensemble(
    model: DenseResidualEnsembleBundle,
    target: CityTensorData,
) -> dict[str, Any]:
    return _evaluate_y_prediction(_predict_dense_residual_ensemble_y(model, target), target.real_y)


def _local_static_scaled_np(data: CityTensorData, scaler: Scaler) -> np.ndarray:
    local = _static_local_features_np(data.observations, data.actions, speed_edges=scaler.local_speed_edges).astype(np.float64)
    mean = scaler.local_mean.detach().cpu().numpy().astype(np.float64)
    std = scaler.local_std.detach().cpu().numpy().astype(np.float64)
    return np.nan_to_num((local - mean) / np.maximum(std, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)


def _rigid_first_delta_features(
    output_name: str,
    data: CityTensorData,
    scaler: Scaler,
    mode: str,
) -> np.ndarray:
    obs_action = data.obs_action()
    pieces: list[np.ndarray] = []
    if mode in {"dense_delta", "dense_static_local_delta"}:
        pieces.append(h2o_features(obs_action))
    if mode in {"static_local_delta", "dense_static_local_delta"}:
        local = _local_static_scaled_np(data, scaler)
        action = (data.actions.astype(np.float64) / 60.0).reshape(-1, 1)
        mechanism = cfcmt_features(output_name, obs_action)
        pieces.append(np.concatenate([mechanism[:, :1], local, local * action], axis=1))
    if not pieces:
        raise ValueError(f"unknown rigid-first delta mode={mode!r}")
    return np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]


def _fit_rigid_first_delta_model(
    source: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    mode: str,
    ridge: float,
) -> RigidFirstDeltaBundle:
    base_y = _predict_cfcmt_y(ridge_model, source.obs_action(), source.sim_y)
    residual = source.real_y.astype(np.float64) - base_y
    stats = {
        name: RidgeStats.zeros(_rigid_first_delta_features(name, source.take(np.arange(min(1, source.n))), scaler, mode).shape[1], 1)
        for name in OUTPUT_NAMES
    }
    for idx, name in enumerate(OUTPUT_NAMES):
        stats[name].add(_rigid_first_delta_features(name, source, scaler, mode), residual[:, idx])
    return RigidFirstDeltaBundle(
        mode=mode,
        beta={name: item.solve(ridge).reshape(-1) for name, item in stats.items()},
    )


def _fit_rigid_first_delta_ensemble(
    source_by_city: dict[str, CityTensorData],
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    mode: str,
    ridge: float,
    weights: dict[str, float],
) -> RigidFirstDeltaEnsembleBundle:
    models = {
        key: _fit_rigid_first_delta_model(data, scaler, ridge_model, mode=mode, ridge=ridge)
        for key, data in source_by_city.items()
    }
    total = sum(max(float(weights.get(key, 0.0)), 0.0) for key in models)
    if total <= 1e-12:
        normalized = {key: 1.0 / float(max(1, len(models))) for key in models}
    else:
        normalized = {key: max(float(weights.get(key, 0.0)), 0.0) / total for key in models}
    return RigidFirstDeltaEnsembleBundle(mode=mode, models=models, weights=normalized)


def _predict_rigid_first_delta_residual(
    model: RigidFirstDeltaBundle,
    data: CityTensorData,
    scaler: Scaler,
) -> np.ndarray:
    residual = np.zeros((data.n, len(OUTPUT_NAMES)), dtype=np.float64)
    for idx, name in enumerate(OUTPUT_NAMES):
        residual[:, idx] = _rigid_first_delta_features(name, data, scaler, model.mode) @ model.beta[name]
    return residual


def _predict_rigid_first_delta_y(
    model: RigidFirstDeltaBundle,
    data: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    alpha: float,
) -> np.ndarray:
    pred = _predict_cfcmt_y(ridge_model, data.obs_action(), data.sim_y)
    if float(alpha) == 0.0:
        return pred
    pred += float(alpha) * _predict_rigid_first_delta_residual(model, data, scaler)
    return pred


def _predict_rigid_first_ensemble_delta_y(
    model: RigidFirstDeltaEnsembleBundle,
    data: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    alpha: float,
) -> np.ndarray:
    pred = _predict_cfcmt_y(ridge_model, data.obs_action(), data.sim_y)
    if float(alpha) == 0.0:
        return pred
    residual = np.zeros_like(pred, dtype=np.float64)
    for key, item in model.models.items():
        residual += float(model.weights.get(key, 0.0)) * _predict_rigid_first_delta_residual(item, data, scaler)
    pred += float(alpha) * residual
    return pred


def _evaluate_y_prediction(pred_y: np.ndarray, real_y: np.ndarray) -> dict[str, Any]:
    sse = np.square(pred_y.astype(np.float64) - real_y.astype(np.float64)).sum(axis=0)
    return _metrics_from_sse(sse, int(real_y.shape[0]))


def _mechanism_output_indices_from_name(mechanism: str) -> list[int]:
    return [OUTPUT_NAMES.index(name) for name in RIGID_FIRST_MECHANISM_OUTPUT_NAMES[mechanism] if name in OUTPUT_NAMES]


def _mechanism_mse_table(metrics: dict[str, Any]) -> dict[str, float]:
    per_output = metrics.get("per_output_mse", {})
    out = {}
    for mechanism, output_names in RIGID_FIRST_MECHANISM_OUTPUT_NAMES.items():
        values = [float(per_output[name]) for name in output_names if name in per_output]
        out[mechanism] = float(np.mean(values)) if values else float("nan")
    return out


def _evaluate_rigid_first_delta(
    model: RigidFirstDeltaBundle,
    target: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    alpha: float,
) -> dict[str, Any]:
    return _evaluate_y_prediction(
        _predict_rigid_first_delta_y(model, target, scaler, ridge_model, alpha=alpha),
        target.real_y,
    )


def _evaluate_rigid_first_ensemble_delta(
    model: RigidFirstDeltaEnsembleBundle,
    target: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    *,
    alpha: float,
) -> dict[str, Any]:
    return _evaluate_y_prediction(
        _predict_rigid_first_ensemble_delta_y(model, target, scaler, ridge_model, alpha=alpha),
        target.real_y,
    )


def _predict_rigid_first_mechanism_delta_y(
    models: dict[str, RigidFirstDeltaBundle],
    data: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    selector: dict[str, Any],
) -> np.ndarray:
    pred = _predict_cfcmt_y(ridge_model, data.obs_action(), data.sim_y)
    selected = selector.get("selected_by_mechanism", {})
    for mechanism, choice in selected.items():
        candidate = str(choice.get("selected_candidate", "rigid_cfcmt"))
        if candidate == "rigid_cfcmt":
            continue
        mode, alpha = _parse_rigid_first_candidate_name(candidate)
        if mode not in models or float(alpha) == 0.0:
            continue
        model = models[mode]
        for idx in _mechanism_output_indices_from_name(mechanism):
            name = OUTPUT_NAMES[idx]
            pred[:, idx] += float(alpha) * (_rigid_first_delta_features(name, data, scaler, model.mode) @ model.beta[name])
    return pred


def _evaluate_rigid_first_mechanism_delta(
    models: dict[str, RigidFirstDeltaBundle],
    target: CityTensorData,
    scaler: Scaler,
    ridge_model: RidgeModelBundle,
    selector: dict[str, Any],
) -> dict[str, Any]:
    return _evaluate_y_prediction(
        _predict_rigid_first_mechanism_delta_y(models, target, scaler, ridge_model, selector),
        target.real_y,
    )


def _select_rigid_first_delta_candidate(
    folds: list[dict[str, Any]],
    *,
    min_win_rate: float,
    min_improvement: float,
    min_unweighted_win_rate: float = 1.0,
    weighting: str = "uniform",
    ensemble_min_win_rate: float | None = None,
    ensemble_min_unweighted_win_rate: float | None = None,
    ensemble_min_max_source_weight: float = 1.0,
    ensemble_max_source_weight: float = 0.0,
) -> dict[str, Any]:
    candidate_names = sorted(
        {
            name
            for fold in folds
            for name in fold.get("rigid_first_candidates", {})
            if name != "rigid_cfcmt"
        }
    )
    summaries = {}
    stable: list[tuple[str, float]] = []
    for name in candidate_names:
        ratio_weight_pairs = [
            (
                fold["rigid_first_candidates"][name].get("vs_rigid_cfcmt"),
                float(fold.get("target_similarity_weight", 1.0)) if weighting == "similarity" else 1.0,
            )
            for fold in folds
            if name in fold.get("rigid_first_candidates", {})
            and fold["rigid_first_candidates"][name].get("vs_rigid_cfcmt") is not None
        ]
        ratios = [float(ratio) for ratio, _weight in ratio_weight_pairs]
        if not ratios:
            continue
        weights = np.asarray([max(float(weight), 0.0) for _ratio, weight in ratio_weight_pairs], dtype=np.float64)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(len(ratios), dtype=np.float64)
        ratio_arr = np.asarray(ratios, dtype=np.float64)
        wins = ratio_arr < 1.0 - float(min_improvement)
        mean_ratio = float(np.average(ratio_arr, weights=weights))
        win_rate = float(np.average(wins.astype(np.float64), weights=weights))
        is_ensemble = name.startswith("ensemble_")
        candidate_min_win_rate = float(ensemble_min_win_rate) if is_ensemble and ensemble_min_win_rate is not None else float(min_win_rate)
        candidate_min_unweighted_win_rate = (
            float(ensemble_min_unweighted_win_rate)
            if is_ensemble and ensemble_min_unweighted_win_rate is not None
            else float(min_unweighted_win_rate)
        )
        source_weight_ok = (not is_ensemble) or float(ensemble_max_source_weight) >= float(ensemble_min_max_source_weight)
        summary = {
            "mean_vs_rigid_cfcmt": mean_ratio,
            "win_rate_vs_rigid_cfcmt": win_rate,
            "unweighted_mean_vs_rigid_cfcmt": float(np.mean(ratio_arr)),
            "unweighted_win_rate_vs_rigid_cfcmt": float(np.mean(wins.astype(np.float64))),
            "weighting": weighting,
            "folds": len(ratios),
            "ensemble": bool(is_ensemble),
            "candidate_min_win_rate": candidate_min_win_rate,
            "candidate_min_unweighted_win_rate": candidate_min_unweighted_win_rate,
            "ensemble_max_source_weight": float(ensemble_max_source_weight) if is_ensemble else None,
            "ensemble_min_max_source_weight": float(ensemble_min_max_source_weight) if is_ensemble else None,
            "source_weight_ok": bool(source_weight_ok),
            "stable": bool(
                mean_ratio < 1.0 - float(min_improvement)
                and win_rate >= candidate_min_win_rate
                and float(np.mean(wins.astype(np.float64))) >= candidate_min_unweighted_win_rate
                and source_weight_ok
            ),
        }
        summaries[name] = summary
        if summary["stable"]:
            stable.append((name, mean_ratio))
    selected = "rigid_cfcmt" if not stable else min(stable, key=lambda item: item[1])[0]
    selected_mode, selected_alpha = _parse_rigid_first_candidate_name(selected)
    return {
        "selected_candidate": selected,
        "selected_mode": selected_mode,
        "selected_alpha": selected_alpha,
        "candidate_summaries": summaries,
        "min_win_rate": float(min_win_rate),
        "min_improvement": float(min_improvement),
        "min_unweighted_win_rate": float(min_unweighted_win_rate),
        "weighting": weighting,
        "ensemble_min_win_rate": None if ensemble_min_win_rate is None else float(ensemble_min_win_rate),
        "ensemble_min_unweighted_win_rate": (
            None if ensemble_min_unweighted_win_rate is None else float(ensemble_min_unweighted_win_rate)
        ),
        "ensemble_min_max_source_weight": float(ensemble_min_max_source_weight),
        "ensemble_max_source_weight": float(ensemble_max_source_weight),
    }


def _select_rigid_first_mechanism_delta_candidate(
    folds: list[dict[str, Any]],
    *,
    min_win_rate: float,
    min_improvement: float,
    min_unweighted_win_rate: float,
    weighting: str,
) -> dict[str, Any]:
    selected_by_mechanism: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    candidate_names = sorted(
        {
            name
            for fold in folds
            for name in fold.get("rigid_first_candidates", {})
            if name != "rigid_cfcmt"
            and fold["rigid_first_candidates"][name].get("mechanism_selectable", True)
        }
    )
    for mechanism in RIGID_FIRST_MECHANISM_OUTPUT_NAMES:
        mechanism_summaries = {}
        stable: list[tuple[str, float]] = []
        for name in candidate_names:
            ratio_weight_pairs = [
                (
                    fold["rigid_first_candidates"][name].get("vs_rigid_cfcmt_by_mechanism", {}).get(mechanism),
                    float(fold.get("target_similarity_weight", 1.0)) if weighting == "similarity" else 1.0,
                )
                for fold in folds
                if name in fold.get("rigid_first_candidates", {})
                and fold["rigid_first_candidates"][name].get("vs_rigid_cfcmt_by_mechanism", {}).get(mechanism) is not None
            ]
            ratios = [float(ratio) for ratio, _weight in ratio_weight_pairs if ratio is not None and math.isfinite(float(ratio))]
            if not ratios:
                continue
            weights = np.asarray([max(float(weight), 0.0) for ratio, weight in ratio_weight_pairs if ratio is not None and math.isfinite(float(ratio))], dtype=np.float64)
            if float(weights.sum()) <= 1e-12:
                weights = np.ones(len(ratios), dtype=np.float64)
            ratio_arr = np.asarray(ratios, dtype=np.float64)
            wins = ratio_arr < 1.0 - float(min_improvement)
            mean_ratio = float(np.average(ratio_arr, weights=weights))
            win_rate = float(np.average(wins.astype(np.float64), weights=weights))
            unweighted_win_rate = float(np.mean(wins.astype(np.float64)))
            unweighted_mean = float(np.mean(ratio_arr))
            stable_candidate = bool(
                mean_ratio < 1.0 - float(min_improvement)
                and win_rate >= float(min_win_rate)
                and unweighted_win_rate >= float(min_unweighted_win_rate)
            )
            mechanism_summaries[name] = {
                "mean_vs_rigid_cfcmt": mean_ratio,
                "win_rate_vs_rigid_cfcmt": win_rate,
                "unweighted_mean_vs_rigid_cfcmt": unweighted_mean,
                "unweighted_win_rate_vs_rigid_cfcmt": unweighted_win_rate,
                "weighting": weighting,
                "folds": len(ratios),
                "stable": stable_candidate,
            }
            if stable_candidate:
                stable.append((name, mean_ratio))
        selected = "rigid_cfcmt" if not stable else min(stable, key=lambda item: item[1])[0]
        selected_mode, selected_alpha = _parse_rigid_first_candidate_name(selected)
        selected_by_mechanism[mechanism] = {
            "selected_candidate": selected,
            "selected_mode": selected_mode,
            "selected_alpha": selected_alpha,
        }
        summaries[mechanism] = mechanism_summaries
    return {
        "selected_by_mechanism": selected_by_mechanism,
        "candidate_summaries_by_mechanism": summaries,
        "min_win_rate": float(min_win_rate),
        "min_improvement": float(min_improvement),
        "min_unweighted_win_rate": float(min_unweighted_win_rate),
        "weighting": weighting,
    }


def _rigid_first_candidate_name(mode: str, alpha: float) -> str:
    return f"{mode}_alpha_{float(alpha):.2f}".replace(".", "p")


def _rigid_first_ensemble_candidate_name(mode: str, alpha: float) -> str:
    return f"ensemble_{_rigid_first_candidate_name(mode, alpha)}"


def _parse_rigid_first_candidate_name(name: str) -> tuple[str | None, float]:
    if name == "rigid_cfcmt":
        return None, 0.0
    marker = "_alpha_"
    if marker not in name:
        return name, 1.0
    mode, alpha_text = name.rsplit(marker, 1)
    return mode, float(alpha_text.replace("p", "."))


def _metrics_from_sse(sse: np.ndarray, n_rows: int) -> dict[str, Any]:
    per_output = sse / float(max(1, n_rows))
    return {
        "total_mse": float(per_output.mean()),
        "dynamic_mse": float(per_output[:-1].mean()),
        "reward_mse": float(per_output[-1]),
        "per_output_mse": {name: float(value) for name, value in zip(OUTPUT_NAMES, per_output)},
    }


def _y_from_prediction(pred: dict[str, Any], scaler: Scaler) -> np.ndarray:
    next_obs = pred["next_observations"] * scaler.obs_std + scaler.obs_mean
    reward = pred["rewards"].reshape(-1, 1) * scaler.reward_std + scaler.reward_mean
    y_cols = [next_obs[:, idx : idx + 1] for idx in DYNAMIC_OUTPUT_INDICES] + [reward]
    return torch.cat(y_cols, dim=1).detach().cpu().numpy()


def _theta_current_features(
    batch: TransitionBatch,
    local_encoder: LocalGraphEncoder | None,
    local_dim: int,
    local_mode: str,
) -> torch.Tensor:
    features = torch.cat([batch.observations, batch.actions], dim=1)
    if local_mode == LOCAL_MODE_STATIC:
        local = batch.metadata.get("local_static_features")
        if not torch.is_tensor(local):
            raise ValueError("deterministic_static local mode requires local_static_features metadata")
        return torch.cat([features, local], dim=1)
    if local_encoder is None:
        return features
    raw = _local_proxy_features(batch, local_encoder.input_dim)
    local = local_encoder(raw)
    return torch.cat([features, local[:, :local_dim]], dim=1)


def _local_proxy_features(batch: TransitionBatch, feature_dim: int) -> torch.Tensor:
    base = torch.cat([batch.observations, batch.actions], dim=1)
    if base.shape[1] >= feature_dim:
        return base[:, :feature_dim]
    pad = base.new_zeros(base.shape[0], feature_dim - base.shape[1])
    return torch.cat([base, pad], dim=1)


def _infer_theta(
    batch: TransitionBatch,
    encoder: TimeVaryingFactorEncoder | None,
    *,
    local_encoder: LocalGraphEncoder | None = None,
    local_dim: int = 0,
    local_mode: str = LOCAL_MODE_NONE,
    history_len: int = 1,
) -> dict[str, torch.Tensor] | None:
    if encoder is None:
        return None
    current = _theta_current_features(batch, local_encoder, local_dim, local_mode)
    if local_mode == LOCAL_MODE_PROXY_ENCODER or history_len <= 1:
        return encoder(current.unsqueeze(1), current.new_ones(current.shape[0], 1))
    windows, masks = build_history_windows(current, history_len=history_len)
    return encoder(windows, masks)


def _train_residual_with_encoder(
    model: CausalFactoredResidualWorldModel,
    real_batch: TransitionBatch,
    encoder: TimeVaryingFactorEncoder,
    *,
    local_encoder: LocalGraphEncoder | None,
    local_dim: int,
    local_mode: str,
    history_len: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict[str, Any]:
    params = list(model.residual_modules.parameters()) + list(encoder.parameters())
    if local_encoder is not None:
        params += list(local_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=float(model.config.get("weight_decay", 0.0)))
    sim_next = real_batch.metadata["sim_next_observations"]
    sim_rewards = real_batch.metadata["sim_rewards"]
    indices = torch.arange(real_batch.batch_size, device=real_batch.device)
    epoch_losses: list[float] = []
    for _epoch in range(max(1, int(epochs))):
        perm = indices[torch.randperm(indices.numel(), device=indices.device)]
        running = []
        for start in range(0, real_batch.batch_size, max(1, int(batch_size))):
            idx = perm[start : start + max(1, int(batch_size))]
            sub = _take_batch(real_batch, idx)
            sub.metadata["sim_next_observations"] = sim_next[idx]
            sub.metadata["sim_rewards"] = sim_rewards[idx]
            theta = _infer_theta(
                sub,
                encoder,
                local_encoder=local_encoder,
                local_dim=local_dim,
                local_mode=local_mode,
                history_len=history_len,
            )
            node_values = _batch_node_values(sub)
            override_values = _batch_node_values(
                sub,
                override_next_observations=sub.metadata["sim_next_observations"],
                override_rewards=sub.metadata["sim_rewards"],
            )
            optimizer.zero_grad()
            losses = []
            for spec in model.mechanism_specs:
                parents, mask = model.parent_inputs(sub, spec.name, node_values=node_values)
                pred = model.residual_modules[spec.name](parents, theta=theta[spec.name], mask=mask)
                target = _target_for_spec(spec, node_values) - _target_for_spec(spec, override_values)
                losses.append(model.residual_modules[spec.name].loss(pred, target, spec.loss_type))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(model.config.get("grad_clip_norm", 10.0)))
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(running) / max(1, len(running)))
    return {
        "residual_trained": True,
        "mode": "paired_joint_factor_encoder",
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "epochs": max(1, int(epochs)),
    }


def _take_batch(batch: TransitionBatch, idx: torch.Tensor) -> TransitionBatch:
    metadata = {
        key: (value[idx] if torch.is_tensor(value) and value.shape[:1] == (batch.batch_size,) else value)
        for key, value in batch.metadata.items()
    }
    return TransitionBatch(
        observations=batch.observations[idx],
        actions=batch.actions[idx],
        rewards=batch.rewards[idx],
        next_observations=batch.next_observations[idx],
        dones=batch.dones[idx],
        metadata=metadata,
    )


def _train_neural_stage(
    source: CityTensorData,
    target: CityTensorData,
    *,
    scaler: Scaler,
    device: torch.device,
    posterior: GraphPosterior,
    specs: list[MechanismSpec],
    args: argparse.Namespace,
    use_latent: bool,
    local_mode: str,
    use_aux_features: bool,
    ridge_model: RidgeModelBundle | None = None,
    anchor_mode: str = ANCHOR_MODE_SIM,
) -> dict[str, Any]:
    source_real = _to_batch(source, scaler, device, use_aux_features)
    source_sim = _sim_batch(source, scaler, device, use_aux_features)
    target_real = _to_batch(target, scaler, device, use_aux_features)
    if anchor_mode == ANCHOR_MODE_RIGID_CFCMT:
        if ridge_model is None:
            raise ValueError("rigid_cfcmt anchor requires ridge_model")
        _install_cfcmt_base(source_real, source, scaler, device, use_aux_features, ridge_model)
        _install_cfcmt_base(target_real, target, scaler, device, use_aux_features, ridge_model)
    elif anchor_mode != ANCHOR_MODE_SIM:
        raise ValueError(f"unknown anchor_mode={anchor_mode!r}")
    model = CausalFactoredResidualWorldModel(
        specs,
        posterior,
        {
            "hidden_dim": args.hidden_dim,
            "batch_size": args.neural_batch_size,
            "train_epochs_sim": args.sim_epochs,
            "train_epochs_residual": args.residual_epochs,
            "lr": args.lr,
            "residual_lr": args.residual_lr,
            "weight_decay": args.weight_decay,
            "parent_threshold": args.parent_threshold,
            "trust": {
                "residual_scale": args.trust_residual_scale,
                "uncertainty_scale": args.trust_uncertainty_scale,
                "graph_uncertainty_scale": args.trust_graph_uncertainty_scale,
                "w_max": 1.0,
            },
        },
    ).to(device)
    sim_metrics = model.fit_sim_modules(source_sim)

    encoder = None
    local_encoder = None
    local_dim = int(args.local_embedding_dim)
    if use_latent:
        input_dim = int(source_real.observations.shape[1]) + int(source_real.actions.shape[1])
        if local_mode == LOCAL_MODE_PROXY_ENCODER:
            local_encoder = LocalGraphEncoder(hidden_dim=max(16, args.hidden_dim), out_dim=local_dim).to(device)
            input_dim += local_dim
        elif local_mode == LOCAL_MODE_STATIC:
            input_dim += int(source_real.metadata["local_static_features"].shape[1])
        elif local_mode != LOCAL_MODE_NONE:
            raise ValueError(f"unknown local_mode={local_mode!r}")
        encoder = TimeVaryingFactorEncoder.from_mechanism_specs(input_dim, specs, hidden_dim=args.hidden_dim).to(device)
        residual_metrics = _train_residual_with_encoder(
            model,
            source_real,
            encoder,
            local_encoder=local_encoder,
            local_dim=local_dim,
            local_mode=local_mode,
            history_len=args.history_len,
            epochs=args.residual_epochs,
            lr=args.residual_lr,
            batch_size=args.neural_batch_size,
        )
    else:
        residual_metrics = model.fit_residual_modules(source_real)

    eval_metrics = _evaluate_world_model(
        model,
        target_real,
        target.real_y,
        scaler,
        args.eval_batch_size,
        encoder=encoder,
        local_encoder=local_encoder,
        local_dim=local_dim,
        local_mode=local_mode,
        history_len=args.history_len,
        trust=False,
    )
    result = {
        "metrics": eval_metrics,
        "sim_training": _compact_training_metrics(sim_metrics),
        "residual_training": _compact_training_metrics(residual_metrics),
        "local_mode": local_mode,
        "anchor_mode": anchor_mode,
        "local_feature_names": list(LOCAL_STATIC_FEATURE_NAMES) if local_mode == LOCAL_MODE_STATIC else [],
        "num_parameters": int(sum(param.numel() for param in model.parameters()) + (sum(param.numel() for param in encoder.parameters()) if encoder else 0) + (sum(param.numel() for param in local_encoder.parameters()) if local_encoder else 0)),
    }
    if use_latent:
        trust_result = _evaluate_world_model(
            model,
            target_real,
            target.real_y,
            scaler,
            args.eval_batch_size,
            encoder=encoder,
            local_encoder=local_encoder,
            local_dim=local_dim,
            local_mode=local_mode,
            history_len=args.history_len,
            trust=True,
        )
        result["trust_gated_metrics"] = trust_result
        if ridge_model is not None:
            fallback_result = _evaluate_world_model_with_cfcmt_fallback(
                model,
                target_real,
                target,
                scaler,
                args.eval_batch_size,
                ridge_model,
                encoder=encoder,
                local_encoder=local_encoder,
                local_dim=local_dim,
                local_mode=local_mode,
                history_len=args.history_len,
            )
            result["cfcmt_fallback_trust_metrics"] = fallback_result
        result["bridge_diagnostics"] = _bridge_diagnostics(
            model,
            target_real,
            posterior,
            encoder,
            local_encoder=local_encoder,
            local_dim=local_dim,
            local_mode=local_mode,
            history_len=args.history_len,
            batch_size=min(args.eval_batch_size, target_real.batch_size),
        )
    return result


def _compact_training_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (float(value) if isinstance(value, (int, float)) else value)
        for key, value in metrics.items()
        if key not in {"loss_history"}
    }


def _evaluate_world_model(
    model: CausalFactoredResidualWorldModel,
    target_batch: TransitionBatch,
    target_real_y: np.ndarray,
    scaler: Scaler,
    batch_size: int,
    *,
    encoder: TimeVaryingFactorEncoder | None,
    local_encoder: LocalGraphEncoder | None,
    local_dim: int,
    local_mode: str,
    history_len: int,
    trust: bool,
) -> dict[str, Any]:
    sse = np.zeros(len(OUTPUT_NAMES), dtype=np.float64)
    trust_estimator = FactorTrustEstimator([spec.name for spec in model.mechanism_specs], _trust_config(model.config, w_min=0.0)) if trust else None
    model.eval()
    if encoder is not None:
        encoder.eval()
    if local_encoder is not None:
        local_encoder.eval()
    with torch.no_grad():
        for start in range(0, target_batch.batch_size, max(1, int(batch_size))):
            end = min(target_batch.batch_size, start + max(1, int(batch_size)))
            sub = _slice_batch(target_batch, start, end)
            theta = _infer_theta(
                sub,
                encoder,
                local_encoder=local_encoder,
                local_dim=local_dim,
                local_mode=local_mode,
                history_len=history_len,
            )
            pred = _predict_sim_anchored(model, sub, theta, trust_estimator=trust_estimator if trust else None)
            y_pred = _y_from_prediction(pred, scaler)
            y_true = target_real_y[start:end].astype(np.float64)
            sse += np.square(y_pred.astype(np.float64) - y_true).sum(axis=0)
    return _metrics_from_sse(sse, target_batch.batch_size)


def _evaluate_world_model_with_cfcmt_fallback(
    model: CausalFactoredResidualWorldModel,
    target_batch: TransitionBatch,
    target_data: CityTensorData,
    scaler: Scaler,
    batch_size: int,
    ridge_model: RidgeModelBundle,
    *,
    encoder: TimeVaryingFactorEncoder | None,
    local_encoder: LocalGraphEncoder | None,
    local_dim: int,
    local_mode: str,
    history_len: int,
) -> dict[str, Any]:
    sse = np.zeros(len(OUTPUT_NAMES), dtype=np.float64)
    trust_estimator = FactorTrustEstimator([spec.name for spec in model.mechanism_specs], _trust_config(model.config, w_min=0.0))
    model.eval()
    if encoder is not None:
        encoder.eval()
    if local_encoder is not None:
        local_encoder.eval()
    with torch.no_grad():
        for start in range(0, target_batch.batch_size, max(1, int(batch_size))):
            end = min(target_batch.batch_size, start + max(1, int(batch_size)))
            sub = _slice_batch(target_batch, start, end)
            theta = _infer_theta(
                sub,
                encoder,
                local_encoder=local_encoder,
                local_dim=local_dim,
                local_mode=local_mode,
                history_len=history_len,
            )
            pred = _predict_sim_anchored(model, sub, theta, trust_estimator=None)
            trust = trust_estimator(pred["mechanism_outputs"], mechanism_uncertainty=pred.get("mechanism_uncertainty"))
            neural_y = _y_from_prediction(pred, scaler).astype(np.float64)
            cfcmt_y = _predict_cfcmt_y(
                ridge_model,
                target_data.obs_action()[start:end],
                target_data.sim_y[start:end],
            )
            blended = cfcmt_y.copy()
            for spec in model.mechanism_specs:
                gate = trust[spec.name].detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
                output_indices = _mechanism_output_indices(spec)
                if not output_indices:
                    continue
                blended[:, output_indices] = (
                    gate * neural_y[:, output_indices]
                    + (1.0 - gate) * cfcmt_y[:, output_indices]
                )
            y_true = target_data.real_y[start:end].astype(np.float64)
            sse += np.square(blended - y_true).sum(axis=0)
    return _metrics_from_sse(sse, target_batch.batch_size)


def _mechanism_output_indices(spec: MechanismSpec) -> list[int]:
    indices = []
    for child in spec.child_names:
        if child == "reward@t1":
            indices.append(len(OUTPUT_NAMES) - 1)
            continue
        obs_name = child[:-3] if child.endswith("@t1") else child
        out_name = f"next_{obs_name}"
        if out_name in OUTPUT_NAMES:
            indices.append(OUTPUT_NAMES.index(out_name))
    return indices


def _predict_sim_anchored(
    model: CausalFactoredResidualWorldModel,
    batch: TransitionBatch,
    theta: dict[str, torch.Tensor] | None,
    *,
    trust_estimator: FactorTrustEstimator | None = None,
) -> dict[str, Any]:
    sim_next = batch.metadata["sim_next_observations"]
    sim_rewards = batch.metadata["sim_rewards"]
    next_obs = sim_next.clone()
    reward = sim_rewards.reshape(-1).clone()
    obs_names = list(batch.metadata.get("obs_names") or OBS_NAMES)
    node_values = _batch_node_values(batch)
    sim_values = _batch_node_values(batch, override_next_observations=sim_next, override_rewards=sim_rewards)
    mechanism_outputs: dict[str, dict[str, torch.Tensor]] = {}
    mechanism_uncertainty: dict[str, torch.Tensor] = {}
    for spec in model.mechanism_specs:
        parents, mask = model.parent_inputs(batch, spec.name, node_values=node_values)
        local_theta = model._theta_for(theta, spec.name, batch.batch_size, parents.device, parents.dtype)
        residual_pred = model.residual_modules[spec.name](parents, theta=local_theta, mask=mask)
        base = _target_for_spec(spec, sim_values)
        residual = residual_pred["mean"]
        mean = base + residual
        mechanism_outputs[spec.name] = {
            "base": base,
            "residual": residual,
            "mean": mean,
            "parent_mask": mask.detach(),
            "parent_names": list(spec.parent_names),
            "child_names": list(spec.child_names),
        }
        mechanism_uncertainty[spec.name] = residual_pred["uncertainty"]

    if trust_estimator is not None:
        trust = trust_estimator(mechanism_outputs, mechanism_uncertainty=mechanism_uncertainty)
    else:
        trust = {}

    for spec in model.mechanism_specs:
        output = mechanism_outputs[spec.name]
        if spec.name in trust:
            gate = trust[spec.name].reshape(-1, 1).to(device=output["mean"].device, dtype=output["mean"].dtype)
            mean = output["base"] + gate * output["residual"]
            output["mean"] = mean
        else:
            mean = output["mean"]
        for offset, child in enumerate(spec.child_names):
            value = mean[:, offset]
            if child == "reward@t1":
                reward = value
            else:
                obs_name = child[:-3]
                if obs_name in obs_names:
                    next_obs[:, obs_names.index(obs_name)] = value
    return {
        "next_observations": next_obs,
        "rewards": reward,
        "reward": reward,
        "mechanism_outputs": mechanism_outputs,
        "mechanism_uncertainty": mechanism_uncertainty,
    }


def _bridge_diagnostics(
    model: CausalFactoredResidualWorldModel,
    batch: TransitionBatch,
    posterior: GraphPosterior,
    encoder: TimeVaryingFactorEncoder,
    *,
    local_encoder: LocalGraphEncoder | None,
    local_dim: int,
    local_mode: str,
    history_len: int,
    batch_size: int,
) -> dict[str, Any]:
    sub = _slice_batch(batch, 0, batch_size)
    trust_estimator = FactorTrustEstimator([spec.name for spec in model.mechanism_specs], _trust_config(model.config, w_min=0.05))

    def theta_provider(item: TransitionBatch) -> dict[str, torch.Tensor]:
        theta = _infer_theta(
            item,
            encoder,
            local_encoder=local_encoder,
            local_dim=local_dim,
            local_mode=local_mode,
            history_len=history_len,
        )
        return theta or {}

    anchored_model = SimAnchoredWorldModelView(model)
    provider = FactorWiseTrustWeightProvider(
        anchored_model,
        trust_estimator,
        WeightComposer(mode="geometric_mean", w_min=0.05, w_max=1.0),
        theta_provider=theta_provider,
        config={"trust_warmup_steps": 0},
    )
    h2o = DummyH2O()
    bridge = H2OFactorTrustBridge(h2o, provider, config={"trust_warmup_steps": 0})
    sim_batch = {
        "observations": sub.observations,
        "actions": sub.actions,
        "rewards": sub.rewards,
        "next_observations": sub.next_observations,
        "dones": sub.dones,
        "obs_names": list(sub.metadata.get("obs_names") or OBS_NAMES),
        "action_names": list(ACTION_NAMES),
        "sim_next_observations": sub.metadata["sim_next_observations"],
        "sim_rewards": sub.metadata["sim_rewards"],
    }
    if torch.is_tensor(sub.metadata.get("local_static_features")):
        sim_batch["local_static_features"] = sub.metadata["local_static_features"]
        sim_batch["local_static_feature_names"] = list(sub.metadata.get("local_static_feature_names") or LOCAL_STATIC_FEATURE_NAMES)
    weights = bridge.compute_sim_weight(sim_batch)
    trainer = CFH2OTrainer(h2o, anchored_model, encoder, posterior, trust_estimator, replay_buffer=None, config={"batch_size": batch_size})
    trainer_weights = trainer.trust_provider.compute_weight(sim_batch, step=0)
    return {
        "bridge_weight_mean": float(weights.mean().detach().cpu()),
        "bridge_weight_min": float(weights.min().detach().cpu()),
        "bridge_weight_max": float(weights.max().detach().cpu()),
        "trainer_provider_weight_mean": float(trainer_weights.mean().detach().cpu()),
        "trainer_graph_entropy": float(trainer._graph_entropy()),
        "h2o_external_provider_installed": bool(h2o.external_sim_weight_provider is not None),
    }


def _trust_config(config: dict[str, Any], *, w_min: float) -> dict[str, float]:
    trust = dict(config.get("trust", {}))
    return {
        "w_min": float(w_min),
        "w_max": float(trust.get("w_max", 1.0)),
        "residual_scale": float(trust.get("residual_scale", 0.0)),
        "uncertainty_scale": float(trust.get("uncertainty_scale", 0.15)),
        "graph_uncertainty_scale": float(trust.get("graph_uncertainty_scale", 0.25)),
        "alignment_scale": float(trust.get("alignment_scale", 0.0)),
        "target_error_scale": float(trust.get("target_error_scale", 0.0)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _ratio_table(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    h2o = methods["h2oplus_dense_ridge"]["total_mse"]
    uncal = methods["uncalibrated"]["total_mse"]
    return {
        name: {
            "total_mse": metrics["total_mse"],
            "vs_h2oplus_dense_ridge": metrics["total_mse"] / h2o if h2o else None,
            "vs_uncalibrated": metrics["total_mse"] / uncal if uncal else None,
        }
        for name, metrics in methods.items()
    }


def _selector_args(args: argparse.Namespace) -> argparse.Namespace:
    values = dict(vars(args))
    values["max_train_rows_per_city"] = int(args.selector_max_train_rows_per_city)
    values["max_eval_rows_per_city"] = int(args.selector_max_eval_rows_per_city)
    values["max_graph_rows"] = int(args.selector_max_graph_rows)
    values["sim_epochs"] = int(args.selector_sim_epochs)
    values["residual_epochs"] = int(args.selector_residual_epochs)
    values["dag_bootstrap_runs"] = int(args.selector_dag_bootstrap_runs)
    values["neural_batch_size"] = min(int(args.neural_batch_size), int(args.selector_batch_size))
    values["eval_batch_size"] = min(int(args.eval_batch_size), int(args.selector_eval_batch_size))
    return argparse.Namespace(**values)


def _fit_source_selector_graph(
    graph_batch: TransitionBatch,
    args: argparse.Namespace,
) -> GraphPosterior:
    discoverer = AutoDAGDiscoverer(
        {
            "bootstrap_runs": args.dag_bootstrap_runs,
            "ridge_l2": args.dag_ridge_l2,
            "score_threshold": args.dag_score_threshold,
            "score_temperature": args.dag_score_temperature,
            "max_parents": args.dag_max_parents,
            "mechanism_parent_threshold": args.parent_threshold,
            "seed": args.seed,
        }
    )
    posterior = discoverer.fit(graph_batch, registry=FeatureRegistry.from_transition_dataset(graph_batch))
    if not bool(args.allow_id_parents):
        posterior = _remove_id_parent_edges(posterior)
    return posterior


def _run_source_leave_one_selector(
    split: dict[str, Any],
    source_keys: list[str],
    city_data: dict[str, CityTensorData],
    args: argparse.Namespace,
    *,
    use_aux_features: bool,
) -> dict[str, Any]:
    selector_args = _selector_args(args)
    device = torch.device(args.device)
    target_key = str(split["target_env"])
    target_full = city_data[target_key]
    candidates = [
        ("no_local", LOCAL_MODE_NONE),
        ("deterministic_static_local", LOCAL_MODE_STATIC),
    ]
    if bool(args.selector_include_proxy):
        candidates.append(("proxy_encoder_local", LOCAL_MODE_PROXY_ENCODER))

    folds = []
    for heldout_key in source_keys:
        train_keys = [key for key in source_keys if key != heldout_key]
        if not train_keys:
            continue
        source_train_by_city = {
            key: _select_city_subset(
                city_data[key],
                selector_args.max_train_rows_per_city,
                selector_args.seed,
                f"{city_data[key].key}:{split['name']}:selector:{heldout_key}:train",
            )
            for key in train_keys
        }
        source_train = _concat_preselected_data(source_train_by_city.values())
        source_eval_full = city_data[heldout_key]
        source_eval = source_eval_full.take(
            _select_rows(
                source_eval_full.n,
                selector_args.max_eval_rows_per_city,
                selector_args.seed,
                f"{split['name']}:selector:{heldout_key}:eval",
            )
        )
        scaler = _fit_scaler(source_train, device, use_aux_features)
        _baseline_metrics, ridge_model = _fit_ridge_baselines(
            source_train,
            source_eval,
            selector_args.ridge,
            selector_args.ridge_chunk_rows,
            return_models=True,
        )
        rigid_total_mse = float(_baseline_metrics["cfcmt_ridge_template"]["total_mse"])
        rigid_mechanism_mse = _mechanism_mse_table(_baseline_metrics["cfcmt_ridge_template"])
        graph_data = None
        posterior = None
        specs = None
        if not bool(args.skip_neural_full):
            graph_data = source_train.take(
                _select_rows(
                    source_train.n,
                    selector_args.max_graph_rows,
                    selector_args.seed,
                    f"{split['name']}:selector:{heldout_key}:graph",
                )
            )
            graph_batch = _to_batch(graph_data, scaler, device, use_aux_features)
            posterior = _fit_source_selector_graph(graph_batch, selector_args)
            specs = _learned_specs(
                posterior,
                latent=True,
                threshold=selector_args.parent_threshold,
                top_k=selector_args.parent_top_k,
                allow_id_parents=bool(selector_args.allow_id_parents),
            )
        candidate_metrics = {}
        candidate_metrics["rigid_cfcmt"] = {
            "total_mse": rigid_total_mse,
            "local_mode": "none",
            "vs_rigid_cfcmt": 1.0,
        }
        rigid_first_candidates = {
            "rigid_cfcmt": {
                "total_mse": rigid_total_mse,
                "mode": None,
                "alpha": 0.0,
                "vs_rigid_cfcmt": 1.0,
                "per_mechanism_mse": rigid_mechanism_mse,
                "vs_rigid_cfcmt_by_mechanism": {name: 1.0 for name in RIGID_FIRST_MECHANISM_OUTPUT_NAMES},
            }
        }
        if bool(args.run_rigid_first_selector):
            for mode in args.rigid_first_delta_modes:
                delta_model = _fit_rigid_first_delta_model(
                    source_train,
                    scaler,
                    ridge_model,
                    mode=mode,
                    ridge=selector_args.ridge,
                )
                for alpha in args.rigid_first_alpha_grid:
                    if float(alpha) == 0.0:
                        continue
                    metric = _evaluate_rigid_first_delta(
                        delta_model,
                        source_eval,
                        scaler,
                        ridge_model,
                        alpha=float(alpha),
                    )
                    candidate_name = _rigid_first_candidate_name(mode, float(alpha))
                    mechanism_mse = _mechanism_mse_table(metric)
                    rigid_first_candidates[candidate_name] = {
                        "total_mse": float(metric["total_mse"]),
                        "mode": mode,
                        "alpha": float(alpha),
                        "vs_rigid_cfcmt": float(metric["total_mse"]) / rigid_total_mse if rigid_total_mse else None,
                        "per_mechanism_mse": mechanism_mse,
                        "vs_rigid_cfcmt_by_mechanism": {
                            mechanism: (
                                float(value) / float(rigid_mechanism_mse[mechanism])
                                if rigid_mechanism_mse.get(mechanism) and math.isfinite(float(rigid_mechanism_mse[mechanism]))
                                else None
                            )
                            for mechanism, value in mechanism_mse.items()
                        },
                    }
        if bool(args.run_rigid_first_ensemble_selector):
            ensemble_weights, ensemble_weight_diagnostics = _source_similarity_weights(
                source_eval_full,
                {key: city_data[key] for key in train_keys},
                args,
            )
            for mode in args.rigid_first_delta_modes:
                ensemble_model = _fit_rigid_first_delta_ensemble(
                    source_train_by_city,
                    scaler,
                    ridge_model,
                    mode=mode,
                    ridge=selector_args.ridge,
                    weights=ensemble_weights,
                )
                for alpha in args.rigid_first_alpha_grid:
                    if float(alpha) == 0.0:
                        continue
                    metric = _evaluate_rigid_first_ensemble_delta(
                        ensemble_model,
                        source_eval,
                        scaler,
                        ridge_model,
                        alpha=float(alpha),
                    )
                    candidate_name = _rigid_first_ensemble_candidate_name(mode, float(alpha))
                    mechanism_mse = _mechanism_mse_table(metric)
                    rigid_first_candidates[candidate_name] = {
                        "total_mse": float(metric["total_mse"]),
                        "mode": mode,
                        "alpha": float(alpha),
                        "ensemble": True,
                        "mechanism_selectable": False,
                        "ensemble_weights": ensemble_model.weights,
                        "ensemble_weight_diagnostics": ensemble_weight_diagnostics,
                        "vs_rigid_cfcmt": float(metric["total_mse"]) / rigid_total_mse if rigid_total_mse else None,
                        "per_mechanism_mse": mechanism_mse,
                        "vs_rigid_cfcmt_by_mechanism": {
                            mechanism: (
                                float(value) / float(rigid_mechanism_mse[mechanism])
                                if rigid_mechanism_mse.get(mechanism) and math.isfinite(float(rigid_mechanism_mse[mechanism]))
                                else None
                            )
                            for mechanism, value in mechanism_mse.items()
                        },
                    }
        if not bool(args.skip_neural_full):
            assert posterior is not None and specs is not None
            for candidate_name, local_mode in candidates:
                stage = _train_neural_stage(
                    source_train,
                    source_eval,
                    scaler=scaler,
                    device=device,
                    posterior=posterior,
                    specs=specs,
                    args=selector_args,
                    use_latent=True,
                    local_mode=local_mode,
                    use_aux_features=use_aux_features,
                    ridge_model=ridge_model,
                )
                metric = stage.get("cfcmt_fallback_trust_metrics", stage["metrics"])
                candidate_metrics[candidate_name] = {
                    "total_mse": float(metric["total_mse"]),
                    "local_mode": local_mode,
                }
        base = candidate_metrics.get("no_local", candidate_metrics["rigid_cfcmt"])["total_mse"]
        for item in candidate_metrics.values():
            item["vs_no_local"] = item["total_mse"] / base if base else None
            item["vs_rigid_cfcmt"] = item["total_mse"] / rigid_total_mse if rigid_total_mse else None
        folds.append(
            {
                "heldout_source_env": heldout_key,
                "train_source_envs": train_keys,
                "train_rows": int(source_train.n),
                "eval_rows": int(source_eval.n),
                "graph_rows": int(graph_data.n) if graph_data is not None else 0,
                "candidates": candidate_metrics,
                "rigid_first_candidates": rigid_first_candidates,
            }
        )

    _annotate_source_target_similarity(folds, target_full, city_data, args)
    target_ensemble_weights, target_ensemble_weight_diagnostics = _source_similarity_weights(
        target_full,
        {key: city_data[key] for key in source_keys},
        args,
    )
    target_ensemble_max_source_weight = max(target_ensemble_weights.values()) if target_ensemble_weights else 0.0

    if bool(args.skip_neural_full):
        static_mean_ratio = None
        static_win_rate = 0.0
        selected = "no_local"
        stable_mean_gain = False
    else:
        static_ratios = [
            fold["candidates"]["deterministic_static_local"]["vs_no_local"]
            for fold in folds
            if fold["candidates"]["deterministic_static_local"]["vs_no_local"] is not None
        ]
        static_wins = [value < 1.0 - float(args.selector_min_improvement) for value in static_ratios]
        static_win_rate = float(np.mean(static_wins)) if static_wins else 0.0
        static_mean_ratio = float(np.mean(static_ratios)) if static_ratios else None
        selected, stable_mean_gain = _select_static_local_candidate(
            static_mean_ratio,
            static_win_rate,
            min_win_rate=float(args.selector_min_win_rate),
            min_improvement=float(args.selector_min_improvement),
        )
    rigid_guard = _select_rigid_guard_candidate(
        folds,
        min_win_rate=float(args.rigid_guard_min_win_rate),
        min_improvement=float(args.rigid_guard_min_improvement),
    )
    rigid_first_delta_selector = (
        _select_rigid_first_delta_candidate(
            folds,
            min_win_rate=float(args.rigid_first_min_win_rate),
            min_improvement=float(args.rigid_first_min_improvement),
            min_unweighted_win_rate=float(args.rigid_first_min_unweighted_win_rate),
            weighting=str(args.rigid_first_selector_weighting),
            ensemble_min_win_rate=float(args.rigid_first_ensemble_min_win_rate),
            ensemble_min_unweighted_win_rate=float(args.rigid_first_ensemble_min_unweighted_win_rate),
            ensemble_min_max_source_weight=float(args.rigid_first_ensemble_min_max_source_weight),
            ensemble_max_source_weight=float(target_ensemble_max_source_weight),
        )
        if bool(args.run_rigid_first_selector)
        else {"selected_candidate": "rigid_cfcmt", "selected_mode": None, "selected_alpha": 0.0, "candidate_summaries": {}}
    )
    rigid_first_mechanism_selector = (
        _select_rigid_first_mechanism_delta_candidate(
            folds,
            min_win_rate=float(args.rigid_first_mechanism_min_win_rate),
            min_improvement=float(args.rigid_first_mechanism_min_improvement),
            min_unweighted_win_rate=float(args.rigid_first_mechanism_min_unweighted_win_rate),
            weighting=str(args.rigid_first_selector_weighting),
        )
        if bool(args.run_rigid_first_mechanism_selector)
        else {"selected_by_mechanism": {}}
    )
    selected_method = {
        "no_local": "neural_autodag_latent_cfcmt_fallback_trust",
        "deterministic_static_local": "neural_autodag_latent_static_local_cfcmt_fallback_trust",
    }[selected]
    guarded_method = {
        "rigid_cfcmt": "cfcmt_ridge_template",
        "no_local": "neural_autodag_latent_cfcmt_fallback_trust",
        "deterministic_static_local": "neural_autodag_latent_static_local_cfcmt_fallback_trust",
        "proxy_encoder_local": "neural_autodag_latent_proxy_local_cfcmt_fallback_trust",
    }[rigid_guard["selected_candidate"]]
    return {
        "enabled": True,
        "definition": "Source-city leave-one validation selector; deterministic local graph is enabled only when it stably improves held-out source-city CFCMT-fallback trust MSE.",
        "candidate_scope": "deterministic_static_local_vs_no_local",
        "rigid_guard_definition": "Full neural candidates must also beat rigid CFCMT on held-out source cities; otherwise target evaluation falls back to rigid CFCMT.",
        "folds": folds,
        "static_local_mean_vs_no_local": static_mean_ratio,
        "static_local_win_rate": static_win_rate,
        "stable_mean_gain": bool(stable_mean_gain),
        "min_win_rate": float(args.selector_min_win_rate),
        "min_improvement": float(args.selector_min_improvement),
        "selected_candidate": selected,
        "selected_target_method": selected_method,
        "rigid_guard": rigid_guard,
        "rigid_guard_selected_target_method": guarded_method,
        "rigid_first_delta_selector": rigid_first_delta_selector,
        "rigid_first_mechanism_delta_selector": rigid_first_mechanism_selector,
        "target_similarity": {
            "target_env": target_key,
            "selector_weighting": str(args.rigid_first_selector_weighting),
            "ensemble_weighting": str(getattr(args, "rigid_first_ensemble_weighting", "similarity")),
            "feature_set": str(getattr(args, "selector_similarity_feature_set", "full")),
            "temperature": float(args.rigid_first_similarity_temperature),
            "max_rows": int(args.selector_similarity_max_rows),
            "ensemble_source_weights": target_ensemble_weights,
            "ensemble_source_weight_diagnostics": target_ensemble_weight_diagnostics,
            "ensemble_max_source_weight": float(target_ensemble_max_source_weight),
        },
    }


def _select_static_local_candidate(
    static_mean_ratio: float | None,
    static_win_rate: float,
    *,
    min_win_rate: float,
    min_improvement: float,
) -> tuple[str, bool]:
    stable_mean_gain = static_mean_ratio is not None and static_mean_ratio < 1.0 - float(min_improvement)
    if stable_mean_gain and float(static_win_rate) >= float(min_win_rate):
        return "deterministic_static_local", True
    return "no_local", bool(stable_mean_gain)


def _select_rigid_guard_candidate(
    folds: list[dict[str, Any]],
    *,
    min_win_rate: float,
    min_improvement: float,
) -> dict[str, Any]:
    candidate_names = sorted(
        {
            name
            for fold in folds
            for name in fold.get("candidates", {})
            if name != "rigid_cfcmt"
        }
    )
    summaries = {}
    stable = []
    for name in candidate_names:
        ratios = [
            fold["candidates"][name].get("vs_rigid_cfcmt")
            for fold in folds
            if name in fold.get("candidates", {}) and fold["candidates"][name].get("vs_rigid_cfcmt") is not None
        ]
        if not ratios:
            continue
        mean_ratio = float(np.mean(ratios))
        win_rate = float(np.mean([ratio < 1.0 - float(min_improvement) for ratio in ratios]))
        summary = {
            "mean_vs_rigid_cfcmt": mean_ratio,
            "win_rate_vs_rigid_cfcmt": win_rate,
            "folds": len(ratios),
            "stable": bool(mean_ratio < 1.0 - float(min_improvement) and win_rate >= float(min_win_rate)),
        }
        summaries[name] = summary
        if summary["stable"]:
            stable.append((name, mean_ratio))
    if not stable:
        selected = "rigid_cfcmt"
    else:
        selected = min(stable, key=lambda item: item[1])[0]
    return {
        "selected_candidate": selected,
        "candidate_summaries": summaries,
        "min_win_rate": float(min_win_rate),
        "min_improvement": float(min_improvement),
    }


def _real_apc_snapshot_proxy_check(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = _resolve_path(root, args.apc_summary)
    if not path.exists():
        return {
            "ok": False,
            "summary_path": str(path),
            "has_stop_event_vehicle_positions": None,
            "has_synchronized_vehicle_snapshot": None,
            "notes": "Austin APC summary is missing; run cf_h2o/eval/austin_apc_summary.py after Austin APC validation.",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    total_transitions = int(sum(int(row.get("transitions") or 0) for row in rows))
    return {
        "ok": bool(data.get("ok", True)),
        "summary_path": str(path),
        "datasets": int(len(rows)),
        "total_transitions": total_transitions,
        "routes": int(sum(int(row.get("routes") or 0) for row in rows)),
        "stops": int(sum(int(row.get("stops") or 0) for row in rows)),
        "trips": int(sum(int(row.get("trips") or 0) for row in rows)),
        "has_stop_event_vehicle_positions": True,
        "has_stop_event_demand": True,
        "has_synchronized_vehicle_snapshot": False,
        "has_counterfactual_holding_action": False,
        "usable_for_static_local_proxy": True,
        "usable_for_true_vehicle_local_graph": False,
        "observed_external_validation_summary": data.get("summary", {}),
        "notes": (
            "CapMetro APC provides observed stop events with vehicle_id and lat/lon, but not synchronized all-vehicle route snapshots. "
            "It is valid for external passive/few-shot APC validation and weak static local proxies; a serious local-graph claim still needs AVL/APC snapshots with neighboring vehicles/stops at the same decision time."
        ),
    }


def run_split(split: dict[str, Any], city_data: dict[str, CityTensorData], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    device = torch.device(args.device)
    source_keys = list(split["source_envs"])
    target_key = str(split["target_env"])
    target_full = city_data[target_key]
    source_train_by_city = {
        key: _select_city_subset(city_data[key], args.max_train_rows_per_city, args.seed, f"{city_data[key].key}:{split['name']}")
        for key in source_keys
    }
    source_train = _concat_preselected_data(source_train_by_city.values())
    target_eval = target_full.take(_select_rows(target_full.n, args.max_eval_rows_per_city, args.seed, f"{split['name']}:eval"))
    use_aux_features = not bool(args.no_aux_features)
    scaler = _fit_scaler(source_train, device, use_aux_features)

    methods, ridge_model = _fit_ridge_baselines(
        source_train,
        target_eval,
        args.ridge,
        args.ridge_chunk_rows,
        return_models=True,
    )
    rigid_first_delta_diagnostics: dict[str, Any] = {}
    rigid_first_delta_models: dict[str, RigidFirstDeltaBundle] = {}
    rigid_first_ensemble_diagnostics: dict[str, Any] = {}
    ensemble_weights: dict[str, float] = {}
    ensemble_weight_diagnostics: dict[str, Any] = {}
    if bool(args.run_rigid_first_ensemble_selector) or bool(args.run_h2oplus_source_ensemble):
        ensemble_weights, ensemble_weight_diagnostics = _source_similarity_weights(
            target_full,
            {key: city_data[key] for key in source_keys},
            args,
        )
    if bool(args.run_h2oplus_source_ensemble):
        dense_ensemble = _fit_dense_residual_ensemble(
            source_train_by_city,
            ridge=args.ridge,
            weights=ensemble_weights,
        )
        methods["h2oplus_source_weighted_dense_ridge"] = _evaluate_dense_residual_ensemble(
            dense_ensemble,
            target_eval,
        )
    for mode in args.rigid_first_delta_modes:
        delta_model = _fit_rigid_first_delta_model(
            source_train,
            scaler,
            ridge_model,
            mode=mode,
            ridge=args.ridge,
        )
        rigid_first_delta_models[mode] = delta_model
        rigid_first_delta_diagnostics[mode] = {
            "feature_dims": {
                name: int(delta_model.beta[name].shape[0])
                for name in OUTPUT_NAMES
            },
            "alpha_grid": [float(alpha) for alpha in args.rigid_first_alpha_grid],
        }
        for alpha in args.rigid_first_alpha_grid:
            if float(alpha) == 0.0:
                continue
            candidate_name = _rigid_first_candidate_name(mode, float(alpha))
            methods[f"rigid_first_{candidate_name}"] = _evaluate_rigid_first_delta(
                delta_model,
                target_eval,
                scaler,
                ridge_model,
                alpha=float(alpha),
            )
        if bool(args.run_rigid_first_ensemble_selector):
            ensemble_model = _fit_rigid_first_delta_ensemble(
                source_train_by_city,
                scaler,
                ridge_model,
                mode=mode,
                ridge=args.ridge,
                weights=ensemble_weights,
            )
            rigid_first_ensemble_diagnostics[mode] = {
                "feature_dims": {
                    name: int(next(iter(ensemble_model.models.values())).beta[name].shape[0])
                    for name in OUTPUT_NAMES
                },
                "alpha_grid": [float(alpha) for alpha in args.rigid_first_alpha_grid],
                "weights": ensemble_model.weights,
                "weight_diagnostics": ensemble_weight_diagnostics,
            }
            for alpha in args.rigid_first_alpha_grid:
                if float(alpha) == 0.0:
                    continue
                candidate_name = _rigid_first_ensemble_candidate_name(mode, float(alpha))
                methods[f"rigid_first_{candidate_name}"] = _evaluate_rigid_first_ensemble_delta(
                    ensemble_model,
                    target_eval,
                    scaler,
                    ridge_model,
                    alpha=float(alpha),
                )
    if bool(args.skip_neural_full):
        source_selector = {"enabled": False}
        if bool(args.run_source_selector):
            source_selector = _run_source_leave_one_selector(
                split,
                source_keys,
                city_data,
                args,
                use_aux_features=use_aux_features,
            )
            rigid_first_choice = source_selector.get("rigid_first_delta_selector", {})
            rigid_first_candidate = rigid_first_choice.get("selected_candidate", "rigid_cfcmt")
            rigid_first_method = (
                "cfcmt_ridge_template"
                if rigid_first_candidate == "rigid_cfcmt"
                else f"rigid_first_{rigid_first_candidate}"
            )
            if rigid_first_method in methods:
                methods["rigid_first_source_gated_delta"] = methods[rigid_first_method]
            mechanism_choice = source_selector.get("rigid_first_mechanism_delta_selector", {})
            if mechanism_choice.get("selected_by_mechanism"):
                methods["rigid_first_mechanism_source_gated_delta"] = _evaluate_rigid_first_mechanism_delta(
                    rigid_first_delta_models,
                    target_eval,
                    scaler,
                    ridge_model,
                    mechanism_choice,
                )
        return {
            "name": split["name"],
            "source_envs": source_keys,
            "target_env": target_key,
            "train_rows": int(source_train.n),
            "eval_rows": int(target_eval.n),
            "graph_rows": 0,
            "methods": methods,
            "ratios": _ratio_table(methods),
            "module_diagnostics": {
                "neural_full_skipped": True,
                "rigid_first_delta": rigid_first_delta_diagnostics,
                "rigid_first_similarity_ensemble_delta": rigid_first_ensemble_diagnostics,
                "source_leave_one_selector": source_selector,
            },
        }
    graph_data = source_train.take(_select_rows(source_train.n, args.max_graph_rows, args.seed, f"{split['name']}:graph"))
    graph_batch = _to_batch(graph_data, scaler, device, use_aux_features)

    template_posterior = _template_posterior(graph_batch)
    neural_template = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=template_posterior,
        specs=_template_specs(latent=False),
        args=args,
        use_latent=False,
        local_mode=LOCAL_MODE_NONE,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
    )
    methods["neural_template_graph"] = neural_template["metrics"]

    discoverer = AutoDAGDiscoverer(
        {
            "bootstrap_runs": args.dag_bootstrap_runs,
            "ridge_l2": args.dag_ridge_l2,
            "score_threshold": args.dag_score_threshold,
            "score_temperature": args.dag_score_temperature,
            "max_parents": args.dag_max_parents,
            "mechanism_parent_threshold": args.parent_threshold,
            "seed": args.seed,
        }
    )
    posterior = discoverer.fit(graph_batch, registry=FeatureRegistry.from_transition_dataset(graph_batch))
    if not bool(args.allow_id_parents):
        posterior = _remove_id_parent_edges(posterior)
    graph_out_dir = root / args.graph_artifact_dir / split["name"]
    discoverer.save(posterior, graph_out_dir)
    learned_no_latent_specs = _learned_specs(
        posterior,
        latent=False,
        threshold=args.parent_threshold,
        top_k=args.parent_top_k,
        allow_id_parents=bool(args.allow_id_parents),
    )
    neural_autodag = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=posterior,
        specs=learned_no_latent_specs,
        args=args,
        use_latent=False,
        local_mode=LOCAL_MODE_NONE,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
    )
    methods["neural_autodag"] = neural_autodag["metrics"]

    learned_latent_specs = _learned_specs(
        posterior,
        latent=True,
        threshold=args.parent_threshold,
        top_k=args.parent_top_k,
        allow_id_parents=bool(args.allow_id_parents),
    )
    neural_latent = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=posterior,
        specs=learned_latent_specs,
        args=args,
        use_latent=True,
        local_mode=LOCAL_MODE_NONE,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
    )
    methods["neural_autodag_latent"] = neural_latent["metrics"]
    methods["neural_autodag_latent_trust_gate"] = neural_latent["trust_gated_metrics"]
    methods["neural_autodag_latent_cfcmt_fallback_trust"] = neural_latent["cfcmt_fallback_trust_metrics"]

    neural_rigid_delta = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=posterior,
        specs=learned_latent_specs,
        args=args,
        use_latent=True,
        local_mode=LOCAL_MODE_NONE,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
        anchor_mode=ANCHOR_MODE_RIGID_CFCMT,
    )
    methods["neural_autodag_latent_rigid_delta"] = neural_rigid_delta["metrics"]
    methods["neural_autodag_latent_rigid_delta_trust_gate"] = neural_rigid_delta["trust_gated_metrics"]
    methods["neural_autodag_latent_rigid_delta_cfcmt_fallback_trust"] = neural_rigid_delta["cfcmt_fallback_trust_metrics"]

    neural_static_local = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=posterior,
        specs=learned_latent_specs,
        args=args,
        use_latent=True,
        local_mode=LOCAL_MODE_STATIC,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
    )
    methods["neural_autodag_latent_static_local"] = neural_static_local["metrics"]
    methods["neural_autodag_latent_static_local_trust_gate"] = neural_static_local["trust_gated_metrics"]
    methods["neural_autodag_latent_static_local_cfcmt_fallback_trust"] = neural_static_local["cfcmt_fallback_trust_metrics"]

    neural_proxy_local = _train_neural_stage(
        source_train,
        target_eval,
        scaler=scaler,
        device=device,
        posterior=posterior,
        specs=learned_latent_specs,
        args=args,
        use_latent=True,
        local_mode=LOCAL_MODE_PROXY_ENCODER,
        use_aux_features=use_aux_features,
        ridge_model=ridge_model,
    )
    methods["neural_autodag_latent_proxy_local"] = neural_proxy_local["metrics"]
    methods["neural_autodag_latent_proxy_local_trust_gate"] = neural_proxy_local["trust_gated_metrics"]
    methods["neural_autodag_latent_proxy_local_cfcmt_fallback_trust"] = neural_proxy_local["cfcmt_fallback_trust_metrics"]

    source_selector = {"enabled": False}
    if bool(args.run_source_selector):
        source_selector = _run_source_leave_one_selector(
            split,
            source_keys,
            city_data,
            args,
            use_aux_features=use_aux_features,
        )
        selected_method = source_selector.get("selected_target_method")
        if selected_method in methods:
            methods["neural_autodag_latent_source_selector_cfcmt_fallback_trust"] = methods[selected_method]
        rigid_guard_method = source_selector.get("rigid_guard_selected_target_method")
        if rigid_guard_method in methods:
            methods["neural_autodag_latent_rigid_guarded_cfcmt_fallback_trust"] = methods[rigid_guard_method]
        rigid_first_choice = source_selector.get("rigid_first_delta_selector", {})
        rigid_first_candidate = rigid_first_choice.get("selected_candidate", "rigid_cfcmt")
        rigid_first_method = (
            "cfcmt_ridge_template"
            if rigid_first_candidate == "rigid_cfcmt"
            else f"rigid_first_{rigid_first_candidate}"
        )
        if rigid_first_method in methods:
            methods["rigid_first_source_gated_delta"] = methods[rigid_first_method]
        mechanism_choice = source_selector.get("rigid_first_mechanism_delta_selector", {})
        if mechanism_choice.get("selected_by_mechanism"):
            methods["rigid_first_mechanism_source_gated_delta"] = _evaluate_rigid_first_mechanism_delta(
                rigid_first_delta_models,
                target_eval,
                scaler,
                ridge_model,
                mechanism_choice,
            )

    return {
        "name": split["name"],
        "source_envs": source_keys,
        "target_env": target_key,
        "train_rows": int(source_train.n),
        "eval_rows": int(target_eval.n),
        "graph_rows": int(graph_data.n),
        "methods": methods,
        "ratios": _ratio_table(methods),
        "module_diagnostics": {
            "template": neural_template,
            "autodag_neural": neural_autodag,
            "latent": neural_latent,
            "latent_rigid_delta": neural_rigid_delta,
            "rigid_first_delta": rigid_first_delta_diagnostics,
            "rigid_first_similarity_ensemble_delta": rigid_first_ensemble_diagnostics,
            "latent_static_local": neural_static_local,
            "latent_proxy_local": neural_proxy_local,
            "source_leave_one_selector": source_selector,
            "autodag_graph": {
                "graph_artifact_dir": str(graph_out_dir),
                "graph_diagnostics": {
                    key: value
                    for key, value in posterior.diagnostics.items()
                    if key not in {"hard_mask", "mechanisms"}
                },
                "mechanism_parent_counts": {spec.name: len(spec.parent_names) for spec in learned_latent_specs},
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    root = _repo_root()
    config_path = _resolve_path(root, args.config)
    config = _read_json(config_path)
    splits = expand_splits(config)
    if args.split_names:
        wanted = set(args.split_names.split(","))
        splits = [split for split in splits if split["name"] in wanted]
    if args.max_splits > 0:
        splits = splits[: args.max_splits]

    city_keys = sorted({key for split in splits for key in list(split["source_envs"]) + [split["target_env"]]})
    t0 = time.time()
    if args.workers > 1 and len(city_keys) > 1:
        city_data = {}
        with futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            pending = {
                executor.submit(_load_city_data, key, config["generated_envs"][key], root, args): key
                for key in city_keys
            }
            for future in futures.as_completed(pending):
                key = pending[future]
                city_data[key] = future.result()
    else:
        city_data = {
            key: _load_city_data(key, config["generated_envs"][key], root, args)
            for key in city_keys
        }
    def run_one_split(index_and_split: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, split = index_and_split
        split_t0 = time.time()
        result = run_split(split, city_data, args, root)
        result["elapsed_sec"] = time.time() - split_t0
        return index, result

    indexed_splits = list(enumerate(splits))
    split_results: list[dict[str, Any] | None] = [None] * len(indexed_splits)
    split_workers = max(1, int(args.split_workers))
    if split_workers > 1 and len(indexed_splits) > 1 and bool(args.skip_neural_full):
        with futures.ThreadPoolExecutor(max_workers=split_workers) as executor:
            pending = {executor.submit(run_one_split, item): item[0] for item in indexed_splits}
            for future in futures.as_completed(pending):
                index, result = future.result()
                split_results[index] = result
                print(json.dumps({"split": result["name"], "ratios": result["ratios"]}, indent=2), flush=True)
    else:
        for item in indexed_splits:
            index, result = run_one_split(item)
            split_results[index] = result
            print(json.dumps({"split": result["name"], "ratios": result["ratios"]}, indent=2), flush=True)
    finalized_split_results = [result for result in split_results if result is not None]

    summary = _summarize(finalized_split_results)
    return {
        "ok": True,
        "validation_level": "full_module_cross_city_pipeline",
        "config": str(config_path),
        "elapsed_sec": time.time() - t0,
        "args": _json_safe(vars(args)),
        "city_rows": {key: data.n for key, data in city_data.items()},
        "splits": finalized_split_results,
        "summary": summary,
        "real_apc_snapshot_proxy_check": _real_apc_snapshot_proxy_check(root, args),
    }


def _summarize(splits: list[dict[str, Any]]) -> dict[str, Any]:
    method_names = sorted({name for split in splits for name in split["methods"]})
    out: dict[str, Any] = {"splits": len(splits), "methods": {}}
    for name in method_names:
        ratios = [
            split["ratios"][name]["vs_h2oplus_dense_ridge"]
            for split in splits
            if name in split["ratios"] and split["ratios"][name]["vs_h2oplus_dense_ridge"] is not None
        ]
        mses = [split["methods"][name]["total_mse"] for split in splits if name in split["methods"]]
        if ratios:
            out["methods"][name] = {
                "mean_total_mse": float(np.mean(mses)),
                "mean_vs_h2oplus_dense_ridge": float(np.mean(ratios)),
                "wins_vs_h2oplus_dense_ridge": int(sum(value < 1.0 for value in ratios)),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/full_module_pipeline_validation.json"))
    parser.add_argument("--graph-artifact-dir", type=Path, default=Path("cf_h2o/results/full_module_graphs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split-workers", type=int, default=1)
    parser.add_argument("--actions", type=float, nargs="+", default=[0.0, 30.0])
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--ridge-chunk-rows", type=int, default=100_000)
    parser.add_argument("--city-tensor-cache-dir", type=Path, default=Path("cf_h2o/results/cache/city_tensors"))
    parser.add_argument("--no-city-tensor-cache", action="store_true")
    parser.add_argument("--no-write-city-tensor-cache", action="store_true")
    parser.add_argument("--refresh-city-tensor-cache", action="store_true")
    parser.add_argument("--city-tensor-cache-tag", default="")
    parser.add_argument("--max-lines-per-city", type=int, default=0)
    parser.add_argument("--max-train-rows-per-city", type=int, default=0)
    parser.add_argument("--max-eval-rows-per-city", type=int, default=0)
    parser.add_argument("--max-graph-rows", type=int, default=0)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--split-names", default="")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--neural-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--sim-epochs", type=int, default=3)
    parser.add_argument("--residual-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--residual-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--history-len", type=int, default=2)
    parser.add_argument("--local-embedding-dim", type=int, default=16)
    parser.add_argument("--skip-neural-full", action="store_true")
    parser.add_argument("--dag-bootstrap-runs", type=int, default=3)
    parser.add_argument("--dag-ridge-l2", type=float, default=1e-3)
    parser.add_argument("--dag-score-threshold", type=float, default=0.04)
    parser.add_argument("--dag-score-temperature", type=float, default=0.04)
    parser.add_argument("--dag-max-parents", type=int, default=8)
    parser.add_argument("--parent-threshold", type=float, default=0.25)
    parser.add_argument("--parent-top-k", type=int, default=8)
    parser.add_argument(
        "--no-aux-features",
        dest="no_aux_features",
        action="store_true",
        default=True,
        help="Disable policy-safe nonlinear auxiliary features for neural modules. This is the default after validation.",
    )
    parser.add_argument(
        "--use-aux-features",
        dest="no_aux_features",
        action="store_false",
        help="Enable auxiliary nonlinear features as an ablation.",
    )
    parser.add_argument(
        "--allow-id-parents",
        dest="allow_id_parents",
        action="store_true",
        default=True,
        help="Allow line_id/bus_id parent nodes in learned AutoDAG specs. This is the default diagnostic setting.",
    )
    parser.add_argument(
        "--forbid-id-parents",
        dest="allow_id_parents",
        action="store_false",
        help="Forbid line_id/bus_id parent nodes as a stricter anti-shortcut ablation.",
    )
    parser.add_argument("--trust-residual-scale", type=float, default=0.0)
    parser.add_argument("--trust-uncertainty-scale", type=float, default=0.15)
    parser.add_argument("--trust-graph-uncertainty-scale", type=float, default=0.25)
    parser.add_argument("--run-source-selector", action="store_true")
    parser.add_argument("--selector-max-train-rows-per-city", type=int, default=5000)
    parser.add_argument("--selector-max-eval-rows-per-city", type=int, default=5000)
    parser.add_argument("--selector-max-graph-rows", type=int, default=5000)
    parser.add_argument("--selector-sim-epochs", type=int, default=1)
    parser.add_argument("--selector-residual-epochs", type=int, default=1)
    parser.add_argument("--selector-dag-bootstrap-runs", type=int, default=1)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--selector-eval-batch-size", type=int, default=4096)
    parser.add_argument("--selector-min-win-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--selector-min-improvement", type=float, default=0.0)
    parser.add_argument("--selector-include-proxy", action="store_true")
    parser.add_argument("--rigid-guard-min-win-rate", type=float, default=1.0)
    parser.add_argument("--rigid-guard-min-improvement", type=float, default=0.0)
    parser.add_argument("--rigid-first-delta-modes", nargs="+", choices=RIGID_FIRST_DELTA_MODES, default=list(RIGID_FIRST_DELTA_MODES))
    parser.add_argument("--rigid-first-alpha-grid", type=float, nargs="+", default=list(RIGID_FIRST_ALPHA_GRID))
    parser.add_argument("--run-rigid-first-selector", dest="run_rigid_first_selector", action="store_true", default=True)
    parser.add_argument("--skip-rigid-first-selector", dest="run_rigid_first_selector", action="store_false")
    parser.add_argument("--run-rigid-first-ensemble-selector", dest="run_rigid_first_ensemble_selector", action="store_true", default=True)
    parser.add_argument("--skip-rigid-first-ensemble-selector", dest="run_rigid_first_ensemble_selector", action="store_false")
    parser.add_argument("--run-h2oplus-source-ensemble", dest="run_h2oplus_source_ensemble", action="store_true", default=True)
    parser.add_argument("--skip-h2oplus-source-ensemble", dest="run_h2oplus_source_ensemble", action="store_false")
    parser.add_argument("--run-rigid-first-mechanism-selector", dest="run_rigid_first_mechanism_selector", action="store_true", default=True)
    parser.add_argument("--skip-rigid-first-mechanism-selector", dest="run_rigid_first_mechanism_selector", action="store_false")
    parser.add_argument("--rigid-first-selector-weighting", choices=["uniform", "similarity"], default="similarity")
    parser.add_argument("--rigid-first-ensemble-weighting", choices=["uniform", "similarity"], default="similarity")
    parser.add_argument("--selector-similarity-feature-set", choices=SIMILARITY_FEATURE_SETS, default="sim_only")
    parser.add_argument("--rigid-first-similarity-temperature", type=float, default=1.0)
    parser.add_argument("--selector-similarity-max-rows", type=int, default=100_000)
    parser.add_argument("--rigid-first-min-win-rate", type=float, default=1.0)
    parser.add_argument("--rigid-first-min-unweighted-win-rate", type=float, default=1.0)
    parser.add_argument("--rigid-first-min-improvement", type=float, default=0.005)
    parser.add_argument("--rigid-first-ensemble-min-win-rate", type=float, default=0.8)
    parser.add_argument("--rigid-first-ensemble-min-unweighted-win-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--rigid-first-ensemble-min-max-source-weight", type=float, default=0.5)
    parser.add_argument("--rigid-first-mechanism-min-win-rate", type=float, default=1.0)
    parser.add_argument("--rigid-first-mechanism-min-unweighted-win-rate", type=float, default=1.0)
    parser.add_argument("--rigid-first-mechanism-min-improvement", type=float, default=0.005)
    parser.add_argument("--apc-summary", type=Path, default=Path("cf_h2o/results/austin_real_apc_validation_summary.json"))
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(_json_safe(result), indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
