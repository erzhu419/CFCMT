"""Validate static local graph features on generated SUMO APC/AVL snapshots.

The Stage 3 generator produces synchronized vehicle-level AVL rows with APC
state overlays. This validation turns those snapshots into one-step occupancy
transitions and tests whether deterministic local route-neighborhood features
improve cross-city transfer over a no-local baseline.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_performance_validation import _repo_root, _resolve_path, _stable_unit


DEFAULT_STAGE3_REPORT = Path("cf_h2o/results/sumo_apc_avl_snapshot_generation_full_day.json")
DEFAULT_OUT = Path("cf_h2o/results/sumo_apc_avl_local_snapshot_validation.json")
NO_LOCAL_FEATURE_NAMES = [
    "occupancy_norm",
    "speed_norm",
    "segment_pos",
    "hour_sin",
    "hour_cos",
    "last_boardings_norm",
    "last_alightings_norm",
    "cumulative_boardings_norm",
    "cumulative_alightings_norm",
    "transition_gap_norm",
]
STATIC_LOCAL_FEATURE_NAMES = [
    "same_line_vehicle_count_norm",
    "same_line_occupancy_mean",
    "same_line_occupancy_std",
    "same_line_occupancy_max",
    "same_line_speed_mean",
    "nearest_distance_norm",
    "nearest_occupancy",
    "nearest_speed",
    "front_segment_gap_norm",
    "front_occupancy",
    "front_speed",
    "back_segment_gap_norm",
    "back_occupancy",
    "back_speed",
    "front_back_occupancy_delta",
    "near_segment_occupancy_mean",
    "near_segment_vehicle_count_norm",
    "segment_headway_imbalance",
    "abs_segment_headway_imbalance",
]


@dataclass
class SnapshotDataset:
    key: str
    label: str
    features_no_local: np.ndarray
    features_static_local: np.ndarray
    target: np.ndarray
    snapshot_times: np.ndarray
    vehicle_capacity: float
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback)
    scale = float(np.nanpercentile(np.abs(finite), 95))
    return scale if math.isfinite(scale) and scale > 1e-9 else float(fallback)


def _select_rows(n_rows: int, max_rows: int, seed: int, label: str) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + int(_stable_unit(label, modulus=1_000_000)))
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False).astype(np.int64))


def _cache_signature(city: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    avl_path = Path(city["avl_path"])
    return {
        "version": 1,
        "avl_path": str(avl_path),
        "avl_mtime": avl_path.stat().st_mtime if avl_path.exists() else None,
        "max_gap_seconds": float(args.max_gap_seconds),
        "max_transitions_per_city": int(args.max_transitions_per_city),
        "vehicle_capacity": float(args.vehicle_capacity or city.get("vehicle_capacity") or 90.0),
        "seed": int(args.seed),
    }


def _cache_path(city: dict[str, Any]) -> Path:
    return Path(city["out_dir"]) / "local_snapshot_dataset.npz"


def _base_features(df: pd.DataFrame, *, vehicle_capacity: float, segment_scale: float) -> np.ndarray:
    timestamp = pd.to_numeric(df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float64)
    hour = (timestamp / 3600.0) % 24.0
    occupancy = pd.to_numeric(df["occupancy"], errors="coerce").to_numpy(dtype=np.float64)
    speed = pd.to_numeric(df["speed_mps"], errors="coerce").to_numpy(dtype=np.float64)
    segment = pd.to_numeric(df["segment_idx"], errors="coerce").to_numpy(dtype=np.float64)
    last_boardings = pd.to_numeric(df["last_boardings"], errors="coerce").to_numpy(dtype=np.float64)
    last_alightings = pd.to_numeric(df["last_alightings"], errors="coerce").to_numpy(dtype=np.float64)
    cum_board = pd.to_numeric(df["cumulative_boardings"], errors="coerce").to_numpy(dtype=np.float64)
    cum_alight = pd.to_numeric(df["cumulative_alightings"], errors="coerce").to_numpy(dtype=np.float64)
    gap = pd.to_numeric(df["transition_gap_seconds"], errors="coerce").to_numpy(dtype=np.float64)
    cum_scale = max(vehicle_capacity, _robust_scale(np.r_[cum_board, cum_alight], fallback=vehicle_capacity))
    return np.column_stack(
        [
            occupancy / vehicle_capacity,
            speed / 25.0,
            segment / segment_scale,
            np.sin(hour / 24.0 * math.tau),
            np.cos(hour / 24.0 * math.tau),
            last_boardings / vehicle_capacity,
            last_alightings / vehicle_capacity,
            cum_board / cum_scale,
            cum_alight / cum_scale,
            gap / max(1.0, 2.0 * float(np.nanmedian(gap[np.isfinite(gap)])) if np.isfinite(gap).any() else 120.0),
        ]
    ).astype(np.float64)


def _static_local_features(df: pd.DataFrame, *, vehicle_capacity: float, segment_scale: float) -> np.ndarray:
    n_rows = len(df)
    features = np.zeros((n_rows, len(STATIC_LOCAL_FEATURE_NAMES)), dtype=np.float64)
    if n_rows == 0:
        return features

    occupancy = pd.to_numeric(df["occupancy"], errors="coerce").to_numpy(dtype=np.float64)
    speed = pd.to_numeric(df["speed_mps"], errors="coerce").to_numpy(dtype=np.float64)
    segment = pd.to_numeric(df["segment_idx"], errors="coerce").to_numpy(dtype=np.float64)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64)
    for _group, idx in df.groupby(["timestamp_sec", "line_uid"], sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        m = int(idx_arr.size)
        if m <= 1:
            features[idx_arr, 0] = 1.0 / 20.0
            features[idx_arr, 1] = occupancy[idx_arr] / vehicle_capacity
            features[idx_arr, 3] = occupancy[idx_arr] / vehicle_capacity
            features[idx_arr, 4] = np.nan_to_num(speed[idx_arr] / 25.0, nan=0.0)
            features[idx_arr, 6] = occupancy[idx_arr] / vehicle_capacity
            features[idx_arr, 7] = np.nan_to_num(speed[idx_arr] / 25.0, nan=0.0)
            features[idx_arr, 15] = occupancy[idx_arr] / vehicle_capacity
            continue

        occ_g = occupancy[idx_arr]
        speed_g = speed[idx_arr]
        seg_g = segment[idx_arr]
        x_g = x[idx_arr]
        y_g = y[idx_arr]
        occ_sum = np.nansum(occ_g)
        occ_sumsq = np.nansum(occ_g**2)
        speed_sum = np.nansum(speed_g)
        speed_present = np.isfinite(speed_g)
        count_other = float(max(1, m - 1))
        dx = x_g[:, None] - x_g[None, :]
        dy = y_g[:, None] - y_g[None, :]
        distance = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(distance, np.inf)
        nearest_pos = np.argmin(distance, axis=1)
        nearest_dist = distance[np.arange(m), nearest_pos]
        nearest_global = idx_arr[nearest_pos]

        for local_pos, global_idx in enumerate(idx_arr):
            other_occ_sum = occ_sum - occupancy[global_idx]
            other_occ_mean = other_occ_sum / count_other
            other_occ_var = max(0.0, (occ_sumsq - occupancy[global_idx] ** 2) / count_other - other_occ_mean**2)
            speed_den = max(1.0, float(speed_present.sum() - int(np.isfinite(speed[global_idx]))))
            features[global_idx, 0] = float(m) / 20.0
            features[global_idx, 1] = other_occ_mean / vehicle_capacity
            features[global_idx, 2] = math.sqrt(other_occ_var) / vehicle_capacity
            features[global_idx, 3] = np.nanmax(np.delete(occ_g, local_pos)) / vehicle_capacity
            features[global_idx, 4] = (speed_sum - (speed[global_idx] if np.isfinite(speed[global_idx]) else 0.0)) / speed_den / 25.0
            features[global_idx, 5] = 0.0 if not np.isfinite(nearest_dist[local_pos]) else nearest_dist[local_pos] / 10_000.0
            features[global_idx, 6] = occupancy[nearest_global[local_pos]] / vehicle_capacity
            features[global_idx, 7] = _safe_num(speed[nearest_global[local_pos]] / 25.0)

            front_global = None
            back_global = None
            if np.isfinite(segment[global_idx]) and np.isfinite(seg_g).sum() > 1:
                higher = np.where(seg_g > segment[global_idx])[0]
                lower = np.where(seg_g < segment[global_idx])[0]
                if higher.size:
                    front_global = idx_arr[higher[np.argmin(seg_g[higher] - segment[global_idx])]]
                if lower.size:
                    back_global = idx_arr[lower[np.argmin(segment[global_idx] - seg_g[lower])]]
            if front_global is None:
                front_global = nearest_global[local_pos]
            if back_global is None:
                back_global = nearest_global[local_pos]

            front_gap = max(0.0, _safe_num(segment[front_global] - segment[global_idx]))
            back_gap = max(0.0, _safe_num(segment[global_idx] - segment[back_global]))
            near_mask = np.abs(seg_g - segment[global_idx]) <= 1.0
            near_mask[local_pos] = False
            if near_mask.any():
                near_occ = float(np.nanmean(occ_g[near_mask]))
                near_count = float(near_mask.sum())
            else:
                near_occ = float(occupancy[global_idx])
                near_count = 1.0
            features[global_idx, 8] = front_gap / segment_scale
            features[global_idx, 9] = occupancy[front_global] / vehicle_capacity
            features[global_idx, 10] = _safe_num(speed[front_global] / 25.0)
            features[global_idx, 11] = back_gap / segment_scale
            features[global_idx, 12] = occupancy[back_global] / vehicle_capacity
            features[global_idx, 13] = _safe_num(speed[back_global] / 25.0)
            features[global_idx, 14] = (occupancy[front_global] - occupancy[back_global]) / vehicle_capacity
            features[global_idx, 15] = near_occ / vehicle_capacity
            features[global_idx, 16] = near_count / 20.0
            denom = max(1.0, front_gap + back_gap)
            features[global_idx, 17] = (front_gap - back_gap) / denom
            features[global_idx, 18] = abs(front_gap - back_gap) / denom
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _make_transitions(avl: pd.DataFrame, *, max_gap_seconds: float) -> pd.DataFrame:
    ordered = avl.sort_values(["vehicle_id", "timestamp_sec"], kind="mergesort").reset_index(drop=True)
    nxt = ordered.groupby("vehicle_id", sort=False).shift(-1)
    gap = pd.to_numeric(nxt["timestamp_sec"], errors="coerce") - pd.to_numeric(ordered["timestamp_sec"], errors="coerce")
    valid = (
        gap.gt(0.0)
        & gap.le(float(max_gap_seconds))
        & (nxt["line_uid"].astype(str) == ordered["line_uid"].astype(str))
        & pd.to_numeric(nxt["occupancy"], errors="coerce").notna()
    )
    transitions = ordered.loc[valid].copy()
    transitions["target_occupancy"] = pd.to_numeric(nxt.loc[valid, "occupancy"], errors="coerce").to_numpy(dtype=np.float64)
    transitions["transition_gap_seconds"] = gap.loc[valid].to_numpy(dtype=np.float64)
    return transitions.reset_index(drop=True)


def _read_avl(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp_sec",
        "city_key",
        "city",
        "vehicle_id",
        "line_key",
        "line_uid",
        "segment_idx",
        "x",
        "y",
        "speed_mps",
        "occupancy",
        "last_boardings",
        "last_alightings",
        "cumulative_boardings",
        "cumulative_alightings",
    ]
    df = pd.read_csv(path, usecols=usecols)
    for column in [
        "timestamp_sec",
        "segment_idx",
        "x",
        "y",
        "speed_mps",
        "occupancy",
        "last_boardings",
        "last_alightings",
        "cumulative_boardings",
        "cumulative_alightings",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["timestamp_sec"].notna() & df["vehicle_id"].notna() & df["line_uid"].notna() & df["occupancy"].notna()]
    return df.reset_index(drop=True)


def _build_dataset(city_key: str, city: dict[str, Any], args: argparse.Namespace) -> SnapshotDataset:
    vehicle_capacity = float(args.vehicle_capacity or city.get("vehicle_capacity") or 90.0)
    signature = _cache_signature(city, args)
    cache = _cache_path(city)
    if not args.rebuild_cache and cache.exists():
        try:
            payload = np.load(cache, allow_pickle=False)
            if json.loads(str(payload["signature"])) == signature:
                summary = json.loads(str(payload["summary"]))
                summary["cache_hit"] = True
                return SnapshotDataset(
                    key=city_key,
                    label=str(city.get("city", city_key)),
                    features_no_local=payload["features_no_local"],
                    features_static_local=payload["features_static_local"],
                    target=payload["target"],
                    snapshot_times=payload["snapshot_times"],
                    vehicle_capacity=vehicle_capacity,
                    summary=summary,
                )
        except Exception:
            pass

    started = time.time()
    avl = _read_avl(Path(city["avl_path"]))
    transitions = _make_transitions(avl, max_gap_seconds=float(args.max_gap_seconds))
    if len(transitions) < int(args.min_transitions):
        raise RuntimeError(f"{city_key}: only {len(transitions)} transitions after filters")
    segment_scale = _robust_scale(pd.to_numeric(transitions["segment_idx"], errors="coerce").to_numpy(dtype=np.float64), fallback=100.0)
    no_local = _base_features(transitions, vehicle_capacity=vehicle_capacity, segment_scale=segment_scale)
    static = np.concatenate(
        [no_local, _static_local_features(transitions, vehicle_capacity=vehicle_capacity, segment_scale=segment_scale)],
        axis=1,
    )
    target = np.clip(pd.to_numeric(transitions["target_occupancy"], errors="coerce").to_numpy(dtype=np.float64) / vehicle_capacity, 0.0, 1.0)
    snapshot_times = pd.to_numeric(transitions["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float64)
    selected = _select_rows(len(target), int(args.max_transitions_per_city), int(args.seed), f"{city_key}:sumo_local_snapshot")
    no_local = no_local[selected]
    static = static[selected]
    target = target[selected]
    snapshot_times = snapshot_times[selected]
    selected_transitions = transitions.iloc[selected]
    summary = {
        "city_key": city_key,
        "label": str(city.get("city", city_key)),
        "transitions": int(len(target)),
        "transitions_before_sampling": int(len(transitions)),
        "snapshots": int(selected_transitions["timestamp_sec"].nunique()),
        "lines": int(selected_transitions["line_uid"].nunique()),
        "vehicles": int(selected_transitions["vehicle_id"].nunique()),
        "vehicle_capacity": vehicle_capacity,
        "segment_scale": float(segment_scale),
        "source_avl_path": str(city["avl_path"]),
        "cache_hit": False,
        "elapsed_sec": time.time() - started,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        signature=json.dumps(signature, sort_keys=True),
        summary=json.dumps(summary, sort_keys=True),
        features_no_local=no_local,
        features_static_local=static,
        target=target,
        snapshot_times=snapshot_times,
    )
    return SnapshotDataset(
        key=city_key,
        label=str(city.get("city", city_key)),
        features_no_local=no_local,
        features_static_local=static,
        target=target,
        snapshot_times=snapshot_times,
        vehicle_capacity=vehicle_capacity,
        summary=summary,
    )


def _build_dataset_worker(payload: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[str, SnapshotDataset]:
    city_key, city, args_dict = payload
    return city_key, _build_dataset(city_key, city, argparse.Namespace(**args_dict))


def _load_datasets(args: argparse.Namespace) -> dict[str, SnapshotDataset]:
    root = _repo_root()
    stage3_report = _read_json(_resolve_path(root, args.stage3_report))
    cities = stage3_report.get("cities", {})
    city_keys = list(args.cities) if args.cities else list(cities.keys())
    payloads = [(city_key, cities[city_key], dict(vars(args))) for city_key in city_keys]
    workers = max(1, min(int(args.workers), len(payloads)))
    datasets: dict[str, SnapshotDataset] = {}
    if workers <= 1:
        for payload in payloads:
            city_key, data = _build_dataset_worker(payload)
            print(f"[local-snapshot] {city_key} transitions={data.summary['transitions']} cache={data.summary['cache_hit']}", flush=True)
            datasets[city_key] = data
    else:
        print(f"[local-snapshot] building datasets with workers={workers}", flush=True)
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_city = {executor.submit(_build_dataset_worker, payload): payload[0] for payload in payloads}
            for future in futures.as_completed(future_to_city):
                city_key, data = future.result()
                print(f"[local-snapshot] {city_key} transitions={data.summary['transitions']} cache={data.summary['cache_hit']}", flush=True)
                datasets[city_key] = data
    return datasets


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xz = (x - mean) / std
    design = np.column_stack([np.ones(xz.shape[0]), xz])
    eye = np.eye(design.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + float(ridge) * eye, design.T @ y)
    return {"mean": mean, "std": std, "beta": beta}


def _ridge_predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xz = (x - model["mean"]) / model["std"]
    design = np.column_stack([np.ones(xz.shape[0]), xz])
    return design @ model["beta"]


def _fit_eval(train: list[SnapshotDataset], eval_data: SnapshotDataset, *, use_static_local: bool, ridge: float) -> dict[str, Any]:
    x_train = np.concatenate([item.features_static_local if use_static_local else item.features_no_local for item in train], axis=0)
    y_train = np.concatenate([item.target for item in train], axis=0)
    x_eval = eval_data.features_static_local if use_static_local else eval_data.features_no_local
    pred = np.clip(_ridge_predict(_ridge_fit(x_train, y_train, ridge), x_eval), 0.0, 1.0)
    err = pred - eval_data.target
    mae = float(np.mean(np.abs(err)))
    return {
        "mse": float(np.mean(err**2)),
        "mae": mae,
        "rmse_passengers": float(math.sqrt(np.mean(err**2)) * eval_data.vehicle_capacity),
        "mae_passengers": float(mae * eval_data.vehicle_capacity),
        "n_train": int(y_train.shape[0]),
        "n_eval": int(eval_data.target.shape[0]),
    }


def _selector_choice(
    source_keys: list[str],
    datasets: dict[str, SnapshotDataset],
    *,
    ridge: float,
    min_win_rate: float,
    min_improvement: float,
) -> dict[str, Any]:
    folds = []
    for heldout in source_keys:
        train_keys = [key for key in source_keys if key != heldout]
        if not train_keys:
            continue
        no = _fit_eval([datasets[key] for key in train_keys], datasets[heldout], use_static_local=False, ridge=ridge)
        local = _fit_eval([datasets[key] for key in train_keys], datasets[heldout], use_static_local=True, ridge=ridge)
        ratio = local["mse"] / no["mse"] if no["mse"] else None
        folds.append({"heldout": heldout, "train": train_keys, "no_local": no, "static_local": local, "static_vs_no_local": ratio})
    ratios = [fold["static_vs_no_local"] for fold in folds if fold["static_vs_no_local"] is not None]
    mean_ratio = float(np.mean(ratios)) if ratios else None
    win_rate = float(np.mean([value < 1.0 - min_improvement for value in ratios])) if ratios else 0.0
    selected = "static_local" if mean_ratio is not None and mean_ratio < 1.0 - min_improvement and win_rate >= min_win_rate else "no_local"
    return {
        "folds": folds,
        "mean_static_vs_no_local": mean_ratio,
        "static_win_rate": win_rate,
        "min_win_rate": float(min_win_rate),
        "min_improvement": float(min_improvement),
        "selected": selected,
    }


def _cross_city_eval(datasets: dict[str, SnapshotDataset], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    keys = sorted(datasets)
    for target in keys:
        source_keys = [key for key in keys if key != target]
        no = _fit_eval([datasets[key] for key in source_keys], datasets[target], use_static_local=False, ridge=args.ridge)
        local = _fit_eval([datasets[key] for key in source_keys], datasets[target], use_static_local=True, ridge=args.ridge)
        selector = _selector_choice(
            source_keys,
            datasets,
            ridge=args.ridge,
            min_win_rate=args.selector_min_win_rate,
            min_improvement=args.selector_min_improvement,
        )
        selected_metrics = local if selector["selected"] == "static_local" else no
        results.append(
            {
                "target": target,
                "sources": source_keys,
                "no_local": no,
                "static_local": local,
                "selector_gated_static_local": selected_metrics,
                "static_vs_no_local": local["mse"] / no["mse"] if no["mse"] else None,
                "selector_vs_no_local": selected_metrics["mse"] / no["mse"] if no["mse"] else None,
                "selector": selector,
            }
        )
    return results


def _temporal_eval(datasets: dict[str, SnapshotDataset], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for key, data in sorted(datasets.items()):
        order = np.argsort(data.snapshot_times)
        cut = max(1, min(len(order) - 1, int(len(order) * float(args.temporal_train_fraction))))
        train_idx = order[:cut]
        eval_idx = order[cut:]
        train = SnapshotDataset(
            key=data.key,
            label=data.label,
            features_no_local=data.features_no_local[train_idx],
            features_static_local=data.features_static_local[train_idx],
            target=data.target[train_idx],
            snapshot_times=data.snapshot_times[train_idx],
            vehicle_capacity=data.vehicle_capacity,
            summary=data.summary,
        )
        eval_data = SnapshotDataset(
            key=data.key,
            label=data.label,
            features_no_local=data.features_no_local[eval_idx],
            features_static_local=data.features_static_local[eval_idx],
            target=data.target[eval_idx],
            snapshot_times=data.snapshot_times[eval_idx],
            vehicle_capacity=data.vehicle_capacity,
            summary=data.summary,
        )
        no = _fit_eval([train], eval_data, use_static_local=False, ridge=args.ridge)
        local = _fit_eval([train], eval_data, use_static_local=True, ridge=args.ridge)
        results.append(
            {
                "target": key,
                "no_local": no,
                "static_local": local,
                "static_vs_no_local": local["mse"] / no["mse"] if no["mse"] else None,
            }
        )
    return results


def _summarize_cross_city(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    no_local_mean_mse = float(np.mean([item["no_local"]["mse"] for item in results]))
    for method in ["no_local", "static_local", "selector_gated_static_local"]:
        mses = [item[method]["mse"] for item in results]
        mean_mse = float(np.mean(mses))
        out[method] = {
            "mean_mse": mean_mse,
            "mean_mse_vs_no_local": mean_mse / no_local_mean_mse if no_local_mean_mse else None,
            "wins_vs_no_local": int(sum(item[method]["mse"] < item["no_local"]["mse"] for item in results)),
        }
    ratios = [item["static_vs_no_local"] for item in results if item["static_vs_no_local"] is not None]
    selector_ratios = [item["selector_vs_no_local"] for item in results if item["selector_vs_no_local"] is not None]
    out["static_local_mean_vs_no_local"] = float(np.mean(ratios)) if ratios else None
    out["selector_mean_vs_no_local"] = float(np.mean(selector_ratios)) if selector_ratios else None
    return out


def _write_summary_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# SUMO APC/AVL Local Snapshot Validation",
        "",
        "Task: one-step occupancy prediction from full-day synchronized SUMO APC/AVL snapshots.",
        "",
        "## Dataset",
        "",
        "| City | Transitions | Raw transitions | Snapshots | Lines | Vehicles | Cache |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["datasets"].values():
        lines.append(
            f"| {item['label']} | {item['transitions']} | {item['transitions_before_sampling']} | "
            f"{item['snapshots']} | {item['lines']} | {item['vehicles']} | {item['cache_hit']} |"
        )
    lines.extend(["", "## Cross-City Results", "", "| Target | No Local MSE | Static Local MSE | Static/No | Selector | Selector/No |", "|---|---:|---:|---:|---|---:|"])
    for item in result["cross_city"]:
        lines.append(
            f"| {item['target']} | {item['no_local']['mse']:.6f} | {item['static_local']['mse']:.6f} | "
            f"{item['static_vs_no_local']:.4f} | {item['selector']['selected']} | {item['selector_vs_no_local']:.4f} |"
        )
    summary = result["summary"]["cross_city"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Static local mean MSE vs no-local: `{summary['static_local']['mean_mse_vs_no_local']:.4f}`.",
            f"- Selector-gated mean MSE vs no-local: `{summary['selector_gated_static_local']['mean_mse_vs_no_local']:.4f}`.",
            f"- Static local mean vs no-local: `{summary['static_local_mean_vs_no_local']:.4f}`.",
            f"- Selector-gated mean vs no-local: `{summary['selector_mean_vs_no_local']:.4f}`.",
            f"- Static local wins vs no-local: `{summary['static_local']['wins_vs_no_local']}/{len(result['cross_city'])}`.",
            f"- Selector-gated wins vs no-local: `{summary['selector_gated_static_local']['wins_vs_no_local']}/{len(result['cross_city'])}`.",
            "",
            "Interpretation: selector-gated static-local should be used for CFCMT local graph claims only when source-city leave-one validation selects it; otherwise no-local remains the conservative variant.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    started = time.time()
    datasets = _load_datasets(args)
    cross_city = _cross_city_eval(datasets, args)
    temporal = _temporal_eval(datasets, args)
    result = {
        "ok": True,
        "validation_level": "sumo_apc_avl_local_snapshot_cross_city",
        "definition": "Cross-city one-step occupancy prediction from generated full-day SUMO APC/AVL snapshots.",
        "stage3_report": str(_resolve_path(root, args.stage3_report)),
        "feature_names": {
            "no_local": list(NO_LOCAL_FEATURE_NAMES),
            "static_local": list(NO_LOCAL_FEATURE_NAMES) + list(STATIC_LOCAL_FEATURE_NAMES),
        },
        "settings": {
            "max_gap_seconds": float(args.max_gap_seconds),
            "max_transitions_per_city": int(args.max_transitions_per_city),
            "ridge": float(args.ridge),
            "selector_min_win_rate": float(args.selector_min_win_rate),
            "selector_min_improvement": float(args.selector_min_improvement),
            "workers": int(args.workers),
        },
        "datasets": {key: data.summary for key, data in sorted(datasets.items())},
        "cross_city": cross_city,
        "temporal_single_city": temporal,
        "summary": {
            "cross_city": _summarize_cross_city(cross_city),
            "temporal_static_mean_vs_no_local": float(np.mean([item["static_vs_no_local"] for item in temporal])),
        },
        "elapsed_sec": time.time() - started,
    }
    out_path = _resolve_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_summary_md(result, out_path.with_suffix(".md"))
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-report", type=Path, default=DEFAULT_STAGE3_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-gap-seconds", type=float, default=180.0)
    parser.add_argument("--min-transitions", type=int, default=100)
    parser.add_argument("--max-transitions-per-city", type=int, default=0, help="0 means all full-day transitions")
    parser.add_argument("--vehicle-capacity", type=float, default=0.0, help="0 means read from Stage 3 city summary")
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--selector-min-win-rate", type=float, default=1.0)
    parser.add_argument("--selector-min-improvement", type=float, default=0.0)
    parser.add_argument("--temporal-train-fraction", type=float, default=0.7)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
