"""Validate static local graph features on GTFS-RT AVL/occupancy snapshots.

This is intentionally not a full CFCMT/H2O+ dynamics benchmark: GTFS-RT
occupancy is a present-time crowding signal, not exact APC board/alight counts,
and the feeds contain no holding actions. The experiment asks a narrower
question: on real synchronized vehicle-position snapshots, do deterministic
local route-neighborhood features improve one-step occupancy transfer?
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
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_performance_validation import _repo_root, _resolve_path


OCCUPANCY_ORDINAL = {
    "EMPTY": 0.0,
    "MANY_SEATS_AVAILABLE": 1.0,
    "FEW_SEATS_AVAILABLE": 2.0,
    "STANDING_ROOM_ONLY": 3.0,
    "CRUSHED_STANDING_ROOM_ONLY": 4.0,
    "FULL": 5.0,
    "NOT_ACCEPTING_PASSENGERS": 5.0,
}
NO_LOCAL_FEATURE_NAMES = [
    "current_occupancy_ordinal",
    "current_occupancy_percentage",
    "has_occupancy_percentage",
    "lat_city_norm",
    "lon_city_norm",
    "current_stop_sequence_norm",
    "has_stop_sequence",
    "speed_norm",
    "has_speed",
    "current_status_norm",
    "direction_norm",
    "hour_sin",
    "hour_cos",
    "gap_seconds_norm",
]
STATIC_LOCAL_FEATURE_NAMES = [
    "same_route_vehicle_count",
    "same_route_occupancy_mean",
    "same_route_occupancy_std",
    "same_route_occupancy_max",
    "same_route_pct_mean",
    "same_route_pct_present_rate",
    "same_route_speed_mean",
    "nearest_distance_km",
    "nearest_occupancy",
    "nearest_speed",
    "front_sequence_gap",
    "front_occupancy",
    "front_speed",
    "back_sequence_gap",
    "back_occupancy",
    "back_speed",
    "front_back_occupancy_delta",
    "nearby_high_occupancy_rate",
]


@dataclass
class SnapshotDataset:
    key: str
    label: str
    frame: pd.DataFrame
    features_no_local: np.ndarray
    features_static_local: np.ndarray
    target: np.ndarray
    snapshot_times: np.ndarray


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _occupancy_value(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    return OCCUPANCY_ORDINAL.get(str(value), np.nan)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _read_snapshot_file(path: Path, source_key: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["source_key"] = source_key
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_ts"], utc=True, errors="coerce")
    df["vehicle_dt"] = pd.to_datetime(df["vehicle_timestamp"], utc=True, errors="coerce")
    df["occupancy_ordinal"] = df["occupancy_status"].map(_occupancy_value).astype(float)
    df["occupancy_pct_norm"] = pd.to_numeric(df["occupancy_percentage"], errors="coerce").astype(float) / 100.0
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["current_stop_sequence"] = pd.to_numeric(df["current_stop_sequence"], errors="coerce")
    df["current_status"] = pd.to_numeric(df["current_status"], errors="coerce")
    df["direction_id"] = pd.to_numeric(df["direction_id"], errors="coerce")
    df = df[df["snapshot_dt"].notna() & df["vehicle_id"].notna() & df["route_id"].notna()]
    df = df[df["latitude"].notna() & df["longitude"].notna() & df["occupancy_ordinal"].notna()]
    return df.reset_index(drop=True)


def _city_norm(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    out = (values - lo) / (hi - lo)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _make_transitions(df: pd.DataFrame, *, max_gap_seconds: float) -> pd.DataFrame:
    ordered = df.sort_values(["vehicle_id", "snapshot_dt", "route_id"]).reset_index(drop=True)
    nxt = ordered.groupby("vehicle_id", sort=False).shift(-1)
    gap = (pd.to_datetime(nxt["snapshot_dt"], utc=True) - ordered["snapshot_dt"]).dt.total_seconds()
    valid = (
        gap.gt(0)
        & gap.le(float(max_gap_seconds))
        & (nxt["route_id"].astype(str) == ordered["route_id"].astype(str))
        & pd.to_numeric(nxt["occupancy_ordinal"], errors="coerce").notna()
    )
    transitions = ordered.loc[valid].copy()
    transitions["target_occupancy_ordinal"] = pd.to_numeric(nxt.loc[valid, "occupancy_ordinal"], errors="coerce").to_numpy(dtype=float)
    transitions["transition_gap_seconds"] = gap.loc[valid].to_numpy(dtype=float)
    return transitions.reset_index(drop=True)


def _base_features(df: pd.DataFrame) -> np.ndarray:
    snapshot_dt = pd.to_datetime(df["snapshot_dt"], utc=True)
    hour = snapshot_dt.dt.hour.to_numpy(dtype=np.float64) + snapshot_dt.dt.minute.to_numpy(dtype=np.float64) / 60.0
    pct = df["occupancy_pct_norm"].to_numpy(dtype=np.float64)
    has_pct = np.isfinite(pct).astype(np.float64)
    pct = np.nan_to_num(pct, nan=-1.0)
    stop_seq = pd.to_numeric(df["current_stop_sequence"], errors="coerce").to_numpy(dtype=np.float64)
    has_seq = np.isfinite(stop_seq).astype(np.float64)
    if np.isfinite(stop_seq).any():
        seq_scale = max(1.0, float(np.nanpercentile(stop_seq[np.isfinite(stop_seq)], 95)))
    else:
        seq_scale = 1.0
    speed = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=np.float64)
    has_speed = np.isfinite(speed).astype(np.float64)
    speed = np.nan_to_num(speed / 35.0, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack(
        [
            df["occupancy_ordinal"].to_numpy(dtype=np.float64) / 5.0,
            pct,
            has_pct,
            _city_norm(df["latitude"]),
            _city_norm(df["longitude"]),
            np.nan_to_num(stop_seq / seq_scale, nan=0.0, posinf=0.0, neginf=0.0),
            has_seq,
            speed,
            has_speed,
            np.nan_to_num(pd.to_numeric(df["current_status"], errors="coerce").to_numpy(dtype=np.float64) / 3.0, nan=0.0),
            np.nan_to_num(pd.to_numeric(df["direction_id"], errors="coerce").to_numpy(dtype=np.float64), nan=0.0),
            np.sin(hour / 24.0 * math.tau),
            np.cos(hour / 24.0 * math.tau),
            np.nan_to_num(df["transition_gap_seconds"].to_numpy(dtype=np.float64) / 7200.0, nan=0.0),
        ]
    ).astype(np.float64)


def _local_features(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    features = np.zeros((n, len(STATIC_LOCAL_FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return features
    occ = df["occupancy_ordinal"].to_numpy(dtype=np.float64)
    pct = df["occupancy_pct_norm"].to_numpy(dtype=np.float64)
    speed = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=np.float64)
    lat = df["latitude"].to_numpy(dtype=np.float64)
    lon = df["longitude"].to_numpy(dtype=np.float64)
    seq = pd.to_numeric(df["current_stop_sequence"], errors="coerce").to_numpy(dtype=np.float64)
    for _group, idx in df.groupby(["snapshot_dt", "route_id"], sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        m = len(idx_arr)
        if m <= 1:
            features[idx_arr, 0] = 1.0
            features[idx_arr, 1] = occ[idx_arr]
            features[idx_arr, 2] = 0.0
            features[idx_arr, 3] = occ[idx_arr]
            features[idx_arr, 4] = np.nan_to_num(pct[idx_arr], nan=-1.0)
            features[idx_arr, 5] = np.isfinite(pct[idx_arr]).astype(float)
            features[idx_arr, 6] = np.nan_to_num(speed[idx_arr] / 35.0, nan=0.0)
            features[idx_arr, 7] = 0.0
            features[idx_arr, 8] = occ[idx_arr]
            continue
        group_occ = occ[idx_arr]
        group_pct = pct[idx_arr]
        group_speed = speed[idx_arr]
        occ_sum = np.nansum(group_occ)
        occ_sumsq = np.nansum(group_occ**2)
        pct_sum = np.nansum(group_pct)
        pct_present = np.isfinite(group_pct)
        speed_sum = np.nansum(group_speed)
        speed_present = np.isfinite(group_speed)
        count_other = float(max(1, m - 1))

        group_lat = lat[idx_arr]
        group_lon = lon[idx_arr]
        distance = _haversine_km(group_lat[:, None], group_lon[:, None], group_lat[None, :], group_lon[None, :])
        np.fill_diagonal(distance, np.inf)
        nearest_pos = np.argmin(distance, axis=1)
        nearest_dist = distance[np.arange(m), nearest_pos]
        nearest_global = idx_arr[nearest_pos]

        group_seq = seq[idx_arr]
        for local_pos, global_idx in enumerate(idx_arr):
            other_occ_sum = occ_sum - occ[global_idx]
            other_occ_mean = other_occ_sum / count_other
            other_occ_var = max(0.0, (occ_sumsq - occ[global_idx] ** 2) / count_other - other_occ_mean**2)
            pct_den = max(1.0, float(pct_present.sum() - int(np.isfinite(pct[global_idx]))))
            speed_den = max(1.0, float(speed_present.sum() - int(np.isfinite(speed[global_idx]))))
            features[global_idx, 0] = float(m)
            features[global_idx, 1] = other_occ_mean / 5.0
            features[global_idx, 2] = math.sqrt(other_occ_var) / 5.0
            features[global_idx, 3] = np.nanmax(np.delete(group_occ, local_pos)) / 5.0
            features[global_idx, 4] = (pct_sum - (pct[global_idx] if np.isfinite(pct[global_idx]) else 0.0)) / pct_den
            features[global_idx, 5] = (pct_present.sum() - int(np.isfinite(pct[global_idx]))) / count_other
            features[global_idx, 6] = (speed_sum - (speed[global_idx] if np.isfinite(speed[global_idx]) else 0.0)) / speed_den / 35.0
            features[global_idx, 7] = 0.0 if not np.isfinite(nearest_dist[local_pos]) else nearest_dist[local_pos] / 10.0
            features[global_idx, 8] = occ[nearest_global[local_pos]] / 5.0
            features[global_idx, 9] = _safe_num(speed[nearest_global[local_pos]] / 35.0)

            front_global = None
            back_global = None
            if np.isfinite(seq[global_idx]) and np.isfinite(group_seq).sum() > 1:
                higher = np.where(group_seq > seq[global_idx])[0]
                lower = np.where(group_seq < seq[global_idx])[0]
                if higher.size:
                    front_global = idx_arr[higher[np.argmin(group_seq[higher] - seq[global_idx])]]
                if lower.size:
                    back_global = idx_arr[lower[np.argmin(seq[global_idx] - group_seq[lower])]]
            if front_global is None:
                front_global = nearest_global[local_pos]
            if back_global is None:
                back_global = nearest_global[local_pos]
            features[global_idx, 10] = max(0.0, _safe_num(seq[front_global] - seq[global_idx])) / 100.0
            features[global_idx, 11] = occ[front_global] / 5.0
            features[global_idx, 12] = _safe_num(speed[front_global] / 35.0)
            features[global_idx, 13] = max(0.0, _safe_num(seq[global_idx] - seq[back_global])) / 100.0
            features[global_idx, 14] = occ[back_global] / 5.0
            features[global_idx, 15] = _safe_num(speed[back_global] / 35.0)
            features[global_idx, 16] = (occ[front_global] - occ[back_global]) / 5.0
            features[global_idx, 17] = float(np.nanmean(np.delete(group_occ, local_pos) >= 3.0))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _load_datasets(args: argparse.Namespace) -> dict[str, SnapshotDataset]:
    root = _repo_root()
    data_dir = _resolve_path(root, args.data_dir)
    paths = {
        "nyc_mta_bus": data_dir / "nyc_mta_bus_vehicle_snapshots.parquet",
        "septa_bus": data_dir / "septa_bus_vehicle_snapshots.parquet",
        "mbta_live": data_dir / "mbta_live_vehicle_snapshots.parquet",
    }
    datasets: dict[str, SnapshotDataset] = {}
    for key, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"missing snapshot parquet: {path}")
        raw = _read_snapshot_file(path, key)
        transitions = _make_transitions(raw, max_gap_seconds=args.max_gap_seconds)
        if len(transitions) < args.min_transitions:
            raise RuntimeError(f"{key}: only {len(transitions)} transitions after filters")
        no_local = _base_features(transitions)
        local = np.concatenate([no_local, _local_features(transitions)], axis=1)
        datasets[key] = SnapshotDataset(
            key=key,
            label=str(transitions["source_label"].dropna().iloc[0]) if transitions["source_label"].notna().any() else key,
            frame=transitions,
            features_no_local=no_local,
            features_static_local=local,
            target=transitions["target_occupancy_ordinal"].to_numpy(dtype=np.float64),
            snapshot_times=pd.to_datetime(transitions["snapshot_dt"], utc=True).to_numpy(),
        )
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
    pred = _ridge_predict(_ridge_fit(x_train, y_train, ridge), x_eval)
    pred_clip = np.clip(pred, 0.0, 5.0)
    rounded = np.rint(pred_clip)
    err = pred_clip - eval_data.target
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "rounded_accuracy": float(np.mean(rounded == eval_data.target)),
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
            frame=data.frame.iloc[train_idx],
            features_no_local=data.features_no_local[train_idx],
            features_static_local=data.features_static_local[train_idx],
            target=data.target[train_idx],
            snapshot_times=data.snapshot_times[train_idx],
        )
        eval_data = SnapshotDataset(
            key=data.key,
            label=data.label,
            frame=data.frame.iloc[eval_idx],
            features_no_local=data.features_no_local[eval_idx],
            features_static_local=data.features_static_local[eval_idx],
            target=data.target[eval_idx],
            snapshot_times=data.snapshot_times[eval_idx],
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
    out = {}
    for method in ["no_local", "static_local", "selector_gated_static_local"]:
        mses = [item[method]["mse"] for item in results]
        out[method] = {
            "mean_mse": float(np.mean(mses)),
            "wins_vs_no_local": int(sum(item[method]["mse"] < item["no_local"]["mse"] for item in results)),
        }
    ratios = [item["static_vs_no_local"] for item in results if item["static_vs_no_local"] is not None]
    selector_ratios = [item["selector_vs_no_local"] for item in results if item["selector_vs_no_local"] is not None]
    out["static_local_mean_vs_no_local"] = float(np.mean(ratios)) if ratios else None
    out["selector_mean_vs_no_local"] = float(np.mean(selector_ratios)) if selector_ratios else None
    return out


def _write_summary_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# GTFS-RT Local Snapshot Validation",
        "",
        "Task: one-step occupancy ordinal prediction from synchronized AVL/occupancy snapshots.",
        "",
        "This is not a full APC/control validation because GTFS-RT occupancy is crowding status/percentage and has no holding action.",
        "",
        "## Dataset",
        "",
        "| Feed | Transitions | Snapshots | Routes |",
        "|---|---:|---:|---:|",
    ]
    for item in result["datasets"].values():
        lines.append(f"| {item['label']} | {item['transitions']} | {item['snapshots']} | {item['routes']} |")
    lines.extend(["", "## Cross-City Results", "", "| Target | No Local MSE | Static Local MSE | Static/No | Selector | Selector/No |", "|---|---:|---:|---:|---|---:|"])
    for item in result["cross_city"]:
        lines.append(
            f"| {item['target']} | {item['no_local']['mse']:.4f} | {item['static_local']['mse']:.4f} | "
            f"{item['static_vs_no_local']:.4f} | {item['selector']['selected']} | {item['selector_vs_no_local']:.4f} |"
        )
    summary = result["summary"]["cross_city"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Static local mean vs no-local: `{summary['static_local_mean_vs_no_local']:.4f}`.",
            f"- Selector-gated mean vs no-local: `{summary['selector_mean_vs_no_local']:.4f}`.",
            f"- Static local wins vs no-local: `{summary['static_local']['wins_vs_no_local']}/3`.",
            f"- Selector-gated wins vs no-local: `{summary['selector_gated_static_local']['wins_vs_no_local']}/3`.",
            "",
            "Interpretation: if static/selector ratios are not clearly below 1.0, use these feeds as evidence that public GTFS-RT occupancy snapshots are useful for data inspection but insufficient for a strong local-graph performance claim; proceed to controlled SUMO APC/AVL snapshot generation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    datasets = _load_datasets(args)
    cross_city = _cross_city_eval(datasets, args)
    temporal = _temporal_eval(datasets, args)
    dataset_summary = {
        key: {
            "label": data.label,
            "transitions": int(len(data.target)),
            "snapshots": int(data.frame["snapshot_ts"].nunique()),
            "routes": int(data.frame["route_id"].nunique()),
            "vehicles": int(data.frame["vehicle_id"].nunique()),
        }
        for key, data in datasets.items()
    }
    result = {
        "ok": True,
        "validation_level": "gtfsrt_local_snapshot_cross_city",
        "definition": "Cross-city one-step occupancy ordinal prediction from public synchronized AVL/occupancy snapshots.",
        "data_dir": str(_resolve_path(root, args.data_dir)),
        "datasets": dataset_summary,
        "feature_names": {
            "no_local": list(NO_LOCAL_FEATURE_NAMES),
            "static_local": list(NO_LOCAL_FEATURE_NAMES) + list(STATIC_LOCAL_FEATURE_NAMES),
        },
        "cross_city": cross_city,
        "temporal_single_city": temporal,
        "summary": {
            "cross_city": _summarize_cross_city(cross_city),
            "temporal_static_mean_vs_no_local": float(np.mean([item["static_vs_no_local"] for item in temporal])),
        },
        "limitations": [
            "GTFS-RT occupancy is crowding status/percentage, not exact APC board/alight counts.",
            "No holding/control action is observed.",
            "NYC and SEPTA archive samples are hourly-spaced snapshots; MBTA live samples are short-interval self-archive snapshots.",
        ],
    }
    out_path = _resolve_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_summary_md(result, out_path.with_suffix(".md"))
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("H2Oplus/downloads/open_transit/gtfsrt_occupancy_snapshots"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/gtfsrt_local_snapshot_validation.json"))
    parser.add_argument("--max-gap-seconds", type=float, default=7200.0)
    parser.add_argument("--min-transitions", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--selector-min-win-rate", type=float, default=1.0)
    parser.add_argument("--selector-min-improvement", type=float, default=0.0)
    parser.add_argument("--temporal-train-fraction", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
