"""Austin CapMetro real APC/AVL validation for CFCMT vs H2O+-style transfer.

This is an external validation slice: models are trained on the existing
static-derived source city bundles and evaluated on public CapMetro APC rows
that include real stop events, boardings/alightings, dwell times, loads, and
vehicle positions. The observed APC data has no counterfactual holding action,
so the validation evaluates the action=0 passive dynamics only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_performance_validation import (
    OUTPUT_NAMES,
    _read_json,
    _repo_root,
    _resolve_path,
    _reward,
    _stable_unit,
    _uncalibrated_transition,
)
from cf_h2o.eval.paper_experiment_suite import (
    ResidualStats,
    build_city_stats,
    _family_sse,
    _fit_cfcmt,
    _fit_h2o,
    _merge_stats,
    _merge_stats_weighted,
    _method_metrics_from_sse,
    _source_similarity_weights,
)


DEFAULT_FEW_SHOT_BUDGETS = [0.0, 0.01, 0.05, 0.10, 0.25]


APC_DATASETS: dict[str, dict[str, str]] = {
    "austin_apc_2021_h2": {
        "id": "im6q-3pc9",
        "label": "CapMetro APC Raw July 2021 - December 2021",
        "source": "https://catalog.data.gov/dataset/apc-raw-july-2021-december-2021",
    },
    "austin_apc_2016_aug": {
        "id": "j7xj-n68t",
        "label": "CapMetro APC Raw August 2016",
        "source": "https://catalog.data.gov/dataset/capmetro-apc-raw-august-2016",
    },
}

APC_FIELDS = [
    "act_trip_start_time",
    "actual_sequence",
    "apc_date_time",
    "block_id",
    "bs_id",
    "close_date_time",
    "current_route_id",
    "day_type_vs",
    "direction_code_id",
    "dwell_time",
    "ext_trip_id",
    "import_error",
    "import_trip_error",
    "max_load",
    "offs",
    "ons",
    "open_date_time",
    "quality_indicator",
    "raw_max_load",
    "raw_off",
    "raw_on",
    "rev_distance",
    "rev_seconds",
    "route_id",
    "sched_time",
    "seg_arr_time",
    "seg_dep_time",
    "start_trip_time",
    "time_id",
    "tp_id",
    "transit_date_time",
    "variation",
    "veh_lat",
    "veh_long",
    "vehicle_id",
]

VALID_WHERE = (
    "bs_id != '0' AND route_id != '0' AND ext_trip_id != '0' "
    "AND veh_lat != '0' AND veh_long != '0' "
    "AND import_error = '0' AND import_trip_error = '0'"
)


def _url_json(url: str, params: dict[str, Any], timeout: int = 120) -> Any:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _socrata_resource_url(dataset_id: str, fmt: str = "csv") -> str:
    return f"https://data.austintexas.gov/resource/{dataset_id}.{fmt}"


def _count_socrata_rows(dataset_id: str, where: str) -> int:
    data = _url_json(_socrata_resource_url(dataset_id, "json"), {"$select": "count(*)", "$where": where})
    return int(data[0]["count"]) if data else 0


def _download_socrata_csv(
    dataset_id: str,
    out_path: Path,
    *,
    where: str,
    fields: list[str],
    page_size: int,
    max_rows: int,
    force: bool,
    workers: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        available = _count_socrata_rows(dataset_id, where)
        return {
            "path": str(out_path),
            "rows_available": available,
            "rows_downloaded": None,
            "skipped": True,
            "source_url": _socrata_resource_url(dataset_id),
        }

    total_available = _count_socrata_rows(dataset_id, where)
    target = total_available if max_rows <= 0 else min(total_available, int(max_rows))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if force and tmp.exists():
        tmp.unlink()

    if tmp.exists():
        with tmp.open("rb") as fh:
            rows_written = max(sum(1 for _ in fh) - 1, 0)
        header = rows_written == 0
        print(f"[apc] resuming {tmp}: {rows_written}/{target} valid rows", flush=True)
    else:
        rows_written = 0
        header = True
    base = _socrata_resource_url(dataset_id, "csv")
    select = ",".join(fields)
    offsets = list(range(rows_written, target, int(page_size)))
    if workers <= 1 or len(offsets) <= 1:
        for offset in offsets:
            _completed_offset, chunk = _download_socrata_chunk(
                base,
                select,
                where,
                offset,
                min(int(page_size), target - offset),
            )
            if chunk.empty:
                break
            chunk.to_csv(tmp, mode="a", index=False, header=header)
            rows_written += int(len(chunk))
            header = False
            print(f"[apc] downloaded {rows_written}/{target} valid rows", flush=True)
            if len(chunk) < min(int(page_size), target - offset):
                break
    else:
        pending: dict[int, futures.Future[tuple[int, pd.DataFrame]]] = {}
        completed: dict[int, pd.DataFrame] = {}
        next_submit = 0
        next_write = rows_written
        max_pending = max(1, int(workers) * 2)
        with futures.ThreadPoolExecutor(max_workers=int(workers)) as executor:
            while next_submit < len(offsets) or pending or next_write in completed:
                while next_submit < len(offsets) and len(pending) < max_pending:
                    offset = offsets[next_submit]
                    limit = min(int(page_size), target - offset)
                    pending[offset] = executor.submit(_download_socrata_chunk, base, select, where, offset, limit)
                    next_submit += 1
                done = [offset for offset, fut in pending.items() if fut.done()]
                if not done:
                    time.sleep(0.1)
                    continue
                for offset in done:
                    completed_offset, chunk = pending.pop(offset).result()
                    completed[completed_offset] = chunk
                while next_write in completed:
                    chunk = completed.pop(next_write)
                    if chunk.empty:
                        next_submit = len(offsets)
                        pending.clear()
                        break
                    chunk.to_csv(tmp, mode="a", index=False, header=header)
                    rows_written += int(len(chunk))
                    next_write += int(len(chunk))
                    header = False
                    print(f"[apc] downloaded {rows_written}/{target} valid rows", flush=True)
                    if len(chunk) < int(page_size):
                        next_submit = len(offsets)
                        pending.clear()
                        break
    tmp.replace(out_path)
    return {
        "path": str(out_path),
        "rows_available": total_available,
        "rows_downloaded": rows_written,
        "skipped": False,
        "source_url": _socrata_resource_url(dataset_id),
    }


def _download_socrata_chunk(base: str, select: str, where: str, offset: int, limit: int) -> tuple[int, pd.DataFrame]:
    params = {
        "$select": select,
        "$where": where,
        "$order": "apc_date_time, route_id, ext_trip_id, actual_sequence",
        "$limit": int(limit),
        "$offset": int(offset),
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            return int(offset), pd.read_csv(url, dtype=str)
        except Exception as exc:
            last_error = exc
            wait = min(30.0, 2.0 * attempt)
            print(
                f"[apc] chunk offset={offset} failed ({type(exc).__name__}); retry {attempt}/5 in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to download Socrata chunk at offset {offset}") from last_error


def _parse_apc_time(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.replace(r"\.0+$", "", regex=True).str.slice(0, 14)
    return pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")


def _seconds_since_midnight(ts: pd.Series) -> np.ndarray:
    return (
        ts.dt.hour.fillna(0).astype(float) * 3600.0
        + ts.dt.minute.fillna(0).astype(float) * 60.0
        + ts.dt.second.fillna(0).astype(float)
    ).to_numpy(dtype=np.float64)


def _direction_binary(values: pd.Series) -> np.ndarray:
    codes = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    unique = {value: idx for idx, value in enumerate(sorted(codes.unique()))}
    return codes.map(unique).fillna(0).astype(int).mod(2).to_numpy(dtype=np.float64)


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _prepare_apc_rows(csv_path: Path, max_rows: int) -> pd.DataFrame:
    read_kwargs = {"dtype": str}
    if max_rows > 0:
        read_kwargs["nrows"] = int(max_rows)
    df = pd.read_csv(csv_path, **read_kwargs)
    if df.empty:
        raise RuntimeError(f"No APC rows in {csv_path}")

    df["apc_ts"] = _parse_apc_time(df["apc_date_time"])
    df["open_ts"] = _parse_apc_time(df["open_date_time"])
    df["close_ts"] = _parse_apc_time(df["close_date_time"])
    df["sched_ts"] = _parse_apc_time(df["sched_time"])
    df["seg_arr_ts"] = _parse_apc_time(df["seg_arr_time"])
    df["seg_dep_ts"] = _parse_apc_time(df["seg_dep_time"])
    for col in (
        "actual_sequence",
        "dwell_time",
        "max_load",
        "offs",
        "ons",
        "raw_max_load",
        "raw_off",
        "raw_on",
        "rev_distance",
        "rev_seconds",
        "route_id",
        "vehicle_id",
    ):
        df[col] = _numeric(df, col)
    df = df[
        df["apc_ts"].notna()
        & (df["route_id"] > 0)
        & (df["vehicle_id"] > 0)
        & (df["actual_sequence"] >= 0)
        & (df["dwell_time"].between(0, 600))
        & (df["rev_seconds"].between(1, 3600))
        & (df["rev_distance"] > 0)
    ].copy()
    if df.empty:
        raise RuntimeError("APC filters removed all rows")

    df["event_seconds"] = df["apc_ts"].astype("int64") / 1e9
    df["hour"] = df["apc_ts"].dt.hour.fillna(0).astype(int)
    df["direction_binary"] = _direction_binary(df["direction_code_id"])
    df["line_key"] = (
        "austin_apc_"
        + df["route_id"].astype(int).astype(str)
        + "_D"
        + df["direction_binary"].astype(int).astype(str)
    )
    df = df.sort_values(["route_id", "direction_code_id", "bs_id", "apc_ts"])
    stop_group = df.groupby(["route_id", "direction_code_id", "bs_id"], sort=False)["event_seconds"]
    df["backward_headway"] = stop_group.diff()
    df["forward_headway"] = -stop_group.diff(-1)
    df["backward_headway"] = df["backward_headway"].clip(lower=10.0, upper=7200.0)
    df["forward_headway"] = df["forward_headway"].clip(lower=10.0, upper=7200.0)
    df = df[df["backward_headway"].notna() & df["forward_headway"].notna()].copy()

    df["target_headway"] = df.groupby(["route_id", "direction_code_id", "hour"])["backward_headway"].transform("median")
    df["target_headway"] = df["target_headway"].clip(lower=60.0, upper=7200.0)
    trip_group_cols = ["ext_trip_id", "vehicle_id", "route_id", "direction_code_id"]
    df = df.sort_values(trip_group_cols + ["actual_sequence", "apc_ts"])
    trip_max_seq = df.groupby(trip_group_cols, sort=False)["actual_sequence"].transform("max").clip(lower=1.0)
    df["station_fraction"] = (df["actual_sequence"] / trip_max_seq).clip(lower=0.0, upper=1.0)
    for col in [
        "forward_headway",
        "backward_headway",
        "ons",
        "dwell_time",
        "target_headway",
        "rev_distance",
        "rev_seconds",
    ]:
        df[f"next_{col}"] = df.groupby(trip_group_cols, sort=False)[col].shift(-1)
    df["next_apc_ts"] = df.groupby(trip_group_cols, sort=False)["apc_ts"].shift(-1)
    df["next_event_seconds"] = df.groupby(trip_group_cols, sort=False)["event_seconds"].shift(-1)
    df["actual_travel_time"] = (df["next_event_seconds"] - df["event_seconds"]).clip(lower=1.0, upper=7200.0)
    scheduled = (df["seg_dep_ts"] - df["seg_arr_ts"]).dt.total_seconds()
    df["scheduled_travel_time"] = scheduled.where(scheduled.between(1, 7200), df["actual_travel_time"])
    df = df[df["next_apc_ts"].notna()].copy()
    return df


def _apc_speed_mps(distance_miles: pd.Series, seconds: pd.Series) -> np.ndarray:
    speed = pd.to_numeric(distance_miles, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    denom = pd.to_numeric(seconds, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    speed = speed * 1609.344 / np.maximum(denom, 1.0)
    return np.clip(speed, 0.8, 25.0)


def _line_apc_stats_worker(payload: tuple[str, pd.DataFrame]) -> ResidualStats:
    line_key, df = payload
    line_codes = np.full(len(df), _stable_unit(str(line_key)), dtype=np.float64)
    action = np.zeros(len(df), dtype=np.float64)
    fwd = df["forward_headway"].to_numpy(dtype=np.float64)
    bwd = df["backward_headway"].to_numpy(dtype=np.float64)
    waiting = df["ons"].clip(lower=0.0).to_numpy(dtype=np.float64)
    target = df["target_headway"].to_numpy(dtype=np.float64)
    stop_duration = df["dwell_time"].clip(lower=0.0, upper=600.0).to_numpy(dtype=np.float64)
    speed = _apc_speed_mps(df["rev_distance"], df["rev_seconds"])
    travel_time = df["scheduled_travel_time"].to_numpy(dtype=np.float64)
    hour = df["hour"].to_numpy(dtype=np.int64)
    sim_time = _seconds_since_midnight(df["apc_ts"])
    bus_id = np.mod(df["vehicle_id"].to_numpy(dtype=np.float64), 400.0) / 400.0
    direction = df["direction_binary"].to_numpy(dtype=np.float64)
    obs_action = np.column_stack(
        [
            line_codes,
            bus_id,
            df["station_fraction"].to_numpy(dtype=np.float64),
            np.clip(hour.astype(np.float64) / 23.0, 0.0, 1.0),
            direction,
            fwd,
            bwd,
            waiting,
            target,
            stop_duration,
            sim_time,
            fwd - bwd,
            fwd,
            bwd,
            speed,
            action,
        ]
    ).astype(np.float32)

    sim_y = np.empty((len(df), len(OUTPUT_NAMES)), dtype=np.float64)
    for h in sorted(np.unique(hour)):
        mask = hour == h
        sim_y[mask, :] = np.column_stack(
            _uncalibrated_transition(
                fwd[mask],
                bwd[mask],
                waiting[mask],
                target[mask],
                speed[mask],
                travel_time[mask],
                action[mask],
                int(h),
            )
        )

    real_fwd = df["next_forward_headway"].to_numpy(dtype=np.float64)
    real_bwd = df["next_backward_headway"].to_numpy(dtype=np.float64)
    real_waiting = df["next_ons"].clip(lower=0.0).to_numpy(dtype=np.float64)
    real_stop = df["next_dwell_time"].clip(lower=0.0, upper=600.0).to_numpy(dtype=np.float64)
    real_speed = _apc_speed_mps(df["next_rev_distance"], df["next_rev_seconds"])
    real_gap = real_fwd - real_bwd
    real_reward = _reward(real_fwd, real_bwd, real_waiting, target, action)
    real_y = np.column_stack(
        [real_fwd, real_bwd, real_waiting, real_stop, real_gap, real_fwd, real_bwd, real_speed, real_reward]
    ).astype(np.float64)
    valid = np.isfinite(obs_action).all(axis=1) & np.isfinite(sim_y).all(axis=1) & np.isfinite(real_y).all(axis=1)
    one = ResidualStats.zeros(f"austin_capmetro_real_apc::{line_key}", "Austin / CapMetro real APC")
    if np.any(valid):
        one.add_arrays(str(line_key), obs_action[valid], real_y[valid] - sim_y[valid])
    return one


def build_real_apc_stats(
    csv_path: Path,
    *,
    max_rows: int = 0,
    workers: int = 1,
) -> tuple[ResidualStats, list[ResidualStats], dict[str, Any]]:
    t0 = time.time()
    df = _prepare_apc_rows(csv_path, max_rows)
    line_keys = sorted(df["line_key"].unique())
    aggregate = ResidualStats.zeros("austin_capmetro_real_apc", "Austin / CapMetro real APC")
    line_stats: list[ResidualStats] = []
    groups = ((str(line_key), group.copy()) for line_key, group in df.groupby("line_key", sort=False))
    if workers <= 1:
        for item in groups:
            one = _line_apc_stats_worker(item)
            aggregate.merge(one)
            line_stats.append(one)
    else:
        group_iter = iter(groups)
        pending: set[futures.Future[ResidualStats]] = set()
        submitted = 0
        completed = 0
        max_pending = max(1, int(workers) * 2)
        with futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
            while completed < len(line_keys):
                while submitted < len(line_keys) and len(pending) < max_pending:
                    pending.add(executor.submit(_line_apc_stats_worker, next(group_iter)))
                    submitted += 1
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    one = fut.result()
                    aggregate.merge(one)
                    line_stats.append(one)
                    completed += 1
                    if completed % 25 == 0 or completed == len(line_keys):
                        print(f"[apc] processed {completed}/{len(line_keys)} line groups", flush=True)

    summary = {
        "csv_path": str(csv_path),
        "rows_loaded_after_filters": int(len(df)),
        "transitions": int(aggregate.n),
        "line_count": int(len(line_keys)),
        "route_count": int(df["route_id"].nunique()),
        "stop_count": int(df["bs_id"].nunique()),
        "trip_count": int(df[["ext_trip_id", "vehicle_id"]].drop_duplicates().shape[0]),
        "date_min": df["apc_ts"].min().isoformat() if len(df) else None,
        "date_max": df["apc_ts"].max().isoformat() if len(df) else None,
        "elapsed_sec": time.time() - t0,
        "notes": "Observed APC stop-event validation; action fixed to 0 because no counterfactual holding action is observed. The passive no-correction baseline is a state-informed one-step transition using APC-derived current state, not a schedule-only free-running simulator.",
    }
    line_stats.sort(key=lambda item: item.line_keys[0] if item.line_keys else item.key)
    return aggregate, line_stats, summary


def _static_city_stats_worker(payload: tuple[str, dict[str, Any], str, dict[str, Any]]) -> tuple[str, ResidualStats, dict[str, Any]]:
    key, spec, root_text, arg_values = payload
    worker_args = SimpleNamespace(**arg_values)
    aggregate, _line_stats, summary = build_city_stats(key, spec, Path(root_text), worker_args)
    return key, aggregate, summary


def _fit_static_sources(
    config: dict[str, Any],
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, ResidualStats], dict[str, Any]]:
    city_stats: dict[str, ResidualStats] = {}
    sanity: dict[str, Any] = {}
    items = list(config["generated_envs"].items())
    arg_values = {
        "max_lines_per_city": args.max_lines_per_city,
        "actions": args.actions,
        "progress_every": args.progress_every,
    }
    if args.workers <= 1 or len(items) <= 1:
        for key, spec in items:
            print(f"[static] building stats for {key}", flush=True)
            aggregate, _line_stats, summary = build_city_stats(key, spec, root, args)
            city_stats[key] = aggregate
            sanity[key] = summary
            print(f"[static] {key}: lines={aggregate.lines_seen} transitions={aggregate.n}", flush=True)
    else:
        payloads = [(key, spec, str(root), arg_values) for key, spec in items]
        with futures.ProcessPoolExecutor(max_workers=min(int(args.workers), len(payloads))) as executor:
            for key, aggregate, summary in executor.map(_static_city_stats_worker, payloads):
                city_stats[key] = aggregate
                sanity[key] = summary
                print(f"[static] {key}: lines={aggregate.lines_seen} transitions={aggregate.n}", flush=True)
    return city_stats, sanity


def _split_target_lines(
    lines: list[ResidualStats],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[ResidualStats], list[ResidualStats], bool]:
    ordered = sorted(lines, key=lambda item: item.line_keys[0] if item.line_keys else item.key)
    if not ordered:
        return [], [], False
    if fraction <= 0.0:
        return [], ordered, False
    if fraction >= 1.0:
        return ordered, ordered, True
    hashes = np.array(
        [
            _stable_unit(f"austin-real-apc-calibration:{seed}:{item.line_keys[0] if item.line_keys else item.key}")
            for item in ordered
        ],
        dtype=np.float64,
    )
    mask = hashes < float(fraction)
    if not np.any(mask):
        mask[int(np.argmin(hashes))] = True
    if np.all(mask):
        mask[int(np.argmax(hashes))] = False
    calibration = [item for item, selected in zip(ordered, mask) if selected]
    evaluation = [item for item, selected in zip(ordered, mask) if not selected]
    return calibration, evaluation, False


def _fit_safe_h2o(stats: ResidualStats, ridge: float) -> np.ndarray | None:
    return _fit_h2o(stats, ridge) if stats.n > 0 else None


def _fit_safe_cfcmt(stats: ResidualStats, ridge: float) -> dict[str, np.ndarray] | None:
    return _fit_cfcmt(stats, ridge) if stats.n > 0 else None


def _linear_gate(stats_xtx: np.ndarray, stats_xty: np.ndarray, beta: np.ndarray) -> np.ndarray:
    b = np.asarray(beta, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    pred_sq = np.einsum("ik,ij,jk->k", b, stats_xtx, b)
    pred_y = np.einsum("ij,ij->j", b, stats_xty)
    alpha = np.divide(pred_y, pred_sq, out=np.zeros_like(pred_y, dtype=np.float64), where=pred_sq > 1e-12)
    return np.clip(alpha, 0.0, 1.0)


def _linear_bias(stats_xtx: np.ndarray, stats_xty: np.ndarray, beta: np.ndarray, n: int) -> np.ndarray:
    b = np.array(beta, dtype=np.float64, copy=True)
    if b.ndim == 1:
        b = b[:, None]
    if n <= 0:
        return b
    feature_sum = stats_xtx[0, :]
    pred_sum = feature_sum @ b
    residual_sum = stats_xty[0, :] - pred_sum
    b[0, :] += residual_sum / float(n)
    return b


def _gate_h2o_beta(calibration: ResidualStats, beta: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    alpha = _linear_gate(calibration.h2o.xtx, calibration.h2o.xty, beta)
    gated = np.asarray(beta, dtype=np.float64) * alpha[None, :]
    return gated, {name: float(value) for name, value in zip(OUTPUT_NAMES, alpha)}


def _bias_h2o_beta(calibration: ResidualStats, beta: np.ndarray) -> np.ndarray:
    return _linear_bias(calibration.h2o.xtx, calibration.h2o.xty, beta, calibration.n)


def _gate_cfcmt_beta(
    calibration: ResidualStats,
    beta: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    gated: dict[str, np.ndarray] = {}
    alpha_by_output: dict[str, float] = {}
    for output_name in OUTPUT_NAMES:
        stats = calibration.cfcmt[output_name]
        coef = np.asarray(beta[output_name], dtype=np.float64).reshape(-1, 1)
        alpha = float(_linear_gate(stats.xtx, stats.xty, coef)[0])
        gated[output_name] = beta[output_name] * alpha
        alpha_by_output[output_name] = alpha
    return gated, alpha_by_output


def _bias_cfcmt_beta(calibration: ResidualStats, beta: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for output_name in OUTPUT_NAMES:
        stats = calibration.cfcmt[output_name]
        out[output_name] = _linear_bias(stats.xtx, stats.xty, beta[output_name], calibration.n).reshape(-1)
    return out


def _metrics_from_optional_sse(sse: np.ndarray | None, n: int) -> dict[str, Any] | None:
    return _method_metrics_from_sse(sse, n) if sse is not None else None


def _evaluate_external(
    external: ResidualStats,
    city_stats: dict[str, ResidualStats],
    sanity: dict[str, Any],
    *,
    ridge: float,
    temperature: float,
    floor: float,
) -> dict[str, Any]:
    target = "austin_capmetro_all"
    sources = [key for key in city_stats if key != target]
    weights = _source_similarity_weights(sanity, target, sources, temperature=temperature, floor=floor)
    train_unweighted = _merge_stats("source_unweighted", "source", [city_stats[key] for key in sources])
    train_weighted = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, weights)
    h2o_beta = _fit_h2o(train_unweighted, ridge)
    h2o_weighted_beta = _fit_h2o(train_weighted, ridge)
    cfcmt_beta = _fit_cfcmt(train_unweighted, ridge)
    cfcmt_weighted_beta = _fit_cfcmt(train_weighted, ridge)

    uncal_sse = external.h2o.sse(None)
    h2o_sse = external.h2o.sse(h2o_beta)
    h2o_weighted_sse = external.h2o.sse(h2o_weighted_beta)
    cfcmt_sse = _family_sse(external.cfcmt, cfcmt_beta)
    cfcmt_weighted_sse = _family_sse(external.cfcmt, cfcmt_weighted_beta)
    passive_metrics = _method_metrics_from_sse(uncal_sse, external.n)
    metrics = {
        "passive_no_correction": passive_metrics,
        "uncalibrated": passive_metrics,
        "h2oplus_dense": _method_metrics_from_sse(h2o_sse, external.n),
        "h2oplus_similarity_weighted": _method_metrics_from_sse(h2o_weighted_sse, external.n),
        "cfcmt_mechanism": _method_metrics_from_sse(cfcmt_sse, external.n),
        "cfcmt_similarity_weighted": _method_metrics_from_sse(cfcmt_weighted_sse, external.n),
    }
    h2o = metrics["h2oplus_dense"]["total_mse"]
    passive = metrics["passive_no_correction"]["total_mse"]
    cfcmt = metrics["cfcmt_mechanism"]["total_mse"]
    weighted = metrics["cfcmt_similarity_weighted"]["total_mse"]
    return {
        "target_env": "austin_capmetro_real_apc",
        "target_city": "Austin / CapMetro real APC",
        "source_envs": sources,
        "source_weights": weights,
        "target_transitions": external.n,
        "target_lines_seen": external.lines_seen,
        "metrics": metrics,
        "comparisons": {
            "h2oplus_vs_passive_total_mse_ratio": h2o / passive if passive else None,
            "cfcmt_vs_passive_total_mse_ratio": cfcmt / passive if passive else None,
            "cfcmt_similarity_weighted_vs_passive_total_mse_ratio": weighted / passive if passive else None,
            "h2oplus_vs_uncalibrated_total_mse_ratio": h2o / passive if passive else None,
            "cfcmt_vs_uncalibrated_total_mse_ratio": cfcmt / passive if passive else None,
            "cfcmt_similarity_weighted_vs_uncalibrated_total_mse_ratio": weighted / passive if passive else None,
            "cfcmt_vs_h2oplus_total_mse_ratio": cfcmt / h2o if h2o else None,
            "cfcmt_similarity_weighted_vs_h2oplus_total_mse_ratio": weighted / h2o if h2o else None,
            "cfcmt_beats_h2oplus": bool(cfcmt < h2o),
            "cfcmt_similarity_weighted_beats_h2oplus": bool(weighted < h2o),
        },
    }


def _evaluate_external_few_shot(
    external_lines: list[ResidualStats],
    city_stats: dict[str, ResidualStats],
    sanity: dict[str, Any],
    *,
    ridge: float,
    budgets: list[float],
    seed: int,
    temperature: float,
    floor: float,
) -> dict[str, Any]:
    target = "austin_capmetro_all"
    sources = [key for key in city_stats if key != target]
    weights = _source_similarity_weights(sanity, target, sources, temperature=temperature, floor=floor)
    source_stats = _merge_stats("source_unweighted", "source", [city_stats[key] for key in sources])
    weighted_source_stats = _merge_stats_weighted("source_similarity_weighted", "source", city_stats, weights)
    source_h2o_beta = _fit_h2o(source_stats, ridge)
    source_weighted_h2o_beta = _fit_h2o(weighted_source_stats, ridge)
    source_cfcmt_beta = _fit_cfcmt(source_stats, ridge)
    source_weighted_cfcmt_beta = _fit_cfcmt(weighted_source_stats, ridge)

    rows = []
    for budget in budgets:
        calibration_lines, evaluation_lines, in_sample_oracle = _split_target_lines(
            external_lines,
            fraction=float(budget),
            seed=seed,
        )
        calibration_stats = _merge_stats(
            f"austin_real_apc::calibration_{budget:g}",
            "Austin / CapMetro real APC",
            calibration_lines,
        )
        evaluation_stats = (
            _merge_stats("austin_real_apc::evaluation_in_sample", "Austin / CapMetro real APC", external_lines)
            if in_sample_oracle
            else _merge_stats(f"austin_real_apc::evaluation_{budget:g}", "Austin / CapMetro real APC", evaluation_lines)
        )
        if evaluation_stats.n <= 0:
            continue

        source_plus_target = _merge_stats(
            f"austin_real_apc::source_plus_target_{budget:g}",
            "source + Austin / CapMetro real APC",
            [source_stats, calibration_stats],
        )
        weighted_source_plus_target = _merge_stats(
            f"austin_real_apc::weighted_source_plus_target_{budget:g}",
            "weighted source + Austin / CapMetro real APC",
            [weighted_source_stats, calibration_stats],
        )
        h2o_source_plus_target = _fit_safe_h2o(source_plus_target, ridge)
        cfcmt_source_plus_target = _fit_safe_cfcmt(source_plus_target, ridge)
        h2o_weighted_source_plus_target = _fit_safe_h2o(weighted_source_plus_target, ridge)
        cfcmt_weighted_source_plus_target = _fit_safe_cfcmt(weighted_source_plus_target, ridge)
        h2o_target_only = _fit_safe_h2o(calibration_stats, ridge)
        cfcmt_target_only = _fit_safe_cfcmt(calibration_stats, ridge)

        if calibration_stats.n > 0:
            gated_h2o_beta, h2o_gate = _gate_h2o_beta(calibration_stats, source_weighted_h2o_beta)
            gated_cfcmt_beta, cfcmt_gate = _gate_cfcmt_beta(calibration_stats, source_weighted_cfcmt_beta)
            bias_h2o_beta = _bias_h2o_beta(calibration_stats, source_weighted_h2o_beta)
            bias_cfcmt_beta = _bias_cfcmt_beta(calibration_stats, source_weighted_cfcmt_beta)
        else:
            gated_h2o_beta, h2o_gate = source_weighted_h2o_beta, {name: 1.0 for name in OUTPUT_NAMES}
            gated_cfcmt_beta, cfcmt_gate = source_weighted_cfcmt_beta, {name: 1.0 for name in OUTPUT_NAMES}
            bias_h2o_beta = source_weighted_h2o_beta
            bias_cfcmt_beta = source_weighted_cfcmt_beta

        passive_sse = evaluation_stats.h2o.sse(None)
        h2o_source_sse = evaluation_stats.h2o.sse(source_h2o_beta)
        h2o_weighted_source_sse = evaluation_stats.h2o.sse(source_weighted_h2o_beta)
        cfcmt_source_sse = _family_sse(evaluation_stats.cfcmt, source_cfcmt_beta)
        cfcmt_weighted_source_sse = _family_sse(evaluation_stats.cfcmt, source_weighted_cfcmt_beta)
        h2o_source_plus_target_sse = (
            evaluation_stats.h2o.sse(h2o_source_plus_target) if h2o_source_plus_target is not None else None
        )
        cfcmt_source_plus_target_sse = (
            _family_sse(evaluation_stats.cfcmt, cfcmt_source_plus_target)
            if cfcmt_source_plus_target is not None
            else None
        )
        h2o_weighted_source_plus_target_sse = (
            evaluation_stats.h2o.sse(h2o_weighted_source_plus_target)
            if h2o_weighted_source_plus_target is not None
            else None
        )
        cfcmt_weighted_source_plus_target_sse = (
            _family_sse(evaluation_stats.cfcmt, cfcmt_weighted_source_plus_target)
            if cfcmt_weighted_source_plus_target is not None
            else None
        )
        h2o_target_only_sse = evaluation_stats.h2o.sse(h2o_target_only) if h2o_target_only is not None else None
        cfcmt_target_only_sse = (
            _family_sse(evaluation_stats.cfcmt, cfcmt_target_only) if cfcmt_target_only is not None else None
        )
        h2o_gate_sse = evaluation_stats.h2o.sse(gated_h2o_beta)
        cfcmt_gate_sse = _family_sse(evaluation_stats.cfcmt, gated_cfcmt_beta)
        h2o_bias_sse = evaluation_stats.h2o.sse(bias_h2o_beta)
        cfcmt_bias_sse = _family_sse(evaluation_stats.cfcmt, bias_cfcmt_beta)

        metrics = {
            "passive_no_correction": _method_metrics_from_sse(passive_sse, evaluation_stats.n),
            "h2oplus_source_only": _method_metrics_from_sse(h2o_source_sse, evaluation_stats.n),
            "h2oplus_weighted_source_only": _method_metrics_from_sse(h2o_weighted_source_sse, evaluation_stats.n),
            "cfcmt_source_only": _method_metrics_from_sse(cfcmt_source_sse, evaluation_stats.n),
            "cfcmt_weighted_source_only": _method_metrics_from_sse(cfcmt_weighted_source_sse, evaluation_stats.n),
            "h2oplus_source_plus_target_budget": _metrics_from_optional_sse(
                h2o_source_plus_target_sse,
                evaluation_stats.n,
            ),
            "cfcmt_source_plus_target_budget": _metrics_from_optional_sse(
                cfcmt_source_plus_target_sse,
                evaluation_stats.n,
            ),
            "h2oplus_weighted_source_plus_target_budget": _metrics_from_optional_sse(
                h2o_weighted_source_plus_target_sse,
                evaluation_stats.n,
            ),
            "cfcmt_weighted_source_plus_target_budget": _metrics_from_optional_sse(
                cfcmt_weighted_source_plus_target_sse,
                evaluation_stats.n,
            ),
            "h2oplus_target_only_budget": _metrics_from_optional_sse(h2o_target_only_sse, evaluation_stats.n),
            "cfcmt_target_only_budget": _metrics_from_optional_sse(cfcmt_target_only_sse, evaluation_stats.n),
            "h2oplus_weighted_source_residual_gate": _method_metrics_from_sse(h2o_gate_sse, evaluation_stats.n),
            "cfcmt_weighted_source_residual_gate": _method_metrics_from_sse(cfcmt_gate_sse, evaluation_stats.n),
            "h2oplus_weighted_source_bias_adapter": _method_metrics_from_sse(h2o_bias_sse, evaluation_stats.n),
            "cfcmt_weighted_source_bias_adapter": _method_metrics_from_sse(cfcmt_bias_sse, evaluation_stats.n),
        }
        passive = metrics["passive_no_correction"]["total_mse"]
        h2o_source = metrics["h2oplus_source_only"]["total_mse"]
        weighted_source = metrics["cfcmt_weighted_source_only"]["total_mse"]
        weighted_gate = metrics["cfcmt_weighted_source_residual_gate"]["total_mse"]
        weighted_bias = metrics["cfcmt_weighted_source_bias_adapter"]["total_mse"]
        weighted_target = metrics["cfcmt_weighted_source_plus_target_budget"]["total_mse"]
        row = {
            "target_line_budget_fraction": float(budget),
            "calibration_lines": len(calibration_lines),
            "evaluation_lines": len(external_lines) if in_sample_oracle else len(evaluation_lines),
            "calibration_transitions": calibration_stats.n,
            "evaluation_transitions": evaluation_stats.n,
            "in_sample_oracle": bool(in_sample_oracle),
            "metrics": metrics,
            "adaptation_parameters": {
                "h2oplus_weighted_source_residual_gate": h2o_gate,
                "cfcmt_weighted_source_residual_gate": cfcmt_gate,
            },
            "comparisons": {
                "cfcmt_weighted_source_only_vs_h2oplus_source_only_ratio": (
                    weighted_source / h2o_source if h2o_source else None
                ),
                "cfcmt_weighted_source_only_vs_passive_ratio": weighted_source / passive if passive else None,
                "cfcmt_weighted_source_plus_target_vs_passive_ratio": weighted_target / passive if passive else None,
                "cfcmt_weighted_source_residual_gate_vs_passive_ratio": weighted_gate / passive if passive else None,
                "cfcmt_weighted_source_bias_adapter_vs_passive_ratio": weighted_bias / passive if passive else None,
                "cfcmt_weighted_source_plus_target_beats_passive": bool(weighted_target < passive),
                "cfcmt_weighted_source_residual_gate_beats_passive": bool(weighted_gate < passive),
                "cfcmt_weighted_source_bias_adapter_beats_passive": bool(weighted_bias < passive),
            },
        }
        rows.append(row)

    best_key = None
    if rows:
        candidate_keys = [
            "cfcmt_weighted_source_plus_target_budget",
            "cfcmt_weighted_source_residual_gate",
            "cfcmt_weighted_source_bias_adapter",
            "cfcmt_target_only_budget",
        ]
        best_by_budget = []
        for row in rows:
            available = [
                (key, row["metrics"][key]["total_mse"])
                for key in candidate_keys
                if row["metrics"].get(key) is not None
            ]
            method, value = min(available, key=lambda item: item[1])
            best_by_budget.append({**row, "best_method": method, "best_total_mse": value})
        best_overall = min(best_by_budget, key=lambda item: item["best_total_mse"])
        best_key = best_overall["best_method"]
        passive_values = [row["metrics"]["passive_no_correction"]["total_mse"] for row in rows]
        summary = {
            "budgets": [row["target_line_budget_fraction"] for row in rows],
            "best_budget": best_overall["target_line_budget_fraction"],
            "best_method": best_key,
            "best_vs_passive_ratio": (
                best_overall["best_total_mse"] / best_overall["metrics"]["passive_no_correction"]["total_mse"]
                if best_overall["metrics"]["passive_no_correction"]["total_mse"]
                else None
            ),
            "budgets_where_gate_beats_passive": [
                row["target_line_budget_fraction"]
                for row in rows
                if row["comparisons"]["cfcmt_weighted_source_residual_gate_beats_passive"]
            ],
            "budgets_where_bias_beats_passive": [
                row["target_line_budget_fraction"]
                for row in rows
                if row["comparisons"]["cfcmt_weighted_source_bias_adapter_beats_passive"]
            ],
            "mean_passive_total_mse": float(np.mean(passive_values)),
        }
    else:
        summary = {"budgets": [], "best_method": best_key}

    return {
        "ok": True,
        "experiment": "austin_real_apc_target_budget_sweep",
        "definition": "Target APC line groups are split into calibration and evaluation sets. Source-only models are trained only on non-Austin static-derived source cities; few-shot variants add or gate residual correction using calibration APC residual labels and evaluate on held-out Austin APC line groups.",
        "source_envs": sources,
        "source_weights": weights,
        "rows": rows,
        "summary": summary,
    }


def _parse_float_list(text: str) -> list[float]:
    return [float(item) for item in str(text).replace(",", " ").split() if item.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    preset = APC_DATASETS[args.dataset]
    raw_dir = _resolve_path(root, args.raw_dir)
    suffix = "full" if args.max_apc_rows <= 0 else f"sample_{args.max_apc_rows}"
    csv_path = raw_dir / f"{args.dataset}_valid_{suffix}.csv"
    download = _download_socrata_csv(
        preset["id"],
        csv_path,
        where=VALID_WHERE,
        fields=APC_FIELDS,
        page_size=args.socrata_page_size,
        max_rows=args.max_apc_rows,
        force=args.force_download,
        workers=args.workers,
    )
    external_stats, external_line_stats, external_summary = build_real_apc_stats(csv_path, max_rows=0, workers=args.workers)
    city_stats, sanity = _fit_static_sources(config, root, args)
    validation = _evaluate_external(
        external_stats,
        city_stats,
        sanity,
        ridge=args.ridge,
        temperature=args.source_weight_temperature,
        floor=args.source_weight_floor,
    )
    few_shot = _evaluate_external_few_shot(
        external_line_stats,
        city_stats,
        sanity,
        ridge=args.ridge,
        budgets=args.few_shot_budgets,
        seed=args.seed,
        temperature=args.source_weight_temperature,
        floor=args.source_weight_floor,
    )
    result = {
        "ok": True,
        "validation_level": "austin_real_apc_external_validation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(_resolve_path(root, args.config)),
        "dataset": {
            **preset,
            "download": download,
            "valid_where": VALID_WHERE,
        },
        "static_training": {
            "max_lines_per_city": args.max_lines_per_city,
            "workers": args.workers,
            "source_envs": validation["source_envs"],
        },
        "real_apc_summary": external_summary,
        "validation": validation,
        "few_shot_validation": few_shot,
    }
    out_path = _resolve_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation["comparisons"], indent=2), flush=True)
    print(json.dumps(few_shot["summary"], indent=2), flush=True)
    print(f"[out] {out_path}", flush=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--dataset", choices=sorted(APC_DATASETS), default="austin_apc_2021_h2")
    parser.add_argument("--raw-dir", type=Path, default=Path("H2Oplus/downloads/open_transit/austin_capmetro/apc"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/austin_real_apc_validation.json"))
    parser.add_argument("--max-apc-rows", type=int, default=0, help="0 means all valid APC rows; >0 for smoke")
    parser.add_argument("--socrata-page-size", type=int, default=50000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-lines-per-city", type=int, default=0, help="0 means all static lines; >0 for smoke")
    parser.add_argument("--actions", type=float, nargs="+", default=[0.0, 30.0])
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--few-shot-budgets", type=_parse_float_list, default=DEFAULT_FEW_SHOT_BUDGETS)
    parser.add_argument("--source-weight-temperature", type=float, default=1.0)
    parser.add_argument("--source-weight-floor", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
