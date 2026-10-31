"""Helpers for using the raw, uncalibrated copied H2O+ bus simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gc
import importlib
import sys
import tracemalloc


@dataclass(frozen=True)
class BusEnvProfile:
    path: str
    data_dir: str
    timetables: int
    routes: int
    stations: int
    state_dim: int
    line_idx: int
    line_headway: float


@dataclass(frozen=True)
class UncalibratedBusEnvSmokeResult:
    raw_profile: BusEnvProfile
    calibrated_profile: BusEnvProfile | None
    raw_cache_entries: int
    cache_growth_after_repeats: int
    traced_growth_bytes: int
    repeat: int


def make_uncalibrated_bus_env(
    *,
    repo_root: str | Path | None = None,
    bus_h2o_root: str | Path | None = None,
    debug: bool = False,
    render: bool = False,
):
    """Instantiate BusSimEnv from ``H2Oplus/bus_h2o`` raw data, not calibrated_env."""

    root = _resolve_bus_h2o_root(repo_root=repo_root, bus_h2o_root=bus_h2o_root)
    _validate_uncalibrated_path(root)
    BusSimEnv, _ = _load_bus_sim_env(root)
    return BusSimEnv(path=str(root), debug=debug, render=render)


def profile_uncalibrated_and_calibrated_envs(config: dict[str, Any] | None = None) -> tuple[BusEnvProfile, BusEnvProfile]:
    """Instantiate raw and calibrated envs once and return lightweight profiles."""

    config = dict(config or {})
    root = _resolve_bus_h2o_root(repo_root=config.get("repo_root"), bus_h2o_root=config.get("bus_h2o_root"))
    _validate_uncalibrated_path(root)
    calibrated = root / "calibrated_env"
    BusSimEnv, _ = _load_bus_sim_env(root)
    raw_env = BusSimEnv(path=str(root), debug=False, render=False)
    calibrated_env = BusSimEnv(path=str(calibrated), debug=False, render=False)
    try:
        return _profile(raw_env), _profile(calibrated_env)
    finally:
        del raw_env
        del calibrated_env
        gc.collect()


def run_uncalibrated_bus_env_smoke(config: dict[str, Any] | None = None) -> UncalibratedBusEnvSmokeResult:
    """Check raw BusSimEnv selection and bounded cache growth across repeats."""

    config = dict(config or {})
    repeat = max(1, int(config.get("repeat", 3)))
    root = _resolve_bus_h2o_root(repo_root=config.get("repo_root"), bus_h2o_root=config.get("bus_h2o_root"))
    _validate_uncalibrated_path(root)
    calibrated = root / "calibrated_env"
    BusSimEnv, env_bus = _load_bus_sim_env(root)

    raw_data_dir = str((root / "data").resolve())
    calibrated_data_dir = str((calibrated / "data").resolve())
    original_cache = dict(getattr(env_bus, "_DATA_CACHE", {}))
    env_bus._DATA_CACHE.pop(raw_data_dir, None)
    env_bus._DATA_CACHE.pop(calibrated_data_dir, None)
    cache_size_before = len(env_bus._DATA_CACHE)

    tracemalloc.start()
    raw_profile = None
    current_after_first = None
    current_after_last = None
    try:
        for idx in range(repeat):
            env = BusSimEnv(path=str(root), debug=False, render=False)
            raw_profile = _profile(env)
            del env
            gc.collect()
            current = tracemalloc.get_traced_memory()[0]
            if idx == 0:
                current_after_first = current
            current_after_last = current

        calibrated_profile = None
        if bool(config.get("compare_calibrated", True)):
            calibrated_env = BusSimEnv(path=str(calibrated), debug=False, render=False)
            calibrated_profile = _profile(calibrated_env)
            del calibrated_env
            gc.collect()

        raw_cache_entries = int(raw_data_dir in env_bus._DATA_CACHE)
        cache_growth = len(env_bus._DATA_CACHE) - cache_size_before
        traced_growth = int(max(0, (current_after_last or 0) - (current_after_first or 0)))
        if raw_profile is None:
            raise RuntimeError("raw BusSimEnv profile was not collected")
        return UncalibratedBusEnvSmokeResult(
            raw_profile=raw_profile,
            calibrated_profile=calibrated_profile,
            raw_cache_entries=raw_cache_entries,
            cache_growth_after_repeats=cache_growth,
            traced_growth_bytes=traced_growth,
            repeat=repeat,
        )
    finally:
        tracemalloc.stop()
        env_bus._DATA_CACHE.clear()
        env_bus._DATA_CACHE.update(original_cache)
        gc.collect()


def _resolve_bus_h2o_root(
    *,
    repo_root: str | Path | None,
    bus_h2o_root: str | Path | None,
) -> Path:
    if bus_h2o_root is not None:
        return Path(bus_h2o_root).expanduser().resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return (Path(repo_root).expanduser().resolve() / "H2Oplus" / "bus_h2o").resolve()


def _validate_uncalibrated_path(path: Path) -> None:
    if path.name == "calibrated_env":
        raise ValueError("uncalibrated BusSimEnv must not point at calibrated_env")
    required = [path / "config.json", path / "data" / "time_table.xlsx", path / "data" / "route_news.xlsx"]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"uncalibrated BusSimEnv path is missing required files: {missing}")


def _load_bus_sim_env(bus_h2o_root: Path):
    root = str(bus_h2o_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("envs.bus_sim_env")
    sim_module = importlib.import_module("sim_core.sim")
    return module.BusSimEnv, sim_module.env_bus


def _profile(env) -> BusEnvProfile:
    return BusEnvProfile(
        path=str(Path(env.path).resolve()),
        data_dir=str((Path(env.path) / "data").resolve()),
        timetables=len(env.timetables),
        routes=len(env.routes),
        stations=len(env.stations),
        state_dim=int(env.state_dim),
        line_idx=int(getattr(env, "line_idx", -1)),
        line_headway=float(getattr(env, "line_headway", 0.0)),
    )
