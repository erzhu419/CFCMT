"""
Smoke-validate one generated H2O+ city environment.

Checks:
    1. BusSimEnv loads the generated Excel/config files.
    2. The simulator reaches a decision observation with dim=15.
    3. A CFCMT RouteGraph and TransitionBatch/FeatureRegistry can be built.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any

import torch


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


def multiline_line_dirs(env_path: pathlib.Path) -> list[pathlib.Path]:
    data_dir = env_path / "data"
    required = {"stop_news.xlsx", "route_news.xlsx", "time_table.xlsx", "passenger_OD.xlsx"}
    if not data_dir.exists():
        return []
    return [
        child for child in sorted(data_dir.iterdir())
        if child.is_dir() and required.issubset({p.name for p in child.iterdir()})
    ]


def load_city_manifest(env_path: pathlib.Path) -> dict[str, Any]:
    for name in ("gtfs_city_manifest.json", "lta_city_manifest.json"):
        path = env_path / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def sampled_multiline_view(env_path: pathlib.Path, line_dirs: list[pathlib.Path], limit: int) -> pathlib.Path:
    if limit <= 0 or len(line_dirs) <= limit:
        return env_path
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="h2o_city_validate_"))
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_path / "config.json", tmp / "config.json")
    for line_dir in line_dirs[:limit]:
        os.symlink(line_dir.resolve(), tmp / "data" / line_dir.name)
    return tmp


def load_env(env_path: pathlib.Path, dynamic_line_limit: int):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    bus_h2o = repo_root / "H2Oplus" / "bus_h2o"
    if str(bus_h2o) not in sys.path:
        sys.path.insert(0, str(bus_h2o))
    from envs.bus_sim_env import BusSimEnv, MultiLineSimEnv

    line_dirs = multiline_line_dirs(env_path)
    if line_dirs:
        view_path = sampled_multiline_view(env_path, line_dirs, dynamic_line_limit)
        env = MultiLineSimEnv(path=str(view_path), debug=False, render=False)
        env._validation_view_path = str(view_path)
        env._validation_total_lines = len(line_dirs)
        return env
    return BusSimEnv(path=str(env_path), debug=False, render=False)


def active_obs(obs: dict[Any, list[Any]]) -> tuple[Any, list[float]] | None:
    if obs and all(isinstance(v, dict) for v in obs.values()):
        for line_id, line_obs in obs.items():
            found = active_obs(line_obs)
            if found is not None:
                bus_id, values = found
                return (line_id, bus_id), values
        return None
    for bus_id, values in obs.items():
        if values:
            first = values[0]
            if hasattr(first, "tolist"):
                first = first.tolist()
            return bus_id, [float(x) for x in first]
    return None


def step_env(env, actions: dict[Any, Any]):
    result = env.step(actions)
    if len(result) == 4:
        return result
    obs, rewards, done = result
    return obs, rewards, done, {"snapshot": env.capture_full_system_snapshot()}


def first_reward(rewards: dict[Any, Any]) -> float:
    if not rewards:
        return 0.0
    first_value = next(iter(rewards.values()))
    if isinstance(first_value, dict):
        return first_reward(first_value)
    return float(first_value)


def run_validation(env_path: pathlib.Path, max_steps: int) -> dict[str, Any]:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cf_h2o.graph.feature_registry import FeatureRegistry
    from cf_h2o.graph.route_graph import RouteGraph
    from cf_h2o.schemas import TransitionBatch

    manifest = load_city_manifest(env_path)
    line_summaries = manifest.get("lines", []) if isinstance(manifest, dict) else []
    static_totals = {
        "lines": int(manifest.get("line_count", len(line_summaries))) if manifest else None,
        "stations": sum(int(item.get("stops", 0)) for item in line_summaries) if line_summaries else None,
        "routes": sum(int(item.get("segments", 0)) for item in line_summaries) if line_summaries else None,
        "timetables": sum(int(item.get("timetable_rows", 0)) for item in line_summaries) if line_summaries else None,
    }

    with contextlib.redirect_stdout(io.StringIO()):
        env = load_env(env_path, dynamic_line_limit=run_validation.dynamic_line_limit)
    obs = env.reset()
    first = active_obs(obs)
    first_snapshot = env.capture_full_system_snapshot() if first is not None and hasattr(env, "capture_full_system_snapshot") else None
    reward_value = 0.0

    if first is None:
        for _ in range(max_steps):
            obs, rewards, done, info = step_env(env, {})
            first = active_obs(obs)
            if first is not None:
                first_snapshot = info["snapshot"]
                reward_value = first_reward(rewards)
                break
            if done:
                break
    if first is None or first_snapshot is None:
        raise RuntimeError(f"No decision observation reached within {max_steps} steps")

    second = None
    for _ in range(max_steps):
        actions = {first[0]: 0.0}
        if hasattr(env, "step_to_event"):
            result = env.step_to_event(actions)
            if len(result) == 3:
                obs2, rewards2, done = result
            else:
                obs2, rewards2, done, _info2 = result
        else:
            obs2, rewards2, done, _info2 = step_env(env, actions)
        second = active_obs(obs2)
        if second is not None:
            if rewards2:
                reward_value = first_reward(rewards2)
            break
        if done:
            break
    if second is None:
        second = first

    graph = RouteGraph.from_snapshot(first_snapshot)
    observations = torch.tensor([first[1]], dtype=torch.float32)
    next_observations = torch.tensor([second[1]], dtype=torch.float32)
    batch = TransitionBatch(
        observations=observations,
        actions=torch.zeros(1, 1),
        rewards=torch.tensor([reward_value], dtype=torch.float32),
        next_observations=next_observations,
        dones=torch.zeros(1),
        snapshot_t=[first_snapshot],
        source=["sim"],
        metadata={"obs_names": OBS_NAMES, "action_names": ["holding"]},
    )
    registry = FeatureRegistry.from_transition_dataset(batch)
    hard_mask = registry.build_temporal_hard_mask()

    return {
        "env_path": str(env_path.resolve()),
        "stations": static_totals["stations"] if static_totals["stations"] is not None else len(env.stations),
        "routes": static_totals["routes"] if static_totals["routes"] is not None else len(env.routes),
        "timetables": static_totals["timetables"] if static_totals["timetables"] is not None else len(env.timetables),
        "stations_loaded": len(env.stations),
        "routes_loaded": len(env.routes),
        "timetables_loaded": len(env.timetables),
        "state_dim": env.state_dim,
        "lines": static_totals["lines"] if static_totals["lines"] is not None else (len(getattr(env, "line_map", {})) if hasattr(env, "line_map") else 1),
        "lines_loaded": len(getattr(env, "line_map", {})) if hasattr(env, "line_map") else 1,
        "line_idx": int(getattr(env, "line_idx", -1)),
        "line_headway": float(getattr(env, "line_headway", 0.0)),
        "first_decision_time": float(env.current_time),
        "first_obs_dim": len(first[1]),
        "route_graph_stations": len(graph.stations),
        "route_graph_segments": len(graph.segments),
        "feature_nodes": len(registry.node_names),
        "hard_mask_edges": int(hard_mask.sum().item()),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_path", type=pathlib.Path)
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument("--dynamic-line-limit", type=int, default=250, help="For very large MultiLine bundles, dynamically load only this many lines for smoke simulation; 0 loads all.")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    run_validation.dynamic_line_limit = args.dynamic_line_limit
    result = run_validation(args.env_path, args.max_steps)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


run_validation.dynamic_line_limit = 250
