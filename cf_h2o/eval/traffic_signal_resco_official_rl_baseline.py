"""Run official RESCO RL baselines and summarize their logs.

This adapter intentionally keeps official RESCO training outside the CFCMT
phase-transfer evaluator.  RESCO agents use their own state, reward, action, and
logging definitions, so the results are reported as a leaderboard-style
cross-check rather than mixed directly into the CFCMT phase-MPC table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import EXTENDED_SCENARIOS
from cf_h2o.eval.traffic_signal_transfer_feasibility import _format_table


DEFAULT_RESC0_REPO = Path("H2Oplus/downloads/traffic_signal_resco/repo")
DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_resco_official_rl_baseline.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_resco_official_rl_baseline.md")
DEFAULT_LOG_DIR = Path("cf_h2o/results/resco_official_rl")
DEFAULT_CACHE_DIR = Path("cf_h2o/results/resco_official_rl_cache")
DEFAULT_ALGORITHMS = ("IDQN", "MPLight")
DEFAULT_SCENARIOS = ("cologne1",)
SUMO_TOOLS = Path("/usr/share/sumo/tools")
SCENARIO_ROUTE_OVERRIDES = {
    "grid4x4": "grid4x4.rou.xml",
    "arterial4x4": "arterial4x4.rou.xml",
}
SCENARIO_ZIPPED_ROUTE_CANDIDATES = {
    "grid4x4": ("grid4x4_1.rou.xml",),
    "arterial4x4": ("arterial4x4_1.rou.xml",),
}


@dataclass(frozen=True)
class OfficialRunSpec:
    scenario: str
    algorithm: str
    episodes: int
    testing: int
    seed: int | None


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_skip_runs(raw: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("skip runs must use ALGORITHM:SCENARIO format")
        algorithm, scenario = (part.strip() for part in item.split(":", 1))
        if not algorithm or not scenario:
            raise argparse.ArgumentTypeError("skip runs must use ALGORITHM:SCENARIO format")
        pairs.append((algorithm, scenario))
    return tuple(pairs)


def _json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path_for_spec(
    cache_dir: Path,
    spec: OfficialRunSpec,
    *,
    extra_args: tuple[str, ...],
    keep_episode_logs: bool,
) -> Path:
    key = {
        "scenario": spec.scenario,
        "algorithm": spec.algorithm,
        "episodes": spec.episodes,
        "testing": spec.testing,
        "seed": spec.seed,
        "extra_args": list(extra_args),
        "keep_episode_logs": keep_episode_logs,
    }
    stem = f"{spec.scenario}_{spec.algorithm}_e{spec.episodes}_t{spec.testing}_s{spec.seed}"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return cache_dir / f"{safe_stem}_{_json_hash(key)}.json"


def load_cached_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if not row.get("ok"):
        return None
    row["cached"] = True
    return row


def write_cached_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _latest_json(log_dir: Path, since: float) -> Path | None:
    candidates = [
        path
        for path in log_dir.glob("*.json")
        if path.is_file() and path.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _values_for_metric(result: dict[str, Any], metric: str) -> list[float]:
    by_hash = result.get(metric, {})
    if not isinstance(by_hash, dict) or not by_hash:
        return []
    values: list[float] = []
    for seq in by_hash.values():
        values.extend(float(value) for value in seq)
    return values


def _tail(values: list[float], n: int) -> list[float]:
    if n <= 0:
        return list(values)
    return list(values[-n:])


def _ensure_route_override_file(bench_dir: Path, scenario: str) -> None:
    route_name = SCENARIO_ROUTE_OVERRIDES.get(scenario)
    if route_name is None:
        return
    env_dir = bench_dir / "environments" / scenario
    dst = env_dir / route_name
    if dst.exists():
        return

    for candidate in SCENARIO_ZIPPED_ROUTE_CANDIDATES.get(scenario, ()):
        src = env_dir / candidate
        if src.exists():
            shutil.copyfile(src, dst)
            return

    zip_path = env_dir / f"{scenario}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing route override {dst} and archive {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for candidate in SCENARIO_ZIPPED_ROUTE_CANDIDATES.get(scenario, ()):
            if candidate in names:
                with zf.open(candidate) as src_fh, dst.open("wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh)
                return
    raise FileNotFoundError(f"{zip_path} contains no route candidate for {scenario}")


def summarize_official_result(
    *,
    spec: OfficialRunSpec,
    result_path: Path | None,
    ok: bool,
    returncode: int,
    elapsed_sec: float,
    stdout_tail: str,
    stderr_tail: str,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ok": bool(ok),
        "scenario": spec.scenario,
        "algorithm": spec.algorithm,
        "episodes": int(spec.episodes),
        "testing": int(spec.testing),
        "seed": spec.seed,
        "returncode": int(returncode),
        "elapsed_sec": float(elapsed_sec),
        "cached": False,
        "result_path": str(result_path) if result_path is not None else None,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if error:
        row["error"] = error
    if not ok or result_path is None or not result_path.exists():
        return row

    result = json.loads(result_path.read_text(encoding="utf-8"))
    parsed_any = False
    for metric in ("timeLoss", "duration", "waitingTime", "queue_lengths", "max_queues", "rewards", "vehicles"):
        values = _values_for_metric(result, metric)
        parsed_any = parsed_any or bool(values)
        test_values = _tail(values, spec.testing)
        row[f"{metric}_episodes"] = values
        row[f"test_mean_{metric}"] = float(np.mean(test_values)) if test_values else float("nan")
        row[f"test_last_{metric}"] = float(test_values[-1]) if test_values else float("nan")
        row[f"all_mean_{metric}"] = float(np.mean(values)) if values else float("nan")
    if not parsed_any:
        row["ok"] = False
        row["error"] = "official RESCO result JSON contained no parsed episode metrics"
    return row


def run_official_resco(
    *,
    repo_root: Path,
    log_dir: Path,
    spec: OfficialRunSpec,
    timeout_sec: float | None,
    extra_args: tuple[str, ...],
    keep_episode_logs: bool,
) -> dict[str, Any]:
    bench_dir = repo_root / "resco_benchmark"
    if not (bench_dir / "main.py").exists():
        raise FileNotFoundError(f"missing official RESCO main.py under {bench_dir}")

    env = os.environ.copy()
    pythonpath = [
        str(repo_root.resolve()),
        str(SUMO_TOOLS),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
    env.setdefault("SUMO_HOME", "/usr/share/sumo")
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        f"@{spec.scenario}",
        f"@{spec.algorithm}",
        f"episodes:{spec.episodes}",
        f"testing:{spec.testing}",
        "save_console_log:False",
        f"delete_episode_logs:{str(not keep_episode_logs)}",
        f"log_dir:{log_dir.resolve()}",
        "libsumo:True",
        "gui:False",
        "save_model:False",
    ]
    if spec.seed is not None:
        cmd.append(f"seed:{spec.seed}")
    if spec.scenario in SCENARIO_ROUTE_OVERRIDES and not any(arg.startswith("route:") for arg in extra_args):
        _ensure_route_override_file(bench_dir, spec.scenario)
        cmd.append(f"route:{SCENARIO_ROUTE_OVERRIDES[spec.scenario]}")
    cmd.extend(extra_args)

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=bench_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.time() - started
    result_path = _latest_json(log_dir, started)
    stdout_tail = "\n".join(proc.stdout.splitlines()[-12:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-12:])
    return summarize_official_result(
        spec=spec,
        result_path=result_path,
        ok=proc.returncode == 0 and result_path is not None,
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=None if proc.returncode == 0 else "official RESCO subprocess failed",
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        groups.setdefault((row["algorithm"], row["scenario"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (algorithm, scenario), items in sorted(groups.items()):
        out.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "runs": len(items),
                "test_timeLoss": float(np.mean([item["test_mean_timeLoss"] for item in items])),
                "test_duration": float(np.mean([item["test_mean_duration"] for item in items])),
                "test_waitingTime": float(np.mean([item["test_mean_waitingTime"] for item in items])),
                "test_queue_lengths": float(np.mean([item["test_mean_queue_lengths"] for item in items])),
                "test_max_queues": float(np.mean([item["test_mean_max_queues"] for item in items])),
                "elapsed_sec": float(np.mean([item["elapsed_sec"] for item in items])),
            }
        )
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary_rows = [
        {
            "algorithm": row["algorithm"],
            "scenario": row["scenario"],
            "runs": row["runs"],
            "test_timeLoss": row["test_timeLoss"],
            "test_duration": row["test_duration"],
            "test_waitingTime": row["test_waitingTime"],
            "test_queue_lengths": row["test_queue_lengths"],
            "test_max_queues": row["test_max_queues"],
            "elapsed_sec": row["elapsed_sec"],
        }
        for row in result["aggregate"]
    ]
    run_rows = [
        {
            "ok": row["ok"],
            "algorithm": row["algorithm"],
            "scenario": row["scenario"],
            "seed": row["seed"],
            "test_timeLoss": row.get("test_mean_timeLoss", float("nan")),
            "test_duration": row.get("test_mean_duration", float("nan")),
            "test_waitingTime": row.get("test_mean_waitingTime", float("nan")),
            "test_queue_lengths": row.get("test_mean_queue_lengths", float("nan")),
            "elapsed_sec": row["elapsed_sec"],
            "cached": row.get("cached", False),
            "result_path": row.get("result_path"),
        }
        for row in result["runs"]
    ]
    lines = [
        "# Official RESCO RL Baseline Adapter",
        "",
        "Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.",
        "",
        "These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.",
        "",
        "## Aggregate",
        "",
        _format_table(
            summary_rows,
            [
                "algorithm",
                "scenario",
                "runs",
                "test_timeLoss",
                "test_duration",
                "test_waitingTime",
                "test_queue_lengths",
                "test_max_queues",
                "elapsed_sec",
            ],
        ),
        "",
        "## Runs",
        "",
        _format_table(
            run_rows,
            [
                "ok",
                "algorithm",
                "scenario",
                "seed",
                "test_timeLoss",
                "test_duration",
                "test_waitingTime",
                "test_queue_lengths",
                "elapsed_sec",
                "cached",
                "result_path",
            ],
        ),
        "",
        "## Notes",
        "",
        "- `testing` episodes are summarized after official RESCO switches the agent into testing mode.",
        "- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.",
        "- Use larger episode budgets before citing these numbers as trained RL baselines.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    repo_root: Path,
    log_dir: Path,
    scenarios: tuple[str, ...],
    algorithms: tuple[str, ...],
    episodes: int,
    testing: int,
    seeds: tuple[int, ...],
    timeout_sec: float | None,
    extra_args: tuple[str, ...],
    keep_episode_logs: bool,
    cache_dir: Path | None = None,
    resume: bool = True,
    skip_runs: tuple[tuple[str, str], ...] = (),
    workers: int = 1,
) -> dict[str, Any]:
    specs: list[OfficialRunSpec] = []
    skip_set = set(skip_runs)
    seed_values: tuple[int | None, ...] = seeds if seeds else (None,)
    for scenario in scenarios:
        for algorithm in algorithms:
            if (algorithm, scenario) in skip_set:
                continue
            for seed in seed_values:
                specs.append(
                    OfficialRunSpec(
                        scenario=scenario,
                        algorithm=algorithm,
                        episodes=episodes,
                        testing=testing,
                        seed=seed,
                    )
                )

    def run_one(spec: OfficialRunSpec) -> dict[str, Any]:
        cache_path = (
            cache_path_for_spec(cache_dir, spec, extra_args=extra_args, keep_episode_logs=keep_episode_logs)
            if cache_dir is not None
            else None
        )
        if resume and cache_path is not None:
            cached = load_cached_row(cache_path)
            if cached is not None:
                return cached
        try:
            row = run_official_resco(
                repo_root=repo_root,
                log_dir=log_dir,
                spec=spec,
                timeout_sec=timeout_sec,
                extra_args=extra_args,
                keep_episode_logs=keep_episode_logs,
            )
        except subprocess.TimeoutExpired as exc:
            row = summarize_official_result(
                spec=spec,
                result_path=None,
                ok=False,
                returncode=124,
                elapsed_sec=float(timeout_sec or 0.0),
                stdout_tail=str(exc.stdout or ""),
                stderr_tail=str(exc.stderr or ""),
                error="timeout",
            )
        except Exception as exc:
            row = summarize_official_result(
                spec=spec,
                result_path=None,
                ok=False,
                returncode=1,
                elapsed_sec=0.0,
                stdout_tail="",
                stderr_tail="",
                error=f"{type(exc).__name__}: {exc}",
            )
        if cache_path is not None:
            write_cached_row(cache_path, row)
        return row

    runs: list[dict[str, Any]] = []
    if workers <= 1 or len(specs) <= 1:
        runs = [run_one(spec) for spec in specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_spec = {pool.submit(run_one, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(future_to_spec):
                try:
                    runs.append(future.result())
                except Exception as exc:
                    spec = future_to_spec[future]
                    runs.append(
                        summarize_official_result(
                            spec=spec,
                            result_path=None,
                            ok=False,
                            returncode=1,
                            elapsed_sec=0.0,
                            stdout_tail="",
                            stderr_tail="",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
    return {
        "experiment": "traffic_signal_resco_official_rl_baseline",
        "setting": {
            "repo_root": str(repo_root),
            "log_dir": str(log_dir),
            "scenarios": list(scenarios),
            "algorithms": list(algorithms),
            "episodes": int(episodes),
            "testing": int(testing),
            "seeds": list(seeds),
            "timeout_sec": timeout_sec,
            "extra_args": list(extra_args),
            "keep_episode_logs": bool(keep_episode_logs),
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "resume": bool(resume),
            "skipped_runs": [{"algorithm": algorithm, "scenario": scenario} for algorithm, scenario in skip_runs],
            "workers": int(workers),
            "protocol_note": "official RESCO state/reward/action/logging; not same-protocol CFCMT phase-MPC",
        },
        "aggregate": aggregate_rows(runs),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_RESC0_REPO)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--scenarios", type=_parse_csv, default=())
    parser.add_argument("--scenario-set", choices=("smoke", "extended"), default="smoke")
    parser.add_argument("--algorithms", type=_parse_csv, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--testing", type=int, default=1)
    parser.add_argument("--seeds", type=_parse_int_csv, default=())
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--extra-args", type=_parse_csv, default=())
    parser.add_argument(
        "--skip-runs",
        type=_parse_skip_runs,
        default=(),
        help="comma-separated ALGORITHM:SCENARIO pairs to skip, e.g. MPLight:cologne8,MPLight:ingolstadt21",
    )
    parser.add_argument("--keep-episode-logs", dest="keep_episode_logs", action="store_true")
    parser.add_argument("--delete-episode-logs", dest="keep_episode_logs", action="store_false")
    parser.set_defaults(keep_episode_logs=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    scenarios = args.scenarios
    if not scenarios:
        scenarios = EXTENDED_SCENARIOS if args.scenario_set == "extended" else DEFAULT_SCENARIOS
    result = run_experiment(
        repo_root=args.repo_root,
        log_dir=args.log_dir,
        scenarios=scenarios,
        algorithms=args.algorithms,
        episodes=args.episodes,
        testing=args.testing,
        seeds=args.seeds,
        timeout_sec=args.timeout_sec,
        extra_args=args.extra_args,
        keep_episode_logs=args.keep_episode_logs,
        cache_dir=args.cache_dir,
        resume=args.resume,
        skip_runs=args.skip_runs,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    failures = [row for row in result["runs"] if not row.get("ok")]
    print(f"runs={len(result['runs'])} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
