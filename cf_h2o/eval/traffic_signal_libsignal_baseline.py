"""Run LibSignal SUMO baselines and summarize stdout metrics.

LibSignal is kept as a leaderboard-style traffic-signal-control baseline.  Its
agents use LibSignal's own state, reward, training loop, and logger, so these
numbers should not be mixed into the same-protocol CFCMT phase-MPC table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.traffic_signal_transfer_feasibility import _format_table


DEFAULT_LIBSIGNAL_REPO = Path("H2Oplus/downloads/traffic_signal_libsignal/repo")
DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_libsignal_baseline.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_libsignal_baseline.md")
DEFAULT_CACHE_DIR = Path("cf_h2o/results/libsignal_baseline_cache")
DEFAULT_RESC0_REPO = Path("H2Oplus/downloads/traffic_signal_resco/repo")
DEFAULT_AGENTS = ("fixedtime", "dqn")
DEFAULT_NETWORKS = ("sumo1x1",)
SUMO_TOOLS = Path("/usr/share/sumo/tools")
DEFAULT_WORK_ROOT = Path("cf_h2o/results/libsignal_workdirs")
SUMO_ASSET_FIXES = {
    "sumo4x4": {
        "target_dir": Path("data/raw_data/grid4x4"),
        "source_dir": Path("resco_benchmark/environments/grid4x4"),
        "files": ("grid4x4.net.xml", "grid4x4.rou.xml"),
    }
}
SUMO_RAW_DATA_DIRS = {
    "sumo1x1": ("cologne1",),
    "sumo1x3": ("cologne3",),
    "sumo4x4": ("grid4x4",),
}

FINAL_RE = re.compile(
    r"Final Travel Time is\s+(?P<travel_time>[-+0-9.eE]+),\s+"
    r"mean rewards:\s+(?P<reward>[-+0-9.eE]+),\s+"
    r"queue:\s+(?P<queue>[-+0-9.eE]+),\s+"
    r"delay:\s+(?P<delay>[-+0-9.eE]+),\s+"
    r"throughput:\s+(?P<throughput>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class LibSignalRunSpec:
    agent: str
    network: str
    episodes: int
    steps: int
    test_steps: int
    seed: int | None


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path_for_spec(
    cache_dir: Path,
    spec: LibSignalRunSpec,
    *,
    interface: str,
    extra_args: tuple[str, ...],
) -> Path:
    key = {
        "agent": spec.agent,
        "network": spec.network,
        "episodes": spec.episodes,
        "steps": spec.steps,
        "test_steps": spec.test_steps,
        "seed": spec.seed,
        "interface": interface,
        "extra_args": list(extra_args),
    }
    stem = f"{spec.network}_{spec.agent}_e{spec.episodes}_st{spec.steps}_ts{spec.test_steps}_s{spec.seed}"
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


def _safe_spec_stem(spec: LibSignalRunSpec) -> str:
    raw = f"{spec.network}_{spec.agent}_e{spec.episodes}_st{spec.steps}_ts{spec.test_steps}_s{spec.seed}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def isolated_repo_path(work_root: Path, spec: LibSignalRunSpec, *, interface: str, extra_args: tuple[str, ...]) -> Path:
    key = {
        "agent": spec.agent,
        "network": spec.network,
        "episodes": spec.episodes,
        "steps": spec.steps,
        "test_steps": spec.test_steps,
        "seed": spec.seed,
        "interface": interface,
        "extra_args": list(extra_args),
    }
    return work_root / f"{_safe_spec_stem(spec)}_{_json_hash(key)}"


def prepare_isolated_repo(
    *,
    repo_root: Path,
    spec: LibSignalRunSpec,
    work_root: Path,
    interface: str,
    extra_args: tuple[str, ...],
) -> Path:
    """Create a per-run LibSignal checkout because upstream mutates configs."""
    run_root = isolated_repo_path(work_root, spec, interface=interface, extra_args=extra_args)
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        repo_root,
        run_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "data"),
    )

    raw_root = run_root / "data" / "raw_data"
    raw_root.mkdir(parents=True, exist_ok=True)
    for raw_dir_name in SUMO_RAW_DATA_DIRS.get(spec.network, ()):
        src = repo_root / "data" / "raw_data" / raw_dir_name
        dst = raw_root / raw_dir_name
        if not src.exists():
            raise FileNotFoundError(f"missing LibSignal raw_data directory: {src}")
        shutil.copytree(src, dst)
    (run_root / "data" / "output_data").mkdir(parents=True, exist_ok=True)
    return run_root


def parse_final_metrics(stdout: str) -> dict[str, float]:
    matches = list(FINAL_RE.finditer(stdout))
    if not matches:
        return {}
    match = matches[-1]
    return {key: float(value) for key, value in match.groupdict().items()}


def _resco_repo_from_libsignal_repo(repo_root: Path) -> Path:
    return repo_root.parent.parent / "traffic_signal_resco" / "repo"


def ensure_sumo_assets(repo_root: Path, network: str, resco_repo: Path | None = None) -> None:
    fix = SUMO_ASSET_FIXES.get(network)
    if fix is None:
        return
    target_dir = repo_root / fix["target_dir"]
    missing = [name for name in fix["files"] if not (target_dir / name).exists()]
    if not missing:
        return

    candidate_repos = []
    if resco_repo is not None:
        candidate_repos.append(resco_repo)
    candidate_repos.append(_resco_repo_from_libsignal_repo(repo_root))
    candidate_repos.append(DEFAULT_RESC0_REPO)
    source_dir = None
    for candidate_repo in candidate_repos:
        candidate_source = candidate_repo / fix["source_dir"]
        if all((candidate_source / name).exists() for name in missing):
            source_dir = candidate_source
            break
    if source_dir is None:
        missing_text = ", ".join(str(target_dir / name) for name in missing)
        raise FileNotFoundError(f"missing LibSignal SUMO assets and no RESCO source found: {missing_text}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        shutil.copyfile(source_dir / name, target_dir / name)


def summarize_run(
    *,
    spec: LibSignalRunSpec,
    ok: bool,
    returncode: int,
    elapsed_sec: float,
    stdout: str,
    stderr: str,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ok": bool(ok),
        "agent": spec.agent,
        "network": spec.network,
        "episodes": int(spec.episodes),
        "steps": int(spec.steps),
        "test_steps": int(spec.test_steps),
        "seed": spec.seed,
        "returncode": int(returncode),
        "elapsed_sec": float(elapsed_sec),
        "cached": False,
        "stdout_tail": "\n".join(stdout.splitlines()[-14:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-14:]),
    }
    if error:
        row["error"] = error
    metrics = parse_final_metrics(stdout)
    if ok and not metrics:
        row["ok"] = False
        row["error"] = "LibSignal stdout contained no final metric line"
    row.update(metrics)
    return row


def run_libsignal(
    *,
    repo_root: Path,
    spec: LibSignalRunSpec,
    timeout_sec: float | None,
    interface: str,
    extra_args: tuple[str, ...],
    resco_repo: Path | None,
    work_root: Path | None,
    isolate_repo: bool,
    keep_workdirs: bool,
    native_threads: int,
) -> dict[str, Any]:
    if not (repo_root / "run.py").exists():
        raise FileNotFoundError(f"missing LibSignal run.py under {repo_root}")
    run_root = repo_root
    if isolate_repo:
        if work_root is None:
            raise ValueError("work_root is required when isolate_repo is enabled")
        run_root = prepare_isolated_repo(
            repo_root=repo_root,
            spec=spec,
            work_root=work_root,
            interface=interface,
            extra_args=extra_args,
        )
    ensure_sumo_assets(run_root, spec.network, resco_repo=resco_repo)

    env = os.environ.copy()
    pythonpath = [str(SUMO_TOOLS), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
    env.setdefault("SUMO_HOME", "/usr/share/sumo")
    if native_threads > 0:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(native_threads)

    prefix_parts = ["cfcmt", spec.agent, spec.network]
    if spec.seed is not None:
        prefix_parts.append(f"seed{spec.seed}")
    prefix = "_".join(prefix_parts)

    cmd = [
        sys.executable,
        "run.py",
        "-a",
        spec.agent,
        "-w",
        "sumo",
        "-n",
        spec.network,
        "--episodes",
        str(spec.episodes),
        "--steps",
        str(spec.steps),
        "--test_steps",
        str(spec.test_steps),
        "--prefix",
        prefix,
        "--interface",
        interface,
        "--ngpu",
        "-1",
    ]
    if spec.seed is not None:
        cmd.extend(["--seed", str(spec.seed)])
    cmd.extend(extra_args)

    try:
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=run_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        elapsed = time.time() - started
        row = summarize_run(
            spec=spec,
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            elapsed_sec=elapsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=None if proc.returncode == 0 else "LibSignal subprocess failed",
        )
        if isolate_repo:
            row["work_dir"] = str(run_root)
        return row
    finally:
        if isolate_repo and not keep_workdirs:
            shutil.rmtree(run_root, ignore_errors=True)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        groups.setdefault((row["agent"], row["network"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (agent, network), items in sorted(groups.items()):
        out.append(
            {
                "agent": agent,
                "network": network,
                "runs": len(items),
                "travel_time": float(np.mean([item["travel_time"] for item in items])),
                "reward": float(np.mean([item["reward"] for item in items])),
                "queue": float(np.mean([item["queue"] for item in items])),
                "delay": float(np.mean([item["delay"] for item in items])),
                "throughput": float(np.mean([item["throughput"] for item in items])),
                "elapsed_sec": float(np.mean([item["elapsed_sec"] for item in items])),
            }
        )
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary_rows = [
        {
            "agent": row["agent"],
            "network": row["network"],
            "runs": row["runs"],
            "travel_time": row["travel_time"],
            "reward": row["reward"],
            "queue": row["queue"],
            "delay": row["delay"],
            "throughput": row["throughput"],
            "elapsed_sec": row["elapsed_sec"],
        }
        for row in result["aggregate"]
    ]
    run_rows = [
        {
            "ok": row["ok"],
            "agent": row["agent"],
            "network": row["network"],
            "seed": row["seed"],
            "travel_time": row.get("travel_time", float("nan")),
            "reward": row.get("reward", float("nan")),
            "queue": row.get("queue", float("nan")),
            "delay": row.get("delay", float("nan")),
            "throughput": row.get("throughput", float("nan")),
            "elapsed_sec": row["elapsed_sec"],
            "cached": row.get("cached", False),
        }
        for row in result["runs"]
    ]
    lines = [
        "# LibSignal SUMO Baseline Adapter",
        "",
        "Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.",
        "",
        "These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.",
        "",
        "## Aggregate",
        "",
        _format_table(
            summary_rows,
            ["agent", "network", "runs", "travel_time", "reward", "queue", "delay", "throughput", "elapsed_sec"],
        ),
        "",
        "## Runs",
        "",
        _format_table(
            run_rows,
            ["ok", "agent", "network", "seed", "travel_time", "reward", "queue", "delay", "throughput", "elapsed_sec", "cached"],
        ),
        "",
        "## Notes",
        "",
        "- The adapter parses the final metric line printed by LibSignal.",
        "- Use larger episode/step budgets before citing RL numbers as trained baselines.",
        "- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    repo_root: Path,
    agents: tuple[str, ...],
    networks: tuple[str, ...],
    episodes: int,
    steps: int,
    test_steps: int,
    seeds: tuple[int, ...],
    timeout_sec: float | None,
    interface: str,
    extra_args: tuple[str, ...],
    cache_dir: Path | None,
    resume: bool,
    resco_repo: Path | None,
    work_root: Path | None,
    isolate_repo: bool,
    keep_workdirs: bool,
    native_threads: int,
    workers: int,
) -> dict[str, Any]:
    seed_values: tuple[int | None, ...] = seeds if seeds else (None,)
    specs = [
        LibSignalRunSpec(
            agent=agent,
            network=network,
            episodes=episodes,
            steps=steps,
            test_steps=test_steps,
            seed=seed,
        )
        for network in networks
        for agent in agents
        for seed in seed_values
    ]

    def run_one(spec: LibSignalRunSpec) -> dict[str, Any]:
        cache_path = (
            cache_path_for_spec(cache_dir, spec, interface=interface, extra_args=extra_args)
            if cache_dir is not None
            else None
        )
        if resume and cache_path is not None:
            cached = load_cached_row(cache_path)
            if cached is not None:
                return cached
        try:
            row = run_libsignal(
                repo_root=repo_root,
                spec=spec,
                timeout_sec=timeout_sec,
                interface=interface,
                extra_args=extra_args,
                resco_repo=resco_repo,
                work_root=work_root,
                isolate_repo=isolate_repo,
                keep_workdirs=keep_workdirs,
                native_threads=native_threads,
            )
        except subprocess.TimeoutExpired as exc:
            row = summarize_run(
                spec=spec,
                ok=False,
                returncode=124,
                elapsed_sec=float(timeout_sec or 0.0),
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
                error="timeout",
            )
        except Exception as exc:
            row = summarize_run(
                spec=spec,
                ok=False,
                returncode=1,
                elapsed_sec=0.0,
                stdout="",
                stderr="",
                error=f"{type(exc).__name__}: {exc}",
            )
        if cache_path is not None:
            write_cached_row(cache_path, row)
        return row

    if workers <= 1 or len(specs) <= 1:
        runs = [run_one(spec) for spec in specs]
    else:
        runs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_spec = {pool.submit(run_one, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    runs.append(future.result())
                except Exception as exc:
                    runs.append(
                        summarize_run(
                            spec=spec,
                            ok=False,
                            returncode=1,
                            elapsed_sec=0.0,
                            stdout="",
                            stderr="",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

    return {
        "experiment": "traffic_signal_libsignal_baseline",
        "setting": {
            "repo_root": str(repo_root),
            "agents": list(agents),
            "networks": list(networks),
            "episodes": int(episodes),
            "steps": int(steps),
            "test_steps": int(test_steps),
            "seeds": list(seeds),
            "timeout_sec": timeout_sec,
            "interface": interface,
            "extra_args": list(extra_args),
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "resume": bool(resume),
            "resco_repo": str(resco_repo) if resco_repo is not None else None,
            "work_root": str(work_root) if work_root is not None else None,
            "isolate_repo": bool(isolate_repo),
            "keep_workdirs": bool(keep_workdirs),
            "native_threads": int(native_threads),
            "workers": int(workers),
            "protocol_note": "LibSignal state/reward/action/logging; not same-protocol CFCMT phase-MPC",
        },
        "aggregate": aggregate_rows(runs),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_LIBSIGNAL_REPO)
    parser.add_argument("--resco-repo", type=Path, default=DEFAULT_RESC0_REPO)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--agents", type=_parse_csv, default=DEFAULT_AGENTS)
    parser.add_argument("--networks", type=_parse_csv, default=DEFAULT_NETWORKS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--test-steps", type=int, default=60)
    parser.add_argument("--seeds", type=_parse_int_csv, default=())
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--interface", choices=("libsumo", "traci"), default="libsumo")
    parser.add_argument("--extra-args", type=_parse_csv, default=())
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--isolate-repo", dest="isolate_repo", action="store_true")
    parser.add_argument("--no-isolate-repo", dest="isolate_repo", action="store_false")
    parser.set_defaults(isolate_repo=True)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--disable-train-test", action="store_true")
    parser.add_argument("--disable-save-model", action="store_true")
    parser.add_argument("--save-rate", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    effective_extra_args = list(args.extra_args)
    if args.torch_threads > 0:
        effective_extra_args.extend(["--torch_threads", str(args.torch_threads)])
    if args.disable_train_test:
        effective_extra_args.extend(["--test_when_train", "false"])
    if args.disable_save_model:
        effective_extra_args.extend(["--save_model", "false"])
    if args.save_rate is not None:
        effective_extra_args.extend(["--save_rate", str(args.save_rate)])

    result = run_experiment(
        repo_root=args.repo_root,
        agents=args.agents,
        networks=args.networks,
        episodes=args.episodes,
        steps=args.steps,
        test_steps=args.test_steps,
        seeds=args.seeds,
        timeout_sec=args.timeout_sec,
        interface=args.interface,
        extra_args=tuple(effective_extra_args),
        cache_dir=args.cache_dir,
        resume=args.resume,
        resco_repo=args.resco_repo,
        work_root=args.work_root,
        isolate_repo=args.isolate_repo,
        keep_workdirs=args.keep_workdirs,
        native_threads=args.native_threads,
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
