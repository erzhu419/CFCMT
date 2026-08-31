#!/usr/bin/env python3
"""Run the seven Chicago safety days with at most one libsumo job per node."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Queue
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.data.acquire_chicago_for_hire_unseen import DATES  # noqa: E402


DEFAULT_SCHEDULER_DIR = Path("/home/erzhu419/mine_code/scheduleurm/skill")
DEFAULT_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
RUNNER_RELATIVE = Path(
    "cf_h2o/eval/traffic_signal_chicago_full_week_admission.py"
)
WRAPPER_RELATIVE = Path("scripts/cluster/run_cfcmt_sumo122.sh")


def _load_scheduler(path: Path):
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def admission_command(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_manifest_sha256: str,
    remote_day_root: Path,
    day_index: int,
) -> str:
    values = [
        str(snapshot_root / WRAPPER_RELATIVE),
        str(snapshot_root / RUNNER_RELATIVE),
        "--package-root",
        str(package_root),
        "--expected-package-manifest-sha256",
        expected_manifest_sha256,
        "--day-index",
        str(day_index),
        "--progress-out",
        str(remote_day_root / "progress.json"),
        "--out",
        str(remote_day_root / "result.json"),
    ]
    execution = " ".join(
        [
            f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))}",
            "PYTHONUNBUFFERED=1",
            *(shlex.quote(value) for value in values),
        ]
    )
    return f"{execution} > {shlex.quote(str(remote_day_root / 'run.log'))} 2>&1"


def _read_remote_json(scheduler: Any, node: str, path: Path) -> dict[str, Any] | None:
    code, stdout, _ = scheduler.run_on(
        node,
        f"test -f {shlex.quote(str(path))} && cat {shlex.quote(str(path))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        return None
    value = json.loads(str(stdout))
    if not isinstance(value, dict):
        raise ValueError(f"expected remote JSON object: {path}")
    return value


def _run_day(
    scheduler: Any,
    *,
    node: str,
    snapshot_root: Path,
    package_root: Path,
    expected_manifest_sha256: str,
    remote_result_root: Path,
    day_index: int,
    timeout_sec: int,
) -> dict[str, Any]:
    day = DATES[day_index].isoformat()
    remote_day_root = remote_result_root / f"day{day_index}_{day}"
    result_path = remote_day_root / "result.json"
    scheduler.run_on(
        node,
        f"mkdir -p {shlex.quote(str(remote_day_root))}",
        timeout=120,
        check=True,
    )
    if int(
        scheduler.run_on(
            node,
            f"test -e {shlex.quote(str(result_path))}",
            timeout=120,
            check=False,
        )[0]
    ) == 0:
        raise FileExistsError(result_path)
    command = admission_command(
        snapshot_root=snapshot_root,
        package_root=package_root,
        expected_manifest_sha256=expected_manifest_sha256,
        remote_day_root=remote_day_root,
        day_index=day_index,
    )
    started = time.perf_counter()
    code, stdout, stderr = scheduler.run_on(
        node, command, timeout=timeout_sec, check=False
    )
    payload = _read_remote_json(scheduler, node, result_path)
    _, log_tail, _ = scheduler.run_on(
        node,
        f"tail -n 120 {shlex.quote(str(remote_day_root / 'run.log'))}",
        timeout=120,
        check=False,
    )
    return {
        "node": node,
        "date": day,
        "day_index": day_index,
        "returncode": int(code),
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": str(stdout)[-2_000:],
        "stderr_tail": str(stderr)[-2_000:],
        "log_tail": str(log_tail)[-16_000:],
        "remote_day_root": str(remote_day_root),
        "result": payload,
    }


def _node_worker(
    scheduler: Any,
    *,
    node: str,
    jobs: Queue[int],
    shared: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        try:
            day_index = jobs.get_nowait()
        except Empty:
            break
        try:
            rows.append(
                _run_day(
                    scheduler,
                    node=node,
                    day_index=day_index,
                    **shared,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "node": node,
                    "date": DATES[day_index].isoformat(),
                    "day_index": day_index,
                    "returncode": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "result": None,
                }
            )
        finally:
            jobs.task_done()
    return rows


def launch_full_week(
    *,
    scheduler_dir: Path,
    snapshot_manifest: Path,
    package_root: Path,
    expected_manifest_sha256: str,
    remote_result_root: Path,
    local_result_root: Path,
    nodes: Sequence[str],
    timeout_sec: int,
) -> dict[str, Any]:
    if not nodes:
        raise ValueError("at least one Chicago safety node is required")
    stage = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    snapshot_root = Path(str(stage["snapshot_root"]))
    scheduler = _load_scheduler(scheduler_dir)
    jobs: Queue[int] = Queue()
    for day_index in range(len(DATES)):
        jobs.put(day_index)
    shared = {
        "snapshot_root": snapshot_root,
        "package_root": package_root,
        "expected_manifest_sha256": expected_manifest_sha256,
        "remote_result_root": remote_result_root,
        "timeout_sec": timeout_sec,
    }
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = [
            executor.submit(
                _node_worker,
                scheduler,
                node=node,
                jobs=jobs,
                shared=shared,
            )
            for node in nodes
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: int(row["day_index"]))
    local_result_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        result = row.get("result")
        if isinstance(result, dict):
            atomic_write_json(
                local_result_root
                / f"day{int(row['day_index'])}_{row['date']}_result.json",
                result,
            )
    complete_indices = {int(row["day_index"]) for row in rows}
    passed = (
        complete_indices == set(range(len(DATES)))
        and len(rows) == len(DATES)
        and all(
            isinstance(row.get("result"), dict)
            and row["result"].get("passed") is True
            for row in rows
        )
    )
    payload = {
        "experiment": "traffic_signal_chicago_full_week_admission_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "snapshot": stage,
        "package_root": str(package_root),
        "expected_package_manifest_sha256": expected_manifest_sha256,
        "remote_result_root": str(remote_result_root),
        "nodes": list(nodes),
        "one_libsumo_process_per_node": True,
        "expected_day_count": len(DATES),
        "day_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    atomic_write_json(local_result_root / "launch_manifest.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--remote-result-root", type=Path, required=True)
    parser.add_argument("--local-result-root", type=Path, required=True)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--timeout-sec", type=int, default=172_800)
    args = parser.parse_args(argv)
    nodes = tuple(value.strip() for value in args.nodes.split(",") if value.strip())
    payload = launch_full_week(
        scheduler_dir=args.scheduler_dir,
        snapshot_manifest=args.snapshot_manifest,
        package_root=args.package_root,
        expected_manifest_sha256=args.expected_package_manifest_sha256,
        remote_result_root=args.remote_result_root,
        local_result_root=args.local_result_root,
        nodes=nodes,
        timeout_sec=args.timeout_sec,
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "day_count": payload["day_count"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
