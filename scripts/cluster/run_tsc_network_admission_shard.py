#!/usr/bin/env python3
"""Run one deterministic shard of an external-network admission matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402


PROTOCOL = "external-network-admission-deterministic-shard-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_jobs(
    *, protocol: dict[str, Any], shard_index: int, shard_count: int
) -> list[tuple[str, int]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid admission shard identity")
    matrix = [
        (str(scenario), int(seed))
        for scenario in protocol["scenarios"]
        for seed in protocol["seeds"]
    ]
    return [
        row for index, row in enumerate(matrix) if index % shard_count == shard_index
    ]


def _command(
    *, protocol: dict[str, Any], scenario: str, seed: int, out: Path
) -> list[str]:
    runtime = dict(protocol["runtime"])
    admission = dict(protocol["admission"])
    conversion = dict(protocol["conversion"])
    return [
        sys.executable,
        "-m",
        "cf_h2o.eval.traffic_signal_external_network_admission",
        "--conversion-root",
        str(conversion["root"]),
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--horizon-sec",
        str(runtime["horizon_sec"]),
        "--expected-sumo-version",
        str(runtime["sumo_version"]),
        "--minimum-controllable-tls",
        str(runtime["minimum_controllable_tls"]),
        "--maximum-collision-count",
        str(admission["maximum_collision_count"]),
        "--maximum-teleport-fraction",
        str(admission["maximum_teleport_fraction"]),
        "--expected-conversion-manifest-sha256",
        str(conversion["manifest_sha256"]),
        "--expected-conversion-tree-sha256",
        str(conversion["tree_sha256"]),
        "--progress-interval-sec",
        "1200",
        "--out",
        str(out),
    ]


def run_shard(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    output_root: Path,
    shard_index: int,
    shard_count: int,
    workers: int,
) -> dict[str, Any]:
    if _sha256(protocol_path) != expected_protocol_sha256:
        raise ValueError("admission protocol identity changed")
    if workers <= 0:
        raise ValueError("workers must be positive")
    protocol = _read_json(protocol_path)
    jobs = build_jobs(
        protocol=protocol,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"shard_{shard_index:02d}_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    for scenario, seed in jobs:
        result = output_root / f"{scenario}_seed{seed}" / "admission.json"
        if result.exists():
            raise FileExistsError(f"refusing to overwrite admission result: {result}")

    def execute(job: tuple[str, int]) -> dict[str, Any]:
        scenario, seed = job
        out = output_root / f"{scenario}_seed{seed}" / "admission.json"
        started = time.monotonic()
        completed = subprocess.run(
            _command(
                protocol=protocol,
                scenario=scenario,
                seed=seed,
                out=out,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "scenario": scenario,
            "seed": seed,
            "returncode": int(completed.returncode),
            "elapsed_seconds": float(time.monotonic() - started),
            "result_present": out.is_file(),
            "stdout_tail": completed.stdout[-1600:],
            "stderr_tail": completed.stderr[-1600:],
        }

    rows = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"SHARD_PROGRESS shard={shard_index}/{shard_count} "
                f"scenario={row['scenario']} seed={row['seed']} "
                f"returncode={row['returncode']} elapsed={row['elapsed_seconds']:.1f}",
                flush=True,
            )
    rows.sort(key=lambda row: (str(row["scenario"]), int(row["seed"])))
    infrastructure_passed = all(
        int(row["returncode"]) == 0 and row["result_present"] is True
        for row in rows
    )
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "matrix_protocol": protocol["protocol"],
        "matrix_protocol_sha256": expected_protocol_sha256,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": workers,
        "job_count": len(jobs),
        "infrastructure_passed": infrastructure_passed,
        "rows": rows,
    }
    atomic_write_json(summary_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    payload = run_shard(
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        output_root=args.output_root,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "shard_index": payload["shard_index"],
                "job_count": payload["job_count"],
                "infrastructure_passed": payload["infrastructure_passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["infrastructure_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
