#!/usr/bin/env python3
"""Submit the frozen V117 confirmation matrix on six CPU nodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v117-pure-waiting-confirmation-launch-v2"
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
NODES = tuple(f"node00{index}" for index in range(1, 7))


def build_specs(
    *,
    snapshot_root: Path,
    freeze_path: Path,
    freeze_sha256: str,
    conversion_root: Path,
    conversion_manifest: Path,
    remote_results_root: Path,
    workers_per_node: int,
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = snapshot_root / "scripts/cluster/run_tsc_pure_waiting_confirmation_shard.py"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                "env",
                f"CFCMT_SOURCE_ROOT={snapshot_root}",
                f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "PYTHONUNBUFFERED=1",
                str(wrapper),
                str(runner),
                "--freeze",
                str(freeze_path),
                "--freeze-sha256",
                str(freeze_sha256),
                "--manifest",
                str(manifest),
                "--conversion-root",
                str(conversion_root),
                "--conversion-manifest",
                str(conversion_manifest),
                "--results-root",
                str(remote_results_root),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(NODES)),
                "--workers",
                str(workers_per_node),
            ]
        ) + " && printf 'TASK_DONE\\n'"
        specs.append(
            {
                "description": f"CFCMT V117 heldout-city confirmation {node}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v117/pure-waiting-confirmation-v2/{node}",
                "resource_family": "CFCMT-v117-pure-waiting-confirmation-v2",
                "vram": 0,
                "ram_mb": 98304,
                "cpu": int(workers_per_node),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(remote_results_root / "_tasks" / node),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--freeze-local", type=Path, required=True)
    parser.add_argument("--freeze-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--workers-per-node", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V117 launch artifact")
    stage = _read_json(args.stage_manifest)
    freeze = _read_json(args.freeze_local)
    freeze_sha = _sha256(args.freeze_local)
    if (
        freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
        or int(freeze.get("matrix_size", -1))
        != len(freeze.get("fresh_seeds", ())) * len(freeze.get("active_arms", ()))
        or int(args.workers_per_node) != 20
    ):
        raise ValueError("V117 launch freeze or worker contract changed")
    snapshot_root = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    check = shlex.join(
        [
            "test",
            "!",
            "-e",
            str(args.remote_results_root),
            "-a",
            "-f",
            str(args.freeze_remote),
            "-a",
            "-f",
            str(args.conversion_manifest),
        ]
    )
    code, _, stderr = scheduler.run_on("node001", check, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"V117 remote preconditions failed: {stderr}")
    code, stdout, stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(args.freeze_remote)]),
        timeout=120,
        check=False,
    )
    remote_sha = str(stdout).strip().split()[0] if str(stdout).strip() else ""
    if int(code) != 0 or remote_sha != freeze_sha:
        raise RuntimeError(
            f"V117 remote freeze identity differs: {remote_sha} != {freeze_sha}; {stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        freeze_path=args.freeze_remote,
        freeze_sha256=freeze_sha,
        conversion_root=args.conversion_root,
        conversion_manifest=args.conversion_manifest,
        remote_results_root=args.remote_results_root,
        workers_per_node=int(args.workers_per_node),
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v117-pure-waiting-confirmation-v2",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "freeze_local": str(args.freeze_local.resolve()),
        "freeze_remote": str(args.freeze_remote),
        "freeze_sha256": freeze_sha,
        "active_arms": freeze["active_arms"],
        "matrix_size": freeze["matrix_size"],
        "remote_results_root": str(args.remote_results_root),
        "workers_per_node": int(args.workers_per_node),
        "task_count": len(specs),
        "specs": specs,
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "matrix_size": payload["matrix_size"],
                "task_count": len(specs),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if completed.returncode != 0:
        raise RuntimeError("V117 scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
