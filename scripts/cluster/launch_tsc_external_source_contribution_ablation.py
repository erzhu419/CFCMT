#!/usr/bin/env python3
"""Launch the eight-rollout target-only ablation on node001-node006."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)


LAUNCH_PROTOCOL = "v90r86-source-contribution-six-node-launch-v1"
PARENT_LAUNCH_PROTOCOL = "v43r39-external-closed-loop-submission-v4"
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json"
)
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _command(
    *,
    snapshot_root: Path,
    node_index: int,
    workers: int,
    results_root: Path,
    parent: dict[str, Any],
    offline_authorization_remote: Path,
    joint_audit_remote: Path,
) -> str:
    values = [
        str(snapshot_root / "scripts/cluster/run_tsc_external_source_contribution_ablation_shard.py"),
        "--protocol",
        str(snapshot_root / PROTOCOL_RELATIVE),
        "--offline-authorization",
        str(offline_authorization_remote),
        "--offline-authorization-sha256",
        str(parent["offline_authorization_sha256"]),
        "--joint-audit",
        str(joint_audit_remote),
        "--joint-audit-sha256",
        str(parent["joint_freeze_audit_sha256"]),
        "--freeze-root",
        str(parent["freeze_root"]),
        "--external-manifest",
        str(snapshot_root / EXTERNAL_MANIFEST_RELATIVE),
        "--external-manifest-sha256",
        str(parent["external_manifest_sha256"]),
        "--conversion-root",
        str(parent["conversion_root"]),
        "--conversion-manifest",
        str(parent["conversion_manifest"]),
        "--conversion-manifest-sha256",
        str(parent["conversion_manifest_sha256"]),
        "--conversion-tree-sha256",
        str(parent["conversion_tree_sha256"]),
        "--results-root",
        str(results_root),
        "--shard-index",
        str(node_index),
        "--shard-count",
        str(len(NODES)),
        "--workers",
        str(workers),
    ]
    command = " ".join(shlex.quote(value) for value in values)
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    return (
        f"cd {shlex.quote(str(snapshot_root))} && "
        f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(parent['conversion_root']))} "
        f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"NUMEXPR_NUM_THREADS=1 {shlex.quote(str(wrapper))} {command}"
    )


def _run_node(scheduler: Any, node: str, command: str) -> dict[str, Any]:
    started = time.monotonic()
    code, stdout, stderr = scheduler.run_on(
        node, command, timeout=600, check=False
    )
    return {
        "node": node,
        "returncode": int(code),
        "elapsed_sec": float(time.monotonic() - started),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(stderr)[-4000:],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--parent-launch-manifest", type=Path, required=True)
    parser.add_argument("--offline-authorization-remote", type=Path, required=True)
    parser.add_argument("--joint-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers-per-node", type=int, default=2)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_results_root.exists():
        raise FileExistsError("refusing to overwrite source-contribution outputs")

    stage = _read_json(args.stage_manifest)
    parent = _read_json(args.parent_launch_manifest)
    snapshot_root = Path(str(stage["snapshot_root"]))
    if not (
        parent.get("protocol") == PARENT_LAUNCH_PROTOCOL
        and int(parent.get("task_count", -1)) == 56
        and len(parent.get("policies", ())) == 7
        and int(args.workers_per_node) == 2
    ):
        raise ValueError("source-contribution parent launch changed")

    scheduler = _load_scheduler(args.scheduler)
    preflight = (
        f"if test -e {shlex.quote(str(args.remote_results_root))}; "
        f"then find {shlex.quote(str(args.remote_results_root))} -type f | wc -l; "
        "else printf '0\\n'; fi"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", preflight, timeout=120, check=False
    )
    if int(code) != 0 or int(str(stdout).strip().splitlines()[-1]) != 0:
        raise RuntimeError(f"source-contribution root is not empty: {stderr or stdout}")

    commands = {
        node: _command(
            snapshot_root=snapshot_root,
            node_index=index,
            workers=int(args.workers_per_node),
            results_root=args.remote_results_root,
            parent=parent,
            offline_authorization_remote=args.offline_authorization_remote,
            joint_audit_remote=args.joint_audit_remote,
        )
        for index, node in enumerate(NODES)
    }
    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        futures = {
            node: pool.submit(_run_node, scheduler, node, command)
            for node, command in commands.items()
        }
        node_results = [futures[node].result() for node in NODES]
    if any(row["returncode"] != 0 for row in node_results):
        raise RuntimeError(f"source-contribution node failure: {node_results}")

    postflight = (
        f"find {shlex.quote(str(args.remote_results_root))} -type f | wc -l"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", postflight, timeout=120, check=False
    )
    remote_file_count = int(str(stdout).strip().splitlines()[-1])
    if int(code) != 0 or remote_file_count != 22:
        raise RuntimeError(
            f"source-contribution file count mismatch: {remote_file_count}; {stderr}"
        )

    local_shards = args.local_results_root / "_shards"
    local_shards.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:"
            f"{args.remote_results_root}/_shards/",
            f"{local_shards}/",
        ],
        check=True,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": str(stage["snapshot_sha256"]),
        "source_tree_sha256": str(stage["source_tree_sha256"]),
        "parent_launch_manifest": str(args.parent_launch_manifest.resolve()),
        "parent_launch_sha256": _sha256(args.parent_launch_manifest),
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "full_rollouts_remain_remote": True,
        "local_materialization": "six_shard_summary_json_files_only",
        "node_count": len(NODES),
        "workers_per_node": int(args.workers_per_node),
        "matrix_size": 8,
        "remote_file_count": remote_file_count,
        "node_results": node_results,
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
