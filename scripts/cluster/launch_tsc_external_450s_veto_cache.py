#!/usr/bin/env python3
"""Launch the v69 450-second cache on node001-node006."""

from __future__ import annotations

import argparse
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

from scripts.cluster import (  # noqa: E402
    launch_tsc_external_long_horizon_target_veto_cache as base,
)
from scripts.cluster.authorize_tsc_external_450s_veto_cache import (  # noqa: E402
    AUDIT_DECISION,
    AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_450s_veto_protocol import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
    _run_direct_specs,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v69r65-external-v9-450s-veto-cache-six-node-launch-v1"
NODES = base.NODES
DIRECT_TIMEOUT_SEC = 28800


def build_specs(**kwargs: Any) -> tuple[list[dict[str, Any]], str]:
    base.PROTOCOL = PROTOCOL
    base.AUDIT_PROTOCOL = AUDIT_PROTOCOL
    base.AUDIT_DECISION = AUDIT_DECISION
    specs, authorization_sha = base.build_specs(**kwargs)
    for spec in specs:
        seed = str(spec["signature"]).rsplit("-", 1)[-1]
        spec.update(
            {
                "description": f"CFCMT v69 aligned 450s veto cache seed {seed}",
                "signature": f"CFCMT/v69r65/450s-veto-cache/seed-{seed}",
                "resource_family": "CFCMT-v69-450s-veto-cache-v1",
            }
        )
    return specs, authorization_sha


def _sync_file_to_shared_remote(
    *, local_path: Path, remote_path: Path, scheduler_path: Path
) -> None:
    scheduler = _load_scheduler(scheduler_path)
    code, _, stderr = scheduler.run_on(
        "node001",
        f"mkdir -p {shlex.quote(str(remote_path.parent))} && "
        f"test ! -e {shlex.quote(str(remote_path))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"remote authorization already exists: {stderr}")
    target = scheduler._ssh_target_for_node("node001")
    shell = scheduler._ssh_rsync_shell_for_node("node001")
    completed = subprocess.run(
        ["rsync", "-a", "-e", shell, str(local_path), f"{target}:{remote_path}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v69 authorization sync failed: {completed.stderr}")


def _wait_for_authorization_visibility(
    *,
    authorization_path: Path,
    expected_sha256: str,
    scheduler_path: Path,
    attempts: int = 12,
) -> dict[str, Any]:
    """Require exact authorization visibility on every node before launch."""

    scheduler = _load_scheduler(scheduler_path)
    for attempt in range(1, int(attempts) + 1):
        rows = []
        for node in NODES:
            command = (
                f"test $(sha256sum {shlex.quote(str(authorization_path))} "
                f"| cut -d' ' -f1) = {shlex.quote(str(expected_sha256))}"
            )
            code, _, stderr = scheduler.run_on(
                node, command, timeout=120, check=False
            )
            rows.append(
                {
                    "node": node,
                    "returncode": int(code),
                    "stderr_tail": str(stderr)[-2000:],
                }
            )
        if all(row["returncode"] == 0 for row in rows):
            return {"passed": True, "attempt": attempt, "nodes": rows}
        if attempt < int(attempts):
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"v69 authorization is not visible on every node: {rows}")


def _sync_cache_with_retries(
    *,
    remote_cache_root: Path,
    local_cache_root: Path,
    scheduler_path: Path,
    attempts: int = 10,
) -> int:
    if local_cache_root.exists():
        raise FileExistsError(f"refusing to overwrite local v69 cache: {local_cache_root}")
    local_cache_root.mkdir(parents=True, exist_ok=False)
    scheduler = _load_scheduler(scheduler_path)
    target = scheduler._ssh_target_for_node("node001")
    shell = scheduler._ssh_rsync_shell_for_node("node001")
    errors: list[str] = []
    for attempt in range(1, int(attempts) + 1):
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--timeout=120",
                "-e",
                shell,
                f"{target}:{remote_cache_root}/",
                f"{local_cache_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return attempt
        errors.append(completed.stderr[-2000:])
        time.sleep(min(5 * attempt, 30))
    raise RuntimeError("v69 cache sync retries exhausted: " + " | ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization-audit-local", type=Path, required=True)
    parser.add_argument("--authorization-audit-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--local-cache-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--cpu-cores", type=int, default=136)
    parser.add_argument("--ram-mb", type=int, default=196608)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    specs, authorization_sha = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        protocol_path=args.protocol,
        authorization_audit_local=args.authorization_audit_local,
        authorization_audit_remote=args.authorization_audit_remote,
        conversion_root=args.conversion_root,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        remote_cache_root=args.remote_cache_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    protocol = _read_json(args.protocol)
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "frozen_protocol_path": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "authorization_audit_sha256": authorization_sha,
        "authorization_decision": AUDIT_DECISION,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "remote_cache_root": str(args.remote_cache_root),
        "local_cache_root": str(args.local_cache_root.resolve()),
        "collection_shards": int(
            protocol["long_horizon_collection"]["collection_shards_per_scenario_seed"]
        ),
        "workers_per_task": int(args.workers),
        "task_count": len(specs),
        "specs": specs,
    }
    try:
        if not args.plan_only:
            base._assert_remote_outputs_absent(
                remote_cache_root=args.remote_cache_root,
                remote_results_root=args.remote_results_root,
                scheduler_path=args.scheduler,
            )
            _sync_file_to_shared_remote(
                local_path=args.authorization_audit_local,
                remote_path=args.authorization_audit_remote,
                scheduler_path=args.scheduler,
            )
            payload["authorization_cross_node_visibility"] = (
                _wait_for_authorization_visibility(
                    authorization_path=args.authorization_audit_remote,
                    expected_sha256=authorization_sha,
                    scheduler_path=args.scheduler,
                )
            )
            direct_results = _run_direct_specs(
                specs,
                scheduler_path=args.scheduler,
                timeout=DIRECT_TIMEOUT_SEC,
            )
            sync_attempts = _sync_cache_with_retries(
                remote_cache_root=args.remote_cache_root,
                local_cache_root=args.local_cache_root,
                scheduler_path=args.scheduler,
            )
            payload.update(
                {
                    "submitted": True,
                    "execution_mode": "direct_run_on_six_physical_nodes",
                    "direct_results": direct_results,
                    "cache_synced": True,
                    "cache_sync_attempts": sync_attempts,
                }
            )
        else:
            payload["execution_mode"] = "plan_only"
    except Exception as exc:
        payload["execution_error"] = repr(exc)
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "parallel_libsumo_workers": len(specs) * int(args.workers),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
