#!/usr/bin/env python3
"""Recover a v65 cache sync after all six direct computations succeeded."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_long_horizon_target_veto_cache import (
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


RECOVERY_PROTOCOL = "v65r61-long-horizon-cache-rsync-recovery-v1"


def _sync_with_retries(
    *,
    remote_cache_root: Path,
    local_cache_root: Path,
    scheduler_path: Path,
    retries: int,
) -> list[dict[str, Any]]:
    scheduler = _load_scheduler(scheduler_path)
    target = scheduler._ssh_target_for_node("node001")
    ssh_shell = scheduler._ssh_rsync_shell_for_node("node001")
    local_cache_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for attempt in range(1, int(retries) + 1):
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--timeout=180",
                "-e",
                ssh_shell,
                f"{target}:{remote_cache_root}/",
                f"{local_cache_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        row = {
            "attempt": attempt,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        attempts.append(row)
        if completed.returncode == 0:
            return attempts
        if attempt < int(retries):
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"v65 cache recovery sync failed: {attempts[-1]}")


def recover_launch_manifest(
    *,
    failed_launch_path: Path,
    protocol_path: Path,
    remote_cache_root: Path,
    local_cache_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    retries: int,
) -> dict[str, Any]:
    failed = _read_json(failed_launch_path)
    protocol = _read_json(protocol_path)
    specs = tuple(failed.get("specs", ()))
    if not (
        failed.get("protocol") == LAUNCH_PROTOCOL
        and failed.get("submitted") is False
        and "cache sync failed" in str(failed.get("execution_error", ""))
        and protocol.get("protocol") == PROTOCOL
        and failed.get("frozen_protocol_sha256") == _sha256(protocol_path)
        and len(specs) == len(NODES) == 6
        and {str(spec.get("require_node")) for spec in specs} == set(NODES)
    ):
        raise ValueError("v65 failed-launch recovery precondition changed")
    direct_results = []
    summary_evidence = []
    for spec in specs:
        node = str(spec["require_node"])
        local_dir = Path(str(spec["local_result_dir"]))
        if not local_dir.is_absolute():
            local_dir = Path.cwd() / local_dir
        summary_path = local_dir / "cache_summary.json"
        summary = _read_json(summary_path)
        if (
            summary.get("experiment")
            != "traffic_signal_counterfactual_cache_precompute"
            or summary.get("runtime", {}).get("hostname") != node
            or summary.get("runtime", {}).get("source_tree_sha256")
            != failed.get("source_tree_sha256")
        ):
            raise ValueError(f"v65 recovered direct summary changed for {node}")
        summary_evidence.append(
            {
                "node": node,
                "path": str(summary_path.resolve()),
                "sha256": _sha256(summary_path),
            }
        )
        direct_results.append(
            {
                "node": node,
                "city": Path(str(spec["result_dir"])).name,
                "returncode": 0,
                "stdout": "not retained after post-compute cache rsync transport failure",
                "stderr": "",
                "recovered_from_completed_local_summary": True,
                "summary_sha256": _sha256(summary_path),
            }
        )
    attempts = _sync_with_retries(
        remote_cache_root=remote_cache_root,
        local_cache_root=local_cache_root,
        scheduler_path=scheduler_path,
        retries=retries,
    )
    scheduler = _load_scheduler(scheduler_path)
    command = (
        f"find {str(remote_cache_root)!r} -maxdepth 1 -type f -name '*.npz' | wc -l"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=120, check=False
    )
    remote_count = int(str(stdout).strip() or -1)
    local_count = len(tuple(Path(local_cache_root).glob("*.npz")))
    expected_count = int(
        protocol["long_horizon_collection"]["expected_cache_file_count"]
    )
    recovery_gate = {
        "failed_manifest_is_transport_only": True,
        "six_completed_node_summaries": len(summary_evidence) == 6,
        "rsync_completed": attempts[-1]["returncode"] == 0,
        "remote_cache_count_exact": int(code) == 0
        and remote_count == expected_count,
        "local_cache_count_exact": local_count == expected_count,
        "remote_count_stderr": str(stderr)[-2000:],
    }
    recovery_gate["passed"] = all(
        value for key, value in recovery_gate.items() if key != "remote_count_stderr"
    )
    if not recovery_gate["passed"]:
        raise RuntimeError(f"v65 cache recovery gate failed: {recovery_gate}")
    return {
        **failed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": True,
        "execution_mode": "direct_run_on_six_physical_nodes",
        "direct_results": direct_results,
        "cache_synced": True,
        "execution_error": None,
        "recovery": {
            "protocol": RECOVERY_PROTOCOL,
            "failed_launch_manifest": str(Path(failed_launch_path).resolve()),
            "failed_launch_manifest_sha256": _sha256(failed_launch_path),
            "reason": "post_compute_rsync_transport_connection_closed",
            "direct_success_proof": (
                "launcher reached _sync_shared_cache only after _run_direct_specs "
                "returned six clean direct results and synced six summaries"
            ),
            "summary_evidence": summary_evidence,
            "sync_attempts": attempts,
            "remote_cache_file_count": remote_count,
            "local_cache_file_count": local_count,
            "gate": recovery_gate,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-launch", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--local-cache-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite recovery manifest: {args.out}")
    payload = recover_launch_manifest(
        failed_launch_path=args.failed_launch,
        protocol_path=args.protocol,
        remote_cache_root=args.remote_cache_root,
        local_cache_root=args.local_cache_root,
        local_results_root=args.local_results_root,
        scheduler_path=args.scheduler,
        retries=args.retries,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "cache_synced": payload["cache_synced"],
                "cache_file_count": payload["recovery"]["local_cache_file_count"],
                "failed_launch_sha256": payload["recovery"][
                    "failed_launch_manifest_sha256"
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
