#!/usr/bin/env python3
"""Launch the frozen v75 paired matrix on node001-node006."""

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

from cf_h2o.eval.traffic_signal_external_local_demand_closed_loop_development import (  # noqa: E402
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    _remote_postflight,
    _remote_preflight,
    _run_direct_without_sync,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v75r71-local-demand-development-six-node-launch-v1"
NODES = tuple(f"node{index:03d}" for index in range(1, 7))
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v58_external_v9_"
    "local_demand_closed_loop_development.json"
)
PROPOSAL_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def _sync_new(
    *, local_path: Path, remote_path: Path, scheduler_path: Path, directory: bool
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
        raise FileExistsError(f"remote v75 input already exists: {remote_path}: {stderr}")
    source = f"{local_path}/" if directory else str(local_path)
    destination = f"{scheduler._ssh_target_for_node('node001')}:{remote_path}"
    if directory:
        scheduler.run_on(
            "node001",
            f"mkdir -p {shlex.quote(str(remote_path))}",
            timeout=120,
            check=True,
        )
        destination += "/"
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--timeout=180",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            source,
            destination,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v75 input sync failed: {completed.stderr[-4000:]}")


def _input_visibility(
    *, checks: Sequence[tuple[Path, str]], scheduler_path: Path
) -> dict[str, Any]:
    scheduler = _load_scheduler(scheduler_path)
    rows = []
    for node in NODES:
        commands = [
            f"test $(sha256sum {shlex.quote(str(path))} | cut -d' ' -f1) = "
            f"{shlex.quote(expected)}"
            for path, expected in checks
        ]
        code, _, stderr = scheduler.run_on(
            node, " && ".join(commands), timeout=180, check=False
        )
        rows.append(
            {
                "node": node,
                "returncode": int(code),
                "stderr_tail": str(stderr)[-2000:],
            }
        )
    if not all(row["returncode"] == 0 for row in rows):
        raise RuntimeError(f"v75 inputs are not visible on every node: {rows}")
    return {"passed": True, "nodes": rows, "file_count": len(checks)}


def _sync_results(
    *,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    attempts: int = 6,
) -> list[dict[str, Any]]:
    if local_results_root.exists():
        raise FileExistsError(f"refusing to overwrite local v75 results: {local_results_root}")
    local_results_root.mkdir(parents=True, exist_ok=False)
    scheduler = _load_scheduler(scheduler_path)
    rows = []
    for attempt in range(1, int(attempts) + 1):
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--timeout=180",
                "-e",
                scheduler._ssh_rsync_shell_for_node("node001"),
                f"{scheduler._ssh_target_for_node('node001')}:{remote_results_root}/",
                f"{local_results_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "attempt": attempt,
                "returncode": int(completed.returncode),
                "stderr_tail": completed.stderr[-3000:],
            }
        )
        if completed.returncode == 0:
            return rows
        time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"v75 result sync failed: {rows[-1]}")


def build_specs(
    *,
    snapshot_root: Path,
    hierarchical_freeze_audit_remote: Path,
    hierarchical_freeze_root_remote: Path,
    veto_freeze_result_remote: Path,
    veto_joint_audit_remote: Path,
    veto_artifact_root_remote: Path,
    conversion_root_remote: Path,
    conversion_manifest_remote: Path,
    remote_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> list[dict[str, Any]]:
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol = _read_json(protocol_path)
    identities = all_rollout_identities(protocol)
    if (
        protocol.get("protocol") != PROTOCOL
        or len(identities) != 48
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("v75 launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_local_demand_closed_loop_development_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--proposal-parent-protocol",
                str(snapshot_root / PROPOSAL_PARENT_RELATIVE),
                "--hierarchical-freeze-audit",
                str(hierarchical_freeze_audit_remote),
                "--hierarchical-freeze-root",
                str(hierarchical_freeze_root_remote),
                "--veto-freeze-result",
                str(veto_freeze_result_remote),
                "--veto-joint-audit",
                str(veto_joint_audit_remote),
                "--veto-artifact-root",
                str(veto_artifact_root_remote),
                "--external-manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root",
                str(conversion_root_remote),
                "--conversion-manifest",
                str(conversion_manifest_remote),
                "--results-root",
                str(remote_results_root),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(NODES)),
                "--workers",
                str(workers),
            ]
        )
        specs.append(
            {
                "project": "CFCMT",
                "signature": f"CFCMT/v75r71/local-demand-development/shard-{shard_index}",
                "description": f"CFCMT v75 local-demand development shard {shard_index}",
                "cmd": command,
                "cwd": str(snapshot_root),
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root_remote),
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                },
                "cpu": int(cpu_cores),
                "ram_mb": int(ram_mb),
                "vram": 0,
                "priority": "high",
                "resource_family": "CFCMT-local-demand-closed-loop-v1",
                "require_node": node,
                "result_dir": str(remote_results_root),
                "skip_launch_staging": True,
                "matrix_rows": len(identities) // len(NODES),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root-remote", type=Path, required=True)
    parser.add_argument("--veto-freeze-result-local", type=Path, required=True)
    parser.add_argument("--veto-freeze-result-remote", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-local", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-local", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-cores", type=int, default=16)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol = _read_json(protocol_path)
    specs = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        hierarchical_freeze_audit_remote=args.hierarchical_freeze_audit_remote,
        hierarchical_freeze_root_remote=args.hierarchical_freeze_root_remote,
        veto_freeze_result_remote=args.veto_freeze_result_remote,
        veto_joint_audit_remote=args.veto_joint_audit_remote,
        veto_artifact_root_remote=args.veto_artifact_root_remote,
        conversion_root_remote=args.conversion_root_remote,
        conversion_manifest_remote=args.conversion_manifest_remote,
        remote_results_root=args.remote_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    artifact_checks = []
    for path in sorted(args.veto_artifact_root_local.rglob("*")):
        if path.is_file():
            artifact_checks.append(
                (
                    args.veto_artifact_root_remote
                    / path.relative_to(args.veto_artifact_root_local),
                    _sha256(path),
                )
            )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "frozen_protocol_sha256": _sha256(protocol_path),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "physical_nodes": list(NODES),
        "matrix_size": len(all_rollout_identities(protocol)),
        "workers_per_shard": int(args.workers),
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "result_protocol": RESULT_PROTOCOL,
        "specs": specs,
    }
    try:
        if args.plan_only:
            payload["execution_mode"] = "plan_only"
        else:
            _sync_new(
                local_path=args.veto_freeze_result_local,
                remote_path=args.veto_freeze_result_remote,
                scheduler_path=args.scheduler,
                directory=False,
            )
            _sync_new(
                local_path=args.veto_joint_audit_local,
                remote_path=args.veto_joint_audit_remote,
                scheduler_path=args.scheduler,
                directory=False,
            )
            _sync_new(
                local_path=args.veto_artifact_root_local,
                remote_path=args.veto_artifact_root_remote,
                scheduler_path=args.scheduler,
                directory=True,
            )
            checks = [
                (args.veto_freeze_result_remote, _sha256(args.veto_freeze_result_local)),
                (args.veto_joint_audit_remote, _sha256(args.veto_joint_audit_local)),
                *artifact_checks,
            ]
            payload["input_visibility"] = _input_visibility(
                checks=checks, scheduler_path=args.scheduler
            )
            payload["remote_preflight"] = _remote_preflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                allow_existing_results=False,
            )
            payload["direct_results"] = _run_direct_without_sync(
                specs, scheduler_path=args.scheduler, timeout=10800
            )
            payload["remote_postflight"] = _remote_postflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                expected_result_file_count=(
                    2 * len(all_rollout_identities(protocol)) + len(NODES)
                ),
            )
            payload["sync_attempts"] = _sync_results(
                remote_results_root=args.remote_results_root,
                local_results_root=args.local_results_root,
                scheduler_path=args.scheduler,
            )
            payload["submitted"] = True
            payload["execution_mode"] = "direct_six_physical_nodes"
    except Exception as exc:
        payload["execution_error"] = repr(exc)[:8000]
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "node_count": len(specs),
                "matrix_size": payload["matrix_size"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
