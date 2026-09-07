#!/usr/bin/env python3
"""Launch the frozen v67 target-veto development matrix on six CPU nodes."""

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

from cf_h2o.eval.traffic_signal_external_target_veto_closed_loop_development import (  # noqa: E402
    PROTOCOL,
    RESULT_PROTOCOL,
    all_method_rollout_identities,
)
from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    _remote_postflight,
    _remote_preflight,
    _run_direct_without_sync,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v68r64-proposal-conditional-development-six-node-launch-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v50_external_v9_proposal_conditional_closed_loop_development.json"
)
PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v49_external_v9_proposal_conditional_veto_successor.json"
)
COLLECTION_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v47_external_v9_long_horizon_target_veto.json"
)
PROPOSAL_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_hierarchical_closed_loop_development.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def _sync_results_resumable(
    *,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    retries: int = 5,
) -> list[dict[str, Any]]:
    if local_results_root.exists():
        raise FileExistsError(
            f"refusing to merge pre-existing formal results: {local_results_root}"
        )
    from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
        _load_scheduler,
    )

    scheduler = _load_scheduler(scheduler_path)
    local_results_root.mkdir(parents=True, exist_ok=False)
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
                scheduler._ssh_rsync_shell_for_node("node001"),
                f"{scheduler._ssh_target_for_node('node001')}:{remote_results_root}/",
                f"{local_results_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        attempts.append(
            {
                "attempt": attempt,
                "returncode": int(completed.returncode),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if completed.returncode == 0:
            return attempts
        if attempt < int(retries):
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"target-veto development sync failed: {attempts[-1]}")


def build_specs(
    *,
    snapshot_root: Path,
    freeze_audit_local: Path,
    freeze_audit_remote: Path,
    freeze_root_remote: Path,
    veto_joint_audit_local: Path,
    veto_joint_audit_remote: Path,
    veto_artifact_root_remote: Path,
    conversion_root_remote: Path,
    conversion_manifest_local: Path,
    conversion_manifest_remote: Path,
    remote_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    parent_path = PROJECT_ROOT / PARENT_RELATIVE
    collection_path = PROJECT_ROOT / COLLECTION_RELATIVE
    proposal_parent_path = PROJECT_ROOT / PROPOSAL_PARENT_RELATIVE
    manifest_path = PROJECT_ROOT / MANIFEST_RELATIVE
    protocol = _read_json(protocol_path)
    identities = all_method_rollout_identities(protocol)
    hashes = {
        "protocol": _sha256(protocol_path),
        "parent_protocol": _sha256(parent_path),
        "collection_protocol": _sha256(collection_path),
        "proposal_parent_protocol": _sha256(proposal_parent_path),
        "freeze_audit": _sha256(freeze_audit_local),
        "veto_joint_audit": _sha256(veto_joint_audit_local),
        "external_manifest": _sha256(manifest_path),
        "conversion_manifest": _sha256(conversion_manifest_local),
    }
    if (
        protocol.get("protocol") != PROTOCOL
        or hashes["parent_protocol"]
        != str(protocol["parent_protocol"]["sha256"])
        or hashes["collection_protocol"]
        != str(protocol["collection_protocol"]["sha256"])
        or hashes["proposal_parent_protocol"]
        != str(protocol["proposal_parent_protocol"]["sha256"])
        or hashes["freeze_audit"]
        != str(protocol["hierarchical_freeze_audit"]["sha256"])
        or hashes["veto_joint_audit"]
        != str(protocol["target_veto_joint_audit"]["sha256"])
        or hashes["external_manifest"]
        != str(protocol["environment"]["external_manifest"]["sha256"])
        or hashes["conversion_manifest"]
        != str(protocol["environment"]["conversion_manifest"]["sha256"])
        or len(identities) != 96
        or set(protocol["development"]["active_cities"])
        & set(protocol["development"]["fallback_cities"])
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("target-veto development launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_target_veto_closed_loop_development_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--parent-protocol",
                str(snapshot_root / PARENT_RELATIVE),
                "--collection-protocol",
                str(snapshot_root / COLLECTION_RELATIVE),
                "--proposal-parent-protocol",
                str(snapshot_root / PROPOSAL_PARENT_RELATIVE),
                "--freeze-audit",
                str(freeze_audit_remote),
                "--freeze-root",
                str(freeze_root_remote),
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
        shard_size = sum(
            index % len(NODES) == shard_index for index in range(len(identities))
        )
        specs.append(
            {
                "description": f"CFCMT v68 proposal-conditional development shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v68r64/proposal-conditional-development/shard-{shard_index}",
                "resource_family": "CFCMT-proposal-conditional-closed-loop-v1",
                "vram": 0,
                "ram_mb": int(ram_mb),
                "cpu": int(cpu_cores),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root_remote),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "result_dir": str(remote_results_root),
                "matrix_rows": int(shard_size),
            }
        )
    if {int(spec["matrix_rows"]) for spec in specs} != {16}:
        raise ValueError("target-veto matrix is not balanced across six nodes")
    return specs, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--freeze-audit-local", type=Path, required=True)
    parser.add_argument("--freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-root-remote", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-local", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-manifest-local", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu-cores", type=int, default=24)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    specs, hashes = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        freeze_audit_local=args.freeze_audit_local,
        freeze_audit_remote=args.freeze_audit_remote,
        freeze_root_remote=args.freeze_root_remote,
        veto_joint_audit_local=args.veto_joint_audit_local,
        veto_joint_audit_remote=args.veto_joint_audit_remote,
        veto_artifact_root_remote=args.veto_artifact_root_remote,
        conversion_root_remote=args.conversion_root_remote,
        conversion_manifest_local=args.conversion_manifest_local,
        conversion_manifest_remote=args.conversion_manifest_remote,
        remote_results_root=args.remote_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "input_hashes": hashes,
        "conversion_tree_sha256": protocol["environment"][
            "conversion_tree_sha256"
        ],
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "workers_per_shard": int(args.workers),
        "node_count": len(specs),
        "matrix_size": sum(int(spec["matrix_rows"]) for spec in specs),
        "specs": specs,
    }
    try:
        if args.direct:
            payload["remote_preflight"] = _remote_preflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                allow_existing_results=bool(args.resume),
            )
            payload["direct_results"] = _run_direct_without_sync(
                specs, scheduler_path=args.scheduler, timeout=7200
            )
            payload["remote_postflight"] = _remote_postflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                expected_result_file_count=(2 * len(all_method_rollout_identities(protocol)))
                + len(NODES),
            )
            payload["sync_attempts"] = _sync_results_resumable(
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
