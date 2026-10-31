#!/usr/bin/env python3
"""Launch the frozen v71 development matrix on six physical CPU nodes."""

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

from cf_h2o.eval.traffic_signal_external_demand_conditioned_closed_loop_development import (  # noqa: E402
    PROTOCOL,
    RESULT_PROTOCOL,
    all_method_rollout_identities,
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


LAUNCH_PROTOCOL = "v71r67-demand-conditioned-development-six-node-launch-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v53_external_v9_"
    "demand_conditioned_closed_loop_development.json"
)
TRAINING_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v52_external_v9_"
    "demand_conditioned_450s_veto_training.json"
)
PROPOSAL_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def _sync_results(
    *,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    retries: int = 8,
) -> list[dict[str, Any]]:
    if local_results_root.exists():
        raise FileExistsError(
            f"refusing to merge pre-existing v71 results: {local_results_root}"
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
    raise RuntimeError(f"v71 result sync failed: {attempts[-1]}")


def _sync_file_absent(
    *, local: Path, remote: Path, scheduler_path: Path
) -> dict[str, Any]:
    scheduler = _load_scheduler(scheduler_path)
    code, _, stderr = scheduler.run_on(
        "node001",
        f"mkdir -p {shlex.quote(str(remote.parent))} && "
        f"test ! -e {shlex.quote(str(remote))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"remote v71 evidence already exists: {remote}: {stderr}")
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            str(local),
            f"{scheduler._ssh_target_for_node('node001')}:{remote}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v71 evidence sync failed: {completed.stderr[-4000:]}")
    return {"remote": str(remote), "sha256": _sha256(local)}


def _sync_directory_absent(
    *, local: Path, remote: Path, scheduler_path: Path
) -> dict[str, Any]:
    scheduler = _load_scheduler(scheduler_path)
    code, _, stderr = scheduler.run_on(
        "node001",
        f"mkdir -p {shlex.quote(str(remote.parent))} && "
        f"test ! -e {shlex.quote(str(remote))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"remote v71 artifact root exists: {remote}: {stderr}")
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{local}/",
            f"{scheduler._ssh_target_for_node('node001')}:{remote}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v71 artifact sync failed: {completed.stderr[-4000:]}")
    local_files = sorted(path for path in local.rglob("*") if path.is_file())
    return {
        "remote": str(remote),
        "file_count": len(local_files),
        "files": {
            str(path.relative_to(local)): _sha256(path) for path in local_files
        },
    }


def _cross_node_input_visibility(
    *,
    expected_files: dict[Path, str],
    scheduler_path: Path,
    attempts: int = 12,
) -> dict[str, Any]:
    """Wait until every physical node reads every staged input at the exact hash."""

    scheduler = _load_scheduler(scheduler_path)
    history = []
    for attempt in range(1, int(attempts) + 1):
        node_rows = []
        for node in NODES:
            command = " && ".join(
                f"test $(sha256sum {shlex.quote(str(path))} | cut -d' ' -f1) = "
                f"{shlex.quote(expected)}"
                for path, expected in expected_files.items()
            )
            code, _, stderr = scheduler.run_on(
                node, command, timeout=120, check=False
            )
            node_rows.append(
                {
                    "node": node,
                    "returncode": int(code),
                    "stderr_tail": str(stderr)[-2000:],
                }
            )
        history.append({"attempt": attempt, "nodes": node_rows})
        if all(row["returncode"] == 0 for row in node_rows):
            return {
                "passed": True,
                "attempt": attempt,
                "expected_file_count": len(expected_files),
                "nodes": node_rows,
            }
        if attempt < int(attempts):
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"v71 inputs not visible on all nodes: {history[-1]}")


def build_specs(
    *,
    snapshot_root: Path,
    freeze_audit_local: Path,
    freeze_audit_remote: Path,
    freeze_root_remote: Path,
    training_result_local: Path,
    training_result_remote: Path,
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
    training_path = PROJECT_ROOT / TRAINING_PROTOCOL_RELATIVE
    proposal_path = PROJECT_ROOT / PROPOSAL_PARENT_RELATIVE
    manifest_path = PROJECT_ROOT / MANIFEST_RELATIVE
    protocol = _read_json(protocol_path)
    identities = all_method_rollout_identities(protocol)
    hashes = {
        "protocol": _sha256(protocol_path),
        "parent_protocol": _sha256(training_path),
        "collection_protocol": _sha256(training_result_local),
        "proposal_parent_protocol": _sha256(proposal_path),
        "freeze_audit": _sha256(freeze_audit_local),
        "veto_joint_audit": _sha256(veto_joint_audit_local),
        "external_manifest": _sha256(manifest_path),
        "conversion_manifest": _sha256(conversion_manifest_local),
    }
    if (
        protocol.get("protocol") != PROTOCOL
        or hashes["parent_protocol"] != protocol["parent_protocol"]["sha256"]
        or hashes["collection_protocol"]
        != protocol["collection_protocol"]["sha256"]
        or hashes["proposal_parent_protocol"]
        != protocol["proposal_parent_protocol"]["sha256"]
        or hashes["freeze_audit"]
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or hashes["veto_joint_audit"]
        != protocol["target_veto_joint_audit"]["sha256"]
        or hashes["external_manifest"]
        != protocol["environment"]["external_manifest"]["sha256"]
        or hashes["conversion_manifest"]
        != protocol["environment"]["conversion_manifest"]["sha256"]
        or len(identities) != 32
        or set(protocol["development"]["active_cities"])
        & set(protocol["development"]["fallback_cities"])
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
        or int(ram_mb) < 16384
    ):
        raise ValueError("v71 development launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/"
        "run_tsc_external_demand_conditioned_closed_loop_development_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--training-protocol",
                str(snapshot_root / TRAINING_PROTOCOL_RELATIVE),
                "--proposal-parent-protocol",
                str(snapshot_root / PROPOSAL_PARENT_RELATIVE),
                "--freeze-audit",
                str(freeze_audit_remote),
                "--freeze-root",
                str(freeze_root_remote),
                "--training-result",
                str(training_result_remote),
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
                "description": f"CFCMT v71 demand-conditioned development shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v71r67/demand-conditioned-development/shard-{shard_index}",
                "resource_family": "CFCMT-demand-conditioned-closed-loop-v1",
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
    if sorted(int(spec["matrix_rows"]) for spec in specs) != [5, 5, 5, 5, 6, 6]:
        raise ValueError("v71 six-node matrix distribution changed")
    return specs, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--freeze-audit-local", type=Path, required=True)
    parser.add_argument("--freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-root-remote", type=Path, required=True)
    parser.add_argument("--training-result-local", type=Path, required=True)
    parser.add_argument("--training-result-remote", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-local", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-local", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-manifest-local", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cpu-cores", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    specs, hashes = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        freeze_audit_local=args.freeze_audit_local,
        freeze_audit_remote=args.freeze_audit_remote,
        freeze_root_remote=args.freeze_root_remote,
        training_result_local=args.training_result_local,
        training_result_remote=args.training_result_remote,
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
                allow_existing_results=False,
            )
            payload["remote_input_sync"] = {
                "training_result": _sync_file_absent(
                    local=args.training_result_local,
                    remote=args.training_result_remote,
                    scheduler_path=args.scheduler,
                ),
                "veto_joint_audit": _sync_file_absent(
                    local=args.veto_joint_audit_local,
                    remote=args.veto_joint_audit_remote,
                    scheduler_path=args.scheduler,
                ),
                "veto_artifact_root": _sync_directory_absent(
                    local=args.veto_artifact_root_local,
                    remote=args.veto_artifact_root_remote,
                    scheduler_path=args.scheduler,
                ),
            }
            expected_remote_files = {
                args.training_result_remote: _sha256(args.training_result_local),
                args.veto_joint_audit_remote: _sha256(args.veto_joint_audit_local),
            }
            for local_path in sorted(
                path
                for path in args.veto_artifact_root_local.rglob("*")
                if path.is_file()
            ):
                expected_remote_files[
                    args.veto_artifact_root_remote
                    / local_path.relative_to(args.veto_artifact_root_local)
                ] = _sha256(local_path)
            payload["cross_node_input_visibility"] = _cross_node_input_visibility(
                expected_files=expected_remote_files,
                scheduler_path=args.scheduler,
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
                expected_result_file_count=(
                    2 * len(all_method_rollout_identities(protocol)) + len(NODES)
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
        payload["execution_error"] = repr(exc)[:12000]
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
