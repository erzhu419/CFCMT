#!/usr/bin/env python3
"""Launch the frozen v89 matrix on node001-node006 and sync summaries only."""

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
from scripts.cluster.freeze_tsc_external_v9_fresh_confirmation import (  # noqa: E402
    NODES,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)


LAUNCH_PROTOCOL = "v89r85-fresh-confirmation-six-node-launch-v1"
V89_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v89_external_v9_"
    "fresh_confirmation.json"
)
V86_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v86_external_v9_"
    "interference_aware_redevelopment.json"
)
V88_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v88_external_v9_"
    "bounded_stay_redevelopment.json"
)


def _command(
    *,
    snapshot_root: Path,
    node_index: int,
    workers: int,
    results_root: Path,
    args: argparse.Namespace,
) -> str:
    paths = {
        "v83": "traffic_signal_tsc_v83_external_v9_uncertainty_horizon_closed_loop_development.json",
        "v84": "traffic_signal_tsc_v84_external_v9_uncertainty_horizon_confirmation.json",
        "v81": "traffic_signal_tsc_v81_external_v9_multihorizon_estimand_aligned_closed_loop_development.json",
        "artifact": "traffic_signal_tsc_v82_external_v9_uncertainty_horizon_artifacts.json",
        "runtime": "traffic_signal_tsc_v46_external_v9_hierarchical_closed_loop_development.json",
        "hierarchy": "traffic_signal_tsc_v44_external_v9_hierarchical_guard_confirmation.json",
        "environment": "traffic_signal_tsc_v41_external_v9_full_budget_refit.json",
        "environment_parent": "traffic_signal_tsc_v40_external_v9_network_repair_confirmation.json",
        "manifest": "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json",
    }
    config = snapshot_root / "cf_h2o/config"
    values = [
        str(snapshot_root / "scripts/cluster/run_tsc_external_fresh_confirmation_shard.py"),
        "--protocol",
        str(snapshot_root / V89_PROTOCOL_RELATIVE),
        "--v86-protocol",
        str(snapshot_root / V86_PROTOCOL_RELATIVE),
        "--v88-protocol",
        str(snapshot_root / V88_PROTOCOL_RELATIVE),
        "--v88-audit",
        str(args.v88_audit_remote),
        "--v83-protocol",
        str(config / paths["v83"]),
        "--v83-audit",
        str(args.v83_audit_remote),
        "--v84-protocol",
        str(config / paths["v84"]),
        "--v84-audit",
        str(args.v84_audit_remote),
        "--v81-protocol",
        str(config / paths["v81"]),
        "--v81-audit",
        str(args.v81_audit_remote),
        "--artifact-protocol",
        str(config / paths["artifact"]),
        "--artifact-result",
        str(args.artifact_result_remote),
        "--artifact-audit",
        str(args.artifact_audit_remote),
        "--artifact-root",
        str(args.artifact_root_remote),
        "--parent-runtime-protocol",
        str(config / paths["runtime"]),
        "--hierarchical-parent",
        str(config / paths["hierarchy"]),
        "--offline-result",
        str(args.offline_result_remote),
        "--support-audit",
        str(args.support_audit_remote),
        "--hierarchical-freeze-audit",
        str(args.hierarchical_freeze_audit_remote),
        "--hierarchical-freeze-root",
        str(args.hierarchical_freeze_root_remote),
        "--environment-protocol",
        str(config / paths["environment"]),
        "--environment-parent",
        str(config / paths["environment_parent"]),
        "--external-manifest",
        str(config / paths["manifest"]),
        "--conversion-root",
        str(args.conversion_root_remote),
        "--conversion-manifest",
        str(args.conversion_manifest_remote),
        "--results-root",
        str(results_root),
        "--shard-index",
        str(node_index),
        "--shard-count",
        str(len(NODES)),
        "--workers",
        str(workers),
    ]
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = " ".join(shlex.quote(value) for value in values)
    return (
        f"cd {shlex.quote(str(snapshot_root))} && "
        f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root_remote))} "
        f"{shlex.quote(str(wrapper))} {command}"
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
    parser.add_argument("--v86-protocol-local", type=Path, required=True)
    parser.add_argument("--v88-protocol-local", type=Path, required=True)
    parser.add_argument("--v88-audit-local", type=Path, required=True)
    parser.add_argument("--v88-audit-remote", type=Path, required=True)
    parser.add_argument("--v83-audit-remote", type=Path, required=True)
    parser.add_argument("--v84-audit-remote", type=Path, required=True)
    parser.add_argument("--v81-audit-remote", type=Path, required=True)
    parser.add_argument("--artifact-result-remote", type=Path, required=True)
    parser.add_argument("--artifact-audit-remote", type=Path, required=True)
    parser.add_argument("--artifact-root-remote", type=Path, required=True)
    parser.add_argument("--offline-result-remote", type=Path, required=True)
    parser.add_argument("--support-audit-remote", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-local-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-remote-root", type=Path, required=True)
    parser.add_argument("--workers-per-node", type=int, default=22)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_results_root.exists():
        raise FileExistsError("refusing to overwrite v89 launch outputs")
    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    snapshot_root = Path(str(stage["snapshot_root"]))
    protocol_path = PROJECT_ROOT / V89_PROTOCOL_RELATIVE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    confirmation = dict(protocol["confirmation"])
    output_roots = dict(protocol["output_roots"])
    if not (
        protocol.get("protocol") == PROTOCOL
        and int(confirmation["matrix_size"]) == 128
        and int(args.workers_per_node) == 22
        and str(args.remote_results_root)
        == str(output_roots["remote_results_root"])
        and str(args.local_results_root.resolve())
        == str(output_roots["local_results_root"])
        and str(args.sealed_v85_local_root.resolve())
        == str(confirmation["sealed_v85_local_root"])
        and str(args.sealed_v85_remote_root)
        == str(confirmation["sealed_v85_remote_root"])
    ):
        raise ValueError("v89 launch protocol or worker allocation changed")
    if args.sealed_v85_local_root.exists():
        raise RuntimeError("sealed v85 local root exists before v89 launch")

    scheduler = _load_scheduler(args.scheduler)
    sealed_command = (
        f"cd /tmp && if test -e "
        f"{shlex.quote(str(args.sealed_v85_remote_root))}; "
        "then printf 'PRESENT\\n'; exit 23; else printf 'ABSENT\\n'; fi"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", sealed_command, timeout=120, check=False
    )
    if int(code) != 0 or str(stdout).strip().splitlines()[-1] != "ABSENT":
        raise RuntimeError(
            f"sealed v85 remote root exists before v89 launch: {stderr or stdout}"
        )
    count_command = (
        f"cd /tmp && if test -e {shlex.quote(str(args.remote_results_root))}; "
        f"then find {shlex.quote(str(args.remote_results_root))} -type f | wc -l; "
        "else printf '0\\n'; fi"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", count_command, timeout=120, check=False
    )
    if int(code) != 0 or int(str(stdout).strip().splitlines()[-1]) != 0:
        raise RuntimeError(f"v89 remote results root is not empty: {stderr or stdout}")

    commands = {
        node: _command(
            snapshot_root=snapshot_root,
            node_index=index,
            workers=int(args.workers_per_node),
            results_root=args.remote_results_root,
            args=args,
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
        raise RuntimeError(f"v89 node failure: {node_results}")

    post_command = (
        f"cd /tmp && find {shlex.quote(str(args.remote_results_root))} "
        "-type f | wc -l"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", post_command, timeout=120, check=False
    )
    remote_file_count = int(str(stdout).strip().splitlines()[-1])
    if int(code) != 0 or remote_file_count != 262:
        raise RuntimeError(
            f"v89 remote file count mismatch: {remote_file_count}; {stderr}"
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
        "protocol_sha256": _sha256(protocol_path),
        "v86_protocol_sha256": _sha256(args.v86_protocol_local),
        "v88_protocol_sha256": _sha256(args.v88_protocol_local),
        "v88_audit_sha256": _sha256(args.v88_audit_local),
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "sealed_v85_preflight": {
            "local_root": str(args.sealed_v85_local_root.resolve()),
            "local_absent": True,
            "remote_root": str(args.sealed_v85_remote_root),
            "remote_absent": True,
        },
        "full_rollouts_remain_remote": True,
        "local_materialization": "six_shard_summary_json_files_only",
        "node_count": len(NODES),
        "workers_per_node": int(args.workers_per_node),
        "matrix_size": 128,
        "remote_file_count": remote_file_count,
        "node_results": node_results,
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
