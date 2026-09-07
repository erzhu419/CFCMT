#!/usr/bin/env python3
"""Run and retrieve the waiting-aligned spatiotemporal latent freeze."""

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
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "v83r79-waiting-aligned-latent-freeze-launch-v1"
ARTIFACT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v71_external_v9_"
    "waiting_aligned_spatiotemporal_latent_artifact.json"
)
DIAGNOSTIC_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v70_external_v9_"
    "waiting_aligned_spatiotemporal_latent_diagnostic.json"
)
CACHE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v67_external_v9_"
    "waiting_aligned_redevelopment_cache.json"
)
PARTITION_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v65_external_v9_"
    "post_validation_redevelopment_partition.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-result-local", type=Path, required=True)
    parser.add_argument("--diagnostic-result-remote", type=Path, required=True)
    parser.add_argument("--cache-audit-local", type=Path, required=True)
    parser.add_argument("--cache-audit-remote", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--local-artifact-root", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--node", default="node004")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--launch-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.local_artifact_root.exists()
        or args.local_out.exists()
        or args.launch_out.exists()
    ):
        raise FileExistsError("refusing to overwrite local latent freeze evidence")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    allowed_nodes = {f"node{index:03d}" for index in range(1, 7)}
    if str(args.node) not in allowed_nodes:
        raise ValueError("latent freeze node is outside node001-node006")
    remote_log = args.remote_out.with_suffix(".log")
    code, _, stderr = scheduler.run_on(
        str(args.node),
        f"test ! -e {shlex.quote(str(args.remote_artifact_root))} && "
        f"test ! -e {shlex.quote(str(args.remote_out))} && "
        f"test ! -e {shlex.quote(str(remote_log))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite remote latent freeze: {stderr}")
    command = shlex.join(
        [
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_freeze",
            "--artifact-protocol",
            str(snapshot / ARTIFACT_PROTOCOL_RELATIVE),
            "--diagnostic-protocol",
            str(snapshot / DIAGNOSTIC_PROTOCOL_RELATIVE),
            "--diagnostic-result",
            str(args.diagnostic_result_remote),
            "--cache-protocol",
            str(snapshot / CACHE_PROTOCOL_RELATIVE),
            "--cache-audit",
            str(args.cache_audit_remote),
            "--cache-root",
            str(args.cache_root),
            "--partition",
            str(snapshot / PARTITION_RELATIVE),
            "--manifest",
            str(snapshot / MANIFEST_RELATIVE),
            "--conversion-root",
            str(args.conversion_root),
            "--output-root",
            str(args.remote_artifact_root),
            "--workers",
            str(args.workers),
            "--out",
            str(args.remote_out),
        ]
    )
    wrapped = (
        f"mkdir -p {shlex.quote(str(args.remote_out.parent))} && "
        f"cd {shlex.quote(str(snapshot))} && "
        f"export CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))} "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 && "
        f"{command} > {shlex.quote(str(remote_log))} 2>&1"
    )
    returncode, stdout, run_stderr = scheduler.run_on(
        str(args.node), wrapped, timeout=int(args.timeout_sec), check=False
    )
    sync_errors: list[str] = []
    if int(returncode) == 0:
        args.local_out.parent.mkdir(parents=True, exist_ok=True)
        target = scheduler._ssh_target_for_node(str(args.node))
        ssh_shell = scheduler._ssh_rsync_shell_for_node(str(args.node))
        for remote, local, trailing in (
            (args.remote_out, args.local_out, False),
            (args.remote_artifact_root, args.local_artifact_root, True),
        ):
            local.parent.mkdir(parents=True, exist_ok=True)
            source = f"{target}:{remote}{'/' if trailing else ''}"
            completed = subprocess.run(
                ["rsync", "-a", "--timeout=300", "-e", ssh_shell, source, str(local)],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                sync_errors.append(completed.stderr[-4000:])
    result = _read_json(args.local_out) if args.local_out.is_file() else None
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "node": str(args.node),
        "artifact_protocol": str(args.artifact_protocol.resolve()),
        "artifact_protocol_sha256": _sha256(args.artifact_protocol),
        "diagnostic_result_local_sha256": _sha256(args.diagnostic_result_local),
        "cache_audit_local_sha256": _sha256(args.cache_audit_local),
        "remote_artifact_root": str(args.remote_artifact_root),
        "remote_out": str(args.remote_out),
        "remote_log": str(remote_log),
        "local_artifact_root": str(args.local_artifact_root.resolve()),
        "local_out": str(args.local_out.resolve()),
        "workers": max(int(args.workers), 1),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "sync_errors": sync_errors,
        "result_status": result.get("status") if result else None,
        "result_decision": result.get("decision") if result else None,
    }
    _atomic_json(args.launch_out, payload)
    if int(returncode) != 0 or sync_errors or result is None:
        raise RuntimeError(
            "waiting latent freeze failed: "
            f"returncode={returncode}, sync_errors={sync_errors}, "
            f"stderr={run_stderr[-2000:]}"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "artifact": result["artifact"]["root"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
