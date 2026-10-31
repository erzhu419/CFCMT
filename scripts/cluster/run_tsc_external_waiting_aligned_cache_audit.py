#!/usr/bin/env python3
"""Run and retrieve the waiting-aligned cache audit from node001."""

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
from scripts.cluster.launch_tsc_external_waiting_aligned_redevelopment_cache import (  # noqa: E402
    MANIFEST_RELATIVE,
    PROTOCOL_RELATIVE,
)


LAUNCH_PROTOCOL = "v83r79-waiting-aligned-cache-audit-launch-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--read-workers", type=int, default=48)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--launch-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.local_out.exists() or args.launch_out.exists():
        raise FileExistsError("refusing to overwrite local waiting cache audit")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    code, _, stderr = scheduler.run_on(
        "node001",
        f"test ! -e {shlex.quote(str(args.remote_out))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite remote cache audit: {stderr}")
    command = shlex.join(
        [
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit",
            "--protocol",
            str(snapshot / PROTOCOL_RELATIVE),
            "--manifest",
            str(snapshot / MANIFEST_RELATIVE),
            "--conversion-root",
            str(args.conversion_root),
            "--cache-root",
            str(args.cache_root),
            "--task-root",
            str(args.task_root),
            "--read-workers",
            str(args.read_workers),
            "--out",
            str(args.remote_out),
        ]
    )
    wrapped = (
        f"cd {shlex.quote(str(snapshot))} && "
        f"export CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))} "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 && "
        f"{command}"
    )
    returncode, stdout, run_stderr = scheduler.run_on(
        "node001", wrapped, timeout=int(args.timeout_sec), check=False
    )
    sync_error = None
    if int(returncode) == 0:
        args.local_out.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--timeout=180",
                "-e",
                scheduler._ssh_rsync_shell_for_node("node001"),
                f"{scheduler._ssh_target_for_node('node001')}:{args.remote_out}",
                str(args.local_out),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            sync_error = completed.stderr[-4000:]
    audit = _read_json(args.local_out) if args.local_out.is_file() else None
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "remote_out": str(args.remote_out),
        "local_out": str(args.local_out.resolve()),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "sync_error": sync_error,
        "audit_status": audit.get("status") if audit else None,
        "audit_decision": audit.get("decision") if audit else None,
    }
    _atomic_json(args.launch_out, payload)
    if int(returncode) != 0 or sync_error is not None or audit is None:
        raise RuntimeError(
            "waiting cache audit failed: "
            f"returncode={returncode}, sync_error={sync_error}, stderr={run_stderr[-2000:]}"
        )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "decision": audit["decision"],
                "cache_files": audit["cache_file_count"],
                "rows": audit["total_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
