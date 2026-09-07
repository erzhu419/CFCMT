#!/usr/bin/env python3
"""Run and retrieve the waiting-aligned direct action-ranker OOF diagnostic."""

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


LAUNCH_PROTOCOL = "v83r79-waiting-aligned-action-ranker-diagnostic-launch-v1"
DIAGNOSTIC_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v69_external_v9_"
    "waiting_aligned_action_ranker_diagnostic.json"
)
CACHE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v67_external_v9_"
    "waiting_aligned_redevelopment_cache.json"
)
VETO_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v68_external_v9_"
    "waiting_aligned_veto_diagnostic.json"
)
PARTITION_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v65_external_v9_"
    "post_validation_redevelopment_partition.json"
)
PARENT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit-local", type=Path, required=True)
    parser.add_argument("--cache-audit-remote", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--veto-result-local", type=Path, required=True)
    parser.add_argument("--veto-result-remote", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--node", default="node003")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--launch-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.local_out.exists() or args.launch_out.exists():
        raise FileExistsError("refusing to overwrite local action-ranker diagnostic")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    if str(args.node) not in {f"node{index:03d}" for index in range(1, 7)}:
        raise ValueError("action-ranker diagnostic node is outside node001-node006")
    remote_log = args.remote_out.with_suffix(".log")
    code, _, stderr = scheduler.run_on(
        str(args.node),
        f"test ! -e {shlex.quote(str(args.remote_out))} && "
        f"test ! -e {shlex.quote(str(remote_log))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(
            f"refusing to overwrite remote action-ranker diagnostic: {stderr}"
        )
    command = shlex.join(
        [
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic",
            "--diagnostic-protocol",
            str(snapshot / DIAGNOSTIC_PROTOCOL_RELATIVE),
            "--cache-protocol",
            str(snapshot / CACHE_PROTOCOL_RELATIVE),
            "--cache-audit",
            str(args.cache_audit_remote),
            "--cache-root",
            str(args.cache_root),
            "--veto-protocol",
            str(snapshot / VETO_PROTOCOL_RELATIVE),
            "--veto-result",
            str(args.veto_result_remote),
            "--partition",
            str(snapshot / PARTITION_RELATIVE),
            "--proposal-parent-protocol",
            str(snapshot / PARENT_PROTOCOL_RELATIVE),
            "--hierarchical-freeze-audit",
            str(args.hierarchical_freeze_audit),
            "--hierarchical-freeze-root",
            str(args.hierarchical_freeze_root),
            "--manifest",
            str(snapshot / MANIFEST_RELATIVE),
            "--conversion-root",
            str(args.conversion_root),
            "--cache-workers",
            str(args.cache_workers),
            "--fold-workers",
            str(args.fold_workers),
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
    sync_error = None
    if int(returncode) == 0:
        args.local_out.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--timeout=300",
                "-e",
                scheduler._ssh_rsync_shell_for_node(str(args.node)),
                f"{scheduler._ssh_target_for_node(str(args.node))}:{args.remote_out}",
                str(args.local_out),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            sync_error = completed.stderr[-4000:]
    result = _read_json(args.local_out) if args.local_out.is_file() else None
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "node": str(args.node),
        "diagnostic_protocol": str(args.diagnostic_protocol.resolve()),
        "diagnostic_protocol_sha256": _sha256(args.diagnostic_protocol),
        "cache_audit_local_sha256": _sha256(args.cache_audit_local),
        "veto_result_local_sha256": _sha256(args.veto_result_local),
        "remote_out": str(args.remote_out),
        "remote_log": str(remote_log),
        "local_out": str(args.local_out.resolve()),
        "cache_workers": int(args.cache_workers),
        "fold_workers": min(max(int(args.fold_workers), 1), 8),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "sync_error": sync_error,
        "result_status": result.get("status") if result else None,
        "result_decision": result.get("decision") if result else None,
    }
    _atomic_json(args.launch_out, payload)
    if int(returncode) != 0 or sync_error is not None or result is None:
        raise RuntimeError(
            "waiting action-ranker diagnostic failed: "
            f"returncode={returncode}, sync_error={sync_error}, "
            f"stderr={run_stderr[-2000:]}"
        )
    selected = result["advance_gate"]["selected_candidate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "groups": result["action_group_count"],
                "selected": selected.get("key") if selected else None,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
