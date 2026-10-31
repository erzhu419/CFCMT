#!/usr/bin/env python3
"""Launch the one-shard waiting-aligned counterfactual pilot on node001."""

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


LAUNCH_PROTOCOL = (
    "v83r79-external-v9-waiting-aligned-counterfactual-pilot-launch-v1"
)
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v66_external_v9_"
    "waiting_aligned_counterfactual_pilot.json"
)
PARTITION_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v65_external_v9_"
    "post_validation_redevelopment_partition.json"
)
SOURCE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v51_external_v9_"
    "estimand_aligned_450s_target_veto_cache.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def _sync_results(
    *, remote_root: Path, local_root: Path, scheduler_path: Path
) -> None:
    if local_root.exists():
        raise FileExistsError(f"refusing to overwrite local pilot root: {local_root}")
    local_root.mkdir(parents=True, exist_ok=False)
    scheduler = _load_scheduler(scheduler_path)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--timeout=180",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:{remote_root}/",
            f"{local_root}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pilot result sync failed: {completed.stderr[-4000:]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--waiting-cache-root", type=Path)
    parser.add_argument("--remote-run-root", type=Path, required=True)
    parser.add_argument("--local-run-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    if _sha256(args.protocol) != _sha256(PROJECT_ROOT / PROTOCOL_RELATIVE):
        raise ValueError("pilot protocol path is not the frozen repository protocol")
    scheduler = _load_scheduler(args.scheduler)
    code, _, stderr = scheduler.run_on(
        "node001",
        f"test ! -e {shlex.quote(str(args.remote_run_root))}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite remote pilot root: {stderr}")
    remote_result = args.remote_run_root / "pilot_result_v1.json"
    remote_cache = (
        Path(args.waiting_cache_root)
        if args.waiting_cache_root is not None
        else args.remote_run_root / "waiting_cache"
    )
    command = shlex.join(
        [
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_pilot",
            "--protocol",
            str(snapshot / PROTOCOL_RELATIVE),
            "--partition",
            str(snapshot / PARTITION_RELATIVE),
            "--source-protocol",
            str(snapshot / SOURCE_PROTOCOL_RELATIVE),
            "--manifest",
            str(snapshot / MANIFEST_RELATIVE),
            "--conversion-root",
            str(args.conversion_root),
            "--legacy-cache-root",
            str(args.legacy_cache_root),
            "--waiting-cache-root",
            str(remote_cache),
            "--out",
            str(remote_result),
        ]
    )
    wrapped = (
        f"mkdir -p {shlex.quote(str(args.remote_run_root))} && "
        f"cd {shlex.quote(str(snapshot))} && "
        f"export CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))} "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 && "
        f"{command}"
    )
    started = datetime.now(timezone.utc)
    returncode, stdout, run_stderr = scheduler.run_on(
        "node001", wrapped, timeout=int(args.timeout_sec), check=False
    )
    sync_error = None
    try:
        _sync_results(
            remote_root=args.remote_run_root,
            local_root=args.local_run_root,
            scheduler_path=args.scheduler,
        )
    except Exception as error:
        sync_error = repr(error)
    local_result = args.local_run_root / "pilot_result_v1.json"
    result = _read_json(local_result) if local_result.is_file() else None
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "submitted": True,
        "node": "node001",
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "conversion_root": str(args.conversion_root),
        "legacy_cache_root": str(args.legacy_cache_root),
        "waiting_cache_root": str(remote_cache),
        "remote_run_root": str(args.remote_run_root),
        "local_run_root": str(args.local_run_root.resolve()),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "sync_error": sync_error,
        "result_status": result.get("status") if result else None,
        "result_decision": result.get("decision") if result else None,
    }
    _atomic_json(args.out, payload)
    if int(returncode) != 0 or sync_error is not None or not result:
        raise RuntimeError(
            "waiting-aligned pilot failed: "
            f"returncode={returncode}, sync_error={sync_error}, stderr={run_stderr[-2000:]}"
        )
    print(
        json.dumps(
            {
                "returncode": int(returncode),
                "status": result["status"],
                "decision": result["decision"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
