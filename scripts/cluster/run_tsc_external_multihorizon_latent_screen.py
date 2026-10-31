#!/usr/bin/env python3
"""Run and retrieve the v77 multihorizon latent OOF screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
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
from scripts.cluster.run_tsc_external_state_conditioned_latent_diagnostic import (  # noqa: E402
    _retrieve_result,
)


LAUNCH_PROTOCOL = "v83r79-multihorizon-latent-screen-launch-v1"
DIAGNOSTIC_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v77_external_v9_"
    "multihorizon_latent_screen.json"
)
CACHE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v76_external_v9_"
    "multihorizon_waiting_cache.json"
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
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit-local", type=Path, required=True)
    parser.add_argument("--cache-audit-remote", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--remote-oof", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--local-oof", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--node", default="node004")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--launch-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(path.exists() for path in (args.local_out, args.local_oof, args.launch_out)):
        raise FileExistsError("refusing to overwrite local multihorizon screen")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    node = str(args.node)
    if node not in {f"node{index:03d}" for index in range(1, 7)}:
        raise ValueError("multihorizon screen node is outside node001-node006")
    remote_log = args.remote_out.with_suffix(".log")
    code, _, stderr = scheduler.run_on(
        node,
        shlex.join(
            [
                "test",
                "!",
                "-e",
                str(args.remote_out),
                "-a",
                "!",
                "-e",
                str(args.remote_oof),
                "-a",
                "!",
                "-e",
                str(remote_log),
            ]
        ),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(
            f"refusing to overwrite remote multihorizon screen: {stderr}"
        )
    command = shlex.join(
        [
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_multihorizon_latent_screen",
            "--diagnostic-protocol",
            str(snapshot / DIAGNOSTIC_PROTOCOL_RELATIVE),
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
            "--cache-workers",
            str(args.cache_workers),
            "--fold-workers",
            str(args.fold_workers),
            "--oof-out",
            str(args.remote_oof),
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
        node,
        wrapped,
        timeout=int(args.timeout_sec),
        check=False,
    )
    remote_log_tail = ""
    if int(returncode) != 0:
        _, remote_log_tail, _ = scheduler.run_on(
            node,
            shlex.join(["tail", "-n", "100", str(remote_log)]),
            timeout=120,
            check=False,
        )
    result_sync_error = None
    oof_sync_error = None
    if int(returncode) == 0:
        result_sync_error = _retrieve_result(
            scheduler=scheduler,
            node=node,
            remote_path=args.remote_out,
            local_path=args.local_out,
        )
        oof_sync_error = _retrieve_result(
            scheduler=scheduler,
            node=node,
            remote_path=args.remote_oof,
            local_path=args.local_oof,
        )
    result = _read_json(args.local_out) if args.local_out.is_file() else None
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "node": node,
        "diagnostic_protocol": str(args.diagnostic_protocol.resolve()),
        "diagnostic_protocol_sha256": _sha256(args.diagnostic_protocol),
        "cache_audit_local_sha256": _sha256(args.cache_audit_local),
        "remote_out": str(args.remote_out),
        "remote_oof": str(args.remote_oof),
        "remote_log": str(remote_log),
        "local_out": str(args.local_out.resolve()),
        "local_oof": str(args.local_oof.resolve()),
        "cache_workers": int(args.cache_workers),
        "fold_workers": min(max(int(args.fold_workers), 1), 8),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "remote_log_tail": str(remote_log_tail)[-8000:],
        "result_sync_error": result_sync_error,
        "oof_sync_error": oof_sync_error,
        "result_status": result.get("status") if result else None,
        "result_decision": result.get("decision") if result else None,
    }
    _atomic_json(args.launch_out, payload)
    if (
        int(returncode) != 0
        or result_sync_error is not None
        or oof_sync_error is not None
        or result is None
    ):
        raise RuntimeError(
            "multihorizon latent screen failed: "
            f"returncode={returncode}, result_sync={result_sync_error}, "
            f"oof_sync={oof_sync_error}, remote_log={remote_log_tail[-2000:]}"
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
