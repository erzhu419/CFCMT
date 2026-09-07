#!/usr/bin/env python3
"""Fit, audit, and retrieve the frozen v82 uncertainty-horizon artifacts."""

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
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_audit import (  # noqa: E402
    AUTHORIZATION_DECISION as AUDIT_AUTHORIZATION_DECISION,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as FREEZE_AUTHORIZATION_DECISION,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "v83r79-uncertainty-horizon-artifact-launch-v1"
ARTIFACT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v82_external_v9_"
    "uncertainty_horizon_artifacts.json"
)
V81_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v81_external_v9_"
    "multihorizon_estimand_aligned_closed_loop_development.json"
)
V79_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v79_external_v9_"
    "multihorizon_state_latent_artifacts.json"
)
SCREEN_PROTOCOL_RELATIVE = Path(
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


def _rsync_from_remote(
    *, scheduler: Any, node: str, source: Path, destination: Path, tree: bool
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--timeout=300",
            "-e",
            scheduler._ssh_rsync_shell_for_node(node),
            f"{scheduler._ssh_target_for_node(node)}:{source}{'/' if tree else ''}",
            str(destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v82 artifact retrieval failed: {completed.stderr[-4000:]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--v81-audit-remote", type=Path, required=True)
    parser.add_argument("--v79-result-remote", type=Path, required=True)
    parser.add_argument("--v79-audit-remote", type=Path, required=True)
    parser.add_argument("--v79-artifact-root", type=Path, required=True)
    parser.add_argument("--screen-result-remote", type=Path, required=True)
    parser.add_argument("--screen-oof-remote", type=Path, required=True)
    parser.add_argument("--cache-audit-remote", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--remote-freeze-result", type=Path, required=True)
    parser.add_argument("--remote-audit-result", type=Path, required=True)
    parser.add_argument("--local-artifact-root", type=Path, required=True)
    parser.add_argument("--local-freeze-result", type=Path, required=True)
    parser.add_argument("--local-audit-result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--node", default="node004")
    parser.add_argument(
        "--scheduleurm-root",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm"),
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    local_outputs = (
        args.local_artifact_root,
        args.local_freeze_result,
        args.local_audit_result,
        args.transport_root,
        args.out,
    )
    if any(path.exists() for path in local_outputs):
        raise FileExistsError("refusing to overwrite local v82 artifact evidence")
    if str(args.node) not in {f"node{index:03d}" for index in range(1, 7)}:
        raise ValueError("v82 artifact node is outside node001-node006")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    remote_outputs = (
        args.remote_artifact_root,
        args.remote_freeze_result,
        args.remote_audit_result,
    )
    preflight = " && ".join(
        shlex.join(["test", "!", "-e", str(path)]) for path in remote_outputs
    )
    code, _, stderr = scheduler.run_on(
        str(args.node), preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite remote v82 evidence: {stderr}")

    wrapper = snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"
    frozen_args = [
        "--v81-protocol",
        str(snapshot / V81_PROTOCOL_RELATIVE),
        "--v81-audit",
        str(args.v81_audit_remote),
        "--v79-protocol",
        str(snapshot / V79_PROTOCOL_RELATIVE),
        "--v79-result",
        str(args.v79_result_remote),
        "--v79-audit",
        str(args.v79_audit_remote),
        "--screen-protocol",
        str(snapshot / SCREEN_PROTOCOL_RELATIVE),
        "--screen-result",
        str(args.screen_result_remote),
        "--screen-oof",
        str(args.screen_oof_remote),
        "--cache-protocol",
        str(snapshot / CACHE_PROTOCOL_RELATIVE),
        "--cache-audit",
        str(args.cache_audit_remote),
        "--partition",
        str(snapshot / PARTITION_RELATIVE),
        "--manifest",
        str(snapshot / MANIFEST_RELATIVE),
    ]
    common = [
        "--artifact-protocol",
        str(snapshot / ARTIFACT_PROTOCOL_RELATIVE),
        *frozen_args,
        "--cache-root",
        str(args.cache_root),
        "--conversion-root",
        str(args.conversion_root),
        "--workers",
        str(max(1, int(args.workers))),
    ]
    freeze_command = shlex.join(
        [
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze",
            *common,
            "--v79-artifact-root",
            str(args.v79_artifact_root),
            "--output-root",
            str(args.remote_artifact_root),
            "--out",
            str(args.remote_freeze_result),
        ]
    )
    audit_command = shlex.join(
        [
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_audit",
            *common,
            "--artifact-result",
            str(args.remote_freeze_result),
            "--artifact-root",
            str(args.remote_artifact_root),
            "--out",
            str(args.remote_audit_result),
        ]
    )
    remote_log = args.remote_freeze_result.with_suffix(".log")
    remote_command = (
        f"cd {shlex.quote(str(snapshot))} && "
        f"export CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))} "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 && "
        f"{freeze_command} > {shlex.quote(str(remote_log))} 2>&1 && "
        f"{audit_command} >> {shlex.quote(str(remote_log))} 2>&1"
    )
    sys.path.insert(0, str(args.scheduleurm_root.resolve()))
    from algorithm.experiments.durable_remote_command import (
        run_durable_remote_capture,
    )

    args.transport_root.mkdir(parents=True, exist_ok=False)
    returncode, stdout, run_stderr, durable = run_durable_remote_capture(
        str(args.node),
        remote_command,
        args.transport_root / str(args.node),
        run_id="cfcmt-v82-uncertainty-horizon-artifacts-v1",
        timeout_s=int(args.timeout_sec),
        poll_interval_s=15.0,
    )
    sync_error = None
    if int(returncode) == 0:
        try:
            _rsync_from_remote(
                scheduler=scheduler,
                node=str(args.node),
                source=args.remote_freeze_result,
                destination=args.local_freeze_result,
                tree=False,
            )
            _rsync_from_remote(
                scheduler=scheduler,
                node=str(args.node),
                source=args.remote_audit_result,
                destination=args.local_audit_result,
                tree=False,
            )
            _rsync_from_remote(
                scheduler=scheduler,
                node=str(args.node),
                source=args.remote_artifact_root,
                destination=args.local_artifact_root,
                tree=True,
            )
        except Exception as exc:
            sync_error = repr(exc)
    freeze_result = (
        _read_json(args.local_freeze_result)
        if args.local_freeze_result.is_file()
        else None
    )
    audit_result = (
        _read_json(args.local_audit_result)
        if args.local_audit_result.is_file()
        else None
    )
    passed = bool(
        int(returncode) == 0
        and sync_error is None
        and freeze_result is not None
        and freeze_result.get("status") == "PASS"
        and freeze_result.get("decision") == FREEZE_AUTHORIZATION_DECISION
        and audit_result is not None
        and audit_result.get("status") == "PASS"
        and audit_result.get("decision") == AUDIT_AUTHORIZATION_DECISION
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "artifact_protocol_sha256": _sha256(
            PROJECT_ROOT / ARTIFACT_PROTOCOL_RELATIVE
        ),
        "node": str(args.node),
        "workers": max(1, int(args.workers)),
        "remote_artifact_root": str(args.remote_artifact_root),
        "remote_freeze_result": str(args.remote_freeze_result),
        "remote_audit_result": str(args.remote_audit_result),
        "remote_log": str(remote_log),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-8000:],
        "stderr_tail": str(run_stderr)[-8000:],
        "durable_transport": durable,
        "sync_error": sync_error,
        "freeze_status": freeze_result.get("status") if freeze_result else None,
        "freeze_decision": freeze_result.get("decision") if freeze_result else None,
        "audit_status": audit_result.get("status") if audit_result else None,
        "audit_decision": audit_result.get("decision") if audit_result else None,
        "passed": passed,
    }
    _atomic_json(args.out, payload)
    if not passed:
        raise RuntimeError(
            "v82 artifact freeze/audit failed: "
            f"returncode={returncode}, sync_error={sync_error}, "
            f"stderr={str(run_stderr)[-2000:]}"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "freeze_decision": payload["freeze_decision"],
                "audit_decision": payload["audit_decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
