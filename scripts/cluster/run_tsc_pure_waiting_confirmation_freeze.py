#!/usr/bin/env python3
"""Freeze V117 on the server and retrieve only the compact JSON contract."""

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
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
    PROTOCOL as CONFIRMATION_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v117-pure-waiting-confirmation-freeze-launch-v2"
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v117_"
    "pure_waiting_heldout_city_confirmation.json"
)


def build_remote_command(
    *,
    snapshot: Path,
    selector_result: Path,
    selector_sha256: str,
    deployment_fit_result: Path,
    deployment_fit_sha256: str,
    remote_out: Path,
) -> str:
    return shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot}",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            str(snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"),
            "-m",
            "cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze",
            "--protocol",
            str(snapshot / PROTOCOL_RELATIVE),
            "--selector-result",
            str(selector_result),
            "--selector-result-sha256",
            str(selector_sha256),
            "--deployment-fit-result",
            str(deployment_fit_result),
            "--deployment-fit-result-sha256",
            str(deployment_fit_sha256),
            "--out",
            str(remote_out),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selector-result-local", type=Path, required=True)
    parser.add_argument("--selector-result-remote", type=Path, required=True)
    parser.add_argument("--deployment-fit-result-local", type=Path, required=True)
    parser.add_argument("--deployment-fit-result-remote", type=Path, required=True)
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--node", default="node001")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--launch-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.local_out.exists() or args.launch_out.exists():
        raise FileExistsError("refusing to overwrite local V117 freeze evidence")

    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != CONFIRMATION_PROTOCOL:
        raise ValueError("V117 confirmation protocol changed")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    selector_sha = _sha256(args.selector_result_local)
    fit_sha = _sha256(args.deployment_fit_result_local)
    protocol_sha = _sha256(args.protocol)
    scheduler = _load_scheduler(args.scheduler)

    precondition = shlex.join(
        [
            "test",
            "!",
            "-e",
            str(args.remote_out),
            "-a",
            "-f",
            str(args.selector_result_remote),
            "-a",
            "-f",
            str(args.deployment_fit_result_remote),
            "-a",
            "-f",
            str(snapshot / PROTOCOL_RELATIVE),
        ]
    )
    code, _, stderr = scheduler.run_on(
        args.node, precondition, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"V117 remote freeze preconditions failed: {stderr}")

    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        args.node,
        shlex.join(
            [
                "sha256sum",
                str(args.selector_result_remote),
                str(args.deployment_fit_result_remote),
                str(snapshot / PROTOCOL_RELATIVE),
            ]
        ),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0]
        for line in str(hash_stdout).splitlines()
        if line.split()
    ]
    expected_hashes = [selector_sha, fit_sha, protocol_sha]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "V117 remote freeze inputs differ from local evidence: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )

    command = build_remote_command(
        snapshot=snapshot,
        selector_result=args.selector_result_remote,
        selector_sha256=selector_sha,
        deployment_fit_result=args.deployment_fit_result_remote,
        deployment_fit_sha256=fit_sha,
        remote_out=args.remote_out,
    )
    returncode, stdout, run_stderr = scheduler.run_on(
        args.node, command, timeout=int(args.timeout_sec), check=False
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
                scheduler._ssh_rsync_shell_for_node(args.node),
                f"{scheduler._ssh_target_for_node(args.node)}:{args.remote_out}",
                str(args.local_out),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            sync_error = completed.stderr[-4000:]

    freeze = _read_json(args.local_out) if args.local_out.is_file() else None
    valid = bool(
        freeze is not None
        and freeze.get("protocol") == FREEZE_PROTOCOL
        and freeze.get("decision") == AUTHORIZATION_DECISION
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "node": str(args.node),
        "protocol_spec": {
            "local": str(args.protocol.resolve()),
            "remote": str(snapshot / PROTOCOL_RELATIVE),
            "sha256": protocol_sha,
        },
        "selector_result": {
            "local": str(args.selector_result_local.resolve()),
            "remote": str(args.selector_result_remote),
            "sha256": selector_sha,
        },
        "deployment_fit_result": {
            "local": str(args.deployment_fit_result_local.resolve()),
            "remote": str(args.deployment_fit_result_remote),
            "sha256": fit_sha,
        },
        "remote_input_sha256": remote_hashes,
        "remote_out": str(args.remote_out),
        "local_out": str(args.local_out.resolve()),
        "returncode": int(returncode),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(run_stderr)[-4000:],
        "sync_error": sync_error,
        "freeze_protocol": freeze.get("protocol") if freeze else None,
        "freeze_decision": freeze.get("decision") if freeze else None,
        "matrix_size": freeze.get("matrix_size") if freeze else None,
        "active_arms": freeze.get("active_arms") if freeze else None,
    }
    _atomic_json(args.launch_out, payload)
    if int(returncode) != 0 or sync_error is not None or not valid:
        raise RuntimeError(
            "V117 server-side freeze failed: "
            f"returncode={returncode}, sync_error={sync_error}, "
            f"stderr={str(run_stderr)[-2000:]}"
        )
    print(
        json.dumps(
            {
                "decision": freeze["decision"],
                "active_arms": freeze["active_arms"],
                "matrix_size": freeze["matrix_size"],
                "out": str(args.local_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
