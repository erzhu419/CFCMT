#!/usr/bin/env python3
"""Run the corrected v2 artifact audit without refitting frozen v79 models."""

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
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_audit import (  # noqa: E402
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as FREEZE_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.run_tsc_external_multihorizon_state_latent_artifact_freeze import (  # noqa: E402
    ARTIFACT_PROTOCOL_RELATIVE,
    MANIFEST_RELATIVE,
    PARTITION_RELATIVE,
)


LAUNCH_PROTOCOL = "v83r79-multihorizon-state-latent-artifact-audit-v2-launch-v1"


def _retrieve(
    *,
    scheduler: Any,
    node: str,
    remote: Path,
    local: Path,
    tree: bool,
) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--timeout=300",
            "-e",
            scheduler._ssh_rsync_shell_for_node(node),
            f"{scheduler._ssh_target_for_node(node)}:{remote}{'/' if tree else ''}",
            str(local),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"v2 audit retrieval failed: {completed.stderr[-4000:]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--remote-freeze-result", type=Path, required=True)
    parser.add_argument("--remote-audit-result", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
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
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(
        path.exists()
        for path in (
            args.local_artifact_root,
            args.local_freeze_result,
            args.local_audit_result,
            args.transport_root,
            args.out,
        )
    ):
        raise FileExistsError("refusing to overwrite local v2 audit evidence")
    stage = _read_json(args.stage_manifest)
    snapshot = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    code, _, stderr = scheduler.run_on(
        str(args.node),
        " && ".join(
            (
                shlex.join(["test", "-d", str(args.remote_artifact_root)]),
                shlex.join(["test", "-f", str(args.remote_freeze_result)]),
                shlex.join(["test", "!", "-e", str(args.remote_audit_result)]),
            )
        ),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise ValueError(f"v2 audit remote preflight failed: {stderr}")
    wrapper = snapshot / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_audit",
            "--artifact-protocol",
            str(snapshot / ARTIFACT_PROTOCOL_RELATIVE),
            "--artifact-result",
            str(args.remote_freeze_result),
            "--artifact-root",
            str(args.remote_artifact_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(snapshot / MANIFEST_RELATIVE),
            "--partition",
            str(snapshot / PARTITION_RELATIVE),
            "--conversion-root",
            str(args.conversion_root),
            "--workers",
            str(max(1, int(args.workers))),
            "--out",
            str(args.remote_audit_result),
        ]
    )
    remote_log = args.remote_audit_result.with_suffix(".log")
    remote_command = (
        f"cd {shlex.quote(str(snapshot))} && "
        f"export CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot))} "
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))} "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 && "
        f"{command} > {shlex.quote(str(remote_log))} 2>&1"
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
        run_id="cfcmt-v79-multihorizon-state-latent-artifact-audit-v2",
        timeout_s=int(args.timeout_sec),
        poll_interval_s=15.0,
    )
    sync_error = None
    if int(returncode) == 0:
        try:
            _retrieve(
                scheduler=scheduler,
                node=str(args.node),
                remote=args.remote_artifact_root,
                local=args.local_artifact_root,
                tree=True,
            )
            _retrieve(
                scheduler=scheduler,
                node=str(args.node),
                remote=args.remote_freeze_result,
                local=args.local_freeze_result,
                tree=False,
            )
            _retrieve(
                scheduler=scheduler,
                node=str(args.node),
                remote=args.remote_audit_result,
                local=args.local_audit_result,
                tree=False,
            )
        except Exception as exc:
            sync_error = repr(exc)
    freeze = (
        _read_json(args.local_freeze_result)
        if args.local_freeze_result.is_file()
        else None
    )
    audit = (
        _read_json(args.local_audit_result)
        if args.local_audit_result.is_file()
        else None
    )
    artifact_hashes_exact = bool(
        freeze
        and all(
            freeze["artifacts"][key]["model"]["sha256"]
            == _sha256(args.local_artifact_root / key / "model.pkl")
            and freeze["artifacts"][key]["certificate"]["sha256"]
            == _sha256(args.local_artifact_root / key / "freeze.json")
            for key in freeze["artifacts"]
        )
    )
    passed = bool(
        int(returncode) == 0
        and sync_error is None
        and freeze
        and freeze.get("protocol") == FREEZE_RESULT_PROTOCOL
        and freeze.get("status") == "PASS"
        and freeze.get("decision") == FREEZE_AUTHORIZATION_DECISION
        and audit
        and audit.get("protocol") == RESULT_PROTOCOL
        and audit.get("status") == "PASS"
        and audit.get("decision") == AUTHORIZATION_DECISION
        and audit.get("integrity_gate", {}).get("passed") is True
        and artifact_hashes_exact
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot),
        "snapshot_sha256": stage["snapshot_sha256"],
        "artifact_protocol_sha256": _sha256(args.artifact_protocol),
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
        "artifact_hashes_exact": artifact_hashes_exact,
        "audit_status": audit.get("status") if audit else None,
        "audit_decision": audit.get("decision") if audit else None,
        "passed": passed,
    }
    _atomic_json(args.out, payload)
    if not passed:
        raise RuntimeError(
            "multihorizon artifact v2 audit failed: "
            f"returncode={returncode}, sync_error={sync_error}, "
            f"stderr={str(run_stderr)[-2000:]}"
        )
    print(json.dumps({"passed": True, "decision": audit["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
