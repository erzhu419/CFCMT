#!/usr/bin/env python3
"""Submit the V122 B100 source-reliability diagnostic."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (  # noqa: E402
    RESULT_PROTOCOL as CROSSFIT_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (  # noqa: E402
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v122-b100-source-reliability-diagnostic-launch-v1"
SIGNATURE = "CFCMT/v122/b100-source-reliability-diagnostic-v1"
TARGET_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    crossfit_artifact_remote: Path,
    crossfit_artifact_sha256: str,
    target_cache_root: Path,
    target_cache_audit_remote: Path,
    remote_output_root: Path,
    node: str,
    cache_workers: int,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / TARGET_MANIFEST_RELATIVE
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic",
            "--fit-result",
            str(fit_result_remote),
            "--fit-result-sha256",
            str(fit_result_sha256),
            "--crossfit-artifact",
            str(crossfit_artifact_remote),
            "--crossfit-artifact-sha256",
            str(crossfit_artifact_sha256),
            "--target-cache-root",
            str(target_cache_root),
            "--target-manifest",
            str(manifest),
            "--target-cache-audit",
            str(target_cache_audit_remote),
            "--conversion-root",
            str(conversion_root),
            "--cache-workers",
            str(cache_workers),
            "--out",
            str(remote_output_root / "result.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V122 B100 source reliability diagnostic",
        "project": "CFCMT",
        "cmd": command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v122-b100-source-reliability-diagnostic",
        "vram": 0,
        "ram_mb": 32768,
        "cpu": 20,
        "priority": "high",
        "require_node": str(node),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--crossfit-result-local", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-audit-local", type=Path, required=True)
    parser.add_argument("--target-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--node", default="node003")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V122 launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V122 cache workers must be in [1, 20]")

    fit = _read_json(args.fit_result_local)
    crossfit = _read_json(args.crossfit_result_local)
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V122 requires the V115 Jinan fit")
    crossfit_summary = crossfit.get("crossfit", {})
    row_count = int(crossfit_summary.get("row_count", -1))
    if (
        crossfit.get("protocol") != CROSSFIT_RESULT_PROTOCOL
        or int(crossfit_summary.get("complete_target_prediction_rows", -1))
        != row_count
        or any(
            int(value) != row_count
            for value in crossfit_summary.get(
                "complete_source_prediction_rows", {}
            ).values()
        )
    ):
        raise ValueError("V120 cross-fit result is incomplete")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    fit_sha = _sha256(args.fit_result_local)
    target_audit_sha = _sha256(args.target_cache_audit_local)
    artifact = crossfit["artifact"]
    crossfit_artifact_remote = Path(artifact["path"])
    crossfit_artifact_sha = str(artifact["sha256"])
    remote_inputs = (
        args.fit_result_remote,
        crossfit_artifact_remote,
        args.target_cache_audit_remote,
    )
    expected_hashes = [fit_sha, crossfit_artifact_sha, target_audit_sha]

    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "-d", str(args.target_cache_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        )
    )
    code, _, stderr = scheduler.run_on(
        args.node, preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V122 preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        args.node,
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote V122 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )

    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=fit_sha,
        crossfit_artifact_remote=crossfit_artifact_remote,
        crossfit_artifact_sha256=crossfit_artifact_sha,
        target_cache_root=args.target_cache_root,
        target_cache_audit_remote=args.target_cache_audit_remote,
        remote_output_root=args.remote_output_root,
        node=args.node,
        cache_workers=args.cache_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v122-b100-source-reliability-diagnostic-v1",
        ],
        input=json.dumps([spec]),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "signature": SIGNATURE,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "crossfit_result": str(args.crossfit_result_local.resolve()),
        "crossfit_result_sha256": _sha256(args.crossfit_result_local),
        "crossfit_artifact_sha256": crossfit_artifact_sha,
        "remote_evidence_sha256": remote_hashes,
        "remote_output_root": str(args.remote_output_root),
        "execution_node": str(args.node),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V122 reliability diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
