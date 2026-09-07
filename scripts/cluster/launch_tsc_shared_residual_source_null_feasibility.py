#!/usr/bin/env python3
"""Submit the frozen V129 shared-residual source-null feasibility test."""

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
from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as SELECTOR_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_anchored_source_null_residual_feasibility import (  # noqa: E402
    ALLOWED_NODES,
    RAM_MB,
    TARGET_MANIFEST_RELATIVE,
    build_spec as _build_v128_spec,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import (  # noqa: E402
    SEEDS,
)


LAUNCH_PROTOCOL = "tsc-v129-shared-residual-source-null-launch-v2"
SIGNATURE = "CFCMT/v129/shared-target-residual-source-null-b500-v2-execution-safe"


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    prediction_artifact_remote: Path,
    prediction_artifact_sha256: str,
    selector_cache_root: Path,
    selector_cache_audit_remote: Path,
    selector_cache_audit_sha256: str,
    remote_output_root: Path,
    nodes: Sequence[str],
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    spec = _build_v128_spec(
        snapshot_root=snapshot_root,
        conversion_root=conversion_root,
        prediction_artifact_remote=prediction_artifact_remote,
        prediction_artifact_sha256=prediction_artifact_sha256,
        selector_cache_root=selector_cache_root,
        selector_cache_audit_remote=selector_cache_audit_remote,
        selector_cache_audit_sha256=selector_cache_audit_sha256,
        remote_output_root=remote_output_root,
        nodes=nodes,
        cache_workers=cache_workers,
        fold_workers=fold_workers,
    )
    old_module = (
        "cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility"
    )
    new_module = (
        "cf_h2o.eval.traffic_signal_shared_residual_source_null_feasibility"
    )
    if spec["cmd"].count(old_module) != 1:
        raise ValueError("V128 base command no longer has one replaceable module")
    spec.update(
        {
            "description": (
                "CFCMT V129 shared target-residual source-null feasibility"
            ),
            "cmd": str(spec["cmd"]).replace(old_module, new_module),
            "signature": SIGNATURE,
            "resource_family": "CFCMT-v129-shared-residual-source-null",
        }
    )
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--prediction-artifact-remote", type=Path, required=True)
    parser.add_argument("--prediction-artifact-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-local", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--nodes", default=",".join(ALLOWED_NODES))
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V129 launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V129 cache workers must be in [1, 20]")
    if not 1 <= int(args.fold_workers) <= 20:
        raise ValueError("V129 fold workers must be in [1, 20]")
    if len(str(args.prediction_artifact_sha256)) != 64:
        raise ValueError("V129 prediction artifact SHA-256 is malformed")
    nodes = tuple(value.strip() for value in args.nodes.split(",") if value.strip())
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("V129 execution nodes must be nonempty and unique")
    if not set(nodes).issubset(ALLOWED_NODES):
        raise ValueError("V129 execution nodes include a non-admitted CPU node")

    selector_audit = _read_json(args.selector_cache_audit_local)
    if (
        selector_audit.get("protocol") != SELECTOR_AUDIT_PROTOCOL
        or selector_audit.get("status") != "PASS"
        or [int(value) for value in selector_audit.get("seeds", ())]
        != list(SEEDS)
    ):
        raise ValueError("V129 selector cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    selector_sha = _sha256(args.selector_cache_audit_local)
    remote_inputs = (
        args.prediction_artifact_remote,
        args.selector_cache_audit_remote,
    )
    expected_hashes = [str(args.prediction_artifact_sha256), selector_sha]
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        )
    )
    for node in nodes:
        code, _, stderr = scheduler.run_on(
            node,
            preflight,
            timeout=120,
            check=False,
        )
        if int(code) != 0:
            raise RuntimeError(f"remote V129 preflight failed on {node}: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        nodes[0],
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0]
        for line in str(hash_stdout).splitlines()
        if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote V129 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        prediction_artifact_remote=args.prediction_artifact_remote,
        prediction_artifact_sha256=str(args.prediction_artifact_sha256),
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        selector_cache_audit_sha256=selector_sha,
        remote_output_root=args.remote_output_root,
        nodes=nodes,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v129-shared-residual-source-null-v1",
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
        "remote_evidence_sha256": remote_hashes,
        "remote_output_root": str(args.remote_output_root),
        "execution_nodes": list(nodes),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V129 shared residual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
