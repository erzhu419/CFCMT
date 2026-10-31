#!/usr/bin/env python3
"""Submit the two-city v47 pressure-regularized artifact freeze."""

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

from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    ALIGNED_AUDIT_DECISION,
    ALIGNED_PROTOCOL,
    V9_ALIGNED_AUDIT_DECISION,
    V9_ALIGNED_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze import (  # noqa: E402
    RESULT_PROTOCOL,
    V9_RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_estimand_aligned_rollouts import (  # noqa: E402
    ALIGNED_PROTOCOL_RELATIVE,
)
from scripts.cluster.launch_tsc_external_full_budget_freeze import (  # noqa: E402
    V9_EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v47r43-external-oof-pressure-regularized-freeze-submission-v1"
V9_LAUNCH_PROTOCOL = (
    "v56r52-external-v9-oof-pressure-regularized-freeze-submission-v1"
)
V9_ALIGNED_PROTOCOL_RELATIVE = (
    "cf_h2o/config/traffic_signal_tsc_v42_external_v9_estimand_aligned_development.json"
)
ASSIGNMENTS = (("node001", "los_angeles"), ("node002", "jinan"))
V9_ASSIGNMENTS = (("node003", "los_angeles"), ("node004", "jinan"))


def build_specs(
    *,
    snapshot_root: Path,
    aligned_joint_audit_local: Path,
    aligned_joint_audit_remote: Path,
    aligned_freeze_root: Path,
    external_cache_root: Path,
    conversion_root: Path,
    expected_external_cache_sha256: str,
    remote_results_root: Path,
    local_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
    generation: str = "v8",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported pressure-freeze generation: {generation}")
    aligned_protocol_relative = (
        ALIGNED_PROTOCOL_RELATIVE
        if generation == "v8"
        else V9_ALIGNED_PROTOCOL_RELATIVE
    )
    external_manifest_relative = (
        EXTERNAL_MANIFEST_RELATIVE
        if generation == "v8"
        else V9_EXTERNAL_MANIFEST_RELATIVE
    )
    expected_aligned_protocol = (
        ALIGNED_PROTOCOL if generation == "v8" else V9_ALIGNED_PROTOCOL
    )
    expected_audit_decision = (
        ALIGNED_AUDIT_DECISION
        if generation == "v8"
        else V9_ALIGNED_AUDIT_DECISION
    )
    assignments = ASSIGNMENTS if generation == "v8" else V9_ASSIGNMENTS
    revision = "v47r43" if generation == "v8" else "v56r52"
    protocol = _read_json(PROJECT_ROOT / aligned_protocol_relative)
    joint = _read_json(aligned_joint_audit_local)
    joint_sha = _sha256(aligned_joint_audit_local)
    if (
        protocol.get("protocol") != expected_aligned_protocol
        or joint.get("status") != "PASS"
        or joint.get("decision") != expected_audit_decision
        or joint_sha != str(protocol["estimand_aligned_joint_audit"]["sha256"])
    ):
        raise ValueError("pressure-regularized freeze authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    aligned_protocol = snapshot_root / aligned_protocol_relative
    external_manifest = snapshot_root / external_manifest_relative
    specs = []
    for node, city in assignments:
        remote_result = Path(remote_results_root) / city
        local_result = Path(local_results_root) / city
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite pressure-regularized freeze: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze",
                "--aligned-protocol",
                str(aligned_protocol),
                "--aligned-joint-audit",
                str(aligned_joint_audit_remote),
                "--expected-aligned-joint-audit-sha256",
                joint_sha,
                "--aligned-freeze-root",
                str(aligned_freeze_root),
                "--external-cache-root",
                str(external_cache_root),
                "--external-manifest",
                str(external_manifest),
                "--conversion-root",
                str(conversion_root),
                "--expected-external-cache-sha256",
                str(expected_external_cache_sha256),
                "--expected-source-tree-sha256",
                str(_read_json(snapshot_root / ".cfcmt_snapshot.json")["source_tree_sha256"])
                if (snapshot_root / ".cfcmt_snapshot.json").is_file()
                else "SOURCE_TREE_FROM_LAUNCH",
                "--city",
                city,
                "--workers",
                str(workers),
                "--model-out",
                str(remote_result / "model.pkl"),
                "--out",
                str(remote_result / "freeze.json"),
            ]
        )
        specs.append(
            {
                "description": f"CFCMT {revision} pressure-regularized freeze {city}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/{revision}/pressure-regularized-freeze-{generation}/{city}"
                ),
                "resource_family": (
                    f"CFCMT-external-pressure-regularized-freeze-{generation}-v1"
                ),
                "vram": 0,
                "ram_mb": int(ram_mb),
                "cpu": int(cpu_cores),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
    return specs, {"aligned_joint_audit": joint_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-local", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-cores", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, hashes = build_specs(
        snapshot_root=snapshot_root,
        aligned_joint_audit_local=args.aligned_joint_audit_local,
        aligned_joint_audit_remote=args.aligned_joint_audit_remote,
        aligned_freeze_root=args.aligned_freeze_root,
        external_cache_root=args.external_cache_root,
        conversion_root=args.conversion_root,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
        generation=args.generation,
    )
    source_tree_sha256 = str(stage["source_tree_sha256"])
    for spec in specs:
        spec["cmd"] = str(spec["cmd"]).replace(
            "SOURCE_TREE_FROM_LAUNCH", source_tree_sha256
        )
    payload: dict[str, Any] = {
        "protocol": (
            LAUNCH_PROTOCOL if args.generation == "v8" else V9_LAUNCH_PROTOCOL
        ),
        "result_protocol": (
            RESULT_PROTOCOL if args.generation == "v8" else V9_RESULT_PROTOCOL
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": source_tree_sha256,
        "generation": args.generation,
        "input_hashes": {
            **hashes,
            "external_cache": args.expected_external_cache_sha256,
        },
        "task_count": len(specs),
        "specs": specs,
    }
    if args.submit:
        completed = subprocess.run(
            [
                str(args.scheduler),
                "submit-jsonl",
                "--stdin",
                "--trusted",
                "--json",
                "--intent-label",
                (
                    "CFCMT-v47r43-pressure-regularized-freeze"
                    if args.generation == "v8"
                    else "CFCMT-v56r52-external-v9-pressure-regularized-freeze"
                ),
            ],
            input=json.dumps(specs),
            text=True,
            capture_output=True,
            check=False,
        )
        payload.update(
            {
                "submitted": completed.returncode == 0,
                "scheduler_returncode": int(completed.returncode),
                "scheduler_stdout": completed.stdout,
                "scheduler_stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            _atomic_json(args.out, payload)
            raise RuntimeError("pressure-regularized freeze submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
