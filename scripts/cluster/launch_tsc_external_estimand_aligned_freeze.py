#!/usr/bin/env python3
"""Submit two estimand-aligned external-city method refits."""

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

from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (  # noqa: E402
    RESULT_PROTOCOL,
    V9_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL,
    V9_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
    SOURCE_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL_RELATIVE,
    V9_EXTERNAL_MANIFEST_RELATIVE,
    V9_PROTOCOL_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v46r42-external-oof-validated-support-freeze-submission-v1"
V9_LAUNCH_PROTOCOL = (
    "v55r51-external-v9-oof-validated-support-freeze-submission-v1"
)
CITY_NODES = {"los_angeles": "node003", "jinan": "node004"}


def build_specs(
    *,
    snapshot_root: Path,
    source_tree_sha256: str,
    source_cache_root: Path,
    source_baseline_result: Path,
    development_audit: Path,
    diagnostic_failure_result: Path,
    parent_joint_audit_local: Path,
    parent_joint_audit_remote: Path,
    external_cache_root: Path,
    conversion_root: Path,
    expected_external_cache_sha256: str,
    remote_results_root: Path,
    local_results_root: Path,
    workers: int,
    fit_workers: int,
    cpu_cores: int,
    ram_mb: int,
    generation: str = "v8",
    failed_v52_audit: Path | None = None,
    network_admission_audit: Path | None = None,
    counterfactual_cache_audit: Path | None = None,
) -> list[dict[str, Any]]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported aligned-freeze generation: {generation}")
    protocol_relative = (
        PROTOCOL_RELATIVE if generation == "v8" else V9_PROTOCOL_RELATIVE
    )
    external_manifest_relative = (
        EXTERNAL_MANIFEST_RELATIVE
        if generation == "v8"
        else V9_EXTERNAL_MANIFEST_RELATIVE
    )
    expected_protocol = PROTOCOL if generation == "v8" else V9_PROTOCOL
    protocol = _read_json(PROJECT_ROOT / protocol_relative)
    if protocol.get("protocol") != expected_protocol:
        raise ValueError("estimand-aligned parent protocol changed")
    parent_path = PROJECT_ROOT / str(protocol["parent_protocol"]["path"])
    if _sha256(parent_path) != str(protocol["parent_protocol"]["sha256"]):
        raise ValueError("estimand-aligned source protocol changed")
    parent = _read_json(parent_path)
    source = dict(parent["source_evidence"])
    cities = dict(protocol["target_protocol"]["external_city_scenarios"])
    if set(cities) != set(CITY_NODES):
        raise ValueError("estimand-aligned city assignment changed")
    parent_joint_sha256 = _sha256(parent_joint_audit_local)

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    source_manifest = snapshot_root / SOURCE_MANIFEST_RELATIVE
    external_manifest = snapshot_root / external_manifest_relative
    protocol_spec = snapshot_root / protocol_relative
    if generation == "v9" and any(
        value is None
        for value in (
            failed_v52_audit,
            network_admission_audit,
            counterfactual_cache_audit,
        )
    ):
        raise ValueError("v9 aligned freeze requires all repair evidence paths")
    specs = []
    for city, node in CITY_NODES.items():
        remote_result = remote_results_root / city
        local_result = local_results_root / city
        if (local_result / "freeze.json").exists() or (local_result / "model.pkl").exists():
            raise FileExistsError(f"refusing to overwrite aligned freeze: {local_result}")
        command_args = [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze",
                "--source-cache-root",
                str(source_cache_root),
                "--source-baseline-result",
                str(source_baseline_result),
                "--source-manifest",
                str(source_manifest),
                "--external-cache-root",
                str(external_cache_root),
                "--external-manifest",
                str(external_manifest),
                "--conversion-root",
                str(conversion_root),
                "--protocol-spec",
                str(protocol_spec),
                "--development-audit",
                str(development_audit),
                "--diagnostic-failure-result",
                str(diagnostic_failure_result),
                "--parent-joint-audit",
                str(parent_joint_audit_remote),
                "--city",
                city,
                "--expected-source-cache-sha256",
                str(source["counterfactual_cache_sha256"]),
                "--expected-source-baseline-sha256",
                str(source["source_rule_baseline_sha256"]),
                "--expected-external-cache-sha256",
                str(expected_external_cache_sha256),
                "--expected-parent-joint-audit-sha256",
                parent_joint_sha256,
                "--expected-source-tree-sha256",
                str(source_tree_sha256),
                "--workers",
                str(workers),
                "--fit-workers",
                str(fit_workers),
                "--model-out",
                str(remote_result / "model.pkl"),
                "--out",
                str(remote_result / "freeze.json"),
            ]
        if generation == "v9":
            command_args.extend(
                [
                    "--failed-v52-audit",
                    str(failed_v52_audit),
                    "--network-admission-audit",
                    str(network_admission_audit),
                    "--counterfactual-cache-audit",
                    str(counterfactual_cache_audit),
                ]
            )
        command = shlex.join(command_args)
        revision = "v46r42" if generation == "v8" else "v55r51"
        specs.append(
            {
                "description": f"CFCMT {revision} OOF-validated support freeze {city}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/{revision}/oof-validated-support-freeze-{generation}/{city}"
                ),
                "resource_family": (
                    f"CFCMT-external-oof-validated-support-freeze-{generation}-v1"
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
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-baseline-result", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--diagnostic-failure-result", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-local", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--cpu-cores", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
    parser.add_argument("--failed-v52-audit", type=Path)
    parser.add_argument("--network-admission-audit", type=Path)
    parser.add_argument("--counterfactual-cache-audit", type=Path)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        source_tree_sha256=str(stage["source_tree_sha256"]),
        source_cache_root=args.source_cache_root,
        source_baseline_result=args.source_baseline_result,
        development_audit=args.development_audit,
        diagnostic_failure_result=args.diagnostic_failure_result,
        parent_joint_audit_local=args.parent_joint_audit_local,
        parent_joint_audit_remote=args.parent_joint_audit_remote,
        external_cache_root=args.external_cache_root,
        conversion_root=args.conversion_root,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        workers=args.workers,
        fit_workers=args.fit_workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
        generation=args.generation,
        failed_v52_audit=args.failed_v52_audit,
        network_admission_audit=args.network_admission_audit,
        counterfactual_cache_audit=args.counterfactual_cache_audit,
    )
    payload: dict[str, Any] = {
        "protocol": (
            LAUNCH_PROTOCOL
            if args.generation == "v8"
            else V9_LAUNCH_PROTOCOL
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
        "source_tree_sha256": stage["source_tree_sha256"],
        "parent_joint_audit_sha256": _sha256(args.parent_joint_audit_local),
        "task_count": len(specs),
        "generation": args.generation,
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
                    "CFCMT-v46r42-oof-validated-support-freeze"
                    if args.generation == "v8"
                    else "CFCMT-v55r51-external-v9-oof-validated-support-freeze"
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
            raise RuntimeError("estimand-aligned freeze submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {"submitted": payload["submitted"], "task_count": len(specs), "out": str(args.out)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
