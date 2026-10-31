#!/usr/bin/env python3
"""Submit the two v43 full-budget seed-blocked OOF plus refit tasks."""

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

from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL,
    V9_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
    SOURCE_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json"
)
V9_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v41_external_v9_full_budget_refit.json"
)
V9_EXTERNAL_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
CITY_NODES = {
    "los_angeles": "node003",
    "jinan": "node004",
}


def build_specs(
    *,
    snapshot_root: Path,
    source_tree_sha256: str,
    source_cache_root: Path,
    source_baseline_result: Path,
    development_audit: Path,
    diagnostic_failure_result: Path,
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
        raise ValueError(f"unsupported full-budget generation: {generation}")
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
        raise ValueError("external full-budget protocol changed")
    parent_spec = dict(protocol["parent_protocol"])
    parent_path = PROJECT_ROOT / str(parent_spec["path"])
    if _sha256(parent_path) != str(parent_spec["sha256"]):
        raise ValueError("external full-budget parent protocol changed")
    parent = _read_json(parent_path)
    source = dict(parent["source_evidence"])
    cities = dict(protocol["target_protocol"]["external_city_scenarios"])
    if set(cities) != set(CITY_NODES):
        raise ValueError("external full-budget city assignment changed")

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
        raise ValueError("v9 full-budget refit requires all repair evidence paths")
    specs = []
    for city, node in CITY_NODES.items():
        remote_result = remote_results_root / city
        local_result = local_results_root / city
        outputs = (
            local_result / "method_freeze.json",
            local_result / "baseline_freeze.json",
            local_result / "method_model.pkl",
            local_result / "baseline_model.pkl",
        )
        if any(path.exists() for path in outputs):
            raise FileExistsError(
                f"refusing to overwrite external full-budget freeze: {local_result}"
            )
        command_args = [
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_external_full_budget_freeze",
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
            "--city",
            city,
            "--expected-source-cache-sha256",
            str(source["counterfactual_cache_sha256"]),
            "--expected-source-baseline-sha256",
            str(source["source_rule_baseline_sha256"]),
            "--expected-external-cache-sha256",
            str(expected_external_cache_sha256),
            "--expected-source-tree-sha256",
            str(source_tree_sha256),
            "--workers",
            str(workers),
            "--fit-workers",
            str(fit_workers),
            "--method-model-out",
            str(remote_result / "method_model.pkl"),
            "--baseline-model-out",
            str(remote_result / "baseline_model.pkl"),
            "--method-out",
            str(remote_result / "method_freeze.json"),
            "--baseline-out",
            str(remote_result / "baseline_freeze.json"),
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
        revision = "v43r39" if generation == "v8" else "v54r50"
        specs.append(
            {
                "description": f"CFCMT {revision} full-budget freeze {city}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/{revision}/full-budget-freeze-{generation}-v1/{city}"
                ),
                "resource_family": (
                    f"CFCMT-external-full-budget-freeze-{generation}-v1"
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
    parser.add_argument(
        "--diagnostic-failure-result", type=Path, required=True
    )
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
            "v43r39-external-full-budget-freeze-submission-v1"
            if args.generation == "v8"
            else "v54r50-external-v9-full-budget-freeze-submission-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "protocol_spec": str(
            (
                PROJECT_ROOT
                / (
                    PROTOCOL_RELATIVE
                    if args.generation == "v8"
                    else V9_PROTOCOL_RELATIVE
                )
            ).resolve()
        ),
        "protocol_spec_sha256": _sha256(
            PROJECT_ROOT
            / (
                PROTOCOL_RELATIVE
                if args.generation == "v8"
                else V9_PROTOCOL_RELATIVE
            )
        ),
        "source_cache_root": str(args.source_cache_root),
        "source_baseline_result": str(args.source_baseline_result),
        "development_audit": str(args.development_audit),
        "diagnostic_failure_result": str(args.diagnostic_failure_result),
        "external_cache_root": str(args.external_cache_root),
        "external_cache_sha256": args.expected_external_cache_sha256,
        "conversion_root": str(args.conversion_root),
        "fit_workers_per_city": int(args.fit_workers),
        "generation": args.generation,
        "failed_v52_audit": str(args.failed_v52_audit)
        if args.failed_v52_audit
        else None,
        "network_admission_audit": str(args.network_admission_audit)
        if args.network_admission_audit
        else None,
        "counterfactual_cache_audit": str(args.counterfactual_cache_audit)
        if args.counterfactual_cache_audit
        else None,
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
                    "CFCMT-v43r39-external-full-budget-freeze"
                    if args.generation == "v8"
                    else "CFCMT-v54r50-external-v9-full-budget-freeze"
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
            raise RuntimeError("external full-budget freeze submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "parallel_fit_processes": len(specs) * int(args.fit_workers),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
