#!/usr/bin/env python3
"""Submit seed-7079 external caches after the joint freeze is authorized."""

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

from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
    PROTOCOL_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


AUTHORIZATION_DECISION = "authorize_external_seed7079_collection"
ASSIGNMENTS = (
    ("node001", "la_1x4", 7079),
    ("node002", "jinan_3x4_real", 7079),
    ("node005", "jinan_3x4_real_2000", 7079),
    ("node006", "jinan_3x4_real_2500", 7079),
)


def build_specs(
    *,
    snapshot_root: Path,
    authorization_audit_local: Path,
    authorization_audit_remote: Path,
    conversion_root: Path,
    conversion_manifest_sha256: str,
    conversion_tree_sha256: str,
    remote_cache_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    collection_shards: int,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], str]:
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    target = dict(protocol["target_protocol"])
    expected_pairs = {
        (str(scenario), int(seed))
        for scenarios in target["external_city_scenarios"].values()
        for scenario in scenarios
        for seed in target["evaluation_seeds"]
    }
    assigned_pairs = {(scenario, seed) for _, scenario, seed in ASSIGNMENTS}
    if assigned_pairs != expected_pairs or len(assigned_pairs) != len(ASSIGNMENTS):
        raise ValueError("external evaluation assignments changed")
    if int(collection_shards) != int(
        target["counterfactual_collection_shards_per_scenario_seed"]
    ):
        raise ValueError("external evaluation shard count changed")
    if int(workers) != int(collection_shards):
        raise ValueError("external evaluation requires one worker per shard")

    authorization = _read_json(authorization_audit_local)
    if (
        authorization.get("status") != "PASS"
        or not bool(authorization.get("gate", {}).get("passed", False))
        or authorization.get("decision") != AUTHORIZATION_DECISION
        or int(authorization.get("evaluation_cache_file_count_at_freeze", -1)) != 0
    ):
        raise ValueError("joint freeze audit does not authorize evaluation cache")
    authorization_sha256 = _sha256(authorization_audit_local)

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / EXTERNAL_MANIFEST_RELATIVE
    specs = []
    for node, scenario, seed in ASSIGNMENTS:
        name = f"{scenario}_seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite external evaluation result: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_cache",
                "--manifest",
                str(manifest),
                "--scenarios",
                scenario,
                "--seeds",
                str(seed),
                "--duration-sec",
                "600",
                "--control-interval-sec",
                "10",
                "--warmup-sec",
                "60",
                "--max-focal-tls",
                "4",
                "--counterfactual-horizon-intervals",
                "6",
                "--collection-shards",
                str(collection_shards),
                "--workers",
                str(workers),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                "phase_pressure",
                "--min-tls-coverage",
                "1",
                "--expected-sumo-version",
                "1.22.0",
                "--expected-input-root",
                str(conversion_root),
                "--expected-input-manifest-sha256",
                str(conversion_manifest_sha256),
                "--expected-input-tree-sha256",
                str(conversion_tree_sha256),
                "--authorization-audit",
                str(authorization_audit_remote),
                "--expected-authorization-sha256",
                authorization_sha256,
                "--required-authorization-decision",
                AUTHORIZATION_DECISION,
                "--out",
                str(remote_result / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT v42r38 held-out evaluation cache {scenario} seed {seed}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/v42r38/evaluation-cache-v8/{scenario}/seed-{seed}"
                ),
                "resource_family": "CFCMT-external-evaluation-cache-v8",
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
                },
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
    return specs, authorization_sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-audit-local", type=Path, required=True)
    parser.add_argument("--authorization-audit-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--collection-shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu-cores", type=int, default=16)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, authorization_sha256 = build_specs(
        snapshot_root=snapshot_root,
        authorization_audit_local=args.authorization_audit_local,
        authorization_audit_remote=args.authorization_audit_remote,
        conversion_root=args.conversion_root,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        remote_cache_root=args.remote_cache_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        collection_shards=args.collection_shards,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    payload: dict[str, Any] = {
        "protocol": "v42r38-external-evaluation-cache-v8-submission-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "authorization_audit_local": str(args.authorization_audit_local.resolve()),
        "authorization_audit_remote": str(args.authorization_audit_remote),
        "authorization_audit_sha256": authorization_sha256,
        "authorization_decision": AUTHORIZATION_DECISION,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "remote_cache_root": str(args.remote_cache_root),
        "collection_shards": int(args.collection_shards),
        "workers_per_task": int(args.workers),
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
                "CFCMT-v42r38-external-evaluation-cache-v8",
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
            raise RuntimeError("external evaluation cache submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "parallel_libsumo_workers": len(specs) * int(args.workers),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
