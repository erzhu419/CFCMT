#!/usr/bin/env python3
"""Submit the fresh seed-8171 cache after the v43 joint freeze passes."""

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
    EVALUATION_SEED,
    PROTOCOL,
    V9_EVALUATION_SEED,
    V9_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_full_budget_freeze import (  # noqa: E402
    V9_EXTERNAL_MANIFEST_RELATIVE,
    V9_PROTOCOL_RELATIVE,
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
AUTHORIZATION_DECISION = "authorize_external_seed8171_collection"
V9_AUTHORIZATION_DECISION = "authorize_external_v9_seed9277_collection"
ASSIGNMENTS = (
    ("node001", "la_1x4", EVALUATION_SEED),
    ("node002", "jinan_3x4_real", EVALUATION_SEED),
    ("node005", "jinan_3x4_real_2000", EVALUATION_SEED),
    ("node006", "jinan_3x4_real_2500", EVALUATION_SEED),
)
V9_ASSIGNMENTS = (
    ("node001", "la_1x4", V9_EVALUATION_SEED),
    ("node002", "jinan_3x4_real", V9_EVALUATION_SEED),
    ("node005", "jinan_3x4_real_2000", V9_EVALUATION_SEED),
    ("node006", "jinan_3x4_real_2500", V9_EVALUATION_SEED),
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
    generation: str = "v8",
) -> tuple[list[dict[str, Any]], str]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported evaluation-cache generation: {generation}")
    protocol_relative = (
        PROTOCOL_RELATIVE if generation == "v8" else V9_PROTOCOL_RELATIVE
    )
    manifest_relative = (
        EXTERNAL_MANIFEST_RELATIVE
        if generation == "v8"
        else V9_EXTERNAL_MANIFEST_RELATIVE
    )
    expected_protocol = PROTOCOL if generation == "v8" else V9_PROTOCOL
    authorization_decision = (
        AUTHORIZATION_DECISION
        if generation == "v8"
        else V9_AUTHORIZATION_DECISION
    )
    assignments = ASSIGNMENTS if generation == "v8" else V9_ASSIGNMENTS
    revision = "v43r39" if generation == "v8" else "v57r53"
    protocol = _read_json(PROJECT_ROOT / protocol_relative)
    if protocol.get("protocol") != expected_protocol:
        raise ValueError("full-budget evaluation protocol changed")
    target = dict(protocol["target_protocol"])
    expected_pairs = {
        (str(scenario), int(seed))
        for scenarios in target["external_city_scenarios"].values()
        for scenario in scenarios
        for seed in target["evaluation_seeds"]
    }
    assigned_pairs = {(scenario, seed) for _, scenario, seed in assignments}
    if assigned_pairs != expected_pairs or len(assigned_pairs) != len(assignments):
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
        or authorization.get("decision") != authorization_decision
        or int(authorization.get("evaluation_cache_file_count_at_freeze", -1))
        != 0
    ):
        raise ValueError("joint freeze does not authorize evaluation cache")
    authorization_sha256 = _sha256(authorization_audit_local)

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / manifest_relative
    specs = []
    for node, scenario, seed in assignments:
        name = f"{scenario}_seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite evaluation result: {local_result}"
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
                authorization_decision,
                "--out",
                str(remote_result / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT {revision} held-out evaluation cache {scenario} seed {seed}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/{revision}/evaluation-cache-{generation}-v1/{scenario}/seed-{seed}"
                ),
                "resource_family": (
                    f"CFCMT-external-full-evaluation-cache-{generation}-v1"
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
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
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
        generation=args.generation,
    )
    authorization_decision = (
        AUTHORIZATION_DECISION
        if args.generation == "v8"
        else V9_AUTHORIZATION_DECISION
    )
    evaluation_seed = (
        EVALUATION_SEED if args.generation == "v8" else V9_EVALUATION_SEED
    )
    payload: dict[str, Any] = {
        "protocol": (
            "v43r39-external-evaluation-cache-submission-v1"
            if args.generation == "v8"
            else "v57r53-external-v9-evaluation-cache-submission-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "authorization_audit_local": str(
            args.authorization_audit_local.resolve()
        ),
        "authorization_audit_remote": str(args.authorization_audit_remote),
        "authorization_audit_sha256": authorization_sha256,
        "authorization_decision": authorization_decision,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "remote_cache_root": str(args.remote_cache_root),
        "evaluation_seed": evaluation_seed,
        "generation": args.generation,
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
                (
                    "CFCMT-v43r39-external-evaluation-cache"
                    if args.generation == "v8"
                    else "CFCMT-v57r53-external-v9-evaluation-cache"
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
