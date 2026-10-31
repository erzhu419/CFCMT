#!/usr/bin/env python3
"""Collect the six-seed v61 hierarchical offline cache on six CPU nodes."""

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

from scripts.cluster.audit_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    AUDIT_DECISION,
    AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_protocol import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
    _run_direct_specs,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v61r57-external-v9-hierarchical-offline-cache-submission-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    protocol_path: Path,
    authorization_audit_local: Path,
    authorization_audit_remote: Path,
    conversion_root: Path,
    conversion_manifest_sha256: str,
    conversion_tree_sha256: str,
    remote_cache_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], str]:
    protocol = _read_json(protocol_path)
    authorization = _read_json(authorization_audit_local)
    authorization_sha = _sha256(authorization_audit_local)
    if (
        protocol.get("protocol") != PROTOCOL
        or authorization.get("protocol") != AUDIT_PROTOCOL
        or authorization.get("status") != "PASS"
        or authorization.get("decision") != AUDIT_DECISION
        or not bool(authorization.get("integrity_gate", {}).get("passed", False))
        or authorization_sha
        != str(protocol["hierarchical_joint_audit"]["sha256"])
        or int(authorization.get("future_cache_file_count_at_freeze", -1)) != 0
    ):
        raise ValueError("hierarchical cache authorization changed")
    offline = dict(protocol["fresh_offline_confirmation"])
    seeds = tuple(int(value) for value in offline["seeds"])
    scenarios = tuple(
        str(scenario)
        for city in sorted(protocol["city_scenarios"])
        for scenario in protocol["city_scenarios"][city]
    )
    if len(seeds) != len(NODES) or len(seeds) != len(set(seeds)):
        raise ValueError("hierarchical cache seed-to-node matrix changed")
    if int(workers) != len(scenarios) * int(
        offline["collection_shards_per_scenario_seed"]
    ):
        raise ValueError("hierarchical cache requires one worker per scenario shard")
    if not 1 <= int(workers) <= int(cpu_cores) <= 192:
        raise ValueError("hierarchical cache CPU reservation is invalid")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    for node, seed in zip(NODES, seeds):
        name = f"seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite hierarchical cache result: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_cache",
                "--manifest",
                str(manifest),
                "--scenarios",
                *scenarios,
                "--seeds",
                str(seed),
                "--duration-sec",
                str(offline["duration_sec"]),
                "--control-interval-sec",
                str(offline["control_interval_sec"]),
                "--warmup-sec",
                str(offline["warmup_sec"]),
                "--max-focal-tls",
                "4",
                "--counterfactual-horizon-intervals",
                str(offline["counterfactual_horizon_intervals"]),
                "--collection-shards",
                str(offline["collection_shards_per_scenario_seed"]),
                "--workers",
                str(workers),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                str(offline["behavior_policy"]),
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
                authorization_sha,
                "--required-authorization-decision",
                AUDIT_DECISION,
                "--out",
                str(remote_result / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": f"CFCMT v61 fresh offline cache seed {seed}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v61r57/hierarchical-offline-cache/seed-{seed}",
                "resource_family": "CFCMT-hierarchical-offline-cache-v1",
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
    return specs, authorization_sha


def _sync_shared_cache(
    *, remote_cache_root: Path, local_cache_root: Path, scheduler_path: Path
) -> None:
    if local_cache_root.exists():
        raise FileExistsError(
            f"refusing to overwrite hierarchical local cache: {local_cache_root}"
        )
    scheduler = _load_scheduler(scheduler_path)
    target = scheduler._ssh_target_for_node("node001")
    ssh_shell = scheduler._ssh_rsync_shell_for_node("node001")
    local_cache_root.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            ssh_shell,
            f"{target}:{remote_cache_root}/",
            f"{local_cache_root}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"hierarchical cache sync failed: {completed.stderr}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization-audit-local", type=Path, required=True)
    parser.add_argument("--authorization-audit-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--local-cache-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--cpu-cores", type=int, default=68)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--direct", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    specs, authorization_sha = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        protocol_path=args.protocol,
        authorization_audit_local=args.authorization_audit_local,
        authorization_audit_remote=args.authorization_audit_remote,
        conversion_root=args.conversion_root,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        remote_cache_root=args.remote_cache_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "frozen_protocol_path": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "authorization_audit_sha256": authorization_sha,
        "authorization_decision": AUDIT_DECISION,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "remote_cache_root": str(args.remote_cache_root),
        "local_cache_root": str(args.local_cache_root.resolve()),
        "collection_shards": 16,
        "workers_per_task": int(args.workers),
        "task_count": len(specs),
        "specs": specs,
    }
    try:
        if args.direct:
            direct_results = _run_direct_specs(
                specs, scheduler_path=args.scheduler, timeout=7200
            )
            _sync_shared_cache(
                remote_cache_root=args.remote_cache_root,
                local_cache_root=args.local_cache_root,
                scheduler_path=args.scheduler,
            )
            payload.update(
                {
                    "submitted": True,
                    "execution_mode": "direct_run_on_six_physical_nodes",
                    "direct_results": direct_results,
                }
            )
        elif args.submit:
            completed = subprocess.run(
                [
                    str(args.scheduler),
                    "submit-jsonl",
                    "--stdin",
                    "--trusted",
                    "--json",
                    "--intent-label",
                    "CFCMT-v61r57-hierarchical-offline-cache",
                ],
                input=json.dumps(specs),
                text=True,
                capture_output=True,
                check=False,
            )
            payload.update(
                {
                    "submitted": completed.returncode == 0,
                    "execution_mode": "scheduler_queue",
                    "scheduler_returncode": int(completed.returncode),
                    "scheduler_stdout": completed.stdout,
                    "scheduler_stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                raise RuntimeError("hierarchical cache submission failed")
    except Exception as exc:
        payload["execution_error"] = repr(exc)
        _atomic_json(args.out, payload)
        raise
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
