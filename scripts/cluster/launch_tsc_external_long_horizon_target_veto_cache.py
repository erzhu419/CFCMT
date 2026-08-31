#!/usr/bin/env python3
"""Collect the v65 full-horizon target-veto cache on six CPU nodes."""

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

from scripts.cluster.authorize_tsc_external_long_horizon_target_veto_cache import (  # noqa: E402
    AUDIT_DECISION,
    AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (  # noqa: E402
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


LAUNCH_PROTOCOL = "v65r61-external-v9-long-horizon-target-veto-cache-launch-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
DIRECT_TIMEOUT_SEC = 28800


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
    collection = dict(protocol.get("long_horizon_collection", {}))
    if (
        protocol.get("protocol") != PROTOCOL
        or authorization.get("protocol") != AUDIT_PROTOCOL
        or authorization.get("status") != "PASS"
        or authorization.get("decision") != AUDIT_DECISION
        or not bool(authorization.get("gate", {}).get("passed", False))
        or authorization.get("frozen_protocol_sha256") != _sha256(protocol_path)
        or any(
            int(value) != 0
            for value in authorization.get(
                "future_file_counts_at_authorization", {}
            ).values()
        )
    ):
        raise ValueError("v65 cache authorization changed")
    seeds = tuple(int(value) for value in collection["seeds"])
    scenarios = tuple(
        str(scenario)
        for city in sorted(protocol["city_scenarios"])
        for scenario in protocol["city_scenarios"][city]
    )
    shards = int(collection["collection_shards_per_scenario_seed"])
    if (
        len(seeds) != len(NODES)
        or len(seeds) != len(set(seeds))
        or len(scenarios) != 4
        or int(workers) != len(scenarios) * shards
        or int(workers) != int(collection["workers_per_seed"])
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
        or int(ram_mb) < 65536
    ):
        raise ValueError("v65 cache resource or matrix contract changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    for node, seed in zip(NODES, seeds):
        name = f"seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite v65 cache task result: {local_result}"
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
                str(collection["duration_sec"]),
                "--control-interval-sec",
                str(collection["control_interval_sec"]),
                "--warmup-sec",
                str(collection["warmup_sec"]),
                "--max-focal-tls",
                str(collection["max_focal_tls"]),
                "--counterfactual-horizon-intervals",
                str(collection["counterfactual_horizon_intervals"]),
                "--collection-shards",
                str(shards),
                "--workers",
                str(workers),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                str(collection["behavior_policy"]),
                "--min-tls-coverage",
                "1",
                "--expected-sumo-version",
                str(protocol["environment"]["sumo_version"]),
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
                "description": f"CFCMT v65 long-horizon target veto cache seed {seed}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v65r61/target-veto-cache/seed-{seed}",
                "resource_family": "CFCMT-v65-long-horizon-target-veto-cache-v1",
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


def _assert_remote_outputs_absent(
    *, remote_cache_root: Path, remote_results_root: Path, scheduler_path: Path
) -> None:
    scheduler = _load_scheduler(scheduler_path)
    command = (
        f"test ! -e {shlex.quote(str(remote_cache_root))} && "
        f"test ! -e {shlex.quote(str(remote_results_root))}"
    )
    code, _, stderr = scheduler.run_on(
        "node001", command, timeout=120, check=False
    )
    if int(code) != 0:
        raise FileExistsError(
            "refusing to overwrite v65 remote output roots: " + str(stderr)
        )


def _sync_shared_cache(
    *, remote_cache_root: Path, local_cache_root: Path, scheduler_path: Path
) -> None:
    if local_cache_root.exists():
        raise FileExistsError(
            f"refusing to overwrite v65 local cache: {local_cache_root}"
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
        raise RuntimeError(f"v65 cache sync failed: {completed.stderr}")


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
    parser.add_argument("--ram-mb", type=int, default=98304)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-cache-sync", action="store_true")
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
            _assert_remote_outputs_absent(
                remote_cache_root=args.remote_cache_root,
                remote_results_root=args.remote_results_root,
                scheduler_path=args.scheduler,
            )
            direct_results = _run_direct_specs(
                specs,
                scheduler_path=args.scheduler,
                timeout=DIRECT_TIMEOUT_SEC,
            )
            if not args.skip_cache_sync:
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
                    "cache_synced": not args.skip_cache_sync,
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
                    "CFCMT-v65r61-long-horizon-target-veto-cache",
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
                raise RuntimeError("v65 cache submission failed")
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
