#!/usr/bin/env python3
"""Collect the frozen 22-seed waiting-aligned cache on six CPU nodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
    _run_direct_specs,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "v83r79-waiting-aligned-cache-six-node-launch-v1"
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v67_external_v9_"
    "waiting_aligned_redevelopment_cache.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    protocol_path: Path,
    conversion_root: Path,
    remote_cache_root: Path,
    remote_task_root: Path,
    local_task_root: Path,
) -> list[dict[str, Any]]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("waiting-aligned redevelopment protocol changed")
    collection = dict(protocol["collection"])
    environment = dict(protocol["environment"])
    if (
        str(remote_cache_root)
        != str(protocol["output_roots"]["remote_cache_root"])
        or str(remote_task_root)
        != str(protocol["output_roots"]["remote_task_root"])
    ):
        raise ValueError("waiting-aligned remote output root changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    for node, seeds in collection["node_assignments"].items():
        values = [int(seed) for seed in seeds]
        workers = int(collection["workers_by_node"][node])
        remote_result = remote_task_root / node
        local_result = local_task_root / node
        if local_result.exists():
            raise FileExistsError(f"refusing to overwrite local cache task: {local_result}")
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_cache",
                "--manifest",
                str(manifest),
                "--scenarios",
                str(collection["scenario"]),
                "--seeds",
                *[str(seed) for seed in values],
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
                str(collection["collection_shards_per_seed"]),
                "--workers",
                str(workers),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                str(collection["behavior_policy"]),
                "--counterfactual-cost-mode",
                str(collection["counterfactual_cost_mode"]),
                "--min-tls-coverage",
                "1",
                "--expected-sumo-version",
                str(environment["sumo_version"]),
                "--expected-input-root",
                str(conversion_root),
                "--expected-input-manifest-sha256",
                str(environment["conversion_manifest_sha256"]),
                "--expected-input-tree-sha256",
                str(environment["conversion_tree_sha256"]),
                "--out",
                str(remote_result / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": f"waiting-aligned cache {node} seeds {values}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v83r79/waiting-cache/{node}",
                "resource_family": "CFCMT-v83-waiting-cache-v1",
                "vram": 0,
                "ram_mb": 196608,
                "cpu": min(workers + 8, 192),
                "priority": "high",
                "require_node": str(node),
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-task-root", type=Path, required=True)
    parser.add_argument("--local-task-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    specs = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        protocol_path=args.protocol,
        conversion_root=args.conversion_root,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
        local_task_root=args.local_task_root,
    )
    scheduler = _load_scheduler(args.scheduler)
    command = (
        f"test ! -e {shlex.quote(str(args.remote_cache_root))} && "
        f"test ! -e {shlex.quote(str(args.remote_task_root))}"
    )
    code, _, stderr = scheduler.run_on(
        "node001", command, timeout=120, check=False
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite remote cache matrix: {stderr}")
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_task_root": str(args.remote_task_root),
        "local_task_root": str(args.local_task_root.resolve()),
        "task_count": len(specs),
        "specs": specs,
    }
    try:
        results = _run_direct_specs(
            specs, scheduler_path=args.scheduler, timeout=int(args.timeout_sec)
        )
        payload.update(
            {
                "submitted": True,
                "execution_mode": "six_node_direct_parallel",
                "direct_results": results,
            }
        )
    except Exception as error:
        payload["execution_error"] = repr(error)
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": True,
                "task_count": len(specs),
                "total_workers": sum(int(spec["cpu"]) - 8 for spec in specs),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
