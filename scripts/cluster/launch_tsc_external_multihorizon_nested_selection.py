#!/usr/bin/env python3
"""Run the frozen v78 nested selection on six CPU nodes durably."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_nested_selection import (  # noqa: E402
    NODES,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _direct_remote_command,
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "v83r79-multihorizon-nested-six-node-durable-launch-v1"
SCREEN_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v77_external_v9_"
    "multihorizon_latent_screen.json"
)
CACHE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v76_external_v9_"
    "multihorizon_waiting_cache.json"
)
PARTITION_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v65_external_v9_"
    "post_validation_redevelopment_partition.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
NESTED_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v78_external_v9_"
    "multihorizon_nested_selection.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    protocol_path: Path,
    screen_result: Path,
    screen_oof: Path,
    cache_audit: Path,
    cache_root: Path,
    conversion_root: Path,
    remote_shard_root: Path,
    local_shard_root: Path,
) -> list[dict[str, Any]]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("multihorizon nested protocol changed")
    nested = dict(protocol["nested_selection"])
    outputs = dict(protocol["output_roots"])
    if (
        str(remote_shard_root) != str(outputs["remote_shard_root"])
        or Path(local_shard_root).resolve()
        != Path(outputs["local_shard_root"]).resolve()
    ):
        raise ValueError("multihorizon nested output roots changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for node in NODES:
        seeds = [int(value) for value in nested["node_assignments"][node]]
        remote_result = remote_shard_root / node
        local_result = local_shard_root / node
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_multihorizon_nested_selection",
                "--nested-protocol",
                str(snapshot_root / NESTED_PROTOCOL_RELATIVE),
                "--screen-protocol",
                str(snapshot_root / SCREEN_PROTOCOL_RELATIVE),
                "--screen-result",
                str(screen_result),
                "--screen-oof",
                str(screen_oof),
                "--cache-protocol",
                str(snapshot_root / CACHE_PROTOCOL_RELATIVE),
                "--cache-audit",
                str(cache_audit),
                "--cache-root",
                str(cache_root),
                "--partition",
                str(snapshot_root / PARTITION_RELATIVE),
                "--manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root",
                str(conversion_root),
                "--node",
                node,
                "--outer-seeds",
                *[str(seed) for seed in seeds],
                "--cache-workers",
                str(nested["cache_workers"]),
                "--fold-workers",
                str(nested["inner_fold_workers"]),
                "--output-root",
                str(remote_result),
                "--out",
                str(remote_result / "summary_v1.json"),
            ]
        )
        specs.append(
            {
                "description": f"multihorizon nested selection {node} {seeds}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v83r79/multihorizon-nested/{node}",
                "resource_family": "CFCMT-v83-multihorizon-nested-v1",
                "vram": 0,
                "ram_mb": 196608,
                "cpu": 32,
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


def _sync_node(scheduler: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    node = str(spec["require_node"])
    local_result = Path(str(spec["local_result_dir"]))
    if local_result.exists():
        raise FileExistsError(f"refusing to overwrite nested sync: {local_result}")
    local_result.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node(node),
            f"{scheduler._ssh_target_for_node(node)}:{spec['result_dir']}/",
            f"{local_result}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "node": node,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-shard-root", type=Path, required=True)
    parser.add_argument("--local-shard-root", type=Path, required=True)
    parser.add_argument(
        "--scheduleurm-root",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm"),
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_shard_root.exists():
        raise FileExistsError("refusing to overwrite local nested launch outputs")
    stage = _read_json(args.stage_manifest)
    specs = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        protocol_path=args.protocol,
        screen_result=args.screen_result,
        screen_oof=args.screen_oof,
        cache_audit=args.cache_audit,
        cache_root=args.cache_root,
        conversion_root=args.conversion_root,
        remote_shard_root=args.remote_shard_root,
        local_shard_root=args.local_shard_root,
    )
    scheduler = _load_scheduler(args.scheduler)
    code, _, stderr = scheduler.run_on(
        "node001",
        shlex.join(["test", "!", "-e", str(args.remote_shard_root)]),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(
            f"refusing to overwrite remote nested shard root: {stderr}"
        )
    sys.path.insert(0, str(args.scheduleurm_root.resolve()))
    from algorithm.experiments.durable_remote_command import (
        run_durable_remote_capture,
    )

    args.transport_root.mkdir(parents=True, exist_ok=False)

    def run(spec: Mapping[str, Any]) -> dict[str, Any]:
        node = str(spec["require_node"])
        returncode, stdout, run_stderr, durable = run_durable_remote_capture(
            node,
            _direct_remote_command(spec),
            args.transport_root / node,
            run_id=f"cfcmt-v78-multihorizon-nested-{node}-v1",
            timeout_s=int(args.timeout_sec),
            poll_interval_s=15.0,
        )
        return {
            "node": node,
            "returncode": int(returncode),
            "stdout_tail": str(stdout)[-8000:],
            "stderr_tail": str(run_stderr)[-8000:],
            "durable_transport": durable,
        }

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        direct_results = list(pool.map(run, specs))
    returncodes_ok = all(item["returncode"] == 0 for item in direct_results)
    sync_results = []
    if returncodes_ok:
        args.local_shard_root.mkdir(parents=True, exist_ok=False)
        for spec in specs:
            sync_results.append(_sync_node(scheduler, spec))
    sync_ok = bool(
        returncodes_ok
        and len(sync_results) == len(specs)
        and all(item["returncode"] == 0 for item in sync_results)
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "remote_shard_root": str(args.remote_shard_root),
        "local_shard_root": str(args.local_shard_root.resolve()),
        "task_count": len(specs),
        "specs": specs,
        "direct_results": direct_results,
        "sync_results": sync_results,
        "passed": sync_ok,
    }
    _atomic_json(args.out, payload)
    if not sync_ok:
        raise RuntimeError(
            "multihorizon nested launch failed: "
            f"direct={[(x['node'], x['returncode']) for x in direct_results]}, "
            f"sync={[(x['node'], x['returncode']) for x in sync_results]}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "nodes": len(specs),
                "outer_folds": sum(
                    len(
                        _read_json(args.protocol)["nested_selection"][
                            "node_assignments"
                        ][node]
                    )
                    for node in NODES
                ),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
