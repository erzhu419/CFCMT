#!/usr/bin/env python3
"""Submit the 22-seed pure-waiting selector cache on six CPU nodes."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (  # noqa: E402
    counterfactual_cost_contract_v5,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v116-pure-waiting-selector-cache-launch-v1"
PROTOCOL_NAME = "tsc-v116-pure-waiting-selector-development-cache-v1"
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    protocol: dict[str, Any],
    conversion_root: Path,
    remote_cache_root: Path,
    remote_task_root: Path,
    selected_nodes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    collection = dict(protocol["collection"])
    conversion = dict(protocol["conversion_input"])
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    selected = (
        set(protocol["node_seed_assignments"])
        if selected_nodes is None
        else set(selected_nodes)
    )
    for node, seeds in protocol["node_seed_assignments"].items():
        if node not in selected:
            continue
        values = [int(seed) for seed in seeds]
        result_root = remote_task_root / str(node)
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
                *(str(seed) for seed in values),
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
                "--rollout-prefix-horizons-sec",
                *(str(value) for value in collection["rollout_prefix_horizons_sec"]),
                "--collection-shards",
                str(collection["collection_shards_per_seed"]),
                "--workers",
                str(collection["workers_per_node"]),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                str(collection["behavior_policy"]),
                "--counterfactual-cost-mode",
                str(collection["counterfactual_cost_mode"]),
                "--min-tls-coverage",
                str(collection["minimum_tls_coverage"]),
                "--expected-sumo-version",
                str(collection["sumo_version"]),
                "--expected-input-root",
                str(conversion_root),
                "--expected-input-manifest-sha256",
                str(conversion["manifest_sha256"]),
                "--expected-input-tree-sha256",
                str(conversion["tree_sha256"]),
                "--out",
                str(result_root / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    "CFCMT V116 pure-waiting selector cache "
                    f"{node}: seeds={','.join(map(str, values))}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v116/pure-waiting-selector-cache/{node}",
                "resource_family": "CFCMT-v116-pure-waiting-selector-cache-v1",
                "vram": 0,
                "ram_mb": 98304,
                "cpu": int(collection["workers_per_node"]),
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
                    "PYTHONUNBUFFERED": "1",
                },
                "result_dir": str(result_root),
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
    parser.add_argument(
        "--nodes",
        nargs="+",
        help="Submit only these frozen node assignments; defaults to all six.",
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite selector cache launch artifact")
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError("pure-waiting selector cache protocol changed")
    collection = dict(protocol["collection"])
    assignments = protocol["node_seed_assignments"]
    selected_nodes = tuple(args.nodes or assignments)
    if (
        len(selected_nodes) != len(set(selected_nodes))
        or not selected_nodes
        or not set(selected_nodes) <= set(assignments)
    ):
        raise ValueError("selector cache node subset is invalid")
    assigned_seeds = [int(seed) for seeds in assignments.values() for seed in seeds]
    frozen_seeds = [int(seed) for seed in collection["seeds"]]
    if (
        tuple(assignments) != tuple(f"node00{index}" for index in range(1, 7))
        or assigned_seeds != frozen_seeds
        or len(assigned_seeds) != len(set(assigned_seeds))
        or len(assigned_seeds) != int(protocol["expected_seed_count"])
        or int(protocol["expected_cache_file_count"])
        != len(assigned_seeds) * int(collection["collection_shards_per_seed"])
        or int(collection["workers_per_node"]) > 20
        or collection.get("counterfactual_cost_contract")
        != counterfactual_cost_contract_v5("halted_queue")
        or str(args.conversion_root) != str(protocol["conversion_input"]["root"])
    ):
        raise ValueError("pure-waiting selector cache assignment contract changed")
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        protocol=protocol,
        conversion_root=args.conversion_root,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
        selected_nodes=selected_nodes,
    )
    scheduler = _load_scheduler(args.scheduler)
    selected_task_roots = [
        args.remote_task_root / node for node in selected_nodes
    ]
    preconditions = ["test"]
    for path in selected_task_roots:
        preconditions.extend(["!", "-e", str(path), "-a"])
    preconditions.pop()
    code, _, stderr = scheduler.run_on(
        "node001",
        shlex.join(preconditions),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"refusing to overwrite selector cache matrix: {stderr}")
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v116-pure-waiting-selector-cache",
        ],
        input=json.dumps(specs),
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
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "conversion_root": str(args.conversion_root),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_task_root": str(args.remote_task_root),
        "selected_nodes": list(selected_nodes),
        "task_count": len(specs),
        "total_requested_workers": sum(int(spec["cpu"]) for spec in specs),
        "specs": specs,
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "total_requested_workers": payload["total_requested_workers"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if completed.returncode != 0:
        raise RuntimeError("selector cache scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
