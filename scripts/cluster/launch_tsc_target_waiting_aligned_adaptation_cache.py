#!/usr/bin/env python3
"""Submit the four-scenario waiting-aligned target adaptation cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


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


LAUNCH_PROTOCOL = "tsc-v114-target-pure-waiting-adaptation-cache-launch-v1"
PROTOCOL_NAME = "tsc-v114-target-pure-waiting-adaptation-cache-v1"
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
PHYSICAL_NODES = tuple(f"node00{index}" for index in range(1, 7))


def _parse_physical_node_map(
    values: Sequence[str], *, selected_nodes: Sequence[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    selected = {str(value) for value in selected_nodes}
    for value in values:
        logical, separator, physical = str(value).partition("=")
        if (
            not separator
            or logical not in selected
            or physical not in PHYSICAL_NODES
            or logical in result
        ):
            raise ValueError(f"invalid target-cache physical node mapping: {value}")
        result[logical] = physical
    resolved = [result.get(node, node) for node in selected_nodes]
    if len(resolved) != len(set(resolved)):
        raise ValueError("target-cache physical node mapping contains collisions")
    return result


def build_specs(
    *,
    snapshot_root: Path,
    protocol: dict[str, Any],
    conversion_root: Path,
    remote_cache_root: Path,
    remote_task_root: Path,
    selected_nodes: Sequence[str] | None = None,
    physical_node_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    collection = dict(protocol["collection"])
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    selected = (
        set(protocol["node_scenario_assignments"])
        if selected_nodes is None
        else set(selected_nodes)
    )
    for node, scenarios in protocol["node_scenario_assignments"].items():
        if node not in selected:
            continue
        physical_node = str((physical_node_overrides or {}).get(node, node))
        result_root = remote_task_root / str(node)
        command = shlex.join(
            [
                "env",
                f"CFCMT_SOURCE_ROOT={snapshot_root}",
                f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "PYTHONUNBUFFERED=1",
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_cache",
                "--manifest",
                str(manifest),
                "--scenarios",
                *(str(value) for value in scenarios),
                "--seeds",
                *(str(value) for value in collection["seeds"]),
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
                "--out",
                str(result_root / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    "CFCMT V114 pure-waiting target adaptation cache "
                    f"{node} on {physical_node}: "
                    f"{','.join(str(value) for value in scenarios)}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v114/target-pure-waiting-adaptation-cache/{node}",
                "resource_family": "CFCMT-v114-target-pure-waiting-adaptation-cache-v1",
                "vram": 0,
                "ram_mb": 98304,
                "cpu": int(collection["workers_per_node"]),
                "priority": "high",
                "require_node": physical_node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
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
        help="Submit only these frozen node assignments; defaults to all four.",
    )
    parser.add_argument(
        "--physical-node-map",
        nargs="*",
        default=(),
        metavar="LOGICAL=PHYSICAL",
        help="Operationally remap selected logical shards while preserving output paths.",
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite target cache launch artifact")
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError("target waiting-cache protocol changed")
    assignments = protocol["node_scenario_assignments"]
    selected_nodes = tuple(args.nodes or assignments)
    physical_node_overrides = _parse_physical_node_map(
        args.physical_node_map,
        selected_nodes=selected_nodes,
    )
    scenarios = [value for values in assignments.values() for value in values]
    collection = protocol["collection"]
    if (
        len(assignments) != 4
        or len(set(assignments)) != 4
        or len(scenarios) != int(protocol["expected_scenario_count"])
        or len(scenarios) != len(set(scenarios))
        or int(protocol["expected_cache_file_count"])
        != len(scenarios)
        * len(collection["seeds"])
        * int(collection["collection_shards_per_seed"])
        or int(collection["workers_per_node"]) > 20
        or collection.get("counterfactual_cost_contract")
        != counterfactual_cost_contract_v5("halted_queue")
        or not selected_nodes
        or len(selected_nodes) != len(set(selected_nodes))
        or not set(selected_nodes) <= set(assignments)
    ):
        raise ValueError("target waiting-cache assignment contract changed")
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        protocol=protocol,
        conversion_root=args.conversion_root,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
        selected_nodes=selected_nodes,
        physical_node_overrides=physical_node_overrides,
    )
    scheduler = _load_scheduler(args.scheduler)
    checks = ["test"]
    for node in selected_nodes:
        checks.extend(
            ["!", "-e", str(args.remote_task_root / node), "-a"]
        )
    checks.pop()
    check_command = shlex.join(checks)
    code, _, stderr = scheduler.run_on(
        "node003", check_command, timeout=120, check=False
    )
    if int(code) != 0:
        raise FileExistsError(
            f"refusing to overwrite remote target cache matrix: {stderr}"
        )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v114-target-pure-waiting-adaptation-cache",
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
        "target_manifest": str((PROJECT_ROOT / MANIFEST_RELATIVE).resolve()),
        "target_manifest_sha256": _sha256(PROJECT_ROOT / MANIFEST_RELATIVE),
        "conversion_root": str(args.conversion_root),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_task_root": str(args.remote_task_root),
        "selected_nodes": list(selected_nodes),
        "execution_node_map": {
            node: physical_node_overrides.get(node, node)
            for node in selected_nodes
        },
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
        raise RuntimeError("target waiting-cache scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
