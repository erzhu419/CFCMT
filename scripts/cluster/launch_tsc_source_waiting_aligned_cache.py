#!/usr/bin/env python3
"""Submit the full 18-scenario waiting-aligned source cache matrix."""

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


LAUNCH_PROTOCOL = "tsc-v114-source-pure-waiting-cache-launch-v1"
PROTOCOL_NAME = "tsc-v114-source-pure-waiting-cache-v1"
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    protocol: dict[str, Any],
    remote_cache_root: Path,
    remote_task_root: Path,
) -> list[dict[str, Any]]:
    collection = dict(protocol["collection"])
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / MANIFEST_RELATIVE
    specs = []
    for node, job in protocol["node_jobs"].items():
        scenarios = [str(value) for value in job["scenarios"]]
        seeds = [int(value) for value in job["seeds"]]
        result_root = remote_task_root / str(node)
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_cache",
                "--manifest",
                str(manifest),
                "--scenarios",
                *(str(value) for value in scenarios),
                "--seeds",
                *(str(value) for value in seeds),
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
                "--allow-zero-retained-scenarios",
                *(
                    str(value)
                    for value in collection.get(
                        "allowed_zero_retained_scenarios", ()
                    )
                    if str(value) in scenarios
                ),
                "--expected-sumo-version",
                str(collection["sumo_version"]),
                "--out",
                str(result_root / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    "CFCMT V114 pure-waiting source cache "
                    f"{node}: {','.join(scenarios)} seeds={','.join(map(str, seeds))}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v114/source-pure-waiting-cache/{node}",
                "resource_family": "CFCMT-v114-source-pure-waiting-cache-v1",
                "vram": 0,
                "ram_mb": 98304,
                "cpu": int(collection["workers_per_node"]),
                "priority": "high",
                "require_node": str(node),
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
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
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-task-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite source cache launch artifact")
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError("source waiting-cache protocol changed")
    jobs = protocol["node_jobs"]
    collection = protocol["collection"]
    all_seeds = {int(value) for value in collection["seeds"]}
    scenario_seeds = [
        (str(scenario), int(seed))
        for job in jobs.values()
        for scenario in job["scenarios"]
        for seed in job["seeds"]
    ]
    scenarios = {scenario for scenario, _ in scenario_seeds}
    if (
        tuple(jobs) != tuple(f"node00{index}" for index in range(1, 7))
        or len(scenarios) != int(protocol["expected_scenario_count"])
        or len(scenario_seeds) != int(protocol["expected_scenario_seed_count"])
        or len(scenario_seeds) != len(set(scenario_seeds))
        or any(
            {seed for name, seed in scenario_seeds if name == scenario}
            != all_seeds
            for scenario in scenarios
        )
        or int(protocol["expected_cache_file_count"])
        != len(scenario_seeds)
        * int(protocol["collection"]["collection_shards_per_seed"])
        or int(protocol["collection"]["workers_per_node"]) > 20
        or collection.get("counterfactual_cost_contract")
        != counterfactual_cost_contract_v5("halted_queue")
    ):
        raise ValueError("source waiting-cache assignment contract changed")
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        protocol=protocol,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
    )
    scheduler = _load_scheduler(args.scheduler)
    check_command = shlex.join(
        [
            "test",
            "!",
            "-e",
            str(args.remote_cache_root),
            "-a",
            "!",
            "-e",
            str(args.remote_task_root),
        ]
    )
    code, _, stderr = scheduler.run_on(
        "node001", check_command, timeout=120, check=False
    )
    if int(code) != 0:
        raise FileExistsError(
            f"refusing to overwrite remote source cache matrix: {stderr}"
        )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v114-source-pure-waiting-cache",
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
        "source_manifest": str((PROJECT_ROOT / MANIFEST_RELATIVE).resolve()),
        "source_manifest_sha256": _sha256(PROJECT_ROOT / MANIFEST_RELATIVE),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_task_root": str(args.remote_task_root),
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
        raise RuntimeError("source waiting-cache scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
