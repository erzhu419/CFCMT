#!/usr/bin/env python3
"""Submit the frozen v8 LA/Jinan adaptation counterfactual cache matrix."""

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

from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    ASSIGNMENTS,
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v26_external_la_jinan_confirmation.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v28_external_la_jinan_v8_manifest.json"
)
V9_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
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
) -> list[dict[str, Any]]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported external cache generation: {generation}")
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    if protocol.get("protocol") != (
        "tsc-v42r38-external-la-jinan-tls-safe-confirmation-v3"
    ):
        raise ValueError("external v8 confirmation protocol changed")
    target = dict(protocol["target_protocol"])
    if int(target["counterfactual_collection_shards_per_scenario_seed"]) != int(
        collection_shards
    ):
        raise ValueError("external adaptation shard count changed")
    expected_pairs = {
        (str(scenario), int(seed))
        for scenarios in target["external_city_scenarios"].values()
        for scenario in scenarios
        for seed in target["adaptation_seeds"]
    }
    assigned_pairs = {(scenario, int(seed)) for _, scenario, seed in ASSIGNMENTS}
    if assigned_pairs != expected_pairs or len(assigned_pairs) != len(ASSIGNMENTS):
        raise ValueError("external adaptation assignments do not cover the matrix")
    if int(workers) != int(collection_shards):
        raise ValueError("one adaptation worker per frozen shard is required")

    manifest_relative = (
        MANIFEST_RELATIVE if generation == "v8" else V9_MANIFEST_RELATIVE
    )
    manifest_payload = _read_json(PROJECT_ROOT / manifest_relative)
    expected_manifest_protocol = {
        "v8": "external-la-jinan-v8-tls-safe-full-network-v1",
        "v9": "external-la-jinan-v9-monotone-virtual-lane-safe-full-network-v1",
    }[generation]
    if manifest_payload.get("protocol") != expected_manifest_protocol:
        raise ValueError(f"external {generation} benchmark manifest changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / manifest_relative
    specs = []
    for node, scenario, seed in ASSIGNMENTS:
        name = f"{scenario}_seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite adaptation cache result: {local_result}"
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
                "--out",
                str(remote_result / "cache_summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT v42r38 external adaptation cache {scenario} "
                    f"seed {seed}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/v42r38/adaptation-cache-{generation}/{scenario}/seed-{seed}"
                ),
                "resource_family": (
                    f"CFCMT-external-counterfactual-cache-{generation}"
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
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
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
    specs = build_specs(
        snapshot_root=snapshot_root,
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
    payload: dict[str, Any] = {
        "protocol": (
            f"v42r38-external-adaptation-cache-{args.generation}-submission-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
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
                f"CFCMT-v42r38-external-adaptation-cache-{args.generation}",
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
            raise RuntimeError("external adaptation cache submission failed")
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
