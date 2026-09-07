"""Audit a cluster node before launching formal traffic-signal experiments."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _scenario_input_fingerprint
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _start_sumo
from cf_h2o.sumo_runtime import load_libsumo, runtime_metadata, source_tree_sha256, sumo_state_directory
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


def _memory_summary() -> dict[str, int]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, raw = line.partition(":")
        if name not in {"MemTotal", "MemAvailable"}:
            continue
        fields = raw.strip().split()
        if fields:
            values[f"{name.lower()}_kb"] = int(fields[0])
    return values


def _run_sumo_smoke(sumocfg: Path) -> dict[str, Any]:
    sumo_api = load_libsumo()
    with sumo_state_directory(prefix="cfcmt-preflight-") as scratch:
        try:
            _start_sumo(
                sumo_api,
                sumocfg,
                seed=20260803,
                tripinfo_output=scratch / "tripinfo.xml",
            )
            initial_time = float(sumo_api.simulation.getTime())
            initial_expected = int(sumo_api.simulation.getMinExpectedNumber())
            for _ in range(3):
                sumo_api.simulationStep()
            return {
                "scenario_sumocfg": str(sumocfg),
                "initial_time": initial_time,
                "final_time": float(sumo_api.simulation.getTime()),
                "initial_min_expected": initial_expected,
                "final_min_expected": int(sumo_api.simulation.getMinExpectedNumber()),
                "traffic_light_count": len(sumo_api.trafficlight.getIDList()),
                "passed": True,
            }
        finally:
            try:
                sumo_api.close()
            except Exception:
                pass


def run_cluster_preflight(
    *,
    manifest_path: Path,
    expected_source_sha256: str,
    expected_sumo_version: str,
    smoke_scenario: str,
    snapshot_manifest: Path | None = None,
    expected_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    actual_source_sha256 = source_tree_sha256()
    if actual_source_sha256 != str(expected_source_sha256):
        raise RuntimeError(
            "cluster source fingerprint mismatch: "
            f"expected {expected_source_sha256}, found {actual_source_sha256}"
        )
    runtime = runtime_metadata()
    if runtime["libsumo_version"] != str(expected_sumo_version):
        raise RuntimeError(
            "cluster SUMO version mismatch: "
            f"expected {expected_sumo_version}, found {runtime['libsumo_version']}"
        )
    hash_seed = runtime.get("determinism_environment", {}).get("PYTHONHASHSEED")
    if hash_seed != "0":
        raise RuntimeError(
            "cluster Python hash seed mismatch: "
            f"expected 0, found {hash_seed!r}"
        )
    snapshot = None
    if snapshot_manifest is not None:
        snapshot = json.loads(Path(snapshot_manifest).read_text(encoding="utf-8"))
        if expected_snapshot_sha256 and snapshot.get("snapshot_sha256") != str(
            expected_snapshot_sha256
        ):
            raise RuntimeError("cluster snapshot manifest fingerprint mismatch")

    manifest = load_traffic_signal_manifest(manifest_path)
    if smoke_scenario not in manifest.sumocfgs:
        raise ValueError(f"smoke scenario is absent from manifest: {smoke_scenario}")
    fingerprints = {
        name: _scenario_input_fingerprint(path)
        for name, path in sorted(manifest.sumocfgs.items())
    }
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    return {
        "experiment": "traffic_signal_cluster_preflight",
        "hostname": socket.gethostname(),
        "runtime": runtime,
        "resources": {
            "logical_cpu_count": int(os.cpu_count() or 0),
            "affinity_cpu_count": len(affinity),
            "affinity_first_cpu": affinity[0] if affinity else None,
            "affinity_last_cpu": affinity[-1] if affinity else None,
            **_memory_summary(),
        },
        "snapshot": snapshot,
        "manifest": manifest.to_dict(),
        "scenario_input_fingerprints": fingerprints,
        "sumo_smoke": _run_sumo_smoke(manifest.sumocfgs[smoke_scenario]),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-sumo-version", required=True)
    parser.add_argument("--smoke-scenario", default="grid4x4")
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--expected-snapshot-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_cluster_preflight(
        manifest_path=args.manifest,
        expected_source_sha256=args.expected_source_sha256,
        expected_sumo_version=args.expected_sumo_version,
        smoke_scenario=args.smoke_scenario,
        snapshot_manifest=args.snapshot_manifest,
        expected_snapshot_sha256=args.expected_snapshot_sha256,
    )
    atomic_write_json(args.out, result)
    print(
        f"preflight passed on {result['hostname']}: "
        f"{result['resources']['affinity_cpu_count']} CPUs, "
        f"SUMO {result['runtime']['libsumo_version']}"
    )


if __name__ == "__main__":
    main()
