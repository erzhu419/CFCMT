#!/usr/bin/env python3
"""Launch the bounded Boston v18 collision diagnostic on libsumo."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    RAM_MB,
    validate_route_result,
)
from scripts.cluster.launch_eth_boston_v13_admission import (
    _common_payload,
    _manifest,
    _submit,
    _verify_remote_inputs,
)
from scripts.cluster.launch_eth_boston_v15_admission import TRIGGER_HORIZON_SEC
from scripts.cluster.launch_eth_boston_v18_admission import validate_trigger_package
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v150-eth-boston-v18-collision-diagnostic-launch-v1"
SIGNATURE = "CFCMT/v150/eth-boston-v18-collision-diagnostic-v1"
TRACE_START_TIME_SEC = 17_240.0
EXPECTED_COLLISION_TIME_SEC = 17_265.0
EXPECTED_COLLISION_LANE = "591691873#0_1"
EXPECTED_COLLIDER_ID = "271533"
EXPECTED_VICTIM_ID = "289458"
TRACE_LANES = (EXPECTED_COLLISION_LANE, "9114638_0")
TRACE_VEHICLES = (EXPECTED_COLLIDER_ID, EXPECTED_VICTIM_ID)
MAXIMUM_TRACE_SAMPLES = 40


def build_spec(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_package_manifest_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    arguments = [
        "env",
        f"CFCMT_SOURCE_ROOT={snapshot_root}",
        "PYTHONUNBUFFERED=1",
        str(wrapper),
        "-m",
        "cf_h2o.eval.traffic_signal_eth_city_collision_diagnostic",
        "--package-root",
        str(package_root),
        "--expected-package-manifest-sha256",
        str(expected_package_manifest_sha256),
        "--expected-city-code",
        "BOS",
        "--expected-sumo-version",
        "1.22.0",
        "--operational-horizon-sec",
        str(TRIGGER_HORIZON_SEC),
        "--expected-collision-time-sec",
        str(EXPECTED_COLLISION_TIME_SEC),
        "--expected-lane-id",
        EXPECTED_COLLISION_LANE,
        "--expected-collider-id",
        EXPECTED_COLLIDER_ID,
        "--expected-victim-id",
        EXPECTED_VICTIM_ID,
        "--trace-start-time-sec",
        str(TRACE_START_TIME_SEC),
    ]
    for lane_id in TRACE_LANES:
        arguments.extend(("--trace-lane-id", lane_id))
    for vehicle_id in TRACE_VEHICLES:
        arguments.extend(("--trace-vehicle-id", vehicle_id))
    arguments.extend(
        (
            "--diagnostic-maximum-samples",
            str(MAXIMUM_TRACE_SAMPLES),
            "--out",
            str(remote_output_root / "result.json"),
        )
    )
    command = shlex.join(arguments) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT Boston v18 read-only collision diagnostic",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_output_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v118-eth-boston-trigger-admission",
        "ram_resource_family": "CFCMT-v150-eth-boston-collision-diagnostic-v18",
        "vram": 0,
        "ram_mb": RAM_MB,
        "cpu": CPU_CORES,
        "priority": "high",
        "allowed_nodes": list(ALLOWED_NODES),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
        "local_result_dir": str(local_output_root.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest-local", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--route-result-local", type=Path, required=True)
    parser.add_argument("--route-result-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite Boston v18 diagnostic evidence")
    snapshot = _read_json(args.stage_manifest)
    snapshot_root = Path(snapshot["snapshot_root"])
    package = _manifest(
        args.package_manifest_local,
        args.expected_package_manifest_sha256,
    )
    validate_trigger_package(package)
    route = _read_json(args.route_result_local)
    validate_route_result(
        route,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
    )
    scheduler = _load_scheduler(args.scheduler)
    _verify_remote_inputs(
        scheduler=scheduler,
        paths_and_hashes=(
            (
                args.package_root / "package_manifest.json",
                args.expected_package_manifest_sha256,
            ),
            (args.route_result_remote, _sha256(args.route_result_local)),
        ),
        absent_path=args.remote_output_root,
    )
    spec = build_spec(
        snapshot_root=snapshot_root,
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
    )
    completed = _submit(
        scheduler_path=args.scheduler,
        spec=spec,
        intent_label="CFCMT-v150-eth-boston-v18-collision-diagnostic",
    )
    payload = _common_payload(
        stage="collision_diagnostic",
        snapshot=snapshot,
        completed=completed,
        spec=spec,
    )
    payload.update(
        {
            "protocol": LAUNCH_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_manifest_sha256": args.expected_package_manifest_sha256,
            "trace_start_time_sec": TRACE_START_TIME_SEC,
            "expected_collision_time_sec": EXPECTED_COLLISION_TIME_SEC,
            "remote_output_root": str(args.remote_output_root),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected Boston v18 collision diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
