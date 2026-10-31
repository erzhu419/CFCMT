#!/usr/bin/env python3
"""Submit the Boston v12 extended trigger-window microscopic admission."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    EXPECTED_TRIP_COUNT,
    RAM_MB,
    TRIGGER_BEGIN_TIME_SEC,
    validate_route_result,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.data.repair_eth_boston_joined_tls_uncontrolled_merge import (
    PACKAGE_PROTOCOL,
    validate_joined_tls_uncontrolled_merge_manifest,
)


LAUNCH_PROTOCOL = "tsc-v118r-eth-boston-trigger-admission-launch-v10"
SIGNATURE = (
    "CFCMT/v118r/eth-boston-trigger-admission-v10-"
    "joined-tls-uncontrolled-merge-yield"
)
TRIGGER_HORIZON_SEC = 7_200
TRIGGER_STOP_TIME_SEC = TRIGGER_BEGIN_TIME_SEC + TRIGGER_HORIZON_SEC
FORMER_FAILURE_TIMES_SEC = (
    9_518.0,
    9_735.0,
    11_304.0,
    11_504.0,
    11_871.0,
    12_279.0,
    13_146.0,
)


def validate_trigger_package(manifest: Mapping[str, Any]) -> None:
    validate_joined_tls_uncontrolled_merge_manifest(manifest)
    microscopic = dict(manifest.get("microscopic_instantiation", {}))
    demand = dict(manifest.get("microscopic_demand_inventory", {}))
    gates = dict(manifest.get("gates", {}))
    begin_sec = float(microscopic.get("begin_sec", -1.0))
    end_sec = float(microscopic.get("end_sec", -1.0))
    if (
        manifest.get("protocol") != PACKAGE_PROTOCOL
        or manifest.get("passed") is not True
        or manifest.get("city_code") != "BOS"
        or gates.get("boston_joined_tls_uncontrolled_merge_yield_repaired")
        is not True
        or microscopic.get("backend_required_by_protocol") != "libsumo"
        or begin_sec != TRIGGER_BEGIN_TIME_SEC
        or end_sec < TRIGGER_STOP_TIME_SEC
        or int(demand.get("trip_count", -1)) != EXPECTED_TRIP_COUNT
        or any(
            failure_time < begin_sec or failure_time > TRIGGER_STOP_TIME_SEC
            for failure_time in FORMER_FAILURE_TIMES_SEC
        )
    ):
        raise ValueError("Boston v12 package does not cover the trigger evidence window")


def build_spec(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_package_manifest_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_eth_city_admission",
            "--package-root",
            str(package_root),
            "--expected-package-manifest-sha256",
            expected_package_manifest_sha256,
            "--expected-city-code",
            "BOS",
            "--expected-sumo-version",
            "1.22.0",
            "--operational-horizon-sec",
            str(TRIGGER_HORIZON_SEC),
            "--progress-interval-sec",
            "900",
            "--state-sample-interval-sec",
            "60",
            "--out",
            str(remote_output_root / "result.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V118r ETH Boston v12 7200s trigger admission",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_output_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v118-eth-boston-trigger-admission",
        "ram_resource_family": "CFCMT-v118-eth-boston-trigger-admission-v12",
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
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest-local", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--route-result-local", type=Path, required=True)
    parser.add_argument("--route-result-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite Boston v12 trigger admission")
    if _sha256(args.package_manifest_local) != args.expected_package_manifest_sha256:
        raise ValueError("local Boston v12 package manifest identity changed")
    package_manifest = _read_json(args.package_manifest_local)
    validate_trigger_package(package_manifest)
    route = _read_json(args.route_result_local)
    validate_route_result(
        route,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
    )
    route_sha256 = _sha256(args.route_result_local)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    package_manifest_remote = args.package_root / "package_manifest.json"
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(
                [
                    "test",
                    "-x",
                    str(snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"),
                ]
            ),
            shlex.join(["test", "-f", str(args.route_result_remote)]),
            shlex.join(["test", "-f", str(package_manifest_remote)]),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote Boston v12 trigger preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(
            ["sha256sum", str(args.route_result_remote), str(package_manifest_remote)]
        ),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    expected_hashes = [route_sha256, args.expected_package_manifest_sha256]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote Boston v12 trigger evidence changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v118r-eth-boston-trigger-admission-v10",
        ],
        input=json.dumps([spec]),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "former_failure_times_sec": list(FORMER_FAILURE_TIMES_SEC),
        "trigger_begin_time_sec": TRIGGER_BEGIN_TIME_SEC,
        "trigger_stop_time_sec": TRIGGER_STOP_TIME_SEC,
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "signature": SIGNATURE,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "route_result_sha256": route_sha256,
        "package_manifest_sha256": args.expected_package_manifest_sha256,
        "remote_output_root": str(args.remote_output_root),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected Boston v12 trigger admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
