#!/usr/bin/env python3
"""Submit the Boston v7 trigger-window microscopic admission."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v118r-eth-boston-trigger-admission-launch-v5"
SIGNATURE = "CFCMT/v118r/eth-boston-trigger-admission-v5-merge-tls-yield"
ALLOWED_NODES = ("node001", "node002", "node003", "node005", "node006")
CPU_CORES = 16
RAM_MB = 24_576
TRIGGER_BEGIN_TIME_SEC = 7_201.0
TRIGGER_HORIZON_SEC = 5_400
TRIGGER_STOP_TIME_SEC = TRIGGER_BEGIN_TIME_SEC + TRIGGER_HORIZON_SEC
FORMER_FAILURE_TIMES_SEC = (9_518.0, 12_279.0)
EXPECTED_TRIP_COUNT = 3_806_510


def validate_route_result(
    result: Mapping[str, Any],
    *,
    expected_package_manifest_sha256: str,
) -> None:
    duarouter = dict(result.get("duarouter", {}))
    if (
        result.get("passed") is not True
        or result.get("decision") != "authorize_microscopic_operational_preflight"
        or result.get("city_code") != "BOS"
        or result.get("package_manifest_sha256")
        != expected_package_manifest_sha256
        or int(result.get("trip_count", -1)) != EXPECTED_TRIP_COUNT
        or int(duarouter.get("returncode", -1)) != 0
        or duarouter.get("ignore_errors") is not False
    ):
        raise ValueError("Boston v7 complete-demand route admission did not pass")


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
        "description": "CFCMT V118r ETH Boston v7 5400s trigger admission",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_output_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v118-eth-boston-trigger-admission",
        "ram_resource_family": "CFCMT-v118-eth-boston-trigger-admission-v7",
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
    if args.out.exists():
        raise FileExistsError("refusing to overwrite Boston trigger launch artifact")
    if _sha256(args.package_manifest_local) != args.expected_package_manifest_sha256:
        raise ValueError("local Boston v7 package manifest identity changed")
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
    code, _, stderr = scheduler.run_on(
        "node001",
        preflight,
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise RuntimeError(f"remote Boston trigger preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(
            [
                "sha256sum",
                str(args.route_result_remote),
                str(package_manifest_remote),
            ]
        ),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0]
        for line in str(hash_stdout).splitlines()
        if line.split()
    ]
    expected_hashes = [route_sha256, args.expected_package_manifest_sha256]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote Boston trigger evidence changed: "
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
            "CFCMT-v118r-eth-boston-trigger-admission-v5",
        ],
        input=json.dumps([spec]),
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected Boston v7 trigger admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
