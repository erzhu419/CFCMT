#!/usr/bin/env python3
"""Submit complete-demand route admission for the Boston v10 package."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from scripts.cluster.launch_eth_boston_route_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    RAM_MB,
)
from scripts.cluster.launch_eth_boston_route_admission_v7 import (
    build_spec as _build_v7_spec,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v118r-eth-boston-route-admission-launch-v8"
SIGNATURE = "CFCMT/v118r/eth-boston-route-admission-v8-permissive-merge-removal"


def build_spec(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_package_manifest_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    spec = _build_v7_spec(
        snapshot_root=snapshot_root,
        package_root=package_root,
        expected_package_manifest_sha256=expected_package_manifest_sha256,
        remote_output_root=remote_output_root,
        local_output_root=local_output_root,
    )
    spec.update(
        {
            "description": "CFCMT V118r ETH Boston v10 complete-demand route admission",
            "signature": SIGNATURE,
            "ram_resource_family": "CFCMT-v118-eth-boston-route-admission-v10",
        }
    )
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest-local", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite Boston v10 route admission")
    if _sha256(args.package_manifest_local) != args.expected_package_manifest_sha256:
        raise ValueError("local Boston v10 package manifest identity changed")
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
            shlex.join(["test", "-f", str(package_manifest_remote)]),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote Boston v10 route preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(package_manifest_remote)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != [args.expected_package_manifest_sha256]:
        raise RuntimeError(
            "remote Boston v10 package identity changed: "
            f"{remote_hashes}; {hash_stderr}"
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
            "CFCMT-v118r-eth-boston-route-admission-v8",
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
        raise RuntimeError("scheduler rejected Boston v10 route admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
