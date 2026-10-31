#!/usr/bin/env python3
"""Replay Boston v9 trigger admission with enriched collision evidence."""

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
from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    validate_route_result,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v7 import (
    build_spec as _build_v7_spec,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v118r-eth-boston-collision-evidence-replay-launch-v8"
SIGNATURE = "CFCMT/v118r/eth-boston-v9-collision-evidence-replay-v8"


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
            "description": "CFCMT Boston v9 collision-evidence trigger replay",
            "signature": SIGNATURE,
            "resource_family": "CFCMT-v118-boston-collision-evidence",
            "ram_resource_family": "CFCMT-v118-boston-collision-evidence-v8",
        }
    )
    return spec


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
        raise FileExistsError("refusing to overwrite Boston collision replay")
    if _sha256(args.package_manifest_local) != args.expected_package_manifest_sha256:
        raise ValueError("local Boston v9 package manifest identity changed")
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
        raise RuntimeError(f"remote Boston collision replay preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(args.route_result_remote), str(package_manifest_remote)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    expected_hashes = [route_sha256, args.expected_package_manifest_sha256]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote Boston collision replay evidence changed: "
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
            "CFCMT-v118r-boston-collision-evidence-replay-v8",
        ],
        input=json.dumps([spec]),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "diagnostic_replay_not_admission",
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
        raise RuntimeError("scheduler rejected Boston collision evidence replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
