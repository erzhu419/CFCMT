#!/usr/bin/env python3
"""Submit Boston full microscopic admission after the v9 trigger passes."""

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
from scripts.cluster.launch_eth_boston_full_admission_v5 import (
    CPU_CORES,
    RAM_MB,
    build_spec as _build_v5_spec,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v7 import (
    ALLOWED_NODES,
    FORMER_FAILURE_TIMES_SEC,
    TRIGGER_BEGIN_TIME_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v118r-eth-boston-full-admission-launch-v7"
SIGNATURE = "CFCMT/v118r/eth-boston-full-admission-v7-ramp-merge-yield"


def validate_trigger_result(
    result: Mapping[str, Any],
    *,
    expected_package_manifest_sha256: str,
) -> None:
    setting = dict(result.get("setting", {}))
    observed = dict(result.get("observed", {}))
    checks = dict(result.get("checks", {}))
    if (
        result.get("mode") != "operational_preflight"
        or result.get("operational_preflight_passed") is not True
        or result.get("scientific_admission_passed") is not False
        or result.get("error") is not None
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise ValueError("Boston v9 trigger-window admission did not pass")
    if setting.get("package_manifest_sha256") != expected_package_manifest_sha256:
        raise ValueError("Boston v9 trigger-window package identity changed")
    if (
        setting.get("city_code") != "BOS"
        or setting.get("backend") != "libsumo"
        or setting.get("expected_sumo_version") != "1.22.0"
        or float(setting.get("begin_time_sec", -1.0)) != TRIGGER_BEGIN_TIME_SEC
        or int(setting.get("operational_horizon_sec", -1)) != TRIGGER_HORIZON_SEC
        or float(setting.get("stop_time_sec", -1.0)) != TRIGGER_STOP_TIME_SEC
        or float(observed.get("final_time_sec", -1.0)) != TRIGGER_STOP_TIME_SEC
    ):
        raise ValueError("Boston v9 trigger-window execution contract changed")
    if any(
        float(observed["final_time_sec"]) <= failure_time
        for failure_time in FORMER_FAILURE_TIMES_SEC
    ):
        raise ValueError("Boston v9 trigger did not cross all prior failures")


def build_spec(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_package_manifest_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    spec = _build_v5_spec(
        snapshot_root=snapshot_root,
        package_root=package_root,
        expected_package_manifest_sha256=expected_package_manifest_sha256,
        remote_output_root=remote_output_root,
        local_output_root=local_output_root,
    )
    spec.update(
        {
            "description": "CFCMT V118r ETH Boston v9 complete-demand libsumo full admission",
            "signature": SIGNATURE,
            "ram_resource_family": "CFCMT-v118-eth-boston-full-admission-v9-complete",
            "allowed_nodes": list(ALLOWED_NODES),
        }
    )
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest-local", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--trigger-result-local", type=Path, required=True)
    parser.add_argument("--trigger-result-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite Boston v9 full launch artifact")
    if _sha256(args.package_manifest_local) != args.expected_package_manifest_sha256:
        raise ValueError("local Boston v9 package manifest identity changed")
    trigger = _read_json(args.trigger_result_local)
    validate_trigger_result(
        trigger,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
    )
    trigger_sha256 = _sha256(args.trigger_result_local)
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
            shlex.join(["test", "-f", str(args.trigger_result_remote)]),
            shlex.join(["test", "-f", str(package_manifest_remote)]),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote Boston v9 full preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(args.trigger_result_remote), str(package_manifest_remote)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in str(hash_stdout).splitlines() if line.split()]
    expected_hashes = [trigger_sha256, args.expected_package_manifest_sha256]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote Boston v9 full identities changed: "
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
            "CFCMT-v118r-eth-boston-full-admission-v7",
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
        "trigger_result_sha256": trigger_sha256,
        "package_manifest_sha256": args.expected_package_manifest_sha256,
        "remote_output_root": str(args.remote_output_root),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected Boston v9 full admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
