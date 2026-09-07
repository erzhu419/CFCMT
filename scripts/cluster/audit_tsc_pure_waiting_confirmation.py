#!/usr/bin/env python3
"""Run the V117 matrix audit remotely and sync only its JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_audit import (  # noqa: E402
    RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--freeze-local", type=Path, required=True)
    parser.add_argument("--freeze-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--remote-audit", type=Path, required=True)
    parser.add_argument("--local-audit", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    args = parser.parse_args(argv)
    if args.local_audit.exists():
        raise FileExistsError("refusing to overwrite local V117 audit")
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    freeze_sha = _sha256(args.freeze_local)
    scheduler = _load_scheduler(args.scheduler)
    code, stdout, stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(args.freeze_remote)]),
        timeout=120,
        check=False,
    )
    remote_sha = str(stdout).strip().split()[0] if str(stdout).strip() else ""
    if int(code) != 0 or remote_sha != freeze_sha:
        raise RuntimeError(
            f"V117 audit freeze differs: {remote_sha} != {freeze_sha}; {stderr}"
        )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_pure_waiting_confirmation_audit",
            "--freeze",
            str(args.freeze_remote),
            "--freeze-sha256",
            freeze_sha,
            "--results-root",
            str(args.remote_results_root),
            "--out",
            str(args.remote_audit),
        ]
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=1200, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"V117 remote audit failed: {stderr or stdout}")
    args.local_audit.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:"
            f"{args.remote_audit}",
            str(args.local_audit),
        ],
        check=True,
    )
    audit = _read_json(args.local_audit)
    if audit.get("protocol") != RESULT_PROTOCOL or audit.get("status") != "PASS":
        raise RuntimeError("synced V117 confirmation audit failed")
    print(
        json.dumps(
            {
                "decision": audit["decision"],
                "confirmation_gate_passed": audit["confirmation_gate_passed"],
                "local_audit": str(args.local_audit),
                "sha256": _sha256(args.local_audit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
