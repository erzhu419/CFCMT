#!/usr/bin/env python3
"""Create V117 freeze evidence beside remote model artifacts and sync JSON only."""

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
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v117_pure_waiting_heldout_city_confirmation.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--selector-result-local", type=Path, required=True)
    parser.add_argument("--selector-result-remote", type=Path, required=True)
    parser.add_argument("--deployment-fit-result-local", type=Path, required=True)
    parser.add_argument("--deployment-fit-result-remote", type=Path, required=True)
    parser.add_argument("--remote-freeze", type=Path, required=True)
    parser.add_argument("--local-freeze", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    args = parser.parse_args(argv)
    if args.local_freeze.exists():
        raise FileExistsError("refusing to overwrite local V117 freeze")
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    selector_sha = _sha256(args.selector_result_local)
    fit_sha = _sha256(args.deployment_fit_result_local)
    scheduler = _load_scheduler(args.scheduler)
    check = shlex.join(
        [
            "test",
            "!",
            "-e",
            str(args.remote_freeze),
            "-a",
            "-f",
            str(args.selector_result_remote),
            "-a",
            "-f",
            str(args.deployment_fit_result_remote),
        ]
    )
    code, _, stderr = scheduler.run_on("node001", check, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"V117 freeze preconditions failed: {stderr}")
    code, stdout, stderr = scheduler.run_on(
        "node001",
        shlex.join(
            [
                "sha256sum",
                str(args.selector_result_remote),
                str(args.deployment_fit_result_remote),
            ]
        ),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(stdout).splitlines() if line.split()
    ]
    if int(code) != 0 or remote_hashes != [selector_sha, fit_sha]:
        raise RuntimeError(
            "V117 remote parent identities differ from local evidence: "
            f"{remote_hashes} != {[selector_sha, fit_sha]}; {stderr}"
        )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze",
            "--protocol",
            str(snapshot_root / PROTOCOL_RELATIVE),
            "--selector-result",
            str(args.selector_result_remote),
            "--selector-result-sha256",
            selector_sha,
            "--deployment-fit-result",
            str(args.deployment_fit_result_remote),
            "--deployment-fit-result-sha256",
            fit_sha,
            "--out",
            str(args.remote_freeze),
        ]
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=600, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"V117 remote freeze failed: {stderr or stdout}")
    args.local_freeze.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:"
            f"{args.remote_freeze}",
            str(args.local_freeze),
        ],
        check=True,
    )
    freeze = _read_json(args.local_freeze)
    if (
        freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
    ):
        raise RuntimeError("synced V117 freeze is invalid")
    print(
        json.dumps(
            {
                "active_arms": freeze["active_arms"],
                "matrix_size": freeze["matrix_size"],
                "local_freeze": str(args.local_freeze),
                "remote_freeze": str(args.remote_freeze),
                "sha256": _sha256(args.local_freeze),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
