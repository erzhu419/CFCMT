#!/usr/bin/env python3
"""Submit the label-free V150B right-of-way signature extraction."""

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

from scripts.cluster.launch_tsc_external_network_admission_v6 import DEFAULT_SCHEDULER


LAUNCH_PROTOCOL = "tsc-v150b-right-of-way-signature-launch-v3"
SIGNATURE = "CFCMT/v150b/right-of-way-signature-v3-admitted-scenarios"
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)


def build_spec(
    *,
    snapshot_root: Path,
    source_cache_root: Path,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_right_of_way_signature",
            "--source-manifest",
            str(snapshot_root / SOURCE_MANIFEST_RELATIVE),
            "--source-cache-root",
            str(source_cache_root),
            "--out",
            str(remote_output_root / "signature.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V150B label-free right-of-way signature",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_output_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v150b-source-identifiability",
        "ram_resource_family": "CFCMT-v150b-source-identifiability",
        "vram": 0,
        "ram_mb": 16_384,
        "cpu": 2,
        "priority": "high",
        "allowed_nodes": ["node001", "node002", "node003", "node005", "node006"],
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
        "local_result_dir": str(local_output_root.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150B signature launch")
    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    snapshot_root = Path(stage["snapshot_root"])
    spec = build_spec(
        snapshot_root=snapshot_root,
        source_cache_root=args.source_cache_root,
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
            "CFCMT-v150b-right-of-way-signature-v3",
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
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_cache_root": str(args.source_cache_root),
        "remote_output_root": str(args.remote_output_root),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150B signature extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
