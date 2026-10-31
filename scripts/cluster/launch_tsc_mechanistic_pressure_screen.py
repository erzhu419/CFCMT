#!/usr/bin/env python3
"""Submit the frozen V131 mechanistic generalized-pressure screen."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as SELECTOR_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_anchored_source_null_residual_feasibility import (  # noqa: E402
    ALLOWED_NODES,
    RAM_MB,
    TARGET_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import (  # noqa: E402
    SEEDS,
)


LAUNCH_PROTOCOL = "tsc-v131-mechanistic-pressure-screen-launch-v1"
SIGNATURE = "CFCMT/v131/mechanistic-generalized-pressure-screen-v1"


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    selector_cache_root: Path,
    selector_cache_audit_remote: Path,
    selector_cache_audit_sha256: str,
    remote_output_root: Path,
    nodes: Sequence[str],
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_mechanistic_pressure_screen",
            "--selector-cache-root",
            str(selector_cache_root),
            "--selector-manifest",
            str(snapshot_root / TARGET_MANIFEST_RELATIVE),
            "--selector-cache-audit",
            str(selector_cache_audit_remote),
            "--selector-cache-audit-sha256",
            selector_cache_audit_sha256,
            "--selector-scenario",
            "jinan_3x4_real",
            "--selector-seeds",
            *(str(value) for value in SEEDS),
            "--selector-collection-shards",
            "32",
            "--conversion-root",
            str(conversion_root),
            "--cache-workers",
            str(cache_workers),
            "--fold-workers",
            str(fold_workers),
            "--out",
            str(remote_output_root / "result.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V131 mechanistic generalized-pressure screen",
        "project": "CFCMT",
        "cmd": command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v131-mechanistic-pressure-screen",
        "vram": 0,
        "ram_mb": RAM_MB,
        "cpu": 20,
        "priority": "high",
        "allowed_nodes": list(nodes),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-local", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--nodes", default=",".join(ALLOWED_NODES))
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V131 launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V131 cache workers must be in [1, 20]")
    if not 1 <= int(args.fold_workers) <= 20:
        raise ValueError("V131 fold workers must be in [1, 20]")
    nodes = tuple(value.strip() for value in args.nodes.split(",") if value.strip())
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("V131 execution nodes must be nonempty and unique")
    if not set(nodes).issubset(ALLOWED_NODES):
        raise ValueError("V131 execution nodes include a non-admitted CPU node")
    selector_audit = _read_json(args.selector_cache_audit_local)
    if (
        selector_audit.get("protocol") != SELECTOR_AUDIT_PROTOCOL
        or selector_audit.get("status") != "PASS"
        or [int(value) for value in selector_audit.get("seeds", ())]
        != list(SEEDS)
    ):
        raise ValueError("V131 selector cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    selector_sha = _sha256(args.selector_cache_audit_local)
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            shlex.join(["test", "-f", str(args.selector_cache_audit_remote)]),
        )
    )
    for node in nodes:
        code, _, stderr = scheduler.run_on(
            node,
            preflight,
            timeout=120,
            check=False,
        )
        if int(code) != 0:
            raise RuntimeError(f"remote V131 preflight failed on {node}: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        nodes[0],
        shlex.join(["sha256sum", str(args.selector_cache_audit_remote)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0]
        for line in str(hash_stdout).splitlines()
        if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != [selector_sha]:
        raise RuntimeError(
            "remote V131 selector evidence changed: "
            f"{remote_hashes} != {[selector_sha]}; {hash_stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        selector_cache_audit_sha256=selector_sha,
        remote_output_root=args.remote_output_root,
        nodes=nodes,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v131-mechanistic-pressure-screen-v1",
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
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "selector_cache_audit_sha256": selector_sha,
        "remote_output_root": str(args.remote_output_root),
        "execution_nodes": list(nodes),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V131 mechanistic pressure screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
