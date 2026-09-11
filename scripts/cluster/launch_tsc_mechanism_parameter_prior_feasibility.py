#!/usr/bin/env python3
"""Submit the seven V150C mechanism-parameter-prior target tasks."""

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
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MECHANISM_ACTION_FEATURES,
    MINIMUM_OOF_GAIN,
    RIDGE_L2,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    SOURCE_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v150c-mechanism-parameter-prior-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150c_mechanism_parameter_prior.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
MODULE = "cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility"
SIGNATURE_PREFIX = "CFCMT/v150c/mechanism-parameter-prior-v1"
NODES = (
    "node001",
    "node002",
    "node003",
    "node005",
    "node006",
    "node001",
    "node002",
)


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("protocol")
        != "tsc-v150c-mechanism-parameter-prior-feasibility-v1"
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("adaptation_fold_count", -1)) != FOLD_COUNT
        or tuple(config.get("mechanism_blocks", ()))
        != tuple(MECHANISM_ACTION_FEATURES)
        or float(config.get("ridge_l2", -1.0)) != RIDGE_L2
        or float(config.get("source_prior_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or float(config.get("minimum_oof_gain", -1.0)) != MINIMUM_OOF_GAIN
    ):
        raise ValueError("V150C frozen protocol configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    source_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
    cache_workers: int,
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for city, node in zip(EXPECTED_CITY_GROUPS, NODES, strict=True):
        remote_city = remote_output_root / city
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
                MODULE,
                "target",
                "--target-city",
                city,
                "--fit-protocol",
                str(snapshot_root / FIT_PROTOCOL_RELATIVE),
                "--source-cache-root",
                str(source_cache_root),
                "--source-manifest",
                str(snapshot_root / SOURCE_MANIFEST_RELATIVE),
                "--source-cache-audit",
                str(source_cache_audit_remote),
                "--source-cache-audit-sha256",
                str(source_cache_audit_sha256),
                "--conversion-root",
                str(conversion_root),
                "--cache-workers",
                str(cache_workers),
                "--out",
                str(remote_city / "result.json"),
            ]
        ) + " && printf 'TASK_DONE\\n'"
        specs.append(
            {
                "description": f"CFCMT V150C mechanism prior target {city}",
                "project": "CFCMT",
                "cmd": "mkdir -p "
                + shlex.quote(str(remote_city))
                + " && "
                + command,
                "cwd": str(snapshot_root),
                "signature": f"{SIGNATURE_PREFIX}/{city}",
                "resource_family": "CFCMT-v150c-mechanism-parameter-prior",
                "ram_resource_family": "CFCMT-v150c-mechanism-parameter-prior",
                "vram": 0,
                "ram_mb": 65_536,
                "cpu": 8,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(remote_city),
                "local_result_dir": str((local_output_root / city).resolve()),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150C launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V150C cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or not source_audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("V150C source cache audit changed")
    source_audit_sha = _sha256(args.source_cache_audit_local)
    snapshot = _read_json(args.stage_manifest)
    snapshot_root = Path(snapshot["snapshot_root"])
    remote_city_roots = [args.remote_output_root / city for city in EXPECTED_CITY_GROUPS]
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "-d", str(args.source_cache_root)]),
            shlex.join(["test", "-d", str(args.conversion_root)]),
            shlex.join(["test", "-f", str(args.source_cache_audit_remote)]),
            shlex.join(["test", "-f", str(snapshot_root / FIT_PROTOCOL_RELATIVE)]),
            shlex.join(["test", "-f", str(snapshot_root / SOURCE_MANIFEST_RELATIVE)]),
            *(
                shlex.join(["test", "!", "-e", str(path)])
                for path in remote_city_roots
            ),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote V150C preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", str(args.source_cache_audit_remote)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in str(stdout).splitlines() if line.split()]
    if int(hash_code) != 0 or remote_hashes != [source_audit_sha]:
        raise RuntimeError(
            f"remote V150C audit identity changed: {remote_hashes}; {hash_stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        source_cache_audit_sha256=source_audit_sha,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        cache_workers=args.cache_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150c-mechanism-parameter-prior-v1",
        ],
        input=json.dumps(specs),
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
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "source_cache_audit_sha256": source_audit_sha,
        "remote_output_root": str(args.remote_output_root),
        "task_specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150C target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
