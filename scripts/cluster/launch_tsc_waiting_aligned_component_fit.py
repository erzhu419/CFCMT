#!/usr/bin/env python3
"""Submit LA and Jinan estimand-aligned B0/B100 mechanism fits."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_source_waiting_aligned_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_target_waiting_aligned_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as TARGET_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "tsc-v115-pure-waiting-component-fit-launch-v3"
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
TARGET_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
CITY_NODES = {"los_angeles": "node001", "jinan": "node002"}


def remote_cache_count_command(paths: Sequence[Path]) -> str:
    substitutions = [
        '"$(' + shlex.join(["find", str(path), "-type", "f", "-name", "*.npz"])
        + ' | wc -l)"'
        for path in paths
    ]
    return "printf '%s\\n' " + " ".join(substitutions)


def build_specs(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    source_cache_root: Path,
    source_cache_audit: Path,
    target_cache_root: Path,
    target_cache_audit: Path,
    remote_output_root: Path,
    cache_workers: int,
    fit_workers: int,
    city_nodes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    execution_nodes = CITY_NODES if city_nodes is None else city_nodes
    if set(execution_nodes) != set(CITY_NODES):
        raise ValueError(
            "component-fit city node map must contain los_angeles and jinan"
        )
    specs = []
    for city in CITY_NODES:
        node = str(execution_nodes[city])
        output_root = remote_output_root / city
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
                "cf_h2o.eval.traffic_signal_waiting_aligned_component_fit",
                "--fit-protocol",
                str(snapshot_root / FIT_PROTOCOL_RELATIVE),
                "--source-cache-root",
                str(source_cache_root),
                "--source-manifest",
                str(snapshot_root / SOURCE_MANIFEST_RELATIVE),
                "--source-cache-audit",
                str(source_cache_audit),
                "--target-cache-root",
                str(target_cache_root),
                "--target-manifest",
                str(snapshot_root / TARGET_MANIFEST_RELATIVE),
                "--target-cache-audit",
                str(target_cache_audit),
                "--city",
                city,
                "--cache-workers",
                str(cache_workers),
                "--fit-workers",
                str(fit_workers),
                "--output-root",
                str(output_root),
            ]
        )
        specs.append(
            {
                "description": f"CFCMT V115 pure-waiting B0/B100 fit: {city}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v115/pure-waiting-fit-v3/{city}",
                "resource_family": "CFCMT-v115-pure-waiting-fit-v3",
                "vram": 0,
                "ram_mb": 98304,
                "cpu": 20,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(output_root),
            }
        )
    return specs


def _require_passed_audit(path: Path, protocol: str) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("protocol") != protocol
        or payload.get("status") != "PASS"
        or not payload.get("gate", {}).get("passed", False)
    ):
        raise ValueError(f"cache audit does not authorize fit: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-audit-local", type=Path, required=True)
    parser.add_argument("--target-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fit-workers", type=int, default=7)
    parser.add_argument("--los-angeles-node", default=CITY_NODES["los_angeles"])
    parser.add_argument("--jinan-node", default=CITY_NODES["jinan"])
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite component-fit launch artifact")
    source_audit = _require_passed_audit(
        args.source_cache_audit_local, SOURCE_AUDIT_PROTOCOL
    )
    target_audit = _require_passed_audit(
        args.target_cache_audit_local, TARGET_AUDIT_PROTOCOL
    )
    if int(args.fit_workers) < 1 or int(args.fit_workers) > 7:
        raise ValueError("component fit-workers must be in [1, 7]")
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        source_cache_root=args.source_cache_root,
        source_cache_audit=args.source_cache_audit_remote,
        target_cache_root=args.target_cache_root,
        target_cache_audit=args.target_cache_audit_remote,
        remote_output_root=args.remote_output_root,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
        city_nodes={
            "los_angeles": str(args.los_angeles_node),
            "jinan": str(args.jinan_node),
        },
    )
    scheduler = _load_scheduler(args.scheduler)
    checks = [
        "test",
        "!",
        "-e",
        str(args.remote_output_root),
        "-a",
        "-d",
        str(args.source_cache_root),
        "-a",
        "-d",
        str(args.target_cache_root),
        "-a",
        "-f",
        str(args.source_cache_audit_remote),
        "-a",
        "-f",
        str(args.target_cache_audit_remote),
    ]
    preflight_node = str(args.los_angeles_node)
    code, _, stderr = scheduler.run_on(
        preflight_node, shlex.join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote fit preconditions failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        preflight_node,
        shlex.join(
            [
                "sha256sum",
                str(args.source_cache_audit_remote),
                str(args.target_cache_audit_remote),
            ]
        ),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    expected_hashes = [
        _sha256(args.source_cache_audit_local),
        _sha256(args.target_cache_audit_local),
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote cache-audit identities differ from local evidence: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    count_code, count_stdout, count_stderr = scheduler.run_on(
        preflight_node,
        remote_cache_count_command(
            [args.source_cache_root, args.target_cache_root]
        ),
        timeout=300,
        check=False,
    )
    try:
        remote_cache_counts = [
            int(line.strip())
            for line in str(count_stdout).splitlines()
            if line.strip()
        ]
    except ValueError as exc:
        raise RuntimeError(
            f"remote cache counts are not integers: {count_stdout}"
        ) from exc
    expected_cache_counts = [
        int(source_audit["cache_file_count"]),
        int(target_audit["cache_file_count"]),
    ]
    if int(count_code) != 0 or remote_cache_counts != expected_cache_counts:
        raise RuntimeError(
            "remote cache counts differ from audited evidence: "
            f"{remote_cache_counts} != {expected_cache_counts}; {count_stderr}"
        )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v115-pure-waiting-component-fit-v3",
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
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_cache_audit": {
            "local": str(args.source_cache_audit_local.resolve()),
            "remote": str(args.source_cache_audit_remote),
            "sha256": _sha256(args.source_cache_audit_local),
            "training_scenario_count": source_audit["training_scenario_count"],
        },
        "target_cache_audit": {
            "local": str(args.target_cache_audit_local.resolve()),
            "remote": str(args.target_cache_audit_remote),
            "sha256": _sha256(args.target_cache_audit_local),
            "scenario_count": target_audit["scenario_count"],
        },
        "remote_output_root": str(args.remote_output_root),
        "task_count": len(specs),
        "fit_workers_per_city": int(args.fit_workers),
        "execution_node_map": {
            "los_angeles": str(args.los_angeles_node),
            "jinan": str(args.jinan_node),
        },
        "remote_audit_sha256": remote_hashes,
        "remote_cache_file_counts": remote_cache_counts,
        "specs": specs,
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if completed.returncode != 0:
        raise RuntimeError("component-fit scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
