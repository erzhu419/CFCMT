#!/usr/bin/env python3
"""Submit the frozen v41 12-network seed-heldout ensemble matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v24_seed_heldout_b100_ensemble.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
ASSIGNMENTS = {
    "node001": ("grid4x4", "arterial4x4"),
    "node002": ("cologne3", "cologne8"),
    "node003": ("ingolstadt7", "ingolstadt21"),
    "node004": ("atlanta_1x5", "hangzhou_4x4"),
    "node005": ("hangzhou_4x4_hetero", "manhattan_28x7"),
    "node006": (
        "saltlake_400s_200w_q1_weekday_peak",
        "saltlake_state_university_q1_weekday_peak",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_specs(
    *,
    snapshot_root: Path,
    cache_root: Path,
    baseline_result: Path,
    remote_results_root: Path,
    local_results_root: Path,
    cache_sha256: str,
    baseline_sha256: str,
    cpu_cores: int,
    ram_mb: int,
    cache_workers: int,
) -> list[dict[str, Any]]:
    manifest = snapshot_root / MANIFEST_RELATIVE
    protocol_spec = snapshot_root / PROTOCOL_RELATIVE
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_seed_heldout_anchored_ensemble_shard.py"
    )
    local_protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    expected_targets = set(local_protocol["development"]["evaluation_targets"])
    assigned = [target for targets in ASSIGNMENTS.values() for target in targets]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected_targets:
        raise ValueError("v41 node assignments do not exactly cover eligible targets")
    specs = []
    for node, targets in ASSIGNMENTS.items():
        for target in targets:
            target_remote = remote_results_root / target
            target_local = local_results_root / target
            if target_local.exists():
                raise FileExistsError(
                    f"refusing to overwrite local v41 target: {target_local}"
                )
            command = shlex.join(
                [
                    str(wrapper),
                    str(runner),
                    "--cache-root",
                    str(cache_root),
                    "--baseline-result",
                    str(baseline_result),
                    "--manifest",
                    str(manifest),
                    "--protocol-spec",
                    str(protocol_spec),
                    "--target",
                    target,
                    "--expected-cache-sha256",
                    cache_sha256,
                    "--expected-baseline-sha256",
                    baseline_sha256,
                    "--out-root",
                    str(remote_results_root),
                    "--cache-workers",
                    str(cache_workers),
                ]
            )
            specs.append(
                {
                    "description": f"CFCMT TSC v41 seed-heldout ensemble {target}",
                    "project": "CFCMT-TSC-v41",
                    "cmd": command,
                    "cwd": str(PROJECT_ROOT),
                    "signature": f"CFCMT/tsc-v41/seed-heldout-ensemble/{target}",
                    "vram": 0,
                    "ram_mb": int(ram_mb),
                    "cpu": int(cpu_cores),
                    "priority": "normal",
                    "require_node": node,
                    "skip_launch_staging": True,
                    "env_spec": "none",
                    "extra_env": {"CFCMT_SOURCE_ROOT": str(snapshot_root)},
                    "result_dir": str(target_remote),
                    "local_result_dir": str(target_local),
                }
            )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--cpu-cores", type=int, default=24)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--cache-workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    local_protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    cache_sha256 = str(local_protocol["combined_counterfactual_cache_sha256"])
    baseline_sha256 = str(local_protocol["expanded_source_rule_baseline_sha256"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        cache_sha256=cache_sha256,
        baseline_sha256=baseline_sha256,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
        cache_workers=args.cache_workers,
    )
    payload: dict[str, Any] = {
        "protocol": "v41r37-seed-heldout-ensemble-scheduleurm-submission-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "cache_sha256": cache_sha256,
        "baseline_sha256": baseline_sha256,
        "assignments": {node: list(targets) for node, targets in ASSIGNMENTS.items()},
        "specs": specs,
    }
    if args.submit:
        completed = subprocess.run(
            [
                str(args.scheduler),
                "submit-jsonl",
                "--stdin",
                "--trusted",
                "--json",
                "--intent-label",
                "CFCMT-v41-seed-heldout-ensemble",
            ],
            input=json.dumps(specs),
            text=True,
            capture_output=True,
            check=False,
        )
        payload.update(
            {
                "submitted": completed.returncode == 0,
                "scheduler_returncode": int(completed.returncode),
                "scheduler_stdout": completed.stdout,
                "scheduler_stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            _atomic_json(args.out, payload)
            raise RuntimeError("scheduleurm v41 bulk submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "snapshot_sha256": stage["snapshot_sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
