#!/usr/bin/env python3
"""Submit the frozen LA/Jinan v6 full-horizon admission matrix."""

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
    "cf_h2o/config/traffic_signal_tsc_v26_external_la_jinan_confirmation.json"
)
ASSIGNMENTS = (
    ("node001", "la_1x4", 5057),
    ("node002", "la_1x4", 6067),
    ("node003", "jinan_3x4_real", 5057),
    ("node004", "jinan_3x4_real", 6067),
    ("node005", "jinan_3x4_real_2000", 5057),
    ("node006", "jinan_3x4_real_2000", 6067),
    ("node001", "jinan_3x4_real_2500", 5057),
    ("node002", "jinan_3x4_real_2500", 6067),
)


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
    conversion_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    conversion_manifest_sha256: str,
    conversion_tree_sha256: str,
    cpu_cores: int,
    ram_mb: int,
    generation: str = "v6",
) -> list[dict[str, Any]]:
    if generation not in {"v6", "v7", "v8", "v9"}:
        raise ValueError(f"unsupported external admission generation: {generation}")
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    city_scenarios = protocol["target_protocol"]["external_city_scenarios"]
    expected_scenarios = {
        scenario
        for scenarios in city_scenarios.values()
        for scenario in scenarios
    }
    expected_seeds = set(protocol["target_protocol"]["adaptation_seeds"])
    assigned_pairs = {(scenario, seed) for _, scenario, seed in ASSIGNMENTS}
    expected_pairs = {
        (scenario, int(seed))
        for scenario in expected_scenarios
        for seed in expected_seeds
    }
    if assigned_pairs != expected_pairs or len(assigned_pairs) != len(ASSIGNMENTS):
        raise ValueError("v6 admission assignments do not exactly cover the matrix")

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for node, scenario, seed in ASSIGNMENTS:
        name = f"{scenario}_seed{seed}"
        remote_result = remote_results_root / name
        local_result = local_results_root / name
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite local v6 admission: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_network_admission",
                "--conversion-root",
                str(conversion_root),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--horizon-sec",
                "3600",
                "--expected-sumo-version",
                "1.22.0",
                "--minimum-controllable-tls",
                "1",
                "--maximum-collision-count",
                "0",
                "--maximum-teleport-fraction",
                "0",
                "--expected-conversion-manifest-sha256",
                conversion_manifest_sha256,
                "--expected-conversion-tree-sha256",
                conversion_tree_sha256,
                "--progress-interval-sec",
                "300",
                "--out",
                str(remote_result / "admission.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT v42r38 external admission {generation} "
                    f"{scenario} seed {seed}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/v42r38/admission-{generation}/"
                    f"{scenario}/seed-{seed}"
                ),
                "resource_family": f"CFCMT-external-admission-{generation}",
                "vram": 0,
                "ram_mb": int(ram_mb),
                "cpu": int(cpu_cores),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {"CFCMT_SOURCE_ROOT": str(snapshot_root)},
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--cpu-cores", type=int, default=2)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument(
        "--generation",
        choices=("v6", "v7", "v8", "v9"),
        default="v6",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
        generation=args.generation,
    )
    payload: dict[str, Any] = {
        "protocol": (
            f"v42r38-external-la-jinan-admission-{args.generation}-submission-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "assignments": [
            {"node": node, "scenario": scenario, "seed": seed}
            for node, scenario, seed in ASSIGNMENTS
        ],
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
                f"CFCMT-v42r38-external-admission-{args.generation}",
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
            raise RuntimeError("scheduleurm v6 admission bulk submission failed")
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
