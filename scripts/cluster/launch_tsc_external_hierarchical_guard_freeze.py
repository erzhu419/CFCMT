#!/usr/bin/env python3
"""Submit the two-city v60 hierarchical successor freeze."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (  # noqa: E402
    RESULT_PROTOCOL as DIAGNOSTIC_PROTOCOL,
    SUCCESSOR_DECISION,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze import (  # noqa: E402
    RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v60r56-external-v9-hierarchical-guard-freeze-submission-v1"
ASSIGNMENTS = (("node001", "los_angeles"), ("node002", "jinan"))
DIRECT_OUTPUT_LIMIT = 64 * 1024
ALIGNED_PROTOCOL_RELATIVE = (
    "cf_h2o/config/traffic_signal_tsc_v42_external_v9_estimand_aligned_development.json"
)
PRESSURE_PROTOCOL_RELATIVE = (
    "cf_h2o/config/traffic_signal_tsc_v43_external_v9_pressure_regularized_development.json"
)


def _load_scheduler(path: Path):
    sys.path.insert(0, str(Path(path).resolve().parent))
    import scheduler  # type: ignore

    return scheduler


def _bounded_direct_output(value: object) -> str:
    output = str(value)
    if len(output) <= DIRECT_OUTPUT_LIMIT:
        return output
    digest = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
    return (
        output[:DIRECT_OUTPUT_LIMIT]
        + "\n... direct output truncated ...\n"
        + f"original_characters={len(output)} sha256={digest}\n"
    )


def _direct_remote_command(spec: Mapping[str, Any]) -> str:
    environment = " ".join(
        f"{key}={shlex.quote(str(value))}"
        for key, value in dict(spec.get("extra_env", {})).items()
    )
    working_directory = Path(str(spec["cwd"]))
    if not working_directory.is_absolute():
        raise ValueError("direct task cwd must be absolute")
    export = f"export {environment} && " if environment else ""
    return (
        f"cd {shlex.quote(str(working_directory))} && "
        f"{export}{str(spec['cmd'])}"
    )


def _run_direct_specs(
    specs: Sequence[Mapping[str, Any]], *, scheduler_path: Path, timeout: int = 600
) -> list[dict[str, Any]]:
    scheduler = _load_scheduler(scheduler_path)

    def run(spec: Mapping[str, Any]) -> dict[str, Any]:
        node = str(spec["require_node"])
        command = _direct_remote_command(spec)
        code, stdout, stderr = scheduler.run_on(
            node, command, timeout=int(timeout), check=False
        )
        return {
            "node": node,
            "city": Path(str(spec["result_dir"])).name,
            "returncode": int(code),
            "stdout": _bounded_direct_output(stdout),
            "stderr": _bounded_direct_output(stderr),
        }

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        results = list(pool.map(run, specs))
    failures = [row for row in results if int(row["returncode"]) != 0]
    if failures:
        raise RuntimeError(f"direct hierarchical freeze failed: {failures}")
    for spec in specs:
        node = str(spec["require_node"])
        target = scheduler._ssh_target_for_node(node)
        ssh_shell = scheduler._ssh_rsync_shell_for_node(node)
        local_result = Path(str(spec["local_result_dir"]))
        local_result.mkdir(parents=True, exist_ok=False)
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                ssh_shell,
                f"{target}:{spec['result_dir']}/",
                f"{local_result}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"direct result sync failed for {node}: {completed.stderr}"
            )
    return results


def build_specs(
    *,
    snapshot_root: Path,
    source_tree_sha256: str,
    aligned_joint_audit_local: Path,
    aligned_joint_audit_remote: Path,
    aligned_freeze_root: Path,
    pressure_joint_audit_local: Path,
    pressure_joint_audit_remote: Path,
    pressure_freeze_root: Path,
    diagnostic_local: Path,
    diagnostic_remote: Path,
    future_cache_root: Path,
    future_seeds: Sequence[int],
    remote_results_root: Path,
    local_results_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    aligned_sha = _sha256(aligned_joint_audit_local)
    pressure_sha = _sha256(pressure_joint_audit_local)
    diagnostic_sha = _sha256(diagnostic_local)
    diagnostic = _read_json(diagnostic_local)
    if (
        diagnostic.get("protocol") != DIAGNOSTIC_PROTOCOL
        or diagnostic.get("decision") != SUCCESSOR_DECISION
        or not bool(diagnostic.get("diagnostic_gate", {}).get("passed", False))
    ):
        raise ValueError("hierarchical freeze diagnostic authorization changed")
    seeds = tuple(int(value) for value in future_seeds)
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("hierarchical freeze future seeds are invalid")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for node, city in ASSIGNMENTS:
        remote_result = Path(remote_results_root) / city
        local_result = Path(local_results_root) / city
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite hierarchical freeze: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze",
                "--aligned-protocol",
                str(snapshot_root / ALIGNED_PROTOCOL_RELATIVE),
                "--aligned-joint-audit",
                str(aligned_joint_audit_remote),
                "--expected-aligned-joint-audit-sha256",
                aligned_sha,
                "--aligned-freeze-root",
                str(aligned_freeze_root),
                "--pressure-protocol",
                str(snapshot_root / PRESSURE_PROTOCOL_RELATIVE),
                "--pressure-joint-audit",
                str(pressure_joint_audit_remote),
                "--expected-pressure-joint-audit-sha256",
                pressure_sha,
                "--pressure-freeze-root",
                str(pressure_freeze_root),
                "--diagnostic",
                str(diagnostic_remote),
                "--expected-diagnostic-sha256",
                diagnostic_sha,
                "--future-evaluation-cache-root",
                str(future_cache_root),
                "--future-evaluation-seeds",
                *(str(seed) for seed in seeds),
                "--expected-source-tree-sha256",
                str(source_tree_sha256),
                "--city",
                city,
                "--model-out",
                str(remote_result / "model.pkl"),
                "--out",
                str(remote_result / "freeze.json"),
            ]
        )
        specs.append(
            {
                "description": f"CFCMT v60 hierarchical guard freeze {city}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v60r56/hierarchical-guard-freeze/{city}",
                "resource_family": "CFCMT-external-hierarchical-guard-freeze-v1",
                "vram": 0,
                "ram_mb": 4096,
                "cpu": 2,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
    return specs, {
        "aligned_joint_audit": aligned_sha,
        "pressure_joint_audit": pressure_sha,
        "diagnostic": diagnostic_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-local", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-local", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--diagnostic-local", type=Path, required=True)
    parser.add_argument("--diagnostic-remote", type=Path, required=True)
    parser.add_argument("--future-cache-root", type=Path, required=True)
    parser.add_argument("--future-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--direct", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, hashes = build_specs(
        snapshot_root=snapshot_root,
        source_tree_sha256=str(stage["source_tree_sha256"]),
        aligned_joint_audit_local=args.aligned_joint_audit_local,
        aligned_joint_audit_remote=args.aligned_joint_audit_remote,
        aligned_freeze_root=args.aligned_freeze_root,
        pressure_joint_audit_local=args.pressure_joint_audit_local,
        pressure_joint_audit_remote=args.pressure_joint_audit_remote,
        pressure_freeze_root=args.pressure_freeze_root,
        diagnostic_local=args.diagnostic_local,
        diagnostic_remote=args.diagnostic_remote,
        future_cache_root=args.future_cache_root,
        future_seeds=args.future_seeds,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "future_evaluation_seeds": [int(value) for value in args.future_seeds],
        "future_cache_root": str(args.future_cache_root),
        "input_hashes": hashes,
        "task_count": len(specs),
        "specs": specs,
    }
    if args.direct:
        try:
            direct_results = _run_direct_specs(
                specs, scheduler_path=args.scheduler
            )
            payload.update(
                {
                    "submitted": True,
                    "execution_mode": "direct_run_on_physical_nodes",
                    "direct_results": direct_results,
                }
            )
        except Exception as exc:
            payload.update(
                {
                    "execution_mode": "direct_run_on_physical_nodes",
                    "direct_error": repr(exc),
                }
            )
            _atomic_json(args.out, payload)
            raise
    elif args.submit:
        completed = subprocess.run(
            [
                str(args.scheduler),
                "submit-jsonl",
                "--stdin",
                "--trusted",
                "--json",
                "--intent-label",
                "CFCMT-v60r56-hierarchical-guard-freeze",
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
                "execution_mode": "scheduler_queue",
            }
        )
        if completed.returncode != 0:
            _atomic_json(args.out, payload)
            raise RuntimeError("hierarchical freeze submission failed")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
