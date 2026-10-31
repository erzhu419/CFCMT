#!/usr/bin/env python3
"""Launch the frozen v64 hierarchical development matrix on six CPU nodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _bounded_direct_output,
    _direct_remote_command,
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


LAUNCH_PROTOCOL = "v64r60-hierarchical-closed-loop-development-six-node-launch-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_hierarchical_closed_loop_development.json"
)
PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v44_external_v9_hierarchical_guard_confirmation.json"
)
ENVIRONMENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v41_external_v9_full_budget_refit.json"
)
ENVIRONMENT_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v40_external_v9_network_repair_confirmation.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    offline_result_local: Path,
    offline_result_remote: Path,
    support_audit_local: Path,
    support_audit_remote: Path,
    freeze_audit_local: Path,
    freeze_audit_remote: Path,
    freeze_root: Path,
    conversion_root: Path,
    conversion_manifest: Path,
    remote_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    parent_path = PROJECT_ROOT / PARENT_RELATIVE
    environment_path = PROJECT_ROOT / ENVIRONMENT_RELATIVE
    environment_parent_path = PROJECT_ROOT / ENVIRONMENT_PARENT_RELATIVE
    manifest_path = PROJECT_ROOT / MANIFEST_RELATIVE
    protocol = _read_json(protocol_path)
    hashes = {
        "protocol": _sha256(protocol_path),
        "parent_protocol": _sha256(parent_path),
        "offline_result": _sha256(offline_result_local),
        "support_audit": _sha256(support_audit_local),
        "freeze_audit": _sha256(freeze_audit_local),
        "environment_protocol": _sha256(environment_path),
        "environment_parent": _sha256(environment_parent_path),
        "external_manifest": _sha256(manifest_path),
    }
    identities = all_rollout_identities(protocol)
    if (
        protocol.get("protocol") != PROTOCOL
        or hashes["parent_protocol"] != str(protocol["parent_protocol"]["sha256"])
        or hashes["offline_result"]
        != str(protocol["offline_authorization"]["sha256"])
        or hashes["support_audit"]
        != str(protocol["support_runtime_equivalence"]["sha256"])
        or hashes["freeze_audit"]
        != str(protocol["hierarchical_freeze_audit"]["sha256"])
        or hashes["environment_protocol"]
        != str(protocol["environment"]["protocol"]["sha256"])
        or hashes["environment_parent"]
        != str(protocol["environment"]["network_parent_protocol"]["sha256"])
        or hashes["external_manifest"]
        != str(protocol["environment"]["external_manifest"]["sha256"])
        or len(identities) != 416
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("hierarchical development launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_hierarchical_closed_loop_development_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--parent-protocol",
                str(snapshot_root / PARENT_RELATIVE),
                "--offline-result",
                str(offline_result_remote),
                "--support-audit",
                str(support_audit_remote),
                "--freeze-audit",
                str(freeze_audit_remote),
                "--freeze-root",
                str(freeze_root),
                "--environment-protocol",
                str(snapshot_root / ENVIRONMENT_RELATIVE),
                "--environment-parent",
                str(snapshot_root / ENVIRONMENT_PARENT_RELATIVE),
                "--external-manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root",
                str(conversion_root),
                "--conversion-manifest",
                str(conversion_manifest),
                "--results-root",
                str(remote_results_root),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(NODES)),
                "--workers",
                str(workers),
            ]
        )
        shard_size = sum(
            index % len(NODES) == shard_index for index in range(len(identities))
        )
        specs.append(
            {
                "description": f"CFCMT v64 hierarchical development shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v64r60/hierarchical-development/shard-{shard_index}",
                "resource_family": "CFCMT-hierarchical-closed-loop-v2",
                "vram": 0,
                "ram_mb": int(ram_mb),
                "cpu": int(cpu_cores),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "result_dir": str(remote_results_root),
                "matrix_rows": int(shard_size),
            }
        )
    return specs, hashes


def _remote_preflight(
    *,
    scheduler_path: Path,
    snapshot_root: Path,
    conversion_root: Path,
    expected_conversion_tree_sha256: str,
    remote_results_root: Path,
    allow_existing_results: bool,
) -> dict[str, Any]:
    scheduler = _load_scheduler(scheduler_path)
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    tree_command = shlex.join(
        [
            str(wrapper),
            "-c",
            (
                "from pathlib import Path; "
                "from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256; "
                f"print(_tree_sha256(Path({str(conversion_root)!r})))"
            ),
        ]
    )
    spec = {
        "cwd": str(snapshot_root),
        "cmd": tree_command,
        "extra_env": {
            "CFCMT_SOURCE_ROOT": str(snapshot_root),
            "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root),
        },
    }
    code, stdout, stderr = scheduler.run_on(
        "node001", _direct_remote_command(spec), timeout=600, check=False
    )
    observed_tree = str(stdout).strip().splitlines()[-1] if str(stdout).strip() else ""
    count_command = (
        f"find {shlex.quote(str(remote_results_root))} -type f 2>/dev/null | wc -l"
    )
    count_code, count_stdout, count_stderr = scheduler.run_on(
        "node001", count_command, timeout=120, check=False
    )
    file_count = int(str(count_stdout).strip() or 0)
    result = {
        "node": "node001",
        "conversion_tree_sha256": observed_tree,
        "conversion_tree_matches": int(code) == 0
        and observed_tree == str(expected_conversion_tree_sha256),
        "future_result_file_count": file_count,
        "future_results_empty": int(count_code) == 0 and file_count == 0,
        "resume_requested": bool(allow_existing_results),
        "stderr": _bounded_direct_output(stderr),
        "count_stderr": _bounded_direct_output(count_stderr),
    }
    result["passed"] = bool(
        result["conversion_tree_matches"]
        and (result["future_results_empty"] or bool(allow_existing_results))
    )
    if not result["passed"]:
        raise RuntimeError(f"hierarchical development remote preflight failed: {result}")
    return result


def _remote_postflight(
    *,
    scheduler_path: Path,
    snapshot_root: Path,
    conversion_root: Path,
    expected_conversion_tree_sha256: str,
    remote_results_root: Path,
    expected_result_file_count: int,
) -> dict[str, Any]:
    scheduler = _load_scheduler(scheduler_path)
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    tree_command = shlex.join(
        [
            str(wrapper),
            "-c",
            (
                "from pathlib import Path; "
                "from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256; "
                f"print(_tree_sha256(Path({str(conversion_root)!r})))"
            ),
        ]
    )
    spec = {
        "cwd": str(snapshot_root),
        "cmd": tree_command,
        "extra_env": {
            "CFCMT_SOURCE_ROOT": str(snapshot_root),
            "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root),
        },
    }
    code, stdout, stderr = scheduler.run_on(
        "node001", _direct_remote_command(spec), timeout=600, check=False
    )
    observed_tree = str(stdout).strip().splitlines()[-1] if str(stdout).strip() else ""
    count_command = (
        f"find {shlex.quote(str(remote_results_root))} -type f 2>/dev/null | wc -l"
    )
    count_code, count_stdout, count_stderr = scheduler.run_on(
        "node001", count_command, timeout=120, check=False
    )
    file_count = int(str(count_stdout).strip() or 0)
    result = {
        "node": "node001",
        "conversion_tree_sha256": observed_tree,
        "conversion_tree_matches": int(code) == 0
        and observed_tree == str(expected_conversion_tree_sha256),
        "result_file_count": file_count,
        "expected_result_file_count": int(expected_result_file_count),
        "result_file_count_matches": int(count_code) == 0
        and file_count == int(expected_result_file_count),
        "stderr": _bounded_direct_output(stderr),
        "count_stderr": _bounded_direct_output(count_stderr),
    }
    result["passed"] = bool(
        result["conversion_tree_matches"] and result["result_file_count_matches"]
    )
    if not result["passed"]:
        raise RuntimeError(f"hierarchical development remote postflight failed: {result}")
    return result


def _run_direct_without_sync(
    specs: Sequence[Mapping[str, Any]], *, scheduler_path: Path, timeout: int
) -> list[dict[str, Any]]:
    scheduler = _load_scheduler(scheduler_path)

    def run(spec: Mapping[str, Any]) -> dict[str, Any]:
        code, stdout, stderr = scheduler.run_on(
            str(spec["require_node"]),
            _direct_remote_command(spec),
            timeout=int(timeout),
            check=False,
        )
        return {
            "node": str(spec["require_node"]),
            "shard": str(spec["signature"]).rsplit("-", 1)[-1],
            "returncode": int(code),
            "stdout": _bounded_direct_output(stdout),
            "stderr": _bounded_direct_output(stderr),
        }

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        rows = list(pool.map(run, specs))
    failures = [row for row in rows if row["returncode"] != 0]
    if failures:
        raise RuntimeError(f"hierarchical development shards failed: {failures}")
    return rows


def _sync_results(
    *, remote_results_root: Path, local_results_root: Path, scheduler_path: Path
) -> None:
    if local_results_root.exists():
        raise FileExistsError(f"refusing to overwrite local results: {local_results_root}")
    scheduler = _load_scheduler(scheduler_path)
    local_results_root.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:{remote_results_root}/",
            f"{local_results_root}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"hierarchical development sync failed: {completed.stderr}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--offline-result-local", type=Path, required=True)
    parser.add_argument("--offline-result-remote", type=Path, required=True)
    parser.add_argument("--support-audit-local", type=Path, required=True)
    parser.add_argument("--support-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-audit-local", type=Path, required=True)
    parser.add_argument("--freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--cpu-cores", type=int, default=68)
    parser.add_argument("--ram-mb", type=int, default=131072)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    specs, hashes = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        offline_result_local=args.offline_result_local,
        offline_result_remote=args.offline_result_remote,
        support_audit_local=args.support_audit_local,
        support_audit_remote=args.support_audit_remote,
        freeze_audit_local=args.freeze_audit_local,
        freeze_audit_remote=args.freeze_audit_remote,
        freeze_root=args.freeze_root,
        conversion_root=args.conversion_root,
        conversion_manifest=args.conversion_manifest,
        remote_results_root=args.remote_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "input_hashes": hashes,
        "conversion_tree_sha256": protocol["environment"][
            "conversion_tree_sha256"
        ],
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "workers_per_shard": int(args.workers),
        "node_count": len(specs),
        "matrix_size": sum(int(spec["matrix_rows"]) for spec in specs),
        "specs": specs,
    }
    try:
        if args.direct:
            payload["remote_preflight"] = _remote_preflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                allow_existing_results=bool(args.resume),
            )
            payload["direct_results"] = _run_direct_without_sync(
                specs, scheduler_path=args.scheduler, timeout=7200
            )
            payload["remote_postflight"] = _remote_postflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root,
                expected_conversion_tree_sha256=str(
                    protocol["environment"]["conversion_tree_sha256"]
                ),
                remote_results_root=args.remote_results_root,
                expected_result_file_count=(2 * len(all_rollout_identities(protocol)))
                + len(NODES),
            )
            _sync_results(
                remote_results_root=args.remote_results_root,
                local_results_root=args.local_results_root,
                scheduler_path=args.scheduler,
            )
            payload["submitted"] = True
            payload["execution_mode"] = "direct_six_physical_nodes"
    except Exception as exc:
        payload["execution_error"] = _bounded_direct_output(repr(exc))
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "node_count": len(specs),
                "matrix_size": payload["matrix_size"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
