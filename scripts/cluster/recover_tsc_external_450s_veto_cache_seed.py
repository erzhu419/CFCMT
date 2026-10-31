#!/usr/bin/env python3
"""Recover one failed v69 cache seed and preserve the failed launch evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cluster.launch_tsc_external_450s_veto_cache import (  # noqa: E402
    DIRECT_TIMEOUT_SEC,
    LAUNCH_PROTOCOL,
    NODES,
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
from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_450s_veto_protocol import (  # noqa: E402
    PROTOCOL,
)


RECOVERY_PROTOCOL = "v69r65-450s-veto-single-seed-compute-recovery-v1"


def _remote_file_exists(
    path: Path, *, node: str, scheduler_path: Path
) -> bool:
    scheduler = _load_scheduler(scheduler_path)
    code, _, _ = scheduler.run_on(
        node,
        f"test -f {shlex.quote(str(path))}",
        timeout=120,
        check=False,
    )
    return int(code) == 0


def _sync_directory(
    *, remote: Path, local: Path, node: str, scheduler_path: Path
) -> None:
    if local.exists():
        raise FileExistsError(f"refusing to overwrite recovered result: {local}")
    local.mkdir(parents=True, exist_ok=False)
    scheduler = _load_scheduler(scheduler_path)
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node(node),
            f"{scheduler._ssh_target_for_node(node)}:{remote}/",
            f"{local}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"v69 result sync failed for {node}: {completed.stderr[-4000:]}"
        )


def _sync_cache(
    *,
    remote_cache_root: Path,
    local_cache_root: Path,
    scheduler_path: Path,
    retries: int,
) -> list[dict[str, Any]]:
    if local_cache_root.exists():
        raise FileExistsError(
            f"refusing to overwrite recovered cache: {local_cache_root}"
        )
    local_cache_root.mkdir(parents=True, exist_ok=False)
    scheduler = _load_scheduler(scheduler_path)
    attempts = []
    for attempt in range(1, int(retries) + 1):
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--timeout=180",
                "-e",
                scheduler._ssh_rsync_shell_for_node("node001"),
                f"{scheduler._ssh_target_for_node('node001')}:{remote_cache_root}/",
                f"{local_cache_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        row = {
            "attempt": attempt,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        attempts.append(row)
        if completed.returncode == 0:
            return attempts
        if attempt < int(retries):
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"v69 cache sync retries exhausted: {attempts[-1]}")


def _summary_evidence(
    *,
    specs: Sequence[Mapping[str, Any]],
    local_results_root: Path,
    source_tree_sha256: str,
) -> list[dict[str, Any]]:
    evidence = []
    for spec in specs:
        node = str(spec["require_node"])
        seed_name = Path(str(spec["result_dir"])).name
        path = local_results_root / seed_name / "cache_summary.json"
        result = _read_json(path)
        setting = dict(result.get("setting", {}))
        runtime = dict(result.get("runtime", {}))
        if (
            result.get("experiment")
            != "traffic_signal_counterfactual_cache_precompute"
            or runtime.get("hostname") != node
            or runtime.get("source_tree_sha256") != source_tree_sha256
            or len(tuple(setting.get("seeds", ()))) != 1
            or seed_name != f"seed{int(tuple(setting['seeds'])[0])}"
        ):
            raise ValueError(f"v69 recovered summary contract changed: {path}")
        evidence.append(
            {
                "node": node,
                "seed": int(tuple(setting["seeds"])[0]),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )
    return evidence


def recover(
    *,
    failed_launch_path: Path,
    protocol_path: Path,
    remote_cache_root: Path,
    local_cache_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    retries: int,
) -> dict[str, Any]:
    failed = _read_json(failed_launch_path)
    protocol = _read_json(protocol_path)
    specs = tuple(dict(row) for row in failed.get("specs", ()))
    failure_text = str(failed.get("execution_error", ""))
    if not (
        failed.get("protocol") == LAUNCH_PROTOCOL
        and failed.get("submitted") is False
        and "direct hierarchical freeze failed" in failure_text
        and protocol.get("protocol") == PROTOCOL
        and failed.get("frozen_protocol_sha256") == _sha256(protocol_path)
        and len(specs) == len(NODES) == 6
        and {str(spec.get("require_node")) for spec in specs} == set(NODES)
        and Path(str(failed.get("remote_cache_root"))) == remote_cache_root
        and all(
            Path(str(spec.get("result_dir"))).parent == remote_results_root
            for spec in specs
        )
    ):
        raise ValueError("v69 failed-launch recovery precondition changed")

    present_before = {}
    for spec in specs:
        summary = Path(str(spec["result_dir"])) / "cache_summary.json"
        present_before[str(spec["require_node"])] = _remote_file_exists(
            summary, node=str(spec["require_node"]), scheduler_path=scheduler_path
        )
    missing = [
        spec for spec in specs if not present_before[str(spec["require_node"])]
    ]
    if len(missing) != 1:
        raise ValueError(
            f"v69 recovery requires exactly one missing seed summary: {present_before}"
        )
    missing_spec = missing[0]
    missing_node = str(missing_spec["require_node"])
    if missing_node not in failure_text:
        raise ValueError("missing v69 node is not identified by failed launcher")

    scheduler = _load_scheduler(scheduler_path)
    code, stdout, stderr = scheduler.run_on(
        missing_node,
        _direct_remote_command(missing_spec),
        timeout=DIRECT_TIMEOUT_SEC,
        check=False,
    )
    recovery_result = {
        "node": missing_node,
        "city": Path(str(missing_spec["result_dir"])).name,
        "returncode": int(code),
        "stdout": _bounded_direct_output(stdout),
        "stderr": _bounded_direct_output(stderr),
        "evidence_kind": "targeted_recovery_returncode_and_final_summary",
        "targeted_recovery": True,
    }
    if int(code) != 0:
        raise RuntimeError(f"v69 targeted seed recovery failed: {recovery_result}")

    for spec in specs:
        _sync_directory(
            remote=Path(str(spec["result_dir"])),
            local=local_results_root / Path(str(spec["result_dir"])).name,
            node=str(spec["require_node"]),
            scheduler_path=scheduler_path,
        )
    sync_attempts = _sync_cache(
        remote_cache_root=remote_cache_root,
        local_cache_root=local_cache_root,
        scheduler_path=scheduler_path,
        retries=retries,
    )
    summary_evidence = _summary_evidence(
        specs=specs,
        local_results_root=local_results_root,
        source_tree_sha256=str(failed["source_tree_sha256"]),
    )
    expected_count = int(
        protocol["long_horizon_collection"]["expected_cache_file_count"]
    )
    remote_command = (
        f"find {shlex.quote(str(remote_cache_root))} -maxdepth 1 -type f "
        "-name '*.npz' | wc -l"
    )
    remote_code, remote_stdout, remote_stderr = scheduler.run_on(
        "node001", remote_command, timeout=120, check=False
    )
    remote_count = int(str(remote_stdout).strip() or -1)
    local_count = len(tuple(local_cache_root.glob("*.npz")))
    gate = {
        "original_failure_preserved": bool(failure_text),
        "exactly_one_summary_missing_before_recovery": len(missing) == 1,
        "missing_node_named_in_original_failure": missing_node in failure_text,
        "targeted_recovery_returncode_clean": int(code) == 0,
        "six_final_summaries_valid": len(summary_evidence) == len(specs) == 6,
        "six_final_physical_hosts_exact": {
            row["node"] for row in summary_evidence
        }
        == set(NODES),
        "cache_sync_clean": sync_attempts[-1]["returncode"] == 0,
        "remote_cache_count_exact": int(remote_code) == 0
        and remote_count == expected_count,
        "local_cache_count_exact": local_count == expected_count,
    }
    gate["passed"] = all(gate.values())
    if not gate["passed"]:
        raise RuntimeError(
            f"v69 targeted recovery gate failed: {gate}; stderr={remote_stderr}"
        )

    recovered_node = missing_node
    direct_results = []
    evidence_by_node = {row["node"]: row for row in summary_evidence}
    for spec in specs:
        node = str(spec["require_node"])
        if node == recovered_node:
            direct_results.append(
                {
                    **recovery_result,
                    "summary_sha256": evidence_by_node[node]["sha256"],
                }
            )
        else:
            direct_results.append(
                {
                    "node": node,
                    "city": Path(str(spec["result_dir"])).name,
                    "returncode": 0,
                    "stdout": "not retained after original six-node launcher raised",
                    "stderr": "",
                    "evidence_kind": (
                        "original_failure_list_excludes_node_and_final_summary_valid"
                    ),
                    "reconstructed_returncode": True,
                    "summary_sha256": evidence_by_node[node]["sha256"],
                }
            )
    return {
        **failed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": True,
        "execution_mode": "direct_run_on_six_physical_nodes",
        "direct_results": direct_results,
        "cache_synced": True,
        "execution_error": None,
        "original_execution_error": failure_text,
        "recovery": {
            "protocol": RECOVERY_PROTOCOL,
            "failed_launch_manifest": str(failed_launch_path.resolve()),
            "failed_launch_manifest_sha256": _sha256(failed_launch_path),
            "reason": "single_seed_compute_startup_failure",
            "missing_before_recovery": [
                {
                    "node": missing_node,
                    "seed_result": Path(str(missing_spec["result_dir"])).name,
                }
            ],
            "present_before_recovery": present_before,
            "targeted_result": recovery_result,
            "summary_evidence": summary_evidence,
            "sync_attempts": sync_attempts,
            "remote_cache_file_count": remote_count,
            "local_cache_file_count": local_count,
            "gate": gate,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-launch", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--local-cache-root", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v69 recovery: {args.out}")
    payload = recover(
        failed_launch_path=args.failed_launch,
        protocol_path=args.protocol,
        remote_cache_root=args.remote_cache_root,
        local_cache_root=args.local_cache_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        scheduler_path=args.scheduler,
        retries=args.retries,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "cache_synced": payload["cache_synced"],
                "recovered_node": payload["recovery"]["missing_before_recovery"][0][
                    "node"
                ],
                "cache_file_count": payload["recovery"][
                    "local_cache_file_count"
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
