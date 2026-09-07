#!/usr/bin/env python3
"""Finalize v69 after successful seed recovery and a result-sync interruption."""

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

from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_450s_veto_protocol import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_450s_veto_cache import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)
from scripts.cluster.recover_tsc_external_450s_veto_cache_seed import (  # noqa: E402
    RECOVERY_PROTOCOL,
)


FINALIZATION_PROTOCOL = "v69r65-450s-veto-post-compute-sync-finalization-v1"


def _sync_directory_resumable(
    *,
    remote: Path,
    local: Path,
    node: str,
    scheduler_path: Path,
    retries: int,
) -> list[dict[str, Any]]:
    local.mkdir(parents=True, exist_ok=True)
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
                scheduler._ssh_rsync_shell_for_node(node),
                f"{scheduler._ssh_target_for_node(node)}:{remote}/",
                f"{local}/",
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
    raise RuntimeError(f"v69 result sync retries exhausted for {node}: {attempts[-1]}")


def _sync_cache_resumable(
    *,
    remote_cache_root: Path,
    local_cache_root: Path,
    scheduler_path: Path,
    retries: int,
) -> list[dict[str, Any]]:
    local_cache_root.mkdir(parents=True, exist_ok=True)
    return _sync_directory_resumable(
        remote=remote_cache_root,
        local=local_cache_root,
        node="node001",
        scheduler_path=scheduler_path,
        retries=retries,
    )


def _summary_evidence(
    *,
    specs: Sequence[Mapping[str, Any]],
    local_results_root: Path,
    source_tree_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        node = str(spec["require_node"])
        seed_name = Path(str(spec["result_dir"])).name
        path = local_results_root / seed_name / "cache_summary.json"
        result = _read_json(path)
        seeds = tuple(int(value) for value in result.get("setting", {}).get("seeds", ()))
        if (
            result.get("experiment")
            != "traffic_signal_counterfactual_cache_precompute"
            or result.get("runtime", {}).get("hostname") != node
            or result.get("runtime", {}).get("source_tree_sha256")
            != source_tree_sha256
            or len(seeds) != 1
            or seed_name != f"seed{seeds[0]}"
        ):
            raise ValueError(f"v69 finalized summary contract changed: {path}")
        rows.append(
            {
                "node": node,
                "seed": seeds[0],
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )
    return rows


def finalize(
    *,
    failed_launch_path: Path,
    protocol_path: Path,
    remote_cache_root: Path,
    local_cache_root: Path,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
    retries: int,
    prior_transport_error: str,
) -> dict[str, Any]:
    failed = _read_json(failed_launch_path)
    protocol = _read_json(protocol_path)
    specs = tuple(dict(row) for row in failed.get("specs", ()))
    original_error = str(failed.get("execution_error", ""))
    failed_specs = [
        spec
        for spec in specs
        if str(spec.get("require_node")) in original_error
        and Path(str(spec.get("result_dir"))).name in original_error
    ]
    if not (
        failed.get("protocol") == LAUNCH_PROTOCOL
        and failed.get("submitted") is False
        and "direct hierarchical freeze failed" in original_error
        and "FileNotFoundError" in original_error
        and "cache_authorization_v1.json" in original_error
        and protocol.get("protocol") == PROTOCOL
        and failed.get("frozen_protocol_sha256") == _sha256(protocol_path)
        and len(specs) == len(NODES) == 6
        and {str(spec.get("require_node")) for spec in specs} == set(NODES)
        and len(failed_specs) == 1
        and str(failed_specs[0]["require_node"]) == "node001"
        and Path(str(failed_specs[0]["result_dir"])).name == "seed80314"
        and Path(str(failed.get("remote_cache_root"))) == remote_cache_root
        and all(
            Path(str(spec["result_dir"])).parent == remote_results_root
            for spec in specs
        )
    ):
        raise ValueError("v69 finalization precondition changed")

    local_before = {
        Path(str(spec["result_dir"])).name: (
            local_results_root
            / Path(str(spec["result_dir"])).name
            / "cache_summary.json"
        ).is_file()
        for spec in specs
    }
    scheduler = _load_scheduler(scheduler_path)
    remote_before = {}
    for spec in specs:
        node = str(spec["require_node"])
        summary = Path(str(spec["result_dir"])) / "cache_summary.json"
        code, _, _ = scheduler.run_on(
            node,
            f"test -f {shlex.quote(str(summary))}",
            timeout=120,
            check=False,
        )
        remote_before[Path(str(spec["result_dir"])).name] = int(code) == 0
    if not all(remote_before.values()):
        raise ValueError(f"v69 computation is incomplete before finalization: {remote_before}")

    result_sync_attempts = {}
    for spec in specs:
        seed_name = Path(str(spec["result_dir"])).name
        result_sync_attempts[seed_name] = _sync_directory_resumable(
            remote=Path(str(spec["result_dir"])),
            local=local_results_root / seed_name,
            node=str(spec["require_node"]),
            scheduler_path=scheduler_path,
            retries=retries,
        )
    cache_sync_attempts = _sync_cache_resumable(
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
    count_command = (
        f"find {shlex.quote(str(remote_cache_root))} -maxdepth 1 -type f "
        "-name '*.npz' | wc -l"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", count_command, timeout=120, check=False
    )
    remote_count = int(str(stdout).strip() or -1)
    local_count = len(tuple(local_cache_root.glob("*.npz")))
    gate = {
        "original_compute_failure_preserved": bool(original_error),
        "original_failure_is_authorization_visibility_only": (
            "FileNotFoundError" in original_error
            and "cache_authorization_v1.json" in original_error
        ),
        "single_failed_identity_exact": len(failed_specs) == 1,
        "six_remote_summaries_before_finalization": all(remote_before.values()),
        "partial_transport_state_observed": sum(local_before.values()) == 5
        and local_before.get("seed50945") is False,
        "prior_transport_error_recorded": "kex_exchange_identification" in prior_transport_error
        and "node006" in prior_transport_error,
        "all_result_syncs_clean": all(
            attempts[-1]["returncode"] == 0
            for attempts in result_sync_attempts.values()
        ),
        "cache_sync_clean": cache_sync_attempts[-1]["returncode"] == 0,
        "six_final_summaries_valid": len(summary_evidence) == 6,
        "six_final_physical_hosts_exact": {
            row["node"] for row in summary_evidence
        }
        == set(NODES),
        "remote_cache_count_exact": int(code) == 0
        and remote_count == expected_count,
        "local_cache_count_exact": local_count == expected_count,
    }
    gate["passed"] = all(gate.values())
    if not gate["passed"]:
        raise RuntimeError(f"v69 finalization gate failed: {gate}; stderr={stderr}")

    evidence_by_node = {row["node"]: row for row in summary_evidence}
    recovered_node = "node001"
    direct_results = []
    for spec in specs:
        node = str(spec["require_node"])
        direct_results.append(
            {
                "node": node,
                "city": Path(str(spec["result_dir"])).name,
                "returncode": 0,
                "stdout": "not retained after launcher/recovery transport exceptions",
                "stderr": "",
                "evidence_kind": (
                    "targeted_recovery_control_flow_and_final_summary"
                    if node == recovered_node
                    else "original_failure_list_excludes_node_and_final_summary"
                ),
                "reconstructed_returncode": True,
                "targeted_recovery": node == recovered_node,
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
        "original_execution_error": original_error,
        "recovery": {
            "protocol": RECOVERY_PROTOCOL,
            "finalization_protocol": FINALIZATION_PROTOCOL,
            "failed_launch_manifest": str(failed_launch_path.resolve()),
            "failed_launch_manifest_sha256": _sha256(failed_launch_path),
            "reason": "authorization_visibility_failure_then_targeted_seed_recovery",
            "targeted_recovery_identity": {
                "node": recovered_node,
                "seed_result": "seed80314",
            },
            "targeted_recovery_success_proof": (
                "The recovery program entered result synchronization only after the "
                "targeted command returned zero; the node001 summary is complete, "
                "source-bound, and included in the exact 768-file cache."
            ),
            "post_compute_transport_failure": {
                "error": prior_transport_error,
                "local_summary_presence_before_finalization": local_before,
            },
            "remote_summary_presence_before_finalization": remote_before,
            "summary_evidence": summary_evidence,
            "result_sync_attempts": result_sync_attempts,
            "sync_attempts": cache_sync_attempts,
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
    parser.add_argument("--prior-transport-error", required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v69 finalization: {args.out}")
    payload = finalize(
        failed_launch_path=args.failed_launch,
        protocol_path=args.protocol,
        remote_cache_root=args.remote_cache_root,
        local_cache_root=args.local_cache_root,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        scheduler_path=args.scheduler,
        retries=args.retries,
        prior_transport_error=args.prior_transport_error,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "cache_synced": payload["cache_synced"],
                "cache_file_count": payload["recovery"]["local_cache_file_count"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
