#!/usr/bin/env python3
"""Launch the frozen v81 validation matrix on node001-node006."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_target_regime_trust_validation import (  # noqa: E402
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    _remote_postflight,
    _remote_preflight,
    _run_direct_without_sync,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)
from scripts.cluster.launch_tsc_external_topology_veto_closed_loop_development import (  # noqa: E402
    _input_visibility,
    _sync_new,
    _sync_results,
)


LAUNCH_PROTOCOL = "v81r77-target-regime-validation-six-node-launch-v1"
NODES = tuple(f"node{index:03d}" for index in range(1, 7))
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v64_external_v9_"
    "target_regime_trust_validation.json"
)
PROPOSAL_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    hierarchical_freeze_audit_remote: Path,
    hierarchical_freeze_root_remote: Path,
    veto_freeze_result_remote: Path,
    veto_joint_audit_remote: Path,
    veto_artifact_root_remote: Path,
    regime_freeze_result_remote: Path,
    regime_artifact_remote: Path,
    conversion_root_remote: Path,
    conversion_manifest_remote: Path,
    remote_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> list[dict[str, Any]]:
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    identities = all_rollout_identities(protocol)
    if (
        protocol.get("protocol") != PROTOCOL
        or len(identities) != 128
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("v81 launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = snapshot_root / "scripts/cluster/run_tsc_external_target_regime_trust_validation_shard.py"
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper), str(runner),
                "--protocol", str(snapshot_root / PROTOCOL_RELATIVE),
                "--proposal-parent-protocol", str(snapshot_root / PROPOSAL_PARENT_RELATIVE),
                "--hierarchical-freeze-audit", str(hierarchical_freeze_audit_remote),
                "--hierarchical-freeze-root", str(hierarchical_freeze_root_remote),
                "--veto-freeze-result", str(veto_freeze_result_remote),
                "--veto-joint-audit", str(veto_joint_audit_remote),
                "--veto-artifact-root", str(veto_artifact_root_remote),
                "--regime-freeze-result", str(regime_freeze_result_remote),
                "--regime-artifact", str(regime_artifact_remote),
                "--external-manifest", str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root", str(conversion_root_remote),
                "--conversion-manifest", str(conversion_manifest_remote),
                "--results-root", str(remote_results_root),
                "--shard-index", str(shard_index),
                "--shard-count", str(len(NODES)),
                "--workers", str(workers),
            ]
        )
        specs.append(
            {
                "project": "CFCMT",
                "signature": f"CFCMT/v81r77/target-regime-validation/shard-{shard_index}",
                "description": f"CFCMT v81 target-regime validation shard {shard_index}",
                "cmd": command,
                "cwd": str(snapshot_root),
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root_remote),
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                },
                "cpu": int(cpu_cores),
                "ram_mb": int(ram_mb),
                "vram": 0,
                "priority": "high",
                "resource_family": "CFCMT-target-regime-validation-v1",
                "require_node": node,
                "result_dir": str(remote_results_root),
                "skip_launch_staging": True,
                "matrix_rows": sum(1 for i in range(len(identities)) if i % len(NODES) == shard_index),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root-remote", type=Path, required=True)
    parser.add_argument("--veto-freeze-result-local", type=Path, required=True)
    parser.add_argument("--veto-freeze-result-remote", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-local", type=Path, required=True)
    parser.add_argument("--veto-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-local", type=Path, required=True)
    parser.add_argument("--veto-artifact-root-remote", type=Path, required=True)
    parser.add_argument("--regime-freeze-result-local", type=Path, required=True)
    parser.add_argument("--regime-freeze-result-remote", type=Path, required=True)
    parser.add_argument("--regime-artifact-local", type=Path, required=True)
    parser.add_argument("--regime-artifact-remote", type=Path, required=True)
    parser.add_argument("--conversion-root-remote", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-cores", type=int, default=16)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol = _read_json(protocol_path)
    specs = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        hierarchical_freeze_audit_remote=args.hierarchical_freeze_audit_remote,
        hierarchical_freeze_root_remote=args.hierarchical_freeze_root_remote,
        veto_freeze_result_remote=args.veto_freeze_result_remote,
        veto_joint_audit_remote=args.veto_joint_audit_remote,
        veto_artifact_root_remote=args.veto_artifact_root_remote,
        regime_freeze_result_remote=args.regime_freeze_result_remote,
        regime_artifact_remote=args.regime_artifact_remote,
        conversion_root_remote=args.conversion_root_remote,
        conversion_manifest_remote=args.conversion_manifest_remote,
        remote_results_root=args.remote_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    artifact_checks = [
        (
            args.veto_artifact_root_remote / path.relative_to(args.veto_artifact_root_local),
            _sha256(path),
        )
        for path in sorted(args.veto_artifact_root_local.rglob("*"))
        if path.is_file()
    ]
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "frozen_protocol_sha256": _sha256(protocol_path),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "physical_nodes": list(NODES),
        "matrix_size": len(all_rollout_identities(protocol)),
        "workers_per_shard": int(args.workers),
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "result_protocol": RESULT_PROTOCOL,
        "specs": specs,
    }
    try:
        if args.plan_only:
            payload["execution_mode"] = "plan_only"
        else:
            for local, remote, directory in (
                (args.veto_freeze_result_local, args.veto_freeze_result_remote, False),
                (args.veto_joint_audit_local, args.veto_joint_audit_remote, False),
                (args.veto_artifact_root_local, args.veto_artifact_root_remote, True),
                (args.regime_freeze_result_local, args.regime_freeze_result_remote, False),
                (args.regime_artifact_local, args.regime_artifact_remote, False),
            ):
                _sync_new(local_path=local, remote_path=remote, scheduler_path=args.scheduler, directory=directory)
            checks = [
                (args.veto_freeze_result_remote, _sha256(args.veto_freeze_result_local)),
                (args.veto_joint_audit_remote, _sha256(args.veto_joint_audit_local)),
                *artifact_checks,
                (args.regime_freeze_result_remote, _sha256(args.regime_freeze_result_local)),
                (args.regime_artifact_remote, _sha256(args.regime_artifact_local)),
            ]
            payload["input_visibility"] = _input_visibility(checks=checks, scheduler_path=args.scheduler)
            payload["remote_preflight"] = _remote_preflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(protocol["environment"]["conversion_tree_sha256"]),
                remote_results_root=args.remote_results_root,
                allow_existing_results=False,
            )
            payload["direct_results"] = _run_direct_without_sync(specs, scheduler_path=args.scheduler, timeout=10800)
            payload["remote_postflight"] = _remote_postflight(
                scheduler_path=args.scheduler,
                snapshot_root=Path(stage["snapshot_root"]),
                conversion_root=args.conversion_root_remote,
                expected_conversion_tree_sha256=str(protocol["environment"]["conversion_tree_sha256"]),
                remote_results_root=args.remote_results_root,
                expected_result_file_count=2 * len(all_rollout_identities(protocol)) + len(NODES),
            )
            payload["sync_attempts"] = _sync_results(
                remote_results_root=args.remote_results_root,
                local_results_root=args.local_results_root,
                scheduler_path=args.scheduler,
            )
            payload["submitted"] = True
            payload["execution_mode"] = "direct_six_physical_nodes"
    except Exception as exc:
        payload["execution_error"] = repr(exc)[:8000]
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(json.dumps({"submitted": payload["submitted"], "node_count": len(specs), "matrix_size": payload["matrix_size"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
