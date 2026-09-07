#!/usr/bin/env python3
"""Submit six node-local process shards for frozen v51 development."""

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

from cf_h2o.eval.traffic_signal_external_execution_trust_region_development import (  # noqa: E402
    FAILED_V50_DECISION,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


TRUST_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v37_external_execution_trust_region_development.json"
)
EXPANDED_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v36_external_policy_horizon_expanded_development.json"
)
ROBUSTNESS_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v35_external_policy_horizon_robustness.json"
)
ADAPTATION_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v34_external_policy_horizon_adaptation.json"
)
PRESSURE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v33_external_pressure_regularized_development.json"
)
LAUNCH_PROTOCOL = "v51r47-execution-trust-region-development-six-node-submission-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")


def build_specs(
    *,
    snapshot_root: Path,
    failed_v50_audit_local: Path,
    failed_v50_audit_remote: Path,
    pressure_joint_audit_local: Path,
    pressure_joint_audit_remote: Path,
    pressure_freeze_root: Path,
    offline_authorization_local: Path,
    offline_authorization_remote: Path,
    parent_joint_audit_local: Path,
    parent_joint_audit_remote: Path,
    freeze_root: Path,
    conversion_root: Path,
    conversion_manifest: Path,
    external_manifest_sha256: str,
    conversion_manifest_sha256: str,
    conversion_tree_sha256: str,
    remote_results_root: Path,
    local_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    trust_path = PROJECT_ROOT / TRUST_PROTOCOL_RELATIVE
    expanded_path = PROJECT_ROOT / EXPANDED_PROTOCOL_RELATIVE
    robustness_path = PROJECT_ROOT / ROBUSTNESS_PROTOCOL_RELATIVE
    adaptation_path = PROJECT_ROOT / ADAPTATION_PROTOCOL_RELATIVE
    pressure_path = PROJECT_ROOT / PRESSURE_PROTOCOL_RELATIVE
    trust = _read_json(trust_path)
    expanded = _read_json(expanded_path)
    robustness = _read_json(robustness_path)
    adaptation = _read_json(adaptation_path)
    failed = _read_json(failed_v50_audit_local)
    hashes = {
        "trust_protocol": _sha256(trust_path),
        "expanded_protocol": _sha256(expanded_path),
        "robustness_protocol": _sha256(robustness_path),
        "adaptation_protocol": _sha256(adaptation_path),
        "pressure_protocol": _sha256(pressure_path),
        "failed_v50_audit": _sha256(failed_v50_audit_local),
        "pressure_joint_audit": _sha256(pressure_joint_audit_local),
        "offline_authorization": _sha256(offline_authorization_local),
        "parent_joint_audit": _sha256(parent_joint_audit_local),
    }
    identities = all_rollout_identities(trust, adaptation)
    identity_seeds = {int(identity[2]) for identity in identities}
    validation_seeds = {
        int(value) for value in trust["new_untouched_validation"]["seeds"]
    }
    prospective_seeds = {
        int(value)
        for value in trust["prospective_confirmation_reservation"]["seeds"]
    }
    city_scope = dict(trust["city_deployment_scope"])
    authorized = (
        trust.get("protocol") == PROTOCOL
        and failed.get("status") == "PASS"
        and failed.get("decision") == FAILED_V50_DECISION
        and hashes["failed_v50_audit"]
        == str(trust["failed_v50_selection_audit"]["sha256"])
        and hashes["expanded_protocol"]
        == str(trust["parent_expanded_protocol"]["sha256"])
        and hashes["adaptation_protocol"]
        == str(trust["candidate_grid_protocol"]["sha256"])
        and hashes["robustness_protocol"]
        == str(expanded["parent_robustness_protocol"]["sha256"])
        and hashes["adaptation_protocol"]
        == str(robustness["parent_adaptation_protocol"]["sha256"])
        and hashes["pressure_protocol"]
        == str(adaptation["parent_pressure_protocol"]["sha256"])
        and hashes["pressure_joint_audit"]
        == str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        and len(identities) == int(trust["development_grid"]["matrix_size"])
        and len(identities) == 240
        and {identity[0] for identity in identities} == {"los_angeles"}
        and trust["development_grid"]["coordination_mode"] == "sparse"
        and city_scope["los_angeles"]["mode"] == "active_development"
        and city_scope["jinan"]["mode"] == "frozen_phase_pressure_fallback"
        and not identity_seeds.intersection(validation_seeds)
        and not identity_seeds.intersection(prospective_seeds)
        and not validation_seeds.intersection(prospective_seeds)
    )
    if not authorized:
        raise ValueError("v51 execution-trust-region launch authorization changed")

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_execution_trust_region_development_shard.py"
    )
    trust_protocol = snapshot_root / TRUST_PROTOCOL_RELATIVE
    expanded_protocol = snapshot_root / EXPANDED_PROTOCOL_RELATIVE
    robustness_protocol = snapshot_root / ROBUSTNESS_PROTOCOL_RELATIVE
    adaptation_protocol = snapshot_root / ADAPTATION_PROTOCOL_RELATIVE
    pressure_protocol = snapshot_root / PRESSURE_PROTOCOL_RELATIVE
    parent_protocol = snapshot_root / PROTOCOL_RELATIVE
    external_manifest = snapshot_root / EXTERNAL_MANIFEST_RELATIVE
    specs = []
    for shard_index, node in enumerate(NODES):
        remote_shard = Path(remote_results_root) / "_shards" / f"shard_{shard_index}.json"
        local_shard = Path(local_results_root) / "_shards" / f"shard_{shard_index}.json"
        if local_shard.exists():
            raise FileExistsError(f"refusing to overwrite v51 shard: {local_shard}")
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--trust-protocol",
                str(trust_protocol),
                "--expanded-protocol",
                str(expanded_protocol),
                "--adaptation-protocol",
                str(adaptation_protocol),
                "--robustness-protocol",
                str(robustness_protocol),
                "--failed-v50-audit",
                str(failed_v50_audit_remote),
                "--expected-failed-v50-audit-sha256",
                hashes["failed_v50_audit"],
                "--pressure-protocol",
                str(pressure_protocol),
                "--pressure-joint-audit",
                str(pressure_joint_audit_remote),
                "--expected-pressure-joint-audit-sha256",
                hashes["pressure_joint_audit"],
                "--pressure-freeze-root",
                str(pressure_freeze_root),
                "--protocol-spec",
                str(parent_protocol),
                "--offline-authorization",
                str(offline_authorization_remote),
                "--expected-offline-authorization-sha256",
                hashes["offline_authorization"],
                "--joint-freeze-audit",
                str(parent_joint_audit_remote),
                "--expected-joint-freeze-sha256",
                hashes["parent_joint_audit"],
                "--freeze-root",
                str(freeze_root),
                "--external-manifest",
                str(external_manifest),
                "--conversion-root",
                str(conversion_root),
                "--expected-external-manifest-sha256",
                external_manifest_sha256,
                "--conversion-manifest",
                str(conversion_manifest),
                "--expected-conversion-manifest-sha256",
                conversion_manifest_sha256,
                "--expected-conversion-tree-sha256",
                conversion_tree_sha256,
                "--expected-sumo-version",
                "1.22.0",
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
            1 for index in range(len(identities)) if index % len(NODES) == shard_index
        )
        specs.append(
            {
                "description": f"CFCMT v51 execution trust-region shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v51r47/execution-trust-region/shard-{shard_index}",
                "resource_family": "CFCMT-v51-execution-trust-region-process-shard-v1",
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
                "result_dir": str(remote_shard.parent),
                "local_result_dir": str(local_shard.parent),
                "matrix_rows": shard_size,
            }
        )
    return specs, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--failed-v50-audit-local", type=Path, required=True)
    parser.add_argument("--failed-v50-audit-remote", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-local", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--offline-authorization-local", type=Path, required=True)
    parser.add_argument("--offline-authorization-remote", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-local", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--cpu-cores", type=int, default=40)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, hashes = build_specs(
        snapshot_root=snapshot_root,
        failed_v50_audit_local=args.failed_v50_audit_local,
        failed_v50_audit_remote=args.failed_v50_audit_remote,
        pressure_joint_audit_local=args.pressure_joint_audit_local,
        pressure_joint_audit_remote=args.pressure_joint_audit_remote,
        pressure_freeze_root=args.pressure_freeze_root,
        offline_authorization_local=args.offline_authorization_local,
        offline_authorization_remote=args.offline_authorization_remote,
        parent_joint_audit_local=args.parent_joint_audit_local,
        parent_joint_audit_remote=args.parent_joint_audit_remote,
        freeze_root=args.freeze_root,
        conversion_root=args.conversion_root,
        conversion_manifest=args.conversion_manifest,
        external_manifest_sha256=args.external_manifest_sha256,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
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
        "input_hashes": hashes,
        "node_count": len(specs),
        "matrix_size": sum(int(spec["matrix_rows"]) for spec in specs),
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
                "CFCMT-v51r47-execution-trust-region-development",
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
            raise RuntimeError("v51 execution-trust-region submission failed")
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
