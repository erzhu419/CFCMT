#!/usr/bin/env python3
"""Submit the v47 contaminated-seed pressure-regularized falsification matrix."""

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

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (  # noqa: E402
    AUTHORIZATION_DECISION,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (  # noqa: E402
    FALSIFICATION_RESULT_PROTOCOL,
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    PROTOCOL,
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


PRESSURE_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v33_external_pressure_regularized_development.json"
)
LAUNCH_PROTOCOL = "v47r43-pressure-regularized-falsification-submission-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")


def build_specs(
    *,
    snapshot_root: Path,
    offline_authorization_local: Path,
    offline_authorization_remote: Path,
    parent_joint_audit_local: Path,
    parent_joint_audit_remote: Path,
    pressure_joint_audit_local: Path,
    pressure_joint_audit_remote: Path,
    pressure_freeze_root: Path,
    freeze_root: Path,
    conversion_root: Path,
    conversion_manifest: Path,
    external_manifest_sha256: str,
    conversion_manifest_sha256: str,
    conversion_tree_sha256: str,
    remote_results_root: Path,
    local_results_root: Path,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    protocol = _read_json(PROJECT_ROOT / PRESSURE_PROTOCOL_RELATIVE)
    authorization = _read_json(offline_authorization_local)
    parent_joint = _read_json(parent_joint_audit_local)
    pressure_joint = _read_json(pressure_joint_audit_local)
    hashes = {
        "offline_authorization": _sha256(offline_authorization_local),
        "parent_joint_audit": _sha256(parent_joint_audit_local),
        "pressure_joint_audit": _sha256(pressure_joint_audit_local),
    }
    if (
        protocol.get("protocol") != PROTOCOL
        or authorization.get("decision") != AUTHORIZATION_DECISION
        or not bool(
            authorization.get("offline_confirmation_gate", {}).get("passed", False)
        )
        or parent_joint.get("status") != "PASS"
        or pressure_joint.get("protocol") != JOINT_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != JOINT_AUDIT_DECISION
        or hashes["pressure_joint_audit"]
        != str(protocol["pressure_regularized_joint_audit"]["sha256"])
    ):
        raise ValueError("v47 pressure-regularized falsification authorization changed")
    scenarios = tuple(
        str(scenario)
        for city_scenarios in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].values()
        for scenario in city_scenarios
    )
    seeds = tuple(
        int(value)
        for value in protocol["contaminated_falsification"]["closed_loop_seeds"]
    )
    policies = tuple(
        str(value) for value in protocol["contaminated_falsification"]["policies"]
    )
    combinations = tuple(
        (scenario, seed, policy)
        for scenario in scenarios
        for seed in seeds
        for policy in policies
    )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    pressure_protocol = snapshot_root / PRESSURE_PROTOCOL_RELATIVE
    parent_protocol = snapshot_root / PROTOCOL_RELATIVE
    external_manifest = snapshot_root / EXTERNAL_MANIFEST_RELATIVE
    specs = []
    for index, (scenario, seed, policy) in enumerate(combinations):
        node = NODES[index % len(NODES)]
        remote_result = Path(remote_results_root) / scenario / f"seed_{seed}" / policy
        local_result = Path(local_results_root) / scenario / f"seed_{seed}" / policy
        if local_result.exists():
            raise FileExistsError(f"refusing to overwrite v47 rollout: {local_result}")
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation",
                "--pressure-protocol",
                str(pressure_protocol),
                "--pressure-joint-audit",
                str(pressure_joint_audit_remote),
                "--expected-pressure-joint-audit-sha256",
                hashes["pressure_joint_audit"],
                "--pressure-freeze-root",
                str(pressure_freeze_root),
                "--analysis-stage",
                "falsification",
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
                str(external_manifest_sha256),
                "--conversion-manifest",
                str(conversion_manifest),
                "--expected-conversion-manifest-sha256",
                str(conversion_manifest_sha256),
                "--expected-conversion-tree-sha256",
                str(conversion_tree_sha256),
                "--expected-sumo-version",
                "1.22.0",
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--policy",
                policy,
                "--tripinfo-out",
                str(remote_result / "tripinfo.xml"),
                "--out",
                str(remote_result / "result.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT v47 falsification {scenario} seed {seed} {policy}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/v47r43/falsification/{scenario}/seed-{seed}/{policy}"
                ),
                "resource_family": "CFCMT-external-pressure-regularized-falsification-v1",
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
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result),
            }
        )
    return specs, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--offline-authorization-local", type=Path, required=True)
    parser.add_argument("--offline-authorization-remote", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-local", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-local", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, hashes = build_specs(
        snapshot_root=snapshot_root,
        offline_authorization_local=args.offline_authorization_local,
        offline_authorization_remote=args.offline_authorization_remote,
        parent_joint_audit_local=args.parent_joint_audit_local,
        parent_joint_audit_remote=args.parent_joint_audit_remote,
        pressure_joint_audit_local=args.pressure_joint_audit_local,
        pressure_joint_audit_remote=args.pressure_joint_audit_remote,
        pressure_freeze_root=args.pressure_freeze_root,
        freeze_root=args.freeze_root,
        conversion_root=args.conversion_root,
        conversion_manifest=args.conversion_manifest,
        external_manifest_sha256=args.external_manifest_sha256,
        conversion_manifest_sha256=args.conversion_manifest_sha256,
        conversion_tree_sha256=args.conversion_tree_sha256,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "result_protocol": FALSIFICATION_RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "input_hashes": hashes,
        "task_count": len(specs),
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
                "CFCMT-v47r43-falsification",
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
            raise RuntimeError("v47 falsification submission failed")
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
