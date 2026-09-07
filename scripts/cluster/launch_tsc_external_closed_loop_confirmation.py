#!/usr/bin/env python3
"""Submit the authorized v43 external closed-loop confirmation matrix."""

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
    METHOD_POLICY,
    POLICIES,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_city_oof_freeze import (  # noqa: E402
    EXTERNAL_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _atomic_json,
    _read_json,
    _sha256,
)


PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json"
)
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")


def build_specs(
    *,
    snapshot_root: Path,
    offline_authorization_local: Path,
    offline_authorization_remote: Path,
    joint_audit_local: Path,
    joint_audit_remote: Path,
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
) -> tuple[list[dict[str, Any]], str, str]:
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    authorization = _read_json(offline_authorization_local)
    joint = _read_json(joint_audit_local)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v43 closed-loop protocol changed")
    authorization_sha256 = _sha256(offline_authorization_local)
    joint_sha256 = _sha256(joint_audit_local)
    if (
        authorization.get("protocol")
        != "tsc-v43r39-external-seed8171-heldout-evaluation-v1"
        or authorization.get("decision") != AUTHORIZATION_DECISION
        or not bool(
            authorization.get("offline_confirmation_gate", {}).get(
                "passed", False
            )
        )
        or authorization.get("joint_freeze_audit_sha256") != joint_sha256
        or joint.get("status") != "PASS"
        or joint.get("decision") != "authorize_external_seed8171_collection"
    ):
        raise ValueError("offline evidence does not authorize closed-loop matrix")

    closed_loop = dict(protocol["closed_loop"])
    if (
        not bool(closed_loop.get("enabled_only_after_offline_gate", False))
        or int(closed_loop.get("horizon_sec", -1)) != 3600
        or tuple((METHOD_POLICY, *closed_loop["baselines"])) != POLICIES
    ):
        raise ValueError("v43 closed-loop matrix contract changed")
    scenarios = tuple(
        scenario
        for city_scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].values()
        for scenario in city_scenarios
    )
    seeds = tuple(
        int(value)
        for value in protocol["target_protocol"]["closed_loop_seeds"]
    )
    combinations = tuple(
        (str(scenario), int(seed), str(policy))
        for scenario in scenarios
        for seed in seeds
        for policy in POLICIES
    )
    if len(combinations) != 4 * 2 * 7 or len(set(combinations)) != len(combinations):
        raise ValueError("v43 closed-loop matrix cardinality changed")

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / EXTERNAL_MANIFEST_RELATIVE
    protocol_path = snapshot_root / PROTOCOL_RELATIVE
    specs = []
    for index, (scenario, seed, policy) in enumerate(combinations):
        node = NODES[index % len(NODES)]
        remote_result = remote_results_root / scenario / f"seed_{seed}" / policy
        local_result = local_results_root / scenario / f"seed_{seed}" / policy
        if local_result.exists():
            raise FileExistsError(
                f"refusing to overwrite closed-loop result: {local_result}"
            )
        command = shlex.join(
            [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_closed_loop_confirmation",
                "--protocol-spec",
                str(protocol_path),
                "--offline-authorization",
                str(offline_authorization_remote),
                "--expected-offline-authorization-sha256",
                authorization_sha256,
                "--joint-freeze-audit",
                str(joint_audit_remote),
                "--expected-joint-freeze-sha256",
                joint_sha256,
                "--freeze-root",
                str(freeze_root),
                "--external-manifest",
                str(manifest),
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
                    f"CFCMT v43r39 closed loop {scenario} seed {seed} {policy}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": (
                    f"CFCMT/v43r39/closed-loop-v4/{scenario}/seed-{seed}/{policy}"
                ),
                "resource_family": "CFCMT-external-closed-loop-v4",
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
    return specs, authorization_sha256, joint_sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--offline-authorization-local", type=Path, required=True)
    parser.add_argument("--offline-authorization-remote", type=Path, required=True)
    parser.add_argument("--joint-audit-local", type=Path, required=True)
    parser.add_argument("--joint-audit-remote", type=Path, required=True)
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
    specs, authorization_sha256, joint_sha256 = build_specs(
        snapshot_root=snapshot_root,
        offline_authorization_local=args.offline_authorization_local,
        offline_authorization_remote=args.offline_authorization_remote,
        joint_audit_local=args.joint_audit_local,
        joint_audit_remote=args.joint_audit_remote,
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
        "protocol": "v43r39-external-closed-loop-submission-v4",
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "offline_authorization_sha256": authorization_sha256,
        "joint_freeze_audit_sha256": joint_sha256,
        "freeze_root": str(args.freeze_root),
        "conversion_root": str(args.conversion_root),
        "conversion_manifest": str(args.conversion_manifest),
        "external_manifest_sha256": args.external_manifest_sha256,
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
        "task_count": len(specs),
        "policies": list(POLICIES),
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
                "CFCMT-v43r39-external-closed-loop",
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
            raise RuntimeError("v43 external closed-loop submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "nodes": list(NODES),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
