#!/usr/bin/env python3
"""Submit estimand-aligned development or prospective rollout matrices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (  # noqa: E402
    AUTHORIZATION_DECISION,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    ALIGNED_AUDIT_DECISION,
    ALIGNED_PROTOCOL,
    CONFIRMATION_RESULT_PROTOCOL,
    DEVELOPMENT_RESULT_PROTOCOL,
    EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL,
    EXECUTION_VARIANTS,
    METHOD,
    POLICIES,
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


ALIGNED_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v32_external_oof_validated_support_development.json"
)
LAUNCH_PROTOCOL = "v46r42-external-oof-validated-support-rollout-submission-v1"
NODES = ("node001", "node002", "node003", "node004", "node005", "node006")


def build_specs(
    *,
    snapshot_root: Path,
    analysis_stage: str,
    offline_authorization_local: Path,
    offline_authorization_remote: Path,
    parent_joint_audit_local: Path,
    parent_joint_audit_remote: Path,
    aligned_joint_audit_local: Path,
    aligned_joint_audit_remote: Path,
    aligned_freeze_root: Path,
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
    scenario_filter: Sequence[str] = (),
    seed_filter: Sequence[int] = (),
    policy_filter: Sequence[str] = (),
    execution_variant_filter: Sequence[str] = (),
    run_tag: str = "",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    protocol = _read_json(PROJECT_ROOT / ALIGNED_PROTOCOL_RELATIVE)
    authorization = _read_json(offline_authorization_local)
    parent_joint = _read_json(parent_joint_audit_local)
    aligned_joint = _read_json(aligned_joint_audit_local)
    if protocol.get("protocol") != ALIGNED_PROTOCOL:
        raise ValueError("estimand-aligned rollout protocol changed")
    hashes = {
        "offline_authorization": _sha256(offline_authorization_local),
        "parent_joint_audit": _sha256(parent_joint_audit_local),
        "aligned_joint_audit": _sha256(aligned_joint_audit_local),
    }
    if (
        authorization.get("decision") != AUTHORIZATION_DECISION
        or not bool(authorization.get("offline_confirmation_gate", {}).get("passed", False))
        or parent_joint.get("status") != "PASS"
        or aligned_joint.get("status") != "PASS"
        or aligned_joint.get("decision") != ALIGNED_AUDIT_DECISION
        or protocol["estimand_aligned_joint_audit"]["sha256"]
        != hashes["aligned_joint_audit"]
    ):
        raise ValueError("estimand-aligned rollout authorization changed")

    scenarios = tuple(
        str(scenario)
        for city_scenarios in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].values()
        for scenario in city_scenarios
    )
    if analysis_stage == "development":
        seeds = tuple(int(value) for value in protocol["development"]["closed_loop_seeds"])
        policies = tuple(str(value) for value in protocol["development"]["policies"])
        result_protocol = DEVELOPMENT_RESULT_PROTOCOL
    elif analysis_stage == "confirmation":
        seeds = tuple(
            int(value)
            for value in protocol["prospective_confirmation_reservation"][
                "closed_loop_seeds"
            ]
        )
        policies = POLICIES
        result_protocol = CONFIRMATION_RESULT_PROTOCOL
    else:
        raise ValueError("analysis stage must be development or confirmation")
    requested_scenarios = tuple(str(value) for value in scenario_filter)
    requested_seeds = tuple(int(value) for value in seed_filter)
    requested_policies = tuple(str(value) for value in policy_filter)
    requested_execution_variants = tuple(
        str(value) for value in execution_variant_filter
    )
    if run_tag and re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_tag)) is None:
        raise ValueError("run tag must contain only letters, numbers, dot, dash, or underscore")
    if requested_scenarios:
        if not set(requested_scenarios) <= set(scenarios):
            raise ValueError("scenario filter is outside the frozen matrix")
        scenarios = tuple(value for value in scenarios if value in set(requested_scenarios))
    if requested_seeds:
        if not set(requested_seeds) <= set(seeds):
            raise ValueError("seed filter is outside the frozen matrix")
        seeds = tuple(value for value in seeds if value in set(requested_seeds))
    if requested_policies:
        if not set(requested_policies) <= set(policies):
            raise ValueError("policy filter is outside the frozen matrix")
        policies = tuple(value for value in policies if value in set(requested_policies))
    if requested_execution_variants:
        if (
            analysis_stage != "development"
            or not set(requested_execution_variants) <= set(EXECUTION_VARIANTS)
            or (requested_policies and set(requested_policies) != {METHOD})
        ):
            raise ValueError(
                "execution variants are frozen-method development diagnostics only"
            )
        eligible_cities = {
            str(city)
            for city, artifact in protocol["city_artifacts"].items()
            if str(artifact["deployment_decision"])
            == "deploy_oof_conformal_trust_region"
        }
        eligible_scenarios = {
            str(scenario)
            for city, city_scenarios in protocol[
                "prospective_confirmation_reservation"
            ]["city_scenarios"].items()
            if str(city) in eligible_cities
            for scenario in city_scenarios
        }
        scenarios = tuple(value for value in scenarios if value in eligible_scenarios)
        policies = (METHOD,)
        result_protocol = EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL
        combinations = tuple(
            (scenario, seed, METHOD, variant)
            for scenario in scenarios
            for seed in seeds
            for variant in requested_execution_variants
        )
    else:
        combinations = tuple(
            (scenario, seed, policy, None)
            for scenario in scenarios
            for seed in seeds
            for policy in policies
        )
    if len(combinations) != len(scenarios) * len(seeds) * (
        len(requested_execution_variants) if requested_execution_variants else len(policies)
    ):
        raise ValueError("estimand-aligned rollout matrix cardinality changed")

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    aligned_protocol = snapshot_root / ALIGNED_PROTOCOL_RELATIVE
    parent_protocol = snapshot_root / PROTOCOL_RELATIVE
    external_manifest = snapshot_root / EXTERNAL_MANIFEST_RELATIVE
    specs = []
    for index, (scenario, seed, policy, execution_variant) in enumerate(combinations):
        node = NODES[index % len(NODES)]
        result_policy = (
            f"{METHOD}_{execution_variant}"
            if execution_variant is not None
            else policy
        )
        remote_result = remote_results_root / scenario / f"seed_{seed}" / result_policy
        local_result = local_results_root / scenario / f"seed_{seed}" / result_policy
        if local_result.exists():
            raise FileExistsError(f"refusing to overwrite aligned rollout: {local_result}")
        command_parts = [
                str(wrapper),
                "-m",
                "cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation",
                "--aligned-protocol",
                str(aligned_protocol),
                "--aligned-joint-audit",
                str(aligned_joint_audit_remote),
                "--expected-aligned-joint-audit-sha256",
                hashes["aligned_joint_audit"],
                "--aligned-freeze-root",
                str(aligned_freeze_root),
                "--analysis-stage",
                analysis_stage,
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
        ]
        if execution_variant is not None:
            command_parts.extend(["--execution-variant", execution_variant])
        command_parts.extend(
            [
                "--tripinfo-out",
                str(remote_result / "tripinfo.xml"),
                "--out",
                str(remote_result / "result.json"),
            ]
        )
        command = shlex.join(command_parts)
        signature = (
            f"CFCMT/v46r42/{analysis_stage}/{scenario}/seed-{seed}/{result_policy}"
        )
        if run_tag:
            signature = f"{signature}/{run_tag}"
        specs.append(
            {
                "description": (
                    f"CFCMT v46r42 {analysis_stage} {scenario} seed {seed} "
                    f"{result_policy}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": signature,
                "resource_family": (
                    "CFCMT-external-oof-execution-diagnostic-v1"
                    if execution_variant is not None
                    else f"CFCMT-external-oof-validated-support-{analysis_stage}-v1"
                ),
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
    return specs, {**hashes, "result_protocol": result_protocol}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--analysis-stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--offline-authorization-local", type=Path, required=True)
    parser.add_argument("--offline-authorization-remote", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-local", type=Path, required=True)
    parser.add_argument("--parent-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-local", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit-remote", type=Path, required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
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
    parser.add_argument("--scenario-filter", action="append", default=[])
    parser.add_argument("--seed-filter", action="append", type=int, default=[])
    parser.add_argument("--policy-filter", action="append", choices=POLICIES, default=[])
    parser.add_argument(
        "--execution-variant", action="append", choices=EXECUTION_VARIANTS, default=[]
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.scenario_filter
        or args.seed_filter
        or args.policy_filter
        or args.execution_variant
    ) and not args.run_tag:
        raise ValueError("a filtered diagnostic subset requires --run-tag")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    specs, hashes = build_specs(
        snapshot_root=snapshot_root,
        analysis_stage=args.analysis_stage,
        offline_authorization_local=args.offline_authorization_local,
        offline_authorization_remote=args.offline_authorization_remote,
        parent_joint_audit_local=args.parent_joint_audit_local,
        parent_joint_audit_remote=args.parent_joint_audit_remote,
        aligned_joint_audit_local=args.aligned_joint_audit_local,
        aligned_joint_audit_remote=args.aligned_joint_audit_remote,
        aligned_freeze_root=args.aligned_freeze_root,
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
        scenario_filter=args.scenario_filter,
        seed_filter=args.seed_filter,
        policy_filter=args.policy_filter,
        execution_variant_filter=args.execution_variant,
        run_tag=args.run_tag,
    )
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "analysis_stage": args.analysis_stage,
        "result_protocol": hashes.pop("result_protocol"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "input_hashes": hashes,
        "matrix_filter": {
            "scenarios": list(args.scenario_filter),
            "seeds": list(args.seed_filter),
            "policies": list(args.policy_filter),
            "execution_variants": list(args.execution_variant),
            "role": (
                "post_failure_diagnostic_subset"
                if args.scenario_filter
                or args.seed_filter
                or args.policy_filter
                or args.execution_variant
                else "frozen_complete_matrix"
            ),
            "run_tag": args.run_tag,
        },
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
                f"CFCMT-v46r42-{args.analysis_stage}",
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
            raise RuntimeError("estimand-aligned rollout submission failed")
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "analysis_stage": args.analysis_stage,
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
