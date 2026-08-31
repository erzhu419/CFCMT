#!/usr/bin/env python3
"""Launch source-city component rollouts directly on Linux CPU nodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)
from scripts.cluster.run_tsc_source_city_component_shard import (  # noqa: E402
    SCENARIOS,
    load_candidate_specs,
    shard_identities,
)


LAUNCH_PROTOCOL = "tsc-v93-source-city-component-direct-launch-v1"
GUARD_PROFILE_PROTOCOL = "tsc-v93-source-city-guard-profile-grid-v1"


def _snapshot_path(snapshot_root: Path, local_path: Path) -> Path:
    resolved = local_path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"snapshot input is outside project: {resolved}") from exc
    return snapshot_root / relative


def _command(
    *,
    snapshot_root: Path,
    candidate_spec: Path,
    protocol: Path,
    external_manifest: Path,
    node_shard_index: int,
    args: argparse.Namespace,
) -> str:
    runner = snapshot_root / "scripts/cluster/run_tsc_source_city_component_shard.py"
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    values = [
        str(runner),
        "--seeds",
        *(str(value) for value in args.seeds),
        "--scenarios",
        *(str(value) for value in args.scenarios),
        "--candidate-spec",
        str(candidate_spec),
        "--candidate-spec-sha256",
        str(args.candidate_spec_sha256),
        "--los-angeles-main-model",
        str(args.los_angeles_main_model),
        "--los-angeles-main-model-sha256",
        str(args.los_angeles_main_model_sha256),
        "--los-angeles-component-bundle",
        str(args.los_angeles_component_bundle),
        "--los-angeles-component-bundle-sha256",
        str(args.los_angeles_component_bundle_sha256),
        "--jinan-main-model",
        str(args.jinan_main_model),
        "--jinan-main-model-sha256",
        str(args.jinan_main_model_sha256),
        "--jinan-component-bundle",
        str(args.jinan_component_bundle),
        "--jinan-component-bundle-sha256",
        str(args.jinan_component_bundle_sha256),
        "--protocol",
        str(protocol),
        "--offline-authorization",
        str(args.offline_authorization),
        "--offline-authorization-sha256",
        str(args.offline_authorization_sha256),
        "--joint-audit",
        str(args.joint_audit),
        "--joint-audit-sha256",
        str(args.joint_audit_sha256),
        "--freeze-root",
        str(args.freeze_root),
        "--external-manifest",
        str(external_manifest),
        "--external-manifest-sha256",
        str(args.external_manifest_sha256),
        "--conversion-root",
        str(args.conversion_root),
        "--conversion-manifest",
        str(args.conversion_manifest),
        "--conversion-manifest-sha256",
        str(args.conversion_manifest_sha256),
        "--conversion-tree-sha256",
        str(args.conversion_tree_sha256),
        "--results-root",
        str(args.remote_results_root),
        "--shard-index",
        str(node_shard_index),
        "--shard-count",
        str(args.shard_count),
        "--workers",
        str(args.workers_per_node),
    ]
    for city in ("los_angeles", "jinan"):
        strict_model = getattr(args, f"{city}_strict_target_only_model")
        strict_sha256 = getattr(
            args, f"{city}_strict_target_only_model_sha256"
        )
        if strict_model is not None:
            option = city.replace("_", "-")
            values.extend(
                [
                    f"--{option}-strict-target-only-model",
                    str(strict_model),
                    f"--{option}-strict-target-only-model-sha256",
                    str(strict_sha256),
                ]
            )
    if args.guard_risk_multiplier is not None:
        values.extend(
            [
                "--guard-risk-multiplier",
                str(args.guard_risk_multiplier),
                "--guard-min-context-trust",
                str(args.guard_min_context_trust),
                "--guard-margin",
                str(args.guard_margin),
                "--guard-max-relative-rule-gap",
                str(args.guard_max_relative_rule_gap),
            ]
        )
    return " ".join(
        [
            f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(args.conversion_root))}",
            "PYTHONUNBUFFERED=1",
            shlex.quote(str(wrapper)),
            *(shlex.quote(value) for value in values),
        ]
    )


def _run_node(
    scheduler: Any,
    node: str,
    command: str,
    timeout_sec: int,
) -> dict[str, Any]:
    started = time.monotonic()
    code, stdout, stderr = scheduler.run_on(
        node, command, timeout=timeout_sec, check=False
    )
    return {
        "node": node,
        "returncode": int(code),
        "elapsed_sec": float(time.monotonic() - started),
        "stdout_tail": str(stdout)[-4000:],
        "stderr_tail": str(stderr)[-4000:],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--candidate-spec-local", type=Path, required=True)
    parser.add_argument("--candidate-spec-sha256", required=True)
    parser.add_argument("--protocol-local", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--external-manifest-local", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--shard-indices", type=int, nargs="+", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers-per-node", type=int, required=True)
    parser.add_argument("--los-angeles-main-model", type=Path, required=True)
    parser.add_argument("--los-angeles-main-model-sha256", required=True)
    parser.add_argument("--los-angeles-component-bundle", type=Path, required=True)
    parser.add_argument("--los-angeles-component-bundle-sha256", required=True)
    parser.add_argument("--jinan-main-model", type=Path, required=True)
    parser.add_argument("--jinan-main-model-sha256", required=True)
    parser.add_argument("--jinan-component-bundle", type=Path, required=True)
    parser.add_argument("--jinan-component-bundle-sha256", required=True)
    parser.add_argument("--los-angeles-strict-target-only-model", type=Path)
    parser.add_argument("--los-angeles-strict-target-only-model-sha256")
    parser.add_argument("--jinan-strict-target-only-model", type=Path)
    parser.add_argument("--jinan-strict-target-only-model-sha256")
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--offline-authorization-sha256", required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--joint-audit-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--guard-profile-spec-local", type=Path)
    parser.add_argument("--guard-profile-spec-sha256")
    parser.add_argument("--guard-profile-key")
    parser.add_argument("--guard-risk-multiplier", type=float)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.0)
    parser.add_argument("--guard-margin", type=float, default=0.0)
    parser.add_argument("--guard-max-relative-rule-gap", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    for city in ("los_angeles", "jinan"):
        strict_values = (
            getattr(args, f"{city}_strict_target_only_model"),
            getattr(args, f"{city}_strict_target_only_model_sha256"),
        )
        if any(value is not None for value in strict_values) and not all(
            value is not None for value in strict_values
        ):
            raise ValueError(
                f"{city} strict target-only model arguments are incomplete"
            )

    if len(args.nodes) != len(args.shard_indices):
        raise ValueError("nodes and shard indices must have the same length")
    if len(set(args.nodes)) != len(args.nodes):
        raise ValueError("launch nodes must be unique")
    if len(set(args.shard_indices)) != len(args.shard_indices):
        raise ValueError("launch shard indices must be unique")
    if any(not 0 <= value < args.shard_count for value in args.shard_indices):
        raise ValueError("launch shard index is outside shard count")
    if args.out.exists() or args.local_results_root.exists():
        raise FileExistsError("refusing to overwrite source-component launch outputs")
    if args.seeds is None:
        candidate_payload = json.loads(
            args.candidate_spec_local.read_text(encoding="utf-8")
        )
        args.seeds = [
            int(value) for value in candidate_payload.get("confirmation_seeds", ())
        ]
        if not args.seeds:
            raise ValueError(
                "seeds were not supplied and candidate spec has no confirmation seeds"
            )

    guard_profile = None
    profile_args = (
        args.guard_profile_spec_local,
        args.guard_profile_spec_sha256,
        args.guard_profile_key,
    )
    if any(value is not None for value in profile_args):
        if not all(value is not None for value in profile_args):
            raise ValueError("guard profile specification arguments are incomplete")
        if _sha256(args.guard_profile_spec_local) != str(
            args.guard_profile_spec_sha256
        ):
            raise ValueError("guard profile specification identity changed")
        profile_payload = json.loads(
            args.guard_profile_spec_local.read_text(encoding="utf-8")
        )
        if profile_payload.get("protocol") != GUARD_PROFILE_PROTOCOL:
            raise ValueError("guard profile protocol changed")
        matches = [
            dict(row)
            for row in profile_payload.get("profiles", ())
            if str(row.get("key")) == str(args.guard_profile_key)
        ]
        if len(matches) != 1:
            raise ValueError("guard profile key is missing or duplicated")
        guard_profile = matches[0]
        frozen_guard = guard_profile.get("guard_config")
        if args.guard_risk_multiplier is not None:
            raise ValueError("do not mix frozen and command-line guard settings")
        if frozen_guard is not None:
            args.guard_risk_multiplier = float(frozen_guard["risk_multiplier"])
            args.guard_min_context_trust = float(
                frozen_guard["min_context_trust"]
            )
            args.guard_margin = float(frozen_guard["margin"])
            args.guard_max_relative_rule_gap = float(
                frozen_guard["max_relative_rule_gap"]
            )

    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    snapshot_root = Path(str(stage["snapshot_root"]))
    candidate_spec = _snapshot_path(snapshot_root, args.candidate_spec_local)
    protocol = _snapshot_path(snapshot_root, args.protocol_local)
    external_manifest = _snapshot_path(snapshot_root, args.external_manifest_local)
    if _sha256(args.candidate_spec_local) != str(args.candidate_spec_sha256):
        raise ValueError("local candidate specification identity changed")
    if _sha256(args.protocol_local) != str(args.protocol_sha256):
        raise ValueError("local full-budget protocol identity changed")
    if _sha256(args.external_manifest_local) != str(
        args.external_manifest_sha256
    ):
        raise ValueError("local external manifest identity changed")
    candidates = load_candidate_specs(args.candidate_spec_local)
    candidate_keys = tuple(str(row["key"]) for row in candidates)
    task_counts = {
        int(index): len(
            shard_identities(
                args.seeds,
                candidate_keys,
                shard_index=int(index),
                shard_count=int(args.shard_count),
                scenarios=args.scenarios,
            )
        )
        for index in args.shard_indices
    }
    if any(value < 1 for value in task_counts.values()):
        raise ValueError("a requested source-component shard is empty")

    scheduler = _load_scheduler(args.scheduler)
    count_command = (
        f"if test -e {shlex.quote(str(args.remote_results_root))}; "
        f"then find {shlex.quote(str(args.remote_results_root))} -type f | wc -l; "
        "else printf '0\\n'; fi"
    )
    code, stdout, stderr = scheduler.run_on(
        args.nodes[0], count_command, timeout=120, check=False
    )
    if int(code) != 0 or int(str(stdout).strip().splitlines()[-1]) != 0:
        raise RuntimeError(f"remote result root is not empty: {stderr or stdout}")

    commands = {
        node: _command(
            snapshot_root=snapshot_root,
            candidate_spec=candidate_spec,
            protocol=protocol,
            external_manifest=external_manifest,
            node_shard_index=int(index),
            args=args,
        )
        for node, index in zip(args.nodes, args.shard_indices)
    }
    with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
        futures = {
            node: pool.submit(
                _run_node,
                scheduler,
                node,
                commands[node],
                int(args.timeout_sec),
            )
            for node in args.nodes
        }
        node_results = [futures[node].result() for node in args.nodes]
    if any(row["returncode"] != 0 for row in node_results):
        raise RuntimeError(f"source-component node failure: {node_results}")

    expected_files = 2 * sum(task_counts.values()) + len(args.nodes)
    post_command = (
        f"find {shlex.quote(str(args.remote_results_root))} -type f | wc -l"
    )
    code, stdout, stderr = scheduler.run_on(
        args.nodes[0], post_command, timeout=120, check=False
    )
    remote_file_count = int(str(stdout).strip().splitlines()[-1])
    if int(code) != 0 or remote_file_count != expected_files:
        raise RuntimeError(
            f"remote source-component file count mismatch: "
            f"{remote_file_count} != {expected_files}; {stderr}"
        )

    local_shards = args.local_results_root / "_shards"
    local_shards.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node(args.nodes[0]),
            f"{scheduler._ssh_target_for_node(args.nodes[0])}:"
            f"{args.remote_results_root}/_shards/",
            f"{local_shards}/",
        ],
        check=True,
    )
    summaries = []
    for index in args.shard_indices:
        path = local_shards / f"shard_{index}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("passed") or int(payload["task_count"]) != task_counts[index]:
            raise RuntimeError(f"source-component shard summary failed: {path}")
        summaries.append(
            {
                "shard_index": int(index),
                "task_count": int(payload["task_count"]),
                "summary_sha256": _sha256(path),
            }
        )
    launch = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": str(stage["snapshot_sha256"]),
        "source_tree_sha256": str(stage["source_tree_sha256"]),
        "candidate_spec_sha256": str(args.candidate_spec_sha256),
        "protocol_sha256": str(args.protocol_sha256),
        "external_manifest_sha256": str(args.external_manifest_sha256),
        "seeds": [int(value) for value in args.seeds],
        "scenarios": [str(value) for value in args.scenarios],
        "guard_config": (
            {
                "enabled": True,
                "risk_multiplier": float(args.guard_risk_multiplier),
                "min_context_trust": float(args.guard_min_context_trust),
                "margin": float(args.guard_margin),
                "max_relative_rule_gap": float(args.guard_max_relative_rule_gap),
            }
            if args.guard_risk_multiplier is not None
            else None
        ),
        "guard_profile": (
            {
                "key": str(args.guard_profile_key),
                "spec_path": str(args.guard_profile_spec_local.resolve()),
                "spec_sha256": str(args.guard_profile_spec_sha256),
            }
            if guard_profile is not None
            else None
        ),
        "artifact_contract": {
            "candidate_spec_local": str(args.candidate_spec_local.resolve()),
            "candidate_spec_remote": str(candidate_spec),
            "protocol_local": str(args.protocol_local.resolve()),
            "protocol_remote": str(protocol),
            "external_manifest_local": str(args.external_manifest_local.resolve()),
            "external_manifest_remote": str(external_manifest),
            "los_angeles_main_model": {
                "path": str(args.los_angeles_main_model),
                "sha256": str(args.los_angeles_main_model_sha256),
            },
            "los_angeles_component_bundle": {
                "path": str(args.los_angeles_component_bundle),
                "sha256": str(args.los_angeles_component_bundle_sha256),
            },
            "los_angeles_strict_target_only_model": (
                {
                    "path": str(args.los_angeles_strict_target_only_model),
                    "sha256": str(
                        args.los_angeles_strict_target_only_model_sha256
                    ),
                }
                if args.los_angeles_strict_target_only_model is not None
                else None
            ),
            "jinan_main_model": {
                "path": str(args.jinan_main_model),
                "sha256": str(args.jinan_main_model_sha256),
            },
            "jinan_component_bundle": {
                "path": str(args.jinan_component_bundle),
                "sha256": str(args.jinan_component_bundle_sha256),
            },
            "jinan_strict_target_only_model": (
                {
                    "path": str(args.jinan_strict_target_only_model),
                    "sha256": str(args.jinan_strict_target_only_model_sha256),
                }
                if args.jinan_strict_target_only_model is not None
                else None
            ),
            "offline_authorization": {
                "path": str(args.offline_authorization),
                "sha256": str(args.offline_authorization_sha256),
            },
            "joint_audit": {
                "path": str(args.joint_audit),
                "sha256": str(args.joint_audit_sha256),
            },
            "freeze_root": str(args.freeze_root),
            "conversion_root": str(args.conversion_root),
            "conversion_manifest": {
                "path": str(args.conversion_manifest),
                "sha256": str(args.conversion_manifest_sha256),
            },
            "conversion_tree_sha256": str(args.conversion_tree_sha256),
        },
        "nodes": list(args.nodes),
        "shard_indices": list(args.shard_indices),
        "shard_count": int(args.shard_count),
        "task_count": int(sum(task_counts.values())),
        "workers_per_node": int(args.workers_per_node),
        "remote_file_count": remote_file_count,
        "remote_results_root": str(args.remote_results_root),
        "local_materialization": "requested_shard_summary_json_files_only",
        "node_results": node_results,
        "summaries": summaries,
    }
    _atomic_json(args.out, launch)
    print(json.dumps(launch, sort_keys=True))
    print(f"Results saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
