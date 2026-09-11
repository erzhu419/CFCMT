#!/usr/bin/env python3
"""Submit seven V150K nested state-utility targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MINIMUM_OOF_GAIN,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    GATE_CONFIG,
    MINIMUM_INTERVENTION_GROUPS,
    MINIMUM_NONDEGRADING_FOLDS,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
    UTILITY_GATE_PROTOCOL,
    V150J_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import DEFAULT_SCHEDULER
from scripts.cluster.launch_tsc_rigid_mechanism_prior_integration import (
    FIT_PROTOCOL_RELATIVE,
    SOURCE_MANIFEST_RELATIVE,
    build_specs as _build_rigid_specs,
)


LAUNCH_PROTOCOL = "tsc-v150k-state-conditioned-source-utility-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150k_state_conditioned_source_utility.json"
)
MODULE = "cf_h2o.eval.traffic_signal_state_conditioned_source_utility"
SIGNATURE_PREFIX = "CFCMT/v150k/state-conditioned-source-utility-v1"
RESOURCE_FAMILY = "CFCMT-v150k-state-conditioned-source-utility"
RAM_MB = 16_384
FIT_WORKERS = 6


def _validate_config(config: Mapping[str, Any]) -> None:
    gate = dict(config.get("development_gate", {}))
    utility = dict(config.get("utility_gate", {}))
    if (
        config.get("protocol")
        != "tsc-v150k-state-conditioned-source-utility-v1"
        or config.get("redeveloped_after") != V150J_AGGREGATE_PROTOCOL
        or config.get("authorization_protocol") != AUTHORIZATION_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("common_evaluation_reserve_budget", -1))
        != TARGET_BUDGET
        or float(config.get("source_prior_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or int(config.get("outer_fold_count", -1)) != FOLD_COUNT
        or int(config.get("inner_fold_count", -1)) != FOLD_COUNT - 1
        or int(config.get("candidate_count_per_target", -1)) != 36
        or config.get("candidate_representation")
        != "single-state-conditioner-v1"
        or config.get("selector")
        != "nested_per_state_source_mechanism_utility"
        or utility.get("protocol") != UTILITY_GATE_PROTOCOL
        or float(utility.get("ridge_alpha", -1.0)) != GATE_CONFIG.ridge_alpha
        or float(utility.get("error_quantile", -1.0))
        != GATE_CONFIG.error_quantile
        or int(utility.get("minimum_records", -1)) != GATE_CONFIG.minimum_records
        or float(utility.get("minimum_predicted_gain", -1.0))
        != GATE_CONFIG.minimum_predicted_gain
        or utility.get("source_identity_feature") is not False
        or float(config.get("minimum_target_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or float(config.get("minimum_identity_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or int(config.get("minimum_nondegrading_outer_folds", -1))
        != MINIMUM_NONDEGRADING_FOLDS
        or int(config.get("minimum_crossfitted_intervention_groups", -1))
        != MINIMUM_INTERVENTION_GROUPS
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or config.get("matched_placebo_pipeline")
        != "separately_nested_same_capacity_utility_gate"
        or config.get("inference_unit") != "city"
        or int(gate.get("minimum_identity_verified_source_cities", -1)) != 2
        or gate.get("require_no_city_regression") is not True
        or gate.get("require_every_admitted_city_improve_rigid") is not True
        or gate.get("require_every_admitted_city_beat_matched_placebo") is not True
        or gate.get("require_exact_fallback_for_rejected_cities") is not True
    ):
        raise ValueError("V150K frozen configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    authorization_result_remote: Path,
    authorization_sha256: str,
    v150j_rejection_remote: Path,
    v150j_rejection_sha256: str,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    source_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
    cache_workers: int,
    runtime_model_root: Path | None = None,
    module: str = MODULE,
    signature_prefix: str = SIGNATURE_PREFIX,
    version_label: str = "V150K",
    resource_family: str = RESOURCE_FAMILY,
    extra_target_args: Sequence[str] = (),
) -> list[dict[str, Any]]:
    specs = _build_rigid_specs(
        snapshot_root=snapshot_root,
        authorization_result_remote=authorization_result_remote,
        authorization_sha256=authorization_sha256,
        conversion_root=conversion_root,
        fit_result_remote=fit_result_remote,
        fit_result_sha256=fit_result_sha256,
        source_cache_root=source_cache_root,
        source_cache_audit_remote=source_cache_audit_remote,
        source_cache_audit_sha256=source_cache_audit_sha256,
        remote_output_root=remote_output_root,
        local_output_root=local_output_root,
        cache_workers=cache_workers,
        module=str(module),
        signature_prefix=str(signature_prefix),
        version_label=str(version_label),
        resource_family=str(resource_family),
        extra_target_args=(
            "--v150j-rejection",
            str(v150j_rejection_remote),
            "--v150j-rejection-sha256",
            str(v150j_rejection_sha256),
            "--fit-workers",
            str(FIT_WORKERS),
            *tuple(str(value) for value in extra_target_args),
        ),
    )
    for spec in specs:
        spec["ram_mb"] = RAM_MB
        spec["ram_resource_family"] = str(resource_family)
        if runtime_model_root is not None:
            city = str(spec["signature"]).rsplit("/", 1)[-1]
            runtime_city = Path(runtime_model_root) / city
            marker = " && printf 'TASK_DONE\\n'"
            command = str(spec["cmd"])
            if not command.endswith(marker):
                raise ValueError("V150K task command completion marker changed")
            command = command[: -len(marker)]
            command += " " + shlex.join(
                ["--runtime-model-out", str(runtime_city / "model.pkl")]
            )
            remote_city = Path(spec["result_dir"])
            spec["cmd"] = (
                shlex.join(["mkdir", "-p", str(remote_city), str(runtime_city)])
                + " && "
                + command.split(" && ", 1)[1]
                + marker
            )
    return specs


def select_target_specs(
    specs: Sequence[Mapping[str, Any]], target_cities: Sequence[str]
) -> list[dict[str, Any]]:
    requested = tuple(dict.fromkeys(str(city) for city in target_cities))
    invalid = sorted(set(requested) - set(EXPECTED_CITY_GROUPS))
    if invalid:
        raise ValueError(f"unknown V150K target cities: {invalid}")
    selected = [
        dict(spec)
        for spec in specs
        if str(spec["signature"]).rsplit("/", 1)[-1] in requested
    ]
    if len(selected) != len(requested):
        raise RuntimeError("V150K target recovery selection is incomplete")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--v150j-rejection-local", type=Path, required=True)
    parser.add_argument("--v150j-rejection-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-model-root",
        type=Path,
        help="remote-only root for per-city frozen runtime model pickles",
    )
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument(
        "--target-city",
        action="append",
        choices=EXPECTED_CITY_GROUPS,
        help="submit only the named failed city; repeat to recover multiple shards",
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150K launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V150K cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))

    authorization = _read_json(args.authorization_result_local)
    rejection = _read_json(args.v150j_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V150K requires passing V150C authorization")
    if (
        rejection.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V150K requires the frozen V150J rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V150K requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V150K source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    target_cities = tuple(args.target_city or EXPECTED_CITY_GROUPS)
    local_inputs = (
        args.authorization_result_local,
        args.v150j_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v150j_rejection_remote,
        args.fit_result_remote,
        args.source_cache_audit_remote,
    )
    expected_hashes = [_sha256(path) for path in local_inputs]
    scheduler = _load_scheduler(args.scheduler)
    remote_city_roots = [args.remote_output_root / city for city in target_cities]
    remote_model_roots = (
        [args.runtime_model_root / city for city in target_cities]
        if args.runtime_model_root is not None
        else []
    )
    checks = [
        shlex.join(["test", "-d", str(args.source_cache_root)]),
        shlex.join(["test", "-d", str(args.conversion_root)]),
        *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_city_roots),
        *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_model_roots),
    ]
    code, _, stderr = scheduler.run_on(
        "node001", " && ".join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V150K preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(stdout).splitlines() if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            f"remote V150K evidence identities changed: {remote_hashes}; {hash_stderr}"
        )
    specs = select_target_specs(build_specs(
        snapshot_root=snapshot_root,
        authorization_result_remote=args.authorization_result_remote,
        authorization_sha256=expected_hashes[0],
        v150j_rejection_remote=args.v150j_rejection_remote,
        v150j_rejection_sha256=expected_hashes[1],
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=expected_hashes[2],
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        source_cache_audit_sha256=expected_hashes[3],
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        cache_workers=args.cache_workers,
        runtime_model_root=args.runtime_model_root,
    ), target_cities)
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150k-state-conditioned-source-utility-v1",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "authorization_result_sha256": expected_hashes[0],
        "v150j_rejection_sha256": expected_hashes[1],
        "fit_result_sha256": expected_hashes[2],
        "source_cache_audit_sha256": expected_hashes[3],
        "remote_output_root": str(args.remote_output_root),
        "runtime_model_root": (
            str(args.runtime_model_root)
            if args.runtime_model_root is not None
            else None
        ),
        "target_cities": list(target_cities),
        "task_specs": specs,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150K target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
