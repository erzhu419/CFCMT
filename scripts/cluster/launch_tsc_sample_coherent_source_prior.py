#!/usr/bin/env python3
"""Submit the 21-target V150I sample-coherent source-prior experiment."""

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
    SOURCE_PRIOR_STRENGTH,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_sample_coherent_source_prior import (
    REFERENCE_BUDGET,
    TARGET_BUDGETS,
    prior_strength_for_budget,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    AGGREGATE_PROTOCOL as V150H_AGGREGATE_PROTOCOL,
    COMMON_EVALUATION_RESERVE_BUDGET,
    PRIMARY_BUDGET,
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


LAUNCH_PROTOCOL = "tsc-v150i-sample-coherent-source-prior-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150i_sample_coherent_source_prior.json"
)
MODULE = "cf_h2o.eval.traffic_signal_sample_coherent_source_prior"
SIGNATURE_PREFIX = "CFCMT/v150i/sample-coherent-source-prior-v1"
RESOURCE_FAMILY = "CFCMT-v150i-sample-coherent-source-prior"
RAM_MB = 8_192


def _validate_config(config: Mapping[str, Any]) -> None:
    schedule = dict(config.get("source_prior_schedule", {}))
    gate = dict(config.get("development_gate", {}))
    observed_schedule = {
        str(budget): float(schedule.get("strength_by_budget", {}).get(str(budget), -1.0))
        for budget in TARGET_BUDGETS
    }
    expected_schedule = {
        str(budget): prior_strength_for_budget(budget) for budget in TARGET_BUDGETS
    }
    if (
        config.get("protocol") != "tsc-v150i-sample-coherent-source-prior-v1"
        or config.get("redeveloped_after") != V150H_AGGREGATE_PROTOCOL
        or config.get("authorization_protocol") != AUTHORIZATION_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or tuple(int(value) for value in config.get("target_budgets", ()))
        != TARGET_BUDGETS
        or int(config.get("primary_budget", -1)) != PRIMARY_BUDGET
        or int(config.get("common_evaluation_reserve_budget", -1))
        != COMMON_EVALUATION_RESERVE_BUDGET
        or int(config.get("adaptation_fold_count", -1)) != FOLD_COUNT
        or int(config.get("candidate_count_per_target", -1)) != 36
        or config.get("selector")
        != "crossfit_complete_source_mechanism_selection"
        or schedule.get("protocol") != "inverse-target-group-count-v1"
        or int(schedule.get("reference_budget", -1)) != REFERENCE_BUDGET
        or float(schedule.get("reference_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or observed_schedule != expected_schedule
        or float(config.get("minimum_target_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or float(config.get("minimum_identity_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or int(config.get("minimum_nondegrading_heldout_folds", -1)) != 4
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or config.get("matched_placebo_follows_source_admission") is not True
        or config.get("inference_unit") != "city"
        or int(gate.get("primary_budget", -1)) != PRIMARY_BUDGET
        or int(gate.get("minimum_identity_verified_source_cities", -1)) != 2
        or gate.get("require_no_city_regression") is not True
        or gate.get("require_every_admitted_city_improve_rigid") is not True
        or gate.get("require_every_admitted_city_beat_matched_placebo") is not True
        or gate.get("require_exact_fallback_for_rejected_cities") is not True
    ):
        raise ValueError("V150I frozen configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    authorization_result_remote: Path,
    authorization_sha256: str,
    v150h_rejection_remote: Path,
    v150h_rejection_sha256: str,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    source_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
    cache_workers: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for budget in TARGET_BUDGETS:
        budget_specs = _build_rigid_specs(
            snapshot_root=snapshot_root,
            authorization_result_remote=authorization_result_remote,
            authorization_sha256=authorization_sha256,
            conversion_root=conversion_root,
            fit_result_remote=fit_result_remote,
            fit_result_sha256=fit_result_sha256,
            source_cache_root=source_cache_root,
            source_cache_audit_remote=source_cache_audit_remote,
            source_cache_audit_sha256=source_cache_audit_sha256,
            remote_output_root=remote_output_root / f"b{budget}",
            local_output_root=local_output_root / f"b{budget}",
            cache_workers=cache_workers,
            module=MODULE,
            signature_prefix=f"{SIGNATURE_PREFIX}/b{budget}",
            version_label=f"V150I-B{budget}",
            resource_family=RESOURCE_FAMILY,
            extra_target_args=(
                "--target-budget",
                str(budget),
                "--v150h-rejection",
                str(v150h_rejection_remote),
                "--v150h-rejection-sha256",
                str(v150h_rejection_sha256),
            ),
        )
        for spec in budget_specs:
            spec["ram_mb"] = RAM_MB
        specs.extend(budget_specs)
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--v150h-rejection-local", type=Path, required=True)
    parser.add_argument("--v150h-rejection-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150I launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V150I cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))

    authorization = _read_json(args.authorization_result_local)
    rejection = _read_json(args.v150h_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V150I requires passing V150C authorization")
    if (
        rejection.get("protocol") != V150H_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V150I requires the frozen V150H rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V150I requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V150I source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    local_inputs = (
        args.authorization_result_local,
        args.v150h_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v150h_rejection_remote,
        args.fit_result_remote,
        args.source_cache_audit_remote,
    )
    expected_hashes = [_sha256(path) for path in local_inputs]
    scheduler = _load_scheduler(args.scheduler)
    remote_city_roots = [
        args.remote_output_root / f"b{budget}" / city
        for budget in TARGET_BUDGETS
        for city in EXPECTED_CITY_GROUPS
    ]
    checks = [
        shlex.join(["test", "-d", str(args.source_cache_root)]),
        shlex.join(["test", "-d", str(args.conversion_root)]),
        *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_city_roots),
    ]
    code, _, stderr = scheduler.run_on(
        "node001", " && ".join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V150I preflight failed: {stderr}")
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
            f"remote V150I evidence identities changed: {remote_hashes}; {hash_stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        authorization_result_remote=args.authorization_result_remote,
        authorization_sha256=expected_hashes[0],
        v150h_rejection_remote=args.v150h_rejection_remote,
        v150h_rejection_sha256=expected_hashes[1],
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=expected_hashes[2],
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        source_cache_audit_sha256=expected_hashes[3],
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        cache_workers=args.cache_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150i-sample-coherent-source-prior-v1",
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
        "v150h_rejection_sha256": expected_hashes[1],
        "fit_result_sha256": expected_hashes[2],
        "source_cache_audit_sha256": expected_hashes[3],
        "remote_output_root": str(args.remote_output_root),
        "task_specs": specs,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150I target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
