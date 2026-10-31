#!/usr/bin/env python3
"""Submit seven V150O dense pressure-pairwise source-utility targets."""

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

from cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility import (
    METHOD_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    V150N_AGGREGATE_PROTOCOL,
    V150N_REJECT_DECISION,
)
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
    V150J_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import DEFAULT_SCHEDULER
from scripts.cluster.launch_tsc_pressure_aligned_source_utility import (
    build_specs as _build_specs,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import (
    select_target_specs,
)


LAUNCH_PROTOCOL = "tsc-v150o-dense-pressure-pairwise-source-utility-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150o_dense_pressure_pairwise_source_utility.json"
)
MODULE = "cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility"
SIGNATURE_PREFIX = "CFCMT/v150o/dense-pressure-pairwise-source-utility-v1"
RESOURCE_FAMILY = "CFCMT-v150o-dense-pressure-pairwise-source-utility"


def _validate_config(config: Mapping[str, Any]) -> None:
    gate = dict(config.get("development_gate", {}))
    utility = dict(config.get("utility_gate", {}))
    if (
        config.get("protocol")
        != "tsc-v150o-dense-pressure-pairwise-source-utility-v1"
        or config.get("redeveloped_after") != V150N_AGGREGATE_PROTOCOL
        or config.get("offline_parent")
        != "tsc-v150k-state-conditioned-source-utility-aggregate-v1"
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
        or config.get("action_pair")
        != "phase_pressure_reference_vs_rigid_target_only"
        or config.get("utility_record_protocol")
        != DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL
        or config.get("selector")
        != "nested_dense_fixed_pair_source_mechanism_utility"
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
        or float(config.get("minimum_source_blind_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or int(config.get("minimum_nondegrading_outer_folds", -1))
        != MINIMUM_NONDEGRADING_FOLDS
        or int(config.get("minimum_crossfitted_intervention_groups", -1))
        != MINIMUM_INTERVENTION_GROUPS
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or config.get("matched_placebo_pipeline")
        != "separately_nested_same_capacity_dense_pairwise_utility_gate"
        or config.get("source_blind_pipeline")
        != "same_target_labels_state_features_and_gate_without_source_evidence"
        or config.get("inference_unit") != "city"
        or int(gate.get("minimum_identity_verified_source_cities", -1)) != 2
        or gate.get("require_no_city_regression") is not True
        or gate.get("require_every_admitted_city_improve_rigid") is not True
        or gate.get("require_every_admitted_city_beat_matched_placebo") is not True
        or gate.get("require_every_admitted_city_beat_source_blind") is not True
        or gate.get("require_exact_fallback_for_rejected_cities") is not True
    ):
        raise ValueError("V150O frozen configuration changed")


def build_specs(**kwargs: Any) -> list[dict[str, Any]]:
    return _build_specs(
        **kwargs,
        module=MODULE,
        signature_prefix=SIGNATURE_PREFIX,
        version_label="V150O",
        resource_family=RESOURCE_FAMILY,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--v150j-rejection-local", type=Path, required=True)
    parser.add_argument("--v150j-rejection-remote", type=Path, required=True)
    parser.add_argument("--v150n-rejection-local", type=Path, required=True)
    parser.add_argument("--v150n-rejection-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--runtime-model-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--target-city", action="append", choices=EXPECTED_CITY_GROUPS)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150O launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V150O cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))

    authorization = _read_json(args.authorization_result_local)
    v150j = _read_json(args.v150j_rejection_local)
    v150n = _read_json(args.v150n_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V150O requires passing V150C authorization")
    if (
        v150j.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or v150j.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V150O requires the frozen V150J rejection")
    if (
        v150n.get("protocol") != V150N_AGGREGATE_PROTOCOL
        or v150n.get("development_gate", {}).get("passed") is not False
        or v150n.get("development_gate", {}).get("decision")
        != V150N_REJECT_DECISION
    ):
        raise ValueError("V150O requires the frozen V150N rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V150O requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V150O source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    target_cities = tuple(args.target_city or EXPECTED_CITY_GROUPS)
    local_inputs = (
        args.authorization_result_local,
        args.v150j_rejection_local,
        args.v150n_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v150j_rejection_remote,
        args.v150n_rejection_remote,
        args.fit_result_remote,
        args.source_cache_audit_remote,
    )
    expected_hashes = [_sha256(path) for path in local_inputs]
    scheduler = _load_scheduler(args.scheduler)
    checks = [
        shlex.join(["test", "-d", str(args.source_cache_root)]),
        shlex.join(["test", "-d", str(args.conversion_root)]),
        *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        *(
            shlex.join(["test", "!", "-e", str(args.remote_output_root / city)])
            for city in target_cities
        ),
        *(
            shlex.join(["test", "!", "-e", str(args.runtime_model_root / city)])
            for city in target_cities
        ),
    ]
    code, _, stderr = scheduler.run_on(
        "node001",
        " && ".join(checks),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V150O preflight failed: {stderr}")
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
            f"remote V150O evidence identities changed: {remote_hashes}; "
            f"{hash_stderr}"
        )
    specs = select_target_specs(
        build_specs(
            snapshot_root=snapshot_root,
            authorization_result_remote=args.authorization_result_remote,
            authorization_sha256=expected_hashes[0],
            v150j_rejection_remote=args.v150j_rejection_remote,
            v150j_rejection_sha256=expected_hashes[1],
            conversion_root=args.conversion_root,
            fit_result_remote=args.fit_result_remote,
            fit_result_sha256=expected_hashes[3],
            source_cache_root=args.source_cache_root,
            source_cache_audit_remote=args.source_cache_audit_remote,
            source_cache_audit_sha256=expected_hashes[4],
            remote_output_root=args.remote_output_root,
            local_output_root=args.local_output_root,
            cache_workers=args.cache_workers,
            runtime_model_root=args.runtime_model_root,
            extra_target_args=(
                "--v150n-rejection",
                str(args.v150n_rejection_remote),
                "--v150n-rejection-sha256",
                expected_hashes[2],
            ),
        ),
        target_cities,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150o-dense-pressure-pairwise-source-utility-v1",
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
        "v150n_rejection_sha256": expected_hashes[2],
        "fit_result_sha256": expected_hashes[3],
        "source_cache_audit_sha256": expected_hashes[4],
        "remote_output_root": str(args.remote_output_root),
        "runtime_model_root": str(args.runtime_model_root),
        "target_cities": list(target_cities),
        "task_specs": specs,
        "method_protocol": METHOD_PROTOCOL,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150O target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
