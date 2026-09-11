#!/usr/bin/env python3
"""Submit seven B100 V153A target-calibrated hierarchical-prior tasks."""

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
from cf_h2o.eval.traffic_signal_hierarchical_mechanism_prior import (
    AGGREGATE_PROTOCOL as V152A_AGGREGATE_PROTOCOL,
    PRIOR_PROTOCOL,
    PRIOR_STRENGTHS,
    V152A_REJECT_DECISION,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    TARGET_BUDGET,
)
from cf_h2o.eval.traffic_signal_target_calibrated_hierarchical_prior import (
    FEATURE_BINDING,
    FOLD_COUNT,
    METHOD_PROTOCOL,
    MINIMUM_INTERVENTION_GROUPS,
    MINIMUM_NONDEGRADING_FOLDS,
    SELECTOR_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
)
from scripts.cluster.launch_tsc_rigid_mechanism_prior_integration import (
    build_specs as _build_specs,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import (
    select_target_specs,
)


LAUNCH_PROTOCOL = "tsc-v153a-target-calibrated-hierarchical-prior-launch-v3"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v153a_target_calibrated_hierarchical_prior.json"
)
MODULE = "cf_h2o.eval.traffic_signal_target_calibrated_hierarchical_prior"
SIGNATURE_PREFIX = "CFCMT/v153a/target-calibrated-hierarchical-prior-v3"
RESOURCE_FAMILY = "CFCMT-v153a-target-calibrated-hierarchical-prior"
RAM_MB = 32_768
FIT_WORKERS = 6


def _validate_config(config: Mapping[str, Any]) -> None:
    selector = dict(config.get("selector", {}))
    gate = dict(config.get("development_gate", {}))
    if (
        config.get("protocol")
        != "tsc-v153a-target-calibrated-hierarchical-prior-v3"
        or config.get("feature_binding") != FEATURE_BINDING
        or config.get("redeveloped_after") != V152A_AGGREGATE_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("target_selector_label_group_count", -1))
        != TARGET_BUDGET
        or int(config.get("target_evaluation_label_count", -1)) != 0
        or config.get("prior_protocol") != PRIOR_PROTOCOL
        or tuple(float(value) for value in config.get("prior_strength_grid", ()))
        != PRIOR_STRENGTHS
        or int(config.get("candidate_mechanism_block_count", -1)) != 6
        or selector.get("protocol") != SELECTOR_PROTOCOL
        or int(selector.get("fold_count", -1)) != FOLD_COUNT
        or int(selector.get("outer_training_group_count", -1)) != 80
        or int(selector.get("inner_training_group_count", -1)) != 60
        or selector.get("inner_fit_excludes_outer_heldout") is not True
        or selector.get("selection_rule")
        != "maximize_worst_comparator_mean_gain"
        or float(selector.get("minimum_mean_gain_each_comparison", -1.0))
        != MINIMUM_OOF_GAIN
        or int(
            selector.get(
                "minimum_nondegrading_folds_each_comparison", -1
            )
        )
        != MINIMUM_NONDEGRADING_FOLDS
        or int(selector.get("minimum_intervention_groups", -1))
        != MINIMUM_INTERVENTION_GROUPS
        or selector.get("target_evaluation_excluded") is not True
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or config.get("matched_placebo_pipeline")
        != "same_candidate_same_target_folds_with_source_labels_permuted_by_stable_city_identity"
        or config.get("zero_centred_control")
        != "same_candidate_same_target_folds_and_source_derived_precision_with_zero_prior_center"
        or int(gate.get("minimum_admitted_target_cities", -1)) != 2
        or gate.get("require_no_target_city_regression") is not True
        or gate.get("require_every_admitted_target_improve_rigid") is not True
        or gate.get("require_every_admitted_target_beat_matched_placebo") is not True
        or gate.get("require_every_admitted_target_beat_zero_centred_prior")
        is not True
        or gate.get("require_exact_fallback_for_rejected_targets") is not True
    ):
        raise ValueError("V153A frozen configuration changed")


def build_specs(**kwargs: Any) -> list[dict[str, Any]]:
    extra_target_args = tuple(kwargs.pop("extra_target_args", ()))
    specs = _build_specs(
        **kwargs,
        module=MODULE,
        signature_prefix=SIGNATURE_PREFIX,
        version_label="V153A-v3",
        resource_family=RESOURCE_FAMILY,
        extra_target_args=(
            "--fit-workers",
            str(FIT_WORKERS),
            *extra_target_args,
        ),
    )
    for spec in specs:
        spec["ram_mb"] = RAM_MB
        spec["ram_resource_family"] = RESOURCE_FAMILY
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--v152a-rejection-local", type=Path, required=True)
    parser.add_argument("--v152a-rejection-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument(
        "--probe-node",
        default="node002",
        help="Connected node used only to verify shared remote inputs.",
    )
    parser.add_argument(
        "--target-city",
        action="append",
        choices=EXPECTED_CITY_GROUPS,
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V153A launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V153A cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))

    authorization = _read_json(args.authorization_result_local)
    rejection = _read_json(args.v152a_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V153A requires passing V150C authorization")
    if (
        rejection.get("protocol") != V152A_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != V152A_REJECT_DECISION
    ):
        raise ValueError("V153A requires the frozen V152A rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V153A requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V153A source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    target_cities = tuple(args.target_city or EXPECTED_CITY_GROUPS)
    local_inputs = (
        args.authorization_result_local,
        args.v152a_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v152a_rejection_remote,
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
    ]
    code, _, stderr = scheduler.run_on(
        str(args.probe_node), " && ".join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V153A preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        str(args.probe_node),
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(stdout).splitlines() if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            f"remote V153A evidence identities changed: {remote_hashes}; "
            f"{hash_stderr}"
        )

    specs = select_target_specs(
        build_specs(
            snapshot_root=snapshot_root,
            authorization_result_remote=args.authorization_result_remote,
            authorization_sha256=expected_hashes[0],
            conversion_root=args.conversion_root,
            fit_result_remote=args.fit_result_remote,
            fit_result_sha256=expected_hashes[2],
            source_cache_root=args.source_cache_root,
            source_cache_audit_remote=args.source_cache_audit_remote,
            source_cache_audit_sha256=expected_hashes[3],
            remote_output_root=args.remote_output_root,
            local_output_root=args.local_output_root,
            cache_workers=args.cache_workers,
            extra_target_args=(
                "--v152a-rejection",
                str(args.v152a_rejection_remote),
                "--v152a-rejection-sha256",
                expected_hashes[1],
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
            "CFCMT-v153a-target-calibrated-hierarchical-prior-v3",
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
        "v152a_rejection_sha256": expected_hashes[1],
        "fit_result_sha256": expected_hashes[2],
        "source_cache_audit_sha256": expected_hashes[3],
        "remote_output_root": str(args.remote_output_root),
        "target_cities": list(target_cities),
        "task_specs": specs,
        "method_protocol": METHOD_PROTOCOL,
        "selector_protocol": SELECTOR_PROTOCOL,
        "feature_binding": FEATURE_BINDING,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V153A target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
