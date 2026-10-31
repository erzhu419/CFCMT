#!/usr/bin/env python3
"""Submit seven leave-target-city-out V151A meta-utility tasks."""

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

from cf_h2o.eval.traffic_signal_cross_city_meta_source_utility import (
    META_MINIMUM_INTERVENTION_GROUPS,
    META_NONDEGRADING_CITY_COUNT,
    METHOD_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    V150O_AGGREGATE_PROTOCOL,
    V150O_REJECT_DECISION,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    GATE_CONFIG,
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


LAUNCH_PROTOCOL = "tsc-v151a-cross-city-meta-source-utility-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v151a_cross_city_meta_source_utility.json"
)
MODULE = "cf_h2o.eval.traffic_signal_cross_city_meta_source_utility"
SIGNATURE_PREFIX = "CFCMT/v151a/cross-city-meta-source-utility-v1"
RESOURCE_FAMILY = "CFCMT-v151a-cross-city-meta-source-utility"
RAM_MB = 32_768


def _validate_config(config: Mapping[str, Any]) -> None:
    utility = dict(config.get("utility_gate", {}))
    meta = dict(config.get("meta_training", {}))
    gate = dict(config.get("development_gate", {}))
    if (
        config.get("protocol") != "tsc-v151a-cross-city-meta-source-utility-v1"
        or config.get("redeveloped_after") != V150O_AGGREGATE_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("target_utility_label_count", -1)) != 0
        or float(config.get("source_prior_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or config.get("action_pair")
        != "phase_pressure_reference_vs_rigid_target_only"
        or config.get("utility_record_protocol")
        != DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL
        or utility.get("protocol") != UTILITY_GATE_PROTOCOL
        or float(utility.get("ridge_alpha", -1.0)) != GATE_CONFIG.ridge_alpha
        or float(utility.get("error_quantile", -1.0))
        != GATE_CONFIG.error_quantile
        or int(utility.get("minimum_records", -1)) != GATE_CONFIG.minimum_records
        or float(utility.get("minimum_predicted_gain", -1.0))
        != GATE_CONFIG.minimum_predicted_gain
        or utility.get("weighting")
        != "equal_pseudo_target_city_then_equal_action_group"
        or utility.get("source_identity_feature") is not False
        or int(meta.get("pseudo_target_city_count", -1)) != 6
        or int(meta.get("candidate_source_city_count_per_pseudo_target", -1)) != 5
        or meta.get("target_excluded_from_utility_labels") is not True
        or meta.get("target_excluded_from_candidate_sources") is not True
        or meta.get("target_excluded_from_feature_scaling_statistics") is not True
        or meta.get("leave_one_pseudo_target_city_out_calibration") is not True
        or float(meta.get("minimum_mean_gain_each_comparison", -1.0))
        != MINIMUM_OOF_GAIN
        or int(
            meta.get(
                "minimum_nondegrading_pseudo_target_cities_each_comparison", -1
            )
        )
        != META_NONDEGRADING_CITY_COUNT
        or int(meta.get("minimum_intervention_groups", -1))
        != META_MINIMUM_INTERVENTION_GROUPS
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or int(gate.get("minimum_admitted_target_cities", -1)) != 2
        or gate.get("require_no_target_city_regression") is not True
        or gate.get("require_every_admitted_target_improve_rigid") is not True
        or gate.get("require_every_admitted_target_beat_matched_placebo") is not True
        or gate.get("require_every_admitted_target_beat_source_blind") is not True
        or gate.get("require_exact_fallback_for_rejected_targets") is not True
    ):
        raise ValueError("V151A frozen configuration changed")


def build_specs(**kwargs: Any) -> list[dict[str, Any]]:
    specs = _build_specs(
        **kwargs,
        module=MODULE,
        signature_prefix=SIGNATURE_PREFIX,
        version_label="V151A",
        resource_family=RESOURCE_FAMILY,
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
    parser.add_argument("--v150j-rejection-local", type=Path, required=True)
    parser.add_argument("--v150j-rejection-remote", type=Path, required=True)
    parser.add_argument("--v150o-rejection-local", type=Path, required=True)
    parser.add_argument("--v150o-rejection-remote", type=Path, required=True)
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
        default="node001",
        help="Connected node used only to verify shared remote inputs.",
    )
    parser.add_argument("--target-city", action="append", choices=EXPECTED_CITY_GROUPS)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V151A launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V151A cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))

    authorization = _read_json(args.authorization_result_local)
    v150j = _read_json(args.v150j_rejection_local)
    v150o = _read_json(args.v150o_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V151A requires passing V150C authorization")
    if (
        v150j.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or v150j.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V151A requires the frozen V150J rejection")
    if (
        v150o.get("protocol") != V150O_AGGREGATE_PROTOCOL
        or v150o.get("development_gate", {}).get("passed") is not False
        or v150o.get("development_gate", {}).get("decision")
        != V150O_REJECT_DECISION
    ):
        raise ValueError("V151A requires the frozen V150O rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V151A requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V151A source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    target_cities = tuple(args.target_city or EXPECTED_CITY_GROUPS)
    local_inputs = (
        args.authorization_result_local,
        args.v150j_rejection_local,
        args.v150o_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v150j_rejection_remote,
        args.v150o_rejection_remote,
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
        raise RuntimeError(f"remote V151A preflight failed: {stderr}")
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
            f"remote V151A evidence identities changed: {remote_hashes}; "
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
            extra_target_args=(
                "--v150o-rejection",
                str(args.v150o_rejection_remote),
                "--v150o-rejection-sha256",
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
            "CFCMT-v151a-cross-city-meta-source-utility-v1",
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
        "v150o_rejection_sha256": expected_hashes[2],
        "fit_result_sha256": expected_hashes[3],
        "source_cache_audit_sha256": expected_hashes[4],
        "remote_output_root": str(args.remote_output_root),
        "target_cities": list(target_cities),
        "task_specs": specs,
        "method_protocol": METHOD_PROTOCOL,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V151A target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
