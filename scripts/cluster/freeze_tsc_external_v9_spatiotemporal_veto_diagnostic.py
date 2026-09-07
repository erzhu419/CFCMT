#!/usr/bin/env python3
"""Freeze the post-v70 spatiotemporal module diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_spatiotemporal_veto_diagnostic import (  # noqa: E402
    DIAGNOSTIC_PROTOCOL,
    VARIANTS,
    feature_names,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


CACHE_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v51_external_v9_"
    "estimand_aligned_450s_target_veto_cache.json"
)
TRAINING_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v52_external_v9_"
    "demand_conditioned_450s_veto_training.json"
)


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def build_protocol(
    *,
    cache_audit_path: Path,
    v70_result_path: Path,
    v70_joint_audit_path: Path,
    previous_failure_path: Path,
    diagnostic_output: Path,
    successor_output_root: Path,
    future_closed_loop_root: Path,
    future_validation_root: Path,
) -> dict[str, Any]:
    cache_protocol = _read_json(CACHE_PROTOCOL_PATH)
    training_protocol = _read_json(TRAINING_PROTOCOL_PATH)
    cache_audit = _read_json(cache_audit_path)
    v70_result = _read_json(v70_result_path)
    v70_audit = _read_json(v70_joint_audit_path)
    previous_failure = _read_json(previous_failure_path)
    proposal_path = Path(training_protocol["proposal_parent_protocol"]["path"])
    freeze_audit_path = Path(training_protocol["hierarchical_freeze_audit"]["path"])
    if (
        cache_audit.get("status") != "PASS"
        or cache_audit.get("decision")
        != "authorize_demand_conditioned_450s_veto_oof_training_only"
        or cache_audit.get("frozen_protocol_sha256")
        != _sha256(CACHE_PROTOCOL_PATH)
        or v70_result.get("status") != "FAIL"
        or v70_result.get("active_cities") != []
        or v70_result.get("decision")
        != "retain_phase_pressure_and_prohibit_v70_closed_loop_development"
        or v70_audit.get("status") != "FAIL"
        or v70_audit.get("decision") != "reject"
        or v70_audit.get("active_cities") != []
        or v70_audit.get("training_result_sha256") != _sha256(v70_result_path)
        or _sha256(proposal_path)
        != training_protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != training_protocol["hierarchical_freeze_audit"]["sha256"]
        or previous_failure.get("protocol")
        != "tsc-v72r68-spatiotemporal-diagnostic-pre-result-failure-v1"
        or previous_failure.get("status") != "FAIL"
        or previous_failure.get("diagnosis", {}).get(
            "scientific_outcome_available"
        )
        is not False
        or previous_failure.get("diagnostic_protocol_sha256")
        != "7305ed72f845b4840ec423392006e0e470dd47fe3e6d11c905dd1b7d8fb89300"
    ):
        raise ValueError("v72 diagnostic parent evidence changed")
    future_counts = {
        "diagnostic_output": int(Path(diagnostic_output).is_file()),
        "successor_output_root": _file_count(successor_output_root),
        "future_closed_loop_root": _file_count(future_closed_loop_root),
        "future_validation_root": _file_count(future_validation_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v72 future evidence already exists: {future_counts}")
    sources = (
        "cf_h2o/eval/traffic_signal_external_spatiotemporal_veto_diagnostic.py",
        "cf_h2o/traffic_signal/demand_timeline_context.py",
        "scripts/cluster/run_tsc_external_spatiotemporal_veto_diagnostic.py",
    )
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v70_negative_pre_spatiotemporal_diagnostic_outcome_freeze",
        "cache_protocol": {
            "path": str(CACHE_PROTOCOL_PATH),
            "sha256": _sha256(CACHE_PROTOCOL_PATH),
        },
        "cache_audit": {
            "path": str(cache_audit_path),
            "sha256": _sha256(cache_audit_path),
            "cache_sha256": cache_audit["cache_sha256"],
        },
        "v70_negative_result": {
            "path": str(v70_result_path),
            "sha256": _sha256(v70_result_path),
            "decision": v70_result["decision"],
        },
        "v70_negative_joint_audit": {
            "path": str(v70_joint_audit_path),
            "sha256": _sha256(v70_joint_audit_path),
            "decision": v70_audit["decision"],
        },
        "previous_pre_result_failure": {
            "path": str(previous_failure_path),
            "sha256": _sha256(previous_failure_path),
            "decision": previous_failure["decision"],
        },
        "proposal_parent_protocol": {
            "path": str(proposal_path),
            "sha256": _sha256(proposal_path),
        },
        "hierarchical_freeze_audit": {
            "path": str(freeze_audit_path),
            "sha256": _sha256(freeze_audit_path),
        },
        "runtime_executable_sources": {
            path: _sha256(Path(path)) for path in sources
        },
        "diagnostic": {
            "classification": "post_v70_disclosed_adaptation_seed_module_diagnostic",
            "seeds": cache_protocol["long_horizon_collection"]["seeds"],
            "city_scenarios": cache_protocol["city_scenarios"],
            "cross_fitting": "leave_one_simulator_seed_out",
            "target": "450_second_group_normalized_candidate_minus_phase_cost",
            "variants": {
                variant: {
                    "feature_names": list(feature_names(variant)),
                    "feature_count": len(feature_names(variant)),
                }
                for variant in VARIANTS
            },
            "acceptance_quantiles": training_protocol["training"][
                "acceptance_quantiles"
            ],
            "model": training_protocol["training"]["model"],
            "bootstrap": training_protocol["training"]["bootstrap"],
            "selection_rule": training_protocol["training"]["selection_rule"],
            "static_global_must_exactly_reproduce_v70_grid": True,
            "counterfactual_label_oracle": (
                "diagnostic_upper_bound_only_never_deployable"
            ),
            "module_advancement_rule": (
                "a_successor_can_be_frozen_only_if_at_least_one_non_oracle_variant_"
                "passes_the_original_v70_selection_rule_and_static_global_reproduces_v70"
            ),
        },
        "future_file_counts_at_freeze": future_counts,
        "validation": training_protocol["validation"],
        "prospective_confirmation": training_protocol[
            "prospective_confirmation"
        ],
        "claim_boundary": (
            "Adaptation-seed module diagnostic after a disclosed negative result. "
            "No model is frozen for closed-loop use, and sealed seeds remain untouched."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--v70-result", type=Path, required=True)
    parser.add_argument("--v70-joint-audit", type=Path, required=True)
    parser.add_argument("--previous-failure", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--successor-output-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--future-validation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v72 diagnostic protocol: {args.out}")
    payload = build_protocol(
        cache_audit_path=args.cache_audit,
        v70_result_path=args.v70_result,
        v70_joint_audit_path=args.v70_joint_audit,
        previous_failure_path=args.previous_failure,
        diagnostic_output=args.diagnostic_output,
        successor_output_root=args.successor_output_root,
        future_closed_loop_root=args.future_closed_loop_root,
        future_validation_root=args.future_validation_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "variants": list(payload["diagnostic"]["variants"]),
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
