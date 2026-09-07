#!/usr/bin/env python3
"""Freeze the v78 paired closed-loop development protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_topology_veto_closed_loop_development import (  # noqa: E402
    CANDIDATE_KEY,
    PHASE_POLICY,
    PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


V60_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v60_external_v9_topology_veto_freeze.json"
)
V59_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v59_external_v9_"
    "historical_prefix_model_diagnostic.json"
)
PROPOSAL_PARENT_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
ABANDONED_V58_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v58_external_v9_"
    "local_demand_closed_loop_development.json"
)
ABANDONED_V75_ROOT = Path(
    "cf_h2o/results/cluster/tsc_v75r71_external_v9_local_demand_"
    "closed_loop_development_20260810"
)
FAILED_V61_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v61_external_v9_"
    "topology_veto_closed_loop_development.json"
)


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def build_protocol(
    *,
    freeze_result_path: Path,
    joint_audit_path: Path,
    artifact_root: Path,
    development_results_root: Path,
    launch_output: Path,
    development_audit_output: Path,
    validation_root: Path,
    prospective_root: Path,
    failed_preflight_path: Path,
) -> dict[str, Any]:
    v60 = _read_json(V60_PATH)
    v59 = _read_json(V59_PATH)
    proposal_parent = _read_json(PROPOSAL_PARENT_PATH)
    abandoned_v58 = _read_json(ABANDONED_V58_PATH)
    freeze_result = _read_json(freeze_result_path)
    joint_audit = _read_json(joint_audit_path)
    failed_preflight = _read_json(failed_preflight_path)
    cache_protocol_path = Path(v60["cache_protocol"]["path"])
    cache_protocol = _read_json(cache_protocol_path)
    hierarchical_audit_path = Path(v60["hierarchical_freeze_audit"]["path"])
    if (
        freeze_result.get("status") != "PASS"
        or freeze_result.get("decision")
        != "authorize_topology_veto_closed_loop_development_only"
        or freeze_result.get("freeze_protocol_sha256") != _sha256(V60_PATH)
        or joint_audit.get("status") != "PASS"
        or joint_audit.get("decision")
        != "authorize_topology_veto_closed_loop_development_only"
        or joint_audit.get("freeze_result", {}).get("sha256")
        != _sha256(freeze_result_path)
        or freeze_result.get("active_cities") != ["jinan"]
        or int(v59["abandoned_v75"]["result_file_count_at_correction_freeze"])
        != 0
        or failed_preflight.get("submitted") is not False
        or "remote preflight failed"
        not in str(failed_preflight.get("execution_error", ""))
        or "direct_results" in failed_preflight
        or _sha256(FAILED_V61_PROTOCOL_PATH)
        != str(failed_preflight.get("frozen_protocol_sha256"))
    ):
        raise ValueError("v78 closed-loop parent evidence changed")
    artifact_files = {
        path.relative_to(artifact_root).as_posix(): _sha256(path)
        for path in sorted(Path(artifact_root).rglob("*"))
        if path.is_file()
    }
    if len(artifact_files) != 4:
        raise ValueError("v78 topology-veto artifact tree changed")
    future_counts = {
        "abandoned_v75_root": _file_count(ABANDONED_V75_ROOT),
        "development_results": _file_count(development_results_root),
        "launch_output": int(Path(launch_output).is_file()),
        "development_audit": int(Path(development_audit_output).is_file()),
        "validation_root": _file_count(validation_root),
        "prospective_root": _file_count(prospective_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v78 future evidence already exists: {future_counts}")
    sources = (
        "cf_h2o/eval/traffic_signal_external_topology_veto_closed_loop_development.py",
        "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py",
        "cf_h2o/traffic_signal/topology_target_veto.py",
        "scripts/cluster/run_tsc_external_topology_veto_closed_loop_development_shard.py",
        "scripts/cluster/launch_tsc_external_topology_veto_closed_loop_development.py",
        "scripts/cluster/audit_tsc_external_topology_veto_closed_loop_development.py",
    )
    old_rule = abandoned_v58["development"]["selection_rule"]
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v77_artifact_audit_pre_adaptation_seed_closed_loop_outcome",
        "failed_v61_preflight": {
            "path": str(failed_preflight_path),
            "sha256": _sha256(failed_preflight_path),
            "rollouts_observed": False,
            "reason": "incorrect_empty_remote_conversion_path",
        },
        "topology_veto_freeze_protocol": {
            "path": str(V60_PATH),
            "sha256": _sha256(V60_PATH),
        },
        "topology_veto_freeze_result": {
            "path": str(freeze_result_path),
            "sha256": _sha256(freeze_result_path),
        },
        "topology_veto_joint_audit": {
            "path": str(joint_audit_path),
            "sha256": _sha256(joint_audit_path),
        },
        "topology_veto_artifact_root": {
            "path": str(Path(artifact_root).resolve()),
            "files": artifact_files,
        },
        "proposal_parent_protocol": {
            "path": str(PROPOSAL_PARENT_PATH),
            "sha256": _sha256(PROPOSAL_PARENT_PATH),
        },
        "hierarchical_freeze_audit": {
            "path": str(hierarchical_audit_path),
            "sha256": _sha256(hierarchical_audit_path),
        },
        "runtime_executable_sources": {
            path: _sha256(Path(path)) for path in sources
        },
        "proposal": {
            "method": "frozen_v60_hierarchical_cfcmt",
            "prediction_horizon_sec": proposal_parent["environment"][
                "prediction_horizon_sec"
            ],
            "target_labels_used_by_proposal": False,
            "city_artifacts": proposal_parent["city_artifacts"],
        },
        "development": {
            "classification": "target_labeled_few_shot_adaptation_development",
            "seeds": v59["diagnostic"]["seeds"],
            "city_scenarios": v59["diagnostic"]["city_scenarios"],
            "active_cities": ["jinan"],
            "fallback_cities": ["los_angeles"],
            "duration_sec": 3600,
            "primary_metric": "mean_tripinfo_waiting_time",
            "phase_policy": PHASE_POLICY,
            "candidate": {
                "key": CANDIDATE_KEY,
                "runtime_policy": (
                    "causal_group_normalized_rigid_advantage_contrast_"
                    "hierarchical_guard"
                ),
                "coordination_mode": "direct",
                "cooldown_intervals": 44,
                "max_simultaneous_overrides": 1,
                "selection_origin": (
                    "predeclared_v58_execution_candidate_reused_after_future_"
                    "feature_rejection_before_any_v75_rollout"
                ),
            },
            "policies": [PHASE_POLICY, CANDIDATE_KEY],
            "paired_matrix_size": 48,
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260821,
                "unit": "paired_simulator_seed_equal_scenarios_within_city",
            },
            "selection_rule": old_rule,
            "single_candidate_no_execution_hyperparameter_search": True,
        },
        "environment": cache_protocol["environment"],
        "future_file_counts_at_freeze": future_counts,
        "future_outputs": {
            "validation_root": str(Path(validation_root).resolve()),
            "prospective_root": str(Path(prospective_root).resolve()),
        },
        "validation": v60["validation"],
        "prospective_confirmation": v60["prospective_confirmation"],
        "claim_boundary": (
            "Only the six adaptation seeds may be used for this paired development "
            "gate. The 16 validation and three prospective seeds remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-result", type=Path, required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--development-results-root", type=Path, required=True)
    parser.add_argument("--launch-output", type=Path, required=True)
    parser.add_argument("--development-audit-output", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--failed-preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v78 protocol: {args.out}")
    payload = build_protocol(
        freeze_result_path=args.freeze_result,
        joint_audit_path=args.joint_audit,
        artifact_root=args.artifact_root,
        development_results_root=args.development_results_root,
        launch_output=args.launch_output,
        development_audit_output=args.development_audit_output,
        validation_root=args.validation_root,
        prospective_root=args.prospective_root,
        failed_preflight_path=args.failed_preflight,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "matrix_size": payload["development"]["paired_matrix_size"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
