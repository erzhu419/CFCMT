#!/usr/bin/env python3
"""Freeze v68 proposal-conditional closed-loop development."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    RESULT_PROTOCOL as V64_ROLLOUT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as TRAINING_DECISION,
    RESULT_PROTOCOL as TRAINING_RESULT_PROTOCOL,
)
from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    AUDIT_PROTOCOL as V64_AUDIT_PROTOCOL,
    REJECTION_DECISION as V64_REJECTION_DECISION,
)
from scripts.cluster.audit_tsc_external_proposal_conditional_veto_successor import (  # noqa: E402
    AUDIT_DECISION as VETO_AUDIT_DECISION,
    AUDIT_PROTOCOL as VETO_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (  # noqa: E402
    PROTOCOL as COLLECTION_PROTOCOL,
    _file_count,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_closed_loop_protocol import (  # noqa: E402
    PROTOCOL as PROPOSAL_PARENT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_proposal_conditional_veto_successor import (  # noqa: E402
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = (
    "tsc-v68r64-external-v9-proposal-conditional-closed-loop-development-v1"
)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260818


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _phase_reference_manifest(
    *, parent: dict[str, Any], v64_audit: dict[str, Any], v64_results_root: Path
) -> list[dict[str, Any]]:
    audited = {
        (
            str(row["city"]),
            str(row["scenario"]),
            int(row["seed"]),
        ): row
        for row in v64_audit.get("rows", ())
        if str(row.get("policy")) == "phase_pressure"
    }
    development = dict(parent["closed_loop_development"])
    rows = []
    for city, scenarios in sorted(development["city_scenarios"].items()):
        for scenario in scenarios:
            for seed in development["seeds"]:
                identity = (str(city), str(scenario), int(seed))
                root = (
                    Path(v64_results_root)
                    / str(city)
                    / str(scenario)
                    / f"seed_{int(seed)}"
                    / "phase_pressure"
                )
                result_path = root / "result.json"
                tripinfo_path = root / "tripinfo.xml"
                result = _read_json(result_path)
                audit_row = audited.get(identity)
                if (
                    audit_row is None
                    or result.get("protocol") != V64_ROLLOUT_PROTOCOL
                    or result.get("policy") != "phase_pressure"
                    or (
                        str(result.get("city")),
                        str(result.get("scenario")),
                        int(result.get("seed", -1)),
                    )
                    != identity
                    or not bool(result.get("metrics", {}).get("ok", False))
                    or _sha256(result_path) != str(audit_row["result_sha256"])
                    or _sha256(tripinfo_path) != str(audit_row["tripinfo_sha256"])
                    or result.get("tripinfo_evidence", {}).get("sha256")
                    != _sha256(tripinfo_path)
                ):
                    raise ValueError(f"v68 phase reference changed: {identity}")
                rows.append(
                    {
                        "city": identity[0],
                        "scenario": identity[1],
                        "seed": identity[2],
                        "result": {
                            "path": _reference(result_path),
                            "sha256": _sha256(result_path),
                        },
                        "tripinfo": {
                            "path": _reference(tripinfo_path),
                            "sha256": _sha256(tripinfo_path),
                        },
                    }
                )
    if len(rows) != 32 or len(
        {(row["city"], row["scenario"], row["seed"]) for row in rows}
    ) != 32:
        raise ValueError("v68 phase-reference matrix changed")
    return rows


def build_protocol(
    *,
    parent_path: Path,
    collection_protocol_path: Path,
    training_result_path: Path,
    veto_joint_audit_path: Path,
    v64_audit_path: Path,
    v64_results_root: Path,
    future_development_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_path)
    collection = _read_json(collection_protocol_path)
    proposal_parent_path = PROJECT_ROOT / str(collection["parent_protocol"]["path"])
    proposal_parent = _read_json(proposal_parent_path)
    training_result = _read_json(training_result_path)
    veto_audit = _read_json(veto_joint_audit_path)
    v64_audit = _read_json(v64_audit_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or collection.get("protocol") != COLLECTION_PROTOCOL
        or parent["collection_protocol"]["sha256"]
        != _sha256(collection_protocol_path)
        or proposal_parent.get("protocol") != PROPOSAL_PARENT_PROTOCOL
        or _sha256(proposal_parent_path) != collection["parent_protocol"]["sha256"]
        or training_result.get("protocol") != TRAINING_RESULT_PROTOCOL
        or training_result.get("status") != "PASS"
        or training_result.get("decision") != TRAINING_DECISION
        or training_result.get("successor_protocol_sha256") != _sha256(parent_path)
        or veto_audit.get("protocol") != VETO_AUDIT_PROTOCOL
        or veto_audit.get("status") != "PASS"
        or veto_audit.get("decision") != VETO_AUDIT_DECISION
        or not bool(veto_audit.get("integrity_gate", {}).get("passed", False))
        or veto_audit.get("training_result_sha256") != _sha256(training_result_path)
        or v64_audit.get("protocol") != V64_AUDIT_PROTOCOL
        or v64_audit.get("status") != "PASS"
        or v64_audit.get("decision") != V64_REJECTION_DECISION
        or not bool(v64_audit.get("integrity_gate", {}).get("passed", False))
        or v64_audit.get("development_protocol_sha256")
        != collection["parent_protocol"]["sha256"]
        or _file_count(future_development_root) != 0
    ):
        raise ValueError("v68 target-veto closed-loop authorization changed")
    phase_references = _phase_reference_manifest(
        parent=collection,
        v64_audit=v64_audit,
        v64_results_root=v64_results_root,
    )
    active_cities = sorted(
        city for city, row in veto_audit["cities"].items() if bool(row["active"])
    )
    fallback_cities = sorted(set(collection["city_scenarios"]) - set(active_cities))
    if not active_cities:
        raise ValueError("v68 requires at least one active city")
    development = dict(collection["closed_loop_development"])
    candidates = list(development["method_candidates"])
    method_matrix = (
        len(development["seeds"])
        * sum(len(values) for values in development["city_scenarios"].values())
        * len(candidates)
    )
    if method_matrix != 96:
        raise ValueError("v68 method-only matrix changed")
    runtime_sources = {
        path: _sha256(PROJECT_ROOT / path)
        for path in (
            "cf_h2o/eval/traffic_signal_external_target_veto_closed_loop_development.py",
            "cf_h2o/traffic_signal/proposal_conditional_target_veto.py",
            "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py",
            "scripts/cluster/run_tsc_external_target_veto_closed_loop_development_shard.py",
        )
    }
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "stage": "post_oof_pre_closed_loop_outcome_development_freeze",
        "parent_protocol": {
            "path": _reference(parent_path),
            "sha256": _sha256(parent_path),
        },
        "collection_protocol": {
            "path": _reference(collection_protocol_path),
            "sha256": _sha256(collection_protocol_path),
        },
        "proposal_parent_protocol": {
            "path": _reference(proposal_parent_path),
            "sha256": _sha256(proposal_parent_path),
        },
        "hierarchical_freeze_audit": proposal_parent[
            "hierarchical_freeze_audit"
        ],
        "successor_training_result": {
            "path": _reference(training_result_path),
            "sha256": _sha256(training_result_path),
        },
        "target_veto_joint_audit": {
            "path": _reference(veto_joint_audit_path),
            "sha256": _sha256(veto_joint_audit_path),
            "decision": VETO_AUDIT_DECISION,
        },
        "v64_integrity_audit": {
            "path": _reference(v64_audit_path),
            "sha256": _sha256(v64_audit_path),
            "decision": V64_REJECTION_DECISION,
            "reuse_scope": "phase_pressure_rows_only",
        },
        "environment": collection["environment"],
        "proposal": collection["proposal"],
        "runtime_executable_sources": runtime_sources,
        "city_target_veto_artifacts": {
            city: {
                "model": row["model"],
                "certificate": row["certificate"],
                "active": bool(row["active"]),
                "selected_gate": row["selected_gate"],
            }
            for city, row in veto_audit["cities"].items()
        },
        "development": {
            "seeds": development["seeds"],
            "city_scenarios": development["city_scenarios"],
            "active_cities": active_cities,
            "fallback_cities": fallback_cities,
            "duration_sec": development["duration_sec"],
            "primary_metric": development["primary_metric"],
            "phase_policy": "phase_pressure",
            "method_candidates": candidates,
            "candidate_count": len(candidates),
            "method_only_matrix_size": method_matrix,
            "reused_phase_matrix_size": len(phase_references),
            "total_paired_matrix_size": method_matrix + len(phase_references),
            "phase_references": phase_references,
            "selection_rule": {
                "active_city": {
                    "minimum_city_mean_relative_improvement": 0.005,
                    "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.01,
                    "maximum_seed_level_city_mean_relative_regression": 0.01,
                    "minimum_improved_seed_fraction": 0.5,
                    "minimum_executed_interventions_per_scenario_seed": 1,
                },
                "fallback_city": {
                    "executed_interventions_must_equal": 0,
                    "mean_waiting_time_must_equal_phase_atol": 1e-12,
                    "mean_travel_time_must_equal_phase_atol": 1e-12,
                    "safety_counts_must_equal_phase": True,
                },
                "all_cities_must_satisfy_declared_active_or_fallback_contract": True,
                "method_teleports_must_not_exceed_paired_phase_pressure": True,
                "method_collision_incidents_must_not_exceed_paired_phase_pressure": True,
                "global_selector": (
                    "lowest equal-active-city robust score among candidates satisfying "
                    "every active and fallback city contract"
                ),
                "robust_score": (
                    "city_mean_relative_delta + 0.5 * seed_delta_std + "
                    "max(worst_seed_delta, 0)"
                ),
                "tie_break": (
                    "lower active-city robust score, lower active-city mean delta, "
                    "fewer interventions, longer cooldown, lexical key"
                ),
                "fallback": "phase_pressure_if_no_candidate_is_feasible",
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired simulator seed with equal scenarios within active city",
            },
            "future_result_file_count_at_freeze": 0,
        },
        "validation": collection["validation"],
        "prospective_confirmation": collection["prospective_confirmation"],
        "claim_boundary": (
            "Outcome-informed proposal-conditional target veto evaluated on disclosed development "
            "seeds. Active cities test improvement; OOF fallback cities test exact "
            "safe abstention. Validation and prospective seeds remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--v64-audit", type=Path, required=True)
    parser.add_argument("--v64-results-root", type=Path, required=True)
    parser.add_argument("--future-development-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v68 protocol: {args.out}")
    payload = build_protocol(
        parent_path=args.parent,
        collection_protocol_path=args.collection_protocol,
        training_result_path=args.training_result,
        veto_joint_audit_path=args.veto_joint_audit,
        v64_audit_path=args.v64_audit,
        v64_results_root=args.v64_results_root,
        future_development_root=args.future_development_root,
        frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "active_cities": payload["development"]["active_cities"],
                "fallback_cities": payload["development"]["fallback_cities"],
                "method_matrix": payload["development"]["method_only_matrix_size"],
                "phase_references": payload["development"]["reused_phase_matrix_size"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
