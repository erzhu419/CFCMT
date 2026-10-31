#!/usr/bin/env python3
"""Freeze v71 demand-conditioned closed-loop development before outcomes exist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_demand_conditioned_closed_loop_development import (  # noqa: E402
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    PROTOCOL,
    TRAINING_PROTOCOL,
    TRAINING_RESULT_PROTOCOL,
    all_method_rollout_identities,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260821


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def _phase_references(
    *, development: Mapping[str, Any]
) -> list[dict[str, Any]]:
    seeds = {int(value) for value in development["seeds"]}
    expected = {
        (str(city), str(scenario), int(seed))
        for city, scenarios in development["city_scenarios"].items()
        for scenario in scenarios
        for seed in seeds
    }
    rows = []
    observed = set()
    for raw in development["phase_references"]:
        row = dict(raw)
        identity = (
            str(row["city"]),
            str(row["scenario"]),
            int(row["seed"]),
        )
        result_path = PROJECT_ROOT / str(row["result"]["path"])
        tripinfo_path = PROJECT_ROOT / str(row["tripinfo"]["path"])
        result = _read_json(result_path)
        if (
            identity in observed
            or identity not in expected
            or _sha256(result_path) != str(row["result"]["sha256"])
            or _sha256(tripinfo_path) != str(row["tripinfo"]["sha256"])
            or result.get("policy") != "phase_pressure"
            or (
                str(result.get("city")),
                str(result.get("scenario")),
                int(result.get("seed", -1)),
            )
            != identity
            or not bool(result.get("metrics", {}).get("ok", False))
            or result.get("tripinfo_evidence", {}).get("sha256")
            != _sha256(tripinfo_path)
        ):
            raise ValueError(f"v71 phase reference changed: {identity}")
        observed.add(identity)
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
    if observed != expected or len(rows) != 32:
        raise ValueError("v71 phase reference matrix is not exact")
    return rows


def build_protocol(
    *,
    training_protocol_path: Path,
    training_result_path: Path,
    veto_joint_audit_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    future_development_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    training_protocol = _read_json(training_protocol_path)
    training_result = _read_json(training_result_path)
    veto_audit = _read_json(veto_joint_audit_path)
    cache_protocol_path = Path(training_protocol["cache_protocol"]["path"])
    cache_protocol = _read_json(cache_protocol_path)
    if (
        training_protocol.get("protocol") != TRAINING_PROTOCOL
        or training_result.get("protocol") != TRAINING_RESULT_PROTOCOL
        or training_result.get("status") != "PASS"
        or training_result.get("training_protocol_sha256")
        != _sha256(training_protocol_path)
        or veto_audit.get("protocol") != JOINT_AUDIT_PROTOCOL
        or veto_audit.get("status") != "PASS"
        or veto_audit.get("decision") != JOINT_AUDIT_DECISION
        or not bool(veto_audit.get("integrity_gate", {}).get("passed", False))
        or veto_audit.get("training_protocol_sha256")
        != _sha256(training_protocol_path)
        or veto_audit.get("training_result_sha256")
        != _sha256(training_result_path)
        or _sha256(cache_protocol_path)
        != training_protocol["cache_protocol"]["sha256"]
        or _sha256(proposal_parent_protocol_path)
        != training_protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != training_protocol["hierarchical_freeze_audit"]["sha256"]
        or _file_count(future_development_root) != 0
    ):
        raise ValueError("v71 closed-loop authorization evidence changed")

    development_parent = dict(training_protocol["closed_loop_development"])
    phase_references = _phase_references(development=development_parent)
    active_cities = sorted(str(value) for value in veto_audit["active_cities"])
    all_cities = set(development_parent["city_scenarios"])
    fallback_cities = sorted(all_cities - set(active_cities))
    if not active_cities or set(active_cities) - all_cities:
        raise ValueError("v71 requires at least one audited active city")
    candidate = dict(development_parent["candidate"])
    city_artifacts = {
        city: {
            "model": row["model"],
            "certificate": row["certificate"],
            "active": bool(row["active"]),
            "selected_gate": row["selected_gate"],
        }
        for city, row in veto_audit["cities"].items()
    }
    if set(city_artifacts) != all_cities or {
        city for city, row in city_artifacts.items() if row["active"]
    } != set(active_cities):
        raise ValueError("v71 active/fallback artifact partition changed")

    runtime_sources = {
        path: _sha256(PROJECT_ROOT / path)
        for path in (
            "cf_h2o/eval/traffic_signal_external_demand_conditioned_closed_loop_development.py",
            "cf_h2o/traffic_signal/demand_conditioned_target_veto.py",
            "cf_h2o/traffic_signal/demand_schedule_context.py",
            "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py",
            "scripts/cluster/run_tsc_external_demand_conditioned_closed_loop_development_shard.py",
        )
    }
    development = {
        "seeds": list(development_parent["seeds"]),
        "city_scenarios": development_parent["city_scenarios"],
        "active_cities": active_cities,
        "fallback_cities": fallback_cities,
        "duration_sec": int(cache_protocol["environment"]["duration_sec"]),
        "primary_metric": "mean_tripinfo_waiting_time",
        "phase_policy": "phase_pressure",
        "candidate": candidate,
        "method_candidates": [candidate],
        "candidate_count": 1,
        "method_only_matrix_size": 32,
        "reused_phase_matrix_size": len(phase_references),
        "total_paired_matrix_size": 32 + len(phase_references),
        "phase_references": phase_references,
        "selection_rule": development_parent["selection_rule"],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired_simulator_seed_equal_scenarios_within_city",
        },
        "future_result_file_count_at_freeze": 0,
    }
    draft = {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "stage": "post_v70_oof_pre_v71_closed_loop_outcome_development_freeze",
        # These aliases preserve the mature paired-development audit contract.
        "parent_protocol": {
            "path": _reference(training_protocol_path),
            "sha256": _sha256(training_protocol_path),
            "semantic_role": "demand_conditioned_training_protocol",
        },
        "collection_protocol": {
            "path": _reference(training_result_path),
            "sha256": _sha256(training_result_path),
            "semantic_role": "demand_conditioned_training_result",
        },
        "training_protocol": {
            "path": _reference(training_protocol_path),
            "sha256": _sha256(training_protocol_path),
        },
        "proposal_parent_protocol": {
            "path": _reference(proposal_parent_protocol_path),
            "sha256": _sha256(proposal_parent_protocol_path),
        },
        "hierarchical_freeze_audit": {
            "path": _reference(freeze_audit_path),
            "sha256": _sha256(freeze_audit_path),
        },
        "target_veto_training_result": {
            "path": _reference(training_result_path),
            "sha256": _sha256(training_result_path),
        },
        "target_veto_joint_audit": {
            "path": _reference(veto_joint_audit_path),
            "sha256": _sha256(veto_joint_audit_path),
            "decision": JOINT_AUDIT_DECISION,
        },
        "environment": cache_protocol["environment"],
        "proposal": cache_protocol["proposal"],
        "demand_schedule_context": cache_protocol["demand_schedule_context"],
        "runtime_executable_sources": runtime_sources,
        "city_target_veto_artifacts": city_artifacts,
        "development": development,
        "validation": training_protocol["validation"],
        "prospective_confirmation": training_protocol[
            "prospective_confirmation"
        ],
        "claim_boundary": (
            "Target-simulator-labeled few-shot veto on disclosed development seeds. "
            "The frozen v60 proposal is unchanged; the v70 model may only veto it. "
            "Validation and prospective seeds remain sealed."
        ),
    }
    identities = all_method_rollout_identities(draft)
    if len(identities) != development["method_only_matrix_size"]:
        raise ValueError("v71 method matrix changed")
    return draft


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--future-development-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v71 protocol: {args.out}")
    payload = build_protocol(
        training_protocol_path=args.training_protocol,
        training_result_path=args.training_result,
        veto_joint_audit_path=args.veto_joint_audit,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
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
                "method_matrix": payload["development"][
                    "method_only_matrix_size"
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
