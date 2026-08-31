"""Evaluate the frozen v56 pressure controller on the fresh v9 seed 9277."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    EXTERNAL_COLLECTION_SHARDS,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    _read_json,
    _resolve_protocol_reference,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    V9_EVALUATION_SEED,
    V9_PROTOCOL as V9_FULL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    V9_JOINT_AUDIT_DECISION,
    V9_JOINT_AUDIT_PROTOCOL,
    V9_PROTOCOL,
    _load_artifacts,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    MIN_MEAN_IMPROVEMENT,
    MIN_OPERATIONAL_TOTAL_VEHICLES,
    _summarize_pressure_regularizer_candidate,
    oof_action_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import PriorRegularizationConfig
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


RESULT_PROTOCOL = "tsc-v58r54-external-v9-pressure-seed9277-heldout-evaluation-v1"
AUTHORIZATION_DECISION = "authorize_external_v9_closed_loop_development_replay"
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 10000
MAX_BOOTSTRAP_UPPER_DELTA = 0.01
MAX_SCENARIO_MEAN_DELTA = 0.01


def _selected_delta_rows(
    records: Sequence[Mapping[str, Any]], regularizer: PriorRegularizationConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        objective = (
            float(record["relative_rule_gap"])
            + float(regularizer.blend_weight) * float(record["predicted_delta"])
            + float(regularizer.risk_multiplier) * float(record["uncertainty"])
        )
        selected = bool(
            record["learned_differs"]
            and float(record["context_trust"])
            >= float(regularizer.min_context_trust)
            and float(record["candidate_features"]["total_veh"])
            >= MIN_OPERATIONAL_TOTAL_VEHICLES
            and objective < 0.0
        )
        rows.append(
            {
                "scenario": str(record["scenario"]),
                "group_id": str(record["group_id"]),
                "selected": selected,
                "objective": float(objective),
                "actual_delta": (
                    float(record["actual_delta"]) if selected else 0.0
                ),
                "predicted_delta": float(record["predicted_delta"]),
                "uncertainty": float(record["uncertainty"]),
                "context_trust": float(record["context_trust"]),
            }
        )
    return rows


def _equal_scenario_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    by_scenario: dict[str, np.ndarray] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        values = np.asarray(
            [
                float(row["actual_delta"])
                for row in rows
                if str(row["scenario"]) == scenario
            ],
            dtype=float,
        )
        if values.size == 0:
            raise ValueError(f"held-out pressure bootstrap lacks {scenario}")
        by_scenario[scenario] = values
    observed = float(np.mean([values.mean() for values in by_scenario.values()]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        draws[index] = float(
            np.mean(
                [
                    values[
                        rng.integers(0, values.size, size=values.size)
                    ].mean()
                    for values in by_scenario.values()
                ]
            )
        )
    return {
        "protocol": "paired-group-equal-scenario-bootstrap-v1",
        "unit": "counterfactual action group within scenario",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "scenario_count": len(by_scenario),
        "group_counts": {
            scenario: int(values.size) for scenario, values in by_scenario.items()
        },
        "observed_mean_delta_vs_phase_pressure": observed,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "one_sided_probability_delta_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _city_gate(
    *, summary: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> dict[str, Any]:
    selected = int(summary["accepted_overrides"])
    observed = float(bootstrap["observed_mean_delta_vs_phase_pressure"])
    upper = float(bootstrap["ci95"][1])
    gate = {
        "minimum_mean_improvement": float(MIN_MEAN_IMPROVEMENT),
        "observed_mean_delta_vs_phase_pressure": observed,
        "bootstrap_95pct_upper_delta": upper,
        "worst_scenario_or_fold_delta": float(summary["worst_stratum_mean_delta"]),
        "accepted_overrides": selected,
        "minimum_overrides": int(summary["minimum_required_overrides"]),
        "mean_improvement_passed": observed <= -float(MIN_MEAN_IMPROVEMENT),
        "bootstrap_upper_passed": upper <= MAX_BOOTSTRAP_UPPER_DELTA,
        "worst_scenario_passed": float(summary["worst_stratum_mean_delta"])
        <= MAX_SCENARIO_MEAN_DELTA,
        "intervention_coverage_passed": selected
        >= int(summary["minimum_required_overrides"]),
    }
    gate["passed"] = all(
        gate[key]
        for key in (
            "mean_improvement_passed",
            "bootstrap_upper_passed",
            "worst_scenario_passed",
            "intervention_coverage_passed",
        )
    )
    return gate


def run_pressure_heldout_evaluation(
    *,
    evaluation_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    full_protocol_path: Path,
    joint_freeze_audit_path: Path,
    expected_joint_freeze_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    expected_evaluation_cache_sha256: str,
    workers: int,
) -> dict[str, Any]:
    full_protocol = _read_json(full_protocol_path)
    joint = _read_json(joint_freeze_audit_path)
    pressure = _read_json(pressure_protocol_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    if full_protocol.get("protocol") != V9_FULL_PROTOCOL:
        raise ValueError("pressure held-out full protocol changed")
    parent_path = _resolve_protocol_reference(
        pressure_protocol_path, str(pressure["parent_protocol"]["path"])
    )
    aligned_parent = _read_json(parent_path)
    full_parent_path = _resolve_protocol_reference(
        parent_path, str(aligned_parent["parent_protocol"]["path"])
    )
    if (
        pressure.get("protocol") != V9_PROTOCOL
        or _sha256(parent_path) != str(pressure["parent_protocol"]["sha256"])
        or _sha256(full_parent_path) != str(aligned_parent["parent_protocol"]["sha256"])
        or _sha256(full_parent_path) != _sha256(full_protocol_path)
        or _sha256(joint_freeze_audit_path) != str(expected_joint_freeze_sha256)
        or joint.get("status") != "PASS"
        or joint.get("decision") != "authorize_external_v9_seed9277_collection"
        or int(joint.get("evaluation_cache_file_count_at_freeze", -1)) != 0
        or _sha256(pressure_joint_audit_path)
        != str(expected_pressure_joint_audit_sha256)
        or pressure_joint.get("protocol") != V9_JOINT_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != V9_JOINT_AUDIT_DECISION
    ):
        raise ValueError("pressure held-out evidence chain changed")

    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in full_protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    cache_sha, cache_files = aggregate_cache_sha256(evaluation_cache_root)
    expected_files = sum(len(values) for values in city_scenarios.values()) * int(
        EXTERNAL_COLLECTION_SHARDS
    )
    if (
        cache_sha != str(expected_evaluation_cache_sha256)
        or cache_files != expected_files
    ):
        raise ValueError("pressure held-out cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=(V9_EVALUATION_SEED,),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    city_results: dict[str, Any] = {}
    for index, (city, scenarios) in enumerate(city_scenarios.items()):
        certificate, model = _load_artifacts(
            protocol=pressure,
            joint=pressure_joint,
            freeze_root=pressure_freeze_root,
            city=city,
        )
        regularizer: PriorRegularizationConfig = model["regularizer"]
        if not regularizer.enabled:
            raise ValueError(f"v56 pressure controller is inactive for {city}")
        dataset = _merge_city_datasets(bank, scenarios, city=city)
        records = oof_action_records(
            dataset,
            models=model["models"],
            selected_candidate=str(model["selected_candidate"]),
            fold_index=0,
            validation_seed=V9_EVALUATION_SEED,
            scenarios=scenarios,
            target_support=model["target_support"],
        )
        summary = _summarize_pressure_regularizer_candidate(
            records,
            blend_weight=float(regularizer.blend_weight),
            risk_multiplier=float(regularizer.risk_multiplier),
            min_context_trust=float(regularizer.min_context_trust),
        )
        selected_rows = _selected_delta_rows(records, regularizer)
        bootstrap = _equal_scenario_bootstrap(
            selected_rows, seed=BOOTSTRAP_SEED + index
        )
        if (
            int(summary["accepted_overrides"])
            != sum(bool(row["selected"]) for row in selected_rows)
            or not np.isclose(
                float(summary["mean_oof_delta_vs_pressure"]),
                float(bootstrap["observed_mean_delta_vs_phase_pressure"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError("pressure held-out decision replay mismatch")
        city_results[city] = {
            "scenarios": list(scenarios),
            "selected_candidate": str(model["selected_candidate"]),
            "regularizer": certificate["deployment"]["regularizer"],
            "decision_count": len(records),
            "fixed_regularizer_summary": summary,
            "bootstrap": bootstrap,
            "gate": _city_gate(summary=summary, bootstrap=bootstrap),
            "selected_action_rows": selected_rows,
            "certificate_sha256": str(
                pressure["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "model_sha256": str(
                pressure["city_artifacts"][city]["model"]["sha256"]
            ),
        }
    gate = {
        "all_cities_present": set(city_results) == set(city_scenarios),
        "all_city_gates_passed": bool(city_results)
        and all(row["gate"]["passed"] for row in city_results.values()),
        "evaluation_seed_is_fresh": V9_EVALUATION_SEED
        not in set(full_protocol["target_protocol"]["adaptation_seeds"]),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "claim_boundary": (
            "Fresh-seed action-regret validation of the frozen v56 controller; this "
            "can authorize development replay but cannot establish closed-loop efficacy."
        ),
        "protocol_spec_sha256": _sha256(full_protocol_path),
        "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
        "pressure_protocol_sha256": _sha256(pressure_protocol_path),
        "pressure_joint_audit_sha256": _sha256(pressure_joint_audit_path),
        "evaluation_cache_aggregate_sha256": cache_sha,
        "evaluation_cache_file_count": cache_files,
        "evaluation_cache_audit": cache_audit,
        "evaluation_seed": V9_EVALUATION_SEED,
        "city_results": city_results,
        "offline_confirmation_gate": gate,
        "decision": (
            AUTHORIZATION_DECISION
            if gate["passed"]
            else "retain_v56_offline_failure_and_prohibit_closed_loop_claim"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--full-protocol", type=Path, required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--expected-evaluation-cache-sha256", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite pressure held-out result: {args.out}")
    payload = run_pressure_heldout_evaluation(
        evaluation_cache_root=args.evaluation_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        full_protocol_path=args.full_protocol,
        joint_freeze_audit_path=args.joint_freeze_audit,
        expected_joint_freeze_sha256=args.expected_joint_freeze_sha256,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
        expected_evaluation_cache_sha256=args.expected_evaluation_cache_sha256,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "gate_passed": payload["offline_confirmation_gate"]["passed"],
                "decision": payload["decision"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["offline_confirmation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
