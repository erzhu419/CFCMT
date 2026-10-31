"""Post-failure, no-tuning diagnosis of hierarchical external-city guards.

This module replays the frozen v55 conformal guard and the frozen v56 pressure
regularizer on the already inspected v9 seed 9277.  It cannot authorize a
confirmatory claim.  A successful result may only justify freezing a successor
controller before collecting any additional counterfactual cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
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
    V9_ALIGNED_AUDIT_DECISION,
    V9_ALIGNED_AUDIT_PROTOCOL,
    V9_ALIGNED_PROTOCOL,
    _load_aligned_artifacts,
    _read_json,
    _resolve_protocol_reference,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    V9_EVALUATION_SEED,
    V9_PROTOCOL as V9_FULL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_heldout_evaluation import (
    RESULT_PROTOCOL as PRESSURE_HELDOUT_PROTOCOL,
    _city_gate,
    _equal_scenario_bootstrap,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    V9_JOINT_AUDIT_DECISION as V9_PRESSURE_AUDIT_DECISION,
    V9_JOINT_AUDIT_PROTOCOL as V9_PRESSURE_AUDIT_PROTOCOL,
    V9_PROTOCOL as V9_PRESSURE_PROTOCOL,
    _load_artifacts as _load_pressure_artifacts,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    MAX_OVERRIDE_FRACTION,
    MIN_OPERATIONAL_TOTAL_VEHICLES,
    MIN_OVERRIDE_FRACTION,
    oof_action_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
    hierarchical_guard_decision_v3,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


RESULT_PROTOCOL = "tsc-v59r55-external-v9-hierarchical-guard-seed9277-diagnostic-v2"
SUCCESSOR_DECISION = "freeze_hierarchical_guard_successor_before_new_seed_collection"
FAILURE_DECISION = "retain_hierarchical_guard_failure_and_prohibit_new_stage_claim"
BOOTSTRAP_SEED = 20260814

PRESSURE_ONLY = "pressure_only"
CONFORMAL_ONLY = "conformal_only"
HIERARCHICAL = "hierarchical_intersection"
VARIANTS = (PRESSURE_ONLY, CONFORMAL_ONLY, HIERARCHICAL)


def _object_sha256(value: object) -> str:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _pressure_decision(
    row: Mapping[str, Any], regularizer: PriorRegularizationConfig
) -> tuple[bool, float]:
    objective = (
        float(row["relative_rule_gap"])
        + float(regularizer.blend_weight) * float(row["predicted_delta"])
        + float(regularizer.risk_multiplier) * float(row["uncertainty"])
    )
    selected = bool(
        regularizer.enabled
        and row["learned_differs"]
        and float(row["context_trust"])
        >= float(regularizer.min_context_trust)
        and float(row["candidate_features"]["total_veh"])
        >= MIN_OPERATIONAL_TOTAL_VEHICLES
        and objective < 0.0
    )
    return selected, float(objective)


def _conformal_decision(
    row: Mapping[str, Any], guard: ContrastGuardConfig
) -> tuple[bool, float]:
    upper = (
        float(row["predicted_delta"])
        + float(guard.risk_multiplier) * float(row["uncertainty"])
        + float(guard.margin)
    )
    selected = bool(
        guard.enabled
        and row["learned_differs"]
        and float(row["context_trust"]) >= float(guard.min_context_trust)
        and float(row["relative_rule_gap"])
        <= float(guard.max_relative_rule_gap) + 1e-12
        and float(row["candidate_features"]["total_veh"])
        >= MIN_OPERATIONAL_TOTAL_VEHICLES
        and upper < 0.0
    )
    return selected, float(upper)


def _record_map(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row["group_id"]): row for row in records}
    if len(mapped) != len(records):
        raise ValueError(f"{label} records contain duplicate action groups")
    return mapped


def _assert_common_prediction(
    pressure: Mapping[str, Any], conformal: Mapping[str, Any]
) -> None:
    for key in ("scenario", "group_id", "learned_differs"):
        if pressure[key] != conformal[key]:
            raise ValueError(f"v55/v56 action identity changed at {key}")
    for key in (
        "predicted_delta",
        "uncertainty",
        "relative_rule_gap",
        "actual_delta",
    ):
        if not np.isclose(
            float(pressure[key]),
            float(conformal[key]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"v55/v56 common model output changed at {key}")
    pressure_features = dict(pressure["candidate_features"])
    conformal_features = dict(conformal["candidate_features"])
    if set(pressure_features) != set(conformal_features):
        raise ValueError("v55/v56 candidate feature set changed")
    if any(
        not np.isclose(
            float(pressure_features[key]),
            float(conformal_features[key]),
            rtol=0.0,
            atol=1e-12,
        )
        for key in pressure_features
    ):
        raise ValueError("v55/v56 candidate features changed")


def hierarchical_variant_rows(
    *,
    pressure_records: Sequence[Mapping[str, Any]],
    conformal_records: Sequence[Mapping[str, Any]],
    regularizer: PriorRegularizationConfig,
    guard: ContrastGuardConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Apply all frozen gates without selecting or tuning any new threshold."""

    pressure_by_group = _record_map(pressure_records, label="pressure")
    conformal_by_group = _record_map(conformal_records, label="conformal")
    if set(pressure_by_group) != set(conformal_by_group):
        raise ValueError("v55/v56 held-out action groups differ")
    rows = {variant: [] for variant in VARIANTS}
    for group_id in sorted(pressure_by_group):
        pressure = pressure_by_group[group_id]
        conformal = conformal_by_group[group_id]
        _assert_common_prediction(pressure, conformal)
        pressure_selected, pressure_objective = _pressure_decision(
            pressure, regularizer
        )
        conformal_selected, conformal_upper = _conformal_decision(
            conformal, guard
        )
        hierarchical = hierarchical_guard_decision_v3(
            learned_differs=bool(pressure["learned_differs"]),
            predicted_delta=float(pressure["predicted_delta"]),
            uncertainty=float(pressure["uncertainty"]),
            relative_rule_gap=float(pressure["relative_rule_gap"]),
            pressure_context_trust=float(pressure["context_trust"]),
            conformal_context_trust=float(conformal["context_trust"]),
            total_vehicles=float(pressure["candidate_features"]["total_veh"]),
            regularizer=regularizer,
            guard=guard,
        )
        if (
            not np.isclose(
                float(hierarchical["pressure_objective"]),
                pressure_objective,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(hierarchical["conformal_upper"]),
                conformal_upper,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError("shared hierarchical decision kernel drifted")
        selections = {
            PRESSURE_ONLY: pressure_selected,
            CONFORMAL_ONLY: conformal_selected,
            HIERARCHICAL: bool(hierarchical["eligible"]),
        }
        for variant, selected in selections.items():
            rows[variant].append(
                {
                    "scenario": str(pressure["scenario"]),
                    "group_id": group_id,
                    "selected": bool(selected),
                    "actual_delta": (
                        float(pressure["actual_delta"]) if selected else 0.0
                    ),
                    "raw_actual_delta": float(pressure["actual_delta"]),
                    "predicted_delta": float(pressure["predicted_delta"]),
                    "uncertainty": float(pressure["uncertainty"]),
                    "relative_rule_gap": float(pressure["relative_rule_gap"]),
                    "pressure_objective": pressure_objective,
                    "conformal_upper": conformal_upper,
                    "pressure_gate_selected": pressure_selected,
                    "conformal_gate_enabled": bool(guard.enabled),
                    "conformal_gate_selected": conformal_selected,
                    "hierarchical_rejection": hierarchical["rejection"],
                    "hierarchical_priority": float(hierarchical["priority"]),
                    "pressure_context_trust": float(pressure["context_trust"]),
                    "conformal_context_trust": float(conformal["context_trust"]),
                    "pressure_local_action_support": float(
                        pressure["local_action_support"]
                    ),
                    "conformal_local_action_support": float(
                        conformal["local_action_support"]
                    ),
                    "total_vehicles": float(
                        pressure["candidate_features"]["total_veh"]
                    ),
                }
            )
    return rows


def _decision_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    scenario_means = {
        scenario: float(
            np.mean(
                [
                    float(row["actual_delta"])
                    for row in rows
                    if str(row["scenario"]) == scenario
                ]
            )
        )
        for scenario in scenarios
    }
    accepted_rows = [row for row in rows if bool(row["selected"])]
    accepted = len(accepted_rows)
    minimum_overrides = max(
        2, int(math.ceil(float(MIN_OVERRIDE_FRACTION) * len(rows)))
    )
    return {
        "decisions": len(rows),
        "accepted_overrides": accepted,
        "minimum_required_overrides": minimum_overrides,
        "override_fraction": float(accepted / max(len(rows), 1)),
        "maximum_override_fraction": float(MAX_OVERRIDE_FRACTION),
        "mean_oof_delta_vs_pressure": float(np.mean(list(scenario_means.values()))),
        "worst_stratum_mean_delta": float(
            max(scenario_means.values(), default=0.0)
        ),
        "scenario_mean_delta": scenario_means,
        "accepted_harmful_actions": int(
            sum(float(row["raw_actual_delta"]) > 0.0 for row in accepted_rows)
        ),
        "accepted_beneficial_actions": int(
            sum(float(row["raw_actual_delta"]) < 0.0 for row in accepted_rows)
        ),
    }


def _variant_result(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_seed: int
) -> dict[str, Any]:
    summary = _decision_summary(rows)
    bootstrap = _equal_scenario_bootstrap(rows, seed=bootstrap_seed)
    if not np.isclose(
        float(summary["mean_oof_delta_vs_pressure"]),
        float(bootstrap["observed_mean_delta_vs_phase_pressure"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("hierarchical diagnostic aggregation mismatch")
    return {
        "summary": summary,
        "bootstrap": bootstrap,
        "gate": _city_gate(summary=summary, bootstrap=bootstrap),
        "action_rows": list(rows),
    }


def _verify_protocol_chain(
    *,
    full_protocol_path: Path,
    full_joint_audit_path: Path,
    expected_full_joint_audit_sha256: str,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
    expected_aligned_joint_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_heldout_result_path: Path,
    expected_pressure_heldout_result_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    full = _read_json(full_protocol_path)
    full_joint = _read_json(full_joint_audit_path)
    aligned = _read_json(aligned_protocol_path)
    aligned_joint = _read_json(aligned_joint_audit_path)
    pressure = _read_json(pressure_protocol_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    pressure_heldout = _read_json(pressure_heldout_result_path)
    aligned_parent = _resolve_protocol_reference(
        aligned_protocol_path, str(aligned["parent_protocol"]["path"])
    )
    pressure_parent = _resolve_protocol_reference(
        pressure_protocol_path, str(pressure["parent_protocol"]["path"])
    )
    if (
        full.get("protocol") != V9_FULL_PROTOCOL
        or _sha256(full_joint_audit_path)
        != str(expected_full_joint_audit_sha256)
        or full_joint.get("status") != "PASS"
        or full_joint.get("decision") != "authorize_external_v9_seed9277_collection"
        or int(full_joint.get("evaluation_cache_file_count_at_freeze", -1)) != 0
        or aligned.get("protocol") != V9_ALIGNED_PROTOCOL
        or _sha256(aligned_parent) != str(aligned["parent_protocol"]["sha256"])
        or _sha256(aligned_parent) != _sha256(full_protocol_path)
        or _sha256(aligned_joint_audit_path)
        != str(expected_aligned_joint_audit_sha256)
        or aligned_joint.get("protocol") != V9_ALIGNED_AUDIT_PROTOCOL
        or aligned_joint.get("status") != "PASS"
        or aligned_joint.get("decision") != V9_ALIGNED_AUDIT_DECISION
        or pressure.get("protocol") != V9_PRESSURE_PROTOCOL
        or _sha256(pressure_parent) != str(pressure["parent_protocol"]["sha256"])
        or _sha256(pressure_parent) != _sha256(aligned_protocol_path)
        or _sha256(pressure_joint_audit_path)
        != str(expected_pressure_joint_audit_sha256)
        or pressure_joint.get("protocol") != V9_PRESSURE_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != V9_PRESSURE_AUDIT_DECISION
        or _sha256(pressure_heldout_result_path)
        != str(expected_pressure_heldout_result_sha256)
        or pressure_heldout.get("protocol") != PRESSURE_HELDOUT_PROTOCOL
        or pressure_heldout.get("decision")
        != "retain_v56_offline_failure_and_prohibit_closed_loop_claim"
    ):
        raise ValueError("hierarchical diagnostic evidence chain changed")
    return full, aligned, aligned_joint, pressure, pressure_joint


def run_hierarchical_guard_diagnostic(
    *,
    evaluation_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    full_protocol_path: Path,
    full_joint_audit_path: Path,
    expected_full_joint_audit_sha256: str,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
    expected_aligned_joint_audit_sha256: str,
    aligned_freeze_root: Path,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    pressure_heldout_result_path: Path,
    expected_pressure_heldout_result_sha256: str,
    expected_evaluation_cache_sha256: str,
    workers: int,
) -> dict[str, Any]:
    full, aligned, aligned_joint, pressure, pressure_joint = _verify_protocol_chain(
        full_protocol_path=full_protocol_path,
        full_joint_audit_path=full_joint_audit_path,
        expected_full_joint_audit_sha256=expected_full_joint_audit_sha256,
        aligned_protocol_path=aligned_protocol_path,
        aligned_joint_audit_path=aligned_joint_audit_path,
        expected_aligned_joint_audit_sha256=expected_aligned_joint_audit_sha256,
        pressure_protocol_path=pressure_protocol_path,
        pressure_joint_audit_path=pressure_joint_audit_path,
        expected_pressure_joint_audit_sha256=expected_pressure_joint_audit_sha256,
        pressure_heldout_result_path=pressure_heldout_result_path,
        expected_pressure_heldout_result_sha256=expected_pressure_heldout_result_sha256,
    )
    pressure_heldout = _read_json(pressure_heldout_result_path)
    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in full["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    cache_sha, cache_files = aggregate_cache_sha256(evaluation_cache_root)
    expected_files = sum(len(values) for values in city_scenarios.values()) * int(
        EXTERNAL_COLLECTION_SHARDS
    )
    if (
        cache_sha != str(expected_evaluation_cache_sha256)
        or cache_sha != str(pressure_heldout["evaluation_cache_aggregate_sha256"])
        or cache_files != expected_files
    ):
        raise ValueError("hierarchical diagnostic evaluation cache changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(
        Path(conversion_root).resolve()
    )
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=(V9_EVALUATION_SEED,),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    city_results: dict[str, Any] = {}
    for city_index, (city, scenarios) in enumerate(sorted(city_scenarios.items())):
        aligned_certificate, aligned_model = _load_aligned_artifacts(
            protocol=aligned,
            joint=aligned_joint,
            aligned_freeze_root=aligned_freeze_root,
            city=city,
        )
        pressure_certificate, pressure_model = _load_pressure_artifacts(
            protocol=pressure,
            joint=pressure_joint,
            freeze_root=pressure_freeze_root,
            city=city,
        )
        aligned_models_sha = _object_sha256(aligned_model["models"])
        pressure_models_sha = _object_sha256(pressure_model["models"])
        if (
            aligned_models_sha != pressure_models_sha
            or aligned_model["selected_candidate"]
            != pressure_model["selected_candidate"]
        ):
            raise ValueError(f"v55/v56 common fitted model changed for {city}")
        dataset = _merge_city_datasets(bank, scenarios, city=city)
        common_kwargs = {
            "selected_candidate": str(aligned_model["selected_candidate"]),
            "fold_index": 0,
            "validation_seed": V9_EVALUATION_SEED,
            "scenarios": scenarios,
        }
        conformal_records = oof_action_records(
            dataset,
            models=aligned_model["models"],
            target_support=aligned_model["target_support"],
            **common_kwargs,
        )
        pressure_records = oof_action_records(
            dataset,
            models=pressure_model["models"],
            target_support=pressure_model["target_support"],
            **common_kwargs,
        )
        regularizer: PriorRegularizationConfig = pressure_model["regularizer"]
        guard: ContrastGuardConfig = aligned_model["guard"]
        variant_rows = hierarchical_variant_rows(
            pressure_records=pressure_records,
            conformal_records=conformal_records,
            regularizer=regularizer,
            guard=guard,
        )
        variants = {
            variant: _variant_result(
                variant_rows[variant],
                bootstrap_seed=BOOTSTRAP_SEED + 10 * city_index + variant_index,
            )
            for variant_index, variant in enumerate(VARIANTS)
        }
        frozen_pressure = pressure_heldout["city_results"][city]
        replay = variants[PRESSURE_ONLY]
        if (
            int(replay["summary"]["accepted_overrides"])
            != int(frozen_pressure["fixed_regularizer_summary"]["accepted_overrides"])
            or not np.isclose(
                float(replay["summary"]["mean_oof_delta_vs_pressure"]),
                float(
                    frozen_pressure["fixed_regularizer_summary"][
                        "mean_oof_delta_vs_pressure"
                    ]
                ),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(f"frozen pressure replay changed for {city}")
        city_results[city] = {
            "scenarios": list(scenarios),
            "selected_candidate": str(aligned_model["selected_candidate"]),
            "common_fitted_model_sha256": aligned_models_sha,
            "common_model_identity_verified": True,
            "aligned_certificate_sha256": str(
                aligned["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "aligned_model_sha256": str(
                aligned["city_artifacts"][city]["model"]["sha256"]
            ),
            "pressure_certificate_sha256": str(
                pressure["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "pressure_model_sha256": str(
                pressure["city_artifacts"][city]["model"]["sha256"]
            ),
            "regularizer": pressure_certificate["deployment"]["regularizer"],
            "conformal_guard": aligned_certificate["deployment"]["guard"],
            "conformal_support": aligned_model["target_support"].diagnostics(),
            "pressure_support": pressure_model["target_support"].diagnostics(),
            "hierarchical_rule": (
                "pressure_and_conformal_intersection"
                if guard.enabled
                else "pressure_gate_with_identity_conformal_filter"
            ),
            "variants": variants,
        }
    gate = {
        "seed9277_declared_post_failure_design_diagnostic": True,
        "no_new_threshold_or_model_selection": True,
        "common_model_identity_verified_all_cities": all(
            row["common_model_identity_verified"] for row in city_results.values()
        ),
        "pressure_failure_reproduced": bool(
            not all(
                row["variants"][PRESSURE_ONLY]["gate"]["passed"]
                for row in city_results.values()
            )
        ),
        "hierarchical_all_city_gates_passed": bool(city_results)
        and all(
            row["variants"][HIERARCHICAL]["gate"]["passed"]
            for row in city_results.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "claim_boundary": (
            "Seed 9277 was inspected after v54 and v56 failed. This is a fixed-"
            "component design diagnostic, not fresh validation, confirmation, or "
            "evidence of closed-loop efficacy."
        ),
        "evaluation_seed": V9_EVALUATION_SEED,
        "component_policy": (
            "No model or threshold is fit here. The v56 pressure regularizer and "
            "v55 conformal guard/support are replayed exactly as frozen."
        ),
        "full_protocol_sha256": _sha256(full_protocol_path),
        "full_joint_audit_sha256": _sha256(full_joint_audit_path),
        "aligned_protocol_sha256": _sha256(aligned_protocol_path),
        "aligned_joint_audit_sha256": _sha256(aligned_joint_audit_path),
        "pressure_protocol_sha256": _sha256(pressure_protocol_path),
        "pressure_joint_audit_sha256": _sha256(pressure_joint_audit_path),
        "parent_failed_pressure_heldout_sha256": _sha256(
            pressure_heldout_result_path
        ),
        "evaluation_cache_aggregate_sha256": cache_sha,
        "evaluation_cache_file_count": cache_files,
        "evaluation_cache_audit": cache_audit,
        "city_results": city_results,
        "diagnostic_gate": gate,
        "decision": SUCCESSOR_DECISION if gate["passed"] else FAILURE_DECISION,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--full-protocol", type=Path, required=True)
    parser.add_argument("--full-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-full-joint-audit-sha256", required=True)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-aligned-joint-audit-sha256", required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--pressure-heldout-result", type=Path, required=True)
    parser.add_argument("--expected-pressure-heldout-result-sha256", required=True)
    parser.add_argument("--expected-evaluation-cache-sha256", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite hierarchical diagnostic: {args.out}"
        )
    payload = run_hierarchical_guard_diagnostic(
        evaluation_cache_root=args.evaluation_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        full_protocol_path=args.full_protocol,
        full_joint_audit_path=args.full_joint_audit,
        expected_full_joint_audit_sha256=args.expected_full_joint_audit_sha256,
        aligned_protocol_path=args.aligned_protocol,
        aligned_joint_audit_path=args.aligned_joint_audit,
        expected_aligned_joint_audit_sha256=args.expected_aligned_joint_audit_sha256,
        aligned_freeze_root=args.aligned_freeze_root,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
        pressure_heldout_result_path=args.pressure_heldout_result,
        expected_pressure_heldout_result_sha256=args.expected_pressure_heldout_result_sha256,
        expected_evaluation_cache_sha256=args.expected_evaluation_cache_sha256,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "gate_passed": payload["diagnostic_gate"]["passed"],
                "decision": payload["decision"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
