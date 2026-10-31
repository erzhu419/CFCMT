"""Dense fixed-pair source utility with source-blind and placebo controls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_bytes,
    _sha256,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    RIDGE_L2,
    _paired_effects,
    _read_json,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    _reference_policy_score,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    _city_bootstrap,
    _validate_authorization,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    GATE_CONFIG,
    MINIMUM_INTERVENTION_GROUPS,
    MINIMUM_NONDEGRADING_FOLDS,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
    V150J_AGGREGATE_PROTOCOL,
    _arm_summary,
    _fit_split,
    _fold_groups,
    _policy_values_for,
    _prepare_problem,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    MECHANISM_ACTION_FEATURES,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES,
    DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
    UtilityRecords,
    apply_dense_pressure_pairwise_utility_gate,
    build_dense_pressure_pairwise_utility_records,
    concatenate_utility_records,
    fit_dense_pressure_pairwise_utility_gate,
    source_blind_candidate_scores,
)


RESULT_PROTOCOL = "tsc-v150o-dense-pressure-pairwise-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150o-dense-pressure-pairwise-source-utility-aggregate-v1"
METHOD_PROTOCOL = "nested-dense-phase-vs-rigid-source-evidence-utility-v1"
UTILITY_GATE_PROTOCOL = "ridge-upper-cost-bound-over-fixed-action-pair-records-v1"
RUNTIME_MODEL_PROTOCOL = (
    "tsc-v150o-dense-pressure-pairwise-source-runtime-model-v1"
)
V150N_AGGREGATE_PROTOCOL = (
    "tsc-v150n-pressure-pairwise-source-utility-aggregate-v1"
)
V150N_REJECT_DECISION = (
    "retain_rigid_target_adaptation_and_reject_v150n_source_claim"
)


def _records(
    prediction: Any,
    scores: Mapping[str, np.ndarray],
    *,
    fold_id: int,
) -> UtilityRecords:
    return build_dense_pressure_pairwise_utility_records(
        prediction.contrast,
        prediction.rigid_score,
        scores,
        actual=prediction.actual,
        fold_id=fold_id,
    )


def _blind_scores(prediction: Any) -> dict[str, np.ndarray]:
    return source_blind_candidate_scores(
        prediction.rigid_score,
        tuple(prediction.candidate_scores),
    )


def _nested_utility_audit(
    problem: Any,
    *,
    fit_workers: int,
) -> tuple[
    dict[str, Any],
    UtilityRecords,
    UtilityRecords,
    UtilityRecords,
]:
    folds = _fold_groups(problem.selected_groups)
    selected_set = set(problem.selected_groups)
    outer_rows: list[dict[str, Any]] = []
    rigid_values: list[float] = []
    source_values: list[float] = []
    placebo_values: list[float] = []
    blind_values: list[float] = []
    final_source_records: list[UtilityRecords] = []
    final_placebo_records: list[UtilityRecords] = []
    final_blind_records: list[UtilityRecords] = []

    for outer in folds:
        outer_index = int(outer["fold_index"])
        outer_heldout = tuple(str(value) for value in outer["heldout_groups"])
        outer_training = selected_set - set(outer_heldout)
        inner_specs = []
        for inner in folds:
            inner_index = int(inner["fold_index"])
            if inner_index == outer_index:
                continue
            inner_heldout = tuple(str(value) for value in inner["heldout_groups"])
            inner_training = outer_training - set(inner_heldout)
            inner_specs.append((inner_index, inner_training, inner_heldout))

        def fit_inner(
            spec: tuple[int, set[str], tuple[str, ...]],
        ) -> tuple[UtilityRecords, UtilityRecords, UtilityRecords]:
            inner_index, inner_training, inner_heldout = spec
            prediction = _fit_split(
                problem,
                training_groups=inner_training,
                evaluation_groups=inner_heldout,
                fit_workers=1,
            )
            return (
                _records(
                    prediction,
                    prediction.candidate_scores,
                    fold_id=inner_index,
                ),
                _records(
                    prediction,
                    prediction.placebo_scores,
                    fold_id=inner_index,
                ),
                _records(
                    prediction,
                    _blind_scores(prediction),
                    fold_id=inner_index,
                ),
            )

        with ThreadPoolExecutor(max_workers=len(inner_specs)) as pool:
            inner_fitted = list(pool.map(fit_inner, inner_specs))
        source_gate = fit_dense_pressure_pairwise_utility_gate(
            concatenate_utility_records([row[0] for row in inner_fitted]),
            config=GATE_CONFIG,
        )
        placebo_gate = fit_dense_pressure_pairwise_utility_gate(
            concatenate_utility_records([row[1] for row in inner_fitted]),
            config=GATE_CONFIG,
        )
        blind_gate = fit_dense_pressure_pairwise_utility_gate(
            concatenate_utility_records([row[2] for row in inner_fitted]),
            config=GATE_CONFIG,
        )
        print(
            "V150O_PROGRESS "
            f"city={problem.city} outer={outer_index + 1}/{len(folds)} "
            "phase=inner_complete",
            flush=True,
        )

        outer_prediction = _fit_split(
            problem,
            training_groups=outer_training,
            evaluation_groups=outer_heldout,
            fit_workers=fit_workers,
        )
        blind_candidates = _blind_scores(outer_prediction)
        final_source_records.append(
            _records(
                outer_prediction,
                outer_prediction.candidate_scores,
                fold_id=outer_index,
            )
        )
        final_placebo_records.append(
            _records(
                outer_prediction,
                outer_prediction.placebo_scores,
                fold_id=outer_index,
            )
        )
        final_blind_records.append(
            _records(
                outer_prediction,
                blind_candidates,
                fold_id=outer_index,
            )
        )
        source_score, source_apply = apply_dense_pressure_pairwise_utility_gate(
            outer_prediction.contrast,
            outer_prediction.rigid_score,
            outer_prediction.candidate_scores,
            source_gate,
            fold_id=outer_index,
        )
        placebo_score, placebo_apply = apply_dense_pressure_pairwise_utility_gate(
            outer_prediction.contrast,
            outer_prediction.rigid_score,
            outer_prediction.placebo_scores,
            placebo_gate,
            fold_id=outer_index,
        )
        blind_score, blind_apply = apply_dense_pressure_pairwise_utility_gate(
            outer_prediction.contrast,
            outer_prediction.rigid_score,
            blind_candidates,
            blind_gate,
            fold_id=outer_index,
        )
        fold_rigid = _policy_values_for(
            outer_prediction,
            outer_prediction.rigid_score,
        )
        fold_source = _policy_values_for(outer_prediction, source_score)
        fold_placebo = _policy_values_for(outer_prediction, placebo_score)
        fold_blind = _policy_values_for(outer_prediction, blind_score)
        rigid_values.extend(fold_rigid.tolist())
        source_values.extend(fold_source.tolist())
        placebo_values.extend(fold_placebo.tolist())
        blind_values.extend(fold_blind.tolist())
        outer_rows.append(
            {
                "outer_fold_index": outer_index,
                "outer_training_group_count": len(outer_training),
                "outer_heldout_group_count": len(outer_heldout),
                "inner_fold_count": len(folds) - 1,
                "gain_vs_rigid": float(np.mean(fold_rigid) - np.mean(fold_source)),
                "gain_vs_matched_placebo": float(
                    np.mean(fold_placebo) - np.mean(fold_source)
                ),
                "gain_vs_source_blind": float(
                    np.mean(fold_blind) - np.mean(fold_source)
                ),
                "source_gate": dict(source_gate.diagnostics),
                "matched_placebo_gate": dict(placebo_gate.diagnostics),
                "source_blind_gate": dict(blind_gate.diagnostics),
                "source_application": source_apply,
                "matched_placebo_application": placebo_apply,
                "source_blind_application": blind_apply,
            }
        )
        print(
            "V150O_PROGRESS "
            f"city={problem.city} outer={outer_index + 1}/{len(folds)} "
            f"phase=outer_complete interventions={source_apply['selected_group_count']}",
            flush=True,
        )

    target_gains = np.asarray(
        [row["gain_vs_rigid"] for row in outer_rows], dtype=float
    )
    placebo_gains = np.asarray(
        [row["gain_vs_matched_placebo"] for row in outer_rows], dtype=float
    )
    blind_gains = np.asarray(
        [row["gain_vs_source_blind"] for row in outer_rows], dtype=float
    )
    source_interventions = sum(
        int(row["source_application"]["selected_group_count"])
        for row in outer_rows
    )

    def nondegrading(values: np.ndarray) -> int:
        return int(np.count_nonzero(values >= 0.0))

    checks = {
        "mean_gain_vs_rigid_at_least_0_0005": float(np.mean(target_gains))
        >= MINIMUM_OOF_GAIN,
        "mean_gain_vs_matched_placebo_at_least_0_0005": float(
            np.mean(placebo_gains)
        )
        >= MINIMUM_OOF_GAIN,
        "mean_gain_vs_source_blind_at_least_0_0005": float(np.mean(blind_gains))
        >= MINIMUM_OOF_GAIN,
        "at_least_four_nondegrading_folds_vs_rigid": nondegrading(target_gains)
        >= MINIMUM_NONDEGRADING_FOLDS,
        "at_least_four_nondegrading_folds_vs_matched_placebo": nondegrading(
            placebo_gains
        )
        >= MINIMUM_NONDEGRADING_FOLDS,
        "at_least_four_nondegrading_folds_vs_source_blind": nondegrading(
            blind_gains
        )
        >= MINIMUM_NONDEGRADING_FOLDS,
        "at_least_five_crossfitted_intervention_groups": source_interventions
        >= MINIMUM_INTERVENTION_GROUPS,
    }
    return (
        {
            "fold_count": len(folds),
            "mean_gain_vs_rigid": float(np.mean(target_gains)),
            "mean_gain_vs_matched_placebo": float(np.mean(placebo_gains)),
            "mean_gain_vs_source_blind": float(np.mean(blind_gains)),
            "nondegrading_fold_count_vs_rigid": nondegrading(target_gains),
            "nondegrading_fold_count_vs_matched_placebo": nondegrading(
                placebo_gains
            ),
            "nondegrading_fold_count_vs_source_blind": nondegrading(blind_gains),
            "source_intervention_group_count": source_interventions,
            "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
            "checks": checks,
            "passed": all(checks.values()),
            "outer_folds": outer_rows,
            "rigid_policy_mean": float(np.mean(rigid_values)),
            "source_policy_mean": float(np.mean(source_values)),
            "matched_placebo_policy_mean": float(np.mean(placebo_values)),
            "source_blind_policy_mean": float(np.mean(blind_values)),
        },
        concatenate_utility_records(final_source_records),
        concatenate_utility_records(final_placebo_records),
        concatenate_utility_records(final_blind_records),
    )


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    v150j_rejection_path: Path,
    expected_v150j_rejection_sha256: str,
    v150n_rejection_path: Path,
    expected_v150n_rejection_sha256: str,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    expected_source_cache_audit_sha256: str,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int = 6,
    runtime_model_path: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V150O target city: {city}")
    authorization = _validate_authorization(
        authorization_result_path,
        expected_authorization_sha256,
    )
    if _sha256(v150j_rejection_path) != expected_v150j_rejection_sha256:
        raise ValueError("V150J rejection identity changed")
    v150j = _read_json(v150j_rejection_path)
    if (
        v150j.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or v150j.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V150O requires the frozen V150J rejection")
    if _sha256(v150n_rejection_path) != expected_v150n_rejection_sha256:
        raise ValueError("V150N rejection identity changed")
    v150n = _read_json(v150n_rejection_path)
    if (
        v150n.get("protocol") != V150N_AGGREGATE_PROTOCOL
        or v150n.get("development_gate", {}).get("passed") is not False
        or v150n.get("development_gate", {}).get("decision")
        != V150N_REJECT_DECISION
    ):
        raise ValueError("V150O requires the frozen V150N rejection")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V150O source cache audit identity changed")
    fit_result = _fit_result_contract(
        fit_result_path,
        expected_sha256=expected_fit_result_sha256,
    )
    problem = _prepare_problem(
        city=city,
        fit_result=fit_result,
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
        fit_workers=fit_workers,
    )
    nested, source_records, placebo_records, blind_records = _nested_utility_audit(
        problem,
        fit_workers=fit_workers,
    )
    source_gate = fit_dense_pressure_pairwise_utility_gate(
        source_records,
        config=GATE_CONFIG,
    )
    placebo_gate = fit_dense_pressure_pairwise_utility_gate(
        placebo_records,
        config=GATE_CONFIG,
    )
    blind_gate = fit_dense_pressure_pairwise_utility_gate(
        blind_records,
        config=GATE_CONFIG,
    )
    final_prediction = _fit_split(
        problem,
        training_groups=problem.selected_groups,
        evaluation_groups=problem.evaluation_groups,
        fit_workers=fit_workers,
    )
    blind_candidates = _blind_scores(final_prediction)
    forced_source, source_apply = apply_dense_pressure_pairwise_utility_gate(
        final_prediction.contrast,
        final_prediction.rigid_score,
        final_prediction.candidate_scores,
        source_gate,
    )
    forced_placebo, placebo_apply = apply_dense_pressure_pairwise_utility_gate(
        final_prediction.contrast,
        final_prediction.rigid_score,
        final_prediction.placebo_scores,
        placebo_gate,
    )
    forced_blind, blind_apply = apply_dense_pressure_pairwise_utility_gate(
        final_prediction.contrast,
        final_prediction.rigid_score,
        blind_candidates,
        blind_gate,
    )
    admitted = bool(nested["passed"])
    fallback = final_prediction.rigid_score.copy()
    selected_source = forced_source if admitted else fallback.copy()
    selected_placebo = forced_placebo if admitted else fallback.copy()
    selected_blind = forced_blind if admitted else fallback.copy()
    phase_score = _reference_policy_score(final_prediction.references)
    arms = {
        "phase_pressure": phase_score,
        "rigid_target_only": final_prediction.rigid_score,
        "source_null_exact_rigid": fallback,
        "forced_dense_source_pairwise": forced_source,
        "selected_dense_source_pairwise": selected_source,
        "forced_dense_matched_placebo": forced_placebo,
        "selected_dense_matched_placebo": selected_placebo,
        "forced_dense_source_blind": forced_blind,
        "selected_dense_source_blind": selected_blind,
    }
    evaluation = _group_subset_v3(
        problem.target_full,
        selected_groups=set(problem.evaluation_groups),
        metadata_updates={"target_data_role": "v150o_outside_b100_evaluation_only"},
    )
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=final_prediction.contrast,
            actual=final_prediction.actual,
            score=score,
            scenarios=problem.scenarios,
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries[
        "source_null_exact_rigid"
    ]:
        raise ValueError("V150O source-null summary differs from rigid CFCMT")

    selector = {
        "candidate": "dense_fixed_phase_vs_rigid_source_mechanism_pool",
        "source_city": "selected_per_state_without_source_identity_feature",
        "mechanism_block": "selected_per_state",
        "admitted": admitted,
        "crossfitted_selection_audit": nested,
        "final_source_gate": dict(source_gate.diagnostics),
        "final_source_application": source_apply,
        "final_matched_placebo_gate": dict(placebo_gate.diagnostics),
        "final_matched_placebo_application": placebo_apply,
        "final_source_blind_gate": dict(blind_gate.diagnostics),
        "final_source_blind_application": blind_apply,
        "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
    }
    result = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "seven-city-dense-pairwise-source-development-target",
        "target_city": city,
        "target_scenarios": list(problem.scenarios),
        "source_city_groups": list(problem.source_order),
        "target_budget": TARGET_BUDGET,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": problem.target_name,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
            "utility_feature_names": list(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES),
            "ridge_l2": RIDGE_L2,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "candidate_count": len(problem.source_order)
            * len(MECHANISM_ACTION_FEATURES),
            "utility_gate_config": {
                "ridge_alpha": GATE_CONFIG.ridge_alpha,
                "error_quantile": GATE_CONFIG.error_quantile,
                "minimum_records": GATE_CONFIG.minimum_records,
                "minimum_predicted_gain": GATE_CONFIG.minimum_predicted_gain,
            },
            "selector": selector,
        },
        "information_budget": {
            "target_adaptation_group_count": len(problem.selected_groups),
            "common_evaluation_reserve_group_count": len(problem.reserve_groups),
            "target_evaluation_group_count": len(problem.evaluation_groups),
            "target_labels_used_for_nested_fixed_pair_utility_fit": True,
            "evaluation_features_or_labels_used_for_fit_or_selection": False,
            "source_labels_used_only_for_mechanism_coefficient_priors": True,
            "source_identity_used_as_utility_feature": False,
            "source_blind_control_uses_same_target_labels_and_state_features": True,
        },
        "selection_audit": problem.selection_audit,
        "evaluation_reserve_audit": problem.reserve_audit,
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "source_disabled_returns_rigid_score_bitwise": True,
            "rigid_scores_bitwise_equal": True,
            "summaries_equal": True,
        },
        "input_audits": {"source_bank": problem.source_bank_audit},
        "inputs": {
            "authorization_result_sha256": expected_authorization_sha256,
            "authorization_protocol": authorization["protocol"],
            "v150j_rejection_sha256": expected_v150j_rejection_sha256,
            "v150j_rejection_protocol": v150j["protocol"],
            "v150n_rejection_sha256": expected_v150n_rejection_sha256,
            "v150n_rejection_protocol": v150n["protocol"],
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "claim_boundary": (
            "V150O is a sequential seven-city B100 target-offline adaptation "
            "experiment. It compares real source evidence with rigid CFCMT, a "
            "separately nested matched placebo and a same-capacity source-blind "
            "utility gate. It is not closed-loop or fresh-city confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }
    if runtime_model_path is not None:
        runtime_payload = {
            "protocol": RUNTIME_MODEL_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": city,
            "target_budget": TARGET_BUDGET,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "admitted": admitted,
            "selected_groups": problem.selected_groups,
            "source_order": problem.source_order,
            "frozen_candidate": problem.frozen_candidate,
            "mechanism_feature_names": problem.mechanism_feature_names,
            "feature_scale": final_prediction.feature_scale,
            "target_beta": final_prediction.target_beta,
            "candidate_betas": dict(final_prediction.candidate_betas),
            "placebo_betas": dict(final_prediction.placebo_betas),
            "rigid_model": final_prediction.rigid_model,
            "source_utility_gate": source_gate,
            "placebo_utility_gate": placebo_gate,
            "source_blind_utility_gate": blind_gate,
            "nested_selection_audit": nested,
            "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
            "fit_contract": {
                "authorization_result_sha256": expected_authorization_sha256,
                "v150j_rejection_sha256": expected_v150j_rejection_sha256,
                "v150n_rejection_sha256": expected_v150n_rejection_sha256,
                "fit_result_sha256": expected_fit_result_sha256,
                "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            },
        }
        _atomic_bytes(
            Path(runtime_model_path),
            pickle.dumps(runtime_payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        result["runtime_model"] = {
            "protocol": RUNTIME_MODEL_PROTOCOL,
            "path": str(Path(runtime_model_path)),
            "sha256": _sha256(Path(runtime_model_path)),
            "size_bytes": int(Path(runtime_model_path).stat().st_size),
            "stored_remote_only": True,
        }
    return result


def _effect_summary(
    by_city: Mapping[str, Mapping[str, Any]],
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    effects, city_means = _paired_effects(by_city, candidate, reference)
    return {
        "seed_unit": _paired_bootstrap(effects),
        "city_unit": {
            **_city_bootstrap(city_means),
            "city_means": city_means,
        },
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
        or any(int(row.get("target_budget", -1)) != TARGET_BUDGET for row in rows)
        or any(
            row.get("method", {}).get("utility_gate_protocol")
            != UTILITY_GATE_PROTOCOL
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V150O aggregate requires seven valid B100 results")

    vs_rigid = _effect_summary(
        by_city,
        "selected_dense_source_pairwise",
        "rigid_target_only",
    )
    vs_placebo = _effect_summary(
        by_city,
        "selected_dense_source_pairwise",
        "selected_dense_matched_placebo",
    )
    vs_blind = _effect_summary(
        by_city,
        "selected_dense_source_pairwise",
        "selected_dense_source_blind",
    )
    admitted = {
        city: bool(by_city[city]["method"]["selector"]["admitted"])
        for city in EXPECTED_CITY_GROUPS
    }
    admitted_cities = [city for city, value in admitted.items() if value]
    rigid_means = vs_rigid["city_unit"]["city_means"]
    placebo_means = vs_placebo["city_unit"]["city_means"]
    blind_means = vs_blind["city_unit"]["city_means"]
    tolerance = 1e-12
    checks = {
        "mean_selected_effect_improves_rigid": vs_rigid["city_unit"]["mean"] < 0.0,
        "mean_selected_effect_beats_matched_placebo": vs_placebo["city_unit"][
            "mean"
        ]
        < 0.0,
        "mean_selected_effect_beats_source_blind": vs_blind["city_unit"]["mean"]
        < 0.0,
        "no_city_regresses_rigid": max(rigid_means.values()) <= tolerance,
        "at_least_two_cities_admit_crossfitted_source": len(admitted_cities) >= 2,
        "every_admitted_city_improves_rigid": all(
            rigid_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_city_beats_matched_placebo": all(
            placebo_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_city_beats_source_blind": all(
            blind_means[city] < 0.0 for city in admitted_cities
        ),
        "every_rejected_city_exactly_falls_back": all(
            abs(rigid_means[city]) <= tolerance
            for city in EXPECTED_CITY_GROUPS
            if not admitted[city]
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_budget": TARGET_BUDGET,
        "source_prior_strength": SOURCE_PRIOR_STRENGTH,
        "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
        "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
        "source_admission_count": len(admitted_cities),
        "admitted_cities": admitted,
        "selected_effect_vs_rigid_target_only": vs_rigid,
        "selected_effect_vs_matched_placebo": vs_placebo,
        "selected_effect_vs_source_blind": vs_blind,
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": checks,
            "passed": passed,
            "decision": (
                "authorize_v150o_dense_pairwise_closed_loop_development"
                if passed
                else "retain_rigid_target_adaptation_and_reject_v150o_source_claim"
            ),
        },
        "claim_boundary": (
            "V150O tests whether dense fixed-pair source evidence adds value "
            "beyond a same-capacity source-blind gate and matched placebo on "
            "seven development cities. Passing cannot establish closed-loop or "
            "fresh-city efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--v150j-rejection", type=Path, required=True)
    target.add_argument("--v150j-rejection-sha256", required=True)
    target.add_argument("--v150n-rejection", type=Path, required=True)
    target.add_argument("--v150n-rejection-sha256", required=True)
    target.add_argument("--fit-result", type=Path, required=True)
    target.add_argument("--fit-result-sha256", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--fit-workers", type=int, default=6)
    target.add_argument("--runtime-model-out", type=Path)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150O result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v150j_rejection_path=args.v150j_rejection,
            expected_v150j_rejection_sha256=args.v150j_rejection_sha256,
            v150n_rejection_path=args.v150n_rejection,
            expected_v150n_rejection_sha256=args.v150n_rejection_sha256,
            fit_result_path=args.fit_result,
            expected_fit_result_sha256=args.fit_result_sha256,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
            fit_workers=args.fit_workers,
            runtime_model_path=args.runtime_model_out,
        )
        status = "DONE"
    else:
        result = aggregate_results([_read_json(path) for path in args.results])
        status = "PASS" if result["development_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
