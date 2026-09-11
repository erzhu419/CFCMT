"""Few-shot target-calibrated robust hierarchical mechanism priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_hierarchical_mechanism_prior import (
    AGGREGATE_PROTOCOL as V152A_AGGREGATE_PROTOCOL,
    PRIOR_PROTOCOL,
    PRIOR_STRENGTHS,
    V152A_REJECT_DECISION,
    _effect_summary,
    _fit_prior_split,
    _policy_values_for,
    _split_candidate_key,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    _arm_summary,
    _fold_groups,
    _read_json,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    _validate_authorization,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    TARGET_BUDGET,
    _prepare_inventory,
    _prepare_problem,
    _prepare_source_statistics_inventory,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v153a-target-calibrated-hierarchical-prior-target-v3"
AGGREGATE_PROTOCOL = (
    "tsc-v153a-target-calibrated-hierarchical-prior-aggregate-v3"
)
METHOD_PROTOCOL = "nested-b100-robust-hierarchical-mechanism-prior-v3"
SELECTOR_PROTOCOL = "five-fold-complete-selector-source-identity-audit-v3"
FEATURE_BINDING = "stored_feature_names"
V153A_REJECT_DECISION = (
    "retain_rigid_target_adaptation_and_reject_v153a_source_claim"
)
FOLD_COUNT = 5
MINIMUM_NONDEGRADING_FOLDS = 4
MINIMUM_INTERVENTION_GROUPS = 5


@dataclass(frozen=True)
class _TargetFold:
    fold_index: int
    heldout_groups: tuple[str, ...]
    rigid_values: np.ndarray
    source_values: Mapping[str, np.ndarray]
    placebo_values: Mapping[str, np.ndarray]
    zero_values: Mapping[str, np.ndarray]
    source_interventions: Mapping[str, int]


def _fold_prediction(
    problem: Any,
    fold: Mapping[str, Any],
) -> _TargetFold:
    prediction = _fit_prior_split(
        problem,
        training_groups=fold["training_groups"],
        evaluation_groups=fold["heldout_groups"],
    )
    return _TargetFold(
        fold_index=int(fold["fold_index"]),
        heldout_groups=tuple(str(value) for value in fold["heldout_groups"]),
        rigid_values=_policy_values_for(prediction, prediction.rigid_score),
        source_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.source_scores.items()
        },
        placebo_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.placebo_scores.items()
        },
        zero_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.zero_scores.items()
        },
        source_interventions=prediction.source_interventions,
    )


def _candidate_diagnostics(
    folds: Sequence[_TargetFold],
) -> dict[str, dict[str, Any]]:
    rows = tuple(folds)
    if not rows:
        raise ValueError("V153A candidate diagnostics require target folds")
    keys = set(rows[0].source_values)
    if not keys or any(
        set(row.source_values) != keys
        or set(row.placebo_values) != keys
        or set(row.zero_values) != keys
        for row in rows
    ):
        raise ValueError("V153A target-fold candidates differ")
    diagnostics: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        gains_rigid = np.asarray(
            [
                np.mean(row.rigid_values) - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        gains_placebo = np.asarray(
            [
                np.mean(row.placebo_values[key])
                - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        gains_zero = np.asarray(
            [
                np.mean(row.zero_values[key]) - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        means = {
            "gain_vs_rigid": float(np.mean(gains_rigid)),
            "gain_vs_matched_placebo": float(np.mean(gains_placebo)),
            "gain_vs_zero_centred_prior": float(np.mean(gains_zero)),
        }
        diagnostics[key] = {
            "means": means,
            "minimum_comparator_gain": min(means.values()),
            "nondegrading_fold_count_vs_rigid": int(
                np.count_nonzero(gains_rigid >= 0.0)
            ),
            "nondegrading_fold_count_vs_matched_placebo": int(
                np.count_nonzero(gains_placebo >= 0.0)
            ),
            "nondegrading_fold_count_vs_zero_centred_prior": int(
                np.count_nonzero(gains_zero >= 0.0)
            ),
            "intervention_group_count": int(
                sum(row.source_interventions[key] for row in rows)
            ),
            "fold_gains": [
                {
                    "fold_index": int(row.fold_index),
                    "gain_vs_rigid": float(gain_rigid),
                    "gain_vs_matched_placebo": float(gain_placebo),
                    "gain_vs_zero_centred_prior": float(gain_zero),
                    "intervention_group_count": int(
                        row.source_interventions[key]
                    ),
                }
                for row, gain_rigid, gain_placebo, gain_zero in zip(
                    rows,
                    gains_rigid,
                    gains_placebo,
                    gains_zero,
                    strict=True,
                )
            ],
        }
    return diagnostics


def _best_candidate(
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> str:
    if not diagnostics:
        raise ValueError("V153A selector has no candidates")
    return max(
        diagnostics,
        key=lambda key: (
            float(diagnostics[key]["minimum_comparator_gain"]),
            float(diagnostics[key]["means"]["gain_vs_rigid"]),
            str(key),
        ),
    )


def _selector_audit(
    folds: Sequence[_TargetFold],
    *,
    inner_folds: Mapping[int, Sequence[_TargetFold]],
) -> dict[str, Any]:
    rows = tuple(sorted(folds, key=lambda row: row.fold_index))
    if len(rows) != FOLD_COUNT or [row.fold_index for row in rows] != list(
        range(FOLD_COUNT)
    ):
        raise ValueError("V153A requires five ordered target OOF folds")
    full_diagnostics = _candidate_diagnostics(rows)
    selected = _best_candidate(full_diagnostics)
    outer_rows = []
    for heldout_index, heldout in enumerate(rows):
        training = tuple(inner_folds[heldout_index])
        training_diagnostics = _candidate_diagnostics(training)
        candidate = _best_candidate(training_diagnostics)
        heldout_diagnostics = _candidate_diagnostics((heldout,))[candidate]
        outer_rows.append(
            {
                "heldout_fold_index": heldout_index,
                "training_fold_indices": [
                    row.fold_index for row in training
                ],
                "outer_training_group_count": TARGET_BUDGET
                - len(heldout.heldout_groups),
                "inner_folds": [
                    {
                        "fold_index": row.fold_index,
                        "training_group_count": TARGET_BUDGET
                        - len(heldout.heldout_groups)
                        - len(row.heldout_groups),
                        "heldout_groups": list(row.heldout_groups),
                    }
                    for row in training
                ],
                "selected_candidate": candidate,
                "heldout_groups": list(heldout.heldout_groups),
                "heldout_means": heldout_diagnostics["means"],
                "heldout_intervention_group_count": int(
                    heldout.source_interventions[candidate]
                ),
            }
        )
    gain_names = (
        "gain_vs_rigid",
        "gain_vs_matched_placebo",
        "gain_vs_zero_centred_prior",
    )
    nested_means = {
        name: float(
            np.mean([row["heldout_means"][name] for row in outer_rows])
        )
        for name in gain_names
    }
    nested_nondegrading = {
        name: int(
            sum(row["heldout_means"][name] >= 0.0 for row in outer_rows)
        )
        for name in gain_names
    }
    nested_interventions = int(
        sum(row["heldout_intervention_group_count"] for row in outer_rows)
    )
    selected_diagnostics = full_diagnostics[selected]
    checks = {
        "full_oof_mean_gain_vs_rigid": selected_diagnostics["means"][
            "gain_vs_rigid"
        ]
        >= MINIMUM_OOF_GAIN,
        "full_oof_mean_gain_vs_matched_placebo": selected_diagnostics[
            "means"
        ]["gain_vs_matched_placebo"]
        >= MINIMUM_OOF_GAIN,
        "full_oof_mean_gain_vs_zero_centred_prior": selected_diagnostics[
            "means"
        ]["gain_vs_zero_centred_prior"]
        >= MINIMUM_OOF_GAIN,
        "nested_mean_gain_vs_rigid": nested_means["gain_vs_rigid"]
        >= MINIMUM_OOF_GAIN,
        "nested_mean_gain_vs_matched_placebo": nested_means[
            "gain_vs_matched_placebo"
        ]
        >= MINIMUM_OOF_GAIN,
        "nested_mean_gain_vs_zero_centred_prior": nested_means[
            "gain_vs_zero_centred_prior"
        ]
        >= MINIMUM_OOF_GAIN,
        "nested_nondegrading_folds_vs_rigid": nested_nondegrading[
            "gain_vs_rigid"
        ]
        >= MINIMUM_NONDEGRADING_FOLDS,
        "nested_nondegrading_folds_vs_matched_placebo": nested_nondegrading[
            "gain_vs_matched_placebo"
        ]
        >= MINIMUM_NONDEGRADING_FOLDS,
        "nested_nondegrading_folds_vs_zero_centred_prior": nested_nondegrading[
            "gain_vs_zero_centred_prior"
        ]
        >= MINIMUM_NONDEGRADING_FOLDS,
        "minimum_nested_intervention_groups": nested_interventions
        >= MINIMUM_INTERVENTION_GROUPS,
    }
    block, strength = _split_candidate_key(selected)
    return {
        "protocol": SELECTOR_PROTOCOL,
        "admitted": all(checks.values()),
        "candidate": selected,
        "mechanism_block": block,
        "prior_strength": strength,
        "selection_rule": "maximize_worst_comparator_mean_gain",
        "minimum_oof_gain": MINIMUM_OOF_GAIN,
        "minimum_nondegrading_folds": MINIMUM_NONDEGRADING_FOLDS,
        "minimum_intervention_groups": MINIMUM_INTERVENTION_GROUPS,
        "selected_full_oof_diagnostics": selected_diagnostics,
        "nested_selector_means": nested_means,
        "nested_selector_nondegrading_fold_counts": nested_nondegrading,
        "nested_selector_intervention_group_count": nested_interventions,
        "nested_selector_folds": outer_rows,
        "checks": checks,
        "candidate_diagnostics": full_diagnostics,
    }


def _fit_target_folds(
    problem: Any,
    *,
    workers: int,
) -> tuple[tuple[_TargetFold, ...], dict[int, tuple[_TargetFold, ...]]]:
    folds = _fold_groups(problem.selected_groups, fold_count=FOLD_COUNT)

    def build(
        fold: Mapping[str, Any],
    ) -> tuple[_TargetFold, tuple[_TargetFold, ...]]:
        outer_index = int(fold["fold_index"])
        outer_training = set(fold["training_groups"])
        inner_predictions = []
        for inner in folds:
            inner_index = int(inner["fold_index"])
            if inner_index == outer_index:
                continue
            inner_spec = {
                "fold_index": inner_index,
                "training_groups": tuple(
                    sorted(outer_training - set(inner["heldout_groups"]))
                ),
                "heldout_groups": inner["heldout_groups"],
            }
            inner_predictions.append(_fold_prediction(problem, inner_spec))
            print(
                "V153A_PROGRESS "
                f"target={problem.city} outer={outer_index} "
                f"inner={inner_index} training_groups="
                f"{len(inner_spec['training_groups'])}",
                flush=True,
            )
        result = _fold_prediction(problem, fold)
        print(
            "V153A_PROGRESS "
            f"target={problem.city} outer={result.fold_index} "
            f"heldout_groups={len(result.heldout_groups)}",
            flush=True,
        )
        return result, tuple(inner_predictions)

    maximum_workers = min(max(int(workers), 1), FOLD_COUNT)
    if maximum_workers == 1:
        rows = tuple(build(fold) for fold in folds)
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            rows = tuple(pool.map(build, folds))
    ordered = tuple(sorted(rows, key=lambda row: row[0].fold_index))
    return (
        tuple(row[0] for row in ordered),
        {row[0].fold_index: row[1] for row in ordered},
    )


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    v152a_rejection_path: Path,
    expected_v152a_rejection_sha256: str,
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
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V153A target city: {city}")
    authorization = _validate_authorization(
        authorization_result_path,
        expected_authorization_sha256,
    )
    if _sha256(v152a_rejection_path) != expected_v152a_rejection_sha256:
        raise ValueError("V152A rejection identity changed")
    rejection = _read_json(v152a_rejection_path)
    if (
        rejection.get("protocol") != V152A_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != V152A_REJECT_DECISION
    ):
        raise ValueError("V153A requires the frozen V152A rejection")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V153A source cache audit identity changed")
    from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
        _fit_result_contract,
    )

    fit_result = _fit_result_contract(
        fit_result_path,
        expected_sha256=expected_fit_result_sha256,
    )
    inventory = _prepare_inventory(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
    )
    inventory_seconds = time.monotonic() - started
    source_cities = tuple(
        value for value in EXPECTED_CITY_GROUPS if value != city
    )
    stage_started = time.monotonic()
    statistics_inventory = _prepare_source_statistics_inventory(
        inventory,
        fit_workers=fit_workers,
        stable_placebo_city_identity=True,
        cities=source_cities,
    )
    source_statistics_seconds = time.monotonic() - stage_started
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
        inventory=inventory,
        stable_placebo_city_identity=True,
        source_statistics_inventory=statistics_inventory,
    )

    stage_started = time.monotonic()
    folds, inner_folds = _fit_target_folds(problem, workers=fit_workers)
    selector = _selector_audit(folds, inner_folds=inner_folds)
    selector_seconds = time.monotonic() - stage_started

    stage_started = time.monotonic()
    prediction = _fit_prior_split(
        problem,
        training_groups=problem.selected_groups,
        evaluation_groups=problem.evaluation_groups,
    )
    target_seconds = time.monotonic() - stage_started
    key = str(selector["candidate"])
    admitted = bool(selector["admitted"])
    fallback = prediction.rigid_score.copy()
    forced_source = prediction.source_scores[key]
    forced_placebo = prediction.placebo_scores[key]
    forced_zero = prediction.zero_scores[key]
    arms = {
        "rigid_target_only": prediction.rigid_score,
        "source_null_exact_rigid": fallback,
        "forced_target_calibrated_source_prior": forced_source,
        "selected_target_calibrated_source_prior": (
            forced_source if admitted else fallback.copy()
        ),
        "forced_matched_placebo_prior": forced_placebo,
        "selected_matched_placebo_prior": (
            forced_placebo if admitted else fallback.copy()
        ),
        "forced_zero_centred_prior": forced_zero,
        "selected_zero_centred_prior": (
            forced_zero if admitted else fallback.copy()
        ),
    }
    evaluation = _group_subset_v3(
        problem.target_full,
        selected_groups=set(problem.evaluation_groups),
        metadata_updates={"target_data_role": "v153a_evaluation_only"},
    )
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=prediction.contrast,
            actual=prediction.actual,
            score=score,
            scenarios=problem.scenarios,
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries[
        "source_null_exact_rigid"
    ]:
        raise ValueError("V153A source-null differs from rigid CFCMT")
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "target-calibrated-hierarchical-prior-development-target",
        "target_city": city,
        "target_budget": TARGET_BUDGET,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "selector_protocol": SELECTOR_PROTOCOL,
            "feature_binding": FEATURE_BINDING,
            "prior_protocol": PRIOR_PROTOCOL,
            "prior_strength_grid": list(PRIOR_STRENGTHS),
            "selector": selector,
            "target_source_prior": prediction.source_prior.diagnostics,
            "target_placebo_prior": prediction.placebo_prior.diagnostics,
            "target_forced_intervention_group_count": int(
                prediction.source_interventions[key]
            ),
        },
        "information_budget": {
            "target_mechanism_adaptation_group_count": len(
                problem.selected_groups
            ),
            "target_selector_label_group_count": len(problem.selected_groups),
            "target_evaluation_label_count": 0,
            "target_evaluation_group_count": len(problem.evaluation_groups),
            "target_evaluation_excluded_from_selector": True,
            "source_city_count": len(source_cities),
            "stable_placebo_seed_by_source_city_identity": True,
        },
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "source_disabled_returns_rigid_score_bitwise": True,
            "summaries_equal": True,
        },
        "inputs": {
            "authorization_protocol": authorization["protocol"],
            "authorization_result_sha256": expected_authorization_sha256,
            "v152a_rejection_sha256": expected_v152a_rejection_sha256,
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "runtime_breakdown_seconds": {
            "shared_inventory": float(inventory_seconds),
            "shared_source_statistics": float(source_statistics_seconds),
            "parallel_target_oof_folds": float(selector_seconds),
            "target_fit": float(target_seconds),
        },
        "runtime_seconds": float(time.monotonic() - started),
        "claim_boundary": (
            "V153A is a B100 few-shot target-adaptation development test. "
            "Target evaluation groups are untouched, but target adaptation "
            "labels select the prior; this is not zero-shot or closed-loop "
            "confirmation."
        ),
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
            row.get("method", {}).get("protocol") != METHOD_PROTOCOL
            or row.get("method", {}).get("feature_binding") != FEATURE_BINDING
            for row in rows
        )
        or any(
            row.get("method", {}).get("selector_protocol")
            != SELECTOR_PROTOCOL
            for row in rows
        )
        or any(
            int(
                row.get("information_budget", {}).get(
                    "target_selector_label_group_count", -1
                )
            )
            != TARGET_BUDGET
            for row in rows
        )
        or any(
            int(
                row.get("information_budget", {}).get(
                    "target_evaluation_label_count", -1
                )
            )
            != 0
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V153A aggregate requires seven valid B100 target results")
    comparisons = {
        "selected_effect_vs_rigid_target_only": _effect_summary(
            by_city,
            "selected_target_calibrated_source_prior",
            "rigid_target_only",
        ),
        "selected_effect_vs_matched_placebo": _effect_summary(
            by_city,
            "selected_target_calibrated_source_prior",
            "selected_matched_placebo_prior",
        ),
        "selected_effect_vs_zero_centred_prior": _effect_summary(
            by_city,
            "selected_target_calibrated_source_prior",
            "selected_zero_centred_prior",
        ),
        "forced_effect_vs_rigid_target_only": _effect_summary(
            by_city,
            "forced_target_calibrated_source_prior",
            "rigid_target_only",
        ),
    }
    admitted = {
        city: bool(by_city[city]["method"]["selector"]["admitted"])
        for city in EXPECTED_CITY_GROUPS
    }
    admitted_cities = [city for city, value in admitted.items() if value]
    rigid_means = comparisons["selected_effect_vs_rigid_target_only"][
        "city_unit"
    ]["city_means"]
    placebo_means = comparisons["selected_effect_vs_matched_placebo"][
        "city_unit"
    ]["city_means"]
    zero_means = comparisons["selected_effect_vs_zero_centred_prior"][
        "city_unit"
    ]["city_means"]
    tolerance = 1e-12
    checks = {
        "mean_selected_effect_improves_rigid": comparisons[
            "selected_effect_vs_rigid_target_only"
        ]["city_unit"]["mean"]
        < 0.0,
        "mean_selected_effect_beats_matched_placebo": comparisons[
            "selected_effect_vs_matched_placebo"
        ]["city_unit"]["mean"]
        < 0.0,
        "mean_selected_effect_beats_zero_centred_prior": comparisons[
            "selected_effect_vs_zero_centred_prior"
        ]["city_unit"]["mean"]
        < 0.0,
        "no_city_regresses_rigid": max(rigid_means.values()) <= tolerance,
        "at_least_two_targets_admit_source_prior": len(admitted_cities) >= 2,
        "every_admitted_target_improves_rigid": all(
            rigid_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_matched_placebo": all(
            placebo_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_zero_centred_prior": all(
            zero_means[city] < 0.0 for city in admitted_cities
        ),
        "every_rejected_target_exactly_falls_back": all(
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
        "selector_protocol": SELECTOR_PROTOCOL,
        "feature_binding": FEATURE_BINDING,
        "prior_protocol": PRIOR_PROTOCOL,
        "source_admission_count": len(admitted_cities),
        "admitted_cities": admitted,
        **comparisons,
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": checks,
            "passed": passed,
            "decision": (
                "authorize_v153a_source_prior_closed_loop_development"
                if passed
                else V153A_REJECT_DECISION
            ),
        },
        "claim_boundary": (
            "V153A tests B100 target-calibrated source priors on seven "
            "development cities. Passing cannot establish closed-loop or "
            "untouched-city efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--v152a-rejection", type=Path, required=True)
    target.add_argument("--v152a-rejection-sha256", required=True)
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
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v152a_rejection_path=args.v152a_rejection,
            expected_v152a_rejection_sha256=args.v152a_rejection_sha256,
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
