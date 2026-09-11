"""Leave-target-city-out meta-utility for source mechanism evidence."""

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
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    _paired_effects,
    _read_json,
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
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
    V150J_AGGREGATE_PROTOCOL,
    _arm_summary,
    _fit_split,
    _policy_values_for,
    _prepare_inventory,
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


RESULT_PROTOCOL = "tsc-v151a-cross-city-meta-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v151a-cross-city-meta-source-utility-aggregate-v1"
METHOD_PROTOCOL = "leave-target-city-out-dense-source-meta-utility-v1"
UTILITY_GATE_PROTOCOL = "equal-city-ridge-upper-cost-bound-v1"
V150O_AGGREGATE_PROTOCOL = (
    "tsc-v150o-dense-pressure-pairwise-source-utility-aggregate-v1"
)
V150O_REJECT_DECISION = (
    "retain_rigid_target_adaptation_and_reject_v150o_source_claim"
)
META_NONDEGRADING_CITY_COUNT = 5
META_MINIMUM_INTERVENTION_GROUPS = 6


@dataclass(frozen=True)
class _MetaDomain:
    city: str
    prediction: Any
    source_records: UtilityRecords
    placebo_records: UtilityRecords
    blind_records: UtilityRecords
    source_order: tuple[str, ...]


def _records(
    prediction: Any,
    scores: Mapping[str, np.ndarray],
    *,
    city: str,
    fold_id: int,
) -> UtilityRecords:
    records = build_dense_pressure_pairwise_utility_records(
        prediction.contrast,
        prediction.rigid_score,
        scores,
        actual=prediction.actual,
        fold_id=fold_id,
    )
    return UtilityRecords(
        features=records.features,
        cost_delta=records.cost_delta,
        group_ids=np.asarray(
            [f"{city}::{value}" for value in records.group_ids],
            dtype=str,
        ),
        candidate_keys=records.candidate_keys,
        fold_ids=np.full(records.features.shape[0], int(fold_id), dtype=int),
    )


def _blind_scores(prediction: Any) -> dict[str, np.ndarray]:
    return source_blind_candidate_scores(
        prediction.rigid_score,
        tuple(prediction.candidate_scores),
    )


def _fit_meta_gate(records: Sequence[UtilityRecords]):
    return fit_dense_pressure_pairwise_utility_gate(
        concatenate_utility_records(records),
        config=GATE_CONFIG,
        balance_folds=True,
    )


def _meta_source_audit(
    domains: Sequence[_MetaDomain], *, workers: int = 1
) -> dict[str, Any]:
    rows = tuple(domains)
    if len(rows) != len(EXPECTED_CITY_GROUPS) - 1:
        raise ValueError("V151A requires six source pseudo-target domains")

    def fit_fold(item: tuple[int, _MetaDomain]) -> dict[str, Any]:
        heldout_index, heldout = item
        training = tuple(row for row in rows if row.city != heldout.city)
        if len(training) != len(rows) - 1:
            raise ValueError("V151A meta fold city identities are not unique")
        source_gate = _fit_meta_gate([row.source_records for row in training])
        placebo_gate = _fit_meta_gate([row.placebo_records for row in training])
        blind_gate = _fit_meta_gate([row.blind_records for row in training])
        source_score, source_apply = apply_dense_pressure_pairwise_utility_gate(
            heldout.prediction.contrast,
            heldout.prediction.rigid_score,
            heldout.prediction.candidate_scores,
            source_gate,
            fold_id=heldout_index,
        )
        placebo_score, placebo_apply = apply_dense_pressure_pairwise_utility_gate(
            heldout.prediction.contrast,
            heldout.prediction.rigid_score,
            heldout.prediction.placebo_scores,
            placebo_gate,
            fold_id=heldout_index,
        )
        blind_score, blind_apply = apply_dense_pressure_pairwise_utility_gate(
            heldout.prediction.contrast,
            heldout.prediction.rigid_score,
            _blind_scores(heldout.prediction),
            blind_gate,
            fold_id=heldout_index,
        )
        rigid_values = _policy_values_for(
            heldout.prediction,
            heldout.prediction.rigid_score,
        )
        source_values = _policy_values_for(heldout.prediction, source_score)
        placebo_values = _policy_values_for(heldout.prediction, placebo_score)
        blind_values = _policy_values_for(heldout.prediction, blind_score)
        result = {
            "heldout_utility_city": heldout.city,
            "training_utility_cities": [row.city for row in training],
            "gain_vs_rigid": float(
                np.mean(rigid_values) - np.mean(source_values)
            ),
            "gain_vs_matched_placebo": float(
                np.mean(placebo_values) - np.mean(source_values)
            ),
            "gain_vs_source_blind": float(
                np.mean(blind_values) - np.mean(source_values)
            ),
            "source_application": source_apply,
            "matched_placebo_application": placebo_apply,
            "source_blind_application": blind_apply,
            "source_gate": dict(source_gate.diagnostics),
            "matched_placebo_gate": dict(placebo_gate.diagnostics),
            "source_blind_gate": dict(blind_gate.diagnostics),
        }
        print(
            "V151A_PROGRESS "
            f"phase=meta_oof heldout={heldout.city} "
            f"interventions={source_apply['selected_group_count']}",
            flush=True,
        )
        return result

    maximum_workers = min(max(int(workers), 1), len(rows))
    if maximum_workers == 1:
        fold_rows = [fit_fold(item) for item in enumerate(rows)]
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            fold_rows = list(pool.map(fit_fold, enumerate(rows)))

    rigid_gains = np.asarray([row["gain_vs_rigid"] for row in fold_rows])
    placebo_gains = np.asarray(
        [row["gain_vs_matched_placebo"] for row in fold_rows]
    )
    blind_gains = np.asarray(
        [row["gain_vs_source_blind"] for row in fold_rows]
    )
    interventions = sum(
        int(row["source_application"]["selected_group_count"])
        for row in fold_rows
    )

    def nondegrading(values: np.ndarray) -> int:
        return int(np.count_nonzero(values >= 0.0))

    checks = {
        "mean_gain_vs_rigid_at_least_0_0005": float(np.mean(rigid_gains))
        >= MINIMUM_OOF_GAIN,
        "mean_gain_vs_matched_placebo_at_least_0_0005": float(
            np.mean(placebo_gains)
        )
        >= MINIMUM_OOF_GAIN,
        "mean_gain_vs_source_blind_at_least_0_0005": float(np.mean(blind_gains))
        >= MINIMUM_OOF_GAIN,
        "at_least_five_of_six_nondegrading_vs_rigid": nondegrading(rigid_gains)
        >= META_NONDEGRADING_CITY_COUNT,
        "at_least_five_of_six_nondegrading_vs_matched_placebo": nondegrading(
            placebo_gains
        )
        >= META_NONDEGRADING_CITY_COUNT,
        "at_least_five_of_six_nondegrading_vs_source_blind": nondegrading(
            blind_gains
        )
        >= META_NONDEGRADING_CITY_COUNT,
        "at_least_six_meta_oof_intervention_groups": interventions
        >= META_MINIMUM_INTERVENTION_GROUPS,
    }
    return {
        "fold_count": len(fold_rows),
        "mean_gain_vs_rigid": float(np.mean(rigid_gains)),
        "mean_gain_vs_matched_placebo": float(np.mean(placebo_gains)),
        "mean_gain_vs_source_blind": float(np.mean(blind_gains)),
        "nondegrading_city_count_vs_rigid": nondegrading(rigid_gains),
        "nondegrading_city_count_vs_matched_placebo": nondegrading(placebo_gains),
        "nondegrading_city_count_vs_source_blind": nondegrading(blind_gains),
        "source_intervention_group_count": interventions,
        "checks": checks,
        "passed": all(checks.values()),
        "folds": fold_rows,
    }


def _build_meta_domains(
    *,
    excluded_target_city: str,
    fit_result: Mapping[str, Any],
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int,
    inventory: Any,
) -> tuple[_MetaDomain, ...]:
    meta_cities = tuple(
        city for city in EXPECTED_CITY_GROUPS if city != excluded_target_city
    )

    def build_domain(item: tuple[int, str]) -> _MetaDomain:
        fold_id, city = item
        problem = _prepare_problem(
            city=city,
            fit_result=fit_result,
            fit_protocol_path=fit_protocol_path,
            source_cache_root=source_cache_root,
            source_manifest_path=source_manifest_path,
            source_cache_audit_path=source_cache_audit_path,
            conversion_root=conversion_root,
            cache_workers=cache_workers,
            fit_workers=1,
            excluded_source_cities=(excluded_target_city,),
            inventory=inventory,
        )
        if excluded_target_city in problem.source_order:
            raise RuntimeError("held-out target leaked into meta source order")
        prediction = _fit_split(
            problem,
            training_groups=problem.selected_groups,
            evaluation_groups=problem.evaluation_groups,
            fit_workers=1,
        )
        forbidden_prefix = f"{excluded_target_city}|"
        if any(
            str(key).startswith(forbidden_prefix)
            for key in (
                *tuple(prediction.candidate_scores),
                *tuple(prediction.placebo_scores),
            )
        ):
            raise RuntimeError("held-out target leaked into meta candidate scores")
        domain = _MetaDomain(
            city=city,
            prediction=prediction,
            source_records=_records(
                prediction,
                prediction.candidate_scores,
                city=city,
                fold_id=fold_id,
            ),
            placebo_records=_records(
                prediction,
                prediction.placebo_scores,
                city=city,
                fold_id=fold_id,
            ),
            blind_records=_records(
                prediction,
                _blind_scores(prediction),
                city=city,
                fold_id=fold_id,
            ),
            source_order=problem.source_order,
        )
        print(
            "V151A_PROGRESS "
            f"target={excluded_target_city} phase=meta_records "
            f"pseudo_target={city} {fold_id + 1}/{len(meta_cities)}",
            flush=True,
        )
        return domain

    maximum_workers = min(max(int(fit_workers), 1), len(meta_cities))
    if maximum_workers == 1:
        domains = [build_domain(item) for item in enumerate(meta_cities)]
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            domains = list(pool.map(build_domain, enumerate(meta_cities)))
    return tuple(domains)


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    v150j_rejection_path: Path,
    expected_v150j_rejection_sha256: str,
    v150o_rejection_path: Path,
    expected_v150o_rejection_sha256: str,
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
    stage_started = started
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V151A target city: {city}")
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
        raise ValueError("V151A requires the frozen V150J rejection")
    if _sha256(v150o_rejection_path) != expected_v150o_rejection_sha256:
        raise ValueError("V150O rejection identity changed")
    v150o = _read_json(v150o_rejection_path)
    if (
        v150o.get("protocol") != V150O_AGGREGATE_PROTOCOL
        or v150o.get("development_gate", {}).get("passed") is not False
        or v150o.get("development_gate", {}).get("decision")
        != V150O_REJECT_DECISION
    ):
        raise ValueError("V151A requires the frozen V150O rejection")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V151A source cache audit identity changed")
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
    inventory_seconds = time.monotonic() - stage_started
    stage_started = time.monotonic()

    meta_domains = _build_meta_domains(
        excluded_target_city=city,
        fit_result=fit_result,
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
        fit_workers=fit_workers,
        inventory=inventory,
    )
    meta_domain_seconds = time.monotonic() - stage_started
    stage_started = time.monotonic()
    meta_audit = _meta_source_audit(meta_domains, workers=fit_workers)
    meta_audit_seconds = time.monotonic() - stage_started
    stage_started = time.monotonic()
    gate_record_sets = (
        [row.source_records for row in meta_domains],
        [row.placebo_records for row in meta_domains],
        [row.blind_records for row in meta_domains],
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        source_gate, placebo_gate, blind_gate = tuple(
            pool.map(_fit_meta_gate, gate_record_sets)
        )
    final_gate_seconds = time.monotonic() - stage_started
    stage_started = time.monotonic()

    target_problem = _prepare_problem(
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
    )
    target_prediction = _fit_split(
        target_problem,
        training_groups=target_problem.selected_groups,
        evaluation_groups=target_problem.evaluation_groups,
        fit_workers=fit_workers,
    )
    target_fit_seconds = time.monotonic() - stage_started
    forced_source, source_apply = apply_dense_pressure_pairwise_utility_gate(
        target_prediction.contrast,
        target_prediction.rigid_score,
        target_prediction.candidate_scores,
        source_gate,
    )
    forced_placebo, placebo_apply = apply_dense_pressure_pairwise_utility_gate(
        target_prediction.contrast,
        target_prediction.rigid_score,
        target_prediction.placebo_scores,
        placebo_gate,
    )
    forced_blind, blind_apply = apply_dense_pressure_pairwise_utility_gate(
        target_prediction.contrast,
        target_prediction.rigid_score,
        _blind_scores(target_prediction),
        blind_gate,
    )
    admitted = bool(meta_audit["passed"])
    fallback = target_prediction.rigid_score.copy()
    arms = {
        "rigid_target_only": target_prediction.rigid_score,
        "source_null_exact_rigid": fallback,
        "forced_meta_source_pairwise": forced_source,
        "selected_meta_source_pairwise": forced_source if admitted else fallback.copy(),
        "forced_meta_matched_placebo": forced_placebo,
        "selected_meta_matched_placebo": (
            forced_placebo if admitted else fallback.copy()
        ),
        "forced_meta_source_blind": forced_blind,
        "selected_meta_source_blind": forced_blind if admitted else fallback.copy(),
    }
    evaluation = _group_subset_v3(
        target_problem.target_full,
        selected_groups=set(target_problem.evaluation_groups),
        metadata_updates={"target_data_role": "v151a_evaluation_only"},
    )
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=target_prediction.contrast,
            actual=target_prediction.actual,
            score=score,
            scenarios=target_problem.scenarios,
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries[
        "source_null_exact_rigid"
    ]:
        raise ValueError("V151A exact fallback differs from rigid CFCMT")

    selector = {
        "admitted": admitted,
        "admission_uses_target_utility_labels": False,
        "meta_source_audit": meta_audit,
        "source_utility_training_cities": [row.city for row in meta_domains],
        "source_candidate_cities_by_pseudo_target": {
            row.city: list(row.source_order) for row in meta_domains
        },
        "final_source_gate": dict(source_gate.diagnostics),
        "final_matched_placebo_gate": dict(placebo_gate.diagnostics),
        "final_source_blind_gate": dict(blind_gate.diagnostics),
        "target_source_application": source_apply,
        "target_matched_placebo_application": placebo_apply,
        "target_source_blind_application": blind_apply,
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "leave-target-city-out-meta-utility-development-target",
        "target_city": city,
        "target_budget": TARGET_BUDGET,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "utility_record_protocol": DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
            "utility_feature_names": list(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES),
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "pseudo_target_candidate_count": (
                (len(EXPECTED_CITY_GROUPS) - 2) * len(MECHANISM_ACTION_FEATURES)
            ),
            "target_candidate_count": (
                (len(EXPECTED_CITY_GROUPS) - 1) * len(MECHANISM_ACTION_FEATURES)
            ),
            "selector": selector,
        },
        "information_budget": {
            "target_mechanism_adaptation_group_count": len(
                target_problem.selected_groups
            ),
            "target_utility_label_count": 0,
            "target_evaluation_group_count": len(target_problem.evaluation_groups),
            "target_excluded_from_meta_utility_labels": True,
            "target_excluded_from_meta_source_candidates": True,
            "target_excluded_from_meta_feature_scaling_statistics": True,
            "source_city_utility_labels_available": True,
        },
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "source_disabled_returns_rigid_score_bitwise": True,
            "summaries_equal": True,
        },
        "inputs": {
            "authorization_protocol": authorization["protocol"],
            "authorization_result_sha256": expected_authorization_sha256,
            "v150j_rejection_sha256": expected_v150j_rejection_sha256,
            "v150o_rejection_sha256": expected_v150o_rejection_sha256,
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "claim_boundary": (
            "V151A is B100 target mechanism adaptation with a utility relation "
            "trained only on the other six cities. It is a seven-city sequential "
            "development test, not closed-loop or untouched-city confirmation."
        ),
        "runtime_breakdown_seconds": {
            "shared_inventory": float(inventory_seconds),
            "parallel_meta_domains": float(meta_domain_seconds),
            "parallel_meta_oof_audit": float(meta_audit_seconds),
            "parallel_final_gates": float(final_gate_seconds),
            "target_fit": float(target_fit_seconds),
        },
        "runtime_seconds": float(time.monotonic() - started),
    }


def _effect_summary(
    by_city: Mapping[str, Mapping[str, Any]],
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    effects, city_means = _paired_effects(by_city, candidate, reference)
    return {
        "seed_unit": _paired_bootstrap(effects),
        "city_unit": {**_city_bootstrap(city_means), "city_means": city_means},
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
            int(row.get("information_budget", {}).get("target_utility_label_count", -1))
            != 0
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V151A aggregate requires seven valid B100 results")

    comparisons = {
        "selected_effect_vs_rigid_target_only": _effect_summary(
            by_city, "selected_meta_source_pairwise", "rigid_target_only"
        ),
        "selected_effect_vs_matched_placebo": _effect_summary(
            by_city,
            "selected_meta_source_pairwise",
            "selected_meta_matched_placebo",
        ),
        "selected_effect_vs_source_blind": _effect_summary(
            by_city,
            "selected_meta_source_pairwise",
            "selected_meta_source_blind",
        ),
    }
    admitted = {
        city: bool(by_city[city]["method"]["selector"]["admitted"])
        for city in EXPECTED_CITY_GROUPS
    }
    admitted_cities = [city for city, value in admitted.items() if value]
    rigid_means = comparisons["selected_effect_vs_rigid_target_only"]["city_unit"][
        "city_means"
    ]
    placebo_means = comparisons["selected_effect_vs_matched_placebo"]["city_unit"][
        "city_means"
    ]
    blind_means = comparisons["selected_effect_vs_source_blind"]["city_unit"][
        "city_means"
    ]
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
        "mean_selected_effect_beats_source_blind": comparisons[
            "selected_effect_vs_source_blind"
        ]["city_unit"]["mean"]
        < 0.0,
        "no_city_regresses_rigid": max(rigid_means.values()) <= tolerance,
        "at_least_two_targets_admit_source_only_meta_gate": len(admitted_cities) >= 2,
        "every_admitted_target_improves_rigid": all(
            rigid_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_matched_placebo": all(
            placebo_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_source_blind": all(
            blind_means[city] < 0.0 for city in admitted_cities
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
        "source_prior_strength": SOURCE_PRIOR_STRENGTH,
        "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
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
                "authorize_v151a_source_meta_utility_closed_loop_development"
                if passed
                else "retain_rigid_target_adaptation_and_reject_v151a_source_claim"
            ),
        },
        "claim_boundary": (
            "V151A tests whether a utility relation trained without the target "
            "city adds source-specific value at B100. Passing cannot establish "
            "closed-loop or untouched-city efficacy."
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
    target.add_argument("--v150o-rejection", type=Path, required=True)
    target.add_argument("--v150o-rejection-sha256", required=True)
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
        raise FileExistsError(f"refusing to overwrite V151A result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v150j_rejection_path=args.v150j_rejection,
            expected_v150j_rejection_sha256=args.v150j_rejection_sha256,
            v150o_rejection_path=args.v150o_rejection,
            expected_v150o_rejection_sha256=args.v150o_rejection_sha256,
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
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
