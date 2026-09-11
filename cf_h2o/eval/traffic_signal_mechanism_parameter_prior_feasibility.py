"""Evaluate mechanism-block source priors with target-only exact fallback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic import (
    _policy_metrics,
    _predictive_metrics,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
    TARGET_BUDGET,
    _reference_policy_score,
    _scenario_seed_policy_summary,
    select_city_budget_groups,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_right_of_way_signature import _read_manifest
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
    _policy_arrays,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    _source_inputs,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    MECHANISM_ACTION_FEATURES,
    equal_city_rms_scale,
    mechanism_parameter_design,
    permute_group_targets,
    ridge_coefficients,
)
from cf_h2o.traffic_signal.right_of_way_context import (
    RIGHT_OF_WAY_ACTION_FEATURES,
    augment_dataset_with_right_of_way_context,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)


RESULT_PROTOCOL = "tsc-v150c-mechanism-parameter-prior-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150c-mechanism-parameter-prior-aggregate-v1"
FOLD_COUNT = 5
RIDGE_L2 = 0.05
SOURCE_PRIOR_STRENGTH = 0.10
MINIMUM_OOF_GAIN = 0.0005
PLACEBO_SEED = 20260908
CONTRAST_FEATURES = (*CONTRAST_FEATURES_V3, *RIGHT_OF_WAY_ACTION_FEATURES)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _group_rows(groups: Sequence[str]) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    values = np.asarray(groups, dtype=str)
    ordered = tuple(dict.fromkeys(values.tolist()))
    return ordered, tuple(np.flatnonzero(values == group) for group in ordered)


def _policy_values(
    actual: np.ndarray,
    score: np.ndarray,
    groups: Sequence[str],
    references: Sequence[bool],
) -> np.ndarray:
    _, rows = _group_rows(groups)
    values, _ = _policy_arrays(
        np.asarray(actual, dtype=float),
        np.asarray(score, dtype=float),
        rows,
        np.asarray(references, dtype=bool),
    )
    return values


def _fold_groups(groups: Sequence[str], fold_count: int = FOLD_COUNT) -> tuple[dict[str, Any], ...]:
    ordered = tuple(sorted(str(value) for value in groups))
    if len(ordered) < int(fold_count) or len(ordered) % int(fold_count):
        raise ValueError("adaptation groups do not form equal mechanism-prior folds")
    folds = []
    for fold_index in range(int(fold_count)):
        heldout = tuple(ordered[fold_index::fold_count])
        heldout_set = set(heldout)
        training = tuple(group for group in ordered if group not in heldout_set)
        folds.append(
            {
                "fold_index": fold_index,
                "training_groups": training,
                "heldout_groups": heldout,
            }
        )
    observed = [group for fold in folds for group in fold["heldout_groups"]]
    if sorted(observed) != list(ordered) or len(set(observed)) != len(ordered):
        raise ValueError("mechanism-prior folds do not partition adaptation groups")
    return tuple(folds)


def _candidate_key(source: str, block: str) -> str:
    return f"{source}|{block}"


def _split_candidate_key(key: str) -> tuple[str, str]:
    parts = str(key).split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid mechanism-prior candidate: {key}")
    return parts[0], parts[1]


def _select_candidate(
    values: Mapping[str, Sequence[float]],
    *,
    target_values: Sequence[float],
    minimum_gain: float,
) -> dict[str, Any]:
    if not values:
        raise ValueError("mechanism-prior selector has no candidates")
    means = {
        str(key): float(np.mean(np.asarray(rows, dtype=float)))
        for key, rows in values.items()
    }
    best = min(means, key=lambda key: (means[key], key))
    target_mean = float(np.mean(np.asarray(target_values, dtype=float)))
    gain = target_mean - means[best]
    admitted = gain >= float(minimum_gain)
    source, block = _split_candidate_key(best)
    return {
        "candidate": best,
        "source_city": source,
        "mechanism_block": block,
        "target_only_mean": target_mean,
        "candidate_mean": means[best],
        "gain_vs_target_only": float(gain),
        "minimum_gain": float(minimum_gain),
        "admitted": bool(admitted),
        "candidate_means": means,
    }


def _fit_with_prior(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    source_coefficients: np.ndarray,
    block_indices: Sequence[int],
    prior_strength: float,
) -> np.ndarray:
    return ridge_coefficients(
        x,
        y,
        groups,
        ridge_l2=RIDGE_L2,
        prior=source_coefficients,
        prior_indices=block_indices,
        prior_strength=prior_strength,
    )


def _arm_summary(
    *,
    absolute: Any,
    contrast: Any,
    actual: np.ndarray,
    score: np.ndarray,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    del absolute
    groups = np.asarray(action_group_ids(contrast), dtype=str)
    unique_groups, group_rows = _group_rows(groups)
    references = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    summary = _scenario_seed_policy_summary(
        actual=actual,
        score=np.asarray(score, dtype=float),
        policy_rows=group_rows,
        references=references,
        unique_groups=unique_groups,
        scenarios=scenarios,
    )
    summary["policy"] = _policy_metrics(actual, score, group_rows, references)
    summary["predictive"] = _predictive_metrics(actual, score, groups, references)
    return summary


def run_target(
    *,
    target_city: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    expected_source_cache_audit_sha256: str,
    conversion_root: Path,
    cache_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V150C target city: {city}")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V150C source cache audit identity changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    protocol, bank, scenarios_by_group, bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    if tuple(sorted(scenarios_by_group)) != EXPECTED_CITY_GROUPS:
        raise ValueError("V150C seven-city inventory changed")

    _, sumocfg_by_scenario = _read_manifest(source_manifest_path)
    contexts = {
        scenario: read_network_right_of_way_context(
            net_file_from_sumocfg(sumocfg_by_scenario[scenario])
        )
        for scenario in bank
    }
    augmented_bank = {
        scenario: augment_dataset_with_right_of_way_context(
            dataset, {scenario: contexts[scenario]}
        )
        for scenario, dataset in bank.items()
    }
    city_datasets = {
        group: _merge_city_datasets(
            augmented_bank, scenarios_by_group[group], city=group
        )
        for group in EXPECTED_CITY_GROUPS
    }
    target_full = city_datasets[city]
    selected_groups, selection_audit = select_city_budget_groups(
        target_full,
        city=city,
        scenarios=scenarios_by_group[city],
    )
    all_target_groups = set(
        str(value) for value in target_full.metadata["action_group_ids"]
    )
    evaluation_groups = all_target_groups - set(selected_groups)
    if not evaluation_groups or evaluation_groups & set(selected_groups):
        raise ValueError("V150C adaptation/evaluation split failed")
    adaptation = _group_subset_v3(
        target_full,
        selected_groups=set(selected_groups),
        metadata_updates={"target_data_role": "v150c_adaptation_only"},
    )
    evaluation = _group_subset_v3(
        target_full,
        selected_groups=evaluation_groups,
        metadata_updates={"target_data_role": "v150c_evaluation_only"},
    )

    city_parts: dict[str, dict[str, Any]] = {}
    for group, absolute in city_datasets.items():
        contrast = build_action_contrast_dataset(
            absolute,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES,
        )
        actual, _, _, references, _ = _normalized_pressure_targets(
            absolute, contrast
        )
        design = mechanism_parameter_design(contrast)
        city_parts[group] = {
            "contrast": contrast,
            "actual": actual,
            "references": references,
            "design": design,
        }

    adaptation_contrast = build_action_contrast_dataset(
        adaptation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    adaptation_actual, _, _, adaptation_references, _ = _normalized_pressure_targets(
        adaptation, adaptation_contrast
    )
    adaptation_design = mechanism_parameter_design(adaptation_contrast)
    evaluation_contrast = build_action_contrast_dataset(
        evaluation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    evaluation_actual, evaluation_unique, evaluation_rows, evaluation_references, _ = (
        _normalized_pressure_targets(evaluation, evaluation_contrast)
    )
    evaluation_design = mechanism_parameter_design(evaluation_contrast)
    if (
        adaptation_design.feature_names != evaluation_design.feature_names
        or adaptation_design.feature_names
        != next(iter(city_parts.values()))["design"].feature_names
    ):
        raise ValueError("V150C mechanism design schema changed across cities")

    source_order = tuple(group for group in EXPECTED_CITY_GROUPS if group != city)
    scale = equal_city_rms_scale(
        [
            city_parts[group]["design"].values
            for group in source_order
        ]
        + [adaptation_design.values]
    )
    source_coefficients: dict[str, np.ndarray] = {}
    placebo_coefficients: dict[str, np.ndarray] = {}
    for source_index, source in enumerate(source_order):
        part = city_parts[source]
        source_x = np.asarray(part["design"].values, dtype=float) / scale
        source_y = np.asarray(part["actual"], dtype=float)
        source_groups = np.asarray(
            action_group_ids(part["contrast"]), dtype=str
        )
        source_coefficients[source] = ridge_coefficients(
            source_x,
            source_y,
            source_groups,
            ridge_l2=RIDGE_L2,
        )
        placebo_y = permute_group_targets(
            source_y,
            source_groups,
            seed=PLACEBO_SEED + source_index,
            references=part["references"],
        )
        placebo_coefficients[source] = ridge_coefficients(
            source_x,
            placebo_y,
            source_groups,
            ridge_l2=RIDGE_L2,
        )

    x_adaptation = np.asarray(adaptation_design.values, dtype=float) / scale
    adaptation_groups = np.asarray(
        action_group_ids(adaptation_contrast), dtype=str
    )
    folds = _fold_groups(selected_groups)
    target_oof_values: list[float] = []
    candidate_oof_values: dict[str, list[float]] = {
        _candidate_key(source, block): []
        for source in source_order
        for block in MECHANISM_ACTION_FEATURES
    }
    placebo_oof_values = {key: [] for key in candidate_oof_values}
    fold_audit = []
    for fold in folds:
        train_mask = np.isin(adaptation_groups, fold["training_groups"])
        heldout_mask = np.isin(adaptation_groups, fold["heldout_groups"])
        if np.any(train_mask & heldout_mask) or not np.all(train_mask | heldout_mask):
            raise ValueError("V150C fold masks do not partition B25 rows")
        beta_target = ridge_coefficients(
            x_adaptation[train_mask],
            adaptation_actual[train_mask],
            adaptation_groups[train_mask],
            ridge_l2=RIDGE_L2,
        )
        heldout_x = x_adaptation[heldout_mask]
        heldout_y = adaptation_actual[heldout_mask]
        heldout_groups = adaptation_groups[heldout_mask]
        heldout_references = adaptation_references[heldout_mask]
        target_oof_values.extend(
            _policy_values(
                heldout_y,
                heldout_x @ beta_target,
                heldout_groups,
                heldout_references,
            ).tolist()
        )
        for source in source_order:
            for block, indices in adaptation_design.block_indices.items():
                key = _candidate_key(source, block)
                beta = _fit_with_prior(
                    x_adaptation[train_mask],
                    adaptation_actual[train_mask],
                    adaptation_groups[train_mask],
                    source_coefficients=source_coefficients[source],
                    block_indices=indices,
                    prior_strength=SOURCE_PRIOR_STRENGTH,
                )
                candidate_oof_values[key].extend(
                    _policy_values(
                        heldout_y,
                        heldout_x @ beta,
                        heldout_groups,
                        heldout_references,
                    ).tolist()
                )
                placebo_beta = _fit_with_prior(
                    x_adaptation[train_mask],
                    adaptation_actual[train_mask],
                    adaptation_groups[train_mask],
                    source_coefficients=placebo_coefficients[source],
                    block_indices=indices,
                    prior_strength=SOURCE_PRIOR_STRENGTH,
                )
                placebo_oof_values[key].extend(
                    _policy_values(
                        heldout_y,
                        heldout_x @ placebo_beta,
                        heldout_groups,
                        heldout_references,
                    ).tolist()
                )
        fold_audit.append(
            {
                **fold,
                "training_row_count": int(np.count_nonzero(train_mask)),
                "heldout_row_count": int(np.count_nonzero(heldout_mask)),
            }
        )

    selector = _select_candidate(
        candidate_oof_values,
        target_values=target_oof_values,
        minimum_gain=MINIMUM_OOF_GAIN,
    )
    placebo_selector = _select_candidate(
        placebo_oof_values,
        target_values=target_oof_values,
        minimum_gain=MINIMUM_OOF_GAIN,
    )
    beta_target = ridge_coefficients(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        ridge_l2=RIDGE_L2,
    )
    source_null_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=np.full(beta_target.shape, 1e6),
        block_indices=adaptation_design.block_indices["queue_service"],
        prior_strength=0.0,
    )
    if not np.array_equal(beta_target, source_null_beta):
        raise ValueError("V150C source-null is not exact target-only")
    x_evaluation = np.asarray(evaluation_design.values, dtype=float) / scale
    target_score = x_evaluation @ beta_target
    source_null_score = x_evaluation @ source_null_beta
    if not np.array_equal(target_score, source_null_score):
        raise ValueError("V150C source-null scores differ from target-only")

    selected_source, selected_block = _split_candidate_key(selector["candidate"])
    forced_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=source_coefficients[selected_source],
        block_indices=adaptation_design.block_indices[selected_block],
        prior_strength=SOURCE_PRIOR_STRENGTH,
    )
    forced_score = x_evaluation @ forced_beta
    selected_score = forced_score if selector["admitted"] else source_null_score.copy()

    placebo_source, placebo_block = _split_candidate_key(
        placebo_selector["candidate"]
    )
    placebo_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=placebo_coefficients[placebo_source],
        block_indices=adaptation_design.block_indices[placebo_block],
        prior_strength=SOURCE_PRIOR_STRENGTH,
    )
    placebo_forced_score = x_evaluation @ placebo_beta
    placebo_score = (
        placebo_forced_score
        if placebo_selector["admitted"]
        else source_null_score.copy()
    )
    phase_score = _reference_policy_score(evaluation_references)
    arms = {
        "phase_pressure": phase_score,
        "target_only_mechanism": target_score,
        "source_null_exact": source_null_score,
        "forced_source_mechanism_prior": forced_score,
        "selected_source_mechanism_prior": selected_score,
        "selected_matched_placebo_prior": placebo_score,
    }
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=evaluation_contrast,
            actual=evaluation_actual,
            score=score,
            scenarios=scenarios_by_group[city],
        )
        for name, score in arms.items()
    }
    if arm_summaries["target_only_mechanism"] != arm_summaries["source_null_exact"]:
        raise ValueError("V150C source-null summary is not exact target-only")

    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "seven-city-mechanism-prior-development-target",
        "target_city": city,
        "target_scenarios": list(scenarios_by_group[city]),
        "source_city_groups": list(source_order),
        "target_budget": TARGET_BUDGET,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": str(protocol["target_name"]),
        "method": {
            "design_feature_names": list(adaptation_design.feature_names),
            "mechanism_blocks": {
                name: [
                    adaptation_design.feature_names[int(index)]
                    for index in indices
                ]
                for name, indices in adaptation_design.block_indices.items()
            },
            "ridge_l2": RIDGE_L2,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "minimum_oof_gain": MINIMUM_OOF_GAIN,
            "placebo_seed": PLACEBO_SEED,
            "selector": selector,
            "placebo_selector": placebo_selector,
        },
        "information_budget": {
            "target_adaptation_group_count": len(selected_groups),
            "target_evaluation_group_count": len(evaluation_unique),
            "evaluation_groups_used_for_fit_or_selection": False,
            "adaptation_labels_used_for_five_fold_source_and_block_selection": True,
            "source_labels_used_only_for_mechanism_coefficient_priors": True,
            "target_evaluation_labels_used_only_for_final_scoring": True,
        },
        "selection_audit": selection_audit,
        "crossfit_folds": fold_audit,
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "prior_strength_zero": True,
            "coefficients_bitwise_equal": True,
            "evaluation_scores_bitwise_equal": True,
            "summaries_equal": True,
        },
        "input_audits": {"source_bank": bank_audit},
        "inputs": {
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "claim_boundary": (
            "V150C is a seven-city development feasibility test. It uses B25 "
            "target outcomes to select one source and one mechanism block, so it "
            "is few-shot target adaptation, not zero-shot or fresh-city confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def _paired_effects(
    results: Mapping[str, Mapping[str, Any]], candidate: str, reference: str
) -> tuple[list[float], dict[str, float]]:
    values = []
    city_means = {}
    for city in EXPECTED_CITY_GROUPS:
        arms = results[city]["arm_summaries"]
        candidate_seeds = arms[candidate]["seed_values"]
        reference_seeds = arms[reference]["seed_values"]
        if set(candidate_seeds) != {str(seed) for seed in SOURCE_SEEDS} or set(
            reference_seeds
        ) != {str(seed) for seed in SOURCE_SEEDS}:
            raise ValueError(f"{city}: V150C seed inventory changed")
        city_values = [
            float(candidate_seeds[str(seed)] - reference_seeds[str(seed)])
            for seed in SOURCE_SEEDS
        ]
        values.extend(city_values)
        city_means[city] = float(np.mean(city_values))
    return values, city_means


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
    ):
        raise ValueError("V150C aggregate requires one valid result per city")
    target_effects, target_city_means = _paired_effects(
        by_city,
        "selected_source_mechanism_prior",
        "target_only_mechanism",
    )
    placebo_effects, placebo_city_means = _paired_effects(
        by_city,
        "selected_source_mechanism_prior",
        "selected_matched_placebo_prior",
    )
    forced_effects, forced_city_means = _paired_effects(
        by_city,
        "forced_source_mechanism_prior",
        "target_only_mechanism",
    )
    target_summary = _paired_bootstrap(target_effects)
    placebo_summary = _paired_bootstrap(placebo_effects)
    forced_summary = _paired_bootstrap(forced_effects)
    admitted_cities = sum(
        bool(row["method"]["selector"]["admitted"]) for row in rows
    )
    gates = {
        "mean_selected_effect_improves_target_only": float(target_summary["mean"])
        < 0.0,
        "selected_effect_bootstrap_upper_below_zero": float(
            target_summary["upper_95"]
        )
        < 0.0,
        "at_least_five_cities_improve_target_only": sum(
            value < 0.0 for value in target_city_means.values()
        )
        >= 5,
        "mean_selected_effect_beats_matched_placebo": float(
            placebo_summary["mean"]
        )
        < 0.0,
        "worst_city_regression_at_most_0p01": max(target_city_means.values())
        <= 0.01,
        "at_least_five_cities_admit_a_source_prior": admitted_cities >= 5,
    }
    passed = all(gates.values())
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development_gate_pass" if passed else "development_gate_reject",
        "city_count": len(rows),
        "target_seed_unit_count": len(target_effects),
        "selected_effect_vs_target_only": {
            **target_summary,
            "city_means": target_city_means,
        },
        "selected_effect_vs_matched_placebo": {
            **placebo_summary,
            "city_means": placebo_city_means,
        },
        "forced_effect_vs_target_only": {
            **forced_summary,
            "city_means": forced_city_means,
        },
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "checks": gates,
            "passed": passed,
            "decision": (
                "promote_mechanism_prior_to_cfcmt_integration"
                if passed
                else "do_not_promote_mechanism_prior"
            ),
        },
        "claim_boundary": (
            "V150C evaluates action-conditioned mechanism-prior transfer on the "
            "seven development cities. A pass authorizes CFCMT integration but "
            "is not closed-loop or untouched-city confirmation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150C result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
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
