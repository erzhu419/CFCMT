"""Screen a low-dimensional generalized-pressure intervention mechanism."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility import (
    CALIBRATION_SEED_COUNT,
    MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION,
    MINIMUM_CALIBRATION_INTERVENTIONS,
    MINIMUM_EFFECT,
    ONE_SIDED_T_CRITICAL_DF9,
    RETENTION_FRACTIONS,
    _arm_summary,
    _group_layout,
    _one_sided_upper,
    _outer_partition,
    _retention_threshold,
    _seed_values,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import (
    PressurePolicySpec,
    generalized_pressure_grid,
    pressure_scores_from_feature_matrix,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RESULT_PROTOCOL = "tsc-v131-mechanistic-pressure-screen-v1"
MODEL_ARM = "mechanistic_generalized_pressure"
RULE_SPECS = generalized_pressure_grid()
TRAINING_T_CRITICAL_DF10 = 1.812
MINIMUM_TRAINING_INTERVENTIONS = 40
MINIMUM_TRAINING_IMPROVING_SEED_FRACTION = 0.60


def _rule_proposals(
    dataset: MechanismDataset,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
    spec: PressurePolicySpec,
) -> dict[str, np.ndarray]:
    rows = np.asarray(policy_rows, dtype=int)
    allowed = np.asarray(eligible[rows], dtype=bool)
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("V131 pressure constraint removed every group action")
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(references[rows], axis=1)
    ]
    raw_scores = pressure_scores_from_feature_matrix(
        dataset.features,
        dataset.feature_names,
        spec,
    )
    scores = np.where(allowed, raw_scores[rows], -np.inf)
    selected_rows = rows[np.arange(rows.shape[0]), np.argmax(scores, axis=1)]
    maxima = np.max(scores, axis=1)
    reference_scores = raw_scores[reference_rows]
    reference_ties = np.isclose(
        reference_scores,
        maxima,
        rtol=0.0,
        atol=1e-12,
    )
    selected_rows = np.where(reference_ties, reference_rows, selected_rows)
    advantage = raw_scores[selected_rows] - reference_scores
    proposed = (selected_rows != reference_rows) & (advantage > 1e-12)
    return {
        "selected_rows": selected_rows,
        "reference_rows": reference_rows,
        "proposed": proposed,
        "predicted_advantage": np.where(proposed, advantage, 0.0),
        "actual_selected_delta": np.asarray(actual[selected_rows], dtype=float),
    }


def _fixed_profile_mask(
    proposals: Mapping[str, np.ndarray],
    *,
    seed_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.asarray(seed_mask, dtype=bool)
        & np.asarray(proposals["proposed"], dtype=bool)
        & (
            np.asarray(proposals["predicted_advantage"], dtype=float)
            >= float(threshold)
        )
    )


def _training_candidates(
    proposals_by_rule: Mapping[str, Mapping[str, np.ndarray]],
    *,
    group_seeds: np.ndarray,
    training_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    training_mask = np.isin(
        group_seeds,
        np.asarray(training_seeds, dtype=int),
    )
    rows: list[dict[str, Any]] = []
    for spec in RULE_SPECS:
        proposals = proposals_by_rule[spec.key]
        proposed = training_mask & np.asarray(proposals["proposed"], dtype=bool)
        for fraction in RETENTION_FRACTIONS:
            threshold = _retention_threshold(
                np.asarray(proposals["predicted_advantage"])[proposed],
                fraction,
            )
            accepted = (
                np.zeros_like(proposed)
                if threshold is None
                else _fixed_profile_mask(
                    proposals,
                    seed_mask=training_mask,
                    threshold=threshold,
                )
            )
            values = _seed_values(
                proposals,
                accepted,
                group_seeds,
                training_seeds,
            )
            seed_values = np.asarray(list(values.values()), dtype=float)
            mean = float(np.mean(seed_values))
            upper = float(
                mean
                + TRAINING_T_CRITICAL_DF10
                * np.std(seed_values, ddof=1)
                / np.sqrt(seed_values.size)
            )
            intervention_count = int(np.count_nonzero(accepted))
            improving_fraction = float(np.mean(seed_values < 0.0))
            rows.append(
                {
                    "policy_spec": spec.to_dict(),
                    "retention_fraction": float(fraction),
                    "predicted_advantage_threshold": (
                        None if threshold is None else float(threshold)
                    ),
                    "intervention_count": intervention_count,
                    "mean": mean,
                    "upper_95_one_sided": upper,
                    "improving_seed_fraction": improving_fraction,
                    "eligible": bool(
                        threshold is not None
                        and intervention_count >= MINIMUM_TRAINING_INTERVENTIONS
                        and mean < 0.0
                        and improving_fraction
                        >= MINIMUM_TRAINING_IMPROVING_SEED_FRACTION
                    ),
                }
            )
    return rows


def _select_training_profile(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in candidates if bool(row["eligible"])]
    if not eligible:
        return {
            "selected": False,
            "policy_spec": None,
            "retention_fraction": 0.0,
            "predicted_advantage_threshold": None,
        }
    selected = min(
        eligible,
        key=lambda row: (
            float(row["upper_95_one_sided"]),
            float(row["mean"]),
            -float(row["retention_fraction"]),
            str(row["policy_spec"]["key"]),
        ),
    )
    return {"selected": True, **dict(selected)}


def _calibrate_profile(
    proposals: Mapping[str, np.ndarray],
    profile: Mapping[str, Any],
    *,
    group_seeds: np.ndarray,
    calibration_seeds: Sequence[int],
) -> dict[str, Any]:
    if not bool(profile["selected"]):
        return {
            "enabled": False,
            "intervention_count": 0,
            "mean": 0.0,
            "upper_95_one_sided": 0.0,
            "improving_seed_fraction": 0.0,
            "seed_values": {str(seed): 0.0 for seed in calibration_seeds},
        }
    calibration_mask = np.isin(
        group_seeds,
        np.asarray(calibration_seeds, dtype=int),
    )
    accepted = _fixed_profile_mask(
        proposals,
        seed_mask=calibration_mask,
        threshold=float(profile["predicted_advantage_threshold"]),
    )
    values = _seed_values(
        proposals,
        accepted,
        group_seeds,
        calibration_seeds,
    )
    seed_values = np.asarray(list(values.values()), dtype=float)
    intervention_count = int(np.count_nonzero(accepted))
    mean = float(np.mean(seed_values))
    upper = _one_sided_upper(seed_values)
    improving_fraction = float(np.mean(seed_values < 0.0))
    return {
        "enabled": bool(
            intervention_count >= MINIMUM_CALIBRATION_INTERVENTIONS
            and mean <= -MINIMUM_EFFECT
            and upper < 0.0
            and improving_fraction
            >= MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
        ),
        "intervention_count": intervention_count,
        "mean": mean,
        "upper_95_one_sided": upper,
        "improving_seed_fraction": improving_fraction,
        "seed_values": {
            str(seed): float(values[int(seed)]) for seed in calibration_seeds
        },
    }


def _outer_fold(
    heldout_seed: int,
    *,
    all_seeds: Sequence[int],
    proposals_by_rule: Mapping[str, Mapping[str, np.ndarray]],
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    training_seeds, calibration_seeds = _outer_partition(
        all_seeds,
        heldout_seed,
    )
    candidates = _training_candidates(
        proposals_by_rule,
        group_seeds=group_seeds,
        training_seeds=training_seeds,
    )
    profile = _select_training_profile(candidates)
    if profile["selected"]:
        proposals = proposals_by_rule[str(profile["policy_spec"]["key"])]
    else:
        proposals = proposals_by_rule[RULE_SPECS[0].key]
    calibration = _calibrate_profile(
        proposals,
        profile,
        group_seeds=group_seeds,
        calibration_seeds=calibration_seeds,
    )
    heldout_mask = group_seeds == int(heldout_seed)
    accepted = (
        _fixed_profile_mask(
            proposals,
            seed_mask=heldout_mask,
            threshold=float(profile["predicted_advantage_threshold"]),
        )
        if calibration["enabled"]
        else np.zeros_like(heldout_mask, dtype=bool)
    )
    heldout_value = _seed_values(
        proposals,
        accepted,
        group_seeds,
        (int(heldout_seed),),
    )[int(heldout_seed)]
    active_values = np.asarray(proposals["actual_selected_delta"])[accepted]
    arm = {
        "heldout_value": float(heldout_value),
        "heldout_intervention_count": int(np.count_nonzero(accepted)),
        "heldout_harmful_intervention_fraction": (
            float(np.mean(active_values > 0.0)) if active_values.size else 0.0
        ),
        "deployment_enabled": bool(calibration["enabled"]),
        "training_profile": profile,
        "training_candidate_count": len(candidates),
        "training_eligible_candidate_count": int(
            sum(bool(row["eligible"]) for row in candidates)
        ),
        "training_leaderboard": sorted(
            candidates,
            key=lambda row: (
                not bool(row["eligible"]),
                float(row["upper_95_one_sided"]),
                float(row["mean"]),
            ),
        )[:10],
        "calibration_profile": calibration,
    }
    return {
        "heldout_seed": int(heldout_seed),
        "training_seeds": list(training_seeds),
        "calibration_seeds": list(calibration_seeds),
        "arms": {MODEL_ARM: arm},
    }


def run_mechanistic_pressure_screen(
    *,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    selector_scenario: str,
    selector_seeds: Sequence[int],
    selector_collection_shards: int,
    conversion_root: Path,
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
    )
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    _assert_pure_waiting_selector_dataset(
        selector,
        target_name="prefix_mean_cost_450s",
    )
    contrast = build_action_contrast_dataset(
        selector,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual, unique_groups, group_rows, _, group_seeds = (
        _normalized_pressure_targets(selector, contrast)
    )
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index],
        dtype=float,
    ) >= -1e-12
    policy_rows, references, layout_seeds = _group_layout(contrast)
    if not np.array_equal(policy_rows, np.vstack(group_rows)) or not np.array_equal(
        layout_seeds,
        group_seeds,
    ):
        raise ValueError("V131 action-group layout changed during normalization")
    proposals_by_rule = {
        spec.key: _rule_proposals(
            contrast,
            actual,
            policy_rows,
            eligible,
            references,
            spec,
        )
        for spec in RULE_SPECS
    }
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) <= CALIBRATION_SEED_COUNT + 1:
        raise ValueError("V131 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    all_seeds=all_seeds,
                    proposals_by_rule=proposals_by_rule,
                    group_seeds=group_seeds,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    summary = _arm_summary(folds, MODEL_ARM)
    passed = bool(
        summary["mean"] <= -MINIMUM_EFFECT and summary["upper_95"] < 0.0
    )
    selected_specs: dict[str, int] = {}
    for fold in folds:
        profile = fold["arms"][MODEL_ARM]["training_profile"]
        if profile["selected"]:
            key = str(profile["policy_spec"]["key"])
            selected_specs[key] = selected_specs.get(key, 0) + 1
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-mechanistic-development-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "target_counterfactual_labels_used_for_rule_selection": True,
            "target_counterfactual_labels_used_for_calibration": True,
            "training_selector_seeds_per_fold": (
                len(all_seeds) - CALIBRATION_SEED_COUNT - 1
            ),
            "calibration_selector_seeds_per_fold": CALIBRATION_SEED_COUNT,
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_selection_or_calibration": False,
        },
        "mechanism_family": {
            "rule_count": len(RULE_SPECS),
            "rule_specs": [spec.to_dict() for spec in RULE_SPECS],
            "retention_fractions": list(RETENTION_FRACTIONS),
            "selection_criterion": (
                "minimum_training_seed_one_sided_upper_bound"
            ),
            "selected_spec_fold_counts": selected_specs,
        },
        "action_constraint": "nondecreasing_instantaneous_service_pressure",
        "calibration_gate": {
            "minimum_interventions": MINIMUM_CALIBRATION_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
            ),
            "one_sided_t_critical_df9": ONE_SIDED_T_CRITICAL_DF9,
        },
        "folds": folds,
        "arm_summaries": {MODEL_ARM: summary},
        "development_gate": {
            "passed": passed,
            "decision": (
                "authorize_source_prior_and_placebo_successor"
                if passed
                else "do_not_add_source_to_mechanistic_pressure_family"
            ),
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "inputs": {
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "claim_boundary": (
            "V131 is an abundant-target low-dimensional mechanism screen. It "
            "uses no source information and cannot support transfer claims."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", required=True)
    parser.add_argument("--selector-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selector-collection-shards", type=int, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V131 result: {args.out}")
    result = run_mechanistic_pressure_screen(
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
        fold_workers=max(1, int(args.fold_workers)),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if result["development_gate"]["passed"] else "REJECT"
                ),
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "arm_summaries": result["arm_summaries"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
