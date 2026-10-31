"""Screen target-side causal models for safe pressure-rule interventions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
    _heldout_policy,
    _outer_partition,
    _policy_proposals,
    _residual_dataset,
    _select_calibration_profile,
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
from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RANKING_PARENTS,
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    ActionAdvantageConfig,
    ActionRankerConfig,
    PairwiseActionAdvantageRegressor,
    PairwiseActionRanker,
    _row_subset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RESULT_PROTOCOL = "tsc-v130-intervention-model-screen-v1"
MODEL_ARMS = (
    "causal_improvement_classifier",
    "group_normalized_advantage",
)
CLASSIFIER_CONFIG = ActionRankerConfig(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=10.0,
    tie_scale_fraction=0.05,
    uncertainty_quantile=0.90,
    random_state=20260803,
)
ADVANTAGE_CONFIG = ActionAdvantageConfig(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=10.0,
    target_clip=5.0,
    uncertainty_quantile=0.90,
    random_state=20260803,
    candidate_only=True,
    balance_candidate_signs=True,
    max_sign_weight_multiplier=4.0,
    target_normalization_protocol=(
        GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    ),
)


def _model(arm: str):
    if arm == "causal_improvement_classifier":
        return PairwiseActionRanker(causal=True, config=CLASSIFIER_CONFIG)
    if arm == "group_normalized_advantage":
        return PairwiseActionAdvantageRegressor(
            causal=True,
            config=ADVANTAGE_CONFIG,
            causal_feature_names=CAUSAL_RANKING_PARENTS,
        )
    raise ValueError(f"unknown V130 model arm: {arm}")


def _outer_fold(
    heldout_seed: int,
    *,
    dataset: MechanismDataset,
    actual: np.ndarray,
    all_seeds: Sequence[int],
    row_seeds: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    training_seeds, calibration_seeds = _outer_partition(
        all_seeds,
        heldout_seed,
    )
    train_mask = np.isin(row_seeds, np.asarray(training_seeds, dtype=int))
    evaluation_seeds = (*calibration_seeds, int(heldout_seed))
    evaluation_mask = np.isin(
        row_seeds,
        np.asarray(evaluation_seeds, dtype=int),
    )
    train = _row_subset(dataset, train_mask)
    evaluation = _row_subset(dataset, evaluation_mask)
    policy_rows, references, group_seeds = _group_layout(evaluation)
    actual_evaluation = np.asarray(actual, dtype=float)[evaluation_mask]
    arms: dict[str, dict[str, Any]] = {}
    for arm in MODEL_ARMS:
        model = _model(arm)
        fit = model.fit(train)
        predicted = np.asarray(
            model.predict(evaluation)["control_cost"]["mean"],
            dtype=float,
        )
        proposals = _policy_proposals(
            predicted,
            actual_evaluation,
            policy_rows,
            np.asarray(eligible[evaluation_mask], dtype=bool),
            references,
        )
        profile = _select_calibration_profile(
            proposals,
            group_seeds,
            calibration_seeds,
        )
        heldout = _heldout_policy(
            proposals,
            profile,
            group_seeds,
            heldout_seed,
        )
        arms[arm] = {
            **heldout,
            "deployment_enabled": bool(profile["enabled"]),
            "calibration_profile": profile,
            "fit": fit,
        }
    return {
        "heldout_seed": int(heldout_seed),
        "training_seeds": list(training_seeds),
        "calibration_seeds": list(calibration_seeds),
        "arms": arms,
    }


def run_intervention_model_screen(
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
    dataset = _residual_dataset(
        contrast,
        actual,
        np.zeros(contrast.size, dtype=float),
    )
    policy_rows = np.vstack(group_rows).astype(int, copy=False)
    row_seeds = np.empty(contrast.size, dtype=int)
    row_seeds[policy_rows] = group_seeds[:, None]
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index],
        dtype=float,
    ) >= -1e-12
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) <= CALIBRATION_SEED_COUNT + 1:
        raise ValueError("V130 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    dataset=dataset,
                    actual=actual,
                    all_seeds=all_seeds,
                    row_seeds=row_seeds,
                    eligible=eligible,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    arm_summaries = {arm: _arm_summary(folds, arm) for arm in MODEL_ARMS}
    arm_gates = {
        arm: {
            "passed": bool(
                summary["mean"] <= -MINIMUM_EFFECT
                and summary["upper_95"] < 0.0
            ),
            "minimum_effect": MINIMUM_EFFECT,
        }
        for arm, summary in arm_summaries.items()
    }
    passing = [arm for arm in MODEL_ARMS if arm_gates[arm]["passed"]]
    selected = (
        min(passing, key=lambda arm: float(arm_summaries[arm]["mean"]))
        if passing
        else None
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-counterfactual-development-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "target_prior_predictions_used": False,
            "target_counterfactual_labels_used_for_fit": True,
            "training_selector_seeds_per_fold": (
                len(all_seeds) - CALIBRATION_SEED_COUNT - 1
            ),
            "calibration_selector_seeds_per_fold": CALIBRATION_SEED_COUNT,
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_fit_or_calibration": False,
        },
        "models": {
            "causal_improvement_classifier": {
                "family": "pairwise_action_improvement_hgb_classifier",
                "config": asdict(CLASSIFIER_CONFIG),
                "causal_feature_names": list(CAUSAL_RANKING_PARENTS),
            },
            "group_normalized_advantage": {
                "family": "candidate_only_sign_balanced_hgb_regressor",
                "config": asdict(ADVANTAGE_CONFIG),
                "causal_feature_names": list(CAUSAL_RANKING_PARENTS),
            },
        },
        "action_constraint": "nondecreasing_instantaneous_service_pressure",
        "calibration_gate": {
            "retention_fractions": list(RETENTION_FRACTIONS),
            "minimum_interventions": MINIMUM_CALIBRATION_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
            ),
            "one_sided_t_critical_df9": ONE_SIDED_T_CRITICAL_DF9,
        },
        "folds": folds,
        "arm_summaries": arm_summaries,
        "arm_gates": arm_gates,
        "development_gate": {
            "passed": selected is not None,
            "selected_model": selected,
            "decision": (
                "authorize_source_veto_placebo_successor"
                if selected is not None
                else "do_not_integrate_source_with_screened_models"
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
            "V130 is an abundant-target model-family development screen. It "
            "uses no source predictions and cannot support transfer claims."
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
        raise FileExistsError(f"refusing to overwrite V130 result: {args.out}")
    result = run_intervention_model_screen(
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
                "status": "PASS" if result["development_gate"]["passed"] else "REJECT",
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "arm_summaries": result["arm_summaries"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
