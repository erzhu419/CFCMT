"""Test source priors against one shared target-only causal residual."""

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
    BASE_ARMS,
    CALIBRATION_SEED_COUNT,
    MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION,
    MINIMUM_CALIBRATION_INTERVENTIONS,
    MINIMUM_EFFECT,
    ONE_SIDED_T_CRITICAL_DF9,
    REPORT_ARMS,
    RESIDUAL_CONFIG,
    RETENTION_FRACTIONS,
    SOURCE_GROUP,
    SOURCE_WEIGHT,
    TARGET_BUDGET,
    _arm_summary,
    _center_on_reference,
    _group_layout,
    _heldout_policy,
    _outer_partition,
    _paired_arm_summary,
    _permute_group_channel,
    _policy_proposals,
    _residual_dataset,
    _select_calibration_profile,
    _source_null_gate,
)
from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    _load_prediction_artifact,
    _prediction,
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
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
    CausalReferenceResidualRegressor,
    _row_subset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RESULT_PROTOCOL = "tsc-v129-shared-residual-source-null-feasibility-v1"


def _shared_prior_predictions(
    shared_residual: np.ndarray,
    priors: Mapping[str, np.ndarray],
    evaluation_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = np.asarray(shared_residual, dtype=float)
    mask = np.asarray(evaluation_mask, dtype=bool)
    predictions = {
        arm: residual + np.asarray(prior, dtype=float)[mask]
        for arm, prior in priors.items()
    }
    if any(value.shape != residual.shape for value in predictions.values()):
        raise ValueError("V129 shared residual and prior rows differ")
    return predictions


def _outer_fold(
    heldout_seed: int,
    *,
    target_residual_dataset: MechanismDataset,
    actual: np.ndarray,
    priors: Mapping[str, np.ndarray],
    all_seeds: Sequence[int],
    row_seeds: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    training_seeds, calibration_seeds = _outer_partition(
        all_seeds, heldout_seed
    )
    train_mask = np.isin(row_seeds, np.asarray(training_seeds, dtype=int))
    evaluation_seeds = (*calibration_seeds, int(heldout_seed))
    evaluation_mask = np.isin(
        row_seeds, np.asarray(evaluation_seeds, dtype=int)
    )
    train = _row_subset(target_residual_dataset, train_mask)
    evaluation = _row_subset(target_residual_dataset, evaluation_mask)
    model = CausalReferenceResidualRegressor(
        config=RESIDUAL_CONFIG,
        state_feature_names=PAIRWISE_CAUSAL_STATE_PARENTS,
        action_feature_names=PAIRWISE_CAUSAL_ACTION_PARENTS,
    )
    shared_fit = model.fit(train)
    shared_residual = np.asarray(
        model.predict(evaluation)["control_cost"]["mean"], dtype=float
    )
    predictions = _shared_prior_predictions(
        shared_residual,
        priors,
        evaluation_mask,
    )
    policy_rows, references, group_seeds = _group_layout(evaluation)
    actual_evaluation = np.asarray(actual, dtype=float)[evaluation_mask]
    arms: dict[str, dict[str, Any]] = {}
    for arm in BASE_ARMS:
        proposals = _policy_proposals(
            predictions[arm],
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
            "residual_fit": "shared_target_only_residual",
        }

    source_gate = _source_null_gate(
        arms["source_aligned"]["calibration_profile"],
        arms["target_only"]["calibration_profile"],
        calibration_seeds,
    )
    placebo_gate = _source_null_gate(
        arms["source_placebo"]["calibration_profile"],
        arms["target_only"]["calibration_profile"],
        calibration_seeds,
    )
    for adaptive, candidate, gate in (
        ("adaptive_source_null", "source_aligned", source_gate),
        ("adaptive_placebo_null", "source_placebo", placebo_gate),
    ):
        selected_arm = candidate if gate["selected_source"] else "target_only"
        selected = arms[selected_arm]
        arms[adaptive] = {
            "heldout_value": selected["heldout_value"],
            "heldout_intervention_count": selected[
                "heldout_intervention_count"
            ],
            "heldout_harmful_intervention_fraction": selected[
                "heldout_harmful_intervention_fraction"
            ],
            "deployment_enabled": selected["deployment_enabled"],
            "selected_arm": selected_arm,
            "source_null_gate": gate,
        }
    return {
        "heldout_seed": int(heldout_seed),
        "training_seeds": list(training_seeds),
        "calibration_seeds": list(calibration_seeds),
        "shared_target_residual_fit": shared_fit,
        "arms": arms,
    }


def run_shared_residual_source_null_feasibility(
    *,
    prediction_artifact_path: Path,
    expected_prediction_artifact_sha256: str,
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
    artifact = _load_prediction_artifact(
        prediction_artifact_path,
        expected_sha256=expected_prediction_artifact_sha256,
    )
    budget_key = str(TARGET_BUDGET)
    if (
        budget_key not in artifact["target_predictions"]
        or SOURCE_GROUP not in artifact["source_predictions"][budget_key]
        or tuple(int(value) for value in artifact.get("selector_seeds", ()))
        != tuple(int(value) for value in selector_seeds)
        or artifact.get("selector_cache_audit_sha256")
        != expected_selector_cache_audit_sha256
    ):
        raise ValueError("V129 frozen V123 input contract changed")
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
    actual, unique_groups, group_rows, references, group_seeds = (
        _normalized_pressure_targets(selector, contrast)
    )
    policy_rows = np.vstack(group_rows).astype(int, copy=False)
    target = _prediction(artifact["target_predictions"][budget_key])
    source = _prediction(
        artifact["source_predictions"][budget_key][SOURCE_GROUP]
    )
    if target.score.size != contrast.size or source.score.size != contrast.size:
        raise ValueError("V129 prediction rows differ from selector rows")
    incremental_source = SOURCE_WEIGHT * (source.score - target.score)
    placebo_source = _permute_group_channel(
        incremental_source,
        policy_rows,
        group_seeds,
    )
    priors = {
        "target_only": _center_on_reference(
            target.score,
            policy_rows,
            references,
        ),
        "source_aligned": _center_on_reference(
            target.score + incremental_source,
            policy_rows,
            references,
        ),
        "source_placebo": _center_on_reference(
            target.score + placebo_source,
            policy_rows,
            references,
        ),
    }
    target_residual_dataset = _residual_dataset(
        contrast,
        actual,
        priors["target_only"],
    )
    row_seeds = np.empty(contrast.size, dtype=int)
    row_seeds[policy_rows] = group_seeds[:, None]
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index], dtype=float
    ) >= -1e-12
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) <= CALIBRATION_SEED_COUNT + 1:
        raise ValueError("V129 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    target_residual_dataset=target_residual_dataset,
                    actual=actual,
                    priors=priors,
                    all_seeds=all_seeds,
                    row_seeds=row_seeds,
                    eligible=eligible,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    arm_summaries = {arm: _arm_summary(folds, arm) for arm in REPORT_ARMS}
    source_vs_target = _paired_arm_summary(
        folds,
        "adaptive_source_null",
        "target_only",
    )
    source_vs_placebo = _paired_arm_summary(
        folds,
        "adaptive_source_null",
        "adaptive_placebo_null",
    )
    source_summary = arm_summaries["adaptive_source_null"]
    passed = bool(
        source_summary["mean"] <= -MINIMUM_EFFECT
        and source_summary["upper_95"] < 0.0
        and source_vs_target["mean"] <= -MINIMUM_EFFECT
        and source_vs_target["upper_95"] < 0.0
        and source_vs_placebo["mean"] <= -MINIMUM_EFFECT
        and source_vs_placebo["upper_95"] < 0.0
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-counterfactual-development-diagnostic",
        "city": artifact["city"],
        "estimand": artifact["estimand"],
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "prior_target_group_budget": TARGET_BUDGET,
            "residual_training_selector_seeds_per_fold": (
                len(all_seeds) - CALIBRATION_SEED_COUNT - 1
            ),
            "source_null_calibration_selector_seeds_per_fold": (
                CALIBRATION_SEED_COUNT
            ),
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_fit_or_calibration": False,
            "target_counterfactual_labels_used_for_residual_fit": True,
        },
        "anchored_prior": {
            "target_budget": TARGET_BUDGET,
            "source_group": SOURCE_GROUP,
            "source_weight": SOURCE_WEIGHT,
            "target_weight": 1.0 - SOURCE_WEIGHT,
            "aligned_prior": "target_b500 + 0.75 * (atlanta_b500 - target_b500)",
            "placebo_protocol": (
                "within-seed whole-action-group permutation of the incremental "
                "source offset"
            ),
        },
        "residual_ranker": {
            "family": "causal_reference_contrast_hgb",
            "config": asdict(RESIDUAL_CONFIG),
            "state_parent_names": list(PAIRWISE_CAUSAL_STATE_PARENTS),
            "action_parent_names": list(PAIRWISE_CAUSAL_ACTION_PARENTS),
            "prior_is_offset_not_feature": True,
            "shared_target_residual_across_all_prior_arms": True,
            "source_specific_target_residual_refit": False,
            "action_constraint": "nondecreasing_instantaneous_service_pressure",
        },
        "calibration_gate": {
            "retention_fractions": list(RETENTION_FRACTIONS),
            "minimum_interventions": MINIMUM_CALIBRATION_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
            ),
            "one_sided_t_critical_df9": ONE_SIDED_T_CRITICAL_DF9,
            "source_null_uses_only_calibration_seeds": True,
        },
        "folds": folds,
        "arm_summaries": arm_summaries,
        "paired_source_contribution": {
            "adaptive_source_null_minus_target_only": source_vs_target,
            "adaptive_source_null_minus_adaptive_placebo_null": (
                source_vs_placebo
            ),
        },
        "development_gate": {
            "minimum_effect": MINIMUM_EFFECT,
            "passed": passed,
            "decision": (
                "justify_total_target_budget_curve"
                if passed
                else "do_not_promote_shared_residual_source_null"
            ),
        },
        "inputs": {
            "prediction_artifact_sha256": expected_prediction_artifact_sha256,
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "claim_boundary": (
            "V129 uses abundant target counterfactual selector labels and is "
            "a shared-residual source-null feasibility test, not few-shot "
            "adaptation or confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--prediction-artifact-sha256", required=True)
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
        raise FileExistsError(f"refusing to overwrite V129 result: {args.out}")
    result = run_shared_residual_source_null_feasibility(
        prediction_artifact_path=args.prediction_artifact,
        expected_prediction_artifact_sha256=args.prediction_artifact_sha256,
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
                "paired_source_contribution": result[
                    "paired_source_contribution"
                ],
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
