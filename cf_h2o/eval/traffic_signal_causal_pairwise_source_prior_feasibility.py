"""Test a causal pairwise source prior with held-out safety calibration."""

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

from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    _load_prediction_artifact,
    _prediction,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
    _stable_group_rows,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_ranker import (
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
    AntisymmetricPairwiseActionRegressor,
    PairwisePreferenceConfig,
    _row_subset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RESULT_PROTOCOL = "tsc-v127-causal-pairwise-source-prior-feasibility-v1"
ARMS = ("target_only", "source_aligned", "source_placebo")
TARGET_BUDGET = 500
SOURCE_GROUP = "atlanta"
SOURCE_WEIGHT = 0.75
CALIBRATION_SEED_COUNT = 5
RETENTION_FRACTIONS = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
MINIMUM_CALIBRATION_INTERVENTIONS = 20
MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION = 0.8
MINIMUM_EFFECT = 0.0005
ONE_SIDED_T_CRITICAL_DF4 = 2.132
PAIRWISE_CONFIG = PairwisePreferenceConfig(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=10.0,
    uncertainty_quantile=0.90,
    random_state=20260803,
)
TARGET_PRIOR_FEATURE = "target_b500_score"
SOURCE_PRIOR_FEATURE = "atlanta_b500_incremental_score"


def _outer_partition(
    all_seeds: Sequence[int], heldout_seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in all_seeds))
    heldout = int(heldout_seed)
    if heldout not in ordered:
        raise ValueError("held-out seed is absent")
    start = ordered.index(heldout)
    calibration = tuple(
        ordered[(start + offset) % len(ordered)]
        for offset in range(1, CALIBRATION_SEED_COUNT + 1)
    )
    training = tuple(
        seed for seed in ordered if seed != heldout and seed not in calibration
    )
    if (
        len(training) + len(calibration) + 1 != len(ordered)
        or set(training) & set(calibration)
        or heldout in training
        or heldout in calibration
    ):
        raise ValueError("V127 outer partition is not disjoint")
    return training, calibration


def _permute_group_channel(
    channel: np.ndarray,
    policy_rows: np.ndarray,
    group_seeds: np.ndarray,
) -> np.ndarray:
    source = np.asarray(channel, dtype=float)
    output = np.empty_like(source)
    rows = np.asarray(policy_rows, dtype=int)
    for seed in sorted(int(value) for value in np.unique(group_seeds)):
        group_indices = np.flatnonzero(group_seeds == seed)
        generator = np.random.default_rng(127000 + seed)
        permutation = generator.permutation(group_indices)
        if np.array_equal(permutation, group_indices):
            permutation = np.roll(permutation, 1)
        output[rows[group_indices]] = source[rows[permutation]]
    return output


def _ranker_dataset(
    contrast: MechanismDataset,
    actual: np.ndarray,
    target_score: np.ndarray,
    source_increment: np.ndarray,
) -> MechanismDataset:
    target = np.asarray(target_score, dtype=float)
    source = np.asarray(source_increment, dtype=float)
    if target.shape != (contrast.size,) or source.shape != (contrast.size,):
        raise ValueError("V127 prior channels differ from selector rows")
    requested = set(
        (*PAIRWISE_CAUSAL_STATE_PARENTS, *PAIRWISE_CAUSAL_ACTION_PARENTS)
    )
    base_indices = [
        index
        for index, name in enumerate(contrast.feature_names)
        if name in requested
    ]
    base_names = tuple(contrast.feature_names[index] for index in base_indices)
    return MechanismDataset(
        feature_names=(
            *base_names,
            TARGET_PRIOR_FEATURE,
            SOURCE_PRIOR_FEATURE,
        ),
        features=np.column_stack(
            [contrast.features[:, base_indices], target, source]
        ),
        context_names=contrast.context_names,
        context=np.asarray(contrast.context, dtype=float),
        priors=contrast.priors,
        targets={**contrast.targets, "interval_cost": np.asarray(actual, dtype=float)},
        domains=np.asarray(contrast.domains),
        metadata=contrast.metadata,
    )


def _group_layout(
    dataset: MechanismDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    unique_groups, group_rows = _stable_group_rows(groups)
    references = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    if any(np.count_nonzero(references[rows]) != 1 for rows in group_rows):
        raise ValueError("V127 group does not have exactly one PhasePressure row")
    action_counts = {int(np.asarray(rows).size) for rows in group_rows}
    if len(action_counts) != 1:
        raise ValueError("V127 requires rectangular action groups")
    seeds = np.asarray(
        [parse_action_group_seed(str(group)) for group in unique_groups], dtype=int
    )
    return np.vstack(group_rows).astype(int, copy=False), references, seeds


def _policy_proposals(
    predicted: np.ndarray,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(policy_rows, dtype=int)
    allowed = np.asarray(eligible[rows], dtype=bool)
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("V127 pressure constraint removed every group action")
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(references[rows], axis=1)
    ]
    scores = np.where(allowed, np.asarray(predicted[rows], dtype=float), np.inf)
    selected_rows = rows[np.arange(rows.shape[0]), np.argmin(scores, axis=1)]
    proposed = selected_rows != reference_rows
    advantage = np.asarray(
        predicted[reference_rows] - predicted[selected_rows], dtype=float
    )
    advantage = np.where(proposed, np.maximum(advantage, 0.0), 0.0)
    return {
        "selected_rows": selected_rows,
        "reference_rows": reference_rows,
        "proposed": proposed,
        "predicted_advantage": advantage,
        "actual_selected_delta": np.asarray(actual[selected_rows], dtype=float),
    }


def _seed_values(
    proposals: Mapping[str, np.ndarray],
    accepted: np.ndarray,
    group_seeds: np.ndarray,
    evaluation_seeds: Sequence[int],
) -> dict[int, float]:
    outcomes = np.where(
        np.asarray(accepted, dtype=bool),
        np.asarray(proposals["actual_selected_delta"], dtype=float),
        0.0,
    )
    return {
        int(seed): float(np.mean(outcomes[group_seeds == int(seed)]))
        for seed in evaluation_seeds
    }


def _one_sided_upper(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("V127 calibration requires at least two seed values")
    return float(
        np.mean(array)
        + ONE_SIDED_T_CRITICAL_DF4
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )


def _retention_threshold(
    advantages: np.ndarray, fraction: float
) -> float | None:
    values = np.sort(np.asarray(advantages, dtype=float))
    if values.size == 0:
        return None
    keep = max(1, int(np.ceil(float(fraction) * values.size)))
    return float(values[-keep])


def _select_calibration_profile(
    proposals: Mapping[str, np.ndarray],
    group_seeds: np.ndarray,
    calibration_seeds: Sequence[int],
) -> dict[str, Any]:
    calibration_mask = np.isin(group_seeds, np.asarray(calibration_seeds, dtype=int))
    proposed = calibration_mask & np.asarray(proposals["proposed"], dtype=bool)
    candidates = []
    for fraction in RETENTION_FRACTIONS:
        threshold = _retention_threshold(
            np.asarray(proposals["predicted_advantage"])[proposed], fraction
        )
        accepted = (
            np.zeros_like(proposed)
            if threshold is None
            else proposed
            & (np.asarray(proposals["predicted_advantage"]) >= threshold)
        )
        values = _seed_values(
            proposals, accepted, group_seeds, calibration_seeds
        )
        seed_array = np.asarray(list(values.values()), dtype=float)
        intervention_count = int(np.count_nonzero(accepted))
        mean = float(np.mean(seed_array))
        upper = _one_sided_upper(seed_array)
        improving_fraction = float(np.mean(seed_array < 0.0))
        authorized = bool(
            intervention_count >= MINIMUM_CALIBRATION_INTERVENTIONS
            and mean <= -MINIMUM_EFFECT
            and upper < 0.0
            and improving_fraction
            >= MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
        )
        candidates.append(
            {
                "retention_fraction": float(fraction),
                "predicted_advantage_threshold": threshold,
                "intervention_count": intervention_count,
                "mean": mean,
                "upper_95_one_sided": upper,
                "improving_seed_fraction": improving_fraction,
                "authorized": authorized,
            }
        )
    authorized = [row for row in candidates if row["authorized"]]
    if not authorized:
        return {
            "enabled": False,
            "predicted_advantage_threshold": None,
            "selected_retention_fraction": 0.0,
            "candidates": candidates,
        }
    selected = min(
        authorized,
        key=lambda row: (
            float(row["mean"]),
            float(row["retention_fraction"]),
        ),
    )
    return {
        "enabled": True,
        "predicted_advantage_threshold": float(
            selected["predicted_advantage_threshold"]
        ),
        "selected_retention_fraction": float(selected["retention_fraction"]),
        "selected_calibration_summary": selected,
        "candidates": candidates,
    }


def _heldout_policy(
    proposals: Mapping[str, np.ndarray],
    profile: Mapping[str, Any],
    group_seeds: np.ndarray,
    heldout_seed: int,
) -> dict[str, Any]:
    heldout = group_seeds == int(heldout_seed)
    if not bool(profile["enabled"]):
        accepted = np.zeros_like(heldout)
    else:
        accepted = (
            heldout
            & np.asarray(proposals["proposed"], dtype=bool)
            & (
                np.asarray(proposals["predicted_advantage"], dtype=float)
                >= float(profile["predicted_advantage_threshold"])
            )
        )
    value = _seed_values(
        proposals, accepted, group_seeds, (int(heldout_seed),)
    )[int(heldout_seed)]
    active_values = np.asarray(proposals["actual_selected_delta"])[accepted]
    return {
        "heldout_value": float(value),
        "heldout_intervention_count": int(np.count_nonzero(accepted)),
        "heldout_harmful_intervention_fraction": (
            float(np.mean(active_values > 0.0)) if active_values.size else 0.0
        ),
    }


def _outer_fold(
    heldout_seed: int,
    *,
    datasets: Mapping[str, MechanismDataset],
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
    arms = {}
    for arm in ARMS:
        train = _row_subset(datasets[arm], train_mask)
        evaluation = _row_subset(datasets[arm], evaluation_mask)
        model = AntisymmetricPairwiseActionRegressor(
            config=PAIRWISE_CONFIG,
            state_feature_names=PAIRWISE_CAUSAL_STATE_PARENTS,
            action_feature_names=(
                *PAIRWISE_CAUSAL_ACTION_PARENTS,
                TARGET_PRIOR_FEATURE,
                SOURCE_PRIOR_FEATURE,
            ),
        )
        fit = model.fit(train)
        predicted = np.asarray(
            model.predict(evaluation)["control_cost"]["mean"], dtype=float
        )
        policy_rows, references, group_seeds = _group_layout(evaluation)
        proposals = _policy_proposals(
            predicted,
            np.asarray(evaluation.targets["interval_cost"], dtype=float),
            policy_rows,
            np.asarray(eligible[evaluation_mask], dtype=bool),
            references,
        )
        profile = _select_calibration_profile(
            proposals, group_seeds, calibration_seeds
        )
        heldout = _heldout_policy(
            proposals, profile, group_seeds, heldout_seed
        )
        arms[arm] = {
            **heldout,
            "calibration_profile": profile,
            "fit": {
                "training_action_group_count": fit["action_group_count"],
                "oriented_pair_count": fit["oriented_pair_count"],
                "training_weighted_mae": fit["training_weighted_mae"],
                "calibration_error": fit["calibration_error"],
                "state_feature_names": fit["state_feature_names"],
                "action_feature_names": fit["action_feature_names"],
            },
        }
    return {
        "heldout_seed": int(heldout_seed),
        "training_seeds": list(training_seeds),
        "calibration_seeds": list(calibration_seeds),
        "arms": arms,
    }


def _arm_summary(folds: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [float(row["arms"][arm]["heldout_value"]) for row in folds]
    return {
        **_paired_bootstrap(values),
        "intervention_count": int(
            sum(
                int(row["arms"][arm]["heldout_intervention_count"])
                for row in folds
            )
        ),
        "enabled_fold_count": int(
            sum(
                bool(row["arms"][arm]["calibration_profile"]["enabled"])
                for row in folds
            )
        ),
    }


def _paired_arm_summary(
    folds: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    return _paired_bootstrap(
        [
            float(row["arms"][candidate]["heldout_value"])
            - float(row["arms"][reference]["heldout_value"])
            for row in folds
        ]
    )


def run_causal_pairwise_source_prior_feasibility(
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
        raise ValueError("V127 frozen V123 input contract changed")
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
        selector, target_name="prefix_mean_cost_450s"
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
        raise ValueError("V127 prediction rows differ from selector rows")
    incremental_source = SOURCE_WEIGHT * (source.score - target.score)
    placebo_source = _permute_group_channel(
        incremental_source, policy_rows, group_seeds
    )
    datasets = {
        "target_only": _ranker_dataset(
            contrast, actual, target.score, np.zeros_like(incremental_source)
        ),
        "source_aligned": _ranker_dataset(
            contrast, actual, target.score, incremental_source
        ),
        "source_placebo": _ranker_dataset(
            contrast, actual, target.score, placebo_source
        ),
    }
    row_seeds = np.empty(contrast.size, dtype=int)
    row_seeds[policy_rows] = group_seeds[:, None]
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = (
        np.asarray(contrast.features[:, pressure_index], dtype=float) >= -1e-12
    )
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) <= CALIBRATION_SEED_COUNT + 1:
        raise ValueError("V127 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    datasets=datasets,
                    all_seeds=all_seeds,
                    row_seeds=row_seeds,
                    eligible=eligible,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    arm_summaries = {arm: _arm_summary(folds, arm) for arm in ARMS}
    source_vs_target = _paired_arm_summary(
        folds, "source_aligned", "target_only"
    )
    source_vs_placebo = _paired_arm_summary(
        folds, "source_aligned", "source_placebo"
    )
    source_summary = arm_summaries["source_aligned"]
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
            "ranker_training_selector_seeds_per_fold": (
                len(all_seeds) - CALIBRATION_SEED_COUNT - 1
            ),
            "ranker_calibration_selector_seeds_per_fold": CALIBRATION_SEED_COUNT,
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_fit_or_calibration": False,
            "target_counterfactual_labels_used_for_ranker_fit": True,
        },
        "source_prior": {
            "source_group": SOURCE_GROUP,
            "source_weight": SOURCE_WEIGHT,
            "target_weight": 1.0 - SOURCE_WEIGHT,
            "selection_evidence": (
                "V123 B500 selected atlanta__source_weight_0.75 in 22/22 "
                "leave-one-selector-seed folds"
            ),
            "incremental_channel": (
                "0.75 * (atlanta_b500_score - target_b500_score)"
            ),
        },
        "pairwise_ranker": {
            "family": "causal_antisymmetric_pairwise_hgb",
            "config": asdict(PAIRWISE_CONFIG),
            "state_parent_names": list(PAIRWISE_CAUSAL_STATE_PARENTS),
            "action_parent_names": list(PAIRWISE_CAUSAL_ACTION_PARENTS)
            + [TARGET_PRIOR_FEATURE, SOURCE_PRIOR_FEATURE],
            "action_constraint": "nondecreasing_instantaneous_service_pressure",
        },
        "calibration_gate": {
            "retention_fractions": list(RETENTION_FRACTIONS),
            "minimum_interventions": MINIMUM_CALIBRATION_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_CALIBRATION_IMPROVING_SEED_FRACTION
            ),
            "one_sided_t_critical_df4": ONE_SIDED_T_CRITICAL_DF4,
        },
        "folds": folds,
        "arm_summaries": arm_summaries,
        "paired_source_contribution": {
            "source_aligned_minus_target_only": source_vs_target,
            "source_aligned_minus_source_placebo": source_vs_placebo,
        },
        "development_gate": {
            "minimum_effect": MINIMUM_EFFECT,
            "passed": passed,
            "decision": (
                "justify_target_budget_curve"
                if passed
                else "do_not_promote_pairwise_source_prior"
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
            "V127 uses abundant target counterfactual selector labels and is "
            "an upper-feasibility test, not few-shot adaptation or confirmation."
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
        raise FileExistsError(f"refusing to overwrite V127 result: {args.out}")
    result = run_causal_pairwise_source_prior_feasibility(
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
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if result["development_gate"]["passed"] else "REJECT"
                ),
                "protocol": result["protocol"],
                "arm_summaries": result["arm_summaries"],
                "paired_source_contribution": result[
                    "paired_source_contribution"
                ],
                "development_gate": result["development_gate"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
