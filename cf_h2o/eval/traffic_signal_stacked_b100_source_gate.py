"""Stacked B100 source gate using strict OOF target-adaptation predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    TARGET_GROUP_BUDGET,
    _atomic_bytes,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    ARTIFACT_PROTOCOL as CROSSFIT_ARTIFACT_PROTOCOL,
    STACKED_GATE_BASE_FEATURES,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    CONFIDENCE_QUANTILES,
    MINIMUM_SOURCE_SUPPORT,
    _build_profiles,
    _fit_result_contract,
    _normalized_pressure_targets,
    _selected_adaptation_dataset,
    _selector_dataset,
    _target_cache_audit_contract,
    nested_leave_one_seed_gate_evaluation,
    select_identifiable_source_profile,
    select_safe_arm_profile,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    MAXIMUM_HARMFUL_OVERRIDE_FRACTION,
    MAXIMUM_INTERVENTION_FRACTION,
    MAXIMUM_SEED_REGRESSION,
    MINIMUM_INTERVENTION_FRACTION,
    MINIMUM_PRESSURE_IMPROVEMENT,
    MINIMUM_SOURCE_CONTRIBUTION,
    _load_models,
    _load_pure_waiting_selector_cache_audit,
    _predict_source_models,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.target_calibrated_source_gate import (
    ARM_NAMES,
    GATE_BASE_FEATURES,
    SOURCE_AGGREGATE_FEATURES,
    aggregate_source_predictions,
    extract_gate_base_features,
    fit_architecture_matched_gate_arms,
    gate_feature_matrix,
    group_conformal_margins,
    permute_source_features_by_group,
)


RESULT_PROTOCOL = "tsc-v121-stacked-b100-causal-source-gate-v1"
MODEL_PROTOCOL = "cfcmt-stacked-b100-causal-source-gate-v1"


def _load_crossfit_artifact(
    path: Path,
    *,
    expected_sha256: str,
    fit_result_sha256: str,
    selected_group_ids: Sequence[str],
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("V120 cross-fit artifact identity changed")
    artifact = pickle.loads(Path(path).read_bytes())
    if (
        artifact.get("protocol") != CROSSFIT_ARTIFACT_PROTOCOL
        or artifact.get("city") != "jinan"
        or artifact.get("estimand") != WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
        or artifact.get("target_name") != "prefix_mean_cost_450s"
        or artifact.get("prior_policy") != "phase_pressure"
        or int(artifact.get("target_group_budget", -1)) != TARGET_GROUP_BUDGET
        or int(artifact.get("fold_count", -1)) != 5
        or tuple(artifact.get("selected_group_ids", ()))
        != tuple(selected_group_ids)
        or artifact.get("fit_result_sha256") != fit_result_sha256
    ):
        raise ValueError("V120 cross-fit artifact contract changed")
    return artifact


def _stacked_base(
    local_features: np.ndarray,
    target_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    local = np.asarray(local_features, dtype=float)
    score, uncertainty, trust = (
        np.asarray(values, dtype=float).reshape(-1)
        for values in target_predictions
    )
    if (
        local.ndim != 2
        or local.shape[0] != score.size
        or uncertainty.shape != score.shape
        or trust.shape != score.shape
        or np.any(uncertainty < 0.0)
        or not np.all(np.isfinite(score))
        or not np.all(np.isfinite(uncertainty))
        or not np.all(np.isfinite(trust))
    ):
        raise ValueError("stacked target predictions are not action-row aligned")
    result = np.column_stack((local, score, uncertainty))
    if result.shape[1] != len(STACKED_GATE_BASE_FEATURES):
        raise ValueError("stacked target feature dimension changed")
    return result


def run_stacked_b100_source_gate(
    *,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    crossfit_artifact_path: Path,
    expected_crossfit_artifact_sha256: str,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    selector_scenario: str,
    selector_seeds: Sequence[int],
    selector_collection_shards: int,
    conversion_root: Path,
    cache_workers: int,
    prediction_workers: int,
    gate_artifact_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if gate_artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite V121 gate: {gate_artifact_path}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    _target_cache_audit_contract(target_cache_audit_path, fit_result)
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
    )
    selected_group_ids = tuple(
        str(value)
        for value in fit_result["information_budget"]["selected_group_ids"]
    )
    crossfit = _load_crossfit_artifact(
        crossfit_artifact_path,
        expected_sha256=expected_crossfit_artifact_sha256,
        fit_result_sha256=expected_fit_result_sha256,
        selected_group_ids=selected_group_ids,
    )
    adaptation, adaptation_bank_audit, adaptation_scenarios = (
        _selected_adaptation_dataset(
            cache_root=target_cache_root,
            manifest_path=target_manifest_path,
            cache_workers=cache_workers,
            selected_group_ids=selected_group_ids,
        )
    )
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    component_artifact = fit_result["model_artifacts"]["source_components_b100"]
    strict_artifact = fit_result["model_artifacts"]["strict_target_b100"]
    bundle, strict, source_models, prior_spec = _load_models(
        component_bundle_path=Path(component_artifact["path"]),
        expected_component_bundle_sha256=str(component_artifact["sha256"]),
        strict_target_only_model_path=Path(strict_artifact["path"]),
        expected_strict_target_only_model_sha256=str(strict_artifact["sha256"]),
        city="jinan",
    )
    if str(prior_spec.key) != "phase_pressure":
        raise ValueError("V121 prior is not PhasePressure")
    adaptation_contrast = build_action_contrast_dataset(
        adaptation,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    selector_contrast = build_action_contrast_dataset(
        selector,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    adaptation_row_groups = np.asarray(
        action_group_ids(adaptation_contrast), dtype=str
    )
    if not np.array_equal(
        adaptation_row_groups,
        np.asarray(crossfit["row_group_ids"], dtype=str),
    ):
        raise ValueError("V120 OOF rows differ from the V121 B100 dataset")
    (
        adaptation_actual,
        adaptation_groups,
        _,
        adaptation_references,
        _,
    ) = _normalized_pressure_targets(adaptation, adaptation_contrast)
    (
        selector_actual,
        selector_groups,
        selector_group_rows,
        selector_references,
        selector_group_seeds,
    ) = _normalized_pressure_targets(selector, selector_contrast)
    if set(selector_group_seeds.tolist()) != set(int(value) for value in selector_seeds):
        raise ValueError("V121 selector seed coverage changed")
    adaptation_source, _, source_order = aggregate_source_predictions(
        crossfit["source_predictions"]
    )
    if source_order != tuple(crossfit["source_group_order"]):
        raise ValueError("V120 source order changed")
    adaptation_local = extract_gate_base_features(adaptation_contrast)
    adaptation_base = _stacked_base(
        adaptation_local, tuple(crossfit["target_predictions"])
    )
    adaptation_placebo = permute_source_features_by_group(
        adaptation_source,
        adaptation_row_groups,
        adaptation_references,
        adaptation_local[:, GATE_BASE_FEATURES.index("delta_service_pressure")],
    )
    adaptation_matrices = {
        "target_only": gate_feature_matrix(
            adaptation_base,
            adaptation_source,
            arm="target_only",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
        "source_aligned": gate_feature_matrix(
            adaptation_base,
            adaptation_source,
            arm="source_aligned",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
        "source_placebo": gate_feature_matrix(
            adaptation_base,
            adaptation_placebo,
            arm="source_placebo",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
    }
    fitted = fit_architecture_matched_gate_arms(
        feature_matrices=adaptation_matrices,
        targets=adaptation_actual,
        groups=adaptation_row_groups,
        is_reference=adaptation_references,
        base_feature_names=STACKED_GATE_BASE_FEATURES,
    )
    margins = group_conformal_margins(
        actual=adaptation_actual,
        oof_predictions=fitted["oof_predictions"],
        groups=adaptation_row_groups,
        is_reference=adaptation_references,
        quantiles=CONFIDENCE_QUANTILES,
    )
    target_score, target_uncertainty, target_trust, _ = _group_adjusted_scores(
        selector_contrast,
        strict["model"].predict(selector_contrast),
        objective_mode=str(strict["objective_mode"]),
    )
    selector_predictions = _predict_source_models(
        source_models=source_models,
        model_contrast=selector_contrast,
        workers=prediction_workers,
    )
    selector_source, selector_score_stack, selector_source_order = (
        aggregate_source_predictions(selector_predictions)
    )
    if selector_source_order != source_order:
        raise ValueError("V121 selector source model order changed")
    selector_local = extract_gate_base_features(selector_contrast)
    selector_base = _stacked_base(
        selector_local, (target_score, target_uncertainty, target_trust)
    )
    selector_row_groups = np.asarray(action_group_ids(selector_contrast), dtype=str)
    selector_placebo = permute_source_features_by_group(
        selector_source,
        selector_row_groups,
        selector_references,
        selector_local[:, GATE_BASE_FEATURES.index("delta_service_pressure")],
    )
    selector_matrices = {
        "target_only": gate_feature_matrix(
            selector_base,
            selector_source,
            arm="target_only",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
        "source_aligned": gate_feature_matrix(
            selector_base,
            selector_source,
            arm="source_aligned",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
        "source_placebo": gate_feature_matrix(
            selector_base,
            selector_placebo,
            arm="source_placebo",
            base_feature_names=STACKED_GATE_BASE_FEATURES,
        ),
    }
    profiles = {}
    compatibility = {}
    for arm in ARM_NAMES:
        predicted = fitted["models"][arm].predict(selector_matrices[arm])
        arm_profiles, arm_compatibility = _build_profiles(
            arm=arm,
            predicted=predicted,
            margins=margins[arm],
            group_rows=selector_group_rows,
            references=selector_references,
            group_seeds=selector_group_seeds,
            normalized_actual=selector_actual,
            source_score_stack=(
                selector_score_stack if arm == "source_aligned" else None
            ),
        )
        profiles.update(arm_profiles)
        compatibility[arm] = arm_compatibility
    nested = nested_leave_one_seed_gate_evaluation(profiles, selector_seeds)
    source_selection = select_identifiable_source_profile(
        profiles, training_seeds=selector_seeds
    )
    target_selection = select_safe_arm_profile(
        {key: row for key, row in profiles.items() if row["arm"] == "target_only"},
        training_seeds=selector_seeds,
    )
    placebo_selection = select_safe_arm_profile(
        {key: row for key, row in profiles.items() if row["arm"] == "source_placebo"},
        training_seeds=selector_seeds,
    )
    deployment_authorized = bool(
        nested["summary"]["source_transfer_gate_passed"]
        and source_selection["selected_profile_key"] is not None
    )
    gate_payload = {
        "protocol": MODEL_PROTOCOL,
        "city": "jinan",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": selected_group_ids,
        "crossfit_artifact_sha256": expected_crossfit_artifact_sha256,
        "source_model_artifact": dict(component_artifact),
        "target_model_artifact": dict(strict_artifact),
        "source_group_order": source_order,
        "base_feature_names": STACKED_GATE_BASE_FEATURES,
        "source_aggregate_feature_names": SOURCE_AGGREGATE_FEATURES,
        "gate_feature_names": fitted["feature_names"],
        "ridge_alpha": fitted["selected_alpha"],
        "models": fitted["models"],
        "conformal_margins": margins,
        "minimum_source_support": MINIMUM_SOURCE_SUPPORT,
        "selected_source_profile_key": source_selection["selected_profile_key"],
        "deployment_authorized": deployment_authorized,
    }
    gate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        gate_artifact_path,
        pickle.dumps(gate_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    round_trip = pickle.loads(gate_artifact_path.read_bytes())
    if (
        round_trip.get("protocol") != MODEL_PROTOCOL
        or set(round_trip.get("models", {})) != set(ARM_NAMES)
        or tuple(round_trip.get("source_group_order", ())) != source_order
    ):
        raise ValueError("V121 gate artifact round trip failed")
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development_selector_not_fresh_city_confirmation",
        "city": "jinan",
        "selector_scenario": selector_scenario,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "information_budget": {
            "target_adaptation_action_groups": TARGET_GROUP_BUDGET,
            "target_adaptation_seeds": list(ADAPTATION_SEEDS),
            "crossfit_training_groups_per_fold": 80,
            "crossfit_heldout_groups_per_fold": 20,
            "deployment_model_target_groups": 100,
            "target_only_target_groups": 100,
            "selector_target_labels_used_for_model_fit": 0,
            "source_city_identity_features": 0,
        },
        "architecture_match": {
            "arms": list(ARM_NAMES),
            "same_model_family": True,
            "same_ridge_alpha": True,
            "same_feature_dimension": True,
            "same_b100_groups": True,
            "same_crossfit_folds": True,
            "target_base_prediction_available_to_all_arms": True,
            "source_placebo_preserves_group_block_marginals": True,
            "selected_alpha": fitted["selected_alpha"],
            "alpha_selection": fitted["alpha_selection_arm"],
        },
        "calibration_oof": {
            "group_count": len(adaptation_groups),
            "row_count": int(adaptation_contrast.size),
            "arm_diagnostics": fitted["arm_diagnostics"],
            "alpha_diagnostics": fitted["alpha_diagnostics"],
            "conformal_margins": margins,
        },
        "selector": {
            "group_count": len(selector_groups),
            "row_count": int(selector_contrast.size),
            "profile_count": len(profiles),
            "compatibility": compatibility,
            "nested_leave_one_seed": nested,
            "full_development_selection": {
                "source_aligned": source_selection,
                "target_only": target_selection,
                "source_placebo": placebo_selection,
            },
        },
        "gate": {
            "passed": deployment_authorized,
            "decision": (
                "freeze_stacked_b100_gate_for_fresh_city_confirmation"
                if deployment_authorized
                else "reject_stacked_source_gate_and_retain_phase_pressure_fallback"
            ),
            "requirements": {
                "minimum_pressure_improvement": MINIMUM_PRESSURE_IMPROVEMENT,
                "minimum_source_contribution": MINIMUM_SOURCE_CONTRIBUTION,
                "maximum_seed_regression": MAXIMUM_SEED_REGRESSION,
                "maximum_harmful_override_fraction": MAXIMUM_HARMFUL_OVERRIDE_FRACTION,
                "intervention_fraction_range": [
                    MINIMUM_INTERVENTION_FRACTION,
                    MAXIMUM_INTERVENTION_FRACTION,
                ],
                "minimum_source_support": MINIMUM_SOURCE_SUPPORT,
            },
        },
        "inputs": {
            "fit_result": {
                "path": str(Path(fit_result_path).resolve()),
                "sha256": expected_fit_result_sha256,
            },
            "crossfit_artifact": {
                "path": str(Path(crossfit_artifact_path).resolve()),
                "sha256": expected_crossfit_artifact_sha256,
                "protocol": CROSSFIT_ARTIFACT_PROTOCOL,
            },
            "selector_cache_audit": selector_audit,
            "source_model_artifact": dict(component_artifact),
            "target_model_artifact": dict(strict_artifact),
            "source_bundle_adaptation_contract_sha256": bundle.get(
                "adaptation_contract_sha256"
            ),
        },
        "data_audits": {
            "adaptation_scenarios": list(adaptation_scenarios),
            "adaptation_bank": adaptation_bank_audit,
            "selector_bank": selector_bank_audit,
        },
        "model_artifact": {
            "path": str(gate_artifact_path.resolve()),
            "sha256": _sha256(gate_artifact_path),
            "size_bytes": gate_artifact_path.stat().st_size,
            "protocol": MODEL_PROTOCOL,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--crossfit-artifact", type=Path, required=True)
    parser.add_argument("--crossfit-artifact-sha256", required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", default="jinan_3x4_real")
    parser.add_argument("--selector-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--selector-collection-shards", type=int, default=32)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--prediction-workers", type=int, default=7)
    parser.add_argument("--gate-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V121 cache workers must be in [1, 20]")
    if not 1 <= int(args.prediction_workers) <= 7:
        raise ValueError("V121 prediction workers must be in [1, 7]")
    result = run_stacked_b100_source_gate(
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        crossfit_artifact_path=args.crossfit_artifact,
        expected_crossfit_artifact_sha256=args.crossfit_artifact_sha256,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        prediction_workers=args.prediction_workers,
        gate_artifact_path=args.gate_artifact,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
