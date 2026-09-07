"""Screen one-step physical mechanisms as surrogates for 450-second waiting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility import (
    MINIMUM_EFFECT,
    RETENTION_FRACTIONS,
    _arm_summary,
    _group_layout,
    _retention_threshold,
    _seed_values,
)
from cf_h2o.eval.traffic_signal_familywise_mechanistic_pressure_screen import (
    FAMILYWISE_ALPHA,
    MULTIPLIER_DRAWS,
    MULTIPLIER_SEED_BASE,
    _max_t_critical,
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
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v134-local-mechanism-surrogate-oracle-v1"
SURROGATE_TARGETS = {
    "queue_propagation": {"one_step_total_queue": 1.0},
    "served_movement": {"one_step_green_queue": 1.0},
    "red_accumulation": {"one_step_red_queue": 1.0},
    "spillback": {"one_step_downstream_occupancy": 1.0},
    "mobility": {"one_step_mean_speed": -1.0},
    "queue_red": {
        "one_step_total_queue": 0.75,
        "one_step_red_queue": 0.25,
    },
    "causal_physical": {
        "one_step_total_queue": 0.60,
        "one_step_red_queue": 0.15,
        "one_step_downstream_occupancy": 0.025,
        "one_step_mean_speed": -0.02,
    },
    "balanced_physical": {
        "one_step_total_queue": 0.40,
        "one_step_green_queue": 0.20,
        "one_step_red_queue": 0.20,
        "one_step_downstream_occupancy": 0.10,
        "one_step_mean_speed": -0.10,
    },
}
MINIMUM_PROFILE_INTERVENTIONS = 40
MINIMUM_PROFILE_IMPROVING_SEED_FRACTION = 0.80
SURROGATE_MULTIPLIER_SEED = MULTIPLIER_SEED_BASE + 134


def _normalized_target_contrasts(
    absolute_dataset: Any,
    contrast_dataset: Any,
    group_rows: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    targets = sorted(
        {target for weights in SURROGATE_TARGETS.values() for target in weights}
    )
    output = {}
    for target_name in targets:
        if target_name not in absolute_dataset.targets:
            raise KeyError(f"V134 cache is missing {target_name!r}")
        absolute = np.asarray(absolute_dataset.targets[target_name], dtype=float)
        delta = np.asarray(contrast_dataset.targets[target_name], dtype=float)
        normalized = np.zeros(contrast_dataset.size, dtype=float)
        for rows in group_rows:
            normalized[rows] = delta[rows] / action_group_range(absolute[rows])
        output[target_name] = normalized
    return output


def _surrogate_scores(
    normalized_targets: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {}
    for name, weights in SURROGATE_TARGETS.items():
        scale = sum(abs(float(value)) for value in weights.values())
        score = sum(
            float(weight) * np.asarray(normalized_targets[target], dtype=float)
            for target, weight in weights.items()
        ) / max(scale, 1e-12)
        result[name] = np.asarray(score, dtype=float)
    return result


def _surrogate_proposals(
    scores: np.ndarray,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(policy_rows, dtype=int)
    allowed = np.asarray(eligible[rows], dtype=bool)
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("V134 pressure constraint removed every group action")
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(references[rows], axis=1)
    ]
    values = np.asarray(scores, dtype=float)
    masked = np.where(allowed, values[rows], np.inf)
    selected_rows = rows[np.arange(rows.shape[0]), np.argmin(masked, axis=1)]
    minima = np.min(masked, axis=1)
    reference_ties = np.isclose(
        values[reference_rows], minima, rtol=0.0, atol=1e-12
    )
    selected_rows = np.where(reference_ties, reference_rows, selected_rows)
    selected_score = values[selected_rows]
    reference_score = values[reference_rows]
    predicted_advantage = reference_score - selected_score
    proposed = (selected_rows != reference_rows) & (predicted_advantage > 1e-12)
    return {
        "selected_rows": selected_rows,
        "reference_rows": reference_rows,
        "proposed": proposed,
        "predicted_advantage": np.where(proposed, predicted_advantage, 0.0),
        "actual_selected_delta": np.asarray(actual[selected_rows], dtype=float),
    }


def _profile_arm(proxy: str, fraction: float) -> str:
    return f"{proxy}__retain_{float(fraction):g}".replace(".", "p")


def _seed_fold(
    seed: int,
    *,
    proposals_by_proxy: Mapping[str, Mapping[str, np.ndarray]],
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    seed_mask = group_seeds == int(seed)
    arms = {}
    for proxy, proposals in proposals_by_proxy.items():
        proposed = seed_mask & np.asarray(proposals["proposed"], dtype=bool)
        for fraction in RETENTION_FRACTIONS:
            threshold = _retention_threshold(
                np.asarray(proposals["predicted_advantage"])[proposed],
                fraction,
            )
            accepted = (
                np.zeros_like(seed_mask, dtype=bool)
                if threshold is None
                else proposed
                & (
                    np.asarray(proposals["predicted_advantage"], dtype=float)
                    >= float(threshold)
                )
            )
            value = _seed_values(
                proposals,
                accepted,
                group_seeds,
                (int(seed),),
            )[int(seed)]
            active = np.asarray(proposals["actual_selected_delta"])[accepted]
            arm = _profile_arm(proxy, fraction)
            arms[arm] = {
                "heldout_value": float(value),
                "heldout_intervention_count": int(np.count_nonzero(accepted)),
                "heldout_harmful_intervention_fraction": (
                    float(np.mean(active > 0.0)) if active.size else 0.0
                ),
                "deployment_enabled": bool(np.any(accepted)),
                "surrogate": proxy,
                "retention_fraction": float(fraction),
                "predicted_advantage_threshold": (
                    None if threshold is None else float(threshold)
                ),
            }
    return {"heldout_seed": int(seed), "arms": arms}


def _familywise_selection(
    folds: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    arm_names = tuple(sorted(folds[0]["arms"]))
    values = np.asarray(
        [
            [float(fold["arms"][arm]["heldout_value"]) for fold in folds]
            for arm in arm_names
        ],
        dtype=float,
    )
    critical = _max_t_critical(
        values,
        alpha=FAMILYWISE_ALPHA,
        draws=MULTIPLIER_DRAWS,
        random_seed=SURROGATE_MULTIPLIER_SEED,
    )
    summaries = {}
    for index, arm in enumerate(arm_names):
        summary = _arm_summary(folds, arm)
        standard_error = float(
            np.std(values[index], ddof=1) / np.sqrt(values.shape[1])
        )
        upper = float(summary["mean"] + critical * standard_error)
        summary = {
            **summary,
            "standard_error": standard_error,
            "simultaneous_upper_95": upper,
            "eligible": bool(
                summary["intervention_count"] >= MINIMUM_PROFILE_INTERVENTIONS
                and summary["mean"] <= -MINIMUM_EFFECT
                and upper < 0.0
                and summary["improving_seed_fraction"]
                >= MINIMUM_PROFILE_IMPROVING_SEED_FRACTION
            ),
            "surrogate": folds[0]["arms"][arm]["surrogate"],
            "retention_fraction": folds[0]["arms"][arm]["retention_fraction"],
        }
        summaries[arm] = summary
    eligible = [arm for arm in arm_names if summaries[arm]["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda arm: (
                float(summaries[arm]["simultaneous_upper_95"]),
                float(summaries[arm]["mean"]),
                arm,
            ),
        )
        if eligible
        else None
    )
    return {
        "passed": selected is not None,
        "selected_profile": selected,
        "eligible_profile_count": len(eligible),
        "familywise_critical_value": float(critical),
        "decision": (
            "authorize_selected_local_mechanism_model_successor"
            if selected is not None
            else "do_not_fit_one_step_surrogate_action_model"
        ),
    }, summaries


def run_local_mechanism_surrogate_oracle(
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
    policy_rows, references, layout_seeds = _group_layout(contrast)
    if not np.array_equal(policy_rows, np.vstack(group_rows)) or not np.array_equal(
        layout_seeds, group_seeds
    ):
        raise ValueError("V134 action-group layout changed during normalization")
    normalized = _normalized_target_contrasts(selector, contrast, group_rows)
    scores = _surrogate_scores(normalized)
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index], dtype=float
    ) >= -1e-12
    proposals = {
        proxy: _surrogate_proposals(
            score,
            actual,
            policy_rows,
            eligible,
            references,
        )
        for proxy, score in scores.items()
    }
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    folds = [
        _seed_fold(
            seed,
            proposals_by_proxy=proposals,
            group_seeds=group_seeds,
        )
        for seed in all_seeds
    ]
    gate, summaries = _familywise_selection(folds)
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "heldout-local-mechanism-surrogate-oracle-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "target_450s_labels_used_for_action_selection": False,
            "same_seed_one_step_counterfactual_labels_used_for_oracle_selection": True,
            "target_450s_labels_used_only_for_surrogate_scoring": True,
        },
        "surrogates": SURROGATE_TARGETS,
        "retention_fractions": list(RETENTION_FRACTIONS),
        "profile_count": len(SURROGATE_TARGETS) * len(RETENTION_FRACTIONS),
        "action_constraint": "nondecreasing_instantaneous_service_pressure",
        "selection_control": {
            "method": "gaussian_multiplier_one_sided_max_t_across_profiles",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "multiplier_draws": MULTIPLIER_DRAWS,
            "multiplier_seed": SURROGATE_MULTIPLIER_SEED,
            "minimum_interventions": MINIMUM_PROFILE_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_PROFILE_IMPROVING_SEED_FRACTION
            ),
        },
        "folds": folds,
        "profile_summaries": summaries,
        "development_gate": gate,
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "inputs": {
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "claim_boundary": (
            "V134 is a same-seed local-mechanism oracle screen. It does not "
            "represent a learned, deployable, zero-shot, or transfer policy."
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V134 result: {args.out}")
    result = run_local_mechanism_surrogate_oracle(
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
    )
    atomic_write_json(args.out, result)
    selected = result["development_gate"]["selected_profile"]
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if result["development_gate"]["passed"] else "REJECT"
                ),
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "selected_summary": (
                    None if selected is None else result["profile_summaries"][selected]
                ),
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
