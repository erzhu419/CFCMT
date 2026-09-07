"""Select generalized-pressure interventions with a simultaneous error bound."""

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
    MINIMUM_EFFECT,
    RETENTION_FRACTIONS,
    _arm_summary,
    _group_layout,
    _retention_threshold,
    _seed_values,
)
from cf_h2o.eval.traffic_signal_mechanistic_pressure_screen import (
    RULE_SPECS,
    _fixed_profile_mask,
    _rule_proposals,
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


RESULT_PROTOCOL = "tsc-v132-familywise-mechanistic-pressure-screen-v1"
MODEL_ARM = "familywise_mechanistic_pressure"
MINIMUM_DEVELOPMENT_INTERVENTIONS = 80
MINIMUM_DEVELOPMENT_IMPROVING_SEED_FRACTION = 0.80
FAMILYWISE_ALPHA = 0.05
MULTIPLIER_DRAWS = 10_000
MULTIPLIER_BATCH_SIZE = 500
MULTIPLIER_SEED_BASE = 132_000
NORMAL_ONE_SIDED_95 = 1.6448536269514722


def _max_t_critical(
    values: np.ndarray,
    *,
    alpha: float,
    draws: int,
    random_seed: int,
    batch_size: int = MULTIPLIER_BATCH_SIZE,
) -> float:
    """Estimate a correlation-aware one-sided max-t critical value."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("V132 max-t values must be profiles by at least two seeds")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("V132 familywise alpha must be in (0, 1)")
    if int(draws) < 1_000:
        raise ValueError("V132 requires at least 1,000 multiplier draws")
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    norms = np.sqrt(np.sum(np.square(centered), axis=1))
    active = norms > 1e-15
    if not np.any(active):
        return NORMAL_ONE_SIDED_95
    standardized = centered[active] / norms[active, None]
    generator = np.random.default_rng(int(random_seed))
    maxima: list[np.ndarray] = []
    remaining = int(draws)
    while remaining:
        count = min(int(batch_size), remaining)
        multipliers = generator.standard_normal((count, matrix.shape[1]))
        maxima.append(np.max(multipliers @ standardized.T, axis=1))
        remaining -= count
    samples = np.concatenate(maxima)
    return float(
        np.quantile(samples, 1.0 - float(alpha), method="higher")
    )


def _development_candidates(
    proposals_by_rule: Mapping[str, Mapping[str, np.ndarray]],
    *,
    group_seeds: np.ndarray,
    development_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    development_mask = np.isin(
        group_seeds,
        np.asarray(development_seeds, dtype=int),
    )
    rows: list[dict[str, Any]] = []
    for spec in RULE_SPECS:
        proposals = proposals_by_rule[spec.key]
        proposed = development_mask & np.asarray(
            proposals["proposed"], dtype=bool
        )
        for fraction in RETENTION_FRACTIONS:
            threshold = _retention_threshold(
                np.asarray(proposals["predicted_advantage"])[proposed],
                fraction,
            )
            accepted = (
                np.zeros_like(development_mask, dtype=bool)
                if threshold is None
                else _fixed_profile_mask(
                    proposals,
                    seed_mask=development_mask,
                    threshold=threshold,
                )
            )
            seed_value_map = _seed_values(
                proposals,
                accepted,
                group_seeds,
                development_seeds,
            )
            seed_values = np.asarray(
                [seed_value_map[int(seed)] for seed in development_seeds],
                dtype=float,
            )
            rows.append(
                {
                    "policy_spec": spec.to_dict(),
                    "retention_fraction": float(fraction),
                    "predicted_advantage_threshold": (
                        None if threshold is None else float(threshold)
                    ),
                    "intervention_count": int(np.count_nonzero(accepted)),
                    "mean": float(np.mean(seed_values)),
                    "standard_error": float(
                        np.std(seed_values, ddof=1)
                        / np.sqrt(seed_values.size)
                    ),
                    "improving_seed_fraction": float(
                        np.mean(seed_values < 0.0)
                    ),
                    "seed_values": seed_values,
                }
            )
    return rows


def _select_familywise_profile(
    candidates: Sequence[Mapping[str, Any]],
    *,
    heldout_seed: int,
    multiplier_draws: int = MULTIPLIER_DRAWS,
) -> tuple[dict[str, Any], float]:
    if not candidates:
        raise ValueError("V132 candidate family is empty")
    values = np.vstack(
        [np.asarray(row["seed_values"], dtype=float) for row in candidates]
    )
    critical = _max_t_critical(
        values,
        alpha=FAMILYWISE_ALPHA,
        draws=multiplier_draws,
        random_seed=MULTIPLIER_SEED_BASE + int(heldout_seed),
    )
    scored: list[dict[str, Any]] = []
    for row in candidates:
        output = {
            key: value for key, value in row.items() if key != "seed_values"
        }
        output["simultaneous_upper_95"] = float(
            float(row["mean"])
            + critical * float(row["standard_error"])
        )
        output["eligible"] = bool(
            row["predicted_advantage_threshold"] is not None
            and int(row["intervention_count"])
            >= MINIMUM_DEVELOPMENT_INTERVENTIONS
            and float(row["mean"]) <= -MINIMUM_EFFECT
            and output["simultaneous_upper_95"] < 0.0
            and float(row["improving_seed_fraction"])
            >= MINIMUM_DEVELOPMENT_IMPROVING_SEED_FRACTION
        )
        scored.append(output)
    eligible = [row for row in scored if row["eligible"]]
    if not eligible:
        return {
            "selected": False,
            "policy_spec": None,
            "retention_fraction": 0.0,
            "predicted_advantage_threshold": None,
            "eligible_candidate_count": 0,
            "leaderboard": sorted(
                scored,
                key=lambda row: (
                    float(row["simultaneous_upper_95"]),
                    float(row["mean"]),
                    str(row["policy_spec"]["key"]),
                ),
            )[:12],
        }, critical
    selected = min(
        eligible,
        key=lambda row: (
            float(row["simultaneous_upper_95"]),
            float(row["mean"]),
            -float(row["retention_fraction"]),
            str(row["policy_spec"]["key"]),
        ),
    )
    return {
        "selected": True,
        **selected,
        "eligible_candidate_count": len(eligible),
        "leaderboard": sorted(
            scored,
            key=lambda row: (
                not bool(row["eligible"]),
                float(row["simultaneous_upper_95"]),
                float(row["mean"]),
                str(row["policy_spec"]["key"]),
            ),
        )[:12],
    }, critical


def _outer_fold(
    heldout_seed: int,
    *,
    all_seeds: Sequence[int],
    proposals_by_rule: Mapping[str, Mapping[str, np.ndarray]],
    group_seeds: np.ndarray,
    multiplier_draws: int = MULTIPLIER_DRAWS,
) -> dict[str, Any]:
    development_seeds = tuple(
        int(seed) for seed in all_seeds if int(seed) != int(heldout_seed)
    )
    if len(development_seeds) + 1 != len(all_seeds):
        raise ValueError("V132 outer partition is not leave-one-seed-out")
    candidates = _development_candidates(
        proposals_by_rule,
        group_seeds=group_seeds,
        development_seeds=development_seeds,
    )
    profile, critical = _select_familywise_profile(
        candidates,
        heldout_seed=heldout_seed,
        multiplier_draws=multiplier_draws,
    )
    proposals = (
        proposals_by_rule[str(profile["policy_spec"]["key"])]
        if profile["selected"]
        else proposals_by_rule[RULE_SPECS[0].key]
    )
    heldout_mask = group_seeds == int(heldout_seed)
    accepted = (
        _fixed_profile_mask(
            proposals,
            seed_mask=heldout_mask,
            threshold=float(profile["predicted_advantage_threshold"]),
        )
        if profile["selected"]
        else np.zeros_like(heldout_mask, dtype=bool)
    )
    heldout_value = _seed_values(
        proposals,
        accepted,
        group_seeds,
        (int(heldout_seed),),
    )[int(heldout_seed)]
    active_values = np.asarray(proposals["actual_selected_delta"])[accepted]
    return {
        "heldout_seed": int(heldout_seed),
        "development_seeds": list(development_seeds),
        "arms": {
            MODEL_ARM: {
                "heldout_value": float(heldout_value),
                "heldout_intervention_count": int(np.count_nonzero(accepted)),
                "heldout_harmful_intervention_fraction": (
                    float(np.mean(active_values > 0.0))
                    if active_values.size
                    else 0.0
                ),
                "deployment_enabled": bool(profile["selected"]),
                "development_profile": profile,
                "familywise_critical_value": float(critical),
                "candidate_count": len(candidates),
            }
        },
    }


def run_familywise_mechanistic_pressure_screen(
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
    multiplier_draws: int = MULTIPLIER_DRAWS,
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
        contrast.features[:, pressure_index], dtype=float
    ) >= -1e-12
    policy_rows, references, layout_seeds = _group_layout(contrast)
    if not np.array_equal(policy_rows, np.vstack(group_rows)) or not np.array_equal(
        layout_seeds, group_seeds
    ):
        raise ValueError("V132 action-group layout changed during normalization")
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
    if len(all_seeds) < 12:
        raise ValueError("V132 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    all_seeds=all_seeds,
                    proposals_by_rule=proposals_by_rule,
                    group_seeds=group_seeds,
                    multiplier_draws=multiplier_draws,
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
        profile = fold["arms"][MODEL_ARM]["development_profile"]
        if profile["selected"]:
            key = str(profile["policy_spec"]["key"])
            selected_specs[key] = selected_specs.get(key, 0) + 1
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-familywise-development-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "target_counterfactual_labels_used_for_profile_selection": True,
            "development_selector_seeds_per_fold": len(all_seeds) - 1,
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_selection_or_authorization": False,
        },
        "mechanism_family": {
            "rule_count": len(RULE_SPECS),
            "profile_count": len(RULE_SPECS) * len(RETENTION_FRACTIONS),
            "rule_specs": [spec.to_dict() for spec in RULE_SPECS],
            "retention_fractions": list(RETENTION_FRACTIONS),
            "selected_spec_fold_counts": selected_specs,
        },
        "selection_control": {
            "method": "gaussian_multiplier_one_sided_max_t",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "multiplier_draws": int(multiplier_draws),
            "multiplier_seed_base": MULTIPLIER_SEED_BASE,
            "minimum_interventions": MINIMUM_DEVELOPMENT_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": (
                MINIMUM_DEVELOPMENT_IMPROVING_SEED_FRACTION
            ),
        },
        "action_constraint": "nondecreasing_instantaneous_service_pressure",
        "folds": folds,
        "arm_summaries": {MODEL_ARM: summary},
        "development_gate": {
            "passed": passed,
            "decision": (
                "authorize_source_prior_and_placebo_successor"
                if passed
                else "close_generalized_pressure_intervention_family"
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
            "V132 is an abundant-target familywise mechanism screen. It uses "
            "no source information and cannot support transfer claims."
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
    parser.add_argument("--multiplier-draws", type=int, default=MULTIPLIER_DRAWS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V132 result: {args.out}")
    result = run_familywise_mechanistic_pressure_screen(
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
        multiplier_draws=int(args.multiplier_draws),
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
