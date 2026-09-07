"""Test whether target action effects repeat across simulator realizations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility import (
    MINIMUM_EFFECT,
    _arm_summary,
    _group_layout,
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


RESULT_PROTOCOL = "tsc-v133-historical-action-identifiability-v1"
MEAN_ARM = "historical_mean_lookup"
CONSENSUS_ARM = "historical_conservative_consensus"
MODEL_ARMS = (MEAN_ARM, CONSENSUS_ARM)
MINIMUM_HISTORY_SUPPORT_FRACTION = 0.80
MINIMUM_HISTORY_IMPROVING_FRACTION = 0.80
ONE_SIDED_T_CRITICAL_DF16 = 1.746
_SEED_TOKEN = re.compile(r"(?<=:)seed[0-9]+(?=:)")


def _history_group_key(group_id: str) -> str:
    value, replacements = _SEED_TOKEN.subn("seed*", str(group_id), count=1)
    if replacements != 1:
        raise ValueError(f"V133 action group lacks one seed token: {group_id!r}")
    return value


def _history_table(
    *,
    groups: np.ndarray,
    candidate_states: np.ndarray,
    actual: np.ndarray,
    eligible: np.ndarray,
    row_seeds: np.ndarray,
    development_seeds: Sequence[int],
) -> dict[tuple[str, str], dict[int, float]]:
    development = set(int(seed) for seed in development_seeds)
    table: dict[tuple[str, str], dict[int, float]] = {}
    for row in np.flatnonzero(np.asarray(eligible, dtype=bool)):
        seed = int(row_seeds[row])
        if seed not in development:
            continue
        key = (
            _history_group_key(str(groups[row])),
            str(candidate_states[row]),
        )
        values = table.setdefault(key, {})
        if seed in values:
            raise ValueError(f"V133 duplicate historical action for seed {seed}: {key}")
        values[seed] = float(actual[row])
    return table


def _history_action_summary(values: Mapping[int, float]) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=float)
    if array.size == 0:
        raise ValueError("V133 historical action summary is empty")
    standard_error = (
        float(np.std(array, ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else float("inf")
    )
    mean = float(np.mean(array))
    return {
        "support_seed_count": int(array.size),
        "mean": mean,
        "standard_error": standard_error,
        "upper_95_one_sided": float(
            mean + ONE_SIDED_T_CRITICAL_DF16 * standard_error
        ),
        "improving_seed_fraction": float(np.mean(array < 0.0)),
    }


def _select_action(
    candidates: Sequence[Mapping[str, Any]],
    *,
    conservative: bool,
) -> Mapping[str, Any] | None:
    admitted = []
    for row in candidates:
        if float(row["mean"]) > -MINIMUM_EFFECT:
            continue
        if conservative and not (
            float(row["upper_95_one_sided"]) < 0.0
            and float(row["improving_seed_fraction"])
            >= MINIMUM_HISTORY_IMPROVING_FRACTION
        ):
            continue
        admitted.append(row)
    if not admitted:
        return None
    if conservative:
        key = lambda row: (
            float(row["upper_95_one_sided"]),
            float(row["mean"]),
            str(row["candidate_state"]),
        )
    else:
        key = lambda row: (
            float(row["mean"]),
            float(row["upper_95_one_sided"]),
            str(row["candidate_state"]),
        )
    return min(admitted, key=key)


def _outer_fold(
    heldout_seed: int,
    *,
    all_seeds: Sequence[int],
    groups: np.ndarray,
    candidate_states: np.ndarray,
    actual: np.ndarray,
    eligible: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    development_seeds = tuple(
        int(seed) for seed in all_seeds if int(seed) != int(heldout_seed)
    )
    minimum_support = int(
        math.ceil(MINIMUM_HISTORY_SUPPORT_FRACTION * len(development_seeds))
    )
    row_seeds = np.empty(groups.shape[0], dtype=int)
    row_seeds[policy_rows] = group_seeds[:, None]
    history = _history_table(
        groups=groups,
        candidate_states=candidate_states,
        actual=actual,
        eligible=eligible,
        row_seeds=row_seeds,
        development_seeds=development_seeds,
    )
    heldout_groups = np.flatnonzero(group_seeds == int(heldout_seed))
    arm_outcomes = {
        arm: np.zeros(heldout_groups.size, dtype=float) for arm in MODEL_ARMS
    }
    arm_interventions = {arm: 0 for arm in MODEL_ARMS}
    arm_harmful = {arm: 0 for arm in MODEL_ARMS}
    exact_key_group_count = 0
    support_qualified_group_count = 0
    for output_index, group_index in enumerate(heldout_groups):
        rows = np.asarray(policy_rows[group_index], dtype=int)
        reference_row = rows[np.argmax(references[rows])]
        group_key = _history_group_key(str(groups[reference_row]))
        summaries = []
        has_exact_key = False
        for row in rows:
            if row == reference_row or not bool(eligible[row]):
                continue
            values = history.get((group_key, str(candidate_states[row])))
            if values is None:
                continue
            has_exact_key = True
            summary = _history_action_summary(values)
            if summary["support_seed_count"] < minimum_support:
                continue
            summaries.append(
                {
                    **summary,
                    "row": int(row),
                    "candidate_state": str(candidate_states[row]),
                }
            )
        exact_key_group_count += int(has_exact_key)
        support_qualified_group_count += int(bool(summaries))
        for arm, conservative in (
            (MEAN_ARM, False),
            (CONSENSUS_ARM, True),
        ):
            selected = _select_action(summaries, conservative=conservative)
            if selected is None:
                continue
            value = float(actual[int(selected["row"])])
            arm_outcomes[arm][output_index] = value
            arm_interventions[arm] += 1
            arm_harmful[arm] += int(value > 0.0)
    arms = {}
    for arm in MODEL_ARMS:
        interventions = int(arm_interventions[arm])
        arms[arm] = {
            "heldout_value": float(np.mean(arm_outcomes[arm])),
            "heldout_intervention_count": interventions,
            "heldout_harmful_intervention_fraction": (
                float(arm_harmful[arm] / interventions)
                if interventions
                else 0.0
            ),
            "deployment_enabled": interventions > 0,
        }
    return {
        "heldout_seed": int(heldout_seed),
        "development_seeds": list(development_seeds),
        "minimum_history_support_seed_count": minimum_support,
        "heldout_group_count": int(heldout_groups.size),
        "exact_history_key_group_count": int(exact_key_group_count),
        "support_qualified_group_count": int(support_qualified_group_count),
        "arms": arms,
    }


def run_historical_action_identifiability(
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
    policy_rows, references, layout_seeds = _group_layout(contrast)
    if not np.array_equal(policy_rows, np.vstack(group_rows)) or not np.array_equal(
        layout_seeds, group_seeds
    ):
        raise ValueError("V133 action-group layout changed during normalization")
    groups = np.asarray(contrast.metadata["action_group_ids"], dtype=str)
    candidate_states = np.asarray(contrast.metadata["candidate_states"], dtype=str)
    if candidate_states.shape != (contrast.size,):
        raise ValueError("V133 candidate states are not row aligned")
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index], dtype=float
    ) >= -1e-12
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) < 12:
        raise ValueError("V133 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    all_seeds=all_seeds,
                    groups=groups,
                    candidate_states=candidate_states,
                    actual=actual,
                    eligible=eligible,
                    policy_rows=policy_rows,
                    references=references,
                    group_seeds=group_seeds,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    summaries = {arm: _arm_summary(folds, arm) for arm in MODEL_ARMS}
    mean_passed = bool(
        summaries[MEAN_ARM]["mean"] <= -MINIMUM_EFFECT
        and summaries[MEAN_ARM]["upper_95"] < 0.0
    )
    consensus_passed = bool(
        summaries[CONSENSUS_ARM]["mean"] <= -MINIMUM_EFFECT
        and summaries[CONSENSUS_ARM]["upper_95"] < 0.0
    )
    if consensus_passed:
        decision = "authorize_local_dynamics_successor_from_stable_history_signal"
    elif mean_passed:
        decision = "history_signal_exists_but_requires_uncertainty_model"
    else:
        decision = "exact_schedule_action_history_not_predictive"
    total_groups = int(sum(row["heldout_group_count"] for row in folds))
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-identifiability-development-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "target_counterfactual_history_used": True,
            "development_selector_seeds_per_fold": len(all_seeds) - 1,
            "heldout_selector_seeds_per_fold": 1,
            "heldout_seed_labels_used_for_lookup_or_selection": False,
            "heldout_static_action_features_used_for_pressure_constraint": True,
        },
        "history_key": "scenario_control_interval_tls_exact_candidate_state",
        "history_gate": {
            "minimum_support_fraction": MINIMUM_HISTORY_SUPPORT_FRACTION,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "conservative_minimum_improving_seed_fraction": (
                MINIMUM_HISTORY_IMPROVING_FRACTION
            ),
            "conservative_upper_95_must_be_negative": True,
            "one_sided_t_critical_df16": ONE_SIDED_T_CRITICAL_DF16,
        },
        "action_constraint": "nondecreasing_instantaneous_service_pressure",
        "coverage": {
            "outer_group_count": total_groups,
            "exact_history_key_group_count": int(
                sum(row["exact_history_key_group_count"] for row in folds)
            ),
            "support_qualified_group_count": int(
                sum(row["support_qualified_group_count"] for row in folds)
            ),
        },
        "folds": folds,
        "arm_summaries": summaries,
        "development_gate": {
            "historical_mean_lookup_passed": mean_passed,
            "historical_conservative_consensus_passed": consensus_passed,
            "passed": consensus_passed,
            "decision": decision,
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "inputs": {
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "claim_boundary": (
            "V133 is an abundant-target cross-realization identifiability "
            "screen. It is not zero-shot, source transfer, or a deployable "
            "closed-loop policy result."
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
        raise FileExistsError(f"refusing to overwrite V133 result: {args.out}")
    result = run_historical_action_identifiability(
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
                "coverage": result["coverage"],
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
