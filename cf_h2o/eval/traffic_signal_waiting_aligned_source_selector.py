"""Waiting-aligned causal source admission with negative-transfer fallback."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_causal_source_components import (
    MODEL_BUNDLE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (
    RESULT_PROTOCOL as PURE_WAITING_SELECTOR_CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_strict_target_only_fit import (
    MODEL_PROTOCOL as STRICT_TARGET_ONLY_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    MODEL_BUNDLE_PROTOCOL as WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL,
    STRICT_TARGET_MODEL_PROTOCOL as WAITING_ALIGNED_STRICT_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY as WAITING_ALIGNED_TARGET_ONLY_FAMILY,
    validate_architecture_matched_target_contract,
    validate_target_label_free_blend_contract,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


LEGACY_RESULT_PROTOCOL = "tsc-v112-waiting-aligned-causal-multisource-selector-v2"
PURE_WAITING_RESULT_PROTOCOL = (
    "tsc-v116-pure-waiting-causal-multisource-selector-v2"
)
# Retained for callers that validate historical v112 selector artifacts.
RESULT_PROTOCOL = LEGACY_RESULT_PROTOCOL
SOURCE_WEIGHT_GRID = (0.25, 0.5, 0.75, 1.0)
RISK_GRID = (0.0, 0.5, 1.0)
TRUST_GRID = (0.25, 0.5, 0.75)
SOURCE_SUPPORT_GRID = (0.6, 0.8, 1.0)
SOURCE_FAMILYWISE_ALPHA = 0.025
TARGET_FAMILYWISE_ALPHA = 0.025
MINIMUM_PRESSURE_IMPROVEMENT = 0.0015
MINIMUM_SOURCE_CONTRIBUTION = 0.0005
MAXIMUM_SEED_REGRESSION = 0.0075
MAXIMUM_HARMFUL_OVERRIDE_FRACTION = 0.42
MINIMUM_INTERVENTION_FRACTION = 0.0025
MAXIMUM_INTERVENTION_FRACTION = 0.05
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 112003
_PREDICTION_CONTEXT: dict[str, Any] | None = None


def _load_pure_waiting_selector_cache_audit(
    path: Path,
    *,
    expected_sha256: str,
    scenario: str,
    seeds: Sequence[int],
    collection_shards: int,
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("selector cache audit identity changed")
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    ordered_seeds = [int(value) for value in seeds]
    expected_files = len(ordered_seeds) * int(collection_shards)
    if (
        not isinstance(audit, dict)
        or audit.get("protocol") != PURE_WAITING_SELECTOR_CACHE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision")
        != "authorize_v116_pure_waiting_nested_source_selection"
        or not audit.get("gate", {}).get("passed", False)
        or audit.get("scenario") != scenario
        or [int(value) for value in audit.get("seeds", ())] != ordered_seeds
        or int(audit.get("seed_count", -1)) != len(ordered_seeds)
        or int(audit.get("shards_per_seed", -1)) != int(collection_shards)
        or int(audit.get("cache_file_count", -1)) != expected_files
        or int(audit.get("expected_cache_file_count", -1)) != expected_files
    ):
        raise ValueError("selector cache audit does not authorize V116 selection")
    return {
        "path": str(Path(path).resolve()),
        "sha256": str(expected_sha256),
        "protocol": str(audit["protocol"]),
        "decision": str(audit["decision"]),
        "cache_file_count": int(audit["cache_file_count"]),
        "seed_count": int(audit["seed_count"]),
        "shards_per_seed": int(audit["shards_per_seed"]),
    }


def _assert_pure_waiting_selector_dataset(
    dataset: Any,
    *,
    target_name: str,
) -> None:
    if (
        dataset.metadata.get("counterfactual_cost_mode") != "halted_queue"
        or dataset.metadata.get("counterfactual_estimand")
        != WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
        or dataset.metadata.get("counterfactual_cost_population")
        != "sumo_last_step_halting_number_on_controlled_lanes"
        or target_name not in dataset.targets
    ):
        raise ValueError("source selector requires pure waiting-aligned targets")
    interval_cost = np.asarray(dataset.targets["interval_cost"], dtype=float)
    selector_target = np.asarray(dataset.targets[target_name], dtype=float)
    if (
        interval_cost.shape != selector_target.shape
        or not np.all(np.isfinite(selector_target))
        or not np.all(selector_target >= 0.0)
        or not np.allclose(
            interval_cost,
            selector_target,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("source selector target is not the pure 450-second cost")


def _stable_group_rows(groups: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return lexicographically ordered groups without rescanning all rows."""

    values = np.asarray(groups, dtype=str)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("action groups must be a nonempty one-dimensional array")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    starts = np.flatnonzero(
        np.concatenate(
            (np.asarray([True]), ordered_values[1:] != ordered_values[:-1])
        )
    )
    stops = np.concatenate((starts[1:], np.asarray([values.size])))
    return ordered_values[starts], tuple(
        order[start:stop] for start, stop in zip(starts, stops, strict=True)
    )


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name(
            "traffic_signal_causal_source_weighting.py"
        ),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _profile_key(
    source_city_group: str | None,
    source_weight: float,
    risk_multiplier: float,
    minimum_context_trust: float,
    minimum_source_support: float = 0.0,
) -> str:
    source = "target_only" if source_city_group is None else source_city_group

    def token(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")

    support = (
        ""
        if float(minimum_source_support) <= 0.0
        else f"__support{token(minimum_source_support)}"
    )
    return (
        f"{source}__w{token(source_weight)}"
        f"__risk{token(risk_multiplier)}"
        f"__trust{token(minimum_context_trust)}"
        f"{support}"
    )


def _paired_confidence(
    values: Sequence[float], *, critical: float
) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("paired confidence input must contain at least two values")
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(array.size))
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "upper_confidence_bound": float(mean + critical * standard_error),
        "worst": float(np.max(array)),
    }


def _profile_training_metrics(
    profile: Mapping[str, Any], seeds: Sequence[int]
) -> dict[str, Any]:
    rows = [profile["seed_metrics"][str(int(seed))] for seed in seeds]
    group_count = int(sum(int(row["group_count"]) for row in rows))
    accepted = int(sum(int(row["accepted_override_count"]) for row in rows))
    harmful = int(sum(int(row["harmful_override_count"]) for row in rows))
    return {
        "seed_deltas": [
            float(row["mean_normalized_delta_vs_phase_pressure"])
            for row in rows
        ],
        "group_count": group_count,
        "accepted_override_count": accepted,
        "harmful_override_count": harmful,
        "intervention_fraction": accepted / max(group_count, 1),
        "harmful_override_fraction": harmful / max(accepted, 1),
    }


def select_negative_transfer_safe_profile(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    training_seeds: Sequence[int],
) -> dict[str, Any]:
    """Select a source profile or fall back without reading held-out seeds."""

    seeds = tuple(sorted(int(value) for value in training_seeds))
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("source admission requires at least two unique seeds")
    target_keys = tuple(
        sorted(
            key
            for key, row in profiles.items()
            if row["source_city_group"] is None
        )
    )
    source_keys = tuple(
        sorted(
            key
            for key, row in profiles.items()
            if row["source_city_group"] is not None
        )
    )
    if not target_keys or not source_keys:
        raise ValueError("source admission matrix is incomplete")
    expected_seeds = {str(seed) for seed in seeds}
    if any(
        not expected_seeds <= set(row["seed_metrics"])
        for row in profiles.values()
    ):
        raise ValueError("source admission profile misses a training seed")

    target_critical = float(
        NormalDist().inv_cdf(1.0 - TARGET_FAMILYWISE_ALPHA / len(target_keys))
    )
    source_critical = float(
        NormalDist().inv_cdf(1.0 - SOURCE_FAMILYWISE_ALPHA / len(source_keys))
    )
    grid: dict[str, Any] = {}
    eligible_target = []
    for key in target_keys:
        profile = profiles[key]
        aggregate = _profile_training_metrics(profile, seeds)
        pressure = _paired_confidence(
            aggregate["seed_deltas"], critical=target_critical
        )
        eligible = bool(
            aggregate["accepted_override_count"] > 0
            and pressure["mean"] <= -MINIMUM_PRESSURE_IMPROVEMENT
            and pressure["upper_confidence_bound"] <= 0.0
            and pressure["worst"] <= MAXIMUM_SEED_REGRESSION
            and aggregate["harmful_override_fraction"]
            <= MAXIMUM_HARMFUL_OVERRIDE_FRACTION
            and MINIMUM_INTERVENTION_FRACTION
            <= aggregate["intervention_fraction"]
            <= MAXIMUM_INTERVENTION_FRACTION
        )
        audit = {
            **{key: profile[key] for key in (
                "profile_key",
                "source_city_group",
                "source_weight",
                "risk_multiplier",
                "minimum_context_trust",
                "minimum_source_support",
            )},
            **aggregate,
            "pressure_comparison": pressure,
            "source_contribution_comparison": None,
            "eligible": eligible,
        }
        grid[key] = audit
        if eligible:
            eligible_target.append(audit)

    eligible_source = []
    for key in source_keys:
        profile = profiles[key]
        target_key = str(profile["target_profile_key"])
        if target_key not in profiles:
            raise ValueError("source profile has no matched target-only guard")
        aggregate = _profile_training_metrics(profile, seeds)
        target_aggregate = _profile_training_metrics(profiles[target_key], seeds)
        pressure = _paired_confidence(
            aggregate["seed_deltas"], critical=source_critical
        )
        contribution_values = np.asarray(
            aggregate["seed_deltas"], dtype=float
        ) - np.asarray(target_aggregate["seed_deltas"], dtype=float)
        contribution = _paired_confidence(
            contribution_values, critical=source_critical
        )
        eligible = bool(
            aggregate["accepted_override_count"] > 0
            and pressure["mean"] <= -MINIMUM_PRESSURE_IMPROVEMENT
            and pressure["upper_confidence_bound"] <= 0.0
            and pressure["worst"] <= MAXIMUM_SEED_REGRESSION
            and contribution["mean"] <= -MINIMUM_SOURCE_CONTRIBUTION
            and contribution["upper_confidence_bound"] <= 0.0
            and aggregate["harmful_override_fraction"]
            <= MAXIMUM_HARMFUL_OVERRIDE_FRACTION
            and MINIMUM_INTERVENTION_FRACTION
            <= aggregate["intervention_fraction"]
            <= MAXIMUM_INTERVENTION_FRACTION
        )
        audit = {
            **{name: profile[name] for name in (
                "profile_key",
                "source_city_group",
                "source_weight",
                "risk_multiplier",
                "minimum_context_trust",
                "minimum_source_support",
                "target_profile_key",
            )},
            **aggregate,
            "pressure_comparison": pressure,
            "source_contribution_comparison": contribution,
            "eligible": eligible,
        }
        grid[key] = audit
        if eligible:
            eligible_source.append(audit)

    def ordering(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        pressure = row["pressure_comparison"]
        return (
            float(pressure["mean"]),
            float(pressure["upper_confidence_bound"]),
            float(row["intervention_fraction"]),
            str(row["profile_key"]),
        )

    if eligible_source:
        selected = min(eligible_source, key=ordering)
        reason = "familywise_source_benefit_over_pressure_and_target_only"
    elif eligible_target:
        selected = min(eligible_target, key=ordering)
        reason = "negative_transfer_suppressed_target_only_fallback"
    else:
        selected = None
        reason = "negative_transfer_suppressed_phase_pressure_fallback"
    return {
        "selection_reason": reason,
        "selected_profile_key": (
            None if selected is None else str(selected["profile_key"])
        ),
        "selected_source_city_group": (
            None if selected is None else selected["source_city_group"]
        ),
        "selected_source_weight": (
            0.0 if selected is None else float(selected["source_weight"])
        ),
        "training_seeds": list(seeds),
        "target_profile_count": len(target_keys),
        "source_profile_count": len(source_keys),
        "target_simultaneous_confidence_multiplier": target_critical,
        "source_simultaneous_confidence_multiplier": source_critical,
        "grid": grid,
    }


def select_pressure_safe_source_profile(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    training_seeds: Sequence[int],
) -> dict[str, Any]:
    """Select a target-label-free source profile using pressure outcomes only."""

    seeds = tuple(sorted(int(value) for value in training_seeds))
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("zero-shot source admission requires unique training seeds")
    keys = tuple(sorted(profiles))
    if not keys or any(profiles[key]["source_city_group"] is None for key in keys):
        raise ValueError("zero-shot admission requires source-only profiles")
    expected_seeds = {str(seed) for seed in seeds}
    if any(not expected_seeds <= set(profiles[key]["seed_metrics"]) for key in keys):
        raise ValueError("zero-shot profile misses a training seed")
    critical = float(
        NormalDist().inv_cdf(1.0 - SOURCE_FAMILYWISE_ALPHA / len(keys))
    )
    grid = {}
    eligible = []
    for key in keys:
        profile = profiles[key]
        aggregate = _profile_training_metrics(profile, seeds)
        pressure = _paired_confidence(
            aggregate["seed_deltas"], critical=critical
        )
        admitted = bool(
            aggregate["accepted_override_count"] > 0
            and pressure["mean"] <= -MINIMUM_PRESSURE_IMPROVEMENT
            and pressure["upper_confidence_bound"] <= 0.0
            and pressure["worst"] <= MAXIMUM_SEED_REGRESSION
            and aggregate["harmful_override_fraction"]
            <= MAXIMUM_HARMFUL_OVERRIDE_FRACTION
            and MINIMUM_INTERVENTION_FRACTION
            <= aggregate["intervention_fraction"]
            <= MAXIMUM_INTERVENTION_FRACTION
        )
        audit = {
            **{
                name: profile[name]
                for name in (
                    "profile_key",
                    "source_city_group",
                    "source_weight",
                    "risk_multiplier",
                    "minimum_context_trust",
                    "minimum_source_support",
                )
            },
            **aggregate,
            "pressure_comparison": pressure,
            "eligible": admitted,
        }
        grid[key] = audit
        if admitted:
            eligible.append(audit)

    def ordering(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        pressure = row["pressure_comparison"]
        return (
            float(pressure["mean"]),
            float(pressure["upper_confidence_bound"]),
            float(row["intervention_fraction"]),
            str(row["profile_key"]),
        )

    selected = min(eligible, key=ordering) if eligible else None
    return {
        "selection_reason": (
            "familywise_zero_shot_source_benefit_over_phase_pressure"
            if selected is not None
            else "zero_shot_source_suppressed_phase_pressure_fallback"
        ),
        "selected_profile_key": (
            None if selected is None else str(selected["profile_key"])
        ),
        "selected_source_city_group": (
            None if selected is None else selected["source_city_group"]
        ),
        "training_seeds": list(seeds),
        "source_profile_count": len(keys),
        "source_simultaneous_confidence_multiplier": critical,
        "grid": grid,
    }


def nested_leave_one_seed_selection(
    profiles: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    ordered = tuple(sorted(int(value) for value in seeds))
    folds = []
    for heldout in ordered:
        training = tuple(seed for seed in ordered if seed != heldout)
        selection = select_negative_transfer_safe_profile(
            profiles, training_seeds=training
        )
        selected_key = selection["selected_profile_key"]
        heldout_row = (
            {
                "mean_normalized_delta_vs_phase_pressure": 0.0,
                "accepted_override_count": 0,
                "harmful_override_count": 0,
                "group_count": next(
                    iter(profiles.values())
                )["seed_metrics"][str(heldout)]["group_count"],
            }
            if selected_key is None
            else profiles[selected_key]["seed_metrics"][str(heldout)]
        )
        folds.append(
            {
                "heldout_seed": heldout,
                "training_seed_count": len(training),
                "selected_profile_key": selected_key,
                "selected_source_city_group": selection[
                    "selected_source_city_group"
                ],
                "selected_source_weight": selection["selected_source_weight"],
                "selection_reason": selection["selection_reason"],
                "heldout_mean_normalized_delta_vs_phase_pressure": float(
                    heldout_row["mean_normalized_delta_vs_phase_pressure"]
                ),
                "heldout_accepted_override_count": int(
                    heldout_row["accepted_override_count"]
                ),
                "heldout_harmful_override_count": int(
                    heldout_row["harmful_override_count"]
                ),
                "heldout_group_count": int(heldout_row["group_count"]),
            }
        )
    deltas = np.asarray(
        [
            row["heldout_mean_normalized_delta_vs_phase_pressure"]
            for row in folds
        ],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        deltas, size=(BOOTSTRAP_REPLICATES, deltas.size), replace=True
    ).mean(axis=1)
    source_folds = sum(
        row["selected_source_city_group"] is not None for row in folds
    )
    harmful_seed_fraction = float(np.mean(deltas > 0.0))
    summary = {
        "mean_normalized_delta_vs_phase_pressure": float(np.mean(deltas)),
        "worst_seed_mean_normalized_delta": float(np.max(deltas)),
        "harmful_seed_fraction": harmful_seed_fraction,
        "source_selected_fold_count": int(source_folds),
        "source_selected_fold_fraction": float(source_folds / len(folds)),
        "bootstrap_95pct_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_nonnegative": float(np.mean(draws >= 0.0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    summary["source_transfer_gate_passed"] = bool(
        summary["source_selected_fold_fraction"] >= 0.5
        and summary["mean_normalized_delta_vs_phase_pressure"]
        <= -MINIMUM_PRESSURE_IMPROVEMENT
        and summary["bootstrap_95pct_ci"][1] < 0.0
        and summary["worst_seed_mean_normalized_delta"]
        <= MAXIMUM_SEED_REGRESSION
        and harmful_seed_fraction <= 0.25
    )
    return {"folds": folds, "summary": summary}


def nested_leave_one_seed_pressure_selection(
    profiles: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Evaluate B0 source admission without target transition labels."""

    ordered = tuple(sorted(int(value) for value in seeds))
    folds = []
    for heldout in ordered:
        training = tuple(seed for seed in ordered if seed != heldout)
        selection = select_pressure_safe_source_profile(
            profiles, training_seeds=training
        )
        selected_key = selection["selected_profile_key"]
        heldout_row = (
            {
                "mean_normalized_delta_vs_phase_pressure": 0.0,
                "accepted_override_count": 0,
                "harmful_override_count": 0,
                "group_count": next(iter(profiles.values()))["seed_metrics"][
                    str(heldout)
                ]["group_count"],
            }
            if selected_key is None
            else profiles[selected_key]["seed_metrics"][str(heldout)]
        )
        folds.append(
            {
                "heldout_seed": heldout,
                "training_seed_count": len(training),
                "selected_profile_key": selected_key,
                "selected_source_city_group": selection[
                    "selected_source_city_group"
                ],
                "selection_reason": selection["selection_reason"],
                "heldout_mean_normalized_delta_vs_phase_pressure": float(
                    heldout_row["mean_normalized_delta_vs_phase_pressure"]
                ),
                "heldout_accepted_override_count": int(
                    heldout_row["accepted_override_count"]
                ),
                "heldout_harmful_override_count": int(
                    heldout_row["harmful_override_count"]
                ),
                "heldout_group_count": int(heldout_row["group_count"]),
            }
        )
    deltas = np.asarray(
        [
            row["heldout_mean_normalized_delta_vs_phase_pressure"]
            for row in folds
        ],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    draws = rng.choice(
        deltas, size=(BOOTSTRAP_REPLICATES, deltas.size), replace=True
    ).mean(axis=1)
    source_folds = sum(
        row["selected_source_city_group"] is not None for row in folds
    )
    harmful_seed_fraction = float(np.mean(deltas > 0.0))
    summary = {
        "mean_normalized_delta_vs_phase_pressure": float(np.mean(deltas)),
        "worst_seed_mean_normalized_delta": float(np.max(deltas)),
        "harmful_seed_fraction": harmful_seed_fraction,
        "source_selected_fold_count": int(source_folds),
        "source_selected_fold_fraction": float(source_folds / len(folds)),
        "bootstrap_95pct_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_nonnegative": float(np.mean(draws >= 0.0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + 1,
    }
    summary["source_transfer_gate_passed"] = bool(
        summary["source_selected_fold_fraction"] >= 0.5
        and summary["mean_normalized_delta_vs_phase_pressure"]
        <= -MINIMUM_PRESSURE_IMPROVEMENT
        and summary["bootstrap_95pct_ci"][1] < 0.0
        and summary["worst_seed_mean_normalized_delta"]
        <= MAXIMUM_SEED_REGRESSION
        and harmful_seed_fraction <= 0.25
    )
    return {"folds": folds, "summary": summary}


def _seed_metrics(
    *,
    seeds: np.ndarray,
    accepted: np.ndarray,
    normalized_delta_if_accepted: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for seed in sorted(int(value) for value in np.unique(seeds)):
        rows = seeds == seed
        accepted_rows = rows & accepted
        selected_delta = np.where(accepted[rows], normalized_delta_if_accepted[rows], 0.0)
        result[str(seed)] = {
            "group_count": int(np.count_nonzero(rows)),
            "accepted_override_count": int(np.count_nonzero(accepted_rows)),
            "harmful_override_count": int(
                np.count_nonzero(
                    accepted_rows & (normalized_delta_if_accepted > 0.0)
                )
            ),
            "mean_normalized_delta_vs_phase_pressure": float(
                np.mean(selected_delta)
            ),
        }
    return result


def _candidate_profiles(
    *,
    source_city_group: str | None,
    source_weight: float,
    score: np.ndarray,
    uncertainty: np.ndarray,
    trust: np.ndarray,
    group_rows: Sequence[np.ndarray],
    pressure_rows: np.ndarray,
    group_seeds: np.ndarray,
    normalized_actual: np.ndarray,
    source_support_scores: np.ndarray | None = None,
    minimum_source_support_values: Sequence[float] = (0.0,),
) -> tuple[dict[str, Any], dict[str, Any]]:
    learned_rows = np.asarray(
        [int(rows[int(np.argmin(score[rows]))]) for rows in group_rows], dtype=int
    )
    differs = learned_rows != pressure_rows
    predicted_delta = score[learned_rows] - score[pressure_rows]
    predicted_uncertainty = uncertainty[learned_rows] + uncertainty[pressure_rows]
    predicted_trust = np.minimum(trust[learned_rows], trust[pressure_rows])
    actual_delta = normalized_actual[learned_rows]
    if source_support_scores is None:
        source_support = np.ones(len(group_rows), dtype=float)
    else:
        support_scores = np.asarray(source_support_scores, dtype=float)
        if support_scores.ndim != 2 or support_scores.shape[1] != score.size:
            raise ValueError("source support scores are not action-row aligned")
        source_support = np.mean(
            support_scores[:, learned_rows]
            < support_scores[:, pressure_rows],
            axis=0,
        )
    profiles = {}
    for risk in RISK_GRID:
        for minimum_trust in TRUST_GRID:
            for minimum_support in minimum_source_support_values:
                accepted = (
                    differs
                    & (predicted_trust >= float(minimum_trust))
                    & (source_support >= float(minimum_support))
                    & (
                        predicted_delta
                        + float(risk) * predicted_uncertainty
                        < 0.0
                    )
                )
                key = _profile_key(
                    source_city_group,
                    source_weight,
                    risk,
                    minimum_trust,
                    minimum_support,
                )
                target_key = _profile_key(None, 0.0, risk, minimum_trust)
                profiles[key] = {
                    "profile_key": key,
                    "source_city_group": source_city_group,
                    "source_weight": float(source_weight),
                    "risk_multiplier": float(risk),
                    "minimum_context_trust": float(minimum_trust),
                    "minimum_source_support": float(minimum_support),
                    "target_profile_key": (
                        None if source_city_group is None else target_key
                    ),
                    "seed_metrics": _seed_metrics(
                        seeds=group_seeds,
                        accepted=accepted,
                        normalized_delta_if_accepted=actual_delta,
                    ),
                }
    compatibility = {
        "proposal_difference_count": int(np.count_nonzero(differs)),
        "proposal_difference_fraction": float(np.mean(differs)),
        "predicted_improvement_fraction": float(np.mean(predicted_delta < 0.0)),
        "mean_predicted_delta_vs_phase_pressure": float(
            np.mean(predicted_delta)
        ),
        "mean_prediction_uncertainty": float(np.mean(predicted_uncertainty)),
        "mean_context_trust": float(np.mean(predicted_trust)),
        "mean_source_support_fraction": float(np.mean(source_support)),
    }
    return profiles, compatibility


def _load_models(
    *,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    strict_target_only_model_path: Path,
    expected_strict_target_only_model_sha256: str,
    city: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    if _sha256(component_bundle_path) != expected_component_bundle_sha256:
        raise ValueError("source-component bundle identity changed")
    if _sha256(strict_target_only_model_path) != expected_strict_target_only_model_sha256:
        raise ValueError("strict target-only model identity changed")
    bundle = pickle.loads(Path(component_bundle_path).read_bytes())
    strict = pickle.loads(Path(strict_target_only_model_path).read_bytes())
    legacy_contract = bool(
        bundle.get("protocol") == MODEL_BUNDLE_PROTOCOL
        and strict.get("protocol") == STRICT_TARGET_ONLY_MODEL_PROTOCOL
        and strict.get("family") == TARGET_ONLY_FAMILY
    )
    waiting_aligned_contract = bool(
        bundle.get("protocol") == WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL
        and strict.get("protocol") == WAITING_ALIGNED_STRICT_MODEL_PROTOCOL
        and strict.get("family") == WAITING_ALIGNED_TARGET_ONLY_FAMILY
        and bundle.get("target_name") == "prefix_mean_cost_450s"
        and strict.get("target_name") == "prefix_mean_cost_450s"
        and int(bundle.get("target_group_budget", -1)) == 100
        and int(strict.get("target_group_budget", -1)) == 100
        and bundle.get("adaptation_contract_sha256")
        == strict.get("adaptation_contract_sha256")
    )
    if (
        not (legacy_contract or waiting_aligned_contract)
        or bundle.get("city") != city
        or strict.get("city") != city
        or bundle.get("main_model_sha256") != strict.get("main_model_sha256")
        or str(bundle["prior_policy"]) != str(strict["prior_policy"])
        or tuple(bundle.get("selected_group_ids", ()))
        != tuple(strict.get("selected_group_ids", ()))
        or int(strict.get("fit_diagnostics", {}).get("source_row_count_consumed", -1))
        != 0
    ):
        raise ValueError("waiting-aligned source model contract changed")
    if waiting_aligned_contract:
        validate_target_label_free_blend_contract(bundle)
        validate_architecture_matched_target_contract(strict)
        if (
            strict.get("selected_candidate") != bundle.get("selected_candidate")
            or strict.get("blend_candidate_selection_sha256")
            != bundle.get("candidate_selection_sha256")
        ):
            raise ValueError(
                "pure-waiting target-only/source component architecture differs"
            )
    selected_candidate = str(bundle["selected_candidate"])
    component_payloads = bundle["component_models"]
    prior_specs = [models.prior_spec for models in component_payloads.values()]
    if (
        not prior_specs
        or any(
            str(spec.key) != str(strict["prior_policy"])
            for spec in prior_specs
        )
    ):
        raise ValueError("source-component prior policies changed")
    prior_spec = prior_specs[0]
    source_models = {
        str(group): FrozenAnchoredBlendModel(
            anchor_model=models.family_models[ANCHOR_FAMILY],
            correction_model=models.family_models[CORRECTION_FAMILY],
            candidate=selected_candidate,
            anchor_objective_mode=models.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=models.objective_modes[CORRECTION_FAMILY],
        )
        for group, models in component_payloads.items()
    }
    return bundle, strict, source_models, prior_spec


def _load_zero_shot_models(
    *,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    city: str,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if _sha256(component_bundle_path) != expected_component_bundle_sha256:
        raise ValueError("zero-shot source-component bundle identity changed")
    bundle = pickle.loads(Path(component_bundle_path).read_bytes())
    component_payloads = bundle.get("component_models", {})
    if (
        bundle.get("protocol") != WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL
        or bundle.get("city") != city
        or bundle.get("target_name") != "prefix_mean_cost_450s"
        or int(bundle.get("target_group_budget", -1)) != 0
        or tuple(bundle.get("selected_group_ids", ()))
        or not component_payloads
        or any(
            int(models.diagnostics.get("target_adaptation_groups", -1)) != 0
            for models in component_payloads.values()
        )
    ):
        raise ValueError("zero-shot source model contract changed")
    validate_target_label_free_blend_contract(bundle)
    prior_specs = [models.prior_spec for models in component_payloads.values()]
    if any(
        str(spec.key) != str(bundle["prior_policy"]) for spec in prior_specs
    ):
        raise ValueError("zero-shot source prior policies changed")
    selected_candidate = str(bundle["selected_candidate"])
    source_models = {
        str(group): FrozenAnchoredBlendModel(
            anchor_model=models.family_models[ANCHOR_FAMILY],
            correction_model=models.family_models[CORRECTION_FAMILY],
            candidate=selected_candidate,
            anchor_objective_mode=models.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=models.objective_modes[CORRECTION_FAMILY],
        )
        for group, models in component_payloads.items()
    }
    return bundle, source_models, prior_specs[0]


def _predict_source_worker(
    source_group: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if _PREDICTION_CONTEXT is None:
        raise RuntimeError("source prediction process context is missing")
    from threadpoolctl import threadpool_limits

    model = _PREDICTION_CONTEXT["source_models"][source_group]
    dataset = _PREDICTION_CONTEXT["model_contrast"]
    with threadpool_limits(limits=1):
        score, uncertainty, trust, _ = _group_adjusted_scores(
            dataset,
            model.predict(dataset),
            objective_mode="control_only",
        )
    return source_group, score, uncertainty, trust


def _predict_source_models(
    *,
    source_models: Mapping[str, Any],
    model_contrast: Any,
    workers: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = tuple(sorted(source_models))
    actual_workers = min(max(int(workers), 1), len(groups))
    if actual_workers == 1:
        global _PREDICTION_CONTEXT
        _PREDICTION_CONTEXT = {
            "source_models": source_models,
            "model_contrast": model_contrast,
        }
        try:
            rows = [_predict_source_worker(group) for group in groups]
        finally:
            _PREDICTION_CONTEXT = None
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("parallel source prediction requires Linux fork")
        _PREDICTION_CONTEXT = {
            "source_models": source_models,
            "model_contrast": model_contrast,
        }
        try:
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                rows = list(pool.map(_predict_source_worker, groups))
        finally:
            _PREDICTION_CONTEXT = None
    return {
        group: (score, uncertainty, trust)
        for group, score, uncertainty, trust in rows
    }


def _build_zero_shot_profiles(
    *,
    source_predictions: Mapping[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    group_rows: Sequence[np.ndarray],
    pressure_rows: np.ndarray,
    group_seeds: np.ndarray,
    normalized_actual: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = {}
    compatibility = {}
    for source_group, values in sorted(source_predictions.items()):
        score, uncertainty, trust = values
        candidate, diagnostics = _candidate_profiles(
            source_city_group=f"b0_{source_group}",
            source_weight=1.0,
            score=score,
            uncertainty=uncertainty,
            trust=trust,
            group_rows=group_rows,
            pressure_rows=pressure_rows,
            group_seeds=group_seeds,
            normalized_actual=normalized_actual,
        )
        profiles.update(candidate)
        compatibility[f"b0_{source_group}"] = diagnostics
    source_order = tuple(sorted(source_predictions))
    score_stack = np.vstack(
        [source_predictions[group][0] for group in source_order]
    )
    uncertainty_stack = np.vstack(
        [source_predictions[group][1] for group in source_order]
    )
    trust_stack = np.vstack(
        [source_predictions[group][2] for group in source_order]
    )
    ensembles = {
        "b0_all_sources_mean": (
            np.mean(score_stack, axis=0),
            np.mean(uncertainty_stack, axis=0),
            np.min(trust_stack, axis=0),
        ),
        "b0_all_sources_median": (
            np.median(score_stack, axis=0),
            np.median(uncertainty_stack, axis=0),
            np.min(trust_stack, axis=0),
        ),
        "b0_all_sources_worst": (
            np.max(score_stack, axis=0),
            np.max(uncertainty_stack, axis=0),
            np.min(trust_stack, axis=0),
        ),
    }
    for name, values in ensembles.items():
        score, uncertainty, trust = values
        candidate, diagnostics = _candidate_profiles(
            source_city_group=name,
            source_weight=1.0,
            score=score,
            uncertainty=uncertainty,
            trust=trust,
            group_rows=group_rows,
            pressure_rows=pressure_rows,
            group_seeds=group_seeds,
            normalized_actual=normalized_actual,
            source_support_scores=score_stack,
            minimum_source_support_values=SOURCE_SUPPORT_GRID,
        )
        profiles.update(candidate)
        compatibility[name] = diagnostics
    return profiles, compatibility


def run_waiting_aligned_source_selector(
    *,
    cache_root: Path,
    manifest_path: Path,
    scenario: str,
    seeds: Sequence[int],
    collection_shards: int,
    target_name: str,
    cache_workers: int,
    prediction_workers: int,
    conversion_root: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    strict_target_only_model_path: Path,
    expected_strict_target_only_model_sha256: str,
    city: str,
    zero_shot_component_bundle_path: Path | None = None,
    expected_zero_shot_component_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    selector_cache_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=scenario,
        seeds=seeds,
        collection_shards=collection_shards,
    )
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    if scenario not in manifest.sumocfgs:
        raise ValueError("waiting-aligned source scenario is absent from manifest")
    selected_manifest = replace(
        manifest,
        scenarios=tuple(
            row for row in manifest.scenarios if row.scenario == scenario
        ),
    )
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        selected_manifest,
        seeds=tuple(int(value) for value in seeds),
        collection_shards=int(collection_shards),
        workers=int(cache_workers),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=city)
    _assert_pure_waiting_selector_dataset(dataset, target_name=target_name)
    bundle, strict, source_models, prior_spec = _load_models(
        component_bundle_path=component_bundle_path,
        expected_component_bundle_sha256=expected_component_bundle_sha256,
        strict_target_only_model_path=strict_target_only_model_path,
        expected_strict_target_only_model_sha256=expected_strict_target_only_model_sha256,
        city=city,
    )
    zero_shot_bundle = None
    zero_shot_models = None
    if (
        zero_shot_component_bundle_path is None
    ) != (
        expected_zero_shot_component_bundle_sha256 is None
    ):
        raise ValueError("zero-shot source bundle arguments are incomplete")
    if zero_shot_component_bundle_path is not None:
        zero_shot_bundle, zero_shot_models, zero_shot_prior = (
            _load_zero_shot_models(
                component_bundle_path=zero_shot_component_bundle_path,
                expected_component_bundle_sha256=str(
                    expected_zero_shot_component_bundle_sha256
                ),
                city=city,
            )
        )
        if (
            str(zero_shot_prior.key) != str(prior_spec.key)
            or zero_shot_bundle.get("adaptation_contract_sha256")
            != bundle.get("adaptation_contract_sha256")
            or zero_shot_bundle.get("candidate_selection_sha256")
            != bundle.get("candidate_selection_sha256")
        ):
            raise ValueError("B0 and B100 model contracts are not aligned")
    model_contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    pressure_contrast = build_action_contrast_dataset(
        dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = np.asarray(action_group_ids(model_contrast), dtype=str)
    if not np.array_equal(groups, action_group_ids(pressure_contrast).astype(str)):
        raise ValueError("model and pressure contrasts changed action-row order")
    unique_groups, group_rows = _stable_group_rows(groups)
    pressure_is_reference = np.asarray(
        pressure_contrast.metadata["is_reference"], dtype=bool
    )
    pressure_rows = np.asarray(
        [int(rows[pressure_is_reference[rows]][0]) for rows in group_rows], dtype=int
    )
    if any(np.count_nonzero(pressure_is_reference[rows]) != 1 for rows in group_rows):
        raise ValueError("phase-pressure contrast has an invalid reference group")
    group_seeds = np.asarray(
        [parse_action_group_seed(str(group)) for group in unique_groups], dtype=int
    )
    if set(group_seeds.tolist()) != set(int(value) for value in seeds):
        raise ValueError("waiting-aligned source selector seed coverage changed")
    absolute_target = np.asarray(dataset.targets[target_name], dtype=float)
    pressure_target = np.asarray(pressure_contrast.targets[target_name], dtype=float)
    normalized_actual = np.zeros(dataset.size, dtype=float)
    for rows in group_rows:
        scale = action_group_range(absolute_target[rows])
        normalized_actual[rows] = pressure_target[rows] / scale

    target_score, target_uncertainty, target_trust, _ = _group_adjusted_scores(
        model_contrast,
        strict["model"].predict(model_contrast),
        objective_mode=str(strict["objective_mode"]),
    )
    profiles, target_compatibility = _candidate_profiles(
        source_city_group=None,
        source_weight=0.0,
        score=target_score,
        uncertainty=target_uncertainty,
        trust=target_trust,
        group_rows=group_rows,
        pressure_rows=pressure_rows,
        group_seeds=group_seeds,
        normalized_actual=normalized_actual,
    )
    compatibility = {"target_only": target_compatibility}
    source_score_audit = {}
    source_predictions = _predict_source_models(
        source_models=source_models,
        model_contrast=model_contrast,
        workers=prediction_workers,
    )
    zero_shot_result = None
    if zero_shot_models is not None:
        zero_predictions = _predict_source_models(
            source_models=zero_shot_models,
            model_contrast=model_contrast,
            workers=prediction_workers,
        )
        zero_profiles, zero_compatibility = _build_zero_shot_profiles(
            source_predictions=zero_predictions,
            group_rows=group_rows,
            pressure_rows=pressure_rows,
            group_seeds=group_seeds,
            normalized_actual=normalized_actual,
        )
        zero_nested = nested_leave_one_seed_pressure_selection(
            zero_profiles, seeds
        )
        zero_full = select_pressure_safe_source_profile(
            zero_profiles, training_seeds=seeds
        )
        zero_shot_result = {
            "model_bundle_path": str(
                Path(zero_shot_component_bundle_path).resolve()
            ),
            "model_bundle_sha256": str(
                expected_zero_shot_component_bundle_sha256
            ),
            "target_transition_label_count": 0,
            "profile_count": len(zero_profiles),
            "profile_definitions": {
                key: {
                    name: value
                    for name, value in row.items()
                    if name != "seed_metrics"
                }
                for key, row in zero_profiles.items()
            },
            "compatibility_diagnostics": zero_compatibility,
            "nested_selection": zero_nested,
            "full_development_selection": zero_full,
            "source_transfer_gate_passed": bool(
                zero_nested["summary"]["source_transfer_gate_passed"]
                and zero_full["selected_source_city_group"] is not None
            ),
        }
    for source_group, values in sorted(source_predictions.items()):
        source_score, source_uncertainty, source_trust = values
        target_std = float(np.std(target_score))
        source_std = float(np.std(source_score))
        score_correlation = (
            float(np.corrcoef(target_score, source_score)[0, 1])
            if target_std > 0.0 and source_std > 0.0
            else None
        )
        source_score_audit[source_group] = {
            "target_score_correlation": score_correlation,
            "mean_absolute_score_difference": float(
                np.mean(np.abs(source_score - target_score))
            ),
            "mean_source_context_trust": float(np.mean(source_trust)),
        }
        for weight in SOURCE_WEIGHT_GRID:
            score = target_score + weight * (source_score - target_score)
            uncertainty = (
                (1.0 - weight) * target_uncertainty
                + weight * source_uncertainty
            )
            trust = (
                source_trust
                if weight == 1.0
                else np.minimum(target_trust, source_trust)
            )
            candidate, candidate_compatibility = _candidate_profiles(
                source_city_group=source_group,
                source_weight=weight,
                score=score,
                uncertainty=uncertainty,
                trust=trust,
                group_rows=group_rows,
                pressure_rows=pressure_rows,
                group_seeds=group_seeds,
                normalized_actual=normalized_actual,
            )
            profiles.update(candidate)
            compatibility[f"{source_group}__w{weight:g}"] = candidate_compatibility

    source_order = tuple(sorted(source_predictions))
    source_score_stack = np.vstack(
        [source_predictions[group][0] for group in source_order]
    )
    source_uncertainty_stack = np.vstack(
        [source_predictions[group][1] for group in source_order]
    )
    source_trust_stack = np.vstack(
        [source_predictions[group][2] for group in source_order]
    )
    ensemble_values = {
        "all_sources_mean": (
            np.mean(source_score_stack, axis=0),
            np.mean(source_uncertainty_stack, axis=0),
            np.min(source_trust_stack, axis=0),
        ),
        "all_sources_median": (
            np.median(source_score_stack, axis=0),
            np.median(source_uncertainty_stack, axis=0),
            np.min(source_trust_stack, axis=0),
        ),
        "all_sources_worst": (
            np.max(source_score_stack, axis=0),
            np.max(source_uncertainty_stack, axis=0),
            np.min(source_trust_stack, axis=0),
        ),
    }
    for ensemble_name, values in ensemble_values.items():
        source_score, source_uncertainty, source_trust = values
        for weight in SOURCE_WEIGHT_GRID:
            score = target_score + weight * (source_score - target_score)
            uncertainty = (
                (1.0 - weight) * target_uncertainty
                + weight * source_uncertainty
            )
            trust = np.minimum(target_trust, source_trust)
            candidate, candidate_compatibility = _candidate_profiles(
                source_city_group=ensemble_name,
                source_weight=weight,
                score=score,
                uncertainty=uncertainty,
                trust=trust,
                group_rows=group_rows,
                pressure_rows=pressure_rows,
                group_seeds=group_seeds,
                normalized_actual=normalized_actual,
                source_support_scores=source_score_stack,
                minimum_source_support_values=SOURCE_SUPPORT_GRID,
            )
            profiles.update(candidate)
            compatibility[f"{ensemble_name}__w{weight:g}"] = (
                candidate_compatibility
            )

    nested = nested_leave_one_seed_selection(profiles, seeds)
    full_selection = select_negative_transfer_safe_profile(
        profiles, training_seeds=seeds
    )
    full_selected_is_source = full_selection["selected_source_city_group"] is not None
    gate_passed = bool(
        nested["summary"]["source_transfer_gate_passed"]
        and full_selected_is_source
    )
    compact_profiles = {
        key: {
            name: value
            for name, value in row.items()
            if name != "seed_metrics"
        }
        for key, row in profiles.items()
    }
    estimand_aligned = bool(
        bundle.get("protocol") == WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL
    )
    return {
        "protocol": (
            PURE_WAITING_RESULT_PROTOCOL
            if estimand_aligned
            else LEGACY_RESULT_PROTOCOL
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "pre-registered_v116_pure_waiting_b0_b100_redevelopment"
            if estimand_aligned
            else "disclosed_post-v112_b2_multisource_consensus_redevelopment"
        ),
        "source_sha256": _source_sha256(),
        "city": city,
        "scenario": scenario,
        "seeds": [int(value) for value in seeds],
        "collection_shards": int(collection_shards),
        "target_name": target_name,
        "cache_audit": cache_audit,
        "selector_cache_audit": selector_cache_audit,
        "action_group_count": len(group_rows),
        "action_count_per_group": sorted({len(rows) for rows in group_rows}),
        "information_budget": {
            "target_model_fit": (
                "strict_target_only_B100_450s_waiting_labels"
                if bundle.get("protocol") == WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL
                else "strict_target_only_B100_one_step_labels"
            ),
            "source_component_fit": (
                "one_source_city_450s_labels_plus_same_B100_450s_target_pool"
                if bundle.get("protocol") == WAITING_ALIGNED_MODEL_BUNDLE_PROTOCOL
                else "one_source_city_plus_same_B100_target_pool"
            ),
            "selector_labels": target_name,
            "selector_protocol": "nested_leave_one_simulator_seed_out",
            "closed_loop_results_consumed": False,
            "multisource_aggregation": ["mean", "median", "worst_source"],
            "blend_candidate_selection": (
                dict(bundle["blend_candidate_selection"])
                if estimand_aligned
                else None
            ),
        },
        "frozen_artifacts": {
            "component_bundle_path": str(Path(component_bundle_path).resolve()),
            "component_bundle_sha256": expected_component_bundle_sha256,
            "strict_target_only_model_path": str(
                Path(strict_target_only_model_path).resolve()
            ),
            "strict_target_only_model_sha256": expected_strict_target_only_model_sha256,
            "main_model_sha256": str(bundle["main_model_sha256"]),
            "model_bundle_protocol": str(bundle["protocol"]),
            "strict_target_model_protocol": str(strict["protocol"]),
            "adaptation_contract_sha256": bundle.get(
                "adaptation_contract_sha256"
            ),
            "prior_policy": str(strict["prior_policy"]),
            "selected_candidate": str(bundle["selected_candidate"]),
            "candidate_selection_sha256": bundle.get(
                "candidate_selection_sha256"
            ),
            "source_city_groups": sorted(source_models),
        },
        "selection_thresholds": {
            "source_weight_grid": list(SOURCE_WEIGHT_GRID),
            "risk_grid": list(RISK_GRID),
            "trust_grid": list(TRUST_GRID),
            "source_support_grid": list(SOURCE_SUPPORT_GRID),
            "source_familywise_alpha": SOURCE_FAMILYWISE_ALPHA,
            "target_familywise_alpha": TARGET_FAMILYWISE_ALPHA,
            "minimum_pressure_improvement": MINIMUM_PRESSURE_IMPROVEMENT,
            "minimum_source_contribution": MINIMUM_SOURCE_CONTRIBUTION,
            "maximum_seed_regression": MAXIMUM_SEED_REGRESSION,
            "maximum_harmful_override_fraction": MAXIMUM_HARMFUL_OVERRIDE_FRACTION,
            "minimum_intervention_fraction": MINIMUM_INTERVENTION_FRACTION,
            "maximum_intervention_fraction": MAXIMUM_INTERVENTION_FRACTION,
        },
        "profile_count": len(profiles),
        "prediction_workers": min(
            max(int(prediction_workers), 1), len(source_models)
        ),
        "profile_definitions": compact_profiles,
        "compatibility_diagnostics": compatibility,
        "source_score_audit": source_score_audit,
        "nested_selection": nested,
        "full_development_selection": full_selection,
        "source_transfer_gate_passed": gate_passed,
        "zero_shot_source_admission": zero_shot_result,
        "decision": (
            "freeze_pure_waiting_multisource_profile_for_v116"
            if gate_passed
            else "reject_current_multisource_components_and_retain_safe_fallback"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--collection-shards", type=int, required=True)
    parser.add_argument("--target-name", default="prefix_mean_cost_450s")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--prediction-workers", type=int, default=7)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--component-bundle", type=Path, required=True)
    parser.add_argument("--component-bundle-sha256", required=True)
    parser.add_argument("--strict-target-only-model", type=Path, required=True)
    parser.add_argument("--strict-target-only-model-sha256", required=True)
    parser.add_argument("--zero-shot-component-bundle", type=Path)
    parser.add_argument("--zero-shot-component-bundle-sha256")
    parser.add_argument("--city", default="jinan")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_waiting_aligned_source_selector(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        scenario=args.scenario,
        seeds=args.seeds,
        collection_shards=args.collection_shards,
        target_name=args.target_name,
        cache_workers=args.cache_workers,
        prediction_workers=args.prediction_workers,
        conversion_root=args.conversion_root,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=(
            args.selector_cache_audit_sha256
        ),
        component_bundle_path=args.component_bundle,
        expected_component_bundle_sha256=args.component_bundle_sha256,
        strict_target_only_model_path=args.strict_target_only_model,
        expected_strict_target_only_model_sha256=args.strict_target_only_model_sha256,
        city=args.city,
        zero_shot_component_bundle_path=args.zero_shot_component_bundle,
        expected_zero_shot_component_bundle_sha256=(
            args.zero_shot_component_bundle_sha256
        ),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "decision": result["decision"],
                "source_transfer_gate_passed": result[
                    "source_transfer_gate_passed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
