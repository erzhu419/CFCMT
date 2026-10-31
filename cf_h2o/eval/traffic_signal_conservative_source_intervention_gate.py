"""Cross-fit a conservative intervention gate over frozen V123 predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    ARTIFACT_PROTOCOL as V123_ARTIFACT_PROTOCOL,
    SOURCE_BLEND_WEIGHTS,
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v124-conservative-source-intervention-gate-v1"
RISK_MULTIPLIERS = (0.0, 0.5, 1.0)
RETENTION_FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.2)
TRUST_QUANTILES = (0.0, 0.5)
PRESSURE_LOSS_QUANTILES = (0.5, 1.0)
ONE_SIDED_T_CRITICAL = 1.725
MINIMUM_SOURCE_CONTRIBUTION = 0.0005


@dataclass(frozen=True)
class GuardProfile:
    risk_multiplier: float
    retention_fraction: float
    trust_quantile: float
    pressure_loss_quantile: float
    require_target_agreement: bool = False

    @property
    def key(self) -> str:
        values = (
            f"r{self.risk_multiplier:g}",
            f"keep{self.retention_fraction:g}",
            f"trust{self.trust_quantile:g}",
            f"pressure{self.pressure_loss_quantile:g}",
            f"agree{int(self.require_target_agreement)}",
        )
        return "__".join(values).replace(".", "p")


@dataclass(frozen=True)
class Prediction:
    score: np.ndarray
    uncertainty: np.ndarray
    trust: np.ndarray


def _load_prediction_artifact(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("V124 prediction artifact identity changed")
    payload = pickle.loads(path.read_bytes())
    if payload.get("protocol") != V123_ARTIFACT_PROTOCOL:
        raise ValueError("V124 requires the frozen V123 prediction artifact")
    return payload


def _prediction(values: Sequence[np.ndarray]) -> Prediction:
    if len(values) != 3:
        raise ValueError("prediction must contain score, uncertainty and trust")
    score, uncertainty, trust = (
        np.asarray(value, dtype=float) for value in values
    )
    if not (score.shape == uncertainty.shape == trust.shape):
        raise ValueError("prediction arrays do not share a shape")
    if not all(np.all(np.isfinite(value)) for value in (score, uncertainty, trust)):
        raise ValueError("prediction arrays contain non-finite values")
    return Prediction(score, np.maximum(uncertainty, 0.0), np.clip(trust, 0.0, 1.0))


def _blend_predictions(
    target: Prediction | None,
    source: Prediction,
    *,
    source_weight: float,
) -> Prediction:
    if target is None:
        if float(source_weight) != 1.0:
            raise ValueError("zero-target prediction requires unit source weight")
        return source
    weight = float(source_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("source weight must be in [0, 1]")
    return Prediction(
        score=(1.0 - weight) * target.score + weight * source.score,
        uncertainty=(
            (1.0 - weight) * target.uncertainty
            + weight * source.uncertainty
        ),
        trust=np.minimum(target.trust, source.trust),
    )


def _uniform_source_prediction(
    source_predictions: Mapping[str, Prediction]
) -> Prediction:
    if not source_predictions:
        raise ValueError("uniform source prediction requires sources")
    rows = tuple(source_predictions[name] for name in sorted(source_predictions))
    return Prediction(
        score=np.mean(np.vstack([row.score for row in rows]), axis=0),
        uncertainty=np.max(
            np.vstack([row.uncertainty for row in rows]), axis=0
        ),
        trust=np.min(np.vstack([row.trust for row in rows]), axis=0),
    )


def _source_candidates(
    target: Prediction | None,
    sources: Mapping[str, Prediction],
) -> dict[str, Prediction]:
    base = dict(sources)
    base["uniform_all_sources"] = _uniform_source_prediction(sources)
    weights = (1.0,) if target is None else SOURCE_BLEND_WEIGHTS
    return {
        f"{name}__source_weight_{weight:g}": _blend_predictions(
            target, source, source_weight=weight
        )
        for name, source in sorted(base.items())
        for weight in weights
    }


def _reference_and_learned_rows(
    prediction: Prediction,
    policy_rows: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(policy_rows, dtype=int)
    if rows.ndim != 2:
        raise ValueError("V124 requires a rectangular action matrix")
    reference_mask = references[rows]
    if not np.all(np.sum(reference_mask, axis=1) == 1):
        raise ValueError("each action group must have one PhasePressure row")
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(reference_mask, axis=1)
    ]
    learned_rows = rows[
        np.arange(rows.shape[0]),
        np.argmin(prediction.score[rows], axis=1),
    ]
    return reference_rows, learned_rows


def _raw_seed_values(
    actual: np.ndarray,
    learned_rows: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[int, float]:
    values = np.asarray(actual[learned_rows], dtype=float)
    return {
        int(seed): float(np.mean(values[group_seeds == seed]))
        for seed in sorted(int(value) for value in np.unique(group_seeds))
    }


def _select_source_candidate(
    candidates: Mapping[str, Prediction],
    *,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
    training_seeds: set[int],
) -> tuple[str, Prediction, dict[str, float]]:
    rows = {}
    for name, prediction in candidates.items():
        _, learned = _reference_and_learned_rows(
            prediction, policy_rows, references
        )
        by_seed = _raw_seed_values(actual, learned, group_seeds)
        rows[name] = float(
            np.mean([by_seed[seed] for seed in sorted(training_seeds)])
        )
    selected = min(rows, key=lambda name: (rows[name], name))
    return selected, candidates[selected], rows


def _guard_feature_arrays(
    prediction: Prediction,
    *,
    policy_rows: np.ndarray,
    references: np.ndarray,
    pressure_delta: np.ndarray,
    target_prediction: Prediction | None,
) -> dict[str, np.ndarray]:
    reference_rows, learned_rows = _reference_and_learned_rows(
        prediction, policy_rows, references
    )
    proposed = learned_rows != reference_rows
    advantage = (
        prediction.score[reference_rows] - prediction.score[learned_rows]
    )
    uncertainty = (
        prediction.uncertainty[reference_rows]
        + prediction.uncertainty[learned_rows]
    )
    trust = np.minimum(
        prediction.trust[reference_rows], prediction.trust[learned_rows]
    )
    pressure_loss = np.maximum(
        -np.asarray(pressure_delta[learned_rows], dtype=float), 0.0
    )
    if target_prediction is None:
        agreement = np.ones(proposed.size, dtype=bool)
    else:
        _, target_rows = _reference_and_learned_rows(
            target_prediction, policy_rows, references
        )
        agreement = learned_rows == target_rows
    return {
        "reference_rows": reference_rows,
        "learned_rows": learned_rows,
        "proposed": proposed,
        "advantage": advantage,
        "uncertainty": uncertainty,
        "trust": trust,
        "pressure_loss": pressure_loss,
        "target_agreement": agreement,
    }


def _profile_acceptance(
    features: Mapping[str, np.ndarray],
    *,
    profile: GuardProfile,
    training_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    proposed_training = training_mask & features["proposed"]
    if not np.any(proposed_training):
        return np.zeros(training_mask.size, dtype=bool), {
            "robust_advantage_threshold": float("inf"),
            "trust_threshold": float("inf"),
            "pressure_loss_threshold": -float("inf"),
        }
    robust_advantage = features["advantage"] - (
        float(profile.risk_multiplier) * features["uncertainty"]
    )
    positive_training = proposed_training & (robust_advantage > 0.0)
    if not np.any(positive_training):
        return np.zeros(training_mask.size, dtype=bool), {
            "robust_advantage_threshold": float("inf"),
            "trust_threshold": float("inf"),
            "pressure_loss_threshold": -float("inf"),
        }
    advantage_threshold = max(
        0.0,
        float(
            np.quantile(
                robust_advantage[positive_training],
                1.0 - float(profile.retention_fraction),
                method="higher",
            )
        ),
    )
    trust_threshold = float(
        np.quantile(
            features["trust"][proposed_training],
            float(profile.trust_quantile),
            method="higher",
        )
    )
    pressure_threshold = float(
        np.quantile(
            features["pressure_loss"][proposed_training],
            float(profile.pressure_loss_quantile),
            method="higher",
        )
    )
    accepted = (
        features["proposed"]
        & (robust_advantage >= advantage_threshold)
        & (features["trust"] >= trust_threshold)
        & (features["pressure_loss"] <= pressure_threshold)
    )
    if profile.require_target_agreement:
        accepted &= features["target_agreement"]
    return accepted, {
        "robust_advantage_threshold": advantage_threshold,
        "trust_threshold": trust_threshold,
        "pressure_loss_threshold": pressure_threshold,
    }


def _one_sided_upper(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float("inf")
    return float(
        np.mean(array)
        + ONE_SIDED_T_CRITICAL * np.std(array, ddof=1) / np.sqrt(array.size)
    )


def _guard_profiles(*, allow_target_agreement: bool) -> tuple[GuardProfile, ...]:
    agreement_values = (False, True) if allow_target_agreement else (False,)
    return tuple(
        GuardProfile(risk, keep, trust, pressure, agreement)
        for risk in RISK_MULTIPLIERS
        for keep in RETENTION_FRACTIONS
        for trust in TRUST_QUANTILES
        for pressure in PRESSURE_LOSS_QUANTILES
        for agreement in agreement_values
    )


def _select_guard(
    prediction: Prediction,
    *,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    pressure_delta: np.ndarray,
    group_seeds: np.ndarray,
    training_seeds: set[int],
    target_prediction: Prediction | None,
    minimum_training_interventions: int = 20,
) -> dict[str, Any]:
    features = _guard_feature_arrays(
        prediction,
        policy_rows=policy_rows,
        references=references,
        pressure_delta=pressure_delta,
        target_prediction=target_prediction,
    )
    training_mask = np.isin(group_seeds, sorted(training_seeds))
    profiles = []
    for profile in _guard_profiles(
        allow_target_agreement=target_prediction is not None
    ):
        accepted, thresholds = _profile_acceptance(
            features, profile=profile, training_mask=training_mask
        )
        accepted_training = accepted & training_mask
        values = np.where(
            accepted,
            np.asarray(actual[features["learned_rows"]], dtype=float),
            0.0,
        )
        seed_values = {
            seed: float(np.mean(values[group_seeds == seed]))
            for seed in sorted(training_seeds)
        }
        active_seeds = int(
            sum(
                bool(np.any(accepted_training & (group_seeds == seed)))
                for seed in training_seeds
            )
        )
        intervention_count = int(np.count_nonzero(accepted_training))
        upper = _one_sided_upper(list(seed_values.values()))
        admissible = bool(
            intervention_count >= int(minimum_training_interventions)
            and active_seeds >= max(2, (len(training_seeds) + 1) // 2)
            and upper < 0.0
        )
        profiles.append(
            {
                "profile": profile,
                "accepted": accepted,
                "thresholds": thresholds,
                "training_seed_values": seed_values,
                "training_mean": float(np.mean(list(seed_values.values()))),
                "training_upper_95_one_sided": upper,
                "training_intervention_count": intervention_count,
                "training_active_seed_count": active_seeds,
                "admissible": admissible,
            }
        )
    admissible = [row for row in profiles if row["admissible"]]
    if not admissible:
        return {
            "enabled": False,
            "profile": None,
            "accepted": np.zeros(group_seeds.size, dtype=bool),
            "thresholds": {},
            "training_seed_values": {
                seed: 0.0 for seed in sorted(training_seeds)
            },
            "training_mean": 0.0,
            "training_upper_95_one_sided": 0.0,
            "training_intervention_count": 0,
            "training_active_seed_count": 0,
            "candidate_profile_count": len(profiles),
            "admissible_profile_count": 0,
        }
    selected = min(
        admissible,
        key=lambda row: (
            row["training_upper_95_one_sided"],
            row["training_mean"],
            row["training_intervention_count"],
            row["profile"].key,
        ),
    )
    return {
        "enabled": True,
        "profile": asdict(selected["profile"]),
        "accepted": selected["accepted"],
        "thresholds": selected["thresholds"],
        "training_seed_values": selected["training_seed_values"],
        "training_mean": selected["training_mean"],
        "training_upper_95_one_sided": selected[
            "training_upper_95_one_sided"
        ],
        "training_intervention_count": selected[
            "training_intervention_count"
        ],
        "training_active_seed_count": selected["training_active_seed_count"],
        "candidate_profile_count": len(profiles),
        "admissible_profile_count": len(admissible),
    }


def _deployed_seed_value(
    guard: Mapping[str, Any],
    prediction: Prediction,
    *,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[float, int]:
    _, learned = _reference_and_learned_rows(prediction, policy_rows, references)
    accepted = np.asarray(guard["accepted"], dtype=bool) & seed_mask
    values = np.where(accepted, actual[learned], 0.0)
    return float(np.mean(values[seed_mask])), int(np.count_nonzero(accepted))


def _paired_dominance(
    source_guard: Mapping[str, Any],
    target_guard: Mapping[str, Any],
    *,
    training_seeds: set[int],
) -> dict[str, Any]:
    differences = np.asarray(
        [
            float(source_guard["training_seed_values"][seed])
            - float(target_guard["training_seed_values"][seed])
            for seed in sorted(training_seeds)
        ],
        dtype=float,
    )
    upper = _one_sided_upper(differences)
    mean = float(np.mean(differences))
    return {
        "mean_source_minus_target": mean,
        "upper_95_one_sided": upper,
        "source_authorized": bool(
            source_guard["enabled"]
            and mean <= -MINIMUM_SOURCE_CONTRIBUTION
            and upper < 0.0
        ),
    }


def _json_guard(guard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in guard.items()
        if key != "accepted"
    }


def run_conservative_gate(
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
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    artifact = _load_prediction_artifact(
        prediction_artifact_path,
        expected_sha256=expected_prediction_artifact_sha256,
    )
    if (
        tuple(int(value) for value in artifact.get("selector_seeds", ()))
        != tuple(int(value) for value in selector_seeds)
        or artifact.get("selector_cache_audit_sha256")
        != expected_selector_cache_audit_sha256
    ):
        raise ValueError("V124 selector identity differs from V123")
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
    feature_index = {
        str(name): index for index, name in enumerate(contrast.feature_names)
    }
    if "delta_service_pressure" not in feature_index:
        raise ValueError("V124 pressure-loss feature is missing")
    pressure_delta = np.asarray(
        contrast.features[:, feature_index["delta_service_pressure"]],
        dtype=float,
    )
    row_count = int(contrast.size)
    source_order = tuple(str(value) for value in artifact["source_group_order"])
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if all_seeds != tuple(sorted(int(value) for value in selector_seeds)):
        raise ValueError("V124 selector seed coverage changed")

    budget_results = {}
    positive_budgets = tuple(int(value) for value in artifact["target_budgets"])
    for budget in (0, *positive_budgets):
        key = str(budget)
        target = (
            None
            if budget == 0
            else _prediction(artifact["target_predictions"][key])
        )
        sources = {
            source: _prediction(artifact["source_predictions"][key][source])
            for source in source_order
        }
        if any(row.score.size != row_count for row in sources.values()) or (
            target is not None and target.score.size != row_count
        ):
            raise ValueError("V124 prediction rows differ from the selector")
        candidates = _source_candidates(target, sources)
        folds = []
        source_seed_values = []
        target_seed_values = []
        source_minus_target = []
        for heldout_seed in all_seeds:
            training_seeds = set(all_seeds) - {heldout_seed}
            candidate_name, selected_prediction, candidate_training = (
                _select_source_candidate(
                    candidates,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    group_seeds=group_seeds,
                    training_seeds=training_seeds,
                )
            )
            source_guard = _select_guard(
                selected_prediction,
                actual=actual,
                policy_rows=policy_rows,
                references=references,
                pressure_delta=pressure_delta,
                group_seeds=group_seeds,
                training_seeds=training_seeds,
                target_prediction=target,
            )
            if target is None:
                target_guard = {
                    "enabled": False,
                    "accepted": np.zeros(group_seeds.size, dtype=bool),
                    "training_seed_values": {
                        seed: 0.0 for seed in sorted(training_seeds)
                    },
                    "training_mean": 0.0,
                    "training_upper_95_one_sided": 0.0,
                }
                dominance = {
                    "mean_source_minus_target": source_guard["training_mean"],
                    "upper_95_one_sided": source_guard[
                        "training_upper_95_one_sided"
                    ],
                    "source_authorized": bool(source_guard["enabled"]),
                }
            else:
                target_guard = _select_guard(
                    target,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    pressure_delta=pressure_delta,
                    group_seeds=group_seeds,
                    training_seeds=training_seeds,
                    target_prediction=None,
                )
                dominance = _paired_dominance(
                    source_guard,
                    target_guard,
                    training_seeds=training_seeds,
                )
            heldout_mask = group_seeds == heldout_seed
            target_value, target_interventions = (
                (0.0, 0)
                if target is None
                else _deployed_seed_value(
                    target_guard,
                    target,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    seed_mask=heldout_mask,
                )
            )
            if dominance["source_authorized"]:
                source_value, source_interventions = _deployed_seed_value(
                    source_guard,
                    selected_prediction,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    seed_mask=heldout_mask,
                )
                deployed_arm = "source_guard"
            else:
                source_value = target_value
                source_interventions = target_interventions
                deployed_arm = "target_guard" if target is not None else "phase_pressure"
            folds.append(
                {
                    "heldout_seed": int(heldout_seed),
                    "selected_source_candidate": candidate_name,
                    "selected_candidate_training_raw_delta": float(
                        candidate_training[candidate_name]
                    ),
                    "source_guard": _json_guard(source_guard),
                    "target_guard": _json_guard(target_guard),
                    "dominance": dominance,
                    "deployed_arm": deployed_arm,
                    "source_policy_delta": source_value,
                    "target_policy_delta": target_value,
                    "source_minus_target": source_value - target_value,
                    "source_intervention_count": source_interventions,
                    "target_intervention_count": target_interventions,
                }
            )
            source_seed_values.append(source_value)
            target_seed_values.append(target_value)
            source_minus_target.append(source_value - target_value)

        source_bootstrap = _paired_bootstrap(source_seed_values)
        target_bootstrap = _paired_bootstrap(target_seed_values)
        contribution_bootstrap = _paired_bootstrap(source_minus_target)
        budget_results[key] = {
            "target_group_budget": budget,
            "folds": folds,
            "source_policy_vs_phase_pressure": source_bootstrap,
            "target_policy_vs_phase_pressure": target_bootstrap,
            "source_minus_target": contribution_bootstrap,
            "source_authorized_fold_count": sum(
                row["dominance"]["source_authorized"] for row in folds
            ),
            "source_intervention_count": sum(
                row["source_intervention_count"] for row in folds
            ),
            "target_intervention_count": sum(
                row["target_intervention_count"] for row in folds
            ),
            "selected_source_candidate_counts": {
                name: sum(
                    row["selected_source_candidate"] == name for row in folds
                )
                for name in sorted(
                    {row["selected_source_candidate"] for row in folds}
                )
            },
        }

    comparable = {
        budget: row
        for budget, row in budget_results.items()
        if int(budget) > 0
    }
    best_budget = min(
        comparable,
        key=lambda budget: (
            comparable[budget]["source_policy_vs_phase_pressure"]["mean"],
            int(budget),
        ),
    )
    best = comparable[best_budget]
    passes = bool(
        best["source_policy_vs_phase_pressure"]["mean"]
        <= -MINIMUM_SOURCE_CONTRIBUTION
        and best["source_policy_vs_phase_pressure"]["upper_95"] < 0.0
        and best["source_minus_target"]["mean"]
        <= -MINIMUM_SOURCE_CONTRIBUTION
        and best["source_minus_target"]["upper_95"] < 0.0
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "cross_fitted_development_not_confirmation",
        "estimand": artifact["estimand"],
        "target_name": artifact["target_name"],
        "city": artifact["city"],
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": row_count,
        "target_budgets": list(positive_budgets),
        "guard_grid": {
            "risk_multipliers": list(RISK_MULTIPLIERS),
            "retention_fractions": list(RETENTION_FRACTIONS),
            "trust_quantiles": list(TRUST_QUANTILES),
            "pressure_loss_quantiles": list(PRESSURE_LOSS_QUANTILES),
            "one_sided_t_critical": ONE_SIDED_T_CRITICAL,
            "minimum_training_interventions": 20,
            "minimum_training_active_seed_fraction": 0.5,
        },
        "budget_results": budget_results,
        "selection": {
            "best_budget": int(best_budget),
            "best_source_policy_mean": best[
                "source_policy_vs_phase_pressure"
            ]["mean"],
            "best_source_policy_upper_95": best[
                "source_policy_vs_phase_pressure"
            ]["upper_95"],
            "best_source_minus_target_mean": best["source_minus_target"][
                "mean"
            ],
            "best_source_minus_target_upper_95": best[
                "source_minus_target"
            ]["upper_95"],
            "minimum_required_effect": MINIMUM_SOURCE_CONTRIBUTION,
            "gate_passed": passes,
            "decision": (
                "authorize_frozen_fresh_city_confirmation"
                if passes
                else "retain_phase_pressure_or_target_guard_fallback"
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V124 result: {args.out}")
    result = run_conservative_gate(
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
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["selection"]["gate_passed"] else "REJECT",
                "protocol": result["protocol"],
                "selection": result["selection"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
