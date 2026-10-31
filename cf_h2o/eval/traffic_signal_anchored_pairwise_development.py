"""Evaluate frozen anchored pairwise candidates on one development network."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    CONFIDENCE_MULTIPLIERS,
    CONSTANT_ALPHAS,
    DISAGREEMENT_CAPS,
    constant_candidate_key,
    local_candidate_key,
)
from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


ANCHOR_FAMILY = GROUP_NORMALIZED_RIGID_FAMILY
CORRECTION_FAMILY = PAIRWISE_PREFERENCE_FAMILY
BASE_FAMILIES = (ANCHOR_FAMILY, CORRECTION_FAMILY)
TARGET_GROUP_BUDGET = 60
SOURCE_SEEDS = (2027, 3037, 4047)
COLLECTION_SHARDS = 16
EXPECTED_CACHE_FILE_COUNT = 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
SCALE_FLOOR = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    development = payload.get("development", {})
    expected_grid = {
        "constant_alpha_grid": list(CONSTANT_ALPHAS),
        "local_disagreement_cap_grid": list(DISAGREEMENT_CAPS),
        "local_confidence_multiplier_grid": list(CONFIDENCE_MULTIPLIERS),
    }
    for key, expected in expected_grid.items():
        observed = [float(value) for value in development.get(key, ())]
        if observed != expected:
            raise ValueError(f"v39 {key} differs from executable grid")
    if int(development.get("target_group_budget", -1)) != TARGET_GROUP_BUDGET:
        raise ValueError("v39 target group budget changed")
    evaluation_targets = [
        str(value) for value in development.get("evaluation_targets", ())
    ]
    if evaluation_targets and (
        len(evaluation_targets) != len(set(evaluation_targets))
        or not evaluation_targets
    ):
        raise ValueError("v39b evaluation targets must be unique and nonempty")
    exclusions = development.get("source_only_capacity_exclusions", {})
    if exclusions:
        if not isinstance(exclusions, dict):
            raise ValueError("v39b source-only exclusions must be an object")
        if set(evaluation_targets) & set(exclusions):
            raise ValueError("v39b evaluation and source-only target sets overlap")
        if any(
            int(count) < 0 or int(count) >= TARGET_GROUP_BUDGET
            for count in exclusions.values()
        ):
            raise ValueError("v39b exclusions must have capacity below B60")
    return payload


def _local_group_alpha(
    *,
    anchor_score: np.ndarray,
    correction_score: np.ndarray,
    anchor_uncertainty: np.ndarray,
    correction_uncertainty: np.ndarray,
    group_rows: np.ndarray,
    disagreement_cap: float,
    confidence_multiplier: float,
) -> tuple[float, dict[str, float | int | bool]]:
    rows = np.asarray(group_rows, dtype=int)
    anchor_values = np.asarray(anchor_score[rows], dtype=float)
    correction_values = np.asarray(correction_score[rows], dtype=float)
    delta = correction_values - anchor_values
    scale = max(
        float(np.ptp(anchor_values)),
        float(np.median(np.asarray(anchor_uncertainty[rows], dtype=float))),
        SCALE_FLOOR,
    )
    disagreement_ratio = float(np.ptp(delta) / scale)
    anchor_local = int(np.argmin(anchor_values))
    correction_local = int(np.argmin(correction_values))
    pairwise_margin_ratio = float(
        max(
            correction_values[anchor_local] - correction_values[correction_local],
            0.0,
        )
        / scale
    )
    uncertainty_ratio = float(
        (
            correction_uncertainty[rows[anchor_local]]
            + correction_uncertainty[rows[correction_local]]
        )
        / scale
    )
    confidence_passed = bool(
        pairwise_margin_ratio + 1e-12
        >= float(confidence_multiplier) * uncertainty_ratio
    )
    alpha = min(
        1.0,
        float(disagreement_cap) / max(disagreement_ratio, 1e-12),
    ) * float(confidence_passed)
    return float(alpha), {
        "anchor_action_local_index": anchor_local,
        "pairwise_action_local_index": correction_local,
        "actions_agree": anchor_local == correction_local,
        "anchor_scale": scale,
        "correction_disagreement_ratio": disagreement_ratio,
        "pairwise_margin_ratio": pairwise_margin_ratio,
        "pairwise_uncertainty_ratio": uncertainty_ratio,
        "confidence_passed": confidence_passed,
    }


def evaluate_anchored_candidates(
    dataset,
    *,
    models,
    excluded_group_ids: Sequence[str],
) -> dict[str, Any]:
    return _evaluate_anchored_model_ensemble(
        dataset,
        model_ensemble=(models,),
        excluded_group_ids=excluded_group_ids,
    )


def evaluate_anchored_ensemble_candidates(
    dataset,
    *,
    model_ensemble: Sequence[Any],
    excluded_group_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a score-bagged ensemble of identically specified base models."""

    models = tuple(model_ensemble)
    if len(models) < 2:
        raise ValueError("anchored ensemble evaluation requires at least two models")
    return _evaluate_anchored_model_ensemble(
        dataset,
        model_ensemble=models,
        excluded_group_ids=excluded_group_ids,
    )


def _combine_anchored_ensemble_scores(
    member_scores: Sequence[np.ndarray],
    member_uncertainties: Sequence[np.ndarray],
    member_trust: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not member_scores or not (
        len(member_scores) == len(member_uncertainties) == len(member_trust)
    ):
        raise ValueError("anchored ensemble score members are incomplete")
    score_stack = np.vstack(member_scores)
    uncertainty_stack = np.vstack(member_uncertainties)
    trust_stack = np.vstack(member_trust)
    if not (
        score_stack.shape == uncertainty_stack.shape == trust_stack.shape
    ):
        raise ValueError("anchored ensemble score member shapes differ")
    score = np.mean(score_stack, axis=0)
    uncertainty = np.sqrt(
        np.mean(
            np.square(uncertainty_stack)
            + np.square(score_stack - score[None, :]),
            axis=0,
        )
    )
    trust = np.mean(trust_stack, axis=0)
    return score, uncertainty, trust


def _evaluate_anchored_model_ensemble(
    dataset,
    *,
    model_ensemble: Sequence[Any],
    excluded_group_ids: Sequence[str],
) -> dict[str, Any]:
    models_sequence = tuple(model_ensemble)
    if not models_sequence:
        raise ValueError("anchored evaluation requires at least one model")
    for models in models_sequence:
        if set(models.family_models) != set(BASE_FAMILIES):
            raise ValueError(
                "anchored evaluation requires exactly the two frozen base families"
            )
    reference_key = str(getattr(models_sequence[0].prior_spec, "key", ""))
    reference_modes = dict(models_sequence[0].objective_modes)
    for models in models_sequence[1:]:
        if str(getattr(models.prior_spec, "key", "")) != reference_key:
            raise ValueError("anchored ensemble prior policies differ")
        if dict(models.objective_modes) != reference_modes:
            raise ValueError("anchored ensemble objective modes differ")
    groups = np.asarray(dataset.metadata["action_group_ids"], dtype=str)
    excluded = {str(value) for value in excluded_group_ids}
    evaluation_groups = set(groups.tolist()) - excluded
    evaluation = _group_subset_v3(
        dataset,
        selected_groups=evaluation_groups,
        metadata_updates={
            "target_data_role": "anchored_pairwise_development_evaluation",
            "excluded_target_group_ids": sorted(excluded),
        },
    )
    contrast = build_action_contrast_dataset(
        evaluation,
        reference_policy=models_sequence[0].prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    contrast_groups = action_group_ids(contrast)
    group_order = np.argsort(contrast_groups, kind="stable")
    sorted_groups = contrast_groups[group_order]
    boundaries = np.concatenate(
        [
            np.asarray([0], dtype=int),
            np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1,
            np.asarray([contrast.size], dtype=int),
        ]
    )
    grouped_rows = [
        group_order[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    family_scores: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family in BASE_FAMILIES:
        member_scores = []
        member_uncertainties = []
        member_trust = []
        for models in models_sequence:
            score, uncertainty, trust, _ = _group_adjusted_scores(
                contrast,
                models.family_models[family].predict(contrast),
                objective_mode=models.objective_modes[family],
            )
            member_scores.append(score)
            member_uncertainties.append(uncertainty)
            member_trust.append(trust)
        score, uncertainty, trust = _combine_anchored_ensemble_scores(
            member_scores,
            member_uncertainties,
            member_trust,
        )
        family_scores[family] = (score, uncertainty, trust)
    anchor_score, anchor_uncertainty, anchor_trust = family_scores[ANCHOR_FAMILY]
    correction_score, correction_uncertainty, correction_trust = family_scores[
        CORRECTION_FAMILY
    ]
    score_delta = correction_score - anchor_score

    def summarize(score: np.ndarray, uncertainty: np.ndarray, alphas: Sequence[float]):
        regrets = []
        optimal = 0
        selected_matches_anchor = 0
        selected_matches_pairwise = 0
        for rows in grouped_rows:
            selected = int(rows[int(np.argmin(score[rows]))])
            anchor_selected = int(rows[int(np.argmin(anchor_score[rows]))])
            pairwise_selected = int(rows[int(np.argmin(correction_score[rows]))])
            values = actual[rows]
            best = float(np.min(values))
            scale = action_group_range(values)
            regrets.append((float(actual[selected]) - best) / scale)
            optimal += int(
                np.isclose(float(actual[selected]), best, rtol=0.0, atol=1e-12)
            )
            selected_matches_anchor += int(selected == anchor_selected)
            selected_matches_pairwise += int(selected == pairwise_selected)
        alpha_values = np.asarray(alphas, dtype=float)
        return {
            "group_count": len(grouped_rows),
            "mean_normalized_action_regret": float(np.mean(regrets)),
            "p90_normalized_action_regret": float(np.quantile(regrets, 0.90)),
            "worst_normalized_action_regret": float(np.max(regrets)),
            "optimal_action_rate": float(optimal / max(len(regrets), 1)),
            "mean_uncertainty": float(np.mean(uncertainty)),
            "mean_alpha": float(np.mean(alpha_values)),
            "p10_alpha": float(np.quantile(alpha_values, 0.10)),
            "p90_alpha": float(np.quantile(alpha_values, 0.90)),
            "full_pairwise_weight_fraction": float(
                np.mean(alpha_values >= 1.0 - 1e-12)
            ),
            "selected_action_matches_anchor_fraction": float(
                selected_matches_anchor / max(len(grouped_rows), 1)
            ),
            "selected_action_matches_pairwise_fraction": float(
                selected_matches_pairwise / max(len(grouped_rows), 1)
            ),
        }

    candidates: dict[str, dict[str, Any]] = {}
    for alpha in CONSTANT_ALPHAS:
        score = anchor_score + float(alpha) * score_delta
        uncertainty = (
            (1.0 - float(alpha)) * anchor_uncertainty
            + float(alpha) * correction_uncertainty
        )
        candidates[constant_candidate_key(alpha)] = summarize(
            score,
            uncertainty,
            [float(alpha)] * len(grouped_rows),
        )

    for cap in DISAGREEMENT_CAPS:
        for confidence in CONFIDENCE_MULTIPLIERS:
            row_alpha = np.zeros(contrast.size, dtype=float)
            group_alphas = []
            group_diagnostics = []
            for rows in grouped_rows:
                alpha, diagnostics = _local_group_alpha(
                    anchor_score=anchor_score,
                    correction_score=correction_score,
                    anchor_uncertainty=anchor_uncertainty,
                    correction_uncertainty=correction_uncertainty,
                    group_rows=rows,
                    disagreement_cap=cap,
                    confidence_multiplier=confidence,
                )
                row_alpha[rows] = alpha
                group_alphas.append(alpha)
                group_diagnostics.append(diagnostics)
            score = anchor_score + row_alpha * score_delta
            uncertainty = (
                (1.0 - row_alpha) * anchor_uncertainty
                + row_alpha * correction_uncertainty
            )
            row = summarize(score, uncertainty, group_alphas)
            row.update(
                {
                    "disagreement_cap": float(cap),
                    "confidence_multiplier": float(confidence),
                    "confidence_pass_fraction": float(
                        np.mean(
                            [
                                bool(item["confidence_passed"])
                                for item in group_diagnostics
                            ]
                        )
                    ),
                    "mean_correction_disagreement_ratio": float(
                        np.mean(
                            [
                                float(item["correction_disagreement_ratio"])
                                for item in group_diagnostics
                            ]
                        )
                    ),
                }
            )
            candidates[local_candidate_key(cap, confidence)] = row
    if set(candidates) != set(CANDIDATE_KEYS):
        raise AssertionError("anchored candidate grid construction changed")
    ensemble_size = len(models_sequence)
    return {
        "protocol": (
            "anchored-pairwise-shrinkage-offline-regret-v1"
            if ensemble_size == 1
            else "anchored-pairwise-cross-fitted-score-ensemble-offline-regret-v1"
        ),
        "model_ensemble_size": ensemble_size,
        "evaluation_group_count": len(grouped_rows),
        "excluded_group_count": len(excluded),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "base_families": {
            ANCHOR_FAMILY: candidates[constant_candidate_key(0.0)],
            CORRECTION_FAMILY: candidates[constant_candidate_key(1.0)],
        },
        "mean_anchor_context_trust": float(np.mean(anchor_trust)),
        "mean_pairwise_context_trust": float(np.mean(correction_trust)),
    }


def run_target_development(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    protocol_spec_path: Path,
    target: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol_spec = _load_protocol(protocol_spec_path)
    baseline_sha256 = _sha256(baseline_result)
    if baseline_sha256 != str(expected_baseline_sha256):
        raise ValueError("expanded source-rule baseline hash mismatch")
    cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    if cache_sha256 != str(expected_cache_sha256):
        raise ValueError("combined counterfactual cache hash mismatch")
    if cache_file_count != EXPECTED_CACHE_FILE_COUNT:
        raise ValueError("combined counterfactual cache file count changed")
    manifest = load_traffic_signal_manifest(manifest_path)
    if target not in manifest.sumocfgs:
        raise ValueError(f"unknown development target: {target}")
    development = protocol_spec.get("development", {})
    evaluation_targets = [
        str(value) for value in development.get("evaluation_targets", ())
    ]
    source_only_exclusions = {
        str(name): int(count)
        for name, count in development.get(
            "source_only_capacity_exclusions", {}
        ).items()
    }
    if evaluation_targets:
        if set(evaluation_targets) | set(source_only_exclusions) != set(
            manifest.sumocfgs
        ):
            raise ValueError("v39b capacity partition does not cover the manifest")
        if target not in evaluation_targets:
            raise ValueError(f"v39b target is source-only after capacity admission: {target}")
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=workers,
    )
    target_groups = np.asarray(
        bank[target].metadata.get("action_group_ids", ()), dtype=str
    )
    if target_groups.shape != (bank[target].size,):
        raise ValueError("v39 target action-group metadata is not row aligned")
    target_safe_complete_group_count = len(set(target_groups.tolist()))
    if target_safe_complete_group_count < TARGET_GROUP_BUDGET:
        raise ValueError("v39 target has fewer than 60 safe complete groups")
    split = _target_adaptation_split_v3(
        bank[target],
        group_budget=TARGET_GROUP_BUDGET,
        selection_seed=20260803,
        calibration_seeds=(4047,),
        calibration_fraction=0.4,
    )
    selected_group_ids = tuple(
        sorted(
            str(value)
            for value in split["selected"].metadata[
                "target_adaptation_selected_group_ids"
            ]
        )
    )
    if len(selected_group_ids) != TARGET_GROUP_BUDGET:
        raise ValueError("v39 target selection did not produce 60 groups")
    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    models = fit_target_screening_models(
        bank,
        target,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=BASE_FAMILIES,
        target_group_budget=TARGET_GROUP_BUDGET,
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(4047,),
        target_calibration_fraction=0.4,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
        target_adaptation_group_ids=selected_group_ids,
    )
    target_city = manifest.city_groups[target]
    expected_heldout = sorted(
        name for name, city in manifest.city_groups.items() if city == target_city
    )
    expected_sources = sorted(set(manifest.sumocfgs) - set(expected_heldout))
    diagnostics = models.diagnostics
    if diagnostics.get("heldout_city_scenarios") != expected_heldout:
        raise ValueError("v39 same-city holdout changed")
    if diagnostics.get("source_scenarios") != expected_sources:
        raise ValueError("v39 source scenario set changed")
    if not set(expected_sources) <= set(source_rule_costs):
        raise ValueError("expanded baseline lacks a v39 source scenario")
    if (
        diagnostics.get("target_adaptation_protocol")
        != "explicit-complete-action-group-list-v1"
        or int(diagnostics.get("target_adaptation_groups", -1))
        != TARGET_GROUP_BUDGET
        or diagnostics.get("target_adaptation_group_ids")
        != list(selected_group_ids)
    ):
        raise ValueError("v39 explicit 60-group fit contract failed")
    evaluation = evaluate_anchored_candidates(
        bank[target],
        models=models,
        excluded_group_ids=selected_group_ids,
    )
    if int(evaluation["excluded_group_count"]) != TARGET_GROUP_BUDGET:
        raise ValueError("v39 evaluation did not exclude all adaptation groups")
    return {
        "protocol": "tsc-v39br35-capacity-admitted-anchored-pairwise-target-development-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "protocol_spec_path": str(protocol_spec_path.resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "protocol_spec": protocol_spec,
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest.to_dict(),
        "cache_root": str(Path(cache_root).resolve()),
        "cache_aggregate_sha256": cache_sha256,
        "cache_file_count": cache_file_count,
        "cache_audit": cache_audit,
        "baseline_result": str(baseline_result.resolve()),
        "baseline_sha256": baseline_sha256,
        "target": target,
        "target_city_group": target_city,
        "heldout_city_scenarios": expected_heldout,
        "source_scenarios": expected_sources,
        "source_city_groups": sorted(
            {manifest.city_groups[name] for name in expected_sources}
        ),
        "target_group_budget": TARGET_GROUP_BUDGET,
        "target_safe_complete_group_count": target_safe_complete_group_count,
        "development_evaluation_targets": evaluation_targets,
        "source_only_capacity_exclusions": source_only_exclusions,
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(BASE_FAMILIES),
        "model_diagnostics": diagnostics,
        "offline_evaluation": evaluation,
        "elapsed_sec": float(time.monotonic() - started),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_target_development(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        target=args.target,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "target": payload["target"],
                "candidate_count": payload["offline_evaluation"]["candidate_count"],
                "evaluation_group_count": payload["offline_evaluation"][
                    "evaluation_group_count"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
