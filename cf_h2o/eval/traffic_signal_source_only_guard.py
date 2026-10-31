"""Freeze a deployment guard using source-city counterfactuals only."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    COLLECTION_SHARDS,
    CORRECTION_FAMILY,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    constant_candidate_key,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    ContrastGuardConfig,
    _calibrate_source_loo_uncertainty_v3,
    _fit_family_with_source_loo_v3,
    _group_adjusted_scores,
    _source_loo_action_records_v3,
    calibrate_contrast_guard_v3,
    merge_counterfactual_datasets_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _relabel_dataset_domain,
)
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    validate_repaired_cache_contract,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.generalized_pressure import (
    PressurePolicySpec,
    generalized_pressure_grid,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


PROTOCOL = "tsc-source-only-anchored-familywise-guard-freeze-v2"
ANCHORED_CANDIDATE = constant_candidate_key(0.9)
ANCHORED_CORRECTION_WEIGHT = 0.9
DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_MAX_RELATIVE_RULE_GAPS = (0.0, 0.10, 0.25, 0.50, 1.0)


def resolve_frozen_prior_policy(key: str) -> PressurePolicySpec:
    specs = {spec.key: spec for spec in generalized_pressure_grid()}
    try:
        return specs[str(key)]
    except KeyError as exc:
        raise ValueError(f"unknown frozen pressure prior: {key!r}") from exc


def anchored_prediction_bundle(
    anchor_bundle: Mapping[str, tuple[Any, Mapping[str, Any]]],
    correction_bundle: Mapping[str, tuple[Any, Mapping[str, Any]]],
    *,
    correction_weight: float = ANCHORED_CORRECTION_WEIGHT,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Blend source-LOO predictions exactly as the frozen online model does."""

    alpha = float(correction_weight)
    if not 0.0 <= alpha <= 1.0 or set(anchor_bundle) != set(correction_bundle):
        raise ValueError("anchored source-LOO prediction bundles are incompatible")
    blended = {}
    for domain in sorted(anchor_bundle):
        anchor_valid, anchor_prediction = anchor_bundle[domain]
        correction_valid, correction_prediction = correction_bundle[domain]
        if (
            anchor_valid is not correction_valid
            and (
                anchor_valid.size != correction_valid.size
                or anchor_valid.feature_names != correction_valid.feature_names
                or not np.array_equal(anchor_valid.features, correction_valid.features)
                or not np.array_equal(anchor_valid.domains, correction_valid.domains)
            )
        ):
            raise ValueError(f"anchored held-out dataset changed for {domain}")
        anchor_score, anchor_uncertainty, anchor_trust, anchor_reference = (
            _group_adjusted_scores(
                anchor_valid,
                anchor_prediction,
                objective_mode="control_only",
            )
        )
        (
            correction_score,
            correction_uncertainty,
            correction_trust,
            correction_reference,
        ) = _group_adjusted_scores(
            correction_valid,
            correction_prediction,
            objective_mode="control_only",
        )
        if not np.array_equal(anchor_reference, correction_reference):
            raise ValueError(f"anchored reference rows changed for {domain}")
        blended[domain] = (
            anchor_valid,
            {
                "control_cost": {
                    "mean": anchor_score + alpha * (
                        correction_score - anchor_score
                    ),
                    "uncertainty": (
                        (1.0 - alpha) * anchor_uncertainty
                        + alpha * correction_uncertainty
                    ),
                    "context_trust": np.minimum(
                        anchor_trust, correction_trust
                    ),
                }
            },
        )
    return blended


def familywise_guard_options(
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    max_relative_rule_gaps: Sequence[float] = DEFAULT_MAX_RELATIVE_RULE_GAPS,
    familywise_alpha: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a source-LOO guard grid with one-sided Bonferroni control."""

    threshold_values = tuple(float(value) for value in thresholds)
    gap_values = tuple(float(value) for value in max_relative_rule_gaps)
    alpha = float(familywise_alpha)
    if (
        not threshold_values
        or not gap_values
        or len(threshold_values) != len(set(threshold_values))
        or len(gap_values) != len(set(gap_values))
        or any(value < 0.0 for value in threshold_values)
        or any(value < 0.0 for value in gap_values)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid family-wise guard grid")
    candidate_count = len(threshold_values) * len(gap_values)
    critical = float(
        NormalDist().inv_cdf(1.0 - alpha / float(candidate_count))
    )
    options = {
        "thresholds": threshold_values,
        "max_relative_rule_gaps": gap_values,
        "selection_mode": "mean_ucb",
        "confidence_z": critical,
    }
    audit = {
        "protocol": "bonferroni-one-sided-source-loo-guard-grid-v1",
        "familywise_alpha": alpha,
        "candidate_count": candidate_count,
        "simultaneous_confidence_multiplier": critical,
        "thresholds": list(threshold_values),
        "max_relative_rule_gaps": list(gap_values),
    }
    return options, audit


def validate_source_only_guard_payload(
    payload: Mapping[str, Any],
    *,
    expected_prior_policy: str,
    family: str = ANCHOR_FAMILY,
) -> tuple[ContrastGuardConfig, float]:
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("scientific_status")
        != "source-only-frozen-before-unseen-target-closed-loop"
        or payload.get("family") != str(family)
        or payload.get("prior_policy") != str(expected_prior_policy)
        or payload.get("target_data_consumed") is not False
        or payload.get("target_closed_loop_consumed") is not False
    ):
        raise ValueError("source-only guard contract changed")
    guard_values = payload.get("guard_config")
    if not isinstance(guard_values, Mapping):
        raise ValueError("source-only guard configuration is absent")
    guard = ContrastGuardConfig(**dict(guard_values))
    scale = float(payload.get("uncertainty_scale", float("nan")))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("source-only guard uncertainty scale is invalid")
    return guard, scale


def load_source_only_guard(
    path: Path,
    *,
    expected_sha256: str,
    expected_prior_policy: str,
    family: str = ANCHOR_FAMILY,
) -> tuple[ContrastGuardConfig, float, dict[str, Any]]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("source-only guard identity changed")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    guard, scale = validate_source_only_guard_payload(
        payload,
        expected_prior_policy=expected_prior_policy,
        family=family,
    )
    return guard, scale, payload


def run_source_only_guard_freeze(
    *,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_repair_audit_path: Path,
    prior_policy: str,
    workers: int,
    fit_workers: int,
    familywise_alpha: float,
    out: Path,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite source-only guard: {out}")
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    source_repair = json.loads(
        Path(source_repair_audit_path).read_text(encoding="utf-8")
    )
    validate_repaired_cache_contract(
        label="source",
        repair_audit=source_repair,
        manifest_scenarios=tuple(source_manifest.sumocfgs),
        seed_count=len(SOURCE_SEEDS),
        collection_shards=COLLECTION_SHARDS,
    )
    source_bank, cache_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    validate_repaired_cache_contract(
        label="source",
        repair_audit=source_repair,
        manifest_scenarios=tuple(source_manifest.sumocfgs),
        seed_count=len(SOURCE_SEEDS),
        collection_shards=COLLECTION_SHARDS,
        cache_audit=cache_audit,
    )
    datasets = [
        _relabel_dataset_domain(
            source_bank[name], str(source_manifest.city_groups[name])
        )
        for name in source_manifest.sumocfgs
    ]
    absolute = merge_counterfactual_datasets_v3(datasets)
    source_domains = tuple(sorted(str(value) for value in np.unique(absolute.domains)))
    if len(source_domains) < 2:
        raise ValueError("source-only guard requires multiple source cities")
    guard_options, familywise_audit = familywise_guard_options(
        familywise_alpha=familywise_alpha
    )
    prior_spec = resolve_frozen_prior_policy(str(prior_policy))
    contrast = build_action_contrast_dataset(
        absolute,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    config = MechanismFitConfig()
    _, anchor_diagnostics, anchor_bundle, anchor_parallel = (
        _fit_family_with_source_loo_v3(
            contrast,
            family=ANCHOR_FAMILY,
            cfcmt_config=config,
            target_domain=None,
            parallel_workers=max(int(fit_workers), 1),
        )
    )
    _, correction_diagnostics, correction_bundle, correction_parallel = (
        _fit_family_with_source_loo_v3(
            contrast,
            family=CORRECTION_FAMILY,
            cfcmt_config=config,
            target_domain=None,
            parallel_workers=max(int(fit_workers), 1),
        )
    )
    blended_bundle = anchored_prediction_bundle(
        anchor_bundle,
        correction_bundle,
    )
    records = _source_loo_action_records_v3(
        contrast,
        family=ANCHOR_FAMILY,
        cfcmt_config=config,
        prediction_bundle=blended_bundle,
    )
    scaled_records, uncertainty = _calibrate_source_loo_uncertainty_v3(records)
    guard, guard_diagnostics = calibrate_contrast_guard_v3(
        contrast,
        family=ANCHOR_FAMILY,
        cfcmt_config=config,
        records_bundle=scaled_records,
        **guard_options,
    )
    scale = float(uncertainty["scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("source-only uncertainty calibration failed")
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "source-only-frozen-before-unseen-target-closed-loop",
        "family": ANCHOR_FAMILY,
        "anchor_family": ANCHOR_FAMILY,
        "correction_family": CORRECTION_FAMILY,
        "anchored_candidate": ANCHORED_CANDIDATE,
        "anchored_correction_weight": ANCHORED_CORRECTION_WEIGHT,
        "prior_policy": str(prior_policy),
        "source_domains": list(source_domains),
        "source_scenario_count": len(source_manifest.sumocfgs),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_repair_audit_sha256": _sha256(source_repair_audit_path),
        "source_cache_audit": cache_audit,
        "target_data_consumed": False,
        "target_closed_loop_consumed": False,
        "guard_config": asdict(guard),
        "uncertainty_scale": scale,
        "uncertainty_calibration": uncertainty,
        "guard_calibration": guard_diagnostics,
        "source_loo_fit": {
            "anchor": {
                "fit": anchor_diagnostics,
                "parallelism": anchor_parallel,
            },
            "correction": {
                "fit": correction_diagnostics,
                "parallelism": correction_parallel,
            },
        },
        "multiple_comparison_control": familywise_audit,
        "runtime_contract": (
            "multiply mixture uncertainty by uncertainty_scale before applying "
            "guard_config; a disabled guard is an exact pressure-prior fallback"
        ),
    }
    validate_source_only_guard_payload(
        payload,
        expected_prior_policy=str(prior_policy),
    )
    _atomic_json(out, payload)
    return payload
