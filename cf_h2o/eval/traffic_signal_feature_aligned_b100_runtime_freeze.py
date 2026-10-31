"""Refit, revalidate, and freeze the V157B B100 runtime arms.

One target model reproduces the original V157A path for audit only.  The
runtime target model instead relabels every selected Jinan row to the common
``jinan`` domain used by the source-component path.  Seven exact V157A source
components and seven capacity-matched label-placebo components share the same
loaded banks and process pool.  A corrected source-minus-target gate must pass
before any runtime bundle is written.  Jinan B100 labels are never permuted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_bytes,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _relabel_dataset_domain,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    ARTIFACT_PROTOCOL as V123_ARTIFACT_PROTOCOL,
    _paired_bootstrap,
    _seed_policy_metrics,
    _target_model,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    _adaptation_inputs,
    _source_inputs,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
    _selector_dataset,
    _target_cache_audit_contract,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    _fit_component_budget,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
    rule_reference_indices,
)
from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
)
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


CONFIG_PROTOCOL = "tsc-v157b-feature-aligned-b100-runtime-refit-freeze-v2"
RESULT_PROTOCOL = "tsc-v157b-feature-aligned-b100-runtime-refit-freeze-result-v2"
RUNTIME_BUNDLE_PROTOCOL = "cfcmt-v157b-feature-aligned-b100-runtime-arms-v2"
ARM_MANIFEST_PROTOCOL = "tsc-v157b-feature-aligned-b100-arm-manifest-v2"
PLACEBO_MAPPING_PROTOCOL = "tsc-v157b-source-label-block-permutation-v1"
V157A_AGGREGATE_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-aggregate-v1"
)
ARM_ORDER = (
    "target_only",
    "uniform_source",
    "source_label_placebo",
    "phase_pressure",
)
LEARNED_ARMS = ARM_ORDER[:-1]
_FREEZE_FIT_STATE: Mapping[str, Any] | None = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class UniformAnchoredRuntimeModel:
    """Return the V157A mean of fixed, group-adjusted component scores."""

    def __init__(self, component_models: Mapping[str, Any]) -> None:
        if not component_models:
            raise ValueError("uniform runtime model requires source components")
        self.component_names = tuple(sorted(str(name) for name in component_models))
        self.component_models = tuple(component_models[name] for name in self.component_names)

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        # V157A adjusted each source component against its PhasePressure row and
        # then averaged the resulting score arrays.  Preserve that order here:
        # averaging raw model outputs first is algebraically equivalent but can
        # differ by one floating-point bit and therefore fails the freeze audit.
        rows = [
            _prediction_arrays(model, dataset) for model in self.component_models
        ]
        names = ("mean", "uncertainty", "context_trust")
        return {
            "control_cost": {
                name: np.mean(
                    np.vstack([row[index] for row in rows]),
                    axis=0,
                )
                for index, name in enumerate(names)
            }
        }


def _anchored_source_components(
    source_payload: Mapping[str, Any],
    *,
    candidate: str,
) -> dict[str, FrozenAnchoredBlendModel]:
    components = dict(source_payload.get("component_models", {}))
    result = {}
    for source_group in sorted(components):
        fitted = components[source_group]
        if (
            set(fitted.family_models) != set(BASE_FAMILIES)
            or set(fitted.objective_modes) != set(BASE_FAMILIES)
        ):
            raise ValueError(f"{source_group}: B100 source component families changed")
        result[source_group] = FrozenAnchoredBlendModel(
            anchor_model=fitted.family_models[ANCHOR_FAMILY],
            correction_model=fitted.family_models[CORRECTION_FAMILY],
            candidate=str(candidate),
            anchor_objective_mode=fitted.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=fitted.objective_modes[CORRECTION_FAMILY],
        )
    return result


def _prediction_arrays(
    model: Any,
    dataset: MechanismDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score, uncertainty, trust, _ = _group_adjusted_scores(
        dataset,
        model.predict(dataset),
        objective_mode="control_only",
    )
    return (
        np.asarray(score, dtype=float),
        np.asarray(uncertainty, dtype=float),
        np.asarray(trust, dtype=float),
    )


def _equal_arrays(
    observed: Sequence[np.ndarray], expected: Sequence[np.ndarray]
) -> bool:
    return len(observed) == len(expected) and all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(observed, expected, strict=True)
    )


def _prediction_identity_diagnostics(
    observed: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Summarize exact prediction-array identity without serializing the arrays."""

    names = ("score", "uncertainty", "trust")
    arrays = {}
    for name, left, right in zip(names, observed, expected, strict=False):
        observed_array = np.asarray(left)
        expected_array = np.asarray(right)
        same_shape = observed_array.shape == expected_array.shape
        array_equal = bool(np.array_equal(observed_array, expected_array))
        if same_shape:
            mismatch_count = int(
                np.count_nonzero(np.not_equal(observed_array, expected_array))
            )
            finite = bool(
                np.all(np.isfinite(observed_array))
                and np.all(np.isfinite(expected_array))
            )
            max_abs_difference = (
                float(np.max(np.abs(observed_array - expected_array)))
                if finite and observed_array.size
                else (0.0 if finite else None)
            )
        else:
            mismatch_count = None
            max_abs_difference = None
        arrays[name] = {
            "array_equal": array_equal,
            "observed_shape": list(observed_array.shape),
            "expected_shape": list(expected_array.shape),
            "mismatch_count": mismatch_count,
            "max_abs_difference": max_abs_difference,
        }
    return {
        "all_array_equal": bool(
            len(observed) == len(expected)
            and len(arrays) == len(names)
            and all(row["array_equal"] for row in arrays.values())
        ),
        "observed_array_count": len(observed),
        "expected_array_count": len(expected),
        "arrays": arrays,
    }


def _one_group(dataset: MechanismDataset) -> MechanismDataset:
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    selected = str(groups[0])
    return _group_subset_v3(
        dataset,
        selected_groups={selected},
        metadata_updates={"identity_audit_role": "single_group_feature_order_check"},
    )


def _reverse_feature_columns(dataset: MechanismDataset) -> MechanismDataset:
    order = np.arange(len(dataset.feature_names) - 1, -1, -1, dtype=int)
    return MechanismDataset(
        feature_names=tuple(dataset.feature_names[index] for index in order),
        features=np.asarray(dataset.features[:, order], dtype=float),
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata=dict(dataset.metadata),
    )


@dataclass(frozen=True)
class SelectorPredictionCache:
    """Full-selector arrays and small feature-binding checks shared by all gates."""

    v157a_audit_target: tuple[np.ndarray, np.ndarray, np.ndarray]
    domain_aligned_target: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_components: Mapping[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    uniform_runtime: tuple[np.ndarray, np.ndarray, np.ndarray]
    v157a_audit_feature_order: Mapping[str, bool]
    domain_aligned_feature_order: Mapping[str, bool]


def _build_selector_prediction_cache(
    *,
    v157a_audit_target_model: Any,
    domain_aligned_target_model: Any,
    source_components: Mapping[str, Any],
    uniform_runtime_model: Any,
    selector_contrast: MechanismDataset,
) -> SelectorPredictionCache:
    """Evaluate each expensive selector prediction once before the freeze gates."""

    source_names = tuple(sorted(str(name) for name in source_components))
    v157a_audit_target = _prediction_arrays(
        v157a_audit_target_model, selector_contrast
    )
    domain_aligned_target = _prediction_arrays(
        domain_aligned_target_model, selector_contrast
    )
    source_arrays = {
        name: _prediction_arrays(source_components[name], selector_contrast)
        for name in source_names
    }
    # Exercise the serialized runtime wrapper once on the full selector.  This
    # deliberately remains distinct from the cached component mean so the exact
    # wrapper-versus-component check still covers the deployed API.
    uniform_runtime = _prediction_arrays(uniform_runtime_model, selector_contrast)

    sample = _one_group(selector_contrast)
    reversed_sample = _reverse_feature_columns(sample)
    source_feature_order = {
        name: _equal_arrays(
            _prediction_arrays(source_components[name], sample),
            _prediction_arrays(source_components[name], reversed_sample),
        )
        for name in source_names
    }

    def target_feature_order(model: Any) -> dict[str, bool]:
        return {
            "target_only": _equal_arrays(
                _prediction_arrays(model, sample),
                _prediction_arrays(model, reversed_sample),
            ),
            **source_feature_order,
        }

    return SelectorPredictionCache(
        v157a_audit_target=v157a_audit_target,
        domain_aligned_target=domain_aligned_target,
        source_components=source_arrays,
        uniform_runtime=uniform_runtime,
        v157a_audit_feature_order=target_feature_order(
            v157a_audit_target_model
        ),
        domain_aligned_feature_order=target_feature_order(
            domain_aligned_target_model
        ),
    )


def validate_reused_prediction_identity(
    *,
    target_model: Any,
    source_components: Mapping[str, Any],
    selector_contrast: MechanismDataset,
    v157a_artifact: Mapping[str, Any],
    comparison: str = "legacy_v115_pickles_vs_v157a_b100_predictions",
    require_exact: bool = True,
    authorize_runtime_reuse: bool = False,
    target_prediction_arrays: Sequence[np.ndarray] | None = None,
    source_prediction_arrays: Mapping[str, Sequence[np.ndarray]] | None = None,
    uniform_runtime_prediction_arrays: Sequence[np.ndarray] | None = None,
    feature_order_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Compare runtime models with V157A arrays, optionally as diagnosis only."""

    expected_sources = tuple(str(value) for value in v157a_artifact["source_group_order"])
    if (
        v157a_artifact.get("protocol") != V123_ARTIFACT_PROTOCOL
        or v157a_artifact.get("city") != "jinan"
        or v157a_artifact.get("estimand") != WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
        or v157a_artifact.get("target_name") != "prefix_mean_cost_450s"
        or v157a_artifact.get("prior_policy") != "phase_pressure"
        or tuple(sorted(source_components)) != tuple(sorted(expected_sources))
    ):
        raise ValueError("V157A prediction artifact contract changed")

    expected_target = tuple(v157a_artifact["target_predictions"]["100"])
    observed_target = (
        tuple(target_prediction_arrays)
        if target_prediction_arrays is not None
        else _prediction_arrays(target_model, selector_contrast)
    )
    target_diagnostics = _prediction_identity_diagnostics(
        observed_target, expected_target
    )
    target_equal = bool(target_diagnostics["all_array_equal"])
    source_diagnostics = {}
    source_equal = {}
    observed_source = {}
    expected_source = {}
    for source_group in expected_sources:
        observed_source[source_group] = (
            tuple(source_prediction_arrays[source_group])
            if source_prediction_arrays is not None
            else _prediction_arrays(source_components[source_group], selector_contrast)
        )
        expected_source[source_group] = tuple(
            v157a_artifact["source_predictions"]["100"][source_group]
        )
        source_diagnostics[source_group] = _prediction_identity_diagnostics(
            observed_source[source_group], expected_source[source_group]
        )
        source_equal[source_group] = bool(
            source_diagnostics[source_group]["all_array_equal"]
        )
    observed_uniform = tuple(
        np.mean(
            np.vstack([observed_source[group][index] for group in expected_sources]),
            axis=0,
        )
        for index in range(3)
    )
    expected_uniform = tuple(
        np.mean(
            np.vstack([expected_source[group][index] for group in expected_sources]),
            axis=0,
        )
        for index in range(3)
    )
    component_mean_diagnostics = _prediction_identity_diagnostics(
        observed_uniform, expected_uniform
    )
    component_mean_equal = bool(component_mean_diagnostics["all_array_equal"])
    runtime_uniform = (
        tuple(uniform_runtime_prediction_arrays)
        if uniform_runtime_prediction_arrays is not None
        else _prediction_arrays(
            UniformAnchoredRuntimeModel(source_components), selector_contrast
        )
    )
    runtime_uniform_diagnostics = _prediction_identity_diagnostics(
        runtime_uniform, expected_uniform
    )
    runtime_uniform_equal = bool(runtime_uniform_diagnostics["all_array_equal"])
    runtime_vs_component_diagnostics = _prediction_identity_diagnostics(
        runtime_uniform, observed_uniform
    )
    if source_prediction_arrays is not None and set(source_prediction_arrays) != set(
        expected_sources
    ):
        raise ValueError("cached V157B source prediction roster changed")
    if feature_order_checks is None:
        sample = _one_group(selector_contrast)
        reversed_sample = _reverse_feature_columns(sample)
        observed_feature_order_checks = {
            "target_only": _equal_arrays(
                _prediction_arrays(target_model, sample),
                _prediction_arrays(target_model, reversed_sample),
            ),
            **{
                source_group: _equal_arrays(
                    _prediction_arrays(source_components[source_group], sample),
                    _prediction_arrays(
                        source_components[source_group], reversed_sample
                    ),
                )
                for source_group in expected_sources
            },
        }
    else:
        observed_feature_order_checks = {
            str(name): bool(passed)
            for name, passed in feature_order_checks.items()
        }
        if set(observed_feature_order_checks) != {"target_only", *expected_sources}:
            raise ValueError("cached V157B feature-order roster changed")
    failed_sections = []
    if not target_equal:
        failed_sections.append("target_only")
    failed_sections.extend(
        f"source:{source_group}"
        for source_group in expected_sources
        if not source_equal[source_group]
    )
    if not component_mean_equal:
        failed_sections.append("uniform_component_mean")
    if not runtime_uniform_equal:
        failed_sections.append("uniform_runtime")
    failed_sections.extend(
        f"feature_order:{name}"
        for name, passed in observed_feature_order_checks.items()
        if not passed
    )
    passed = not failed_sections
    reuse_authorized = bool(passed and authorize_runtime_reuse)
    audit = {
        "protocol": "v157b-v157a-b100-prediction-identity-audit-v2",
        "array_equality": "numpy_array_equal",
        "comparison": str(comparison),
        "target_score_uncertainty_trust_equal": target_equal,
        "per_source_score_uncertainty_trust_equal": source_equal,
        "uniform_component_mean_score_uncertainty_trust_equal": component_mean_equal,
        "uniform_runtime_score_uncertainty_trust_equal": runtime_uniform_equal,
        "selector_row_count": int(selector_contrast.size),
        "feature_order_invariance_on_one_group": observed_feature_order_checks,
        "failed_sections": failed_sections,
        "target_only": target_diagnostics,
        "per_source": source_diagnostics,
        "uniform_component_mean": component_mean_diagnostics,
        "uniform_runtime": runtime_uniform_diagnostics,
        "uniform_runtime_vs_component_mean": runtime_vs_component_diagnostics,
        "passed": passed,
        "runtime_reuse_authorized": reuse_authorized,
        "decision": (
            "exact_refit_identity_validated_for_audit_only"
            if require_exact and passed and not reuse_authorized
            else "exact_refit_identity_validated"
            if require_exact and passed and reuse_authorized
            else "diagnostic_only_do_not_reuse"
        ),
    }
    if not passed:
        print(
            "V157B_PREDICTION_IDENTITY_DIAGNOSTIC="
            + json.dumps(
                audit,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        if require_exact:
            if comparison == "legacy_v115_pickles_vs_v157a_b100_predictions":
                raise ValueError(
                    "V115 runtime pickles do not reproduce corrected V157A B100 arrays"
                )
            raise ValueError(
                "V157B refits do not reproduce corrected V157A B100 arrays"
            )
    return audit


def domain_aligned_source_gate(
    *,
    selector: MechanismDataset,
    selector_contrast: MechanismDataset,
    target_model: Any,
    source_model: Any,
    selector_seeds: Sequence[int],
    mean_maximum: float,
    upper_95_maximum: float,
    target_prediction_score: np.ndarray | None = None,
    source_prediction_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Re-evaluate fixed uniform source against the corrected target comparator."""

    actual, _, group_rows, references, group_seeds = _normalized_pressure_targets(
        selector, selector_contrast
    )
    target_score = (
        np.asarray(target_prediction_score, dtype=float)
        if target_prediction_score is not None
        else _prediction_arrays(target_model, selector_contrast)[0]
    )
    source_score = (
        np.asarray(source_prediction_score, dtype=float)
        if source_prediction_score is not None
        else _prediction_arrays(source_model, selector_contrast)[0]
    )
    target_by_seed = _seed_policy_metrics(
        actual, target_score, group_rows, references, group_seeds
    )
    source_by_seed = _seed_policy_metrics(
        actual, source_score, group_rows, references, group_seeds
    )
    expected_seeds = tuple(sorted(int(value) for value in selector_seeds))
    observed_seeds = tuple(sorted((int(value) for value in target_by_seed), key=int))
    if (
        observed_seeds != tuple(sorted(expected_seeds))
        or set(source_by_seed) != set(target_by_seed)
    ):
        raise ValueError("V157B domain-aligned gate seed roster changed")
    differences = [
        float(
            source_by_seed[str(seed)]["mean_normalized_delta_vs_phase_pressure"]
            - target_by_seed[str(seed)]["mean_normalized_delta_vs_phase_pressure"]
        )
        for seed in expected_seeds
    ]
    interval = _paired_bootstrap(differences)
    passed = bool(
        interval["mean"] <= float(mean_maximum)
        and interval["upper_95"] < float(upper_95_maximum)
    )
    return {
        "protocol": "v157b-domain-aligned-source-minus-target-gate-v2",
        "comparison": "uniform_source_minus_domain_aligned_target_only",
        "candidate": "uniform_all_sources__source_weight_1",
        "source_weight": 1.0,
        "unit": "selector_seed_mean_normalized_cost",
        "selector_seed_order": list(expected_seeds),
        "source_minus_target_by_seed": differences,
        "paired_bootstrap": interval,
        "thresholds": {
            "mean_maximum": float(mean_maximum),
            "upper_95_strict_maximum": float(upper_95_maximum),
        },
        "target_only_seed_metrics": target_by_seed,
        "uniform_source_seed_metrics": source_by_seed,
        "uniform_source_mean_normalized_delta_vs_phase_pressure": float(
            np.mean(
                [
                    source_by_seed[str(seed)][
                        "mean_normalized_delta_vs_phase_pressure"
                    ]
                    for seed in expected_seeds
                ]
            )
        ),
        "passed": passed,
        "decision": (
            "authorize_runtime_freeze" if passed else "reject_runtime_freeze"
        ),
    }


def derive_placebo_donor_mapping(
    groups: Sequence[str] | np.ndarray,
    references: Sequence[bool] | np.ndarray,
    *,
    scenario: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Derive a deranged donor map without reading any outcome labels."""

    group_values = np.asarray(groups, dtype=str)
    reference_mask = np.asarray(references, dtype=bool)
    if group_values.ndim != 1 or reference_mask.shape != group_values.shape:
        raise ValueError("placebo group and reference rows are not aligned")
    rows = {
        group: np.flatnonzero(group_values == group)
        for group in sorted(set(group_values.tolist()))
    }
    buckets: dict[int, list[str]] = {}
    for group, group_rows in rows.items():
        if np.count_nonzero(reference_mask[group_rows]) != 1:
            raise ValueError(f"{scenario}/{group}: expected one PhasePressure reference")
        buckets.setdefault(int(group_rows.size), []).append(group)

    mapping: dict[str, str] = {}
    bucket_audit = {}
    for action_count, names in sorted(buckets.items()):
        if len(names) < 2:
            raise ValueError(
                f"{scenario}: action-count bucket {action_count} has no placebo donor"
            )
        bucket_seed = int.from_bytes(
            hashlib.sha256(
                f"{PLACEBO_MAPPING_PROTOCOL}|{int(seed)}|{scenario}|{action_count}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            byteorder="little",
            signed=False,
        )
        shuffled = list(names)
        np.random.default_rng(bucket_seed).shuffle(shuffled)
        donors = shuffled[1:] + shuffled[:1]
        for destination, donor in zip(shuffled, donors, strict=True):
            mapping[destination] = donor
        bucket_audit[str(action_count)] = {
            "group_count": len(names),
            "bucket_seed": bucket_seed,
        }
    if set(mapping) != set(rows) or any(mapping[name] == name for name in mapping):
        raise ValueError("placebo donor map is incomplete or contains a fixed point")
    if set(mapping.values()) != set(mapping):
        raise ValueError("placebo donor map is not one-to-one")
    mapping_rows = [
        {
            "destination_group": group,
            "donor_group": mapping[group],
            "action_count": int(rows[group].size),
        }
        for group in sorted(mapping)
    ]
    return mapping, {
        "protocol": PLACEBO_MAPPING_PROTOCOL,
        "scenario": str(scenario),
        "seed": int(seed),
        "group_count": len(mapping),
        "bucket_count": len(buckets),
        "buckets": bucket_audit,
        "mapping_sha256": _canonical_sha256(mapping_rows),
        "mapping_uses": ["scenario", "action_group_id", "action_count", "seed"],
        "outcome_labels_used_to_derive_mapping": 0,
        "all_groups_moved": True,
        "mapping_rows": mapping_rows,
    }


def source_label_block_placebo(
    dataset: MechanismDataset,
    *,
    scenario: str,
    seed: int,
    target_names: Sequence[str],
) -> tuple[MechanismDataset, dict[str, Any]]:
    """Move complete donor non-reference delta vectors onto recipient states."""

    groups = np.asarray(action_group_ids(dataset), dtype=str)
    reference_by_group = rule_reference_indices(dataset, policy="phase_pressure")
    references = np.asarray(
        [index == int(reference_by_group[group]) for index, group in enumerate(groups)],
        dtype=bool,
    )
    mapping, audit = derive_placebo_donor_mapping(
        groups, references, scenario=scenario, seed=seed
    )
    group_rows = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))
    }
    targets = {
        name: np.asarray(values, dtype=float).copy()
        for name, values in dataset.targets.items()
    }
    requested = tuple(str(name) for name in target_names)
    if not requested or any(name not in targets for name in requested):
        raise ValueError(f"{scenario}: placebo target schema changed")
    for target_name in requested:
        original = np.asarray(dataset.targets[target_name], dtype=float)
        output = original.copy()
        for destination, donor in mapping.items():
            destination_rows = group_rows[destination]
            donor_rows = group_rows[donor]
            destination_reference = destination_rows[references[destination_rows]]
            donor_reference = donor_rows[references[donor_rows]]
            destination_nonreference = destination_rows[~references[destination_rows]]
            donor_nonreference = donor_rows[~references[donor_rows]]
            if (
                destination_reference.size != 1
                or donor_reference.size != 1
                or destination_nonreference.size != donor_nonreference.size
            ):
                raise ValueError(f"{scenario}: placebo action block shape changed")
            baseline = float(original[int(destination_reference[0])])
            donor_delta = (
                original[donor_nonreference]
                - float(original[int(donor_reference[0])])
            )
            output[int(destination_reference[0])] = baseline
            output[destination_nonreference] = baseline + donor_delta
        targets[target_name] = output
    metadata = {
        **dict(dataset.metadata),
        "source_label_placebo": {
            key: value for key, value in audit.items() if key != "mapping_rows"
        },
    }
    return (
        MechanismDataset(
            feature_names=dataset.feature_names,
            features=dataset.features,
            context_names=dataset.context_names,
            context=dataset.context,
            priors=dataset.priors,
            targets=targets,
            domains=dataset.domains,
            metadata=metadata,
        ),
        audit,
    )


def _fit_task_roster(source_groups: Sequence[str]) -> tuple[tuple[str, str | None], ...]:
    groups = tuple(sorted(str(value) for value in source_groups))
    return (
        ("target_v157a_audit", None),
        ("target_domain_aligned", None),
        *(("source_aligned", group) for group in groups),
        *(("source_label_placebo", group) for group in groups),
    )


def _freeze_fit_worker(task: tuple[str, str | None]) -> dict[str, Any]:
    if _FREEZE_FIT_STATE is None:
        raise RuntimeError("V157B refit worker state is missing")
    from threadpoolctl import threadpool_limits

    arm, source_group = task
    state = _FREEZE_FIT_STATE
    with threadpool_limits(limits=1):
        if arm in {"target_v157a_audit", "target_domain_aligned"}:
            if source_group is not None:
                raise ValueError("V157B target refit has a source group")
            model = _target_model(
                state[
                    "adaptation_dataset"
                    if arm == "target_v157a_audit"
                    else "domain_aligned_adaptation_dataset"
                ],
                group_ids=state["selected_group_ids"],
                candidate=str(state["candidate"]),
            )
            return {"arm": arm, "source_group": None, "model": model}
        if source_group is None or arm not in {
            "source_aligned",
            "source_label_placebo",
        }:
            raise ValueError(f"unknown V157B refit task: {task}")
        component_state = state[
            "aligned_states" if arm == "source_aligned" else "placebo_states"
        ][source_group]
        fitted = _fit_component_budget(
            component_state,
            target_group_ids=state["selected_group_ids"],
        )
    expected_domains = {str(source_group), "jinan"}
    if set(fitted.diagnostics["source_domains"]) != expected_domains:
        raise ValueError(f"{source_group}: {arm} information boundary changed")
    return {
        "arm": str(arm),
        "source_group": str(source_group),
        "fitted": fitted,
        "diagnostics": fitted.diagnostics,
    }


def _fit_runtime_components(
    *,
    adaptation_dataset: MechanismDataset,
    selected_group_ids: Sequence[str],
    candidate: str,
    aligned_states: Mapping[str, Mapping[str, Any]],
    placebo_states: Mapping[str, Mapping[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if tuple(sorted(aligned_states)) != tuple(sorted(placebo_states)):
        raise ValueError("V157B aligned and placebo source rosters differ")
    tasks = _fit_task_roster(aligned_states)
    actual_workers = min(max(int(workers), 1), len(tasks))
    global _FREEZE_FIT_STATE
    _FREEZE_FIT_STATE = {
        "adaptation_dataset": adaptation_dataset,
        "domain_aligned_adaptation_dataset": _relabel_dataset_domain(
            adaptation_dataset, "jinan"
        ),
        "selected_group_ids": tuple(str(value) for value in selected_group_ids),
        "candidate": str(candidate),
        "aligned_states": aligned_states,
        "placebo_states": placebo_states,
    }
    try:
        if actual_workers == 1:
            return [_freeze_fit_worker(task) for task in tasks]
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("parallel V157B B100 refit requires Linux fork")
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=mp.get_context("fork"),
        ) as pool:
            return list(pool.map(_freeze_fit_worker, tasks))
    finally:
        _FREEZE_FIT_STATE = None


def _fit_structure(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    fit = dict(diagnostics.get("fit", {}))
    anchor = dict(fit.get(ANCHOR_FAMILY, {}))
    correction = dict(fit.get(CORRECTION_FAMILY, {}))
    anchor_keys = (
        "estimator",
        "causal",
        "feature_names",
        "feature_count",
        "rows",
        "training_rows",
        "candidate_only",
        "target_normalization_protocol",
    )
    correction_keys = (
        "estimator",
        "causal",
        "state_feature_names",
        "action_feature_names",
        "state_feature_count",
        "action_feature_count",
        "action_group_count",
        "unordered_pair_count",
        "oriented_pair_count",
        "antisymmetric_augmentation",
        "target_normalization_protocol",
    )
    return {
        "anchor": {name: anchor.get(name) for name in anchor_keys},
        "correction": {name: correction.get(name) for name in correction_keys},
        "source_scenarios": list(diagnostics.get("source_scenarios", ())),
        "target_adaptation_group_ids": list(
            diagnostics.get("target_adaptation_group_ids", ())
        ),
        "source_domains": list(diagnostics.get("source_domains", ())),
    }


def load_runtime_arms(
    bundle_path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = Path(bundle_path)
    if expected_sha256 is not None and _sha256(path) != str(expected_sha256):
        raise ValueError("V157B runtime bundle identity changed")
    payload = pickle.loads(path.read_bytes())
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != RUNTIME_BUNDLE_PROTOCOL
        or tuple(payload.get("arm_order", ())) != ARM_ORDER
        or set(payload.get("arms", {})) != set(ARM_ORDER)
        or payload["arms"]["phase_pressure"].get("model") is not None
    ):
        raise ValueError("V157B runtime bundle contract changed")
    return payload


def runtime_arm_model(payload: Mapping[str, Any], arm: str) -> Any | None:
    if payload.get("protocol") != RUNTIME_BUNDLE_PROTOCOL or arm not in ARM_ORDER:
        raise ValueError(f"unknown V157B runtime arm: {arm}")
    return payload["arms"][arm].get("model")


def run_feature_aligned_b100_runtime_freeze(
    *,
    config_path: Path,
    v157a_aggregate_path: Path,
    v157a_prediction_artifact_path: Path,
    fit_result_path: Path,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite V157B freeze: {output_root}")
    config = _read_json(config_path)
    identities = dict(config.get("frozen_identities", {}))
    if (
        config.get("protocol") != CONFIG_PROTOCOL
        or tuple(config.get("arm_order", ())) != ARM_ORDER
        or tuple(config.get("contrast_features", ())) != tuple(CONTRAST_FEATURES_V3)
        or tuple(config.get("rigid_model_feature_names", ()))
        != tuple(CAUSAL_RIGID_RANKING_PARENTS)
    ):
        raise ValueError("V157B config contract changed")
    local_inputs = {
        "v157a_aggregate": v157a_aggregate_path,
        "v157a_prediction_artifact": v157a_prediction_artifact_path,
        "fit_result": fit_result_path,
        "source_cache_audit": source_cache_audit_path,
        "target_cache_audit": target_cache_audit_path,
        "selector_cache_audit": selector_cache_audit_path,
    }
    observed_hashes = {name: _sha256(path) for name, path in local_inputs.items()}
    expected_hashes = {name: identities[f"{name}_sha256"] for name in local_inputs}
    if observed_hashes != expected_hashes:
        raise ValueError(f"V157B frozen input identity changed: {observed_hashes}")

    aggregate = _read_json(v157a_aggregate_path)
    b100 = aggregate.get("b100_branch_authorization", {})
    if (
        aggregate.get("protocol") != V157A_AGGREGATE_PROTOCOL
        or b100.get("passed") is not True
        or aggregate.get("budget_comparison", {}).get("100", {}).get(
            "corrected", {}
        ).get("selected_candidate_counts")
        != {"uniform_all_sources__source_weight_1": 22}
    ):
        raise ValueError("V157A does not authorize the uniform B100 runtime freeze")
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=identities["fit_result_sha256"]
    )
    _target_cache_audit_contract(target_cache_audit_path, fit_result)
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    selector_seeds = tuple(int(value) for value in config["selector_seeds"])
    selector_scenario = str(config["selector_scenario"])
    selector_shards = int(config["selector_collection_shards"])
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=identities["selector_cache_audit_sha256"],
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_shards,
    )
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_shards,
        cache_workers=cache_workers,
    )
    selector_contrast = build_action_contrast_dataset(
        selector,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    v157a_artifact = pickle.loads(Path(v157a_prediction_artifact_path).read_bytes())
    expected_groups = tuple(str(value) for value in config["b100_group_ids"])
    if (
        tuple(v157a_artifact.get("selected_group_ids", {}).get(100, ()))
        != expected_groups
        or tuple(int(value) for value in v157a_artifact.get("selector_seeds", ()))
        != selector_seeds
        or tuple(v157a_artifact.get("source_group_order", ()))
        != tuple(config["source_group_order"])
    ):
        raise ValueError("V157A artifact B100 groups changed")
    protocol, source_bank, source_scenarios, source_bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    adaptation, target_context, target_bank_audit, target_scenarios = (
        _adaptation_inputs(
            cache_root=target_cache_root,
            manifest_path=target_manifest_path,
            cache_workers=cache_workers,
            selected_group_ids=expected_groups,
        )
    )
    placebo_seed = int(config["source_label_placebo"]["seed"])
    target_names = tuple(config["source_label_placebo"]["permuted_target_names"])
    placebo_bank = {}
    mapping_rows = {}
    mapping_summaries = {}
    for scenario, dataset in source_bank.items():
        placebo_bank[scenario], audit = source_label_block_placebo(
            dataset,
            scenario=scenario,
            seed=placebo_seed,
            target_names=target_names,
        )
        mapping_rows[scenario] = audit.pop("mapping_rows")
        mapping_summaries[scenario] = audit
    candidate = str(config["frozen_candidate"])
    rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    aligned_target_key = "waiting_aligned_target_jinan_v123"
    placebo_target_key = "waiting_aligned_target_jinan_v157b_v2_placebo"
    aligned_states = {}
    placebo_states = {}
    for source_group, scenarios in source_scenarios.items():
        aligned_bank = {scenario: source_bank[scenario] for scenario in scenarios}
        aligned_bank[aligned_target_key] = adaptation
        placebo_fit_bank = {
            scenario: placebo_bank[scenario] for scenario in scenarios
        }
        placebo_fit_bank[placebo_target_key] = adaptation
        city_groups = {scenario: source_group for scenario in scenarios}
        aligned_city_groups = {**city_groups, aligned_target_key: "jinan"}
        placebo_city_groups = {**city_groups, placebo_target_key: "jinan"}
        common = {
            "selected_group_ids": expected_groups,
            "target_static_context": target_context,
            "prior_policy": "phase_pressure",
            "source_rule_specs": rule_specs,
            "city": "jinan",
        }
        aligned_states[source_group] = {
            **common,
            "fit_bank": aligned_bank,
            "target_key": aligned_target_key,
            "city_groups": aligned_city_groups,
        }
        placebo_states[source_group] = {
            **common,
            "fit_bank": placebo_fit_bank,
            "target_key": placebo_target_key,
            "city_groups": placebo_city_groups,
        }
    if tuple(sorted(aligned_states)) != tuple(config["source_group_order"]):
        raise ValueError("V157B source group order changed before refit")

    fitted_rows = _fit_runtime_components(
        adaptation_dataset=adaptation,
        selected_group_ids=expected_groups,
        candidate=candidate,
        aligned_states=aligned_states,
        placebo_states=placebo_states,
        workers=fit_workers,
    )
    target_audit_rows = [
        row for row in fitted_rows if row["arm"] == "target_v157a_audit"
    ]
    target_runtime_rows = [
        row for row in fitted_rows if row["arm"] == "target_domain_aligned"
    ]
    if len(target_audit_rows) != 1 or len(target_runtime_rows) != 1:
        raise ValueError("V157B target refit roster changed")
    v157a_target_audit_model = target_audit_rows[0]["model"]
    target_model = target_runtime_rows[0]["model"]
    aligned_fits = {
        row["source_group"]: row
        for row in fitted_rows
        if row["arm"] == "source_aligned"
    }
    placebo_fits = {
        row["source_group"]: row
        for row in fitted_rows
        if row["arm"] == "source_label_placebo"
    }
    if (
        tuple(sorted(aligned_fits)) != tuple(config["source_group_order"])
        or tuple(sorted(placebo_fits)) != tuple(config["source_group_order"])
    ):
        raise ValueError("V157B source refit roster is incomplete")
    source_components = _anchored_source_components(
        {
            "component_models": {
                group: row["fitted"] for group, row in aligned_fits.items()
            }
        },
        candidate=candidate,
    )
    placebo_components = _anchored_source_components(
        {
            "component_models": {
                group: row["fitted"] for group, row in placebo_fits.items()
            }
        },
        candidate=candidate,
    )
    source_runtime = UniformAnchoredRuntimeModel(source_components)
    selector_predictions = _build_selector_prediction_cache(
        v157a_audit_target_model=v157a_target_audit_model,
        domain_aligned_target_model=target_model,
        source_components=source_components,
        uniform_runtime_model=source_runtime,
        selector_contrast=selector_contrast,
    )
    refit_identity_audit = validate_reused_prediction_identity(
        target_model=v157a_target_audit_model,
        source_components=source_components,
        selector_contrast=selector_contrast,
        v157a_artifact=v157a_artifact,
        comparison="v157b_v2_exact_b100_refits_vs_v157a_b100_predictions",
        require_exact=True,
        authorize_runtime_reuse=False,
        target_prediction_arrays=selector_predictions.v157a_audit_target,
        source_prediction_arrays=selector_predictions.source_components,
        uniform_runtime_prediction_arrays=selector_predictions.uniform_runtime,
        feature_order_checks=selector_predictions.v157a_audit_feature_order,
    )
    runtime_domain_alignment_diagnostic = validate_reused_prediction_identity(
        target_model=target_model,
        source_components=source_components,
        selector_contrast=selector_contrast,
        v157a_artifact=v157a_artifact,
        comparison=(
            "domain_aligned_runtime_target_and_exact_source_refits_vs_"
            "v157a_predictions"
        ),
        require_exact=False,
        target_prediction_arrays=selector_predictions.domain_aligned_target,
        source_prediction_arrays=selector_predictions.source_components,
        uniform_runtime_prediction_arrays=selector_predictions.uniform_runtime,
        feature_order_checks=selector_predictions.domain_aligned_feature_order,
    )
    if not all(
        runtime_domain_alignment_diagnostic[
            "per_source_score_uncertainty_trust_equal"
        ].values()
    ):
        raise ValueError("V157B aligned source refits lost V157A identity")

    placebo_structure_equal = {
        source_group: _fit_structure(placebo_fits[source_group]["diagnostics"])
        == _fit_structure(aligned_fits[source_group]["diagnostics"])
        for source_group in sorted(aligned_fits)
    }
    expected_anchor_names = tuple(config["rigid_model_feature_names"])
    expected_anchor_config = dict(config["anchor_hgb_settings"])
    expected_correction_config = dict(config["correction_hgb_settings"])
    runtime_setting_equal = {
        "target_v157a_audit": bool(
            tuple(v157a_target_audit_model.anchor_model.feature_names)
            == expected_anchor_names
            and asdict(v157a_target_audit_model.anchor_model.config)
            == expected_anchor_config
            and asdict(v157a_target_audit_model.correction_model.config)
            == expected_correction_config
        ),
        "target_only": bool(
            tuple(target_model.anchor_model.feature_names) == expected_anchor_names
            and asdict(target_model.anchor_model.config) == expected_anchor_config
            and asdict(target_model.correction_model.config)
            == expected_correction_config
        ),
        **{
            f"source_aligned:{source_group}": bool(
                tuple(model.anchor_model.feature_names) == expected_anchor_names
                and asdict(model.anchor_model.config) == expected_anchor_config
                and asdict(model.correction_model.config)
                == expected_correction_config
            )
            for source_group, model in source_components.items()
        },
        **{
            f"source_label_placebo:{source_group}": bool(
                tuple(model.anchor_model.feature_names) == expected_anchor_names
                and asdict(model.anchor_model.config) == expected_anchor_config
                and asdict(model.correction_model.config)
                == expected_correction_config
            )
            for source_group, model in placebo_components.items()
        },
    }
    if not all(placebo_structure_equal.values()) or not all(
        runtime_setting_equal.values()
    ):
        raise ValueError("V157B refit or placebo capacity contract changed")
    placebo_runtime = UniformAnchoredRuntimeModel(placebo_components)

    gate_spec = dict(config["domain_aligned_source_gate"])
    if (
        gate_spec.get("protocol")
        != "v157b-domain-aligned-source-minus-target-gate-v2"
        or gate_spec.get("candidate") != "uniform_all_sources__source_weight_1"
        or float(gate_spec.get("source_weight", float("nan"))) != 1.0
        or int(gate_spec.get("selector_seed_count", -1)) != len(selector_seeds)
    ):
        raise ValueError("V157B domain-aligned source gate contract changed")
    source_gate = domain_aligned_source_gate(
        selector=selector,
        selector_contrast=selector_contrast,
        target_model=target_model,
        source_model=source_runtime,
        selector_seeds=selector_seeds,
        mean_maximum=float(gate_spec["mean_maximum"]),
        upper_95_maximum=float(gate_spec["upper_95_strict_maximum"]),
        target_prediction_score=selector_predictions.domain_aligned_target[0],
        source_prediction_score=selector_predictions.uniform_runtime[0],
    )
    v1_failure_provenance = dict(config["v1_failure_provenance"])
    if not source_gate["passed"]:
        output_root.mkdir(parents=True, exist_ok=False)
        rejected = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": (
                "domain_aligned_source_gate_rejected_runtime_not_frozen"
            ),
            "city": "jinan",
            "arm_order": list(ARM_ORDER),
            "runtime_artifacts_written": False,
            "v157c_authorized": False,
            "v1_failure_provenance": v1_failure_provenance,
            "v157a_exact_refit_identity_audit": refit_identity_audit,
            "runtime_domain_alignment_diagnostic": (
                runtime_domain_alignment_diagnostic
            ),
            "domain_aligned_source_gate": source_gate,
            "b100_fit_roster": {
                "target_v157a_audit_only": 1,
                "target_domain_aligned_runtime": 1,
                "source_aligned_components": len(source_components),
                "source_label_placebo_components": len(placebo_components),
                "total_fit_tasks": len(fitted_rows),
            },
            "placebo_capacity_audit": {
                "scenario_summaries": mapping_summaries,
                "fit_structure_equal": placebo_structure_equal,
                "runtime_feature_and_hgb_settings_equal": runtime_setting_equal,
                "passed": True,
            },
            "inputs": observed_hashes,
            "bank_audits": {
                "source": source_bank_audit,
                "target": target_bank_audit,
                "selector": selector_bank_audit,
                "selector_cache": selector_audit,
            },
            "claim_boundary": (
                "V157B v2 rejected the corrected source-minus-target premise; "
                "no runtime bundle or V157C authorization was created."
            ),
        }
        _atomic_json(output_root / "result.json", rejected)
        return rejected

    output_root.mkdir(parents=True, exist_ok=False)
    mapping_path = output_root / "placebo_donor_mapping.json"
    mapping_payload = {
        "protocol": PLACEBO_MAPPING_PROTOCOL,
        "seed": placebo_seed,
        "mapping_derived_from_training_bank_only": True,
        "evaluation_outcomes_consumed": 0,
        "target_city_labels_permuted": False,
        "scenarios": mapping_rows,
    }
    _atomic_json(mapping_path, mapping_payload)
    runtime_path = output_root / "runtime_models.pkl"
    runtime_payload = {
        "protocol": RUNTIME_BUNDLE_PROTOCOL,
        "city": "jinan",
        "prior_policy": "phase_pressure",
        "contrast_features": tuple(CONTRAST_FEATURES_V3),
        "rigid_model_feature_names": tuple(CAUSAL_RIGID_RANKING_PARENTS),
        "b100_group_ids": expected_groups,
        "source_group_order": tuple(config["source_group_order"]),
        "arm_order": ARM_ORDER,
        "runtime_target_comparator": "domain_aligned_b100_target_only_v2",
        "arms": {
            "target_only": {
                "kind": "learned_model",
                "model": target_model,
                "source_rows": False,
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "kind": "learned_model",
                "model": source_runtime,
                "source_rows": True,
                "source_labels_permuted": False,
                "training_path": "exact_v157a_source_component_refits",
            },
            "source_label_placebo": {
                "kind": "learned_model",
                "model": placebo_runtime,
                "source_rows": True,
                "source_labels_permuted": True,
                "target_city_labels_permuted": False,
            },
            "phase_pressure": {
                "kind": "reference_policy",
                "model": None,
                "policy": "phase_pressure",
            },
        },
    }
    _atomic_bytes(
        runtime_path,
        pickle.dumps(runtime_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    round_trip = load_runtime_arms(runtime_path)
    if any(runtime_arm_model(round_trip, arm) is None for arm in LEARNED_ARMS):
        raise ValueError("V157B learned runtime arm round trip failed")

    manifest_path = output_root / "arm_manifest.json"
    arm_manifest = {
        "protocol": ARM_MANIFEST_PROTOCOL,
        "city": "jinan",
        "arm_order": list(ARM_ORDER),
        "runtime_bundle": {
            "path": str(runtime_path.resolve()),
            "sha256": _sha256(runtime_path),
            "size_bytes": runtime_path.stat().st_size,
            "protocol": RUNTIME_BUNDLE_PROTOCOL,
        },
        "arms": {
            "target_only": {
                "kind": "learned_model",
                "bundle_key": "target_only",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "kind": "learned_model",
                "bundle_key": "uniform_source",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_path": "exact_v157a_source_component_refits",
                "source_component_count": len(source_components),
            },
            "source_label_placebo": {
                "kind": "learned_model",
                "bundle_key": "source_label_placebo",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "source_component_count": len(placebo_components),
                "source_labels_permuted": True,
                "target_city_labels_permuted": False,
            },
            "phase_pressure": {
                "kind": "reference_policy",
                "policy": "phase_pressure",
                "bundle_key": "phase_pressure",
            },
        },
        "scope": "one_frozen_focal_tls_action_then_phase_pressure_continuation",
    }
    _atomic_json(manifest_path, arm_manifest)
    result = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "runtime_arms_frozen_branch_not_run",
        "city": "jinan",
        "arm_order": list(ARM_ORDER),
        "runtime_artifacts_written": True,
        "v157c_authorized": True,
        "information_budget": {
            "target_action_groups": 100,
            "target_group_ids_sha256": config["b100_group_ids_sha256"],
            "source_group_count": len(source_components),
            "v1_legacy_models_used_at_runtime": False,
            "b100_fit_roster": {
                "target_v157a_audit_only": 1,
                "target_domain_aligned_runtime": 1,
                "source_aligned_components": len(source_components),
                "source_label_placebo_components": len(placebo_components),
                "total_fit_tasks": len(fitted_rows),
                "shared_process_pool_workers": min(
                    max(int(fit_workers), 1), len(fitted_rows)
                ),
            },
        },
        "v1_failure_provenance": v1_failure_provenance,
        "v157a_exact_refit_identity_audit": refit_identity_audit,
        "runtime_domain_alignment_diagnostic": runtime_domain_alignment_diagnostic,
        "domain_aligned_source_gate": source_gate,
        "placebo": {
            "protocol": PLACEBO_MAPPING_PROTOCOL,
            "seed": placebo_seed,
            "mapping_artifact": {
                "path": str(mapping_path.resolve()),
                "sha256": _sha256(mapping_path),
                "size_bytes": mapping_path.stat().st_size,
            },
            "scenario_summaries": mapping_summaries,
            "fit_diagnostics": {
                source_group: row["diagnostics"]
                for source_group, row in placebo_fits.items()
            },
            "capacity_match_audit": {
                "fit_structure_equal": placebo_structure_equal,
                "runtime_feature_and_hgb_settings_equal": runtime_setting_equal,
                "passed": True,
            },
            "same_model_families": list(BASE_FAMILIES),
            "same_target_b100_groups": True,
            "target_city_labels_permuted": False,
            "evaluation_outcomes_consumed": 0,
        },
        "artifacts": {
            "runtime_models": arm_manifest["runtime_bundle"],
            "arm_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "protocol": ARM_MANIFEST_PROTOCOL,
            },
        },
        "inputs": observed_hashes,
        "bank_audits": {
            "source": source_bank_audit,
            "target": target_bank_audit,
            "selector": selector_bank_audit,
            "selector_cache": selector_audit,
        },
        "target_scenarios": list(target_scenarios),
        "fit_protocol": protocol["protocol"],
        "claim_boundary": (
            "V157B v2 freezes three learned action rankers only after the exact "
            "V157A audit and corrected domain-aligned source gate pass. This is "
            "model preparation, not a branch or controller-performance result."
        ),
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v157a-aggregate", type=Path, required=True)
    parser.add_argument("--v157a-prediction-artifact", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=16)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_feature_aligned_b100_runtime_freeze(
        config_path=args.config,
        v157a_aggregate_path=args.v157a_aggregate,
        v157a_prediction_artifact_path=args.v157a_prediction_artifact,
        fit_result_path=args.fit_result,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
        output_root=args.output_root,
    )
    print(json.dumps({"status": result["scientific_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
