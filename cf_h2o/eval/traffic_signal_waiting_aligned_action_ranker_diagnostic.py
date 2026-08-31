"""Seed-blocked OOF diagnostic for waiting-aligned direct action ranking."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    frozen_proposal_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_veto_diagnostic import (
    RESULT_PROTOCOL as VETO_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_ranker import (
    ActionAdvantageConfig,
    AntisymmetricPairwiseActionRegressor,
    PairwiseActionAdvantageRegressor,
    PairwisePreferenceConfig,
    group_normalized_action_target,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_action_ranker_diagnostic import (
    PARTITION_PROTOCOL,
    PROTOCOL,
    VETO_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_veto_diagnostic import (
    PROTOCOL as VETO_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-waiting-aligned-action-ranker-diagnostic-result-v1"
AUTHORIZATION_DECISION = "authorize_waiting_aligned_action_ranker_artifact_freeze"
REJECTION_DECISION = "retain_phase_pressure_and_reject_waiting_aligned_action_ranker"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _row_subset(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    """Subset every row-aligned field, including action-state metadata."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (dataset.size,):
        raise ValueError("row subset mask is not aligned with the dataset")
    subset = dataset.subset(mask)
    metadata = dict(dataset.metadata)
    for key, value in tuple(metadata.items()):
        array = np.asarray(value)
        if array.shape == (dataset.size,):
            metadata[key] = array[mask].tolist()
    return MechanismDataset(
        feature_names=subset.feature_names,
        features=subset.features,
        context_names=subset.context_names,
        context=subset.context,
        priors=subset.priors,
        targets=subset.targets,
        domains=subset.domains,
        metadata=metadata,
    )


def _build_model(spec: Mapping[str, Any]) -> Any:
    family = str(spec["family"])
    config = dict(spec["config"])
    if family == "antisymmetric_pairwise":
        return AntisymmetricPairwiseActionRegressor(
            config=PairwisePreferenceConfig(**config),
            state_feature_names=tuple(spec["state_feature_names"]),
            action_feature_names=tuple(spec["action_feature_names"]),
        )
    if family == "group_normalized_advantage":
        return PairwiseActionAdvantageRegressor(
            causal=True,
            config=ActionAdvantageConfig(**config),
            causal_feature_names=tuple(spec["causal_feature_names"]),
        )
    raise ValueError(f"unknown waiting-aligned ranker family: {family!r}")


def _argmin_with_reference_tie_break(
    rows: np.ndarray,
    values: np.ndarray,
    *,
    reference: int,
    candidate_states: np.ndarray,
) -> int:
    minimum = float(np.min(values[rows]))
    tied = rows[np.isclose(values[rows], minimum, rtol=0.0, atol=1e-12)]
    if int(reference) in {int(value) for value in tied}:
        return int(reference)
    return min(
        (int(value) for value in tied),
        key=lambda row: (str(candidate_states[row]), row),
    )


def _fit_oof_fold(
    *,
    heldout_seed: int,
    contrast: MechanismDataset,
    model_specs: Sequence[Mapping[str, Any]],
    scenario: str,
    expected_action_count: int,
) -> dict[str, Any]:
    groups = action_group_ids(contrast).astype(str)
    unique_groups = np.unique(groups)
    seed_by_group = {
        str(group): parse_action_group_seed(str(group)) for group in unique_groups
    }
    training_mask = np.asarray(
        [seed_by_group[str(group)] != int(heldout_seed) for group in groups],
        dtype=bool,
    )
    validation_mask = ~training_mask
    training = _row_subset(contrast, training_mask)
    validation = _row_subset(contrast, validation_mask)
    training_groups = set(
        str(value) for value in training.metadata["action_group_ids"]
    )
    validation_groups = set(
        str(value) for value in validation.metadata["action_group_ids"]
    )
    if (
        not training_groups
        or not validation_groups
        or any(
            parse_action_group_seed(group) == int(heldout_seed)
            for group in training_groups
        )
        or any(
            parse_action_group_seed(group) != int(heldout_seed)
            for group in validation_groups
        )
        or training_groups.intersection(validation_groups)
    ):
        raise RuntimeError("waiting-aligned OOF seed partition leaked")

    validation_group_ids = action_group_ids(validation).astype(str)
    is_reference = np.asarray(validation.metadata["is_reference"], dtype=bool)
    candidate_states = np.asarray(
        validation.metadata.get(
            "candidate_states", [f"row{index}" for index in range(validation.size)]
        ),
        dtype=str,
    )
    if candidate_states.shape != (validation.size,):
        raise ValueError("candidate-state metadata is not aligned after OOF subsetting")
    actual_raw = np.asarray(validation.targets["interval_cost"], dtype=float)
    actual_normalized, _ = group_normalized_action_target(
        validation, "interval_cost"
    )
    if not np.isfinite(actual_raw).all() or not np.isfinite(actual_normalized).all():
        raise ValueError("waiting-aligned OOF targets are not finite")

    model_results = []
    for model_spec in model_specs:
        model = _build_model(model_spec)
        fit = model.fit(training)
        prediction = model.predict(validation)["control_cost"]
        score = np.asarray(prediction["mean"], dtype=float)
        uncertainty = np.asarray(prediction["uncertainty"], dtype=float)
        trust = np.asarray(prediction["context_trust"], dtype=float)
        if not (
            np.isfinite(score).all()
            and np.isfinite(uncertainty).all()
            and np.isfinite(trust).all()
        ):
            raise ValueError("waiting-aligned ranker emitted non-finite predictions")
        records = []
        for group in sorted(validation_groups):
            rows = np.flatnonzero(validation_group_ids == group)
            references = rows[is_reference[rows]]
            if rows.size != int(expected_action_count) or references.size != 1:
                raise ValueError(
                    f"action group {group!r} violates the frozen eight-action contract"
                )
            reference = int(references[0])
            selected = _argmin_with_reference_tie_break(
                rows,
                score,
                reference=reference,
                candidate_states=candidate_states,
            )
            oracle = _argmin_with_reference_tie_break(
                rows,
                actual_raw,
                reference=reference,
                candidate_states=candidate_states,
            )
            records.append(
                {
                    "model_key": str(model_spec["key"]),
                    "model_family": str(model_spec["family"]),
                    "heldout_seed": int(heldout_seed),
                    "simulator_seed": int(heldout_seed),
                    "scenario": str(scenario),
                    "group_id": str(group),
                    "action_count": int(rows.size),
                    "selected_candidate_state": str(candidate_states[selected]),
                    "reference_candidate_state": str(candidate_states[reference]),
                    "oracle_candidate_state": str(candidate_states[oracle]),
                    "selected_differs": bool(selected != reference),
                    "selected_is_oracle": bool(selected == oracle),
                    "predicted_delta": float(score[selected] - score[reference]),
                    "selected_uncertainty": float(
                        uncertainty[selected] + uncertainty[reference]
                    ),
                    "selected_context_trust": float(
                        min(trust[selected], trust[reference])
                    ),
                    "actual_group_normalized_delta": float(
                        actual_normalized[selected] - actual_normalized[reference]
                    ),
                    "actual_raw_delta": float(
                        actual_raw[selected] - actual_raw[reference]
                    ),
                    "oracle_group_normalized_delta": float(
                        actual_normalized[oracle] - actual_normalized[reference]
                    ),
                    "normalized_regret_to_oracle": float(
                        actual_normalized[selected] - actual_normalized[oracle]
                    ),
                }
            )
        model_results.append(
            {
                "model_key": str(model_spec["key"]),
                "model_family": str(model_spec["family"]),
                "fit_diagnostics": fit,
                "records": records,
            }
        )
    return {
        "heldout_seed": int(heldout_seed),
        "training_group_count": len(training_groups),
        "validation_group_count": len(validation_groups),
        "training_validation_disjoint": not bool(
            training_groups.intersection(validation_groups)
        ),
        "heldout_absent_from_training": all(
            parse_action_group_seed(group) != int(heldout_seed)
            for group in training_groups
        ),
        "model_results": model_results,
    }


def _bootstrap_seed_means(
    seed_means: Mapping[int, float],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [float(seed_means[seed]) for seed in sorted(seed_means)], dtype=float
    )
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("ranker bootstrap requires finite seed means")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.choice(
        values, size=(int(replicates), values.size), replace=True
    ).mean(axis=1)
    return {
        "unit": "simulator_seed_equal_scenario_mean",
        "replicates": int(replicates),
        "seed": int(bootstrap_seed),
        "observed": float(values.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _seed_means(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    delta_key: str,
) -> dict[int, float]:
    result = {}
    for seed in seeds:
        values = [
            float(row[delta_key])
            for row in records
            if int(row["simulator_seed"]) == int(seed)
        ]
        if not values:
            raise ValueError(f"ranker records do not cover seed {seed}")
        result[int(seed)] = float(np.mean(values))
    return result


def _policy_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    delta_key: str,
    active_key: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    seed_means = _seed_means(records, seeds=seeds, delta_key=delta_key)
    values = np.asarray(list(seed_means.values()), dtype=float)
    active = [row for row in records if bool(row[active_key])]
    harmful_fraction = (
        float(np.mean([float(row[delta_key]) > 0.0 for row in active]))
        if active
        else 0.0
    )
    return {
        "group_count": len(records),
        "active_group_count": len(active),
        "active_seed_count": len(
            {int(row["simulator_seed"]) for row in active}
        ),
        "harmful_active_group_fraction": harmful_fraction,
        "equal_seed_mean_delta": float(np.mean(values)),
        "seed_delta_std": float(np.std(values)),
        "worst_seed_delta": float(np.max(values)),
        "seed_mean_delta": {
            str(seed): float(seed_means[seed]) for seed in sorted(seed_means)
        },
        "bootstrap": _bootstrap_seed_means(
            seed_means,
            replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["seed"]),
        ),
    }


def summarize_gate_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    risk_multiplier: float,
    minimum_context_trust: float,
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated = []
    for raw in records:
        row = dict(raw)
        upper = float(row["predicted_delta"]) + float(risk_multiplier) * float(
            row["selected_uncertainty"]
        )
        accepted = bool(
            row["selected_differs"]
            and float(row["selected_context_trust"])
            >= float(minimum_context_trust)
            and upper < 0.0
        )
        evaluated.append(
            {
                **row,
                "predicted_upper": upper,
                "ranker_accepted": accepted,
                "deployed_delta": (
                    float(row["actual_group_normalized_delta"])
                    if accepted
                    else 0.0
                ),
            }
        )
    seed_means = _seed_means(evaluated, seeds=seeds, delta_key="deployed_delta")
    values = np.asarray(list(seed_means.values()), dtype=float)
    accepted = [row for row in evaluated if bool(row["ranker_accepted"])]
    retained_seeds = {int(row["simulator_seed"]) for row in accepted}
    harmful_fraction = (
        float(
            np.mean(
                [
                    float(row["actual_group_normalized_delta"]) > 0.0
                    for row in accepted
                ]
            )
        )
        if accepted
        else 1.0
    )
    interval = _bootstrap_seed_means(
        seed_means,
        replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    mean_delta = float(np.mean(values))
    seed_std = float(np.std(values))
    worst_seed = float(np.max(values))
    gates = {
        "minimum_retained_groups": len(accepted)
        >= int(selection_rule["minimum_retained_groups_per_city"]),
        "minimum_retained_seed_fraction": len(retained_seeds) / max(len(seeds), 1)
        >= float(selection_rule["minimum_retained_seed_fraction"]),
        "mean_delta": mean_delta
        <= float(selection_rule["maximum_equal_seed_scenario_mean_delta"]),
        "bootstrap_upper": float(interval["ci95"][1])
        <= float(selection_rule["maximum_bootstrap_95pct_upper_mean_delta"]),
        "worst_seed": worst_seed
        <= float(selection_rule["maximum_worst_seed_mean_delta"]),
        "harmful_fraction": harmful_fraction
        <= float(selection_rule["maximum_harmful_group_fraction"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "key": (
            f"{records[0]['model_key']}::r{str(float(risk_multiplier)).replace('.', 'p')}"
            f"::t{str(float(minimum_context_trust)).replace('.', 'p')}"
        ),
        "model_key": str(records[0]["model_key"]),
        "model_family": str(records[0]["model_family"]),
        "risk_multiplier": float(risk_multiplier),
        "minimum_context_trust": float(minimum_context_trust),
        "decision_count": len(evaluated),
        "ranker_nonreference_count": sum(
            bool(row["selected_differs"]) for row in evaluated
        ),
        "retained_group_count": len(accepted),
        "retained_seed_count": len(retained_seeds),
        "retained_seed_fraction": len(retained_seeds) / max(len(seeds), 1),
        "harmful_group_fraction": harmful_fraction,
        "mean_delta": mean_delta,
        "seed_delta_std": seed_std,
        "worst_seed_delta": worst_seed,
        "robust_score": mean_delta + 0.5 * seed_std + max(worst_seed, 0.0),
        "bootstrap": interval,
        "seed_mean_delta": {
            str(seed): float(seed_means[seed]) for seed in sorted(seed_means)
        },
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def select_ranker_gate(
    oof_records: Sequence[Mapping[str, Any]],
    *,
    model_specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    rule = dict(selection["selection_rule"])
    bootstrap = dict(selection["bootstrap"])
    grid = []
    for spec in model_specs:
        records = [
            row for row in oof_records if str(row["model_key"]) == str(spec["key"])
        ]
        if not records:
            raise ValueError(f"ranker OOF records missing model {spec['key']!r}")
        for candidate in selection["gate_candidates"]:
            grid.append(
                summarize_gate_candidate(
                    records,
                    seeds=seeds,
                    risk_multiplier=float(candidate["risk_multiplier"]),
                    minimum_context_trust=float(
                        candidate["minimum_context_trust"]
                    ),
                    selection_rule=rule,
                    bootstrap=bootstrap,
                )
            )
    feasible = [row for row in grid if bool(row["feasible"])]
    selected = min(
        feasible,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_group_count"]),
            -float(row["risk_multiplier"]),
            -float(row["minimum_context_trust"]),
            str(row["key"]),
        ),
        default=None,
    )
    return {
        "decision": (
            "freeze_selected_direct_ranker"
            if selected is not None
            else "fallback_to_phase_pressure"
        ),
        "selected": selected,
        "feasible_candidate_count": len(feasible),
        "grid": grid,
    }


def _baseline_summaries(
    *,
    model_records: Sequence[Mapping[str, Any]],
    proposal_records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    proposal_by_group = {str(row["group_id"]): row for row in proposal_records}
    first_model = str(model_records[0]["model_key"])
    group_rows = [
        row for row in model_records if str(row["model_key"]) == first_model
    ]
    if set(proposal_by_group) != {str(row["group_id"]) for row in group_rows}:
        raise ValueError("immutable CFCMT comparator does not cover ranker groups")
    baseline_rows = []
    for row in group_rows:
        proposal = proposal_by_group[str(row["group_id"])]
        old_active = bool(proposal["short_proposed"])
        baseline_rows.append(
            {
                "simulator_seed": int(row["simulator_seed"]),
                "phase_delta": 0.0,
                "phase_active": False,
                "oracle_delta": float(row["oracle_group_normalized_delta"]),
                "oracle_active": float(row["oracle_group_normalized_delta"]) < 0.0,
                "old_cfcmt_delta": (
                    float(proposal["actual_group_normalized_delta"])
                    if old_active
                    else 0.0
                ),
                "old_cfcmt_active": old_active,
            }
        )
    summaries = {
        "phase_pressure_reference": _policy_summary(
            baseline_rows,
            seeds=seeds,
            delta_key="phase_delta",
            active_key="phase_active",
            bootstrap=bootstrap,
        ),
        "immutable_v60_hierarchical_cfcmt": _policy_summary(
            baseline_rows,
            seeds=seeds,
            delta_key="old_cfcmt_delta",
            active_key="old_cfcmt_active",
            bootstrap=bootstrap,
        ),
        "counterfactual_oracle_best_of_eight": _policy_summary(
            baseline_rows,
            seeds=seeds,
            delta_key="oracle_delta",
            active_key="oracle_active",
            bootstrap=bootstrap,
        ),
    }
    for model_key in sorted({str(row["model_key"]) for row in model_records}):
        rows = [row for row in model_records if str(row["model_key"]) == model_key]
        enriched = [
            {
                **dict(row),
                "ungated_active": bool(row["selected_differs"]),
            }
            for row in rows
        ]
        summary = _policy_summary(
            enriched,
            seeds=seeds,
            delta_key="actual_group_normalized_delta",
            active_key="ungated_active",
            bootstrap=bootstrap,
        )
        summary["selected_oracle_fraction"] = float(
            np.mean([bool(row["selected_is_oracle"]) for row in rows])
        )
        summary["mean_normalized_regret_to_oracle"] = float(
            np.mean([float(row["normalized_regret_to_oracle"]) for row in rows])
        )
        summaries[f"ungated::{model_key}"] = summary
    return summaries


def run_diagnostic(
    *,
    diagnostic_protocol_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    veto_protocol_path: Path,
    veto_result_path: Path,
    partition_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    protocol = _read_json(diagnostic_protocol_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    veto_protocol = _read_json(veto_protocol_path)
    veto_result = _read_json(veto_result_path)
    partition = _read_json(partition_path)
    parent = _read_json(proposal_parent_protocol_path)
    parent_audit = _read_json(hierarchical_freeze_audit_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    evidence_ok = (
        protocol.get("protocol") == PROTOCOL
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("gate", {}).get("passed") is True
        and veto_protocol.get("protocol") == VETO_PROTOCOL
        and veto_result.get("protocol") == VETO_RESULT_PROTOCOL
        and veto_result.get("status") == "PASS"
        and veto_result.get("decision") == VETO_REJECTION_DECISION
        and partition.get("protocol") == PARTITION_PROTOCOL
        and parent_audit.get("status") == "PASS"
        and _sha256(cache_protocol_path) == frozen.get("cache_protocol_sha256")
        and _sha256(cache_audit_path) == frozen.get("cache_audit_sha256")
        and _sha256(veto_protocol_path) == frozen.get("veto_protocol_sha256")
        and _sha256(veto_result_path) == frozen.get("veto_result_sha256")
        and _sha256(partition_path) == frozen.get("partition_sha256")
        and _sha256(proposal_parent_protocol_path)
        == frozen.get("proposal_parent_protocol_sha256")
        and _sha256(hierarchical_freeze_audit_path)
        == frozen.get("hierarchical_freeze_audit_sha256")
        and _sha256(manifest_path) == frozen.get("manifest_sha256")
    )
    if not evidence_ok:
        raise ValueError("waiting-aligned action-ranker evidence changed")

    training = dict(protocol["training"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    sealed = {
        int(value)
        for value in (
            list(training["sealed_confirmatory_seeds"])
            + list(training["sealed_prospective_seeds"])
        )
    }
    if set(seeds).intersection(sealed):
        raise ValueError("ranker development seeds overlap sealed validation seeds")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("waiting-aligned ranker scenario is not unique")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(int(cache_workers), 1),
    )
    _, frozen_model = _load_frozen_city_model(
        city=str(training["city"]),
        freeze_root=hierarchical_freeze_root,
        protocol=parent,
        freeze_audit=parent_audit,
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=str(training["city"]))
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(training["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = action_group_ids(contrast).astype(str)
    group_seed_set = {parse_action_group_seed(group) for group in np.unique(groups)}
    if group_seed_set != set(seeds):
        raise ValueError("ranker contrast does not cover the development seed set")
    proposal_records = frozen_proposal_records(
        contrast, frozen_model=frozen_model, scenarios=(scenario,)
    )

    model_specs = tuple(training["model_candidates"])
    workers = min(max(int(fold_workers), 1), 8, len(seeds))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        folds = list(
            pool.map(
                lambda seed: _fit_oof_fold(
                    heldout_seed=int(seed),
                    contrast=contrast,
                    model_specs=model_specs,
                    scenario=scenario,
                    expected_action_count=int(
                        training["expected_action_count_per_group"]
                    ),
                ),
                seeds,
            )
        )
    oof_records = [
        row
        for fold in folds
        for model_result in fold["model_results"]
        for row in model_result["records"]
    ]
    expected_groups = int(cache_audit["total_groups"])
    expected_records = expected_groups * len(model_specs)
    unique_record_keys = {
        (str(row["model_key"]), str(row["group_id"])) for row in oof_records
    }
    action_counts = {int(row["action_count"]) for row in oof_records}
    fit_contracts = [
        result["fit_diagnostics"]
        for fold in folds
        for result in fold["model_results"]
    ]
    fold_summaries = [
        {
            key: value
            for key, value in fold.items()
            if key != "model_results"
        }
        | {
            "model_results": [
                {
                    "model_key": result["model_key"],
                    "model_family": result["model_family"],
                    "fit_diagnostics": result["fit_diagnostics"],
                    "validation_record_count": len(result["records"]),
                }
                for result in fold["model_results"]
            ]
        }
        for fold in folds
    ]
    pairwise_fit_contract = all(
        fit.get("preference_protocol") == "all_action_antisymmetric_pairwise_v1"
        and fit.get("uses_context_features") is False
        for fit in fit_contracts
        if "preference_protocol" in fit
    )
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "development_seed_set_exact": group_seed_set == set(seeds),
        "sealed_seed_overlap_absent": not bool(set(seeds).intersection(sealed)),
        "fold_count_exact": len(folds) == len(seeds),
        "fold_seed_partition_disjoint": all(
            bool(fold["training_validation_disjoint"])
            and bool(fold["heldout_absent_from_training"])
            for fold in folds
        ),
        "validation_groups_exact": sum(
            int(fold["validation_group_count"]) for fold in folds
        )
        == expected_groups,
        "oof_record_count_exact": len(oof_records) == expected_records,
        "oof_group_model_keys_unique": len(unique_record_keys) == expected_records,
        "eight_actions_per_group": action_counts
        == {int(training["expected_action_count_per_group"])},
        "immutable_cfcmt_group_count_exact": len(proposal_records)
        == expected_groups,
        "pairwise_fit_contract_exact": pairwise_fit_contract,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]

    selection = select_ranker_gate(
        oof_records,
        model_specs=model_specs,
        seeds=seeds,
        selection=protocol["selection"],
    )
    baseline_summaries = _baseline_summaries(
        model_records=oof_records,
        proposal_records=proposal_records,
        seeds=seeds,
        bootstrap=protocol["selection"]["bootstrap"],
    )
    advance = bool(integrity["passed"] and selection["selected"] is not None)
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": AUTHORIZATION_DECISION if advance else REJECTION_DECISION,
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "veto_result_sha256": _sha256(veto_result_path),
        "bank_audit": bank_audit,
        "dataset_rows": int(contrast.size),
        "action_group_count": expected_groups,
        "model_count": len(model_specs),
        "fold_workers": workers,
        "folds": fold_summaries,
        "oof_record_count": len(oof_records),
        "oof_records": oof_records,
        "offline_comparators": baseline_summaries,
        "gate_selection": selection,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "selected_candidate": selection["selected"],
        },
        "adaptive_status": protocol["adaptive_status"],
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--veto-protocol", type=Path, required=True)
    parser.add_argument("--veto-result", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite action-ranker result: {args.out}")
    result = run_diagnostic(
        diagnostic_protocol_path=args.diagnostic_protocol,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        veto_protocol_path=args.veto_protocol,
        veto_result_path=args.veto_result,
        partition_path=args.partition,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    atomic_write_json(args.out, result)
    selected = result["advance_gate"]["selected_candidate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "groups": result["action_group_count"],
                "models": result["model_count"],
                "selected": selected.get("key") if selected else None,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
