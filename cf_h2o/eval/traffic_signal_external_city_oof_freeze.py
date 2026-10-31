"""Freeze leakage-safe external-city CFCMT models from adaptation-only caches.

This stage is deliberately separated from held-out cache collection.  It loads
the immutable 18-network source evidence and the external adaptation seeds,
selects a city-level B100 pool without looking at counterfactual outcomes,
cross-fits the frozen v41 model families, applies the already frozen v41
selector, and serializes the five fitted models for later read-only evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    BASE_FAMILIES,
    COLLECTION_SHARDS,
    SOURCE_SEEDS,
    evaluate_anchored_candidates,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import CANDIDATE_KEYS
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    SELECTOR_SPECS,
    select_target_candidate,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    merge_counterfactual_datasets_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _coverage_first_group_order_v3,
    _group_seed_v3,
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import (
    MechanismDataset,
    MechanismFitConfig,
)


TARGET_GROUP_BUDGET = 100
ADAPTATION_SEEDS = (5057, 6067)
EVALUATION_SEEDS = (7079,)
FOLD_COUNT = 5
GROUPS_PER_FOLD = TARGET_GROUP_BUDGET // FOLD_COUNT
TRAINING_GROUPS_PER_FOLD = TARGET_GROUP_BUDGET - GROUPS_PER_FOLD
SELECTION_SEED = 20260803
EXTERNAL_COLLECTION_SHARDS = 16
FROZEN_SELECTOR_KEY = "scope_all_tau_0_rho_0p1_lambda_0"
MODEL_ARTIFACT_PROTOCOL = "cfcmt-external-city-cross-fitted-model-ensemble-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes(path, encoded)


def _with_metadata(
    dataset: MechanismDataset,
    updates: Mapping[str, Any],
) -> MechanismDataset:
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata={**dict(dataset.metadata), **dict(updates)},
    )


def _validate_protocol(
    protocol_spec_path: Path,
    development_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(Path(protocol_spec_path).read_text(encoding="utf-8"))
    if protocol.get("protocol") != (
        "tsc-v42r38-external-la-jinan-tls-safe-confirmation-v3"
    ):
        raise ValueError("external confirmation protocol changed")
    target = dict(protocol.get("target_protocol", {}))
    expected_scalars = {
        "target_group_budget_per_city": TARGET_GROUP_BUDGET,
        "cross_fit_folds": FOLD_COUNT,
        "groups_per_fold": GROUPS_PER_FOLD,
        "model_training_groups_per_fold": TRAINING_GROUPS_PER_FOLD,
        "model_ensemble_size": FOLD_COUNT,
        "selection_seed": SELECTION_SEED,
        "counterfactual_collection_shards_per_scenario_seed": (
            EXTERNAL_COLLECTION_SHARDS
        ),
    }
    for key, expected in expected_scalars.items():
        if int(target.get(key, -1)) != expected:
            raise ValueError(f"external target protocol changed: {key}")
    if tuple(int(value) for value in target.get("adaptation_seeds", ())) != (
        ADAPTATION_SEEDS
    ):
        raise ValueError("external adaptation seed partition changed")
    if tuple(int(value) for value in target.get("evaluation_seeds", ())) != (
        EVALUATION_SEEDS
    ):
        raise ValueError("external evaluation seed partition changed")
    if set(ADAPTATION_SEEDS) & set(EVALUATION_SEEDS):
        raise ValueError("external adaptation and evaluation seeds overlap")
    if tuple(str(value) for value in target.get("base_families", ())) != BASE_FAMILIES:
        raise ValueError("external base-family identities changed")
    if not bool(target.get("city_is_the_transfer_and_reporting_unit", False)):
        raise ValueError("external city reporting unit is not frozen")
    if not bool(target.get("equal_scenario_weight_within_city", False)):
        raise ValueError("external equal-scenario weighting is not frozen")
    if not bool(target.get("coverage_first_sampling_across_city_scenarios", False)):
        raise ValueError("external coverage-first sampling is not frozen")
    if not bool(target.get("scenario_balanced_target_group_quotas", False)):
        raise ValueError("external scenario-balanced B100 quotas are not frozen")
    if not bool(
        target.get("adaptation_seed_balanced_target_group_quotas", False)
    ):
        raise ValueError("external adaptation-seed-balanced B100 quotas are not frozen")
    if not bool(target.get("evaluation_cache_must_be_collected_after_model_freeze", False)):
        raise ValueError("external post-freeze evaluation-cache rule is missing")

    frozen = dict(protocol.get("development_gate", {}).get("frozen_selector", {}))
    if str(frozen.get("key")) != FROZEN_SELECTOR_KEY:
        raise ValueError("external frozen selector identity changed")
    expected_selector = SELECTOR_SPECS[FROZEN_SELECTOR_KEY]
    observed_selector = {
        "scope": str(frozen.get("scope")),
        "minimum_improvement": float(frozen.get("minimum_improvement", float("nan"))),
        "maximum_fold_regression": float(
            frozen.get("maximum_fold_regression", float("nan"))
        ),
        "standard_error_multiplier": float(
            frozen.get("standard_error_multiplier", float("nan"))
        ),
    }
    if observed_selector != expected_selector:
        raise ValueError("external frozen selector parameters changed")

    expected_audit_sha = str(
        protocol.get("development_gate", {}).get("v41_audit_sha256", "")
    )
    if _sha256(development_audit_path) != expected_audit_sha:
        raise ValueError("v41 development audit SHA-256 mismatch")
    audit = json.loads(Path(development_audit_path).read_text(encoding="utf-8"))
    gate = dict(audit.get("development_gate", {}))
    selected = dict(gate.get("global_selection", {}))
    if (
        audit.get("status") != "PASS"
        or not bool(audit.get("integrity_passed", False))
        or gate.get("decision")
        != "freeze_cross_fitted_anchored_selector_for_external_confirmation"
        or not bool(gate.get("global_gate", {}).get("passed", False))
        or selected.get("selected_selector") != FROZEN_SELECTOR_KEY
    ):
        raise ValueError("v41 development audit does not authorize external freeze")
    return protocol, audit


def _balanced_quotas(total: int, labels: Sequence[str]) -> dict[str, int]:
    names = tuple(str(value) for value in labels)
    if int(total) <= 0 or not names or len(names) != len(set(names)):
        raise ValueError("balanced quota labels must be unique and nonempty")
    base, remainder = divmod(int(total), len(names))
    return {
        name: base + int(index < remainder) for index, name in enumerate(names)
    }


def _select_city_adaptation_groups(
    bank: Mapping[str, MechanismDataset],
    *,
    city: str,
    scenarios: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    quotas = _balanced_quotas(TARGET_GROUP_BUDGET, scenarios)
    selected: list[str] = []
    scenario_rows: dict[str, Any] = {}
    seed_totals = {seed: 0 for seed in ADAPTATION_SEEDS}
    for scenario_index, scenario in enumerate(scenarios):
        dataset = bank[str(scenario)]
        groups = sorted(
            set(str(value) for value in dataset.metadata.get("action_group_ids", ()))
        )
        observed_seeds = {_group_seed_v3(group) for group in groups}
        if observed_seeds != set(ADAPTATION_SEEDS):
            raise ValueError(
                f"external adaptation cache seed coverage changed for {scenario}: "
                f"{sorted(observed_seeds)}"
            )
        quota = quotas[str(scenario)]
        per_seed = _balanced_quotas(
            quota,
            tuple(
                str(seed)
                for seed in (
                    ADAPTATION_SEEDS
                    if scenario_index % 2 == 0
                    else tuple(reversed(ADAPTATION_SEEDS))
                )
            ),
        )
        scenario_selected: list[str] = []
        for seed in ADAPTATION_SEEDS:
            candidates = [group for group in groups if _group_seed_v3(group) == seed]
            requested = per_seed[str(seed)]
            if len(candidates) < requested:
                raise ValueError(
                    f"insufficient groups for {scenario}/seed={seed}: "
                    f"{len(candidates)} < {requested}"
                )
            ordered = _coverage_first_group_order_v3(
                dataset,
                candidates,
                selection_seed=SELECTION_SEED,
                role=f"external_city_b100:{city}:{scenario}:seed{seed}",
            )
            chosen = ordered[:requested]
            scenario_selected.extend(chosen)
            seed_totals[seed] += len(chosen)
        if len(scenario_selected) != quota:
            raise AssertionError("external per-scenario B100 quota failed")
        selected.extend(scenario_selected)
        scenario_rows[str(scenario)] = {
            "available_group_count": len(groups),
            "selected_group_count": len(scenario_selected),
            "selected_group_count_by_seed": {
                str(seed): sum(_group_seed_v3(group) == seed for group in scenario_selected)
                for seed in ADAPTATION_SEEDS
            },
            "selected_group_ids": sorted(scenario_selected),
        }
    if len(selected) != TARGET_GROUP_BUDGET or len(set(selected)) != len(selected):
        raise AssertionError("external city B100 selection is incomplete or duplicated")
    if max(seed_totals.values()) - min(seed_totals.values()) > 1:
        raise ValueError("external city B100 pool is not adaptation-seed balanced")
    return tuple(sorted(selected)), {
        "protocol": "city-scenario-and-seed-balanced-coverage-first-b100-v1",
        "city": city,
        "selection_seed": SELECTION_SEED,
        "scenario_quotas": quotas,
        "selected_group_count_by_seed": {
            str(seed): seed_totals[seed] for seed in ADAPTATION_SEEDS
        },
        "scenarios": scenario_rows,
    }


def _scenario_from_group(group_id: str, scenarios: Sequence[str]) -> str:
    matches = [
        str(scenario)
        for scenario in scenarios
        if str(group_id).startswith(f"{scenario}:")
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot identify one external scenario for group {group_id}")
    return matches[0]


def assign_scenario_seed_stratified_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[str],
    expected_group_count: int = TARGET_GROUP_BUDGET,
    fold_count: int = FOLD_COUNT,
) -> dict[str, Any]:
    """Assign equal folds while balancing scenario and simulator-seed strata."""

    scenario_names = tuple(str(value) for value in scenarios)
    if len(records) != int(expected_group_count):
        raise ValueError("external fold assignment received the wrong group count")
    if int(expected_group_count) % int(fold_count):
        raise ValueError("external target groups must divide evenly across folds")
    normalized = []
    for row in records:
        group = str(row["group_id"])
        normalized.append(
            {
                "group_id": group,
                "scenario": _scenario_from_group(group, scenario_names),
                "simulator_seed": int(row["simulator_seed"]),
                "snapshot_time_sec": float(row["snapshot_time_sec"]),
                "tls_id": str(row["tls_id"]),
            }
        )
    group_ids = [row["group_id"] for row in normalized]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("external fold assignment contains duplicate groups")
    if {row["simulator_seed"] for row in normalized} != set(ADAPTATION_SEEDS):
        raise ValueError("external fold assignment seed coverage changed")

    scenario_counts = {
        scenario: sum(row["scenario"] == scenario for row in normalized)
        for scenario in scenario_names
    }
    fold_size = int(expected_group_count) // int(fold_count)
    base_targets = {
        scenario: scenario_counts[scenario] // int(fold_count)
        for scenario in scenario_names
    }
    scenario_fold_targets = {
        scenario: [base_targets[scenario]] * int(fold_count)
        for scenario in scenario_names
    }
    extra_load = [0] * int(fold_count)
    for scenario in scenario_names:
        extras = scenario_counts[scenario] % int(fold_count)
        ranked_folds = sorted(
            range(int(fold_count)),
            key=lambda fold: (
                extra_load[fold],
                _stable_digest(
                    f"{SELECTION_SEED}:scenario-fold-target:{scenario}:{fold}"
                ),
            ),
        )
        for fold in ranked_folds[:extras]:
            scenario_fold_targets[scenario][fold] += 1
            extra_load[fold] += 1
    if [sum(scenario_fold_targets[s][fold] for s in scenario_names) for fold in range(int(fold_count))] != [fold_size] * int(fold_count):
        raise ValueError("external scenario fold targets cannot form equal folds")

    assignments: dict[str, int] = {}
    scenario_seed_fold_counts: dict[str, dict[int, list[int]]] = {}
    for scenario in scenario_names:
        scenario_seed_rows = {
            seed: sorted(
                (
                    row
                    for row in normalized
                    if row["scenario"] == scenario
                    and row["simulator_seed"] == seed
                ),
                key=lambda row: (
                    row["snapshot_time_sec"],
                    row["tls_id"],
                    _stable_digest(row["group_id"]),
                ),
            )
            for seed in ADAPTATION_SEEDS
        }
        seed_bases = {
            seed: len(rows) // int(fold_count)
            for seed, rows in scenario_seed_rows.items()
        }
        remaining_column_extras = [
            scenario_fold_targets[scenario][fold] - sum(seed_bases.values())
            for fold in range(int(fold_count))
        ]
        if any(value < 0 or value > len(ADAPTATION_SEEDS) for value in remaining_column_extras):
            raise ValueError("external scenario/seed fold margins are infeasible")
        seed_targets = {
            seed: [seed_bases[seed]] * int(fold_count)
            for seed in ADAPTATION_SEEDS
        }
        for seed in ADAPTATION_SEEDS:
            extras = len(scenario_seed_rows[seed]) % int(fold_count)
            eligible = [
                fold
                for fold in range(int(fold_count))
                if remaining_column_extras[fold] > 0
            ]
            if len(eligible) < extras:
                raise ValueError("external scenario/seed extra margins are infeasible")
            ranked = sorted(
                eligible,
                key=lambda fold: (
                    -remaining_column_extras[fold],
                    _stable_digest(
                        f"{SELECTION_SEED}:scenario-seed-extra:"
                        f"{scenario}:{seed}:{fold}"
                    ),
                ),
            )
            for fold in ranked[:extras]:
                seed_targets[seed][fold] += 1
                remaining_column_extras[fold] -= 1
        if any(remaining_column_extras):
            raise ValueError("external scenario/seed fold margins were not filled")
        if [sum(seed_targets[seed][fold] for seed in ADAPTATION_SEEDS) for fold in range(int(fold_count))] != scenario_fold_targets[scenario]:
            raise AssertionError("external scenario/seed fold margins changed")

        scenario_seed_fold_counts[scenario] = seed_targets
        for seed in ADAPTATION_SEEDS:
            rows = scenario_seed_rows[seed]
            cursor = 0
            fold_order = sorted(
                range(int(fold_count)),
                key=lambda fold: _stable_digest(
                    f"{SELECTION_SEED}:scenario-seed-fold-order:"
                    f"{scenario}:{seed}:{fold}"
                ),
            )
            for fold in fold_order:
                count = seed_targets[seed][fold]
                for row in rows[cursor : cursor + count]:
                    assignments[row["group_id"]] = fold
                cursor += count
            if cursor != len(rows):
                raise AssertionError("external scenario/seed rows were not assigned")

    fold_group_counts = [
        sum(fold == index for fold in assignments.values())
        for index in range(int(fold_count))
    ]
    if fold_group_counts != [fold_size] * int(fold_count):
        raise ValueError(f"external fold balance failed: {fold_group_counts}")
    for scenario in scenario_names:
        values = scenario_fold_targets[scenario]
        if max(values) - min(values) > 1:
            raise ValueError("external scenario balance differs by more than one group")
        for seed in ADAPTATION_SEEDS:
            counts = scenario_seed_fold_counts[scenario][seed]
            if max(counts) - min(counts) > 1:
                raise ValueError(
                    f"external scenario/seed fold balance failed: {scenario}/{seed}"
                )
    return {
        "protocol": "scenario-seed-time-tls-balanced-five-fold-v1",
        "fold_count": int(fold_count),
        "group_count": int(expected_group_count),
        "groups_per_fold": fold_size,
        "assignments": assignments,
        "fold_group_counts": fold_group_counts,
        "scenario_group_counts": scenario_counts,
        "scenario_fold_group_counts": scenario_fold_targets,
        "scenario_seed_fold_group_counts": {
            scenario: {
                str(seed): counts
                for seed, counts in scenario_seed_fold_counts[scenario].items()
            }
            for scenario in scenario_names
        },
    }


def _merge_city_datasets(
    bank: Mapping[str, MechanismDataset],
    scenarios: Sequence[str],
    *,
    city: str,
) -> MechanismDataset:
    datasets = [bank[str(scenario)] for scenario in scenarios]
    merged = (
        datasets[0]
        if len(datasets) == 1
        else merge_counterfactual_datasets_v3(datasets)
    )
    return _with_metadata(
        merged,
        {
            "scenario": f"external_city_{city}",
            "external_city": city,
            "external_city_scenarios": list(scenarios),
            "external_city_merge_protocol": "target-only-equal-scenario-unit-v1",
        },
    )


def _fit_explicit_target_groups(
    *,
    bank: Mapping[str, MechanismDataset],
    target: str,
    target_group_ids: Sequence[str],
    city_groups: Mapping[str, str],
    source_rule_costs: Mapping[str, Mapping[str, float]],
    source_rule_specs: Mapping[str, Any],
    target_static_context: Sequence[float],
):
    return fit_target_screening_models(
        bank,
        target,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=BASE_FAMILIES,
        target_group_budget=len(target_group_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=city_groups,
        target_adaptation_group_ids=target_group_ids,
        target_static_context=target_static_context,
    )


def _equal_scenario_fold_evaluation(
    validation: MechanismDataset,
    *,
    models: Any,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    groups = set(str(value) for value in validation.metadata["action_group_ids"])
    scenario_results: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_groups = {
            group for group in groups if group.startswith(f"{scenario}:")
        }
        if not scenario_groups:
            raise ValueError(f"OOF fold has no groups from scenario {scenario}")
        subset = _group_subset_v3(
            validation,
            selected_groups=scenario_groups,
            metadata_updates={"target_data_role": "external_oof_scenario_validation"},
        )
        scenario_results[str(scenario)] = evaluate_anchored_candidates(
            subset,
            models=models,
            excluded_group_ids=(),
        )
    candidates = {}
    for candidate in CANDIDATE_KEYS:
        rows = [
            scenario_results[str(scenario)]["candidates"][candidate]
            for scenario in scenarios
        ]
        candidates[candidate] = {
            "equal_scenario_mean_normalized_action_regret": float(
                np.mean([row["mean_normalized_action_regret"] for row in rows])
            ),
            "equal_scenario_mean_alpha": float(
                np.mean([row["mean_alpha"] for row in rows])
            ),
            "scenario_group_counts": {
                str(scenario): int(
                    scenario_results[str(scenario)]["evaluation_group_count"]
                )
                for scenario in scenarios
            },
        }
    return {
        "protocol": "equal-scenario-weighted-external-oof-regret-v1",
        "scenarios": scenario_results,
        "candidates": candidates,
    }


def run_external_city_oof_freeze(
    *,
    source_cache_root: Path,
    source_baseline_result: Path,
    source_manifest_path: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    development_audit_path: Path,
    city: str,
    expected_source_cache_sha256: str,
    expected_source_baseline_sha256: str,
    expected_external_cache_sha256: str,
    expected_source_tree_sha256: str | None,
    workers: int,
    model_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol, development_audit = _validate_protocol(
        protocol_spec_path, development_audit_path
    )
    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("external freeze source-tree SHA-256 mismatch")

    source_evidence = dict(protocol["source_evidence"])
    if str(source_evidence["counterfactual_cache_sha256"]) != str(
        expected_source_cache_sha256
    ):
        raise ValueError("source cache expectation differs from frozen protocol")
    if str(source_evidence["source_rule_baseline_sha256"]) != str(
        expected_source_baseline_sha256
    ):
        raise ValueError("source baseline expectation differs from frozen protocol")
    if _sha256(source_manifest_path) != str(source_evidence["manifest_sha256"]):
        raise ValueError("source manifest SHA-256 mismatch")
    if _sha256(source_baseline_result) != str(expected_source_baseline_sha256):
        raise ValueError("source rule baseline SHA-256 mismatch")
    source_cache_sha, source_cache_files = aggregate_cache_sha256(source_cache_root)
    if (
        source_cache_sha != str(expected_source_cache_sha256)
        or source_cache_files != 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
    ):
        raise ValueError("source counterfactual cache identity changed")

    external_manifest_expected = str(
        protocol["counterfactual_cache"]["manifest_sha256"]
    )
    if _sha256(external_manifest_path) != external_manifest_expected:
        raise ValueError("external manifest SHA-256 mismatch")
    external_cache_sha, external_cache_files = aggregate_cache_sha256(
        external_cache_root
    )
    expected_external_cache_files = (
        sum(
            len(names)
            for names in protocol["target_protocol"]["external_city_scenarios"].values()
        )
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if (
        external_cache_sha != str(expected_external_cache_sha256)
        or external_cache_files != expected_external_cache_files
    ):
        raise ValueError("external adaptation cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    external_city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    if city not in external_city_scenarios:
        raise ValueError(f"unknown external confirmation city: {city}")
    scenarios = external_city_scenarios[city]
    if set(scenarios) != {
        name for name, group in external_manifest.city_groups.items() if group == city
    }:
        raise ValueError("external manifest city grouping changed")

    source_bank, source_cache_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    external_bank, external_cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        external_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    selected_group_ids, selection_audit = _select_city_adaptation_groups(
        external_bank, city=city, scenarios=scenarios
    )
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={"target_data_role": "external_city_b100_adaptation_pool"},
    )

    target_key = f"external_city_{city}"
    scenario_static_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_static_context = np.mean(
        np.vstack([scenario_static_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    fit_bank = dict(source_bank)
    fit_bank[target_key] = selected_dataset
    city_groups = dict(source_manifest.city_groups)
    city_groups[target_key] = city
    if any(group in {"los_angeles", "jinan"} for group in source_manifest.city_groups.values()):
        raise ValueError("an external confirmation city entered the source bank")

    records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=TARGET_GROUP_BUDGET,
    )
    fold_assignment = assign_scenario_seed_stratified_folds(
        records,
        scenarios=scenarios,
        expected_group_count=TARGET_GROUP_BUDGET,
        fold_count=FOLD_COUNT,
    )
    baseline = json.loads(source_baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    if not set(source_manifest.sumocfgs) <= set(source_rule_costs):
        raise ValueError("source baseline does not cover all source scenarios")
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}

    fold_results = []
    model_ensemble = []
    for fold_index in range(FOLD_COUNT):
        fold_started = time.monotonic()
        validation_ids = tuple(
            sorted(
                group
                for group, fold in fold_assignment["assignments"].items()
                if int(fold) == fold_index
            )
        )
        validation_set = set(validation_ids)
        training_ids = tuple(
            group for group in selected_group_ids if group not in validation_set
        )
        if (
            len(training_ids) != TRAINING_GROUPS_PER_FOLD
            or len(validation_ids) != GROUPS_PER_FOLD
        ):
            raise ValueError("external OOF fold train/validation counts changed")
        models = _fit_explicit_target_groups(
            bank=fit_bank,
            target=target_key,
            target_group_ids=training_ids,
            city_groups=city_groups,
            source_rule_costs=source_rule_costs,
            source_rule_specs=source_rule_specs,
            target_static_context=city_static_context,
        )
        diagnostics = models.diagnostics
        expected_fitted_domains = set(source_manifest.city_groups.values()) | {city}
        if (
            diagnostics.get("heldout_city_scenarios") != [target_key]
            or set(diagnostics.get("source_scenarios", ()))
            != set(source_manifest.sumocfgs)
            or set(diagnostics.get("source_domains", ()))
            != expected_fitted_domains
            or any(
                external_city in set(diagnostics.get("source_domains", ()))
                for external_city in set(external_city_scenarios) - {city}
            )
            or diagnostics.get("target_static_context_protocol")
            != "explicit-label-free-city-static-context-v1"
            or not np.allclose(
                np.asarray(diagnostics.get("target_static_context", ()), dtype=float),
                city_static_context,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("external target/source isolation contract failed")
        validation = _group_subset_v3(
            selected_dataset,
            selected_groups=validation_set,
            metadata_updates={
                "target_data_role": "external_city_cross_fit_oof_validation",
                "cross_fit_fold_index": fold_index,
            },
        )
        oof_evaluation = _equal_scenario_fold_evaluation(
            validation,
            models=models,
            scenarios=scenarios,
        )
        model_ensemble.append(models)
        fold_results.append(
            {
                "fold_index": fold_index,
                "training_group_ids": list(training_ids),
                "validation_group_ids": list(validation_ids),
                "model_diagnostics": diagnostics,
                "oof_evaluation": oof_evaluation,
                "elapsed_sec": float(time.monotonic() - fold_started),
            }
        )
        print(
            "CFCMT_EXTERNAL_OOF_PROGRESS "
            f"city={city} fold={fold_index + 1}/{FOLD_COUNT} "
            f"elapsed={fold_results[-1]['elapsed_sec']:.1f}s",
            flush=True,
        )
        gc.collect()

    oof_candidates = {}
    for candidate in CANDIDATE_KEYS:
        fold_regrets = [
            float(
                fold["oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_normalized_action_regret"
                ]
            )
            for fold in fold_results
        ]
        fold_alphas = [
            float(
                fold["oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_alpha"
                ]
            )
            for fold in fold_results
        ]
        oof_candidates[candidate] = {
            "fold_regrets": fold_regrets,
            "mean_regret": float(np.mean(fold_regrets)),
            "mean_alpha": float(np.mean(fold_alphas)),
        }
    target_selection = select_target_candidate(
        oof_candidates,
        **SELECTOR_SPECS[FROZEN_SELECTOR_KEY],
    )

    model_payload = {
        "protocol": MODEL_ARTIFACT_PROTOCOL,
        "city": city,
        "target_key": target_key,
        "scenarios": tuple(scenarios),
        "selected_candidate": target_selection["selected_candidate"],
        "selector_key": FROZEN_SELECTOR_KEY,
        "base_families": BASE_FAMILIES,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
        "model_ensemble": tuple(model_ensemble),
    }
    model_bytes = pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_bytes(model_out, model_bytes)
    model_sha256 = _sha256(model_out)
    loaded = pickle.loads(model_out.read_bytes())
    if (
        loaded.get("protocol") != MODEL_ARTIFACT_PROTOCOL
        or loaded.get("city") != city
        or loaded.get("selected_candidate")
        != target_selection["selected_candidate"]
        or len(loaded.get("model_ensemble", ())) != FOLD_COUNT
        or any(
            set(models.family_models) != set(BASE_FAMILIES)
            for models in loaded.get("model_ensemble", ())
        )
    ):
        raise ValueError("external frozen model artifact failed round-trip audit")

    return {
        "protocol": "tsc-v42r38-external-city-b100-oof-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "protocol_spec_path": str(Path(protocol_spec_path).resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "development_audit_path": str(Path(development_audit_path).resolve()),
        "development_audit_sha256": _sha256(development_audit_path),
        "development_decision": development_audit["development_gate"]["decision"],
        "source_manifest_path": str(Path(source_manifest_path).resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_cache_root": str(Path(source_cache_root).resolve()),
        "source_cache_aggregate_sha256": source_cache_sha,
        "source_cache_file_count": source_cache_files,
        "source_cache_audit": source_cache_audit,
        "source_baseline_result": str(Path(source_baseline_result).resolve()),
        "source_baseline_sha256": _sha256(source_baseline_result),
        "external_manifest_path": str(Path(external_manifest_path).resolve()),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "external_cache_root": str(Path(external_cache_root).resolve()),
        "external_cache_aggregate_sha256": external_cache_sha,
        "external_cache_file_count": external_cache_files,
        "external_cache_audit": external_cache_audit,
        "city": city,
        "target_key": target_key,
        "scenarios": list(scenarios),
        "source_scenarios": list(source_manifest.sumocfgs),
        "source_city_groups": sorted(set(source_manifest.city_groups.values())),
        "external_targets_must_not_source_each_other": True,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "adaptation_seeds": list(ADAPTATION_SEEDS),
        "evaluation_seeds_not_loaded": list(EVALUATION_SEEDS),
        "selection_audit": selection_audit,
        "selected_group_ids": list(selected_group_ids),
        "city_static_context": {
            "protocol": "equal-scenario-mean-label-free-static-context-v1",
            "context_names": list(selected_dataset.context_names),
            "scenario_values": {
                scenario: scenario_static_contexts[scenario].tolist()
                for scenario in scenarios
            },
            "city_value": city_static_context.tolist(),
        },
        "fold_assignment": fold_assignment,
        "fold_results": fold_results,
        "oof_candidates": oof_candidates,
        "frozen_selector_key": FROZEN_SELECTOR_KEY,
        "target_selection": target_selection,
        "selected_candidate": target_selection["selected_candidate"],
        "model_artifact": {
            "protocol": MODEL_ARTIFACT_PROTOCOL,
            "path": str(Path(model_out).resolve()),
            "sha256": model_sha256,
            "size_bytes": Path(model_out).stat().st_size,
            "round_trip_passed": True,
            "ensemble_size": FOLD_COUNT,
        },
        "freeze_gate": {
            "adaptation_only": True,
            "evaluation_seed_absent": True,
            "target_cities_isolated": True,
            "equal_scenario_oof_weighting": True,
            "model_artifact_round_trip": True,
            "passed": True,
        },
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-baseline-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-baseline-sha256", required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_external_city_oof_freeze(
        source_cache_root=args.source_cache_root,
        source_baseline_result=args.source_baseline_result,
        source_manifest_path=args.source_manifest,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        development_audit_path=args.development_audit,
        city=args.city,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
        expected_source_baseline_sha256=args.expected_source_baseline_sha256,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        workers=args.workers,
        model_out=args.model_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": payload["city"],
                "selected_candidate": payload["selected_candidate"],
                "model_sha256": payload["model_artifact"]["sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
