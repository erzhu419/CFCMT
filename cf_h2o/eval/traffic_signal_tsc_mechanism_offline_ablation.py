"""Read-only cached action-group validation for TSC mechanism variants."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_MECHANISM_DEFINITIONS,
    CONTRAST_FEATURES_V3,
    _fit_family_v3,
    _group_adjusted_scores,
    merge_counterfactual_datasets_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _fit_target_models,
    _group_subset_v3,
    _merge_collection_shards_v3,
    _merge_scenario_seed_datasets_v3,
    _relabel_dataset_domain,
    _static_context_from_dataset,
    _strict_city_fold,
    _target_adaptation_split_v3,
    _validate_counterfactual_cache_dataset,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
    select_contextual_policy_from_domain_costs,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset, MechanismFitConfig
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    FusedCausalMechanismAdvantageModel,
    FusedPhysicalResidualCausalMechanismAdvantageModel,
)


DEFAULT_FAMILIES = (
    "causal_rigid_advantage",
    "causal_target_only",
    "cfcmt_physical_mechanism",
    "cfcmt_physical_fused",
)


@dataclass
class OfflineScreeningModels:
    """Minimum fitted-model contract needed by cached action-regret screening."""

    family_models: dict[str, Any]
    prior_spec: Any
    objective_modes: dict[str, str]
    diagnostics: dict[str, Any]


_OFFLINE_PROCESS_STATE: dict[str, Any] | None = None


def load_frozen_counterfactual_bank(
    cache_root: Path,
    manifest: TrafficSignalBenchmarkManifest,
    *,
    seeds: Sequence[int],
    collection_shards: int,
    workers: int,
    scenario_seeds: Mapping[str, Sequence[int]] | None = None,
) -> tuple[dict[str, MechanismDataset], dict[str, Any]]:
    """Load an exact cache by embedded identity without synthesizing misses."""

    scenarios = tuple(manifest.sumocfgs)
    requested_seeds = tuple(int(value) for value in seeds)
    if scenario_seeds is None:
        if not requested_seeds:
            raise ValueError("uniform counterfactual seed list cannot be empty")
        requested_by_scenario = {
            scenario: requested_seeds for scenario in scenarios
        }
    else:
        if requested_seeds:
            raise ValueError(
                "uniform seeds and scenario_seeds are mutually exclusive"
            )
        observed_scenarios = {str(name) for name in scenario_seeds}
        if observed_scenarios != set(scenarios):
            raise ValueError(
                "scenario_seeds must exactly cover the manifest: "
                f"missing={sorted(set(scenarios) - observed_scenarios)}, "
                f"extra={sorted(observed_scenarios - set(scenarios))}"
            )
        requested_by_scenario = {}
        for scenario in scenarios:
            scenario_values = tuple(
                int(value) for value in scenario_seeds[scenario]
            )
            if not scenario_values or len(scenario_values) != len(
                set(scenario_values)
            ):
                raise ValueError(
                    f"scenario seed list must be nonempty and unique: {scenario}"
                )
            requested_by_scenario[scenario] = scenario_values
    paths = sorted(Path(cache_root).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no counterfactual cache files under {cache_root}")
    with ThreadPoolExecutor(max_workers=min(max(int(workers), 1), len(paths), 32)) as pool:
        loaded = list(pool.map(load_mechanism_dataset, paths))

    shards: dict[tuple[str, int], dict[int, MechanismDataset]] = {}
    identity_sha = set()
    for path, dataset in zip(paths, loaded, strict=True):
        identity = dict(dataset.metadata.get("counterfactual_cache_identity", {}))
        _validate_counterfactual_cache_dataset(dataset, identity, path)
        scenario = str(identity.get("scenario", ""))
        seed = int(identity.get("seed", -1))
        shard = int(identity.get("collection_shard_index", -1))
        shard_count = int(identity.get("collection_shard_count", -1))
        if scenario not in requested_by_scenario or seed not in set(
            requested_by_scenario[scenario]
        ):
            continue
        if shard_count != int(collection_shards) or not 0 <= shard < shard_count:
            raise ValueError(f"unexpected shard identity at {path}: {identity}")
        key = (scenario, seed)
        if shard in shards.setdefault(key, {}):
            raise ValueError(f"duplicate cache shard for {key} index={shard}")
        shards[key][shard] = dataset
        identity_sha.add(
            json.dumps(identity, sort_keys=True, separators=(",", ":"))
        )

    expected_keys = {
        (scenario, seed)
        for scenario in scenarios
        for seed in requested_by_scenario[scenario]
    }
    if set(shards) != expected_keys:
        missing = sorted(expected_keys - set(shards))
        extra = sorted(set(shards) - expected_keys)
        raise ValueError(f"cache scenario/seed coverage mismatch: missing={missing}, extra={extra}")
    merged_by_scenario_seed = {}
    for scenario, seed in sorted(expected_keys):
        indexed = shards[(scenario, seed)]
        expected_indices = set(range(int(collection_shards)))
        if set(indexed) != expected_indices:
            raise ValueError(
                f"incomplete cache shards for {scenario}/seed={seed}: "
                f"found={sorted(indexed)}"
            )
        merged_by_scenario_seed[(scenario, seed)] = _merge_collection_shards_v3(
            scenario,
            seed,
            [indexed[index] for index in range(int(collection_shards))],
        )
    bank = {
        scenario: _merge_scenario_seed_datasets_v3(
            scenario,
            [
                merged_by_scenario_seed[(scenario, seed)]
                for seed in requested_by_scenario[scenario]
            ],
        )
        for scenario in scenarios
    }
    audit = {
        "protocol": (
            "embedded-identity-read-only-complete-shard-load-v1"
            if scenario_seeds is None
            else "embedded-identity-read-only-mixed-seed-shard-load-v2"
        ),
        "cache_root": str(Path(cache_root).resolve()),
        "file_count": len(paths),
        "used_file_count": len(expected_keys) * int(collection_shards),
        "scenario_count": len(scenarios),
        "collection_shards": int(collection_shards),
        "unique_identities": len(identity_sha),
        "rows_by_scenario": {name: int(dataset.size) for name, dataset in bank.items()},
        "groups_by_scenario": {
            name: len(set(str(value) for value in dataset.metadata["action_group_ids"]))
            for name, dataset in bank.items()
        },
    }
    if scenario_seeds is None:
        audit["seeds"] = list(requested_seeds)
    else:
        audit["seeds_by_scenario"] = {
            scenario: list(requested_by_scenario[scenario])
            for scenario in scenarios
        }
    return bank, audit


def evaluate_target_action_groups(
    dataset: MechanismDataset,
    *,
    models: Any,
    excluded_group_ids: Sequence[str],
    physical_blend_weights: Sequence[float] = (),
    physical_disagreement_caps: Sequence[float] = (),
) -> dict[str, Any]:
    groups = np.asarray(dataset.metadata["action_group_ids"], dtype=str)
    excluded = {str(value) for value in excluded_group_ids}
    evaluation_groups = set(groups.tolist()) - excluded
    evaluation = _group_subset_v3(
        dataset,
        selected_groups=evaluation_groups,
        metadata_updates={
            "target_data_role": "offline_ablation_evaluation",
            "excluded_target_group_ids": sorted(excluded),
        },
    )
    contrast = build_action_contrast_dataset(
        evaluation,
        reference_policy=models.prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual_raw = np.asarray(contrast.targets["interval_cost"], dtype=float)
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

    def summarize(
        score: np.ndarray,
        uncertainty: np.ndarray,
        trust: np.ndarray,
    ) -> dict[str, Any]:
        regrets = []
        selected_deltas = []
        optimal = 0
        for group_rows in grouped_rows:
            selected = int(group_rows[int(np.argmin(score[group_rows]))])
            values = actual_raw[group_rows]
            best = float(np.min(values))
            scale = action_group_range(values)
            regrets.append((float(actual_raw[selected]) - best) / scale)
            selected_deltas.append(float(actual_raw[selected]) / scale)
            optimal += int(
                np.isclose(float(actual_raw[selected]), best, rtol=0.0, atol=1e-12)
            )
        return {
            "group_count": len(regrets),
            "mean_normalized_action_regret": float(np.mean(regrets)),
            "p90_normalized_action_regret": float(np.quantile(regrets, 0.90)),
            "worst_normalized_action_regret": float(np.max(regrets)),
            "optimal_action_rate": float(optimal / max(len(regrets), 1)),
            "mean_selected_normalized_delta": float(np.mean(selected_deltas)),
            "mean_uncertainty": float(np.mean(uncertainty)),
            "mean_context_trust": float(np.mean(trust)),
        }

    rows = {}
    family_scores: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family, model in models.family_models.items():
        score, uncertainty, trust, _ = _group_adjusted_scores(
            contrast,
            model.predict(contrast),
            objective_mode=models.objective_modes[family],
        )
        family_scores[str(family)] = (score, uncertainty, trust)
        rows[str(family)] = summarize(score, uncertainty, trust)

    rigid_key = "causal_rigid_advantage"
    physical_key = "cfcmt_physical_mechanism"
    if (physical_blend_weights or physical_disagreement_caps) and not {
        rigid_key,
        physical_key,
    } <= set(family_scores):
        raise ValueError(
            "physical residual diagnostics require causal_rigid_advantage and "
            "cfcmt_physical_mechanism"
        )
    for weight in physical_blend_weights:
        value = float(weight)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"physical blend weight must be in [0, 1], got {value}")
        rigid_score, rigid_uncertainty, rigid_trust = family_scores[rigid_key]
        physical_score, physical_uncertainty, physical_trust = family_scores[physical_key]
        score = rigid_score + value * (physical_score - rigid_score)
        uncertainty = (
            (1.0 - value) * rigid_uncertainty + value * physical_uncertainty
        )
        trust = np.minimum(rigid_trust, physical_trust)
        suffix = str(value).replace(".", "p")
        rows[f"cfcmt_physical_blend_{suffix}"] = summarize(
            score,
            uncertainty,
            trust,
        )

    score_disagreement_ranges: list[float] = []
    if physical_disagreement_caps:
        rigid_score, rigid_uncertainty, rigid_trust = family_scores[rigid_key]
        physical_score, physical_uncertainty, physical_trust = family_scores[physical_key]
        score_delta = physical_score - rigid_score
        score_disagreement_ranges = [
            float(np.ptp(score_delta[group_rows])) for group_rows in grouped_rows
        ]
        for cap in physical_disagreement_caps:
            value = float(cap)
            if value <= 0.0:
                raise ValueError(
                    f"physical disagreement cap must be positive, got {value}"
                )
            row_gate = np.ones(contrast.size, dtype=float)
            group_gates = []
            for group_rows, disagreement in zip(
                grouped_rows,
                score_disagreement_ranges,
                strict=True,
            ):
                gate = min(1.0, value / max(float(disagreement), 1e-12))
                row_gate[group_rows] = gate
                group_gates.append(gate)
            score = rigid_score + row_gate * score_delta
            uncertainty = (
                (1.0 - row_gate) * rigid_uncertainty
                + row_gate * physical_uncertainty
            )
            trust = np.minimum(rigid_trust, physical_trust)
            suffix = str(value).replace(".", "p")
            summary = summarize(score, uncertainty, trust)
            summary.update(
                {
                    "mean_mechanism_gate": float(np.mean(group_gates)),
                    "p10_mechanism_gate": float(np.quantile(group_gates, 0.10)),
                    "full_mechanism_gate_fraction": float(
                        np.mean(np.asarray(group_gates) >= 1.0 - 1e-12)
                    ),
                }
            )
            rows[f"cfcmt_physical_disagreement_cap_{suffix}"] = summary
    return {
        "protocol": "disjoint-target-matched-action-group-offline-regret-v1",
        "evaluation_group_count": len(evaluation_groups),
        "excluded_group_count": len(excluded),
        "physical_blend_weights": [float(value) for value in physical_blend_weights],
        "physical_disagreement_caps": [
            float(value) for value in physical_disagreement_caps
        ],
        "physical_score_disagreement_range": (
            {
                "mean": float(np.mean(score_disagreement_ranges)),
                "p90": float(np.quantile(score_disagreement_ranges, 0.90)),
                "max": float(np.max(score_disagreement_ranges)),
            }
            if score_disagreement_ranges
            else None
        ),
        "families": rows,
    }


def fit_target_screening_models(
    bank: Mapping[str, MechanismDataset],
    target: str,
    *,
    fit_config: MechanismFitConfig,
    source_rule_costs: Mapping[str, Mapping[str, float]],
    source_rule_specs: Mapping[str, Any],
    model_families: Sequence[str],
    target_group_budget: int,
    target_adaptation_selection_seed: int,
    target_calibration_seeds: Sequence[int],
    target_calibration_fraction: float,
    source_selector_min_context_domains: int,
    scenario_city_groups: Mapping[str, str],
    target_adaptation_group_ids: Sequence[str] | None = None,
    target_static_context: Sequence[float] | None = None,
    prior_policy_override: str | None = None,
) -> OfflineScreeningModels:
    """Fit candidate predictors without deployment guard/regularizer LOO.

    The model's own nested source-city validation remains intact.  This path
    omits only the second, deployment-specific source-LOO layer used to tune a
    closed-loop guard and prior regularizer, neither of which is consumed by
    the offline action-ranking screen.
    """

    target_city_group = str(scenario_city_groups[target])
    source_names, heldout_city_names = _strict_city_fold(
        tuple(bank), target, scenario_city_groups
    )
    source_datasets = [
        _relabel_dataset_domain(bank[name], str(scenario_city_groups[name]))
        for name in source_names
    ]
    if target_adaptation_group_ids is None:
        split = _target_adaptation_split_v3(
            bank[target],
            group_budget=int(target_group_budget),
            selection_seed=int(target_adaptation_selection_seed),
            calibration_seeds=target_calibration_seeds,
            calibration_fraction=float(target_calibration_fraction),
        )
        target_adaptation_raw = split["adaptation"]
        target_adaptation_protocol = "frozen-adaptation-calibration-split-v1"
    else:
        requested_groups = tuple(
            sorted(str(value) for value in target_adaptation_group_ids)
        )
        if len(requested_groups) != len(set(requested_groups)):
            raise ValueError("explicit target adaptation groups must be unique")
        if len(requested_groups) != int(target_group_budget):
            raise ValueError(
                "explicit target adaptation group count must equal target budget"
            )
        available_groups = set(
            str(value)
            for value in bank[target].metadata.get("action_group_ids", ())
        )
        unknown_groups = set(requested_groups) - available_groups
        if unknown_groups:
            raise ValueError(
                f"unknown explicit target adaptation groups: {sorted(unknown_groups)}"
            )
        target_adaptation_raw = _group_subset_v3(
            bank[target],
            selected_groups=set(requested_groups),
            metadata_updates={
                "target_adaptation_group_budget": int(target_group_budget),
                "target_adaptation_groups_selected": len(requested_groups),
                "target_adaptation_selected_group_ids": list(requested_groups),
                "target_data_role": "explicit_model_adaptation",
                "target_role_groups": len(requested_groups),
            },
        )
        target_adaptation_protocol = "explicit-complete-action-group-list-v1"
    target_adaptation = _relabel_dataset_domain(
        target_adaptation_raw, target_city_group
    )
    if target_adaptation.size:
        source_datasets.append(target_adaptation)
    absolute = merge_counterfactual_datasets_v3(source_datasets)
    if target_static_context is None:
        target_context = _static_context_from_dataset(bank[target])
        target_context_protocol = "single-scenario-row-constant-static-context-v1"
    else:
        target_context = np.asarray(target_static_context, dtype=float)
        expected_shape = (len(bank[target].context_names),)
        if target_context.shape != expected_shape or not np.all(
            np.isfinite(target_context)
        ):
            raise ValueError(
                "explicit target static context must be finite and match context schema"
            )
        target_context_protocol = "explicit-label-free-city-static-context-v1"
    source_city_groups = sorted(
        {str(scenario_city_groups[name]) for name in source_names}
    )
    selector_costs = {
        city_group: {
            policy: float(
                np.mean(
                    [
                        source_rule_costs[name][policy]
                        for name in source_names
                        if str(scenario_city_groups[name]) == city_group
                    ]
                )
            )
            for policy in source_rule_specs
        }
        for city_group in source_city_groups
    }
    if target_city_group in selector_costs:
        raise AssertionError("offline screen target city entered source selector")
    selector_contexts = {
        city_group: np.mean(
            np.vstack(
                [
                    bank[name].context
                    for name in source_names
                    if str(scenario_city_groups[name]) == city_group
                ]
            ),
            axis=0,
        )
        for city_group in source_city_groups
    }
    if prior_policy_override is None:
        selected_prior_policy, prior_diagnostics = (
            select_contextual_policy_from_domain_costs(
                selector_costs,
                selector_contexts,
                target_context=target_context,
                policies=tuple(source_rule_specs),
                min_context_domains=int(source_selector_min_context_domains),
            )
        )
        prior_policy = str(selected_prior_policy)
    else:
        prior_policy = str(prior_policy_override)
        prior_diagnostics = {
            "protocol": "explicit-counterfactual-prior-override-v1",
            "selected": prior_policy,
            "source_domain_count": len(source_city_groups),
            "source_domains": list(source_city_groups),
            "contextual_selector_bypassed": True,
            "reason": "prior fixed by parent all-source target model",
        }
    if prior_policy not in source_rule_specs:
        raise ValueError(f"unknown explicit offline-screening prior: {prior_policy!r}")
    if prior_policy_override is not None:
        prior_diagnostics = {
            **dict(prior_diagnostics),
            "selected": prior_policy,
            "estimand_alignment_override": {
                "protocol": "counterfactual-rollout-anchor-alignment-v1",
                "policy": prior_policy,
                "uses_target_transition_labels": False,
            },
        }
    prior_spec = source_rule_specs[prior_policy]
    contrast = build_action_contrast_dataset(
        absolute,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    fitted_domains = set(str(value) for value in np.unique(contrast.domains))
    if int(target_group_budget) <= 0 and target_city_group in fitted_domains:
        raise AssertionError("zero-shot target city entered offline screening fit")

    family_models: dict[str, Any] = {}
    fit_diagnostics: dict[str, Any] = {}
    family_times: dict[str, float] = {}
    for family_index, family in enumerate(model_families, start=1):
        family_started = time.monotonic()
        reused_family = {
            "cfcmt_fused": "cfcmt_mechanism",
            "cfcmt_physical_fused": "cfcmt_physical_mechanism",
            "cfcmt_physical_latent_fused": "cfcmt_physical_latent_mechanism",
        }.get(str(family))
        source_model = family_models.get(reused_family) if reused_family else None
        if source_model is not None and family == "cfcmt_fused":
            model = FusedCausalMechanismAdvantageModel(
                CONTRAST_MECHANISM_DEFINITIONS,
                target_domain=target_city_group,
            )
            diagnostics = model.fit_from_fitted_source(contrast, source_model)
        elif source_model is not None and family in {
            "cfcmt_physical_fused",
            "cfcmt_physical_latent_fused",
        }:
            model = FusedPhysicalResidualCausalMechanismAdvantageModel(
                CONTRAST_MECHANISM_DEFINITIONS,
                target_domain=target_city_group,
                latent=family == "cfcmt_physical_latent_fused",
            )
            diagnostics = model.fit_from_fitted_source(contrast, source_model)
        else:
            model, diagnostics = _fit_family_v3(
                contrast,
                str(family),
                fit_config,
                target_domain=target_city_group,
            )
        family_models[str(family)] = model
        fit_diagnostics[str(family)] = diagnostics
        family_times[str(family)] = float(time.monotonic() - family_started)
        print(
            "CFCMT_OFFLINE_FAMILY_PROGRESS "
            f"target={target} {family_index}/{len(model_families)} "
            f"family={family} elapsed={family_times[str(family)]:.1f}s",
            flush=True,
        )
    return OfflineScreeningModels(
        family_models=family_models,
        prior_spec=prior_spec,
        objective_modes={str(family): "control_only" for family in model_families},
        diagnostics={
            "protocol": "model-internal-validation-only-no-deployment-guard-loo-v1",
            "target": target,
            "target_city_group": target_city_group,
            "heldout_city_scenarios": sorted(heldout_city_names),
            "source_scenarios": sorted(source_names),
            "source_domains": sorted(fitted_domains),
            "prior_policy": str(prior_policy),
            "prior_policy_override": (
                str(prior_policy_override)
                if prior_policy_override is not None
                else None
            ),
            "prior_selection": prior_diagnostics,
            "fit": fit_diagnostics,
            "family_elapsed_sec": family_times,
            "target_adaptation_groups": int(
                target_adaptation.metadata.get("target_role_groups", 0)
            ),
            "target_adaptation_protocol": target_adaptation_protocol,
            "target_static_context_protocol": target_context_protocol,
            "target_static_context": target_context.tolist(),
            "target_adaptation_group_ids": sorted(
                set(
                    str(value)
                    for value in target_adaptation.metadata.get(
                        "action_group_ids", ()
                    )
                )
            ),
        },
    )


def _fit_offline_target_result(
    target: str,
    *,
    bank: Mapping[str, MechanismDataset],
    city_groups: Mapping[str, str],
    source_rule_costs: Mapping[str, Mapping[str, float]],
    source_rule_specs: Mapping[str, Any],
    model_families: Sequence[str],
    physical_blend_weights: Sequence[float],
    physical_disagreement_caps: Sequence[float],
    target_group_budget: int,
    fit_parallel_workers: int,
    fit_protocol: str,
) -> dict[str, Any]:
    target_started = time.monotonic()
    fit_kwargs = {
        "fit_config": MechanismFitConfig(),
        "source_rule_costs": source_rule_costs,
        "source_rule_specs": source_rule_specs,
        "model_families": model_families,
        "target_group_budget": int(target_group_budget),
        "target_adaptation_selection_seed": 20260803,
        "target_calibration_seeds": (4047,),
        "target_calibration_fraction": 0.4,
        "source_selector_min_context_domains": 5,
        "scenario_city_groups": city_groups,
    }
    if fit_protocol == "screening":
        models = fit_target_screening_models(
            bank,
            target,
            **fit_kwargs,
        )
    elif fit_protocol == "deployment":
        models = _fit_target_models(
            bank,
            target,
            **fit_kwargs,
            min_target_calibration_groups=4,
            target_guard_selection_mode="strict_worst_domain",
            fit_parallel_workers=fit_parallel_workers,
        )
    else:
        raise ValueError(f"unknown offline fit protocol: {fit_protocol!r}")
    split = _target_adaptation_split_v3(
        bank[target],
        group_budget=int(target_group_budget),
        selection_seed=20260803,
        calibration_seeds=(4047,),
        calibration_fraction=0.4,
    )
    excluded_groups = split["selected"].metadata[
        "target_adaptation_selected_group_ids"
    ]
    return {
        "target": target,
        "city_group": city_groups[target],
        "target_group_budget": int(target_group_budget),
        "excluded_group_ids": sorted(str(value) for value in excluded_groups),
        "offline_evaluation": evaluate_target_action_groups(
            bank[target],
            models=models,
            excluded_group_ids=excluded_groups,
            physical_blend_weights=physical_blend_weights,
            physical_disagreement_caps=physical_disagreement_caps,
        ),
        "model_diagnostics": models.diagnostics,
        "elapsed_sec": float(time.monotonic() - target_started),
    }


def _offline_process_worker(target: str) -> dict[str, Any]:
    if _OFFLINE_PROCESS_STATE is None:
        raise RuntimeError("offline process worker state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_offline_target_result(
            str(target),
            **_OFFLINE_PROCESS_STATE,
        )


def run_offline_ablation(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    targets: Sequence[str],
    target_group_budget: int,
    model_families: Sequence[str],
    workers: int,
    fit_parallel_workers: int,
    target_workers: int,
    target_executor: str,
    fit_protocol: str,
    physical_blend_weights: Sequence[float],
    physical_disagreement_caps: Sequence[float],
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = load_traffic_signal_manifest(manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(2027, 3037, 4047),
        collection_shards=16,
        workers=workers,
    )
    unknown_targets = set(targets) - set(bank)
    if unknown_targets:
        raise ValueError(f"unknown targets: {sorted(unknown_targets)}")
    baseline = json.loads(Path(baseline_result).read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    if any(set(source_rule_specs) - set(source_rule_costs[name]) for name in bank):
        raise ValueError("baseline result is missing generalized source-rule costs")

    worker_state = {
        "bank": bank,
        "city_groups": manifest.city_groups,
        "source_rule_costs": source_rule_costs,
        "source_rule_specs": source_rule_specs,
        "model_families": tuple(model_families),
        "physical_blend_weights": tuple(float(value) for value in physical_blend_weights),
        "physical_disagreement_caps": tuple(
            float(value) for value in physical_disagreement_caps
        ),
        "target_group_budget": int(target_group_budget),
        "fit_parallel_workers": int(fit_parallel_workers),
        "fit_protocol": str(fit_protocol),
    }

    def fit_target(target: str) -> dict[str, Any]:
        return _fit_offline_target_result(target, **worker_state)

    actual_target_workers = min(max(int(target_workers), 1), len(targets))
    if actual_target_workers == 1:
        target_results = []
        for index, target in enumerate(targets, start=1):
            target_result = fit_target(target)
            target_results.append(target_result)
            print(
                "CFCMT_OFFLINE_PROGRESS "
                f"{index}/{len(targets)} target={target} "
                f"elapsed={target_result['elapsed_sec']:.1f}s",
                flush=True,
            )
    elif target_executor == "thread":
        with ThreadPoolExecutor(max_workers=actual_target_workers) as pool:
            futures = {target: pool.submit(fit_target, target) for target in targets}
            target_results = []
            for index, target in enumerate(targets, start=1):
                target_result = futures[target].result()
                target_results.append(target_result)
                print(
                    "CFCMT_OFFLINE_PROGRESS "
                    f"{index}/{len(targets)} target={target} "
                    f"elapsed={target_result['elapsed_sec']:.1f}s",
                    flush=True,
                )
    elif target_executor == "process":
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("process target executor requires Linux fork support")
        global _OFFLINE_PROCESS_STATE
        _OFFLINE_PROCESS_STATE = worker_state
        try:
            with ProcessPoolExecutor(
                max_workers=actual_target_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {
                    target: pool.submit(_offline_process_worker, target)
                    for target in targets
                }
                target_results = []
                for index, target in enumerate(targets, start=1):
                    target_result = futures[target].result()
                    target_results.append(target_result)
                    print(
                        "CFCMT_OFFLINE_PROGRESS "
                        f"{index}/{len(targets)} target={target} "
                        f"elapsed={target_result['elapsed_sec']:.1f}s",
                        flush=True,
                    )
        finally:
            _OFFLINE_PROCESS_STATE = None
    else:
        raise ValueError(f"unknown target executor: {target_executor!r}")
    return {
        "protocol": "tsc-physical-residual-progressive-offline-ablation-v2",
        "fit_protocol": str(fit_protocol),
        "cache_audit": cache_audit,
        "baseline_result": str(Path(baseline_result).resolve()),
        "manifest": manifest.to_dict(),
        "target_group_budget": int(target_group_budget),
        "model_families": list(model_families),
        "physical_blend_weights": [float(value) for value in physical_blend_weights],
        "physical_disagreement_caps": [
            float(value) for value in physical_disagreement_caps
        ],
        "parallelism": {
            "target_workers_requested": int(target_workers),
            "target_workers_actual": int(actual_target_workers),
            "target_executor": str(target_executor),
            "fit_parallel_workers_per_target": int(fit_parallel_workers),
        },
        "targets": target_results,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cf_h2o/config/traffic_signal_cross_city_v1.json"),
    )
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--target-group-budget", type=int, default=0)
    parser.add_argument("--model-families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fit-parallel-workers", type=int, default=7)
    parser.add_argument("--target-workers", type=int, default=1)
    parser.add_argument(
        "--target-executor",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument(
        "--fit-protocol",
        choices=("screening", "deployment"),
        default="screening",
    )
    parser.add_argument(
        "--physical-blend-weights",
        nargs="*",
        type=float,
        default=(),
    )
    parser.add_argument(
        "--physical-disagreement-caps",
        nargs="*",
        type=float,
        default=(),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_offline_ablation(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        targets=tuple(args.targets),
        target_group_budget=args.target_group_budget,
        model_families=tuple(args.model_families),
        workers=args.workers,
        fit_parallel_workers=args.fit_parallel_workers,
        target_workers=args.target_workers,
        target_executor=args.target_executor,
        fit_protocol=args.fit_protocol,
        physical_blend_weights=tuple(args.physical_blend_weights),
        physical_disagreement_caps=tuple(args.physical_disagreement_caps),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "targets": len(result["targets"])}))


if __name__ == "__main__":
    main()
