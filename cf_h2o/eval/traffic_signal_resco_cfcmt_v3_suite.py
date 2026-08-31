"""Parallel strict leave-one-city-out suite for action-contrast CFCMT."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import libsumo_version, load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import EXTENDED_SCENARIOS
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2_suite import (
    paired_hierarchical_bootstrap,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_CORE_FAMILY_V3,
    CFCMT_HIERARCHY_LAYERS_V3,
    CFCMT_HIERARCHY_PROTOCOL_V3,
    CFCMT_TARGET_SPECIALIST_FAMILY_V3,
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
    CONTRAST_FEATURES_V3,
    FittedContrastModelsV3,
    MODEL_FAMILIES_V3,
    OUTPUT_NAMES_V4,
    ONE_STEP_OUTPUT_NAMES_V4,
    POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5,
    normalize_rollout_prefix_horizons,
    rollout_prefix_target_name,
    STATE_SNAPSHOT_PROTOCOL_V3,
    STRICT_SAFETY_MONITORING_V3,
    calibrate_target_policy_selection_v3,
    collect_counterfactual_transitions_v3,
    evaluate_policy_v3,
    fit_contrast_models_v3,
    has_fitted_target_specialist_v3,
    merge_counterfactual_datasets_v3,
    select_target_capacity_layers_v3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    DEFAULT_ENV_ROOT,
    SUMO_EXECUTION_PROTOCOL,
    _scenario_sumocfg,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset, MechanismFitConfig
from cf_h2o.traffic_signal.action_contrast import (
    build_action_contrast_dataset,
    select_contextual_policy_from_domain_costs,
)
from cf_h2o.traffic_signal.action_ranker import ACTION_TARGET_SCALE_PROTOCOL
from cf_h2o.traffic_signal.generalized_pressure import (
    MAX_PRESSURE_SPEC,
    PHASE_PRESSURE_SPEC,
    SPILLBACK_PRESSURE_SPEC,
    PressurePolicySpec,
    generalized_pressure_grid,
    pressure_spec,
)
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
    save_mechanism_dataset,
)
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


LEGACY_COUNTERFACTUAL_CACHE_VERSION_V18 = (
    "v18-benchmark-aware-symmetric-safety-censor-least-covered-two-pass-20260808"
)
COUNTERFACTUAL_CACHE_VERSION = (
    "v19-policy-consistent-one-step-mechanisms-two-pass-20260809"
)
MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION = (
    "v20-policy-consistent-multihorizon-prefix-costs-two-pass-20260830"
)
SUPPORTED_COUNTERFACTUAL_CACHE_VERSIONS = frozenset(
    {
        LEGACY_COUNTERFACTUAL_CACHE_VERSION_V18,
        COUNTERFACTUAL_CACHE_VERSION,
        MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    }
)
SOURCE_RULE_CACHE_VERSION = (
    "v5-benchmark-collision-audited-no-teleport-full-input-hash-sumo122-20260808"
)
DEFAULT_CACHE_ROOT_V3 = Path("cf_h2o/results/traffic_signal_resco_cfcmt_v3_cache")
DEFAULT_CROSS_CITY_MANIFEST = Path("cf_h2o/config/traffic_signal_cross_city_v1.json")
TARGET_DEPLOYMENT_FIT_PROTOCOL_V3 = (
    "adaptation_only_model_disjoint_calibration_guard_v1"
)
TARGET_GROUP_SELECTION_PROTOCOL_V3 = (
    "coverage-first-per-tls-disjoint-seed-role-v1"
)

DEFAULT_METRICS_V3 = (
    "mean_queue",
    "p90_queue",
    "mean_queue_per_lane",
    "p90_queue_per_lane",
    "mean_queue_proxy",
    "mean_queue_proxy_per_lane",
    "halted_vehicle_hours",
    "active_vehicle_hours",
    "pending_vehicle_hours",
    "system_vehicle_hours",
    "mean_active_vehicles",
    "mean_active_vehicles_per_controlled_lane",
    "mean_system_vehicles",
    "mean_system_vehicles_per_controlled_lane",
    "active_vehicles_at_horizon",
    "pending_vehicles_at_horizon",
    "mean_tripinfo_duration",
    "mean_tripinfo_waiting_time",
    "mean_tripinfo_time_loss",
    "throughput_ratio",
    "departure_service_ratio",
    "completion_ratio",
    "starting_teleports",
    "ending_teleports",
    "collision_events",
    "collision_event_steps",
    "collision_incidents",
    "emergency_stops",
    "teleport_rate",
    "collision_rate",
    "collision_incident_rate",
    "emergency_stop_rate",
)


def _resolve_scenario_sumocfgs(
    *,
    env_root: Path,
    scenarios: Sequence[str],
    scenario_sumocfgs: Mapping[str, Path] | None,
) -> dict[str, Path]:
    provided = {str(name): Path(path).resolve() for name, path in (scenario_sumocfgs or {}).items()}
    resolved = {}
    for scenario in scenarios:
        path = provided.get(str(scenario))
        if path is None:
            path = _scenario_sumocfg(Path(env_root), str(scenario)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing SUMO configuration for {scenario}: {path}")
        resolved[str(scenario)] = path
    return resolved


def _default_city_group(scenario: str) -> str:
    scenario = str(scenario)
    if scenario.startswith("cologne"):
        return "cologne"
    if scenario.startswith("ingolstadt"):
        return "ingolstadt"
    if scenario in {"grid4x4", "arterial4x4"}:
        return "resco_synthetic"
    return scenario


def _resolve_city_groups(
    scenarios: Sequence[str],
    scenario_city_groups: Mapping[str, str] | None,
) -> dict[str, str]:
    provided = {str(name): str(group) for name, group in (scenario_city_groups or {}).items()}
    result = {
        str(scenario): provided.get(str(scenario), _default_city_group(str(scenario)))
        for scenario in scenarios
    }
    if any(not group.strip() for group in result.values()):
        raise ValueError("every scenario requires a non-empty city group")
    if len(set(result.values())) < 3:
        raise ValueError("strict leave-one-city-out evaluation requires at least three city groups")
    return result


def _strict_city_fold(
    scenarios: Sequence[str],
    target: str,
    scenario_city_groups: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    target_group = str(scenario_city_groups[target])
    source = [name for name in scenarios if str(scenario_city_groups[name]) != target_group]
    heldout = [name for name in scenarios if str(scenario_city_groups[name]) == target_group]
    if target not in heldout or set(source) & set(heldout):
        raise AssertionError("invalid strict leave-one-city-out fold")
    return source, heldout

PRIMARY_POLICY_PRIORITY_V3 = (
    "cfcmt_fused_contrast_regularized",
    "cfcmt_fused_contrast_guard",
    "cfcmt_fused_contrast_mpc",
    "cfcmt_mechanism_contrast_regularized",
    "cfcmt_mechanism_contrast_guard",
    "cfcmt_mechanism_contrast_mpc",
    "cfcmt_boosted_contrast_regularized",
    "cfcmt_boosted_contrast_guard",
    "cfcmt_boosted_contrast_mpc",
    "causal_target_adapter_contrast_regularized",
    "causal_target_adapter_contrast_guard",
    "causal_target_adapter_contrast_mpc",
    "causal_target_only_contrast_regularized",
    "causal_target_only_contrast_guard",
    "causal_target_only_contrast_mpc",
    "dense_balanced_advantage_contrast_regularized",
    "dense_balanced_advantage_contrast_guard",
    "dense_balanced_advantage_contrast_mpc",
    "causal_rigid_advantage_contrast_regularized",
    "causal_rigid_advantage_contrast_guard",
    "causal_rigid_advantage_contrast_mpc",
    "causal_balanced_advantage_contrast_regularized",
    "causal_balanced_advantage_contrast_guard",
    "causal_balanced_advantage_contrast_mpc",
    "causal_core_advantage_contrast_regularized",
    "causal_core_advantage_contrast_guard",
    "causal_core_advantage_contrast_mpc",
    "causal_advantage_contrast_regularized",
    "causal_advantage_contrast_guard",
    "causal_advantage_contrast_mpc",
    "causal_ranker_contrast_regularized",
    "causal_ranker_contrast_guard",
    "causal_ranker_contrast_mpc",
    "cfcmt_contrast_regularized",
    "cfcmt_contrast_guard",
    "cfcmt_contrast_mpc",
)


def _map_fresh_sumo_processes(worker: Any, jobs: Sequence[Mapping[str, Any]], workers: int) -> list[Any]:
    """Run every libsumo rollout in a newly spawned process.

    libsumo does not fully reset all stochastic state across repeated
    ``close/start`` cycles in one Python process, even with the same SUMO seed.
    Recycling after every task is therefore part of the paired-seed protocol.
    """

    if int(workers) <= 1:
        # Even the serial path needs process isolation between jobs.
        process_count = 1
    else:
        process_count = min(int(workers), len(jobs))
    if not jobs:
        return []
    context = mp.get_context("spawn")
    with context.Pool(processes=process_count, maxtasksperchild=1) as pool:
        results = []
        started = time.monotonic()
        report_every = max(len(jobs) // 20, 1)
        for completed, result in enumerate(
            pool.imap_unordered(worker, jobs, chunksize=1), start=1
        ):
            results.append(result)
            if completed == 1 or completed == len(jobs) or completed % report_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed / elapsed
                eta = (len(jobs) - completed) / max(rate, 1e-9)
                print(
                    f"CFCMT_PROGRESS {completed}/{len(jobs)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s rate={rate:.3f}/s",
                    flush=True,
                )
        return results


DEFAULT_POLICIES_V3 = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
    "selected_source_prior",
    "simulator_contrast_mpc",
    "dense_contrast_mpc",
    "sparse_contrast_mpc",
    "cfcmt_contrast_mpc",
    "dense_boosted_contrast_mpc",
    "causal_boosted_contrast_mpc",
    "cfcmt_boosted_contrast_mpc",
    "cfcmt_mechanism_contrast_mpc",
    "dense_ranker_contrast_mpc",
    "causal_ranker_contrast_mpc",
    "simulator_contrast_regularized",
    "dense_contrast_regularized",
    "sparse_contrast_regularized",
    "cfcmt_contrast_regularized",
    "dense_boosted_contrast_regularized",
    "causal_boosted_contrast_regularized",
    "cfcmt_boosted_contrast_regularized",
    "cfcmt_mechanism_contrast_regularized",
    "dense_ranker_contrast_regularized",
    "causal_ranker_contrast_regularized",
    "simulator_contrast_guard",
    "dense_contrast_guard",
    "sparse_contrast_guard",
    "cfcmt_contrast_guard",
    "dense_boosted_contrast_guard",
    "causal_boosted_contrast_guard",
    "cfcmt_boosted_contrast_guard",
    "cfcmt_mechanism_contrast_guard",
    "dense_ranker_contrast_guard",
    "causal_ranker_contrast_guard",
)


def _collect_worker(payload: Mapping[str, Any]) -> tuple[str, int, int, MechanismDataset]:
    scenario = str(payload["scenario"])
    seed = int(payload["seed"])
    shard_index = int(payload["collection_shard_index"])
    cache_identity = dict(payload["cache_identity"])
    cache_path = Path(payload["cache_path"]) if payload.get("cache_path") else None
    if cache_path is not None and cache_path.exists():
        dataset = load_mechanism_dataset(cache_path)
        _validate_counterfactual_cache_dataset(dataset, cache_identity, cache_path)
        return scenario, seed, shard_index, dataset
    try:
        dataset = collect_counterfactual_transitions_v3(
            sumo_api=_load_libsumo(),
            sumocfg=Path(payload["sumocfg"]),
            scenario=scenario,
            duration_sec=float(payload["duration_sec"]),
            control_interval_sec=int(payload["control_interval_sec"]),
            warmup_sec=float(payload["warmup_sec"]),
            seed=int(payload["seed"]),
            max_focal_tls=int(payload["max_focal_tls"]),
            counterfactual_horizon_intervals=int(
                payload["counterfactual_horizon_intervals"]
            ),
            counterfactual_cost_mode=str(
                payload.get("counterfactual_cost_mode", "system_vehicle_load")
            ),
            behavior_policy=str(payload["behavior_policy"]),
            collection_shard_index=shard_index,
            collection_shard_count=int(payload["collection_shard_count"]),
            rollout_prefix_horizons_sec=payload.get(
                "rollout_prefix_horizons_sec"
            ),
        )
    except Exception as error:
        raise RuntimeError(
            "counterfactual collection failed for "
            f"scenario={scenario}, seed={seed}, shard={shard_index}/"
            f"{int(payload['collection_shard_count'])}: {error}"
        ) from error
    dataset = replace(
        dataset,
        metadata={
            **dict(dataset.metadata),
            "counterfactual_cache_identity": cache_identity,
        },
    )
    _validate_counterfactual_cache_dataset(dataset, cache_identity, cache_path)
    if cache_path is not None:
        save_mechanism_dataset(cache_path, dataset)
    return scenario, seed, shard_index, dataset


def _validate_counterfactual_cache_dataset(
    dataset: MechanismDataset,
    expected_identity: Mapping[str, Any],
    cache_path: Path | None,
) -> None:
    location = str(cache_path) if cache_path is not None else "<uncached dataset>"
    actual_identity = dataset.metadata.get("counterfactual_cache_identity")
    if actual_identity != dict(expected_identity):
        raise ValueError(f"counterfactual cache identity mismatch at {location}")
    output_names = tuple(
        str(name)
        for name in dataset.metadata.get(
            "mechanism_output_names",
            tuple(dataset.targets),
        )
    )
    if (
        not output_names
        or set(output_names) != set(dataset.targets)
        or set(output_names) != set(dataset.priors)
    ):
        raise ValueError(f"counterfactual cache output schema mismatch at {location}")
    cache_version = str(expected_identity.get("version", ""))
    if cache_version in {
        COUNTERFACTUAL_CACHE_VERSION,
        MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    }:
        expected_rollout_horizon = int(expected_identity["control_interval_sec"]) * max(
            int(expected_identity["counterfactual_horizon_intervals"]),
            1,
        )
        estimands = dict(dataset.metadata.get("mechanism_estimands", {}))
        one_step_estimand = dict(estimands.get("one_step_local_state", {}))
        cost_mode = str(
            expected_identity.get(
                "counterfactual_cost_mode", "system_vehicle_load"
            )
        )
        expected_estimand = (
            WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5
            if cost_mode == "halted_queue"
            else POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4
        )
        expected_prefix_horizons = normalize_rollout_prefix_horizons(
            expected_identity.get("rollout_prefix_horizons_sec", ()),
            rollout_horizon_sec=expected_rollout_horizon,
        )
        expected_prefix_names = tuple(
            rollout_prefix_target_name(value)
            for value in expected_prefix_horizons
        )
        expected_output_names = (*OUTPUT_NAMES_V4, *expected_prefix_names)
        current_estimand_failed = (
            output_names != expected_output_names
            or dataset.metadata.get("counterfactual_estimand")
            != expected_estimand
            or str(
                dataset.metadata.get(
                    "counterfactual_cost_mode", "system_vehicle_load"
                )
            )
            != cost_mode
            or int(dataset.metadata.get("local_mechanism_horizon_sec", -1))
            != int(expected_identity["control_interval_sec"])
            or int(dataset.metadata.get("rollout_value_horizon_sec", -1))
            != expected_rollout_horizon
            or tuple(one_step_estimand.get("output_names", ()))
            != ONE_STEP_OUTPUT_NAMES_V4
            or int(one_step_estimand.get("horizon_sec", -1))
            != int(expected_identity["control_interval_sec"])
            or int(one_step_estimand.get("analytic_prior_horizon_sec", -1))
            != int(expected_identity["control_interval_sec"])
            or tuple(
                int(value)
                for value in dataset.metadata.get(
                    "rollout_prefix_horizons_sec", ()
                )
            )
            != expected_prefix_horizons
        )
        if cache_version == COUNTERFACTUAL_CACHE_VERSION:
            current_estimand_failed = (
                current_estimand_failed or bool(expected_prefix_horizons)
            )
        else:
            prefix_estimand = dict(
                estimands.get("rollout_prefix_action_values", {})
            )
            current_estimand_failed = current_estimand_failed or (
                not expected_prefix_horizons
                or tuple(prefix_estimand.get("output_names", ()))
                != expected_prefix_names
                or tuple(
                    int(value)
                    for value in prefix_estimand.get("horizons_sec", ())
                )
                != expected_prefix_horizons
                or prefix_estimand.get("aggregation")
                != "arithmetic_mean_of_one_second_cost_samples_from_branch_start"
                or prefix_estimand.get("policy")
                != "candidate_action_then_phase_pressure"
                or prefix_estimand.get("cost_mode") != cost_mode
            )
        if current_estimand_failed:
            raise ValueError(
                f"counterfactual cache policy-consistent estimand mismatch at {location}"
            )
        if expected_rollout_horizon in expected_prefix_horizons:
            full_prefix = np.asarray(
                dataset.targets[
                    rollout_prefix_target_name(expected_rollout_horizon)
                ],
                dtype=float,
            )
            if not np.allclose(
                full_prefix,
                np.asarray(dataset.targets["interval_cost"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"counterfactual full-horizon prefix mismatch at {location}"
                )
    if dataset.metadata.get("state_snapshot_protocol") != STATE_SNAPSHOT_PROTOCOL_V3:
        raise ValueError(f"counterfactual cache snapshot protocol mismatch at {location}")
    if dataset.metadata.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
        raise ValueError(f"counterfactual cache SUMO execution protocol mismatch at {location}")
    if dataset.metadata.get("strict_safety_monitoring") != STRICT_SAFETY_MONITORING_V3:
        raise ValueError(f"counterfactual cache safety monitoring protocol mismatch at {location}")
    behavior_safety = dict(dataset.metadata.get("behavior_safety_audit", {}))
    if (
        not behavior_safety.get("no_teleport_passed", False)
        or int(behavior_safety.get("starting_teleports", -1)) != 0
        or int(behavior_safety.get("ending_teleports", -1)) != 0
        or any(
            int(behavior_safety.get(name, -1)) < 0
            for name in (
                "raw_collision_events",
                "collision_event_steps",
                "unique_collision_incidents",
            )
        )
    ):
        raise ValueError(f"counterfactual behavior safety audit failed at {location}")
    counterfactual_safety = dict(
        dataset.metadata.get("counterfactual_safety_audit", {})
    )
    retained_groups = len(
        {str(value) for value in dataset.metadata.get("action_group_ids", ())}
    )
    safety_counts = {
        name: int(counterfactual_safety.get(name, -1))
        for name in (
            "candidate_groups",
            "retained_groups",
            "censored_groups",
            "candidate_branches",
            "attempted_branches",
            "retained_branches",
            "censored_branches",
            "observed_unsafe_branches",
            "starting_teleports",
            "ending_teleports",
        )
    }
    censor_accounting_failed = (
        counterfactual_safety.get("protocol")
        != COUNTERFACTUAL_SAFETY_PROTOCOL_V3
        or not counterfactual_safety.get("no_teleport_passed", False)
        or not counterfactual_safety.get(
            "symmetric_group_censoring_passed", False
        )
        or any(value < 0 for value in safety_counts.values())
        or safety_counts["starting_teleports"] != 0
        or safety_counts["ending_teleports"] != 0
        or safety_counts["candidate_groups"]
        != safety_counts["retained_groups"] + safety_counts["censored_groups"]
        or safety_counts["candidate_branches"]
        != safety_counts["retained_branches"] + safety_counts["censored_branches"]
        or safety_counts["retained_groups"] != retained_groups
        or safety_counts["retained_branches"] != dataset.size
        or len(counterfactual_safety.get("censored_group_ids", ()))
        != safety_counts["censored_groups"]
        or len(
            set(counterfactual_safety.get("collision_incident_signatures", ()))
        )
        != int(
            counterfactual_safety.get(
                "unique_collision_incidents_observed", -1
            )
        )
        or not math.isclose(
            float(counterfactual_safety.get("censored_group_fraction", -1.0)),
            safety_counts["censored_groups"]
            / max(safety_counts["candidate_groups"], 1),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if censor_accounting_failed:
        raise ValueError(
            f"counterfactual symmetric safety audit failed at {location}: "
            f"{counterfactual_safety}"
        )
    focal_protocol = str(dataset.metadata.get("focal_selection_protocol", ""))
    if focal_protocol != str(STATE_SNAPSHOT_PROTOCOL_V3["focal_selection"]):
        raise ValueError(f"counterfactual cache focal selection protocol mismatch at {location}")
    controllable_tls = {
        str(tls_id) for tls_id in dataset.metadata.get("controllable_tls_ids", ())
    }
    focal_counts = {
        str(tls_id): int(value)
        for tls_id, value in dict(
            dataset.metadata.get("focal_selection_count_by_tls", {})
        ).items()
    }
    if set(focal_counts) != controllable_tls or any(
        value < 0 for value in focal_counts.values()
    ):
        raise ValueError(f"counterfactual cache focal selection counts are invalid at {location}")
    if int(dataset.metadata.get("focal_selection_min_count", -1)) != min(
        focal_counts.values(), default=0
    ) or int(dataset.metadata.get("focal_selection_max_count", -1)) != max(
        focal_counts.values(), default=0
    ):
        raise ValueError(f"counterfactual cache focal selection bounds are invalid at {location}")
    behavior_trace = str(dataset.metadata.get("behavior_trace_sha256", ""))
    if len(behavior_trace) != 64 or any(character not in "0123456789abcdef" for character in behavior_trace):
        raise ValueError(f"counterfactual cache behavior trace is invalid at {location}")
    replay = dataset.metadata.get("counterfactual_replay_audit", {})
    tolerance = float(replay.get("absolute_tolerance", 1e-8))
    replay_checks = int(replay.get("checks", -1))
    replay_not_applicable = bool(replay.get("not_applicable", False))
    replay_failed = (
        not replay.get("passed", False)
        or float(replay.get("max_abs_difference", float("inf"))) > tolerance
        or replay_not_applicable != (replay_checks == 0)
    )
    if dataset.size > 0:
        replay_failed = replay_failed or replay_checks < 1
    else:
        expected_trailing_empty = (
            replay_not_applicable
            and replay_checks == 0
            and int(dataset.metadata.get("counterfactual_branches", -1)) == 0
            and int(dataset.metadata.get("eligible_collection_opportunities", -1))
            <= int(expected_identity["collection_shard_index"])
        )
        expected_safety_censored_empty = (
            safety_counts["candidate_groups"] > 0
            and safety_counts["retained_groups"] == 0
            and safety_counts["censored_groups"]
            == safety_counts["candidate_groups"]
            and replay_checks >= 0
        )
        expected_empty = expected_trailing_empty or expected_safety_censored_empty
        replay_failed = replay_failed or not expected_empty
    if replay_failed:
        raise ValueError(f"counterfactual cache replay audit failed at {location}: {replay}")
    for values in (dataset.features, dataset.context, *dataset.priors.values(), *dataset.targets.values()):
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError(f"counterfactual cache contains non-finite values at {location}")
    scenario = str(expected_identity["scenario"])
    expected_domains = {scenario} if dataset.size else set()
    if set(np.asarray(dataset.domains, dtype=str).tolist()) != expected_domains:
        raise ValueError(f"counterfactual cache domain mismatch at {location}")
    aligned = {}
    for key in ("action_group_ids", "row_tls", "row_times", "candidate_states"):
        values = list(dataset.metadata.get(key, ()))
        if len(values) != dataset.size:
            raise ValueError(f"counterfactual cache row metadata is misaligned at {location}: {key}")
        aligned[key] = values
    action_keys = list(zip(aligned["action_group_ids"], aligned["candidate_states"]))
    if len(set(action_keys)) != len(action_keys):
        raise ValueError(f"counterfactual cache has duplicate action rows at {location}")


def _counterfactual_cache_identity(
    *,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    behavior_policy: str,
    collection_shard_index: int,
    collection_shard_count: int,
    counterfactual_cost_mode: str = "system_vehicle_load",
    rollout_prefix_horizons_sec: Sequence[int] | None = None,
) -> dict[str, Any]:
    rollout_horizon_sec = int(control_interval_sec) * max(
        int(counterfactual_horizon_intervals), 1
    )
    prefix_horizons = normalize_rollout_prefix_horizons(
        rollout_prefix_horizons_sec,
        rollout_horizon_sec=rollout_horizon_sec,
    )
    identity = {
        "version": (
            MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
            if prefix_horizons
            else COUNTERFACTUAL_CACHE_VERSION
        ),
        "libsumo_version": libsumo_version(),
        "scenario": str(scenario),
        "input": _scenario_input_fingerprint(Path(sumocfg)),
        "duration_sec": float(duration_sec),
        "control_interval_sec": int(control_interval_sec),
        "warmup_sec": float(warmup_sec),
        "seed": int(seed),
        "max_focal_tls": int(max_focal_tls),
        "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
        "behavior_policy": str(behavior_policy),
        "collection_shard_index": int(collection_shard_index),
        "collection_shard_count": int(collection_shard_count),
        "state_snapshot_protocol": dict(STATE_SNAPSHOT_PROTOCOL_V3),
        "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
    }
    if prefix_horizons:
        identity["rollout_prefix_horizons_sec"] = list(prefix_horizons)
    if str(counterfactual_cost_mode) != "system_vehicle_load":
        identity["counterfactual_cost_mode"] = str(counterfactual_cost_mode)
    return identity


@functools.lru_cache(maxsize=128)
def _scenario_input_fingerprint(sumocfg: Path) -> str:
    """Hash exact SUMO inputs rather than mutable file timestamps."""

    sumocfg = Path(sumocfg).resolve()
    root = ET.parse(sumocfg).getroot()
    paths = {sumocfg}
    for tag in ("net-file", "route-files", "additional-files"):
        for item in root.findall(f".//{tag}"):
            for value in str(item.attrib.get("value", "")).split(","):
                value = value.strip()
                if value:
                    paths.add((sumocfg.parent / value).resolve())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(f"SUMO input referenced by {sumocfg} is missing: {path}")
        try:
            label = str(path.relative_to(sumocfg.parent))
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _counterfactual_cache_path(
    *,
    cache_root: Path,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    behavior_policy: str = "phase_pressure",
    counterfactual_cost_mode: str = "system_vehicle_load",
    collection_shard_index: int = 0,
    collection_shard_count: int = 1,
    rollout_prefix_horizons_sec: Sequence[int] | None = None,
) -> Path:
    sumocfg = Path(sumocfg)
    key = _counterfactual_cache_identity(
        sumocfg=sumocfg,
        scenario=scenario,
        duration_sec=duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seed=seed,
        max_focal_tls=max_focal_tls,
        counterfactual_horizon_intervals=counterfactual_horizon_intervals,
        behavior_policy=behavior_policy,
        counterfactual_cost_mode=counterfactual_cost_mode,
        collection_shard_index=collection_shard_index,
        collection_shard_count=collection_shard_count,
        rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
    )
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return Path(cache_root) / f"{scenario}__{digest}.npz"


def collect_counterfactual_bank_v3(
    *,
    env_root: Path,
    scenarios: Sequence[str],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    workers: int,
    cache_root: Path | None = DEFAULT_CACHE_ROOT_V3,
    behavior_policy: str = "phase_pressure",
    counterfactual_cost_mode: str = "system_vehicle_load",
    scenario_sumocfgs: Mapping[str, Path] | None = None,
    collection_shards: int = 1,
    rollout_prefix_horizons_sec: Sequence[int] | None = None,
) -> dict[str, MechanismDataset]:
    if seeds is None:
        if seed is None:
            raise ValueError("counterfactual collection requires seed or seeds")
        seeds = (int(seed),)
    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("counterfactual source seed list cannot be empty")
    if int(collection_shards) < 1:
        raise ValueError("collection_shards must be positive")
    resolved_sumocfgs = _resolve_scenario_sumocfgs(
        env_root=env_root,
        scenarios=scenarios,
        scenario_sumocfgs=scenario_sumocfgs,
    )
    payloads = [
        {
            "scenario": scenario,
            "sumocfg": str(resolved_sumocfgs[scenario]),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(source_seed),
            "max_focal_tls": int(max_focal_tls),
            "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
            "behavior_policy": str(behavior_policy),
            "counterfactual_cost_mode": str(counterfactual_cost_mode),
            "collection_shard_index": int(shard_index),
            "collection_shard_count": int(collection_shards),
            "rollout_prefix_horizons_sec": list(
                normalize_rollout_prefix_horizons(
                    rollout_prefix_horizons_sec,
                    rollout_horizon_sec=int(control_interval_sec)
                    * max(int(counterfactual_horizon_intervals), 1),
                )
            ),
            "cache_identity": _counterfactual_cache_identity(
                sumocfg=resolved_sumocfgs[scenario],
                scenario=scenario,
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=source_seed,
                max_focal_tls=max_focal_tls,
                counterfactual_horizon_intervals=counterfactual_horizon_intervals,
                behavior_policy=behavior_policy,
                counterfactual_cost_mode=counterfactual_cost_mode,
                collection_shard_index=shard_index,
                collection_shard_count=collection_shards,
                rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
            ),
            "cache_path": str(
                _counterfactual_cache_path(
                    cache_root=Path(cache_root),
                    sumocfg=resolved_sumocfgs[scenario],
                    scenario=scenario,
                    duration_sec=duration_sec,
                    control_interval_sec=control_interval_sec,
                    warmup_sec=warmup_sec,
                    seed=source_seed,
                    max_focal_tls=max_focal_tls,
                    counterfactual_horizon_intervals=counterfactual_horizon_intervals,
                    behavior_policy=behavior_policy,
                    counterfactual_cost_mode=counterfactual_cost_mode,
                    collection_shard_index=shard_index,
                    collection_shard_count=collection_shards,
                    rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
                )
            )
            if cache_root is not None
            else None,
        }
        for scenario in scenarios
        for source_seed in seeds
        for shard_index in range(int(collection_shards))
    ]
    cached_payloads = []
    missing_payloads = []
    for payload in payloads:
        destination = payload.get("cache_path")
        if destination and Path(destination).is_file():
            cached_payloads.append(payload)
        else:
            missing_payloads.append(payload)
    cached: list[tuple[str, int, int, MechanismDataset]] = []
    if cached_payloads:
        # Cache reads do not touch libsumo state, so process isolation would
        # only add hundreds of Python imports to every experiment budget.
        read_workers = min(max(int(workers), 1), len(cached_payloads), 32)
        with ThreadPoolExecutor(max_workers=read_workers) as pool:
            cached = list(pool.map(_collect_worker, cached_payloads))
    collected = [
        *cached,
        *_map_fresh_sumo_processes(_collect_worker, missing_payloads, workers),
    ]
    by_scenario_seed: dict[tuple[str, int], list[tuple[int, MechanismDataset]]] = {
        (scenario, source_seed): [] for scenario in scenarios for source_seed in seeds
    }
    for scenario, source_seed, shard_index, dataset in collected:
        by_scenario_seed[(scenario, source_seed)].append((shard_index, dataset))
    return {
        scenario: _merge_scenario_seed_datasets_v3(
            scenario,
            [
                _merge_collection_shards_v3(
                    scenario,
                    source_seed,
                    [
                        dataset
                        for _, dataset in sorted(
                            by_scenario_seed[(scenario, source_seed)],
                            key=lambda item: item[0],
                        )
                    ],
                )
                for source_seed in seeds
            ],
        )
        for scenario in scenarios
    }


def _merge_collection_shards_v3(
    scenario: str,
    seed: int,
    datasets: Sequence[MechanismDataset],
) -> MechanismDataset:
    if not datasets:
        raise ValueError(f"no counterfactual shards collected for {scenario} seed={seed}")
    expected_count = int(datasets[0].metadata.get("collection_shard_count", 1))
    indices = sorted(int(item.metadata.get("collection_shard_index", 0)) for item in datasets)
    if len(datasets) != expected_count or indices != list(range(expected_count)):
        raise ValueError(
            f"incomplete counterfactual shard set for {scenario} seed={seed}: "
            f"expected {expected_count}, found {indices}"
        )
    if expected_count == 1:
        return datasets[0]
    opportunity_counts = {
        int(item.metadata.get("eligible_collection_opportunities", -1))
        for item in datasets
    }
    if len(opportunity_counts) != 1 or next(iter(opportunity_counts)) < 0:
        raise ValueError(
            f"counterfactual shards disagree on decision opportunities for {scenario} seed={seed}"
        )
    behavior_traces = {
        str(item.metadata.get("behavior_trace_sha256", "")) for item in datasets
    }
    behavior_interval_counts = {
        int(item.metadata.get("behavior_interval_count", -1)) for item in datasets
    }
    phase_execution_audits = {
        json.dumps(item.metadata.get("phase_execution_audit", {}), sort_keys=True)
        for item in datasets
    }
    focal_selection_protocols = {
        str(item.metadata.get("focal_selection_protocol", "")) for item in datasets
    }
    focal_selection_counts = {
        json.dumps(item.metadata.get("focal_selection_count_by_tls", {}), sort_keys=True)
        for item in datasets
    }
    safety_monitoring_protocols = {
        json.dumps(item.metadata.get("strict_safety_monitoring", {}), sort_keys=True)
        for item in datasets
    }
    if (
        len(behavior_traces) != 1
        or len(behavior_interval_counts) != 1
        or len(phase_execution_audits) != 1
        or len(focal_selection_protocols) != 1
        or "" in focal_selection_protocols
        or len(focal_selection_counts) != 1
        or safety_monitoring_protocols
        != {json.dumps(STRICT_SAFETY_MONITORING_V3, sort_keys=True)}
    ):
        raise ValueError(
            f"counterfactual shards disagree on uninterrupted behavior trace for "
            f"{scenario} seed={seed}"
        )
    merged = merge_counterfactual_datasets_v3(datasets)
    metadata = {
        **dict(merged.metadata),
        "scenario": scenario,
        "seed": int(seed),
        "duration_sec": float(datasets[0].metadata["duration_sec"]),
        "control_interval_sec": int(datasets[0].metadata["control_interval_sec"]),
        "warmup_sec": float(datasets[0].metadata["warmup_sec"]),
        "max_focal_tls": int(datasets[0].metadata["max_focal_tls"]),
        "collection_shard_count": expected_count,
        "collection_shard_indices": indices,
        "eligible_collection_opportunities": next(iter(opportunity_counts)),
        "behavior_trace_sha256": next(iter(behavior_traces)),
        "behavior_interval_count": next(iter(behavior_interval_counts)),
        "captured_snapshot_count": int(
            sum(int(item.metadata.get("captured_snapshot_count", 0)) for item in datasets)
        ),
        "counterfactual_branches": int(
            sum(int(item.metadata["counterfactual_branches"]) for item in datasets)
        ),
        "state_restores": int(sum(int(item.metadata["state_restores"]) for item in datasets)),
        "phase_execution_audit": dict(datasets[0].metadata.get("phase_execution_audit", {})),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        "focal_selection_protocol": next(iter(focal_selection_protocols)),
        "focal_selection_count_by_tls": dict(
            datasets[0].metadata.get("focal_selection_count_by_tls", {})
        ),
        "focal_selection_min_count": int(
            datasets[0].metadata.get("focal_selection_min_count", 0)
        ),
        "focal_selection_max_count": int(
            datasets[0].metadata.get("focal_selection_max_count", 0)
        ),
        "controllable_tls_ids": sorted(
            {
                str(tls_id)
                for item in datasets
                for tls_id in item.metadata.get("controllable_tls_ids", ())
            }
        ),
        "covered_tls_ids": sorted(
            {
                str(tls_id)
                for item in datasets
                for tls_id in item.metadata.get("covered_tls_ids", ())
            }
        ),
    }
    metadata["controllable_tls_count"] = len(metadata["controllable_tls_ids"])
    metadata["covered_tls_count"] = len(metadata["covered_tls_ids"])
    metadata["tls_coverage_fraction"] = metadata["covered_tls_count"] / max(
        metadata["controllable_tls_count"], 1
    )
    return MechanismDataset(
        feature_names=merged.feature_names,
        features=merged.features,
        context_names=merged.context_names,
        context=merged.context,
        priors=merged.priors,
        targets=merged.targets,
        domains=merged.domains,
        metadata=metadata,
    )


def _merge_scenario_seed_datasets_v3(
    scenario: str,
    datasets: Sequence[MechanismDataset],
) -> MechanismDataset:
    if not datasets:
        raise ValueError(f"no counterfactual datasets collected for {scenario}")
    if len(datasets) == 1:
        return datasets[0]
    focal_protocols = {
        str(item.metadata.get("focal_selection_protocol", "")) for item in datasets
    }
    if len(focal_protocols) != 1 or "" in focal_protocols:
        raise ValueError(
            f"counterfactual seeds disagree on focal selection protocol for {scenario}"
        )
    if any(
        item.metadata.get("strict_safety_monitoring")
        != STRICT_SAFETY_MONITORING_V3
        for item in datasets
    ):
        raise ValueError(
            f"counterfactual seeds disagree on safety monitoring protocol for {scenario}"
        )
    merged = merge_counterfactual_datasets_v3(datasets)
    metadata = {
        **dict(merged.metadata),
        "scenario": scenario,
        "source_seeds": [int(item.metadata["seed"]) for item in datasets],
        "behavior_trace_sha256_by_seed": {
            str(int(item.metadata["seed"])): str(item.metadata["behavior_trace_sha256"])
            for item in datasets
        },
        "behavior_interval_count_by_seed": {
            str(int(item.metadata["seed"])): int(item.metadata["behavior_interval_count"])
            for item in datasets
        },
        "eligible_collection_opportunities_by_seed": {
            str(int(item.metadata["seed"])): int(
                item.metadata["eligible_collection_opportunities"]
            )
            for item in datasets
        },
        "focal_selection_protocol": str(
            datasets[0].metadata.get("focal_selection_protocol", "")
        ),
        "focal_selection_min_count_by_seed": {
            str(int(item.metadata["seed"])): int(
                item.metadata.get("focal_selection_min_count", 0)
            )
            for item in datasets
        },
        "focal_selection_max_count_by_seed": {
            str(int(item.metadata["seed"])): int(
                item.metadata.get("focal_selection_max_count", 0)
            )
            for item in datasets
        },
        "captured_snapshot_count": int(
            sum(int(item.metadata.get("captured_snapshot_count", 0)) for item in datasets)
        ),
        "duration_sec_per_seed": float(datasets[0].metadata["duration_sec"]),
        "counterfactual_branches": int(
            sum(int(item.metadata["counterfactual_branches"]) for item in datasets)
        ),
        "state_restores": int(sum(int(item.metadata["state_restores"]) for item in datasets)),
        "phase_execution_audit": _sum_numeric(
            [item.metadata.get("phase_execution_audit", {}) for item in datasets]
        ),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        "controllable_tls_ids": sorted(
            {
                str(tls_id)
                for item in datasets
                for tls_id in item.metadata.get("controllable_tls_ids", ())
            }
        ),
        "covered_tls_ids": sorted(
            {
                str(tls_id)
                for item in datasets
                for tls_id in item.metadata.get("covered_tls_ids", ())
            }
        ),
    }
    metadata["controllable_tls_count"] = len(metadata["controllable_tls_ids"])
    metadata["covered_tls_count"] = len(metadata["covered_tls_ids"])
    metadata["tls_coverage_fraction"] = metadata["covered_tls_count"] / max(
        metadata["controllable_tls_count"], 1
    )
    return MechanismDataset(
        feature_names=merged.feature_names,
        features=merged.features,
        context_names=merged.context_names,
        context=merged.context,
        priors=merged.priors,
        targets=merged.targets,
        domains=merged.domains,
        metadata=metadata,
    )


def _target_adaptation_subset_v3(
    dataset: MechanismDataset,
    *,
    group_budget: int,
    selection_seed: int,
) -> MechanismDataset:
    if int(group_budget) < 0:
        raise ValueError("target adaptation group budget must be nonnegative")
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    if groups.shape != (dataset.size,):
        raise ValueError("target adaptation requires row-aligned action groups")
    unique_groups = sorted(set(groups.tolist()))
    budget = min(max(int(group_budget), 0), len(unique_groups))
    ranked = sorted(
        unique_groups,
        key=lambda group: hashlib.sha256(f"{selection_seed}:{group}".encode("utf-8")).hexdigest(),
    )
    selected_groups = set(ranked[:budget])
    return _group_subset_v3(
        dataset,
        selected_groups=selected_groups,
        metadata_updates={
            "target_adaptation_group_budget": int(group_budget),
            "target_adaptation_groups_selected": int(budget),
            "target_adaptation_selection_seed": int(selection_seed),
            "target_adaptation_selected_group_ids": sorted(selected_groups),
        },
    )


def _group_subset_v3(
    dataset: MechanismDataset,
    *,
    selected_groups: set[str],
    metadata_updates: Mapping[str, Any] | None = None,
) -> MechanismDataset:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    if groups.shape != (dataset.size,):
        raise ValueError("group subset requires row-aligned action groups")
    mask = np.asarray([group in selected_groups for group in groups], dtype=bool)
    subset = dataset.subset(mask)
    metadata = dict(dataset.metadata)
    for key, value in tuple(metadata.items()):
        array = np.asarray(value)
        if array.shape == (dataset.size,):
            metadata[key] = array[mask].tolist()
    parent_safety_audit = dict(
        dataset.metadata.get("counterfactual_safety_audit", {})
    )
    if parent_safety_audit:
        if (
            parent_safety_audit.get("protocol")
            != COUNTERFACTUAL_SAFETY_PROTOCOL_V3
            or not parent_safety_audit.get("no_teleport_passed", False)
            or not parent_safety_audit.get(
                "symmetric_group_censoring_passed", False
            )
            or int(parent_safety_audit.get("retained_branches", -1))
            != dataset.size
        ):
            raise ValueError(
                "cannot derive a target subset from a dataset with an invalid "
                "counterfactual safety audit"
            )
        selected_row_groups = np.asarray(groups[mask], dtype=str)
        retained_group_count = len(set(selected_row_groups.tolist()))
        retained_branch_count = int(mask.sum())
        parent_audit_sha256 = hashlib.sha256(
            json.dumps(
                parent_safety_audit,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        metadata["counterfactual_safety_audit"] = {
            "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
            "audit_scope": "materialized_retained_group_subset_v1",
            "candidate_groups": retained_group_count,
            "retained_groups": retained_group_count,
            "censored_groups": 0,
            "candidate_branches": retained_branch_count,
            "attempted_branches": retained_branch_count,
            "retained_branches": retained_branch_count,
            "censored_branches": 0,
            "unevaluated_censored_branches": 0,
            "observed_unsafe_branches": 0,
            "raw_collision_events_observed": 0,
            "collision_event_steps_observed": 0,
            "unique_collision_incidents_observed": 0,
            "collision_incident_signatures": [],
            "censored_group_fraction": 0.0,
            "censored_group_ids": [],
            "collision_samples": [],
            "starting_teleports": 0,
            "ending_teleports": 0,
            "no_teleport_passed": True,
            "symmetric_group_censoring_passed": True,
            "parent_safety_audit_sha256": parent_audit_sha256,
            "parent_safety_summary": {
                key: int(parent_safety_audit.get(key, 0))
                for key in (
                    "candidate_groups",
                    "retained_groups",
                    "censored_groups",
                    "candidate_branches",
                    "retained_branches",
                    "censored_branches",
                    "observed_unsafe_branches",
                )
            },
        }
    metadata.update(dict(metadata_updates or {}))
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


def _group_seed_v3(group: str) -> int | None:
    match = re.search(r":seed(-?\d+):", str(group))
    return int(match.group(1)) if match else None


def _group_tls_map_v3(dataset: MechanismDataset) -> dict[str, str] | None:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    row_tls = np.asarray(dataset.metadata.get("row_tls", ()), dtype=str)
    if row_tls.shape != (dataset.size,):
        return None
    mapping: dict[str, str] = {}
    for group, tls_id in zip(groups, row_tls, strict=True):
        previous = mapping.setdefault(str(group), str(tls_id))
        if previous != str(tls_id):
            raise ValueError(f"action group {group!r} spans multiple TLS IDs")
    return mapping


def _coverage_first_group_order_v3(
    dataset: MechanismDataset,
    candidates: Sequence[str],
    *,
    selection_seed: int,
    role: str,
) -> list[str]:
    group_tls = _group_tls_map_v3(dataset)

    def digest(value: str) -> str:
        return hashlib.sha256(
            f"{selection_seed}:{role}:{value}".encode("utf-8")
        ).hexdigest()

    if group_tls is None:
        return sorted(candidates, key=digest)
    by_tls: dict[str, list[str]] = {}
    for group in candidates:
        by_tls.setdefault(group_tls[str(group)], []).append(str(group))
    for tls_id in by_tls:
        by_tls[tls_id].sort(key=digest)
    tls_order = sorted(by_tls, key=lambda tls_id: digest(f"tls:{tls_id}"))
    ordered = []
    depth = 0
    while len(ordered) < len(candidates):
        added = False
        for tls_id in tls_order:
            values = by_tls[tls_id]
            if depth < len(values):
                ordered.append(values[depth])
                added = True
        if not added:
            break
        depth += 1
    if len(ordered) != len(candidates):
        raise AssertionError("coverage-first target group ordering lost candidates")
    return ordered


def _dataset_tls_ids_v3(dataset: MechanismDataset) -> list[str]:
    row_tls = np.asarray(dataset.metadata.get("row_tls", ()), dtype=str)
    return sorted(set(row_tls.tolist())) if row_tls.shape == (dataset.size,) else []


def _target_adaptation_split_v3(
    dataset: MechanismDataset,
    *,
    group_budget: int,
    selection_seed: int,
    calibration_seeds: Sequence[int],
    calibration_fraction: float = 0.20,
) -> dict[str, MechanismDataset]:
    if int(group_budget) < 0:
        raise ValueError("target adaptation group budget must be nonnegative")
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    if groups.shape != (dataset.size,):
        raise ValueError("target adaptation split requires row-aligned action groups")
    unique_groups = sorted(set(groups.tolist()))
    budget = min(int(group_budget), len(unique_groups))
    calibration_seed_set = {int(seed) for seed in calibration_seeds}
    calibration_candidates = [
        group for group in unique_groups if _group_seed_v3(group) in calibration_seed_set
    ]
    adaptation_candidates = [
        group for group in unique_groups if _group_seed_v3(group) not in calibration_seed_set
    ]

    desired_calibration = 0
    if budget >= 2 and calibration_candidates:
        desired_calibration = min(
            max(int(np.ceil(float(calibration_fraction) * budget)), 1),
            budget - 1,
        )
    calibration_count = min(desired_calibration, len(calibration_candidates))
    adaptation_count = min(budget - calibration_count, len(adaptation_candidates))
    if adaptation_count + calibration_count < budget:
        calibration_count = min(
            budget - adaptation_count,
            len(calibration_candidates),
        )

    def rank(candidates: Sequence[str], role: str) -> list[str]:
        return _coverage_first_group_order_v3(
            dataset,
            candidates,
            selection_seed=selection_seed,
            role=role,
        )

    adaptation_groups = set(rank(adaptation_candidates, "adaptation")[:adaptation_count])
    calibration_groups = set(rank(calibration_candidates, "calibration")[:calibration_count])
    selected_groups = adaptation_groups | calibration_groups
    common = {
        "target_adaptation_group_budget": int(group_budget),
        "target_adaptation_groups_selected": len(selected_groups),
        "target_adaptation_selection_seed": int(selection_seed),
        "target_calibration_seeds": sorted(calibration_seed_set),
        "target_adaptation_selected_group_ids": sorted(selected_groups),
        "target_group_selection_protocol": TARGET_GROUP_SELECTION_PROTOCOL_V3,
    }
    return {
        "adaptation": _group_subset_v3(
            dataset,
            selected_groups=adaptation_groups,
            metadata_updates={
                **common,
                "target_data_role": "model_adaptation",
                "target_role_groups": len(adaptation_groups),
            },
        ),
        "calibration": _group_subset_v3(
            dataset,
            selected_groups=calibration_groups,
            metadata_updates={
                **common,
                "target_data_role": "policy_selection_calibration",
                "target_role_groups": len(calibration_groups),
            },
        ),
        "selected": _group_subset_v3(
            dataset,
            selected_groups=selected_groups,
            metadata_updates={
                **common,
                "target_data_role": "budget_union_audit_only",
                "target_role_groups": len(selected_groups),
            },
        ),
    }


def _primary_reference_policy_v3(
    policies: Sequence[str],
    declared: str | None = None,
) -> str | None:
    requested = tuple(str(policy) for policy in policies)
    if declared is not None:
        selected = str(declared).strip()
        if not selected:
            raise ValueError("primary_reference_policy must be non-empty when declared")
        if selected not in requested:
            raise ValueError(
                "declared primary_reference_policy is absent from requested policies: "
                f"{selected}"
            )
        return selected
    return next(
        (policy for policy in PRIMARY_POLICY_PRIORITY_V3 if policy in requested),
        None,
    )


def _json_ready(value: Any) -> Any:
    """Convert a result tree to strict, portable JSON values."""

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


def _relabel_dataset_domain(dataset: MechanismDataset, city_group: str) -> MechanismDataset:
    metadata = {**dict(dataset.metadata), "city_group": str(city_group)}
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=np.asarray([str(city_group)] * dataset.size),
        metadata=metadata,
    )


def _static_context_from_dataset(dataset: MechanismDataset) -> np.ndarray:
    if dataset.size < 1:
        raise ValueError("static target context requires at least one dataset row")
    context = np.asarray(dataset.context, dtype=float)
    reference = context[0]
    if not np.allclose(context, reference[None, :], rtol=0.0, atol=1e-12):
        raise ValueError("scenario context varies by transition and is not label-free static data")
    return reference.copy()


def _fit_target_models(
    bank: Mapping[str, MechanismDataset],
    target: str,
    *,
    fit_config: MechanismFitConfig,
    source_rule_costs: Mapping[str, Mapping[str, float]],
    source_rule_specs: Mapping[str, PressurePolicySpec],
    model_families: Sequence[str],
    target_group_budget: int,
    target_adaptation_selection_seed: int,
    target_calibration_seeds: Sequence[int],
    min_target_calibration_groups: int,
    target_calibration_fraction: float,
    target_guard_selection_mode: str,
    source_selector_min_context_domains: int,
    scenario_city_groups: Mapping[str, str],
    fit_parallel_workers: int = 1,
) -> FittedContrastModelsV3:
    target_city_group = str(scenario_city_groups[target])
    source_names, heldout_city_names = _strict_city_fold(
        tuple(bank), target, scenario_city_groups
    )
    if target in source_names or set(source_names) & set(heldout_city_names):
        raise AssertionError("target city entered the source set")
    source_datasets = [
        _relabel_dataset_domain(bank[name], str(scenario_city_groups[name]))
        for name in source_names
    ]
    target_split = _target_adaptation_split_v3(
        bank[target],
        group_budget=target_group_budget,
        selection_seed=target_adaptation_selection_seed,
        calibration_seeds=target_calibration_seeds,
        calibration_fraction=target_calibration_fraction,
    )
    target_adaptation = _relabel_dataset_domain(
        target_split["adaptation"], target_city_group
    )
    target_calibration = target_split["calibration"]
    target_selected = _relabel_dataset_domain(target_split["selected"], target_city_group)
    target_context = _static_context_from_dataset(bank[target])
    all_target_tls_ids = _dataset_tls_ids_v3(bank[target])
    selected_tls_ids = _dataset_tls_ids_v3(target_selected)
    adaptation_tls_ids = _dataset_tls_ids_v3(target_adaptation)
    calibration_tls_ids = _dataset_tls_ids_v3(target_calibration)
    available_target_groups = len(
        set(str(value) for value in bank[target].metadata.get("action_group_ids", ()))
    )
    source_city_groups = sorted({str(scenario_city_groups[name]) for name in source_names})
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
        raise AssertionError("target city entered source rule selector")
    candidate_policies = tuple(source_rule_specs)
    prior_policy, prior_diagnostics = select_contextual_policy_from_domain_costs(
        selector_costs,
        {
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
        },
        target_context=target_context,
        policies=candidate_policies,
        min_context_domains=int(source_selector_min_context_domains),
    )
    fit_kwargs = {
        "cfcmt_config": fit_config,
        "target_context": target_context,
        "prior_policy_override": source_rule_specs[prior_policy],
        "prior_selection_diagnostics": prior_diagnostics,
        "model_families": model_families,
        "target_domain": target_city_group,
        "parallel_workers": max(int(fit_parallel_workers), 1),
    }
    target_selection: dict[str, Any] = {
        "eligible": False,
        "reason": "no_target_calibration_requested",
        "group_count": int(len(set(target_calibration.metadata["action_group_ids"]))),
        "guards": {},
        "regularizers": {},
        "diagnostics": {},
    }
    selection_models: FittedContrastModelsV3 | None = None
    if target_calibration.size and int(target_selection["group_count"]) >= int(
        min_target_calibration_groups
    ):
        selection_parts = [*source_datasets]
        if target_adaptation.size:
            selection_parts.append(target_adaptation)
        selection_models = fit_contrast_models_v3(
            merge_counterfactual_datasets_v3(selection_parts),
            **fit_kwargs,
        )
        if target_adaptation.size:
            selection_models.target_support = TargetActionSupport.fit(
                build_action_contrast_dataset(
                    target_adaptation,
                    reference_policy=selection_models.prior_spec,
                    contrast_features=CONTRAST_FEATURES_V3,
                )
            )
        target_selection = calibrate_target_policy_selection_v3(
            target_calibration,
            fitted_models=selection_models,
            cfcmt_config=fit_config,
            min_calibration_groups=min_target_calibration_groups,
            guard_selection_mode=target_guard_selection_mode,
        )

    if target_selection["eligible"]:
        if selection_models is None:
            raise AssertionError(
                "eligible target calibration has no fitted adaptation model"
            )
        # Deploy the exact predictor evaluated by the disjoint calibration
        # groups. Re-fitting on those groups would invalidate the safety gate.
        models = selection_models
        models.guards.update(target_selection["guards"])
        models.regularizers.update(target_selection["regularizers"])
        deployment_model_groups = int(
            target_adaptation.metadata["target_role_groups"]
        )
        deployment_model_group_ids = sorted(
            set(
                str(group)
                for group in target_adaptation.metadata.get(
                    "action_group_ids", ()
                )
            )
        )
        calibration_refit = False
    else:
        final_parts = [*source_datasets]
        if target_adaptation.size:
            final_parts.append(target_adaptation)
        source = merge_counterfactual_datasets_v3(final_parts)
        models = fit_contrast_models_v3(
            source,
            **fit_kwargs,
        )
        if target_adaptation.size:
            models.target_support = TargetActionSupport.fit(
                build_action_contrast_dataset(
                    target_adaptation,
                    reference_policy=models.prior_spec,
                    contrast_features=CONTRAST_FEATURES_V3,
                )
            )
            for family in models.family_models:
                models.guards[family] = replace(
                    models.guards[family], enabled=False
                )
                models.regularizers[family] = replace(
                    models.regularizers[family],
                    enabled=False,
                )
        deployment_model_groups = int(
            target_adaptation.metadata["target_role_groups"]
        )
        deployment_model_group_ids = sorted(
            set(
                str(group)
                for group in target_adaptation.metadata.get(
                    "action_group_ids", ()
                )
            )
        )
        calibration_refit = False
    target_specialist_fitted = (
        CFCMT_TARGET_SPECIALIST_FAMILY_V3 in models.family_models
        and has_fitted_target_specialist_v3(
            models.family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
        )
    )
    if target_selection["eligible"]:
        hierarchy_selection = select_target_capacity_layers_v3(
            target_specialist_fitted=target_specialist_fitted,
            guards=models.guards,
            regularizers=models.regularizers,
            calibration_diagnostics=target_selection["diagnostics"],
        )
        models.hierarchy_layers.update(hierarchy_selection["layers"])
    else:
        hierarchy_selection = {
            "protocol": "source-calibrated-default-hierarchy",
            "applied": False,
            "reason": str(target_selection["reason"]),
            "target_specialist_fitted": bool(target_specialist_fitted),
            "layers": {
                mode: tuple(layers)
                for mode, layers in models.hierarchy_layers.items()
            },
        }
    local_graph_selector = _apply_local_graph_selector_v3(models)
    models.diagnostics["cfcmt_hierarchy"].update(
        {
            "target_specialist_fitted": bool(target_specialist_fitted),
            "target_capacity_selection": hierarchy_selection,
            "enabled_layers": {
                mode: list(layers)
                for mode, layers in models.hierarchy_layers.items()
            },
        }
    )
    fitted_domains = set(models.diagnostics["source_domains"])
    if int(target_group_budget) <= 0 and target_city_group in fitted_domains:
        raise AssertionError("zero-shot target city appears in fitted V3 diagnostics")
    if target_selected.size > 0 and target_city_group not in fitted_domains:
        raise AssertionError("target adaptation labels were not included in fitted V3 diagnostics")
    models.diagnostics["target_adaptation"] = {
        "target": target,
        "target_city_group": target_city_group,
        "heldout_city_scenarios": sorted(heldout_city_names),
        "source_city_groups": source_city_groups,
        "source_scenarios": sorted(source_names),
        "target_context_provenance": "static_sumocfg_network_phase_and_route_summary",
        "target_context_transition_labels_used": False,
        "requested_group_budget": int(target_group_budget),
        "selected_groups": int(target_selected.metadata["target_adaptation_groups_selected"]),
        "available_target_groups": int(available_target_groups),
        "selected_group_fraction": float(
            int(target_selected.metadata["target_adaptation_groups_selected"])
            / max(available_target_groups, 1)
        ),
        "adaptation_groups": int(target_adaptation.metadata["target_role_groups"]),
        "calibration_groups": int(target_calibration.metadata["target_role_groups"]),
        "rows": int(target_selected.size),
        "adaptation_rows": int(target_adaptation.size),
        "calibration_rows": int(target_calibration.size),
        "selected_group_ids": sorted(
            str(group)
            for group in target_selected.metadata.get(
                "target_adaptation_selected_group_ids", ()
            )
        ),
        "adaptation_group_ids": sorted(
            set(
                str(group)
                for group in target_adaptation.metadata.get("action_group_ids", ())
            )
        ),
        "calibration_group_ids": sorted(
            set(
                str(group)
                for group in target_calibration.metadata.get("action_group_ids", ())
            )
        ),
        "target_group_selection_protocol": TARGET_GROUP_SELECTION_PROTOCOL_V3,
        "available_target_tls_ids": all_target_tls_ids,
        "available_target_tls_count": len(all_target_tls_ids),
        "selected_tls_ids": selected_tls_ids,
        "selected_tls_count": len(selected_tls_ids),
        "selected_tls_fraction": float(
            len(selected_tls_ids) / max(len(all_target_tls_ids), 1)
        ),
        "adaptation_tls_ids": adaptation_tls_ids,
        "adaptation_tls_count": len(adaptation_tls_ids),
        "adaptation_tls_fraction": float(
            len(adaptation_tls_ids) / max(len(all_target_tls_ids), 1)
        ),
        "calibration_tls_ids": calibration_tls_ids,
        "calibration_tls_count": len(calibration_tls_ids),
        "calibration_tls_fraction": float(
            len(calibration_tls_ids) / max(len(all_target_tls_ids), 1)
        ),
        "selection_seed": int(target_adaptation_selection_seed),
        "calibration_seeds": [int(seed) for seed in target_calibration_seeds],
        "evaluation_seed_overlap": False,
        "label_type": "matched_target_simulator_counterfactual_action_outcomes",
        "calibration_fraction": float(target_calibration_fraction),
        "guard_selection_mode": str(target_guard_selection_mode),
        "deployment_fit_protocol": TARGET_DEPLOYMENT_FIT_PROTOCOL_V3,
        "deployment_model_groups": int(deployment_model_groups),
        "deployment_model_group_ids": deployment_model_group_ids,
        "calibration_groups_refit_into_deployment_model": bool(
            calibration_refit
        ),
        "local_action_support": models.target_support.diagnostics()
        if models.target_support is not None
        else None,
        "local_graph_selector": local_graph_selector,
        "policy_selection": {
            "eligible": bool(target_selection["eligible"]),
            "reason": str(target_selection["reason"]),
            "group_count": int(target_selection["group_count"]),
            "min_calibration_groups": int(min_target_calibration_groups),
            "selected_guards": {
                family: asdict(config)
                for family, config in target_selection["guards"].items()
            },
            "selected_regularizers": {
                family: asdict(config)
                for family, config in target_selection["regularizers"].items()
            },
            "diagnostics": target_selection["diagnostics"],
            "hierarchy": hierarchy_selection,
        },
    }
    models.diagnostics["model_fit_parallelism"] = {
        "protocol": "target_process_plus_source_loo_threads_v1",
        "source_loo_workers_per_target": max(int(fit_parallel_workers), 1),
    }
    return models


def _apply_local_graph_selector_v3(
    models: FittedContrastModelsV3,
    *,
    core_family: str = CFCMT_CORE_FAMILY_V3,
    local_family: str = "causal_balanced_advantage",
    min_source_mean_gain: float = 0.001,
    min_improving_source_domains: int = 2,
) -> dict[str, Any]:
    if core_family not in models.family_models or local_family not in models.family_models:
        return {
            "applicable": False,
            "selected": core_family if core_family in models.family_models else None,
            "reason": "paired_core_and_local_families_not_fitted",
        }
    source = models.diagnostics["guard_calibration"][local_family]
    selected = source.get("selected")
    domain_values = selected.get("domain_mean_delta", {}) if selected else {}
    improving_domains = sum(float(value) < 0.0 for value in domain_values.values())
    source_mean_gain = -float(selected["mean_actual_delta"]) if selected else 0.0
    source_stable = (
        selected is not None
        and source_mean_gain >= float(min_source_mean_gain)
        and improving_domains >= int(min_improving_source_domains)
        and float(selected["worst_domain_delta"]) <= 0.0
    )
    target_enabled = bool(models.guards[local_family].enabled)
    local_enabled = source_stable and target_enabled
    if not local_enabled:
        models.guards[local_family] = replace(models.guards[local_family], enabled=False)
        models.regularizers[local_family] = replace(
            models.regularizers[local_family],
            enabled=False,
        )
    return {
        "applicable": True,
        "selected": local_family if local_enabled else core_family,
        "reason": "source_loo_and_target_calibration_passed"
        if local_enabled
        else "local_graph_source_stability_gate_failed",
        "source_stable": bool(source_stable),
        "target_enabled": bool(target_enabled),
        "min_source_mean_gain": float(min_source_mean_gain),
        "source_mean_gain": float(source_mean_gain),
        "min_improving_source_domains": int(min_improving_source_domains),
        "improving_source_domains": int(improving_domains),
    }


def _fit_worker(payload: Mapping[str, Any]) -> tuple[str, FittedContrastModelsV3]:
    from threadpoolctl import threadpool_limits

    target = str(payload["target"])
    with threadpool_limits(limits=1):
        return target, _fit_target_models(
            payload["bank"],
            target,
            fit_config=payload["fit_config"],
            source_rule_costs=payload["source_rule_costs"],
            source_rule_specs=payload["source_rule_specs"],
            model_families=payload["model_families"],
            target_group_budget=int(payload["target_group_budget"]),
            target_adaptation_selection_seed=int(payload["target_adaptation_selection_seed"]),
            target_calibration_seeds=payload["target_calibration_seeds"],
            min_target_calibration_groups=int(payload["min_target_calibration_groups"]),
            target_calibration_fraction=float(payload["target_calibration_fraction"]),
            target_guard_selection_mode=str(payload["target_guard_selection_mode"]),
            source_selector_min_context_domains=int(
                payload["source_selector_min_context_domains"]
            ),
            scenario_city_groups=payload["scenario_city_groups"],
            fit_parallel_workers=int(payload.get("fit_parallel_workers", 1)),
        )


def fit_target_models_parallel_v3(
    bank: Mapping[str, MechanismDataset],
    *,
    fit_config: MechanismFitConfig,
    source_rule_costs: Mapping[str, Mapping[str, float]],
    source_rule_specs: Mapping[str, PressurePolicySpec],
    model_families: Sequence[str],
    target_group_budget: int,
    target_adaptation_selection_seed: int,
    target_calibration_seeds: Sequence[int],
    min_target_calibration_groups: int,
    target_calibration_fraction: float,
    target_guard_selection_mode: str,
    source_selector_min_context_domains: int,
    workers: int,
    scenario_city_groups: Mapping[str, str],
) -> dict[str, FittedContrastModelsV3]:
    targets = tuple(bank)
    process_workers = min(max(int(workers), 1), len(targets))
    fit_parallel_workers = max(int(workers) // max(process_workers, 1), 1)
    fit_started = time.monotonic()
    if int(workers) <= 1:
        models = {}
        for index, target in enumerate(targets, start=1):
            models[target] = _fit_target_models(
                bank,
                target,
                fit_config=fit_config,
                source_rule_costs=source_rule_costs,
                source_rule_specs=source_rule_specs,
                model_families=model_families,
                target_group_budget=target_group_budget,
                target_adaptation_selection_seed=target_adaptation_selection_seed,
                target_calibration_seeds=target_calibration_seeds,
                min_target_calibration_groups=min_target_calibration_groups,
                target_calibration_fraction=target_calibration_fraction,
                target_guard_selection_mode=target_guard_selection_mode,
                source_selector_min_context_domains=(
                    source_selector_min_context_domains
                ),
                scenario_city_groups=scenario_city_groups,
                fit_parallel_workers=1,
            )
            elapsed = time.monotonic() - fit_started
            print(
                "CFCMT_FIT_PROGRESS "
                f"{index}/{len(targets)} target={target} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return models
    context = mp.get_context("spawn")
    models: dict[str, FittedContrastModelsV3] = {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(targets)), mp_context=context) as pool:
        futures = [
            pool.submit(
                _fit_worker,
                {
                    "target": target,
                    "bank": bank,
                    "fit_config": fit_config,
                    "source_rule_costs": source_rule_costs,
                    "source_rule_specs": source_rule_specs,
                    "model_families": tuple(model_families),
                    "target_group_budget": int(target_group_budget),
                    "target_adaptation_selection_seed": int(target_adaptation_selection_seed),
                    "target_calibration_seeds": tuple(int(seed) for seed in target_calibration_seeds),
                    "min_target_calibration_groups": int(min_target_calibration_groups),
                    "target_calibration_fraction": float(target_calibration_fraction),
                    "target_guard_selection_mode": str(target_guard_selection_mode),
                    "source_selector_min_context_domains": int(
                        source_selector_min_context_domains
                    ),
                    "scenario_city_groups": dict(scenario_city_groups),
                    "fit_parallel_workers": int(fit_parallel_workers),
                },
            )
            for target in targets
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            target, model = future.result()
            models[target] = model
            elapsed = time.monotonic() - fit_started
            print(
                "CFCMT_FIT_PROGRESS "
                f"{index}/{len(targets)} target={target} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return {target: models[target] for target in targets}


def _source_rule_cache_identity_v3(
    *,
    sumocfg: Path,
    scenario: str,
    policy_spec: PressurePolicySpec,
    seed: int,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    sumo_version: str,
) -> dict[str, Any]:
    return {
        "version": SOURCE_RULE_CACHE_VERSION,
        "sumo_version": str(sumo_version),
        "scenario": str(scenario),
        "input": _scenario_input_fingerprint(Path(sumocfg)),
        "policy_spec": policy_spec.to_dict(),
        "seed": int(seed),
        "duration_sec": float(duration_sec),
        "control_interval_sec": int(control_interval_sec),
        "warmup_sec": float(warmup_sec),
        "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
    }


def _source_rule_cache_path_v3(
    cache_root: Path,
    identity: Mapping[str, Any],
) -> Path:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return Path(cache_root) / f"source_rule_{hashlib.sha256(encoded).hexdigest()}.json"


def _validate_source_rule_row_v3(
    row: Mapping[str, Any],
    *,
    location: str,
) -> None:
    metrics = dict(row.get("metrics", {}))
    if not metrics.get("ok", False):
        raise ValueError(f"source rule rollout failed at {location}")
    if metrics.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
        raise ValueError(f"source rule SUMO execution protocol mismatch at {location}")
    if metrics.get("strict_safety_monitoring") != STRICT_SAFETY_MONITORING_V3:
        raise ValueError(f"source rule safety monitoring protocol mismatch at {location}")
    collision_events = int(metrics.get("collision_events", -1))
    collision_incidents = int(metrics.get("collision_incidents", -1))
    if (
        collision_events < 0
        or collision_incidents < 0
        or collision_incidents > collision_events
    ):
        raise ValueError(
            f"source rule rollout has an invalid collision audit at {location}: "
            f"events={collision_events}, incidents={collision_incidents}"
        )
    teleports = {
        name: int(metrics.get(name, -1))
        for name in ("starting_teleports", "ending_teleports")
    }
    if any(value != 0 for value in teleports.values()):
        raise ValueError(
            f"source rule rollout has teleport events at {location}: {teleports}"
        )


def _rule_calibration_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario = str(payload["scenario"])
    policy_key = str(payload["policy_key"])
    configured_spec = pressure_spec(payload["pressure_spec"])
    cache_identity = dict(payload["cache_identity"])
    cache_path = Path(payload["cache_path"]) if payload.get("cache_path") else None
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("identity") != cache_identity:
            raise ValueError(f"source rule cache identity mismatch at {cache_path}")
        row = dict(cached.get("row", {}))
        _validate_source_rule_row_v3(row, location=str(cache_path))
        return row
    metrics = evaluate_policy_v3(
        sumo_api=_load_libsumo(),
        sumocfg=Path(payload["sumocfg"]),
        scenario=scenario,
        policy="configured_pressure",
        models=None,
        duration_sec=float(payload["duration_sec"]),
        control_interval_sec=int(payload["control_interval_sec"]),
        warmup_sec=float(payload["warmup_sec"]),
        seed=int(payload["seed"]),
        tripinfo_output=Path(payload["tripinfo_output"]),
        pressure_spec_override=configured_spec,
    )
    row = {
        "scenario": scenario,
        "policy": policy_key,
        "pressure_spec": configured_spec.to_dict(),
        "seed": int(payload["seed"]),
        "metrics": metrics,
    }
    _validate_source_rule_row_v3(
        row,
        location=f"{scenario}/{policy_key}/seed={int(payload['seed'])}",
    )
    if cache_path is not None and metrics.get("ok", False):
        atomic_write_json(cache_path, {"identity": cache_identity, "row": row})
    return row


def _collect_source_rule_rows_v3(
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    cached_jobs = [
        job
        for job in jobs
        if job.get("cache_path") and Path(str(job["cache_path"])).is_file()
    ]
    missing_jobs = [job for job in jobs if job not in cached_jobs]
    cached_rows: list[dict[str, Any]] = []
    read_workers = 0
    if cached_jobs:
        read_workers = min(max(int(workers), 1), len(cached_jobs), 32)
        with ThreadPoolExecutor(max_workers=read_workers) as pool:
            cached_rows = list(pool.map(_rule_calibration_worker, cached_jobs))
    print(
        "CFCMT_CACHE source_rules "
        f"hits={len(cached_jobs)} misses={len(missing_jobs)} "
        f"read_workers={read_workers}",
        flush=True,
    )
    fresh_rows = (
        _map_fresh_sumo_processes(_rule_calibration_worker, missing_jobs, workers)
        if missing_jobs
        else []
    )
    rows = [*cached_rows, *fresh_rows]
    rows.sort(key=lambda row: (str(row["scenario"]), str(row["policy"]), int(row["seed"])))
    return rows


def collect_source_rule_costs_v3(
    *,
    env_root: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    workers: int,
    policy_specs: Sequence[PressurePolicySpec] | None = None,
    scenario_sumocfgs: Mapping[str, Path] | None = None,
    source_rule_cache_root: Path | None = None,
    expected_sumo_version: str | None = None,
    tripinfo_root: Path = Path(
        "cf_h2o/results/tripinfo/traffic_signal_resco_cfcmt_v3_source_rules"
    ),
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    specs = tuple(policy_specs or generalized_pressure_grid())
    spec_by_key = {spec.key: spec for spec in specs}
    if len(spec_by_key) != len(specs):
        raise ValueError("source pressure grid contains duplicate policy keys")
    policies = tuple(spec_by_key)
    resolved_sumocfgs = _resolve_scenario_sumocfgs(
        env_root=env_root,
        scenarios=scenarios,
        scenario_sumocfgs=scenario_sumocfgs,
    )
    actual_sumo_version = libsumo_version()
    if expected_sumo_version and actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            "source rule cache SUMO version mismatch: "
            f"expected {expected_sumo_version!r}, found {actual_sumo_version!r}"
        )
    jobs = [
        {
            "scenario": scenario,
            "sumocfg": str(resolved_sumocfgs[scenario]),
            "policy_key": policy,
            "pressure_spec": spec_by_key[policy].to_dict(),
            "seed": int(seed),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "cache_identity": _source_rule_cache_identity_v3(
                sumocfg=resolved_sumocfgs[scenario],
                scenario=scenario,
                policy_spec=spec_by_key[policy],
                seed=int(seed),
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                sumo_version=actual_sumo_version,
            ),
            "tripinfo_output": str(
                Path(tripinfo_root) / f"{scenario}__{policy}__seed{seed}.xml"
            ),
        }
        for scenario in scenarios
        for policy in policies
        for seed in seeds
    ]
    for job in jobs:
        job["cache_path"] = (
            str(
                _source_rule_cache_path_v3(
                    Path(source_rule_cache_root), job["cache_identity"]
                )
            )
            if source_rule_cache_root is not None
            else None
        )
    rows = _collect_source_rule_rows_v3(jobs, workers=workers)
    for row in rows:
        _validate_source_rule_row_v3(
            row,
            location=(
                f"{row.get('scenario')}/{row.get('policy')}/seed={row.get('seed')}"
            ),
        )
    costs = {
        scenario: {
            policy: float(
                np.mean(
                    [
                        row["metrics"]["mean_system_vehicles_per_controlled_lane"]
                        for row in rows
                        if row["scenario"] == scenario and row["policy"] == policy
                    ]
                )
            )
            for policy in policies
        }
        for scenario in scenarios
    }
    rows.sort(key=lambda row: (row["scenario"], row["seed"], row["policy"]))
    return costs, rows


def _evaluate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    scenario = str(payload["scenario"])
    policy = str(payload["policy"])
    with threadpool_limits(limits=1):
        metrics = evaluate_policy_v3(
            sumo_api=_load_libsumo(),
            sumocfg=Path(payload["sumocfg"]),
            scenario=scenario,
            policy=policy,
            models=payload["models"],
            duration_sec=float(payload["duration_sec"]),
            control_interval_sec=int(payload["control_interval_sec"]),
            warmup_sec=float(payload["warmup_sec"]),
            seed=int(payload["seed"]),
            tripinfo_output=Path(payload["tripinfo_output"]),
        )
    return {
        "target": scenario,
        "policy": policy,
        "seed": int(payload["seed"]),
        "metrics": metrics,
    }


def _policy_family_from_name_v3(policy: str) -> str | None:
    return next(
        (family for family in MODEL_FAMILIES_V3 if policy.startswith(f"{family}_contrast_")),
        None,
    )


def select_target_closed_loop_policies_v3(
    models: Mapping[str, FittedContrastModelsV3],
    *,
    env_root: Path,
    policies: Sequence[str],
    seeds: Sequence[int],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    workers: int,
    tripinfo_root: Path,
    min_mean_improvement: float = 0.01,
    refinement_min_mean_improvement: float = 0.002,
    worst_seed_tolerance: float = 0.05,
    mean_p90_tolerance: float = 0.10,
    throughput_tolerance: float = 0.01,
    completion_tolerance: float = 0.005,
    departure_service_tolerance: float = 0.005,
    teleport_rate_tolerance: float = 0.0,
    collision_rate_tolerance: float = 0.0,
    scenario_sumocfgs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    selection_seeds = tuple(int(seed) for seed in seeds)
    requested_candidate_policies = tuple(
        policy
        for policy in policies
        if _policy_family_from_name_v3(policy) is not None
        and (policy.endswith("_guard") or policy.endswith("_regularized"))
    )
    candidate_policy_set = set(requested_candidate_policies)
    for policy in requested_candidate_policies:
        if _policy_family_from_name_v3(policy) == "cfcmt_mechanism":
            for dependency in (
                CFCMT_CORE_FAMILY_V3,
                CFCMT_TARGET_SPECIALIST_FAMILY_V3,
            ):
                candidate_policy_set.add(
                    policy.replace("cfcmt_mechanism", dependency, 1)
                )
    candidate_policies = tuple(sorted(candidate_policy_set))
    active_pairs = []
    diagnostics: dict[str, Any] = {target: {} for target in models}
    for target, fitted in models.items():
        for policy in candidate_policies:
            family = _policy_family_from_name_v3(policy)
            config = (
                fitted.guards[family]
                if policy.endswith("_guard")
                else fitted.regularizers[family]
            )
            hierarchy_active = False
            if (
                family == "cfcmt_mechanism"
                and CFCMT_CORE_FAMILY_V3 in fitted.family_models
            ):
                core_config = (
                    fitted.guards[CFCMT_CORE_FAMILY_V3]
                    if policy.endswith("_guard")
                    else fitted.regularizers[CFCMT_CORE_FAMILY_V3]
                )
                target_config = None
                if (
                    CFCMT_TARGET_SPECIALIST_FAMILY_V3 in fitted.family_models
                    and has_fitted_target_specialist_v3(
                        fitted.family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
                    )
                ):
                    target_config = (
                        fitted.guards[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
                        if policy.endswith("_guard")
                        else fitted.regularizers[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
                    )
                hierarchy_active = bool(
                    core_config.enabled
                    or (target_config is not None and target_config.enabled)
                )
            dependency_only_target = (
                family == CFCMT_TARGET_SPECIALIST_FAMILY_V3
                and policy not in requested_candidate_policies
                and not has_fitted_target_specialist_v3(
                    fitted.family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
                )
            )
            if (config.enabled or hierarchy_active) and not dependency_only_target:
                active_pairs.append((target, policy))
            else:
                diagnostics[target][policy] = {
                    "accepted": False,
                    "reason": "offline_target_selector_disabled",
                    "seed_results": [],
                }
    if not selection_seeds or not active_pairs:
        return {
            "enabled": bool(selection_seeds),
            "selection_seeds": list(selection_seeds),
            "active_candidate_count": len(active_pairs),
            "diagnostics": diagnostics,
            "seed_results": [],
        }

    active_targets = sorted({target for target, _ in active_pairs})
    resolved_sumocfgs = _resolve_scenario_sumocfgs(
        env_root=env_root,
        scenarios=active_targets,
        scenario_sumocfgs=scenario_sumocfgs,
    )
    selection_references = {}
    target_rollout_policies: dict[str, set[str]] = {
        target: {"selected_source_prior"} for target in active_targets
    }
    for target, policy in active_pairs:
        family = _policy_family_from_name_v3(policy)
        reference = "selected_source_prior"
        if family == "cfcmt_mechanism":
            reference = policy.replace(
                "cfcmt_mechanism", CFCMT_CORE_FAMILY_V3, 1
            )
        selection_references[(target, policy)] = reference
        target_rollout_policies[target].update((policy, reference))
    jobs = [
        {
            "scenario": target,
            "sumocfg": str(resolved_sumocfgs[target]),
            "policy": policy,
            "models": models[target],
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "tripinfo_output": str(
                tripinfo_root / f"{target}__{policy}__selection_seed{seed}.xml"
            ),
        }
        for target in active_targets
        for seed in selection_seeds
        for policy in sorted(target_rollout_policies[target])
    ]
    rows = _map_fresh_sumo_processes(_evaluate_worker, jobs, workers)
    failures = [row for row in rows if not row["metrics"].get("ok")]
    if failures:
        raise RuntimeError(f"target closed-loop policy selection failed: {failures[:3]}")
    by_key = {
        (str(row["target"]), str(row["policy"]), int(row["seed"])): row["metrics"]
        for row in rows
    }
    for target, policy in active_pairs:
        reference_policy = selection_references[(target, policy)]
        paired = []
        for seed in selection_seeds:
            baseline = by_key[(target, reference_policy, seed)]
            candidate = by_key[(target, policy, seed)]
            baseline_system_tts = float(
                baseline["mean_system_vehicles_per_controlled_lane"]
            )
            baseline_p90_queue = float(baseline["p90_queue_per_lane"])
            paired.append(
                {
                    "seed": int(seed),
                    "system_tts_delta": float(
                        candidate["mean_system_vehicles_per_controlled_lane"]
                        - baseline_system_tts
                    ),
                    "system_tts_relative_delta": float(
                        (
                            candidate["mean_system_vehicles_per_controlled_lane"]
                            - baseline_system_tts
                        )
                        / max(abs(baseline_system_tts), 1e-12)
                    ),
                    "p90_queue_delta": float(
                        candidate["p90_queue_per_lane"] - baseline_p90_queue
                    ),
                    "p90_queue_relative_delta": float(
                        (candidate["p90_queue_per_lane"] - baseline_p90_queue)
                        / max(abs(baseline_p90_queue), 1e-12)
                    ),
                    "throughput_delta": float(
                        candidate["throughput_ratio"] - baseline["throughput_ratio"]
                    ),
                    "completion_delta": float(
                        candidate["completion_ratio"] - baseline["completion_ratio"]
                    ),
                    "departure_service_delta": float(
                        candidate["departure_service_ratio"]
                        - baseline["departure_service_ratio"]
                    ),
                    "teleport_rate_delta": float(
                        candidate["teleport_rate"] - baseline["teleport_rate"]
                    ),
                    "collision_rate_delta": float(
                        candidate["collision_rate"] - baseline["collision_rate"]
                    ),
                }
            )
        mean_delta = float(
            np.mean([row["system_tts_relative_delta"] for row in paired])
        )
        worst_delta = float(
            np.max([row["system_tts_relative_delta"] for row in paired])
        )
        mean_p90_delta = float(
            np.mean([row["p90_queue_relative_delta"] for row in paired])
        )
        worst_throughput_delta = float(np.min([row["throughput_delta"] for row in paired]))
        worst_completion_delta = float(np.min([row["completion_delta"] for row in paired]))
        worst_departure_service_delta = float(
            np.min([row["departure_service_delta"] for row in paired])
        )
        worst_teleport_rate_delta = float(
            np.max([row["teleport_rate_delta"] for row in paired])
        )
        worst_collision_rate_delta = float(
            np.max([row["collision_rate_delta"] for row in paired])
        )
        required_improvement = (
            float(refinement_min_mean_improvement)
            if reference_policy != "selected_source_prior"
            else float(min_mean_improvement)
        )
        accepted = (
            mean_delta < -required_improvement
            and worst_delta <= float(worst_seed_tolerance)
            and mean_p90_delta <= float(mean_p90_tolerance)
            and worst_throughput_delta >= -float(throughput_tolerance)
            and worst_completion_delta >= -float(completion_tolerance)
            and worst_departure_service_delta >= -float(departure_service_tolerance)
            and worst_teleport_rate_delta <= float(teleport_rate_tolerance)
            and worst_collision_rate_delta <= float(collision_rate_tolerance)
        )
        diagnostics[target][policy] = {
            "accepted": bool(accepted),
            "reason": "disjoint_target_closed_loop_noninferiority"
            if accepted
            else "target_closed_loop_evidence_failed",
            "mean_relative_system_tts_delta": mean_delta,
            "worst_seed_relative_system_tts_delta": worst_delta,
            "mean_relative_p90_queue_delta": mean_p90_delta,
            "selection_reference_policy": reference_policy,
            "required_mean_relative_improvement": required_improvement,
            "worst_seed_throughput_delta": worst_throughput_delta,
            "worst_seed_completion_delta": worst_completion_delta,
            "worst_seed_departure_service_delta": worst_departure_service_delta,
            "worst_seed_teleport_rate_delta": worst_teleport_rate_delta,
            "worst_seed_collision_rate_delta": worst_collision_rate_delta,
            "seed_results": paired,
        }
        if not accepted:
            family = _policy_family_from_name_v3(policy)
            if policy.endswith("_guard"):
                models[target].guards[family] = replace(models[target].guards[family], enabled=False)
            else:
                models[target].regularizers[family] = replace(
                    models[target].regularizers[family],
                    enabled=False,
                )
    for target, policy in active_pairs:
        if _policy_family_from_name_v3(policy) != "cfcmt_mechanism":
            continue
        diagnostic = diagnostics[target][policy]
        hierarchy_mode = "guard" if policy.endswith("_guard") else "regularized"
        if diagnostic["accepted"]:
            family_config = (
                models[target].guards["cfcmt_mechanism"]
                if policy.endswith("_guard")
                else models[target].regularizers["cfcmt_mechanism"]
            )
            models[target].hierarchy_layers[hierarchy_mode] = CFCMT_HIERARCHY_LAYERS_V3
            diagnostic["effective_layer"] = "validated_adaptive_hierarchy"
            diagnostic["effective_layers"] = list(CFCMT_HIERARCHY_LAYERS_V3)
            diagnostic["mechanism_refinement_enabled"] = bool(family_config.enabled)
            continue
        core_policy = policy.replace(
            "cfcmt_mechanism", CFCMT_CORE_FAMILY_V3, 1
        )
        target_policy = policy.replace(
            "cfcmt_mechanism", CFCMT_TARGET_SPECIALIST_FAMILY_V3, 1
        )
        fallback_candidates = []
        core_diagnostic = diagnostics[target].get(core_policy, {})
        if core_diagnostic.get("accepted"):
            fallback_candidates.append(
                (
                    float(core_diagnostic["mean_relative_system_tts_delta"]),
                    "causal_core_fallback",
                    core_policy,
                )
            )
        target_diagnostic = diagnostics[target].get(target_policy, {})
        if (
            target_diagnostic.get("accepted")
            and has_fitted_target_specialist_v3(
                models[target].family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
            )
        ):
            fallback_candidates.append(
                (
                    float(target_diagnostic["mean_relative_system_tts_delta"]),
                    "target_specialist",
                    target_policy,
                )
            )
        if fallback_candidates:
            _, selected_layer, evidence_policy = min(fallback_candidates)
            models[target].hierarchy_layers[hierarchy_mode] = (selected_layer,)
            diagnostic.update(
                {
                    "accepted": True,
                    "reason": "adaptive_hierarchy_rejected_validated_child_fallback_accepted",
                    "effective_layer": selected_layer,
                    "effective_layers": [selected_layer],
                    "fallback_evidence_policy": evidence_policy,
                }
            )
        else:
            models[target].hierarchy_layers[hierarchy_mode] = ()
            diagnostic["effective_layer"] = "pressure_prior_fallback"
            diagnostic["effective_layers"] = []
    for target, fitted in models.items():
        fitted.diagnostics["cfcmt_hierarchy"]["closed_loop_enabled_layers"] = {
            mode: list(layers)
            for mode, layers in fitted.hierarchy_layers.items()
        }
        fitted.diagnostics["target_closed_loop_policy_selection"] = diagnostics[target]
    rows.sort(key=lambda row: (row["target"], row["seed"], row["policy"]))
    return {
        "enabled": True,
        "selection_seeds": list(selection_seeds),
        "active_candidate_count": len(active_pairs),
        "thresholds": {
            "min_mean_relative_improvement": float(min_mean_improvement),
            "refinement_min_mean_relative_improvement": float(
                refinement_min_mean_improvement
            ),
            "worst_seed_relative_tolerance": float(worst_seed_tolerance),
            "mean_relative_p90_tolerance": float(mean_p90_tolerance),
            "throughput_tolerance": float(throughput_tolerance),
            "completion_tolerance": float(completion_tolerance),
            "departure_service_tolerance": float(departure_service_tolerance),
            "teleport_rate_tolerance": float(teleport_rate_tolerance),
            "collision_rate_tolerance": float(collision_rate_tolerance),
        },
        "diagnostics": diagnostics,
        "seed_results": rows,
    }


def _sum_numeric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    keys = sorted({key for row in rows for key in row if not key.endswith("_rate")})
    result: dict[str, float | int] = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        result[key] = float(sum(float(value) for value in values)) if any(isinstance(value, float) for value in values) else int(sum(values))
    if int(result.get("decisions", 0)) > 0:
        decisions = max(int(result["decisions"]), 1)
        proposed = max(int(result.get("proposed_overrides", 0)), 1)
        result["agreement_rate"] = int(result.get("prior_agreements", 0)) / decisions
        result["proposal_rate"] = int(result.get("proposed_overrides", 0)) / decisions
        result["acceptance_rate"] = int(result.get("accepted_overrides", 0)) / proposed
    return result


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str],
    scenario_city_groups: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = sorted({str(row["target"]) for row in rows})
    target_rows = []
    for target in targets:
        policy_metrics = {}
        for policy in policies:
            items = [
                row["metrics"]
                for row in rows
                if row["target"] == target and row["policy"] == policy and row["metrics"].get("ok")
            ]
            metrics = {
                metric: float(np.mean([item[metric] for item in items])) if items else float("nan")
                for metric in DEFAULT_METRICS_V3
            }
            metrics["seed_count"] = len(items)
            if items and (policy.endswith("_guard") or policy.endswith("_regularized")):
                metrics["guard_audit"] = _sum_numeric([item.get("guard_audit", {}) for item in items])
            if items and policy != "fixed_program":
                metrics["phase_execution_audit"] = _sum_numeric(
                    [item.get("phase_execution_audit", {}) for item in items]
                )
            policy_metrics[policy] = metrics
        target_rows.append(
            {
                "target": target,
                "city_group": str((scenario_city_groups or {}).get(target, target)),
                "policy_metrics": policy_metrics,
            }
        )
    city_groups = sorted({str(row["city_group"]) for row in target_rows})
    aggregate = {
        policy: {
            **{
                metric: float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    target["policy_metrics"][policy][metric]
                                    for target in target_rows
                                    if target["city_group"] == city_group
                                ]
                            )
                            for city_group in city_groups
                        ]
                    )
                )
                for metric in DEFAULT_METRICS_V3
            },
            "target_count": len(target_rows),
            "city_group_count": len(city_groups),
        }
        for policy in policies
    }
    return target_rows, aggregate


def _paired_city_effects(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str],
    reference: str,
    metric: str,
) -> dict[str, Any]:
    """Compute paired relative effects before equal-city aggregation."""

    city_groups = sorted({str(row["city_group"]) for row in target_rows})
    result = {}
    for policy in policies:
        city_effects = {}
        network_effects = {}
        for row in target_rows:
            baseline = float(row["policy_metrics"][reference][metric])
            candidate = float(row["policy_metrics"][policy][metric])
            effect = (candidate - baseline) / max(abs(baseline), 1e-12)
            network_effects[str(row["target"])] = float(effect)
        for city_group in city_groups:
            values = [
                network_effects[str(row["target"])]
                for row in target_rows
                if str(row["city_group"]) == city_group
            ]
            city_effects[city_group] = float(np.mean(values))
        result[policy] = {
            "reference": reference,
            "metric": metric,
            "mean_relative_effect": float(np.mean(list(city_effects.values()))),
            "median_relative_effect": float(np.median(list(city_effects.values()))),
            "win_city_groups": int(sum(value < 0.0 for value in city_effects.values())),
            "city_group_count": len(city_groups),
            "city_relative_effect": city_effects,
            "network_relative_effect": network_effects,
        }
    return result


def run_leave_one_network_out_v3(
    *,
    env_root: Path = DEFAULT_ENV_ROOT,
    scenarios: Sequence[str] = EXTENDED_SCENARIOS,
    policies: Sequence[str] = DEFAULT_POLICIES_V3,
    seeds: Sequence[int] = (131, 337, 911),
    source_duration_sec: float = 600.0,
    duration_sec: float = 600.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    source_seed: int = 2027,
    source_seeds: Sequence[int] | None = None,
    source_policy_seeds: Sequence[int] = (1709, 2027),
    max_focal_tls: int = 4,
    counterfactual_horizon_intervals: int = 3,
    collection_shards: int = 1,
    workers: int = 8,
    tripinfo_root: Path = Path("cf_h2o/results/tripinfo/traffic_signal_resco_cfcmt_v3"),
    fit_config: MechanismFitConfig | None = None,
    source_rule_specs: Sequence[PressurePolicySpec] | None = None,
    cache_root: Path | None = DEFAULT_CACHE_ROOT_V3,
    source_rule_cache_root: Path | None = None,
    target_group_budget: int = 0,
    target_adaptation_selection_seed: int = 20260803,
    target_calibration_seeds: Sequence[int] = (4047,),
    min_target_calibration_groups: int = 4,
    target_calibration_fraction: float = 0.40,
    target_guard_selection_mode: str = "strict_worst_domain",
    source_selector_min_context_domains: int = 5,
    source_behavior_policy: str = "phase_pressure",
    target_closed_loop_selection_seeds: Sequence[int] = (),
    target_closed_loop_selection_duration_sec: float | None = None,
    target_closed_loop_min_improvement: float = 0.01,
    target_closed_loop_refinement_min_improvement: float = 0.002,
    target_closed_loop_worst_tolerance: float = 0.05,
    min_source_tls_coverage: float = 0.0,
    scenario_sumocfgs: Mapping[str, Path] | None = None,
    scenario_city_groups: Mapping[str, str] | None = None,
    benchmark_manifest: Mapping[str, Any] | None = None,
    primary_reference_policy: str | None = None,
) -> dict[str, Any]:
    # Process-level parallelism owns the CPU budget. HGB/BLAS threads inside
    # each spawned worker otherwise multiply the requested worker count.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    scenarios = tuple(scenarios)
    resolved_sumocfgs = _resolve_scenario_sumocfgs(
        env_root=env_root,
        scenarios=scenarios,
        scenario_sumocfgs=scenario_sumocfgs,
    )
    resolved_city_groups = _resolve_city_groups(scenarios, scenario_city_groups)
    policies = tuple(policies)
    primary_reference = _primary_reference_policy_v3(
        policies,
        declared=primary_reference_policy,
    )
    seeds = tuple(int(seed) for seed in seeds)
    if int(target_group_budget) < 0:
        raise ValueError("target_group_budget must be nonnegative")
    target_calibration_seeds = tuple(int(seed) for seed in target_calibration_seeds)
    if int(min_target_calibration_groups) < 1:
        raise ValueError("min_target_calibration_groups must be positive")
    if not 0.0 <= float(target_calibration_fraction) < 1.0:
        raise ValueError("target_calibration_fraction must be in [0, 1)")
    if target_guard_selection_mode not in {"strict_worst_domain", "mean_ucb"}:
        raise ValueError("unknown target_guard_selection_mode")
    if source_behavior_policy not in {"phase_pressure", "exploratory_mixture"}:
        raise ValueError("unknown source_behavior_policy")
    if int(source_selector_min_context_domains) < 2:
        raise ValueError("source_selector_min_context_domains must be at least two")
    if not 0.0 <= float(min_source_tls_coverage) <= 1.0:
        raise ValueError("min_source_tls_coverage must be in [0, 1]")
    target_closed_loop_selection_seeds = tuple(
        int(seed) for seed in target_closed_loop_selection_seeds
    )
    overlap = set(target_closed_loop_selection_seeds) & set(seeds)
    if overlap:
        raise ValueError(
            f"target closed-loop selection seeds overlap evaluation seeds: {sorted(overlap)}"
        )
    fit_config = fit_config or MechanismFitConfig()
    source_rule_specs = tuple(source_rule_specs or generalized_pressure_grid())
    source_rule_spec_map = {spec.key: spec for spec in source_rule_specs}
    requested_model_families = tuple(
        family
        for family in MODEL_FAMILIES_V3
        if any(policy.startswith(f"{family}_contrast_") for policy in policies)
    )
    model_family_set = set(requested_model_families)
    if "cfcmt_mechanism" in model_family_set:
        model_family_set.update(
            (
                CFCMT_CORE_FAMILY_V3,
                CFCMT_TARGET_SPECIALIST_FAMILY_V3,
            )
        )
    model_families = tuple(
        family for family in MODEL_FAMILIES_V3 if family in model_family_set
    )
    source_seeds = tuple(int(seed) for seed in (source_seeds or (source_seed,)))
    bank = collect_counterfactual_bank_v3(
        env_root=env_root,
        scenarios=scenarios,
        duration_sec=source_duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seeds=source_seeds,
        max_focal_tls=max_focal_tls,
        counterfactual_horizon_intervals=counterfactual_horizon_intervals,
        workers=workers,
        cache_root=cache_root,
        behavior_policy=source_behavior_policy,
        scenario_sumocfgs=resolved_sumocfgs,
        collection_shards=collection_shards,
    )
    insufficient_coverage = {
        name: float(dataset.metadata.get("tls_coverage_fraction", 0.0))
        for name, dataset in bank.items()
        if float(dataset.metadata.get("tls_coverage_fraction", 0.0))
        + 1e-12
        < float(min_source_tls_coverage)
    }
    if insufficient_coverage:
        raise ValueError(
            "source counterfactual TLS coverage is below the requested floor: "
            f"{insufficient_coverage}"
        )
    source_rule_costs, source_rule_rows = collect_source_rule_costs_v3(
        env_root=env_root,
        scenarios=scenarios,
        seeds=tuple(int(seed) for seed in source_policy_seeds),
        duration_sec=source_duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        workers=workers,
        policy_specs=source_rule_specs,
        scenario_sumocfgs=resolved_sumocfgs,
        source_rule_cache_root=source_rule_cache_root,
        tripinfo_root=tripinfo_root / "source_rules",
    )
    models = fit_target_models_parallel_v3(
        bank,
        fit_config=fit_config,
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_spec_map,
        model_families=model_families,
        target_group_budget=target_group_budget,
        target_adaptation_selection_seed=target_adaptation_selection_seed,
        target_calibration_seeds=target_calibration_seeds,
        min_target_calibration_groups=min_target_calibration_groups,
        target_calibration_fraction=target_calibration_fraction,
        target_guard_selection_mode=target_guard_selection_mode,
        source_selector_min_context_domains=source_selector_min_context_domains,
        workers=workers,
        scenario_city_groups=resolved_city_groups,
    )
    closed_loop_selection = (
        select_target_closed_loop_policies_v3(
            models,
            env_root=env_root,
            policies=policies,
            seeds=target_closed_loop_selection_seeds,
            duration_sec=float(target_closed_loop_selection_duration_sec or duration_sec),
            control_interval_sec=control_interval_sec,
            warmup_sec=warmup_sec,
            workers=workers,
            tripinfo_root=tripinfo_root / "target_policy_selection",
            min_mean_improvement=target_closed_loop_min_improvement,
            refinement_min_mean_improvement=(
                target_closed_loop_refinement_min_improvement
            ),
            worst_seed_tolerance=target_closed_loop_worst_tolerance,
            scenario_sumocfgs=resolved_sumocfgs,
        )
        if int(target_group_budget) > 0 and target_closed_loop_selection_seeds
        else {
            "enabled": False,
            "reason": "zero_shot_or_no_target_closed_loop_selection_seeds",
            "selection_seeds": list(target_closed_loop_selection_seeds),
            "diagnostics": {},
            "seed_results": [],
        }
    )

    jobs = [
        {
            "scenario": target,
            "sumocfg": str(resolved_sumocfgs[target]),
            "policy": policy,
            "models": models[target],
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "tripinfo_output": str(tripinfo_root / f"{target}__{policy}__seed{seed}.xml"),
        }
        for target in scenarios
        for seed in seeds
        for policy in policies
    ]
    rows = _map_fresh_sumo_processes(_evaluate_worker, jobs, workers)
    rows.sort(key=lambda row: (row["target"], row["seed"], row["policy"]))
    target_rows, aggregate = _aggregate_rows(
        rows,
        policies=policies,
        scenario_city_groups=resolved_city_groups,
    )
    effect_reference = "phase_pressure" if "phase_pressure" in policies else None
    paired_city_effects = (
        _paired_city_effects(
            target_rows,
            policies=policies,
            reference=effect_reference,
            metric="mean_system_vehicles_per_controlled_lane",
        )
        if effect_reference is not None
        else {}
    )
    for policy, effect in paired_city_effects.items():
        aggregate[policy]["paired_city_mean_relative_system_tts"] = float(
            effect["mean_relative_effect"]
        )
    bootstrap = (
        [
            paired_hierarchical_bootstrap(
                rows,
                reference=primary_reference,
                baseline=baseline,
                metric="mean_system_vehicles_per_controlled_lane",
                effect="relative",
                clusters=resolved_city_groups,
            )
            for baseline in policies
            if baseline != primary_reference
        ]
        if primary_reference is not None
        else []
    )
    return {
        "experiment": "traffic_signal_cfcmt_v4_action_contrast_strict_leave_one_city_out",
        "runtime": _runtime_metadata(),
        "setting": {
            "env_root": str(env_root),
            "scenario_sumocfgs": {
                scenario: str(path) for scenario, path in resolved_sumocfgs.items()
            },
            "scenario_city_groups": dict(resolved_city_groups),
            "city_group_count": len(set(resolved_city_groups.values())),
            "fold_protocol": "strict_leave_one_city_group_out",
            "benchmark_manifest": dict(benchmark_manifest or {}),
            "scenarios": list(scenarios),
            "policies": list(policies),
            "seeds": list(seeds),
            "source_duration_sec": float(source_duration_sec),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "source_seed": int(source_seed),
            "source_seeds": list(source_seeds),
            "source_policy_seeds": [int(seed) for seed in source_policy_seeds],
            "max_focal_tls": int(max_focal_tls),
            "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
            "counterfactual_horizon_sec": int(control_interval_sec)
            * max(int(counterfactual_horizon_intervals), 1),
            "collection_shards": int(collection_shards),
            "workers": int(workers),
            "model_fit_process_workers": min(max(int(workers), 1), len(scenarios)),
            "model_fit_source_loo_workers_per_target": max(
                int(workers) // min(max(int(workers), 1), len(scenarios)),
                1,
            ),
            "fit_config": asdict(fit_config),
            "source_rule_specs": [spec.to_dict() for spec in source_rule_specs],
            "source_rule_cache_version": SOURCE_RULE_CACHE_VERSION,
            "source_rule_cache_root": (
                str(source_rule_cache_root)
                if source_rule_cache_root is not None
                else None
            ),
            "fitted_model_families": list(model_families),
            "counterfactual_cache_root": str(cache_root) if cache_root is not None else None,
            "target_group_budget": int(target_group_budget),
            "target_adaptation_selection_seed": int(target_adaptation_selection_seed),
            "target_calibration_seeds": list(target_calibration_seeds),
            "min_target_calibration_groups": int(min_target_calibration_groups),
            "target_calibration_fraction": float(target_calibration_fraction),
            "target_guard_selection_mode": str(target_guard_selection_mode),
            "target_deployment_fit_protocol": (
                TARGET_DEPLOYMENT_FIT_PROTOCOL_V3
            ),
            "action_target_scale_protocol": ACTION_TARGET_SCALE_PROTOCOL,
            "cfcmt_hierarchy_protocol": CFCMT_HIERARCHY_PROTOCOL_V3,
            "target_group_selection_protocol": (
                TARGET_GROUP_SELECTION_PROTOCOL_V3
            ),
            "source_selector_min_context_domains": int(
                source_selector_min_context_domains
            ),
            "target_information_budget": "static_network_summary_plus_online_state_no_target_transition_labels"
            if int(target_group_budget) <= 0
            else "static_network_summary_plus_matched_target_simulator_counterfactual_groups",
            "source_supervision": (
                "matched_finite_horizon_sumo_counterfactuals_on_symmetric_safe_support"
            ),
            "source_behavior_policy": source_behavior_policy,
            "target_closed_loop_selection_seeds": list(target_closed_loop_selection_seeds),
            "target_closed_loop_selection_duration_sec": float(
                target_closed_loop_selection_duration_sec or duration_sec
            ),
            "target_closed_loop_min_improvement": float(target_closed_loop_min_improvement),
            "target_closed_loop_refinement_min_improvement": float(
                target_closed_loop_refinement_min_improvement
            ),
            "target_closed_loop_worst_tolerance": float(target_closed_loop_worst_tolerance),
            "min_source_tls_coverage": float(min_source_tls_coverage),
            "signal_protocol": (
                "safe_min_green_yellow_all_red_full_junction_occupancy_clearance_v4"
            ),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "sumo_process_isolation": "spawn_maxtasksperchild_1",
            "primary_reference_policy": primary_reference,
            "primary_reference_policy_source": (
                "explicit_cli"
                if primary_reference_policy is not None
                else "legacy_priority_fallback"
            ),
            "primary_performance_metric": "mean_system_vehicles_per_controlled_lane",
            "cross_network_effect": "paired_relative_change",
        },
        "source_counterfactual_diagnostics": {
            name: {
                "rows": dataset.size,
                "groups": len(set(dataset.metadata["action_group_ids"])),
                "branches": dataset.metadata["counterfactual_branches"],
                "restores": dataset.metadata["state_restores"],
                "phase_execution_audit": dataset.metadata["phase_execution_audit"],
                "counterfactual_cost_scope": dataset.metadata.get("counterfactual_cost_scope"),
                "state_snapshot_protocol": dataset.metadata.get("state_snapshot_protocol"),
                "strict_safety_monitoring": dataset.metadata.get(
                    "strict_safety_monitoring"
                ),
                "behavior_safety_audit": dataset.metadata.get(
                    "behavior_safety_audit"
                ),
                "counterfactual_safety_audit": dataset.metadata.get(
                    "counterfactual_safety_audit"
                ),
                "behavior_trace_sha256_by_seed": dataset.metadata.get(
                    "behavior_trace_sha256_by_seed",
                    {
                        str(dataset.metadata.get("seed")): dataset.metadata.get(
                            "behavior_trace_sha256"
                        )
                    },
                ),
                "behavior_interval_count_by_seed": dataset.metadata.get(
                    "behavior_interval_count_by_seed",
                    {
                        str(dataset.metadata.get("seed")): int(
                            dataset.metadata.get("behavior_interval_count", 0)
                        )
                    },
                ),
                "eligible_collection_opportunities_by_seed": dataset.metadata.get(
                    "eligible_collection_opportunities_by_seed",
                    {
                        str(dataset.metadata.get("seed")): int(
                            dataset.metadata.get("eligible_collection_opportunities", 0)
                        )
                    },
                ),
                "counterfactual_replay_audit": dataset.metadata.get(
                    "counterfactual_replay_audit"
                ),
                "controllable_tls_count": int(dataset.metadata.get("controllable_tls_count", 0)),
                "covered_tls_count": int(dataset.metadata.get("covered_tls_count", 0)),
                "tls_coverage_fraction": float(dataset.metadata.get("tls_coverage_fraction", 0.0)),
            }
            for name, dataset in bank.items()
        },
        "source_rule_policy_costs": source_rule_costs,
        "source_rule_seed_results": source_rule_rows,
        "model_diagnostics": {target: model.diagnostics for target, model in models.items()},
        "target_closed_loop_policy_selection": closed_loop_selection,
        "selected_prior_policy": {target: model.prior_policy for target, model in models.items()},
        "selected_prior_spec": {target: model.prior_spec.to_dict() for target, model in models.items()},
        "guard_configs": {
            target: {family: asdict(config) for family, config in model.guards.items()}
            for target, model in models.items()
        },
        "prior_regularization_configs": {
            target: {family: asdict(config) for family, config in model.regularizers.items()}
            for target, model in models.items()
        },
        "aggregate": aggregate,
        "paired_city_effects": paired_city_effects,
        "targets": target_rows,
        "hierarchical_bootstrap": bootstrap,
        "seed_results": rows,
    }


def write_markdown_v3(result: Mapping[str, Any], path: Path) -> None:
    policies = result["setting"]["policies"]
    target_group_budget = int(result["setting"].get("target_group_budget", 0))
    information_statement = (
        "Each target fold uses static summaries and online state only; no target transition labels are used."
        if target_group_budget <= 0
        else f"Each target fold additionally uses {target_group_budget} matched target-simulator action groups for adaptation."
    )
    lines = [
        "# CFCMT V3 Action-Contrast RESCO Benchmark",
        "",
        f"Source labels are matched safe one-step counterfactuals. {information_statement}",
        "",
        "| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in sorted(
        policies,
        key=lambda name: result["aggregate"][name]["mean_system_vehicles_per_controlled_lane"],
    ):
        metrics = result["aggregate"][policy]
        lines.append(
            f"| {policy} | {metrics['mean_system_vehicles_per_controlled_lane']:.4f} | "
            f"{metrics['mean_queue_per_lane']:.4f} | {metrics['p90_queue_per_lane']:.4f} | "
            f"{metrics['completion_ratio']:.4f} |"
        )
    lines.extend(["", "## Selected Rule Prior", "", "| Target | Prior |", "|---|---|"])
    for target, policy in result["selected_prior_policy"].items():
        lines.append(f"| {target} | {policy} |")
    lines.extend(
        [
            "",
            "## Hierarchical Bootstrap",
            "",
            "| Baseline | Relative delta | 95% CI | Wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["hierarchical_bootstrap"]:
        lines.append(
            f"| {row['baseline']} | {row.get('mean_delta', float('nan')):.4f} | "
            f"[{row.get('ci_low', float('nan')):.4f}, {row.get('ci_high', float('nan')):.4f}] | "
            f"{row.get('win_networks', 0)}/{row.get('network_count', 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES_V3))
    parser.add_argument("--primary-reference-policy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[131, 337, 911])
    parser.add_argument("--source-duration-sec", type=float, default=600.0)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--source-seed", type=int, default=2027)
    parser.add_argument("--source-seeds", nargs="+", type=int)
    parser.add_argument("--source-policy-seeds", nargs="+", type=int, default=[1709, 2027])
    parser.add_argument(
        "--source-rule-grid",
        choices=("generalized", "legacy", "fixed_max", "fixed_phase"),
        default="generalized",
    )
    parser.add_argument("--max-focal-tls", type=int, default=4)
    parser.add_argument("--counterfactual-horizon-intervals", type=int, default=3)
    parser.add_argument("--collection-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT_V3)
    parser.add_argument("--source-rule-cache-root", type=Path)
    parser.add_argument(
        "--tripinfo-root",
        type=Path,
        default=Path("cf_h2o/results/tripinfo/traffic_signal_resco_cfcmt_v3"),
    )
    parser.add_argument("--target-group-budget", type=int, default=0)
    parser.add_argument("--target-adaptation-selection-seed", type=int, default=20260803)
    parser.add_argument("--target-calibration-seeds", nargs="+", type=int, default=[4047])
    parser.add_argument("--min-target-calibration-groups", type=int, default=4)
    parser.add_argument("--target-calibration-fraction", type=float, default=0.40)
    parser.add_argument(
        "--target-guard-selection-mode",
        choices=("strict_worst_domain", "mean_ucb"),
        default="strict_worst_domain",
    )
    parser.add_argument("--source-selector-min-context-domains", type=int, default=5)
    parser.add_argument(
        "--source-behavior-policy",
        choices=("phase_pressure", "exploratory_mixture"),
        default="phase_pressure",
    )
    parser.add_argument("--target-closed-loop-selection-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--target-closed-loop-selection-duration-sec", type=float)
    parser.add_argument("--target-closed-loop-min-improvement", type=float, default=0.01)
    parser.add_argument(
        "--target-closed-loop-refinement-min-improvement",
        type=float,
        default=0.002,
    )
    parser.add_argument("--target-closed-loop-worst-tolerance", type=float, default=0.05)
    parser.add_argument("--min-source-tls-coverage", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_cfcmt_v3_validation.json"))
    parser.add_argument("--md-out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_cfcmt_v3_validation.md"))
    args = parser.parse_args()
    manifest = load_traffic_signal_manifest(args.manifest) if args.manifest else None
    if manifest is not None:
        available = manifest.sumocfgs
        scenarios = tuple(args.scenarios or available)
        unknown = set(scenarios) - set(available)
        if unknown:
            parser.error(f"scenarios are absent from manifest: {sorted(unknown)}")
        scenario_sumocfgs = {name: available[name] for name in scenarios}
        scenario_city_groups = {name: manifest.city_groups[name] for name in scenarios}
        manifest_metadata = manifest.to_dict()
    else:
        scenarios = tuple(args.scenarios or EXTENDED_SCENARIOS)
        scenario_sumocfgs = None
        scenario_city_groups = None
        manifest_metadata = None
    source_rule_specs = {
        "generalized": generalized_pressure_grid(),
        "legacy": (MAX_PRESSURE_SPEC, PHASE_PRESSURE_SPEC, SPILLBACK_PRESSURE_SPEC),
        "fixed_max": (MAX_PRESSURE_SPEC,),
        "fixed_phase": (PHASE_PRESSURE_SPEC,),
    }[args.source_rule_grid]
    result = run_leave_one_network_out_v3(
        env_root=args.env_root,
        scenarios=scenarios,
        policies=args.policies,
        seeds=args.seeds,
        source_duration_sec=args.source_duration_sec,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        source_seed=args.source_seed,
        source_seeds=args.source_seeds,
        source_policy_seeds=args.source_policy_seeds,
        max_focal_tls=args.max_focal_tls,
        counterfactual_horizon_intervals=args.counterfactual_horizon_intervals,
        collection_shards=args.collection_shards,
        workers=args.workers,
        cache_root=args.cache_root,
        source_rule_cache_root=args.source_rule_cache_root,
        tripinfo_root=args.tripinfo_root,
        source_rule_specs=source_rule_specs,
        target_group_budget=args.target_group_budget,
        target_adaptation_selection_seed=args.target_adaptation_selection_seed,
        target_calibration_seeds=args.target_calibration_seeds,
        min_target_calibration_groups=args.min_target_calibration_groups,
        target_calibration_fraction=args.target_calibration_fraction,
        target_guard_selection_mode=args.target_guard_selection_mode,
        source_selector_min_context_domains=args.source_selector_min_context_domains,
        source_behavior_policy=args.source_behavior_policy,
        target_closed_loop_selection_seeds=args.target_closed_loop_selection_seeds,
        target_closed_loop_selection_duration_sec=args.target_closed_loop_selection_duration_sec,
        target_closed_loop_min_improvement=args.target_closed_loop_min_improvement,
        target_closed_loop_refinement_min_improvement=(
            args.target_closed_loop_refinement_min_improvement
        ),
        target_closed_loop_worst_tolerance=args.target_closed_loop_worst_tolerance,
        min_source_tls_coverage=args.min_source_tls_coverage,
        scenario_sumocfgs=scenario_sumocfgs,
        scenario_city_groups=scenario_city_groups,
        benchmark_manifest=manifest_metadata,
        primary_reference_policy=args.primary_reference_policy,
    )
    atomic_write_json(args.out, _json_ready(result))
    write_markdown_v3(result, args.md_out)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
