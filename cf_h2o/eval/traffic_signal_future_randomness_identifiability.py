"""Separate fixed traffic states from future SUMO randomness.

The historical counterfactual caches save SUMO RNG state so that matched action
branches are exactly reproducible.  This diagnostic deliberately omits RNG state
from a small set of snapshots, reloads each snapshot under several future seeds,
and estimates the conditional action-cost difference at a fixed decision state.
It does not replace or mutate any historical cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    _advance_with_actions,
    _candidate_for_state,
    _counterfactual_branch_outcome_v3,
    _counterfactual_outcome_vector,
    _controlled_lane_set,
    _new_safety_ledger,
    _read_graph_states_v3,
    _restore_executors,
    _rule_candidate,
    _scenario_context_v3,
    _tls_phase_infos,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    _reload_sumo_state,
    _start_sumo,
)
from cf_h2o.sumo_runtime import (
    libsumo_version,
    load_libsumo,
    runtime_metadata,
    sumo_state_directory,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.dynamic_graph_context import build_signal_routing_graph
from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors


RESULT_PROTOCOL = "tsc-v150f-fixed-state-future-randomness-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150f-fixed-state-future-randomness-aggregate-v1"
SNAPSHOT_PROTOCOL = {
    "save_rng": False,
    "precision": 8,
    "thread_rngs": 1,
    "reload_mode": "full_network_load_with_new_future_seed",
    "behavior_capture": "uninterrupted_first_pass",
    "action_pairing": "common_future_seed_within_fixed_state",
}
MINIMUM_VALID_FUTURES_PER_CHECKPOINT = 3


def _snapshot_contains_rng_state(path: Path) -> bool:
    for _, element in ET.iterparse(path, events=("start",)):
        if str(element.tag).rsplit("}", 1)[-1] == "rngState":
            return True
    return False


def _action_pair(
    *, state: Any, executor: Any, info: Any
) -> tuple[Any, Any] | None:
    feasible = tuple(str(value) for value in executor.feasible_states_now())
    if len(feasible) < 2:
        return None
    reference = _rule_candidate(state, executor, "phase_pressure")
    alternatives = tuple(value for value in feasible if value != reference.state)
    if not alternatives:
        return None
    alternative = min(
        (_candidate_for_state(info, value) for value in alternatives),
        key=lambda candidate: (
            abs(int(candidate.green_count) - int(reference.green_count)),
            int(candidate.phase_index),
            str(candidate.state),
        ),
    )
    return reference, alternative


def summarize_fixed_state_records(
    records: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-8
) -> dict[str, Any]:
    if not records:
        raise ValueError("future-randomness summary requires records")
    by_checkpoint: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        checkpoint = int(record["checkpoint_index"])
        by_checkpoint.setdefault(checkpoint, []).append(record)

    checkpoint_rows: list[dict[str, Any]] = []
    within_variances: list[float] = []
    checkpoint_means: list[float] = []
    for checkpoint, rows in sorted(by_checkpoint.items()):
        valid = [row for row in rows if bool(row.get("valid", False))]
        utilities = np.asarray(
            [float(row["reference_minus_alternative_utility"]) for row in valid],
            dtype=float,
        )
        reference_costs = np.asarray(
            [float(row["reference_cumulative_cost"]) for row in valid], dtype=float
        )
        alternative_costs = np.asarray(
            [float(row["alternative_cumulative_cost"]) for row in valid], dtype=float
        )
        if utilities.size:
            mean_utility = float(np.mean(utilities))
            utility_variance = float(np.var(utilities, ddof=1)) if utilities.size > 1 else 0.0
            utility_range = float(np.ptp(utilities))
            within_variances.append(utility_variance)
            checkpoint_means.append(mean_utility)
            sign = np.sign(mean_utility)
            sign_agreement = float(
                np.mean(np.sign(utilities) == sign) if sign else np.mean(utilities == 0.0)
            )
            distinct_outcomes = len(
                {
                    (
                        tuple(np.round(row["reference_outcome_vector"], 8)),
                        tuple(np.round(row["alternative_outcome_vector"], 8)),
                    )
                    for row in valid
                }
            )
        else:
            mean_utility = None
            utility_variance = None
            utility_range = None
            sign_agreement = None
            distinct_outcomes = 0
        checkpoint_rows.append(
            {
                "checkpoint_index": checkpoint,
                "future_seed_count": len(rows),
                "valid_pair_count": len(valid),
                "mean_reference_minus_alternative_utility": mean_utility,
                "utility_variance_across_future_seeds": utility_variance,
                "utility_range_across_future_seeds": utility_range,
                "utility_sign_agreement_fraction": sign_agreement,
                "reference_cost_range": (
                    float(np.ptp(reference_costs)) if reference_costs.size else None
                ),
                "alternative_cost_range": (
                    float(np.ptp(alternative_costs)) if alternative_costs.size else None
                ),
                "distinct_paired_outcome_count": int(distinct_outcomes),
                "future_seed_changes_outcome": bool(distinct_outcomes > 1),
                "future_seed_changes_cost_label": bool(
                    utilities.size
                    and max(
                        float(np.ptp(reference_costs)),
                        float(np.ptp(alternative_costs)),
                        float(np.ptp(utilities)),
                    )
                    > float(tolerance)
                ),
            }
        )

    within = float(np.mean(within_variances)) if within_variances else 0.0
    between = (
        float(np.var(np.asarray(checkpoint_means, dtype=float), ddof=1))
        if len(checkpoint_means) > 1
        else 0.0
    )
    total = within + between
    replay_checks = [row for row in records if "replay_max_abs_difference" in row]
    replay_max = max(
        (float(row["replay_max_abs_difference"]) for row in replay_checks),
        default=float("inf"),
    )
    return {
        "checkpoint_count": len(by_checkpoint),
        "valid_checkpoint_count": sum(
            row["valid_pair_count"] > 0 for row in checkpoint_rows
        ),
        "valid_future_pair_count": sum(
            int(row["valid_pair_count"]) for row in checkpoint_rows
        ),
        "future_randomness_effective_checkpoint_count": sum(
            bool(row["future_seed_changes_outcome"]) for row in checkpoint_rows
        ),
        "future_randomness_effective": any(
            bool(row["future_seed_changes_outcome"]) for row in checkpoint_rows
        ),
        "future_randomness_cost_label_effective_checkpoint_count": sum(
            bool(row["future_seed_changes_cost_label"]) for row in checkpoint_rows
        ),
        "future_randomness_cost_label_effective": any(
            bool(row["future_seed_changes_cost_label"]) for row in checkpoint_rows
        ),
        "same_seed_replay_check_count": len(replay_checks),
        "same_seed_replay_max_abs_difference": replay_max,
        "same_seed_replay_passed": bool(
            replay_checks and replay_max <= float(tolerance)
        ),
        "utility_variance_decomposition": {
            "mean_within_state_future_variance": within,
            "between_state_mean_variance": between,
            "within_state_fraction": float(within / total) if total > 0.0 else 0.0,
        },
        "checkpoints": checkpoint_rows,
    }


def run_scenario(
    *,
    manifest_path: Path,
    target_city: str,
    scenario: str,
    base_seed: int,
    future_seeds: Sequence[int],
    base_trace_duration_sec: float,
    warmup_sec: float,
    checkpoint_count: int,
    checkpoint_spacing_sec: float,
    control_interval_sec: int,
    counterfactual_horizon_intervals: int,
    counterfactual_cost_mode: str,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    specs = {item.scenario: item for item in manifest.scenarios}
    if scenario not in specs or specs[scenario].city_group != str(target_city):
        raise ValueError("future-randomness scenario/city assignment changed")
    seeds = tuple(int(value) for value in future_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique future seeds are required")
    if int(checkpoint_count) < 1 or int(control_interval_sec) < 1:
        raise ValueError("checkpoint count and control interval must be positive")

    sumo_api = load_libsumo()
    sumocfg = specs[scenario].sumocfg
    records: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    snapshot_rng_tags: list[bool] = []
    _start_sumo(
        sumo_api,
        sumocfg,
        int(base_seed),
        save_state_rng=False,
    )
    try:
        infos = _tls_phase_infos(sumo_api)
        routing_graph = build_signal_routing_graph(sumo_api, infos)
        context = _scenario_context_v3(
            sumocfg=sumocfg,
            infos=infos,
            control_interval_sec=int(control_interval_sec),
        )
        executors = build_safe_phase_executors(sumo_api, infos)
        controlled_lanes = tuple(sorted(_controlled_lane_set(infos)))
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(base_trace_duration_sec)
        next_checkpoint_time = begin_time + float(warmup_sec)
        interval_index = 0
        with sumo_state_directory(prefix=f"cfcmt_v150f_{scenario}_") as state_dir:
            while (
                float(sumo_api.simulation.getTime()) < end_time
                and int(sumo_api.simulation.getMinExpectedNumber()) > 0
                and len(snapshots) < int(checkpoint_count)
            ):
                states = _read_graph_states_v3(
                    sumo_api, infos, context, routing_graph
                )
                reference_actions = {
                    tls_id: _rule_candidate(
                        states[tls_id], executors[tls_id], "phase_pressure"
                    )
                    for tls_id in sorted(infos)
                }
                now = float(sumo_api.simulation.getTime())
                if now + 1e-9 >= next_checkpoint_time:
                    eligible = []
                    for tls_id in sorted(infos):
                        pair = _action_pair(
                            state=states[tls_id],
                            executor=executors[tls_id],
                            info=infos[tls_id],
                        )
                        if pair is not None:
                            eligible.append((tls_id, pair))
                    if eligible:
                        focal_id, pair = max(
                            eligible,
                            key=lambda item: (
                                float(sum(states[item[0]].q_by_lane.values())),
                                str(item[0]),
                            ),
                        )
                        state_path = state_dir / f"checkpoint_{len(snapshots):03d}.xml"
                        sumo_api.simulation.saveState(str(state_path))
                        has_rng = _snapshot_contains_rng_state(state_path)
                        snapshot_rng_tags.append(has_rng)
                        snapshots.append(
                            {
                                "checkpoint_index": len(snapshots),
                                "time_sec": now,
                                "state_path": state_path,
                                "focal_id": focal_id,
                                "states": states,
                                "executor_snapshots": {
                                    tls_id: executor.snapshot()
                                    for tls_id, executor in executors.items()
                                },
                                "reference": pair[0],
                                "alternative": pair[1],
                            }
                        )
                        checkpoint_metadata.append(
                            {
                                "checkpoint_index": len(snapshots) - 1,
                                "time_sec": now,
                                "interval_index": interval_index,
                                "focal_tls": focal_id,
                                "reference_state": pair[0].state,
                                "alternative_state": pair[1].state,
                                "snapshot_contains_rng_state": has_rng,
                            }
                        )
                        next_checkpoint_time = now + float(checkpoint_spacing_sec)
                remaining = max(
                    int(np.ceil(end_time - float(sumo_api.simulation.getTime()))), 0
                )
                if remaining <= 0:
                    break
                _advance_with_actions(
                    sumo_api=sumo_api,
                    executors=executors,
                    actions={
                        tls_id: candidate.state
                        for tls_id, candidate in reference_actions.items()
                    },
                    control_interval_sec=min(int(control_interval_sec), remaining),
                    collision_mode="audit",
                    safety_ledger=_new_safety_ledger(),
                )
                interval_index += 1

            if len(snapshots) != int(checkpoint_count):
                raise RuntimeError(
                    f"captured {len(snapshots)} of {int(checkpoint_count)} requested "
                    f"fixed-state checkpoints for {scenario}"
                )
            if any(snapshot_rng_tags):
                raise RuntimeError("RNG-free SUMO snapshot unexpectedly contains rngState")

            for snapshot in snapshots:
                for seed_index, future_seed in enumerate(seeds):
                    action_outcomes: dict[str, np.ndarray] = {}
                    action_costs: dict[str, float] = {}
                    valid = True
                    failure: str | None = None
                    failure_action: str | None = None
                    replay_difference: float | None = None
                    for action_name, candidate in (
                        ("reference", snapshot["reference"]),
                        ("alternative", snapshot["alternative"]),
                    ):
                        try:
                            _reload_sumo_state(
                                sumo_api,
                                sumocfg,
                                int(future_seed),
                                snapshot["state_path"],
                                begin_time=float(snapshot["time_sec"]),
                                save_state_rng=False,
                            )
                            _restore_executors(
                                executors, snapshot["executor_snapshots"]
                            )
                            costs, one_step, terminal = _counterfactual_branch_outcome_v3(
                                sumo_api=sumo_api,
                                infos=infos,
                                states=snapshot["states"],
                                executors=executors,
                                focal_id=str(snapshot["focal_id"]),
                                candidate=candidate,
                                context=context,
                                control_interval_sec=int(control_interval_sec),
                                counterfactual_horizon_intervals=int(
                                    counterfactual_horizon_intervals
                                ),
                                controlled_lane_count=max(len(controlled_lanes), 1),
                                controlled_lanes=controlled_lanes,
                                counterfactual_cost_mode=str(
                                    counterfactual_cost_mode
                                ),
                                safety_ledger=_new_safety_ledger(),
                            )
                            vector = _counterfactual_outcome_vector(
                                costs, one_step, terminal
                            )
                            action_outcomes[action_name] = vector
                            action_costs[action_name] = float(np.sum(costs))
                            if action_name == "reference" and seed_index == 0:
                                _reload_sumo_state(
                                    sumo_api,
                                    sumocfg,
                                    int(future_seed),
                                    snapshot["state_path"],
                                    begin_time=float(snapshot["time_sec"]),
                                    save_state_rng=False,
                                )
                                _restore_executors(
                                    executors, snapshot["executor_snapshots"]
                                )
                                replay_costs, replay_one, replay_terminal = (
                                    _counterfactual_branch_outcome_v3(
                                        sumo_api=sumo_api,
                                        infos=infos,
                                        states=snapshot["states"],
                                        executors=executors,
                                        focal_id=str(snapshot["focal_id"]),
                                        candidate=candidate,
                                        context=context,
                                        control_interval_sec=int(control_interval_sec),
                                        counterfactual_horizon_intervals=int(
                                            counterfactual_horizon_intervals
                                        ),
                                        controlled_lane_count=max(
                                            len(controlled_lanes), 1
                                        ),
                                        controlled_lanes=controlled_lanes,
                                        counterfactual_cost_mode=str(
                                            counterfactual_cost_mode
                                        ),
                                        safety_ledger=_new_safety_ledger(),
                                    )
                                )
                                replay = _counterfactual_outcome_vector(
                                    replay_costs, replay_one, replay_terminal
                                )
                                replay_difference = float(
                                    np.max(np.abs(vector - replay))
                                )
                        except Exception as error:
                            valid = False
                            failure = f"{type(error).__name__}: {error}"
                            failure_action = action_name
                            break
                    record: dict[str, Any] = {
                        "checkpoint_index": int(snapshot["checkpoint_index"]),
                        "time_sec": float(snapshot["time_sec"]),
                        "focal_tls": str(snapshot["focal_id"]),
                        "future_seed": int(future_seed),
                        "valid": bool(valid),
                        "failure": failure,
                        "failure_action": failure_action,
                    }
                    if replay_difference is not None:
                        record["replay_max_abs_difference"] = replay_difference
                    if valid:
                        reference_cost = action_costs["reference"]
                        alternative_cost = action_costs["alternative"]
                        record.update(
                            {
                                "reference_cumulative_cost": reference_cost,
                                "alternative_cumulative_cost": alternative_cost,
                                "reference_minus_alternative_utility": (
                                    reference_cost - alternative_cost
                                ),
                                "alternative_minus_reference_cost": (
                                    alternative_cost - reference_cost
                                ),
                                "reference_outcome_vector": action_outcomes[
                                    "reference"
                                ].tolist(),
                                "alternative_outcome_vector": action_outcomes[
                                    "alternative"
                                ].tolist(),
                            }
                        )
                    records.append(record)
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass

    summary = summarize_fixed_state_records(records)
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "seven-city-development-randomness-diagnostic",
        "target_city": str(target_city),
        "scenario": str(scenario),
        "sumocfg": str(sumocfg),
        "base_seed": int(base_seed),
        "future_seeds": list(seeds),
        "snapshot_protocol": dict(SNAPSHOT_PROTOCOL),
        "historical_cache_contrast": {
            "historical_save_rng": True,
            "historical_independent_future_replicates_per_fixed_state": 0,
            "this_diagnostic_save_rng": False,
            "this_diagnostic_independent_future_replicates_per_fixed_state": len(
                seeds
            ),
        },
        "counterfactual_protocol": {
            "base_trace_duration_sec": float(base_trace_duration_sec),
            "warmup_sec": float(warmup_sec),
            "checkpoint_count": int(checkpoint_count),
            "checkpoint_spacing_sec": float(checkpoint_spacing_sec),
            "control_interval_sec": int(control_interval_sec),
            "horizon_intervals": int(counterfactual_horizon_intervals),
            "cost_mode": str(counterfactual_cost_mode),
            "reference_action": "phase_pressure",
            "alternative_action": "closest_green_count_feasible_nonreference",
        },
        "checkpoints": checkpoint_metadata,
        "records": records,
        "summary": summary,
        "runtime": {
            **runtime_metadata(),
            "libsumo_version": libsumo_version(sumo_api),
        },
        "claim_boundary": (
            "This diagnostic estimates future-seed variability at fixed SUMO "
            "states. It is development evidence only and does not establish "
            "source-city transfer benefit or closed-loop superiority."
        ),
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    cities = [str(row.get("target_city")) for row in rows]
    if len(rows) != 7 or len(set(cities)) != 7:
        raise ValueError("V150F aggregate requires seven distinct city results")
    if any(row.get("protocol") != RESULT_PROTOCOL for row in rows):
        raise ValueError("V150F result protocol changed")
    if any(row.get("snapshot_protocol") != SNAPSHOT_PROTOCOL for row in rows):
        raise ValueError("V150F snapshot protocol changed")
    summaries = {str(row["target_city"]): dict(row["summary"]) for row in rows}
    replay_passed = all(
        bool(summary["same_seed_replay_passed"]) for summary in summaries.values()
    )
    rng_omitted = all(
        not any(bool(item["snapshot_contains_rng_state"]) for item in row["checkpoints"])
        for row in rows
    )
    outcome_effective_cities = sorted(
        city
        for city, summary in summaries.items()
        if bool(summary["future_randomness_effective"])
    )
    cost_label_effective_cities = sorted(
        city
        for city, summary in summaries.items()
        if bool(summary["future_randomness_cost_label_effective"])
    )
    invalid_by_city = {
        str(row["target_city"]): {
            "invalid_pair_count": sum(
                not bool(record.get("valid", False)) for record in row["records"]
            ),
            "reference_failure_count": sum(
                record.get("failure_action") == "reference"
                for record in row["records"]
            ),
            "alternative_failure_count": sum(
                record.get("failure_action") == "alternative"
                for record in row["records"]
            ),
        }
        for row in rows
    }
    analyzable_checkpoints = [
        checkpoint
        for summary in summaries.values()
        for checkpoint in summary["checkpoints"]
        if int(checkpoint["valid_pair_count"]) > 0
    ]
    sign_unstable_checkpoints = sum(
        checkpoint["utility_sign_agreement_fraction"] is not None
        and float(checkpoint["utility_sign_agreement_fraction"]) < 1.0
        for checkpoint in analyzable_checkpoints
    )
    city_within_state_fractions = {
        city: float(
            summary["utility_variance_decomposition"]["within_state_fraction"]
        )
        for city, summary in summaries.items()
    }
    every_checkpoint_has_minimum_futures = all(
        int(checkpoint["valid_pair_count"])
        >= MINIMUM_VALID_FUTURES_PER_CHECKPOINT
        for summary in summaries.values()
        for checkpoint in summary["checkpoints"]
    )
    integrity_passed = bool(
        rng_omitted and replay_passed and every_checkpoint_has_minimum_futures
    )
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development-diagnostic-complete"
            if integrity_passed
            else "development-diagnostic-partial"
        ),
        "city_count": len(rows),
        "fixed_state_count": sum(
            int(summary["checkpoint_count"]) for summary in summaries.values()
        ),
        "valid_future_pair_count": sum(
            int(summary["valid_future_pair_count"])
            for summary in summaries.values()
        ),
        "integrity_gate": {
            "rng_state_omitted_from_every_snapshot": rng_omitted,
            "same_seed_replay_passed_every_city": replay_passed,
            "minimum_valid_futures_per_checkpoint": (
                MINIMUM_VALID_FUTURES_PER_CHECKPOINT
            ),
            "every_checkpoint_has_minimum_valid_futures": (
                every_checkpoint_has_minimum_futures
            ),
            "passed": integrity_passed,
        },
        "future_randomness_outcome_effective_cities": outcome_effective_cities,
        "future_randomness_outcome_effective_city_count": len(
            outcome_effective_cities
        ),
        "future_randomness_cost_label_effective_cities": (
            cost_label_effective_cities
        ),
        "future_randomness_cost_label_effective_city_count": len(
            cost_label_effective_cities
        ),
        "invalid_future_pairs_by_city": invalid_by_city,
        "utility_sign_stability": {
            "analyzable_checkpoint_count": len(analyzable_checkpoints),
            "sign_unstable_checkpoint_count": int(sign_unstable_checkpoints),
        },
        "within_state_future_variance_fraction": {
            "city_values": city_within_state_fractions,
            "city_macro_mean": float(
                np.mean(list(city_within_state_fractions.values()))
            ),
        },
        "city_summaries": summaries,
        "interpretation": (
            "Among valid branches, different future seeds changed the paired "
            "action-cost label in "
            f"{len(cost_label_effective_cities)}/7 cities. Future-seed averaging "
            "is required where that variance is material, while the separate "
            "between-state component must not be attributed to future randomness."
            if cost_label_effective_cities
            else "The tested action-cost labels were deterministic after conditioning "
            "on state, even if microscopic outcomes differed; observed cross-seed "
            "reversals therefore do not identify future-randomness label noise here."
        ),
        "claim_boundary": (
            "V150F diagnoses outcome-label variance. It neither selects a source "
            "nor authorizes source-aware closed-loop evaluation."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    scenario_parser = subparsers.add_parser("scenario")
    scenario_parser.add_argument("--manifest", type=Path, required=True)
    scenario_parser.add_argument("--target-city", required=True)
    scenario_parser.add_argument("--scenario", required=True)
    scenario_parser.add_argument("--base-seed", type=int, default=5057)
    scenario_parser.add_argument("--future-seeds", nargs="+", type=int, required=True)
    scenario_parser.add_argument("--base-trace-duration-sec", type=float, default=1200.0)
    scenario_parser.add_argument("--warmup-sec", type=float, default=300.0)
    scenario_parser.add_argument("--checkpoint-count", type=int, default=3)
    scenario_parser.add_argument("--checkpoint-spacing-sec", type=float, default=180.0)
    scenario_parser.add_argument("--control-interval-sec", type=int, default=30)
    scenario_parser.add_argument("--counterfactual-horizon-intervals", type=int, default=3)
    scenario_parser.add_argument(
        "--counterfactual-cost-mode", default="system_vehicle_load"
    )
    scenario_parser.add_argument("--out", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150F result: {args.out}")
    if args.mode == "scenario":
        result = run_scenario(
            manifest_path=args.manifest,
            target_city=args.target_city,
            scenario=args.scenario,
            base_seed=args.base_seed,
            future_seeds=args.future_seeds,
            base_trace_duration_sec=args.base_trace_duration_sec,
            warmup_sec=args.warmup_sec,
            checkpoint_count=args.checkpoint_count,
            checkpoint_spacing_sec=args.checkpoint_spacing_sec,
            control_interval_sec=args.control_interval_sec,
            counterfactual_horizon_intervals=args.counterfactual_horizon_intervals,
            counterfactual_cost_mode=args.counterfactual_cost_mode,
        )
        status = "DONE"
    else:
        result = aggregate_results([_read_json(path) for path in args.results])
        status = "PASS" if result["integrity_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
