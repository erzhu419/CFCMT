"""Run V157C native-prefix, one-focal-TLS Jinan action branches.

The module deliberately contains no SUMO checkpoint restoration.  Every model
branch starts at t=0, follows PhasePressure to one fixed checkpoint, changes at
most one focal traffic light for one 10-second interval, and follows
PhasePressure for the remaining 440 seconds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_CORE_FAMILY_V3,
    FittedContrastModelsV3,
    _controlled_lane_set,
    _group_adjusted_scores,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import pressure_spec


PROTOCOL = "tsc-v157c-jinan-native-one-action-branches-v1"
SEED_RESULT_PROTOCOL = "tsc-v157c-jinan-native-one-action-seed-result-v1"
AGGREGATE_PROTOCOL = "tsc-v157c-jinan-native-one-action-aggregate-v1"
INPUT_VALIDATION_PROTOCOL = "tsc-v157c-jinan-native-one-action-input-validation-v1"
EVALUATE_POLICY_V3_REQUIRED_KEYWORDS = (
    "sumo_api",
    "sumocfg",
    "scenario",
    "policy",
    "models",
    "duration_sec",
    "control_interval_sec",
    "warmup_sec",
    "seed",
    "tripinfo_output",
    "residual_coordination_mode",
    "residual_cooldown_intervals_override",
    "step_observer",
)
V157A_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-aggregate-v1"
)
RUNTIME_BUNDLE_PROTOCOL = "cfcmt-v157b-feature-aligned-b100-runtime-arms-v2"
ARM_MANIFEST_PROTOCOL = "tsc-v157b-feature-aligned-b100-arm-manifest-v2"
LEARNED_ARMS = ("target_only", "uniform_source", "source_label_placebo")
ALL_ARMS = (*LEARNED_ARMS, "phase_pressure")
SCENARIO = "jinan_3x4_real"
SEEDS = (80314, 88625, 27178)
CHECKPOINTS_SEC = (300, 480, 660)
CONTROL_INTERVAL_SEC = 10
HORIZON_SEC = 450
RUNTIME_POLICY = f"{CFCMT_CORE_FAMILY_V3}_contrast_source_utility"
FOCAL_TLS_RULE = (
    "among_action_eligible_tls_where_uniform_source_and_target_only_select_"
    "different_states_choose_largest_maximum_of_their_predicted_pp_advantages_"
    "then_lexicographically_first_tls;_if_none_choose_the_same_way_over_all_"
    "action_eligible_tls_and_record_an_exact_zero_source_vs_target_action_contrast"
)
ACTION_ROSTER = (
    "all_three_learned_arms_are_scored_on_each_action_eligible_tls_on_the_shared_"
    "phase_pressure_trajectory_before_any_450_second_outcome_is_observed"
)

EXECUTOR_FIELDS = (
    "mode",
    "current_state",
    "current_green_state",
    "target_green_state",
    "green_elapsed_sec",
    "remaining_sec",
    "pending_all_red_sec",
)


def _runtime_models(
    originator: Any,
    *,
    prediction_horizon_sec: int,
) -> FittedContrastModelsV3:
    family = CFCMT_CORE_FAMILY_V3
    prior = pressure_spec("phase_pressure")
    return FittedContrastModelsV3(
        family_models={family: None},
        prior_policy=prior.key,
        prior_spec=prior,
        prediction_horizon_sec=int(prediction_horizon_sec),
        objective_modes={family: "control_only"},
        guards={},
        regularizers={},
        target_support=None,
        hierarchy_layers={},
        diagnostics={"runtime": str(originator.selection_layer)},
        action_originator=originator,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    execution = dict(protocol.get("execution", {}))
    cost = dict(protocol.get("cost", {}))
    state = dict(protocol.get("same_state_contract", {}))
    runtime = dict(protocol.get("runtime_arm_semantics", {}))
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("scientific_status") != "prepared_not_run"
        or protocol.get("city") != "jinan"
        or protocol.get("scenario") != SCENARIO
        or tuple(int(value) for value in protocol.get("seeds", ())) != SEEDS
        or tuple(int(value) for value in protocol.get("checkpoints_sec", ()))
        != CHECKPOINTS_SEC
        or int(protocol.get("control_interval_sec", -1)) != CONTROL_INTERVAL_SEC
        or int(protocol.get("horizon_sec", -1)) != HORIZON_SEC
        or tuple(protocol.get("arms", ())) != ALL_ARMS
        or protocol.get("learned_arm_bundle_protocol") != RUNTIME_BUNDLE_PROTOCOL
        or protocol.get("learned_arm_manifest_protocol") != ARM_MANIFEST_PROTOCOL
        or runtime.get("target_only_training_domain_handling")
        != "relabel_all_b100_rows_to_jinan"
        or runtime.get("v157a_exact_target_refit_used_at_runtime") is not False
        or runtime.get("uniform_source_training_path")
        != "exact_v157a_source_component_refits"
        or protocol.get("reference_policy") != "phase_pressure"
        or protocol.get("branch_estimand")
        != "one_focal_tls_action_for_10_seconds_then_phase_pressure_for_440_seconds"
        or protocol.get("focal_tls_rule") != FOCAL_TLS_RULE
        or state.get("construction")
        != "fresh_native_t0_phase_pressure_prefix_for_every_branch"
        or state.get("snapshot_restore_used") is not False
        or state.get("action_eligibility")
        != "executor_feasible_states_now_count_greater_than_1_at_checkpoint"
        or state.get("zero_action_eligible_checkpoint")
        != "invalid_without_checkpoint_shift_or_drop"
        or state.get("action_roster") != ACTION_ROSTER
        or cost.get("name") != "prefix_mean_cost_450s"
        or cost.get("normalization") != "controlled_lane_count"
        or execution.get("evaluate_policy") != "evaluate_policy_v3"
        or execution.get("residual_coordination_mode") != "direct"
        or int(execution.get("residual_cooldown_intervals", -1)) != 0
        or int(execution.get("native_runs_per_seed", -1)) != 10
        or int(execution.get("total_native_runs", -1)) != 30
    ):
        raise ValueError("V157C frozen protocol changed")


def validate_authorization(
    protocol: Mapping[str, Any], authorization_path: Path
) -> dict[str, Any]:
    expected = dict(protocol["authorized_by"])
    if _sha256(authorization_path) != str(expected["sha256"]):
        raise ValueError("V157A authorization artifact identity changed")
    authorization = _read_json(authorization_path)
    branch = dict(authorization.get("b100_branch_authorization", {}))
    if (
        authorization.get("protocol") != V157A_PROTOCOL
        or branch.get("passed") is not True
        or branch.get("decision") != expected["required_decision"]
    ):
        raise ValueError("V157A did not authorize the V157C branch protocol")
    return authorization


@dataclass(frozen=True)
class V157CActionDecision:
    reference_index: int
    proposed_index: int
    selected_index: int
    learned_differs: bool
    eligible: bool
    priority: float
    rejection: str | None
    arm: str
    tls_id: str
    time_sec: float
    proposed_state: str
    selected_state: str
    reference_state: str
    predicted_score: float
    reference_score: float
    forced_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_metadata(
    dataset: Any, reference_index: int
) -> tuple[str, float, tuple[str, ...]]:
    tls_values = tuple(str(value) for value in dataset.metadata.get("row_tls", ()))
    time_values = tuple(
        float(value) for value in dataset.metadata.get("row_times", ())
    )
    states = tuple(
        str(value) for value in dataset.metadata.get("candidate_states", ())
    )
    if (
        len(tls_values) != int(dataset.size)
        or len(time_values) != int(dataset.size)
        or len(states) != int(dataset.size)
        or len(set(tls_values)) != 1
        or len(set(time_values)) != 1
        or not 0 <= int(reference_index) < int(dataset.size)
    ):
        raise ValueError("V157C requires one row-aligned TLS action group")
    return tls_values[0], time_values[0], states


def phase_pressure_decision(
    dataset: Any,
    *,
    reference_index: int,
    arm: str,
    rejection: str,
) -> V157CActionDecision:
    tls_id, time_sec, states = _row_metadata(dataset, reference_index)
    reference = int(reference_index)
    return V157CActionDecision(
        reference_index=reference,
        proposed_index=reference,
        selected_index=reference,
        learned_differs=False,
        eligible=False,
        priority=0.0,
        rejection=str(rejection),
        arm=str(arm),
        tls_id=tls_id,
        time_sec=time_sec,
        proposed_state=states[reference],
        selected_state=states[reference],
        reference_state=states[reference],
        predicted_score=0.0,
        reference_score=0.0,
        forced_reference=True,
    )


def score_model_decision(
    model: Any,
    dataset: Any,
    *,
    reference_index: int,
    arm: str,
) -> V157CActionDecision:
    tls_id, time_sec, states = _row_metadata(dataset, reference_index)
    reference = int(reference_index)
    score, _, _, _ = _group_adjusted_scores(
        dataset,
        model.predict(dataset),
        objective_mode="control_only",
    )
    scores = np.asarray(score, dtype=float)
    if scores.shape != (int(dataset.size),) or not np.all(np.isfinite(scores)):
        raise ValueError(f"V157C {arm} model returned invalid scores")
    selected = int(np.argmin(scores))
    differs = selected != reference
    return V157CActionDecision(
        reference_index=reference,
        proposed_index=selected,
        selected_index=selected,
        learned_differs=differs,
        eligible=differs,
        priority=max(float(scores[reference] - scores[selected]), 0.0),
        rejection=None,
        arm=str(arm),
        tls_id=tls_id,
        time_sec=time_sec,
        proposed_state=states[selected],
        selected_state=states[selected],
        reference_state=states[reference],
        predicted_score=float(scores[selected]),
        reference_score=float(scores[reference]),
        forced_reference=False,
    )


def force_phase_pressure(
    decision: V157CActionDecision, *, reason: str
) -> V157CActionDecision:
    return replace(
        decision,
        selected_index=decision.reference_index,
        learned_differs=False,
        eligible=False,
        priority=0.0,
        rejection=str(reason),
        selected_state=decision.reference_state,
        forced_reference=True,
    )


def _checkpoint(time_sec: float) -> int | None:
    rounded = int(round(float(time_sec)))
    if abs(float(time_sec) - rounded) > 1e-9 or rounded not in CHECKPOINTS_SEC:
        return None
    return rounded


def _checkpoint_scorable_tls_ids(
    states: Mapping[str, Any], executors: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return exactly the TLS roster that V3 will pass to an originator."""

    return tuple(
        sorted(
            str(tls_id)
            for tls_id in states
            if len(tuple(executors[tls_id].feasible_states_now())) > 1
        )
    )


def _shared_interval_checkpoint(states: Mapping[str, Any]) -> int | None:
    checkpoints = {
        _checkpoint(float(state.sim_time)) for state in states.values()
    }
    if len(checkpoints) != 1:
        raise ValueError("V157C TLS states do not share one interval time")
    return next(iter(checkpoints))


class RosterProbeOriginator:
    """Score all learned arms on PP states while executing PP throughout."""

    selection_layer = "v157c_shared_pp_roster_probe"

    def __init__(self, models: Mapping[str, Any]) -> None:
        if tuple(models) != LEARNED_ARMS:
            raise ValueError("V157C roster models are incomplete or reordered")
        self.models = dict(models)
        self.roster: dict[int, dict[str, dict[str, dict[str, Any]]]] = {
            checkpoint: {} for checkpoint in CHECKPOINTS_SEC
        }
        self.controlled_lanes: tuple[str, ...] = ()
        self.tls_ids: tuple[str, ...] = ()
        self.scorable_tls_ids: dict[int, tuple[str, ...] | None] = {
            checkpoint: None for checkpoint in CHECKPOINTS_SEC
        }

    def prepare_interval(self, **kwargs: Any) -> None:
        states = dict(kwargs["states"])
        infos = {tls_id: state.info for tls_id, state in states.items()}
        lanes = tuple(sorted(_controlled_lane_set(infos)))
        tls_ids = tuple(sorted(states))
        if self.controlled_lanes and self.controlled_lanes != lanes:
            raise ValueError("V157C controlled lanes changed within one trajectory")
        if self.tls_ids and self.tls_ids != tls_ids:
            raise ValueError("V157C TLS roster changed within one trajectory")
        self.controlled_lanes = lanes
        self.tls_ids = tls_ids
        checkpoint = _shared_interval_checkpoint(states)
        if checkpoint is not None:
            scorable = _checkpoint_scorable_tls_ids(states, kwargs["executors"])
            previous = self.scorable_tls_ids[checkpoint]
            if previous is not None and previous != scorable:
                raise ValueError(
                    f"V157C checkpoint {checkpoint} scorable TLS roster changed"
                )
            self.scorable_tls_ids[checkpoint] = scorable

    def select(self, dataset: Any, *, reference_index: int) -> V157CActionDecision:
        tls_id, time_sec, _ = _row_metadata(dataset, reference_index)
        checkpoint = _checkpoint(time_sec)
        if checkpoint is None:
            return phase_pressure_decision(
                dataset,
                reference_index=reference_index,
                arm="phase_pressure",
                rejection="outside_fixed_checkpoint",
            )
        decisions = {
            arm: score_model_decision(
                self.models[arm],
                dataset,
                reference_index=reference_index,
                arm=arm,
            ).to_dict()
            for arm in LEARNED_ARMS
        }
        if tls_id in self.roster[checkpoint]:
            raise ValueError("V157C scored a checkpoint TLS more than once")
        self.roster[checkpoint][tls_id] = decisions
        driver = V157CActionDecision(**decisions[LEARNED_ARMS[0]])
        return force_phase_pressure(driver, reason="shared_phase_pressure_probe")

    def finalized_roster(self) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
        for checkpoint in CHECKPOINTS_SEC:
            scorable = self.scorable_tls_ids[checkpoint]
            if not scorable:
                raise ValueError(
                    f"V157C checkpoint {checkpoint} has no scorable TLS"
                )
            if tuple(sorted(self.roster[checkpoint])) != scorable:
                raise ValueError(f"V157C checkpoint {checkpoint} roster is incomplete")
            if any(
                tuple(rows) != LEARNED_ARMS
                for rows in self.roster[checkpoint].values()
            ):
                raise ValueError("V157C learned-arm roster is incomplete")
        return self.roster


def select_focal_tls(
    checkpoint_roster: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    if not checkpoint_roster:
        raise ValueError("V157C focal selection requires a checkpoint roster")
    tls_ids = tuple(sorted(str(value) for value in checkpoint_roster))
    for tls_id in tls_ids:
        if tuple(checkpoint_roster[tls_id]) != LEARNED_ARMS:
            raise ValueError("V157C focal selection arm roster changed")

    disagreements = tuple(
        tls_id
        for tls_id in tls_ids
        if checkpoint_roster[tls_id]["uniform_source"]["selected_state"]
        != checkpoint_roster[tls_id]["target_only"]["selected_state"]
    )
    eligible = disagreements or tls_ids

    def key(tls_id: str) -> tuple[float, float, str]:
        source = checkpoint_roster[tls_id]["uniform_source"]
        target = checkpoint_roster[tls_id]["target_only"]
        return (
            -max(float(source["priority"]), float(target["priority"])),
            -float(source["priority"]),
            str(tls_id),
        )

    selected = min(eligible, key=key)
    selected_rows = checkpoint_roster[selected]
    return {
        "focal_tls": selected,
        "selection_basis": (
            "source_target_action_disagreement_then_predicted_advantage"
            if disagreements
            else "no_source_target_action_disagreement_zero_primary_contrast"
        ),
        "source_target_disagreement_tls_count": len(disagreements),
        "source_target_selected_actions_differ": bool(disagreements),
        "outcomes_used_for_selection": False,
        "selected_actions": {
            arm: str(selected_rows[arm]["selected_state"])
            for arm in LEARNED_ARMS
        },
        "reference_action": str(selected_rows["target_only"]["reference_state"]),
    }


class OneActionBranchOriginator:
    """Expose one learned focal-TLS action and force PP everywhere else."""

    selection_layer = "v157c_one_focal_tls_action"

    def __init__(
        self,
        *,
        model: Any,
        arm: str,
        checkpoint_sec: int,
        focal_tls: str,
        expected_roster: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if arm not in LEARNED_ARMS or int(checkpoint_sec) not in CHECKPOINTS_SEC:
            raise ValueError("unknown V157C branch arm or checkpoint")
        self.model = model
        self.arm = str(arm)
        self.checkpoint_sec = int(checkpoint_sec)
        self.focal_tls = str(focal_tls)
        self.expected_roster = dict(expected_roster)
        self.checkpoint_decisions: dict[str, dict[str, Any]] = {}
        self.controlled_lanes: tuple[str, ...] = ()
        self.tls_ids: tuple[str, ...] = ()
        self.scorable_tls_ids: tuple[str, ...] | None = None

    def prepare_interval(self, **kwargs: Any) -> None:
        states = dict(kwargs["states"])
        infos = {tls_id: state.info for tls_id, state in states.items()}
        self.controlled_lanes = tuple(sorted(_controlled_lane_set(infos)))
        self.tls_ids = tuple(sorted(states))
        if _shared_interval_checkpoint(states) == self.checkpoint_sec:
            scorable = _checkpoint_scorable_tls_ids(
                states, kwargs["executors"]
            )
            if (
                self.scorable_tls_ids is not None
                and self.scorable_tls_ids != scorable
            ):
                raise ValueError("V157C branch scorable TLS roster changed")
            self.scorable_tls_ids = scorable

    def select(self, dataset: Any, *, reference_index: int) -> V157CActionDecision:
        tls_id, time_sec, _ = _row_metadata(dataset, reference_index)
        if abs(time_sec - self.checkpoint_sec) > 1e-9:
            return phase_pressure_decision(
                dataset,
                reference_index=reference_index,
                arm=self.arm,
                rejection="outside_one_action_interval",
            )
        decision = score_model_decision(
            self.model,
            dataset,
            reference_index=reference_index,
            arm=self.arm,
        )
        actual = decision if tls_id == self.focal_tls else force_phase_pressure(
            decision, reason="nonfocal_tls"
        )
        self.checkpoint_decisions[tls_id] = decision.to_dict()
        return actual

    def validate_checkpoint_roster(self) -> None:
        if not self.scorable_tls_ids:
            raise ValueError("V157C branch checkpoint has no scorable TLS")
        expected_tls_ids = tuple(sorted(self.expected_roster))
        if self.scorable_tls_ids != expected_tls_ids:
            raise ValueError("V157C branch scorable TLS roster changed")
        if tuple(sorted(self.checkpoint_decisions)) != self.scorable_tls_ids:
            raise ValueError("V157C branch checkpoint roster is incomplete")
        for tls_id in self.scorable_tls_ids:
            expected = dict(self.expected_roster[tls_id][self.arm])
            if self.checkpoint_decisions[tls_id] != expected:
                raise ValueError(
                    f"V157C {self.arm} decision changed at {tls_id}"
                )


def physical_snapshot(api: Any, executors: Mapping[str, Any]) -> dict[str, Any]:
    vehicles = {
        vehicle_id: {
            "lane_id": str(api.vehicle.getLaneID(vehicle_id)),
            "route_index": int(api.vehicle.getRouteIndex(vehicle_id)),
            "route": tuple(str(value) for value in api.vehicle.getRoute(vehicle_id)),
            "lane_position": float(api.vehicle.getLanePosition(vehicle_id)),
            "speed": float(api.vehicle.getSpeed(vehicle_id)),
        }
        for vehicle_id in sorted(api.vehicle.getIDList())
    }
    executor_states = {}
    for tls_id in sorted(executors):
        snapshot = executors[tls_id].snapshot()
        executor_states[str(tls_id)] = {
            key: snapshot[key] for key in EXECUTOR_FIELDS
        }
    return {
        "time_sec": float(api.simulation.getTime()),
        "vehicles": vehicles,
        "executors": executor_states,
        "pending_ids": tuple(sorted(api.simulation.getPendingVehicles())),
    }


def compare_physical(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, tolerance: float = 0.0
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    max_numeric_difference = 0.0

    def mismatch(path: str, before: Any, after: Any) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 20:
            mismatches.append({"path": path, "expected": before, "actual": after})

    if float(expected["time_sec"]) != float(actual["time_sec"]):
        mismatch("time_sec", expected["time_sec"], actual["time_sec"])
    if tuple(expected["pending_ids"]) != tuple(actual["pending_ids"]):
        mismatch("pending_ids", expected["pending_ids"], actual["pending_ids"])
    expected_vehicles = dict(expected["vehicles"])
    actual_vehicles = dict(actual["vehicles"])
    if set(expected_vehicles) != set(actual_vehicles):
        mismatch(
            "vehicle_ids", sorted(expected_vehicles), sorted(actual_vehicles)
        )
    for vehicle_id in sorted(set(expected_vehicles) & set(actual_vehicles)):
        before = expected_vehicles[vehicle_id]
        after = actual_vehicles[vehicle_id]
        for key in ("lane_id", "route_index", "route"):
            if before[key] != after[key]:
                mismatch(f"vehicles.{vehicle_id}.{key}", before[key], after[key])
        for key in ("lane_position", "speed"):
            difference = abs(float(before[key]) - float(after[key]))
            max_numeric_difference = max(max_numeric_difference, difference)
            if difference > float(tolerance):
                mismatch(f"vehicles.{vehicle_id}.{key}", before[key], after[key])
    expected_executors = dict(expected["executors"])
    actual_executors = dict(actual["executors"])
    if set(expected_executors) != set(actual_executors):
        mismatch(
            "executor_ids", sorted(expected_executors), sorted(actual_executors)
        )
    for tls_id in sorted(set(expected_executors) & set(actual_executors)):
        for key in EXECUTOR_FIELDS:
            if expected_executors[tls_id][key] != actual_executors[tls_id][key]:
                mismatch(
                    f"executors.{tls_id}.{key}",
                    expected_executors[tls_id][key],
                    actual_executors[tls_id][key],
                )
    return {
        "passed": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "max_position_or_speed_difference": max_numeric_difference,
    }


def _new_window_record() -> dict[str, Any]:
    return {
        "samples": [],
        "starting_teleports": 0,
        "ending_teleports": 0,
        "collision_events": 0,
    }


def _record_step_diagnostics(api: Any, record: dict[str, Any]) -> None:
    record["starting_teleports"] += int(
        api.simulation.getStartingTeleportNumber()
    )
    record["ending_teleports"] += int(api.simulation.getEndingTeleportNumber())
    record["collision_events"] += len(tuple(api.simulation.getCollisions()))


def _sample(api: Any, controlled_lanes: Sequence[str], time_sec: float) -> dict[str, Any]:
    lane_count = len(controlled_lanes)
    if lane_count <= 0:
        raise ValueError("V157C controlled-lane set is empty")
    return {
        "time_sec": float(time_sec),
        "cost": float(
            sum(
                api.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in controlled_lanes
            )
        )
        / lane_count,
        "active": int(api.vehicle.getIDCount()),
        "pending": len(api.simulation.getPendingVehicles()),
        "arrived": int(api.simulation.getArrivedNumber()),
        "departed": int(api.simulation.getDepartedNumber()),
    }


def _window_summary(record: Mapping[str, Any], start: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(record["samples"])
    costs = [float(row["cost"]) for row in samples]
    arrived = sum(int(row["arrived"]) for row in samples)
    departed = sum(int(row["departed"]) for row in samples)
    end_active = int(samples[-1]["active"]) if samples else None
    accounting = (
        end_active == len(start["vehicles"]) + departed - arrived
        if end_active is not None
        else False
    )
    return {
        "sample_count": len(samples),
        "mean_halted_per_controlled_lane": (
            float(np.mean(costs)) if costs else None
        ),
        "halted_vehicle_seconds_per_controlled_lane": float(np.sum(costs)),
        "arrived": arrived,
        "departed": departed,
        "end_active": end_active,
        "end_pending": int(samples[-1]["pending"]) if samples else None,
        "active_population_accounting_passed": bool(accounting),
        "starting_teleports": int(record["starting_teleports"]),
        "ending_teleports": int(record["ending_teleports"]),
        "collision_events": int(record["collision_events"]),
    }


def _load_arm_models(bundle_path: Path, expected_sha256: str) -> dict[str, Any]:
    from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze import (
        load_runtime_arms,
        runtime_arm_model,
    )

    payload = load_runtime_arms(
        bundle_path,
        expected_sha256=str(expected_sha256),
    )
    if payload.get("protocol") != RUNTIME_BUNDLE_PROTOCOL:
        raise ValueError("V157B runtime bundle protocol changed")
    arms = dict(payload.get("arms", {}))
    target = dict(arms.get("target_only", {}))
    uniform = dict(arms.get("uniform_source", {}))
    if (
        payload.get("runtime_target_comparator")
        != "domain_aligned_b100_target_only_v2"
        or target.get("training_domain_handling")
        != "relabel_all_b100_rows_to_jinan"
        or target.get("v157a_exact_target_refit_used_at_runtime") is not False
        or uniform.get("training_path")
        != "exact_v157a_source_component_refits"
        or uniform.get("source_labels_permuted") is not False
    ):
        raise ValueError("V157B runtime arm semantics changed")
    models = {
        arm: runtime_arm_model(payload, arm)
        for arm in LEARNED_ARMS
    }
    if any(model is None for model in models.values()):
        raise ValueError("V157C learned arm resolved to no model")
    return models


def _validate_arm_manifest(
    path: Path,
    *,
    expected_sha256: str,
    bundle_path: Path,
    bundle_sha256: str,
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("V157B arm manifest identity changed")
    manifest = _read_json(path)
    arms = dict(manifest.get("arms", {}))
    bundle = dict(manifest.get("runtime_bundle", {}))
    if (
        manifest.get("protocol") != ARM_MANIFEST_PROTOCOL
        or manifest.get("city") != "jinan"
        or tuple(manifest.get("arm_order", ())) != ALL_ARMS
        or Path(bundle.get("path", "")).resolve() != bundle_path.resolve()
        or bundle.get("sha256") != str(bundle_sha256)
        or bundle.get("protocol") != RUNTIME_BUNDLE_PROTOCOL
        or int(bundle.get("size_bytes", -1)) != bundle_path.stat().st_size
        or set(arms) != set(ALL_ARMS)
        or any(
            arms[arm].get("kind") != "learned_model"
            or arms[arm].get("bundle_key") != arm
            or arms[arm].get("predict_interface")
            != "model.predict(action_contrast_dataset)"
            for arm in LEARNED_ARMS
        )
        or arms["target_only"].get("training_domain_handling")
        != "relabel_all_b100_rows_to_jinan"
        or arms["target_only"].get("v157a_exact_target_refit_used_at_runtime")
        is not False
        or arms["uniform_source"].get("training_path")
        != "exact_v157a_source_component_refits"
        or arms["phase_pressure"]
        != {
            "kind": "reference_policy",
            "policy": "phase_pressure",
            "bundle_key": "phase_pressure",
        }
        or manifest.get("scope")
        != "one_frozen_focal_tls_action_then_phase_pressure_continuation"
    ):
        raise ValueError("V157B arm manifest contract changed")
    return manifest


def validate_inputs(
    *,
    protocol_path: Path,
    authorization_path: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    arm_manifest_path: Path,
    arm_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate frozen V157C model inputs without SUMO or output writes."""

    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    evaluator_interface = validate_evaluate_policy_v3_interface()
    authorization = validate_authorization(protocol, authorization_path)
    if _sha256(runtime_bundle_path) != str(runtime_bundle_sha256):
        raise ValueError("V157B runtime bundle identity changed")
    arm_manifest = _validate_arm_manifest(
        arm_manifest_path,
        expected_sha256=arm_manifest_sha256,
        bundle_path=runtime_bundle_path,
        bundle_sha256=runtime_bundle_sha256,
    )
    models = _load_arm_models(runtime_bundle_path, runtime_bundle_sha256)
    return {
        "protocol": INPUT_VALIDATION_PROTOCOL,
        "status": "PASS",
        "parent_protocol": protocol["protocol"],
        "authorization_protocol": authorization["protocol"],
        "arm_manifest_protocol": arm_manifest["protocol"],
        "runtime_bundle_protocol": RUNTIME_BUNDLE_PROTOCOL,
        "learned_arms": list(models),
        "evaluate_policy_v3_interface": evaluator_interface,
        "sumo_started": False,
        "artifacts_written": False,
    }


def validate_evaluate_policy_v3_interface(
    evaluator: Any | None = None,
) -> dict[str, Any]:
    """Check the evaluator keyword contract without starting SUMO."""

    if evaluator is None:
        from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3

        evaluator = v3.evaluate_policy_v3
    try:
        signature = inspect.signature(evaluator)
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot inspect evaluate_policy_v3 interface") from exc
    parameters = signature.parameters
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    keyword_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    missing = [
        name
        for name in EVALUATE_POLICY_V3_REQUIRED_KEYWORDS
        if not accepts_var_keyword
        and (
            name not in parameters
            or parameters[name].kind not in keyword_kinds
        )
    ]
    if missing:
        raise ValueError(
            "evaluate_policy_v3 does not accept V157C keyword arguments: "
            + ", ".join(missing)
        )
    return {
        "passed": True,
        "required_keyword_arguments": list(
            EVALUATE_POLICY_V3_REQUIRED_KEYWORDS
        ),
        "accepts_var_keyword": accepts_var_keyword,
    }


def _evaluate(
    *,
    api: Any,
    v3: Any,
    sumocfg: Path,
    seed: int,
    duration_sec: int,
    originator: Any,
    observer: Any,
    tripinfo_path: Path,
) -> dict[str, Any]:
    try:
        return v3.evaluate_policy_v3(
            sumo_api=api,
            sumocfg=sumocfg,
            scenario=SCENARIO,
            policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=HORIZON_SEC),
            duration_sec=int(duration_sec),
            control_interval_sec=CONTROL_INTERVAL_SEC,
            warmup_sec=0,
            seed=int(seed),
            tripinfo_output=tripinfo_path,
            residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0,
            step_observer=observer,
        )
    finally:
        tripinfo_path.unlink(missing_ok=True)


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "ok",
            "error",
            "starting_teleports",
            "ending_teleports",
            "collision_events",
            "collision_incidents",
            "emergency_stops",
        )
    }


def _intervention_trace_checks(
    trace: Sequence[Mapping[str, Any]],
    *,
    decision: Mapping[str, Any],
    checkpoint: int,
    focal_tls: str,
) -> dict[str, bool]:
    expected_count = int(
        str(decision["selected_state"]) != str(decision["reference_state"])
    )
    exact_count = len(trace) == expected_count
    return {
        "at_most_one_learned_action": exact_count,
        "learned_action_only_at_focal_checkpoint": all(
            float(row["time_sec"]) == float(checkpoint)
            and str(row["tls_id"]) == str(focal_tls)
            for row in trace
        ),
        "learned_action_execution_effective": bool(
            expected_count == 0
            or (
                exact_count
                and trace[0].get("execution_effective") is True
            )
        ),
    }


def run_seed(
    *,
    protocol_path: Path,
    authorization_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    arm_manifest_path: Path,
    arm_manifest_sha256: str,
    seed: int,
    scratch_root: Path,
    output: Path,
) -> dict[str, Any]:
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3

    started = time.monotonic()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    validate_authorization(protocol, authorization_path)
    if int(seed) not in SEEDS:
        raise ValueError("V157C seed is outside the frozen roster")
    if output.exists() or scratch_root.exists():
        raise FileExistsError("refusing to overwrite V157C output or scratch")
    if _sha256(manifest_path) != protocol["network_manifest"]["sha256"]:
        raise ValueError("V157C network manifest identity changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    benchmark = load_traffic_signal_manifest(manifest_path)
    scenario_specs = {row.scenario: row for row in benchmark.scenarios}
    if (
        SCENARIO not in scenario_specs
        or scenario_specs[SCENARIO].city_group != "jinan"
    ):
        raise ValueError("V157C Jinan scenario mapping changed")
    sumocfg = Path(scenario_specs[SCENARIO].sumocfg)
    if not sumocfg.is_file():
        raise FileNotFoundError(sumocfg)
    if _sha256(runtime_bundle_path) != str(runtime_bundle_sha256):
        raise ValueError("V157B runtime bundle identity changed")
    arm_manifest = _validate_arm_manifest(
        arm_manifest_path,
        expected_sha256=arm_manifest_sha256,
        bundle_path=runtime_bundle_path,
        bundle_sha256=runtime_bundle_sha256,
    )
    models = _load_arm_models(runtime_bundle_path, runtime_bundle_sha256)
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("V157C requires SUMO 1.22.0")
    scratch_root.mkdir(parents=True)
    output.mkdir(parents=True)

    probe = RosterProbeOriginator(models)
    baseline_snapshots: dict[int, dict[str, Any]] = {}
    baseline_windows = {
        checkpoint: _new_window_record() for checkpoint in CHECKPOINTS_SEC
    }

    def baseline_observer(*, stage: str, time_sec: float, executors: Any) -> None:
        if stage != "after_executor_advance":
            return
        now = int(round(float(time_sec)))
        if abs(float(time_sec) - now) > 1e-9:
            return
        if now in CHECKPOINTS_SEC:
            baseline_snapshots[now] = physical_snapshot(api, executors)
        for checkpoint, record in baseline_windows.items():
            if checkpoint < now <= checkpoint + HORIZON_SEC:
                record["samples"].append(
                    _sample(api, probe.controlled_lanes, float(time_sec))
                )
                _record_step_diagnostics(api, record)

    baseline_metrics = _evaluate(
        api=api,
        v3=v3,
        sumocfg=sumocfg,
        seed=int(seed),
        duration_sec=max(CHECKPOINTS_SEC) + HORIZON_SEC,
        originator=probe,
        observer=baseline_observer,
        tripinfo_path=scratch_root / "phase_pressure.xml",
    )
    roster = probe.finalized_roster()
    focal = {
        checkpoint: select_focal_tls(roster[checkpoint])
        for checkpoint in CHECKPOINTS_SEC
    }
    baseline_checks = {
        "evaluator_complete": baseline_metrics.get("ok") is True,
        "no_learned_interventions": baseline_metrics.get(
            "accepted_intervention_trace", []
        )
        == [],
        "all_checkpoint_states_captured": tuple(sorted(baseline_snapshots))
        == CHECKPOINTS_SEC,
        "all_windows_complete": all(
            len(record["samples"]) == HORIZON_SEC
            for record in baseline_windows.values()
        ),
        "controlled_lanes_nonempty": bool(probe.controlled_lanes),
    }
    baseline_summaries = {
        checkpoint: _window_summary(
            baseline_windows[checkpoint], baseline_snapshots[checkpoint]
        )
        for checkpoint in CHECKPOINTS_SEC
    }

    branch_summaries: dict[int, dict[str, dict[str, Any]]] = {
        checkpoint: {} for checkpoint in CHECKPOINTS_SEC
    }
    full_branches: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS_SEC:
        for arm in LEARNED_ARMS:
            originator = OneActionBranchOriginator(
                model=models[arm],
                arm=arm,
                checkpoint_sec=checkpoint,
                focal_tls=focal[checkpoint]["focal_tls"],
                expected_roster=roster[checkpoint],
            )
            branch_snapshot: dict[str, Any] = {}
            record = _new_window_record()

            def branch_observer(
                *, stage: str, time_sec: float, executors: Any
            ) -> None:
                if stage != "after_executor_advance":
                    return
                now = int(round(float(time_sec)))
                if abs(float(time_sec) - now) > 1e-9:
                    return
                if now == checkpoint:
                    branch_snapshot.update(physical_snapshot(api, executors))
                if checkpoint < now <= checkpoint + HORIZON_SEC:
                    record["samples"].append(
                        _sample(api, originator.controlled_lanes, float(time_sec))
                    )
                    _record_step_diagnostics(api, record)

            metrics = _evaluate(
                api=api,
                v3=v3,
                sumocfg=sumocfg,
                seed=int(seed),
                duration_sec=checkpoint + HORIZON_SEC,
                originator=originator,
                observer=branch_observer,
                tripinfo_path=scratch_root / f"t{checkpoint}_{arm}.xml",
            )
            originator.validate_checkpoint_roster()
            physical = compare_physical(
                baseline_snapshots[checkpoint], branch_snapshot, tolerance=0.0
            )
            decision = originator.checkpoint_decisions[
                focal[checkpoint]["focal_tls"]
            ]
            trace = list(metrics.get("accepted_intervention_trace", []))
            checks = {
                "evaluator_complete": metrics.get("ok") is True,
                "physical_start_exact": physical["passed"],
                "full_450_second_horizon": len(record["samples"]) == HORIZON_SEC,
                "active_population_accounting": False,
                "checkpoint_roster_exact": True,
                **_intervention_trace_checks(
                    trace,
                    decision=decision,
                    checkpoint=checkpoint,
                    focal_tls=focal[checkpoint]["focal_tls"],
                ),
            }
            summary = _window_summary(record, branch_snapshot)
            checks["active_population_accounting"] = summary[
                "active_population_accounting_passed"
            ]
            summary.update(
                {
                    "status": "PASS" if all(checks.values()) else "INVALID",
                    "checks": checks,
                    "physical_start_comparison": physical,
                    "selected_action": decision["selected_state"],
                    "reference_action": decision["reference_state"],
                    "predicted_advantage_vs_phase_pressure": decision["priority"],
                    "accepted_intervention_count": len(trace),
                    "runtime": _compact_metrics(metrics),
                }
            )
            branch_summaries[checkpoint][arm] = summary
            full_branches[f"t{checkpoint}:{arm}"] = {
                "checkpoint_snapshot": branch_snapshot,
                "checkpoint_decisions": originator.checkpoint_decisions,
                "samples": record["samples"],
                "accepted_intervention_trace": trace,
            }

    identity_checks: dict[int, dict[str, bool]] = {}
    for checkpoint in CHECKPOINTS_SEC:
        rows = branch_summaries[checkpoint]
        actions = {
            **{arm: rows[arm]["selected_action"] for arm in LEARNED_ARMS},
            "phase_pressure": focal[checkpoint]["reference_action"],
        }
        costs = {
            **{
                arm: [
                    row["cost"]
                    for row in full_branches[f"t{checkpoint}:{arm}"]["samples"]
                ]
                for arm in LEARNED_ARMS
            },
            "phase_pressure": [
                row["cost"] for row in baseline_windows[checkpoint]["samples"]
            ],
        }
        checks = {}
        for index, left in enumerate(ALL_ARMS):
            for right in ALL_ARMS[index + 1 :]:
                if actions[left] == actions[right]:
                    checks[f"{left}_equals_{right}"] = costs[left] == costs[right]
        identity_checks[checkpoint] = checks

    windows: dict[int, dict[str, Any]] = {}
    for checkpoint in CHECKPOINTS_SEC:
        costs = {
            arm: float(
                branch_summaries[checkpoint][arm][
                    "mean_halted_per_controlled_lane"
                ]
            )
            for arm in LEARNED_ARMS
        }
        costs["phase_pressure"] = float(
            baseline_summaries[checkpoint]["mean_halted_per_controlled_lane"]
        )
        windows[checkpoint] = {
            "checkpoint_sec": checkpoint,
            "scorable_tls_ids": list(probe.scorable_tls_ids[checkpoint] or ()),
            "scorable_tls_count": len(roster[checkpoint]),
            "focal": focal[checkpoint],
            "costs": costs,
            "contrasts": {
                "uniform_source_minus_target_only": costs["uniform_source"]
                - costs["target_only"],
                "uniform_source_minus_source_label_placebo": costs[
                    "uniform_source"
                ]
                - costs["source_label_placebo"],
                **{
                    f"{arm}_minus_phase_pressure": costs[arm]
                    - costs["phase_pressure"]
                    for arm in LEARNED_ARMS
                },
            },
            "phase_pressure": baseline_summaries[checkpoint],
            "learned_arms": branch_summaries[checkpoint],
            "same_selected_action_replay_checks": identity_checks[checkpoint],
        }

    all_checks = {
        **{f"baseline_{key}": value for key, value in baseline_checks.items()},
        "all_branch_contracts": all(
            row["status"] == "PASS"
            for checkpoint in branch_summaries.values()
            for row in checkpoint.values()
        ),
        "same_action_trajectories_exact": all(
            all(checks.values()) for checks in identity_checks.values()
        ),
        "run_count_exact": 1 + len(CHECKPOINTS_SEC) * len(LEARNED_ARMS) == 10,
    }
    result = {
        "protocol": SEED_RESULT_PROTOCOL,
        "status": "PASS" if all(all_checks.values()) else "INVALID",
        "parent_protocol": PROTOCOL,
        "seed": int(seed),
        "scenario": SCENARIO,
        "checks": all_checks,
        "controlled_lane_count": len(probe.controlled_lanes),
        "tls_count": len(probe.tls_ids),
        "checkpoints_sec": list(CHECKPOINTS_SEC),
        "windows": {str(key): value for key, value in windows.items()},
        "baseline_runtime": _compact_metrics(baseline_metrics),
        "runtime_bundle": {
            "path": str(runtime_bundle_path),
            "sha256": str(runtime_bundle_sha256),
            "protocol": RUNTIME_BUNDLE_PROTOCOL,
        },
        "arm_manifest": {
            "path": str(arm_manifest_path),
            "sha256": str(arm_manifest_sha256),
            "protocol": arm_manifest.get("protocol"),
        },
        "native_simulation_count": 10,
        "full_traces_server_only": str(output / "full_traces_server_only.json"),
        "collision_handling": "diagnostic_not_primary_gate",
        "elapsed_sec": time.monotonic() - started,
    }
    (output / "full_traces_server_only.json").write_text(
        json.dumps(
            {
                "baseline_snapshots": baseline_snapshots,
                "baseline_windows": baseline_windows,
                "model_roster": roster,
                "branches": full_branches,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(scratch_root)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "seed": int(seed),
                "elapsed_sec": result["elapsed_sec"],
            }
        ),
        flush=True,
    )
    return result


def aggregate_seed_results(
    protocol: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_protocol(protocol)
    by_seed = {int(row["seed"]): row for row in results}
    if (
        tuple(sorted(by_seed)) != tuple(sorted(SEEDS))
        or len(by_seed) != len(results)
        or any(
            row.get("protocol") != SEED_RESULT_PROTOCOL
            or row.get("parent_protocol") != PROTOCOL
            or row.get("scenario") != SCENARIO
            or row.get("status") != "PASS"
            for row in results
        )
    ):
        raise ValueError("V157C seed result roster is incomplete or invalid")

    contrast_names = (
        "uniform_source_minus_target_only",
        "uniform_source_minus_source_label_placebo",
        "target_only_minus_phase_pressure",
        "uniform_source_minus_phase_pressure",
        "source_label_placebo_minus_phase_pressure",
    )
    seed_rows = {}
    for seed in SEEDS:
        windows = by_seed[seed]["windows"]
        seed_rows[str(seed)] = {
            "window_count": len(CHECKPOINTS_SEC),
            "mean_costs": {
                arm: float(
                    np.mean(
                        [
                            windows[str(checkpoint)]["costs"][arm]
                            for checkpoint in CHECKPOINTS_SEC
                        ]
                    )
                )
                for arm in ALL_ARMS
            },
            "mean_contrasts": {
                name: float(
                    np.mean(
                        [
                            windows[str(checkpoint)]["contrasts"][name]
                            for checkpoint in CHECKPOINTS_SEC
                        ]
                    )
                )
                for name in contrast_names
            },
            "source_target_disagreement_window_count": sum(
                bool(
                    windows[str(checkpoint)]["focal"][
                        "source_target_selected_actions_differ"
                    ]
                )
                for checkpoint in CHECKPOINTS_SEC
            ),
        }

    comparisons = {
        name: {
            "mean_of_seed_means": float(
                np.mean(
                    [seed_rows[str(seed)]["mean_contrasts"][name] for seed in SEEDS]
                )
            ),
            "seed_means": {
                str(seed): seed_rows[str(seed)]["mean_contrasts"][name]
                for seed in SEEDS
            },
            "improving_seed_count": sum(
                seed_rows[str(seed)]["mean_contrasts"][name] < 0.0
                for seed in SEEDS
            ),
        }
        for name in contrast_names
    }
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "status": "COMPLETE",
        "parent_protocol": PROTOCOL,
        "scenario": SCENARIO,
        "seeds": list(SEEDS),
        "checkpoints_sec": list(CHECKPOINTS_SEC),
        "native_simulation_count": sum(
            int(row["native_simulation_count"]) for row in results
        ),
        "seed_unit_results": seed_rows,
        "comparisons": comparisons,
        "source_target_disagreement_window_count": sum(
            row["source_target_disagreement_window_count"]
            for row in seed_rows.values()
        ),
        "identical_action_windows_retained_as_zero": True,
        "outcomes_used_for_checkpoint_or_focal_selection": False,
        "scientific_status": "targeted_development_diagnostic_complete",
        "claim_boundary": protocol["claim_boundary"],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation_parser = subparsers.add_parser("validate-inputs")
    validation_parser.add_argument("--protocol", type=Path, required=True)
    validation_parser.add_argument("--authorization", type=Path, required=True)
    validation_parser.add_argument("--runtime-bundle", type=Path, required=True)
    validation_parser.add_argument("--runtime-bundle-sha256", required=True)
    validation_parser.add_argument("--arm-manifest", type=Path, required=True)
    validation_parser.add_argument("--arm-manifest-sha256", required=True)

    seed_parser = subparsers.add_parser("run-seed")
    seed_parser.add_argument("--protocol", type=Path, required=True)
    seed_parser.add_argument("--authorization", type=Path, required=True)
    seed_parser.add_argument("--manifest", type=Path, required=True)
    seed_parser.add_argument("--conversion-root", type=Path, required=True)
    seed_parser.add_argument("--runtime-bundle", type=Path, required=True)
    seed_parser.add_argument("--runtime-bundle-sha256", required=True)
    seed_parser.add_argument("--arm-manifest", type=Path, required=True)
    seed_parser.add_argument("--arm-manifest-sha256", required=True)
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_parser.add_argument("--scratch-root", type=Path, required=True)
    seed_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--protocol", type=Path, required=True)
    aggregate_parser.add_argument(
        "--seed-results", nargs=len(SEEDS), type=Path, required=True
    )
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-inputs":
        result = validate_inputs(
            protocol_path=args.protocol,
            authorization_path=args.authorization,
            runtime_bundle_path=args.runtime_bundle,
            runtime_bundle_sha256=args.runtime_bundle_sha256,
            arm_manifest_path=args.arm_manifest,
            arm_manifest_sha256=args.arm_manifest_sha256,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "run-seed":
        result = run_seed(
            protocol_path=args.protocol,
            authorization_path=args.authorization,
            manifest_path=args.manifest,
            conversion_root=args.conversion_root,
            runtime_bundle_path=args.runtime_bundle,
            runtime_bundle_sha256=args.runtime_bundle_sha256,
            arm_manifest_path=args.arm_manifest,
            arm_manifest_sha256=args.arm_manifest_sha256,
            seed=args.seed,
            scratch_root=args.scratch_root,
            output=args.output,
        )
        return 0 if result["status"] == "PASS" else 2
    if args.output.exists():
        raise FileExistsError("refusing to overwrite V157C aggregate")
    protocol = _read_json(args.protocol)
    result = aggregate_seed_results(
        protocol, [_read_json(path) for path in args.seed_results]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol": result["protocol"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
