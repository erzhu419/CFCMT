"""Native-prefix action-ranking calibration for the final Jinan remediation.

V158 collects one hundred action groups on ten new development seeds.  Every
non-reference action is evaluated from a fresh t=0 PhasePressure prefix, so no
SUMO state reload enters the labels.  A single predeclared causal residual
regressor is cross-fitted with target, true-source, or source-label-placebo
prediction columns.  The fitted policies retain PhasePressure as an explicit
zero-cost fallback and forbid a learned stay action from cancelling a PP
switch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_CORE_FAMILY_V3,
    CONTRAST_FEATURES_V3,
    _controlled_lane_set,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import (
    RUNTIME_BUNDLE_PROTOCOL,
    RUNTIME_POLICY,
    V157CActionDecision,
    _checkpoint_scorable_tls_ids,
    _compact_metrics,
    _load_arm_models,
    _new_window_record,
    _record_step_diagnostics,
    _runtime_models,
    _sample,
    _window_summary,
    compare_physical,
    force_phase_pressure,
    phase_pressure_decision,
    physical_snapshot,
    score_model_decision,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.action_ranker import (
    CausalReferenceResidualRegressor,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
    PairwisePreferenceConfig,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import (
    load_mechanism_dataset,
    save_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PROTOCOL = "tsc-v158-jinan-native-prefix-ranking-calibration-v1"
SEED_RESULT_PROTOCOL = "tsc-v158-jinan-native-prefix-seed-bank-v1"
FIT_RESULT_PROTOCOL = "tsc-v158-jinan-native-prefix-ranking-fit-v1"
CALIBRATOR_BUNDLE_PROTOCOL = "cfcmt-v158-jinan-native-prefix-calibrators-v1"
RESERVE_SEED_PROTOCOL = "tsc-v158-jinan-native-prefix-reserve-seed-v1"
RESERVE_AGGREGATE_PROTOCOL = "tsc-v158-jinan-native-prefix-reserve-aggregate-v1"

CONTROL_INTERVAL_SEC = 10
HORIZON_SEC = 450
BASE_ARMS = ("target_only", "uniform_source", "source_label_placebo")
CALIBRATED_ARMS = ("target_native", "source_native", "placebo_native")
ALL_RESERVE_ARMS = (*CALIBRATED_ARMS, "phase_pressure")
PREDICTION_FIELDS = ("score", "uncertainty", "trust")
RAW_PREDICTION_FEATURES = tuple(
    f"v157b_{arm}_{field}" for arm in BASE_ARMS for field in PREDICTION_FIELDS
)
TARGET_META_FEATURES = tuple(
    f"native_target_{field}" for field in PREDICTION_FIELDS
)
AUX_META_FEATURES = tuple(f"native_aux_{field}" for field in PREDICTION_FIELDS)
META_ACTION_FEATURES = (*TARGET_META_FEATURES, *AUX_META_FEATURES)
TOLERANCE = 1e-12
DEVELOPMENT_SEEDS = (
    180314,
    188625,
    127178,
    177524,
    163775,
    150945,
    173412,
    161481,
    150744,
    134310,
)
CROSSFIT_FOLDS = tuple(
    DEVELOPMENT_SEEDS[index : index + 2] for index in range(0, 10, 2)
)
SCENARIO_TIME_BINS = {
    "jinan_3x4_real": ((300, 600), (1050, 1350), (1800, 2100), (2550, 2850)),
    "jinan_3x4_real_2000": ((300, 600), (1050, 1350), (1800, 2100)),
    "jinan_3x4_real_2500": ((300, 600), (1050, 1350), (1800, 2100)),
}
RESERVE_SEEDS = tuple(seed + 100000 for seed in DEVELOPMENT_SEEDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _execution_snapshot_identity() -> dict[str, str] | None:
    root_value = os.environ.get("CFCMT_SOURCE_ROOT")
    if not root_value:
        return None
    root = Path(root_value).resolve()
    marker = _read_json(root / ".cfcmt_snapshot.json")
    return {
        "root": str(root),
        "protocol": str(marker.get("protocol", "")),
        "snapshot_sha256": str(marker.get("snapshot_sha256", "")),
        "source_tree_sha256": str(marker.get("source_tree_sha256", "")),
    }


def _existing_result(
    path: Path,
    *,
    protocol: str,
    statuses: set[str],
    seed: int | None = None,
) -> dict[str, Any] | None:
    """Return an intact terminal result, or None for an absent/partial result."""

    try:
        result = _read_json(path)
        if (
            result.get("protocol") != protocol
            or result.get("status") not in statuses
            or (seed is not None and int(result.get("seed", -1)) != int(seed))
        ):
            return None
        current_snapshot = _execution_snapshot_identity()
        if (
            current_snapshot is not None
            and result.get("execution_snapshot") != current_snapshot
        ):
            return None
        if protocol == SEED_RESULT_PROTOCOL:
            bank = Path(str(result["bank"]))
            if not bank.is_file() or _sha256(bank) != result["bank_sha256"]:
                return None
        elif protocol == FIT_RESULT_PROTOCOL:
            for key in ("native_b100", "calibrators"):
                artifact = result["artifacts"][key]
                artifact_path = Path(str(artifact["path"]))
                if not artifact_path.is_file() or _sha256(artifact_path) != artifact["sha256"]:
                    return None
            if not Path(str(result["artifacts"]["oof_server_only"])).is_file():
                return None
        return result
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _prepare_directory_run(
    *,
    output: Path,
    scratch: Path | None,
    reuse_complete: bool,
    protocol: str,
    statuses: set[str],
    seed: int | None = None,
) -> dict[str, Any] | None:
    if not reuse_complete:
        return None
    existing = _existing_result(
        output / "result.json", protocol=protocol, statuses=statuses, seed=seed
    )
    if existing is not None:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        return existing
    shutil.rmtree(output, ignore_errors=True)
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    return None


def _strict_time(states: Mapping[str, Any]) -> int:
    values = {float(state.sim_time) for state in states.values()}
    if len(values) != 1:
        raise ValueError("V158 TLS states do not share one simulation time")
    value = values.pop()
    rounded = int(round(value))
    if abs(value - rounded) > TOLERANCE:
        raise ValueError("V158 requires integral-second controller states")
    return rounded


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    development = dict(protocol.get("development", {}))
    crossfit = dict(protocol.get("crossfit", {}))
    model = dict(protocol.get("model", {}))
    action = dict(protocol.get("action_selection", {}))
    estimand = dict(protocol.get("native_estimand", {}))
    reserve = dict(protocol.get("reserve", {}))
    scenarios = dict(development.get("scenario_time_bins_sec", {}))
    seeds = tuple(int(value) for value in development.get("seeds", ()))
    folds = tuple(
        tuple(int(value) for value in fold) for fold in crossfit.get("folds", ())
    )
    flattened = tuple(value for fold in folds for value in fold)
    group_count = len(seeds) * sum(len(value) for value in scenarios.values())
    expected_branches = group_count * 7
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("scientific_status") != "prepared_not_run"
        or protocol.get("city") != "jinan"
        or seeds != DEVELOPMENT_SEEDS
        or folds != CROSSFIT_FOLDS
        or flattened != DEVELOPMENT_SEEDS
        or {
            str(name): tuple((int(start), int(end)) for start, end in bins)
            for name, bins in scenarios.items()
        }
        != SCENARIO_TIME_BINS
        or any(
            int(start) % CONTROL_INTERVAL_SEC
            or int(end) % CONTROL_INTERVAL_SEC
            or int(start) < 0
            or int(end) <= int(start)
            for bins in scenarios.values()
            for start, end in bins
        )
        or int(development.get("groups_per_seed", -1)) != 10
        or int(development.get("action_groups", -1)) != group_count
        or group_count != 100
        or int(development.get("candidate_actions_per_group", -1)) != 8
        or int(development.get("expected_native_action_branches", -1))
        != expected_branches
        or expected_branches != 700
        or tuple(protocol.get("v157b_runtime", {}).get("base_arms", ()))
        != BASE_ARMS
        or protocol.get("v157b_runtime", {}).get("bundle_protocol")
        != RUNTIME_BUNDLE_PROTOCOL
        or tuple(model.get("arms", ())) != CALIBRATED_ARMS
        or model.get("family") != "CausalReferenceResidualRegressor"
        or float(crossfit.get("threshold", float("nan"))) != 0.0
        or action.get("reference_policy") != "phase_pressure"
        or action.get("no_stay_override_of_pp_switch") is not True
        or estimand.get("construction")
        != "fresh_t0_phase_pressure_prefix_for_every_nonreference_action"
        or estimand.get("snapshot_restore_used") is not False
        or int(estimand.get("action_duration_sec", -1)) != 10
        or int(estimand.get("continuation_duration_sec", -1)) != 440
        or int(estimand.get("horizon_sec", -1)) != HORIZON_SEC
        or int(estimand.get("samples", -1)) != HORIZON_SEC
        or tuple(reserve.get("arms", ())) != ALL_RESERVE_ARMS
        or int(reserve.get("duration_sec", -1)) != 3600
        or reserve.get("residual_coordination_mode") != "sparse"
        or int(reserve.get("residual_cooldown_intervals", -1)) != 44
        or reserve.get("no_stay_override_of_pp_switch") is not True
        or tuple(int(value) for value in reserve.get("seeds", ()))
        != RESERVE_SEEDS
    ):
        raise ValueError("V158 frozen protocol changed")


def _prediction_vectors(model: Any, dataset: MechanismDataset) -> dict[str, np.ndarray]:
    score, uncertainty, trust, _ = _group_adjusted_scores(
        dataset,
        model.predict(dataset),
        objective_mode="control_only",
    )
    result = {
        "score": np.asarray(score, dtype=float),
        "uncertainty": np.asarray(uncertainty, dtype=float),
        "trust": np.asarray(trust, dtype=float),
    }
    if any(values.shape != (dataset.size,) for values in result.values()):
        raise ValueError("V158 base model returned misaligned prediction arrays")
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("V158 base model returned nonfinite prediction arrays")
    return result


def _copy_dataset(dataset: MechanismDataset) -> MechanismDataset:
    return MechanismDataset(
        feature_names=tuple(dataset.feature_names),
        features=np.asarray(dataset.features, dtype=float).copy(),
        context_names=tuple(dataset.context_names),
        context=np.asarray(dataset.context, dtype=float).copy(),
        priors={name: np.asarray(values, dtype=float).copy() for name, values in dataset.priors.items()},
        targets={name: np.asarray(values, dtype=float).copy() for name, values in dataset.targets.items()},
        domains=np.asarray(dataset.domains, dtype=str).copy(),
        metadata=dict(dataset.metadata),
    )


def _focal_tls(
    roster: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    tls_ids = tuple(sorted(str(value) for value in roster))
    if not tls_ids:
        raise ValueError("V158 focal selection requires a nonempty roster")
    disagreements = tuple(
        tls_id
        for tls_id in tls_ids
        if roster[tls_id]["uniform_source"]["selected_state"]
        != roster[tls_id]["target_only"]["selected_state"]
    )
    eligible = disagreements or tls_ids

    def key(tls_id: str) -> tuple[float, float, str]:
        source = roster[tls_id]["uniform_source"]
        target = roster[tls_id]["target_only"]
        return (
            -max(float(source["priority"]), float(target["priority"])),
            -float(source["priority"]),
            str(tls_id),
        )

    selected = min(eligible, key=key)
    return {
        "tls_id": selected,
        "source_target_disagreement": bool(disagreements),
        "disagreement_tls_count": len(disagreements),
        "selection_basis": (
            "source_target_disagreement_then_maximum_predicted_advantage"
            if disagreements
            else "maximum_predicted_advantage"
        ),
    }


class DevelopmentProbeOriginator:
    """Select the first complete eight-action state in each frozen time bin."""

    selection_layer = "v158_native_development_probe"

    def __init__(
        self,
        *,
        api: Any,
        models: Mapping[str, Any],
        scenario: str,
        seed: int,
        bins: Sequence[Sequence[int]],
    ) -> None:
        if tuple(models) != BASE_ARMS:
            raise ValueError("V158 base arm order changed")
        self.api = api
        self.models = dict(models)
        self.scenario = str(scenario)
        self.seed = int(seed)
        self.bins = tuple((int(start), int(end)) for start, end in bins)
        self.selected: list[dict[str, Any]] = []
        self.controlled_lanes: tuple[str, ...] = ()
        self.tls_ids: tuple[str, ...] = ()
        self._slot: int | None = None
        self._time = -1
        self._expected: tuple[str, ...] = ()
        self._executors: Mapping[str, Any] = {}
        self._decisions: dict[str, dict[str, dict[str, Any]]] = {}
        self._datasets: dict[str, MechanismDataset] = {}

    def prepare_interval(self, **kwargs: Any) -> None:
        states = dict(kwargs["states"])
        executors = dict(kwargs["executors"])
        infos = {tls_id: state.info for tls_id, state in states.items()}
        lanes = tuple(sorted(_controlled_lane_set(infos)))
        tls_ids = tuple(sorted(states))
        if self.controlled_lanes and lanes != self.controlled_lanes:
            raise ValueError("V158 controlled lanes changed within a trajectory")
        if self.tls_ids and tls_ids != self.tls_ids:
            raise ValueError("V158 TLS roster changed within a trajectory")
        self.controlled_lanes = lanes
        self.tls_ids = tls_ids
        self._time = _strict_time(states)
        self._executors = executors
        self._decisions = {}
        self._datasets = {}
        self._slot = next(
            (
                index
                for index, (start, end) in enumerate(self.bins)
                if index >= len(self.selected) and start <= self._time < end
            ),
            None,
        )
        self._expected = (
            _checkpoint_scorable_tls_ids(states, executors)
            if self._slot is not None
            else ()
        )

    def select(self, dataset: MechanismDataset, *, reference_index: int) -> V157CActionDecision:
        if self._slot is None:
            return phase_pressure_decision(
                dataset,
                reference_index=reference_index,
                arm="phase_pressure",
                rejection="outside_unfilled_v158_bin",
            )
        tls_values = tuple(str(value) for value in dataset.metadata.get("row_tls", ()))
        if len(set(tls_values)) != 1:
            raise ValueError("V158 probe received a multi-TLS action dataset")
        tls_id = tls_values[0]
        decisions = {
            arm: score_model_decision(
                self.models[arm], dataset, reference_index=reference_index, arm=arm
            ).to_dict()
            for arm in BASE_ARMS
        }
        self._decisions[tls_id] = decisions
        self._datasets[tls_id] = _copy_dataset(dataset)
        driver = V157CActionDecision(**decisions["target_only"])
        if tuple(sorted(self._decisions)) == self._expected:
            complete = {
                key: value
                for key, value in self._decisions.items()
                if self._datasets[key].size == 8
            }
            if complete and self._slot == len(self.selected):
                focal = _focal_tls(complete)
                selected_tls = focal["tls_id"]
                selected_decisions = complete[selected_tls]
                group_id = (
                    f"{self.scenario}:seed{self.seed}:slot{self._slot}:"
                    f"t{self._time}:{selected_tls}"
                )
                dataset_copy = self._datasets[selected_tls]
                states = tuple(
                    str(value)
                    for value in dataset_copy.metadata.get("candidate_states", ())
                )
                if len(states) != 8 or len(set(states)) != 8:
                    raise ValueError("V158 selected action group is not eight distinct states")
                vectors = {
                    arm: {
                        key: values.tolist()
                        for key, values in _prediction_vectors(
                            self.models[arm], dataset_copy
                        ).items()
                    }
                    for arm in BASE_ARMS
                }
                self.selected.append(
                    {
                        "group_id": group_id,
                        "slot": self._slot,
                        "scenario": self.scenario,
                        "seed": self.seed,
                        "checkpoint_sec": self._time,
                        "tls_id": selected_tls,
                        "reference_index": int(
                            selected_decisions["target_only"]["reference_index"]
                        ),
                        "candidate_states": list(states),
                        "focal_selection": focal,
                        "base_decisions": selected_decisions,
                        "base_predictions": vectors,
                        "dataset": dataset_copy,
                        "baseline_snapshot": physical_snapshot(self.api, self._executors),
                        "baseline_window": _new_window_record(),
                    }
                )
        return force_phase_pressure(driver, reason="v158_native_probe_executes_pp")


class FixedActionBranchOriginator:
    """Apply one frozen candidate state at one focal TLS, then return to PP."""

    selection_layer = "v158_native_fixed_action_branch"

    def __init__(self, group: Mapping[str, Any], candidate_index: int) -> None:
        self.group = group
        self.candidate_index = int(candidate_index)
        self.checkpoint_sec = int(group["checkpoint_sec"])
        self.focal_tls = str(group["tls_id"])
        self.candidate_state = str(group["candidate_states"][self.candidate_index])
        self.reference_index = int(group["reference_index"])
        self.reference_state = str(group["candidate_states"][self.reference_index])
        self.controlled_lanes: tuple[str, ...] = ()
        self.dataset_match: dict[str, Any] | None = None
        self.decision: dict[str, Any] | None = None

    def prepare_interval(self, **kwargs: Any) -> None:
        states = dict(kwargs["states"])
        infos = {tls_id: state.info for tls_id, state in states.items()}
        self.controlled_lanes = tuple(sorted(_controlled_lane_set(infos)))

    def select(self, dataset: MechanismDataset, *, reference_index: int) -> V157CActionDecision:
        tls_id = str(dataset.metadata["row_tls"][0])
        time_sec = float(dataset.metadata["row_times"][0])
        if int(round(time_sec)) != self.checkpoint_sec:
            return phase_pressure_decision(
                dataset,
                reference_index=reference_index,
                arm="v158_fixed_action",
                rejection="outside_fixed_native_checkpoint",
            )
        if tls_id != self.focal_tls:
            return phase_pressure_decision(
                dataset,
                reference_index=reference_index,
                arm="v158_fixed_action",
                rejection="nonfocal_tls",
            )
        expected = self.group["dataset"]
        states = tuple(str(value) for value in dataset.metadata["candidate_states"])
        features_equal = bool(np.array_equal(dataset.features, expected.features))
        context_equal = bool(np.array_equal(dataset.context, expected.context))
        self.dataset_match = {
            "feature_names_exact": tuple(dataset.feature_names) == tuple(expected.feature_names),
            "context_names_exact": tuple(dataset.context_names) == tuple(expected.context_names),
            "features_exact": features_equal,
            "context_exact": context_equal,
            "candidate_states_exact": states == tuple(self.group["candidate_states"]),
            "reference_index_exact": int(reference_index) == self.reference_index,
        }
        if not all(self.dataset_match.values()):
            raise ValueError(f"{self.group['group_id']}: native branch start dataset changed")
        selected = self.candidate_index
        decision = V157CActionDecision(
            reference_index=int(reference_index),
            proposed_index=selected,
            selected_index=selected,
            learned_differs=selected != int(reference_index),
            eligible=selected != int(reference_index),
            priority=1.0 if selected != int(reference_index) else 0.0,
            rejection=None,
            arm="v158_fixed_action",
            tls_id=tls_id,
            time_sec=time_sec,
            proposed_state=states[selected],
            selected_state=states[selected],
            reference_state=states[int(reference_index)],
            predicted_score=-1.0 if selected != int(reference_index) else 0.0,
            reference_score=0.0,
            forced_reference=False,
        )
        self.decision = decision.to_dict()
        return decision


def _evaluate_originator(
    *,
    api: Any,
    v3: Any,
    sumocfg: Path,
    scenario: str,
    seed: int,
    duration_sec: int,
    originator: Any,
    observer: Any,
    tripinfo_path: Path,
    residual_coordination_mode: str = "direct",
    cooldown_intervals: int = 0,
) -> dict[str, Any]:
    try:
        return v3.evaluate_policy_v3(
            sumo_api=api,
            sumocfg=sumocfg,
            scenario=str(scenario),
            policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=HORIZON_SEC),
            duration_sec=int(duration_sec),
            control_interval_sec=CONTROL_INTERVAL_SEC,
            warmup_sec=0,
            seed=int(seed),
            tripinfo_output=tripinfo_path,
            residual_coordination_mode=str(residual_coordination_mode),
            residual_cooldown_intervals_override=int(cooldown_intervals),
            step_observer=observer,
        )
    finally:
        tripinfo_path.unlink(missing_ok=True)


def _record_selected_windows(
    *, api: Any, probe: DevelopmentProbeOriginator, time_sec: float
) -> None:
    now = int(round(float(time_sec)))
    if abs(float(time_sec) - now) > TOLERANCE:
        return
    for group in probe.selected:
        checkpoint = int(group["checkpoint_sec"])
        if checkpoint < now <= checkpoint + HORIZON_SEC:
            record = group["baseline_window"]
            record["samples"].append(_sample(api, probe.controlled_lanes, float(now)))
            _record_step_diagnostics(api, record)


def _scenario_paths(manifest_path: Path) -> dict[str, Path]:
    manifest = load_traffic_signal_manifest(manifest_path)
    return {
        spec.scenario: Path(spec.sumocfg)
        for spec in manifest.scenarios
        if spec.city_group == "jinan"
    }


def _build_seed_bank(groups: Sequence[Mapping[str, Any]]) -> MechanismDataset:
    if not groups:
        raise ValueError("V158 cannot build an empty seed bank")
    base = groups[0]["dataset"]
    feature_names = (*base.feature_names, *RAW_PREDICTION_FEATURES)
    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    domains: list[str] = []
    absolute_costs: list[float] = []
    relative_costs: list[float] = []
    metadata: dict[str, list[Any]] = {
        "action_group_ids": [],
        "is_reference": [],
        "reference_rows": [],
        "row_tls": [],
        "row_times": [],
        "candidate_states": [],
        "row_seeds": [],
        "row_scenarios": [],
    }
    for group in groups:
        dataset = group["dataset"]
        if (
            dataset.size != 8
            or tuple(dataset.feature_names) != tuple(base.feature_names)
            or tuple(dataset.context_names) != tuple(base.context_names)
        ):
            raise ValueError("V158 seed groups changed feature or action schemas")
        reference = int(group["reference_index"])
        outcomes = dict(group["native_outcomes"])
        if set(outcomes) != set(range(8)):
            raise ValueError(f"{group['group_id']}: native outcomes are incomplete")
        costs = np.asarray([outcomes[index]["cost"] for index in range(8)], dtype=float)
        if not np.all(np.isfinite(costs)):
            raise ValueError(f"{group['group_id']}: native outcomes are nonfinite")
        prediction_columns = []
        for arm in BASE_ARMS:
            values = group["base_predictions"][arm]
            prediction_columns.extend(
                np.asarray(values[field], dtype=float) for field in PREDICTION_FIELDS
            )
        augmented = np.column_stack([dataset.features, *prediction_columns])
        features.append(augmented)
        contexts.append(np.asarray(dataset.context, dtype=float))
        domains.extend([str(group["scenario"])] * 8)
        absolute_costs.extend(costs.tolist())
        relative_costs.extend((costs - costs[reference]).tolist())
        offset = len(metadata["action_group_ids"])
        metadata["action_group_ids"].extend([str(group["group_id"])] * 8)
        metadata["is_reference"].extend([index == reference for index in range(8)])
        metadata["reference_rows"].extend([offset + reference] * 8)
        metadata["row_tls"].extend([str(group["tls_id"])] * 8)
        metadata["row_times"].extend([float(group["checkpoint_sec"])] * 8)
        metadata["candidate_states"].extend(group["candidate_states"])
        metadata["row_seeds"].extend([int(group["seed"])] * 8)
        metadata["row_scenarios"].extend([str(group["scenario"])] * 8)
    size = len(absolute_costs)
    metadata.update(
        {
            "protocol": SEED_RESULT_PROTOCOL,
            "native_prefix": True,
            "snapshot_restore_used": False,
            "action_group_count": len(groups),
            "candidate_actions_per_group": 8,
        }
    )
    return MechanismDataset(
        feature_names=tuple(feature_names),
        features=np.vstack(features),
        context_names=tuple(base.context_names),
        context=np.vstack(contexts),
        priors={"interval_cost": np.zeros(size, dtype=float)},
        targets={
            "interval_cost": np.asarray(relative_costs, dtype=float),
            "absolute_cost": np.asarray(absolute_costs, dtype=float),
        },
        domains=np.asarray(domains, dtype=str),
        metadata=metadata,
    )


def collect_development_seed(
    *,
    protocol_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    seed: int,
    scratch_root: Path,
    output: Path,
) -> dict[str, Any]:
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3

    started = time.monotonic()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    seeds = tuple(int(value) for value in protocol["development"]["seeds"])
    if int(seed) not in seeds:
        raise ValueError("V158 development seed is outside the frozen roster")
    if output.exists() or scratch_root.exists():
        raise FileExistsError("refusing to overwrite V158 collection output or scratch")
    if _sha256(manifest_path) != protocol["network_manifest"]["sha256"]:
        raise ValueError("V158 network manifest identity changed")
    if _sha256(runtime_bundle_path) != str(runtime_bundle_sha256):
        raise ValueError("V158 base runtime bundle identity changed")
    if str(runtime_bundle_sha256) != protocol["v157b_runtime"]["bundle_sha256"]:
        raise ValueError("V158 base runtime hash differs from the frozen protocol")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    paths = _scenario_paths(manifest_path)
    bins_by_scenario = protocol["development"]["scenario_time_bins_sec"]
    if set(paths) != set(bins_by_scenario):
        raise ValueError("V158 Jinan scenario roster changed")
    models = _load_arm_models(runtime_bundle_path, runtime_bundle_sha256)
    if tuple(models) != BASE_ARMS:
        raise ValueError("V158 base runtime arm order changed")
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("V158 requires SUMO 1.22.0")
    scratch_root.mkdir(parents=True)
    output.mkdir(parents=True)
    groups: list[dict[str, Any]] = []
    baseline_runtime: dict[str, Any] = {}
    branch_count = 0
    for scenario in bins_by_scenario:
        bins = bins_by_scenario[scenario]
        probe = DevelopmentProbeOriginator(
            api=api,
            models=models,
            scenario=scenario,
            seed=int(seed),
            bins=bins,
        )

        def baseline_observer(*, stage: str, time_sec: float, executors: Any) -> None:
            if stage == "after_executor_advance":
                _record_selected_windows(api=api, probe=probe, time_sec=time_sec)

        duration = max(int(end) for _, end in bins) + HORIZON_SEC
        metrics = _evaluate_originator(
            api=api,
            v3=v3,
            sumocfg=paths[scenario],
            scenario=scenario,
            seed=int(seed),
            duration_sec=duration,
            originator=probe,
            observer=baseline_observer,
            tripinfo_path=scratch_root / f"{scenario}_pp.xml",
        )
        baseline_runtime[scenario] = _compact_metrics(metrics)
        if len(probe.selected) != len(bins):
            raise ValueError(f"{scenario}: V158 did not fill every frozen time bin")
        for group in probe.selected:
            pp_summary = _window_summary(
                group["baseline_window"], group["baseline_snapshot"]
            )
            pp_checks = {
                "baseline_complete": metrics.get("ok") is True,
                "no_baseline_learned_interventions": metrics.get(
                    "accepted_intervention_trace", []
                )
                == [],
                "full_450_second_window": pp_summary["sample_count"] == HORIZON_SEC,
                "active_population_accounting": pp_summary[
                    "active_population_accounting_passed"
                ],
                "no_prefix_or_window_teleport": (
                    int(metrics.get("starting_teleports", -1)) == 0
                    and int(metrics.get("ending_teleports", -1)) == 0
                    and pp_summary["starting_teleports"] == 0
                    and pp_summary["ending_teleports"] == 0
                ),
            }
            if not all(pp_checks.values()):
                print(
                    json.dumps(
                        {
                            "v158_invalid_baseline": group["group_id"],
                            "checks": pp_checks,
                            "summary": pp_summary,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                raise ValueError(f"{group['group_id']}: invalid PP baseline window")
            group["phase_pressure"] = {**pp_summary, "checks": pp_checks}
            group["native_outcomes"] = {
                int(group["reference_index"]): {
                    "cost": pp_summary["mean_halted_per_controlled_lane"],
                    "summary": pp_summary,
                    "source": "uninterrupted_phase_pressure_baseline",
                }
            }
            for candidate_index in range(8):
                if candidate_index == int(group["reference_index"]):
                    continue
                originator = FixedActionBranchOriginator(group, candidate_index)
                record = _new_window_record()
                branch_snapshot: dict[str, Any] = {}

                def branch_observer(
                    *, stage: str, time_sec: float, executors: Any
                ) -> None:
                    if stage != "after_executor_advance":
                        return
                    now = int(round(float(time_sec)))
                    checkpoint = int(group["checkpoint_sec"])
                    if now == checkpoint:
                        branch_snapshot.update(physical_snapshot(api, executors))
                    if checkpoint < now <= checkpoint + HORIZON_SEC:
                        record["samples"].append(
                            _sample(api, originator.controlled_lanes, float(now))
                        )
                        _record_step_diagnostics(api, record)

                branch_metrics = _evaluate_originator(
                    api=api,
                    v3=v3,
                    sumocfg=paths[scenario],
                    scenario=scenario,
                    seed=int(seed),
                    duration_sec=int(group["checkpoint_sec"]) + HORIZON_SEC,
                    originator=originator,
                    observer=branch_observer,
                    tripinfo_path=(
                        scratch_root
                        / f"{scenario}_slot{group['slot']}_action{candidate_index}.xml"
                    ),
                )
                summary = _window_summary(record, branch_snapshot)
                physical = compare_physical(
                    group["baseline_snapshot"], branch_snapshot, tolerance=0.0
                )
                trace = list(branch_metrics.get("accepted_intervention_trace", []))
                checks = {
                    "evaluator_complete": branch_metrics.get("ok") is True,
                    "physical_start_exact": physical["passed"],
                    "checkpoint_dataset_exact": bool(
                        originator.dataset_match
                        and all(originator.dataset_match.values())
                    ),
                    "fixed_action_decision_recorded": originator.decision is not None,
                    "full_450_second_window": summary["sample_count"] == HORIZON_SEC,
                    "active_population_accounting": summary[
                        "active_population_accounting_passed"
                    ],
                    "no_prefix_or_window_teleport": (
                        int(branch_metrics.get("starting_teleports", -1)) == 0
                        and int(branch_metrics.get("ending_teleports", -1)) == 0
                        and summary["starting_teleports"] == 0
                        and summary["ending_teleports"] == 0
                    ),
                    "one_effective_focal_intervention": (
                        len(trace) == 1
                        and float(trace[0].get("time_sec", -1))
                        == float(group["checkpoint_sec"])
                        and str(trace[0].get("tls_id")) == str(group["tls_id"])
                        and trace[0].get("execution_effective") is True
                    ),
                }
                if not all(checks.values()):
                    print(
                        json.dumps(
                            {
                                "v158_invalid_native_branch": group["group_id"],
                                "candidate_index": candidate_index,
                                "checks": checks,
                                "summary": summary,
                                "dataset_match": originator.dataset_match,
                                "physical_start_comparison": physical,
                                "accepted_intervention_trace": trace,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    raise ValueError(
                        f"{group['group_id']} action {candidate_index}: invalid native branch"
                    )
                group["native_outcomes"][candidate_index] = {
                    "cost": summary["mean_halted_per_controlled_lane"],
                    "summary": summary,
                    "checks": checks,
                    "physical_start_comparison": physical,
                    "runtime": _compact_metrics(branch_metrics),
                    "source": "fresh_t0_one_action_then_phase_pressure",
                }
                branch_count += 1
                print(
                    json.dumps(
                        {
                            "seed": int(seed),
                            "scenario": scenario,
                            "completed_native_branches": branch_count,
                            "expected_native_branches": 70,
                            "group": group["group_id"],
                            "action": candidate_index,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            groups.append(group)
    if len(groups) != int(protocol["development"]["groups_per_seed"]):
        raise ValueError("V158 seed action-group count changed")
    if branch_count != 7 * len(groups):
        raise ValueError("V158 native non-reference branch count changed")
    bank = _build_seed_bank(groups)
    bank_path = output / "native_bank.npz"
    save_mechanism_dataset(bank_path, bank)
    compact_groups = []
    full_traces: dict[str, Any] = {}
    for group in groups:
        compact_groups.append(
            {
                "group_id": group["group_id"],
                "slot": group["slot"],
                "scenario": group["scenario"],
                "seed": group["seed"],
                "checkpoint_sec": group["checkpoint_sec"],
                "tls_id": group["tls_id"],
                "reference_index": group["reference_index"],
                "candidate_states": group["candidate_states"],
                "focal_selection": group["focal_selection"],
                "base_decisions": group["base_decisions"],
                "native_costs": [
                    group["native_outcomes"][index]["cost"] for index in range(8)
                ],
                "collision_events_by_action": [
                    group["native_outcomes"][index]["summary"]["collision_events"]
                    for index in range(8)
                ],
            }
        )
        full_traces[group["group_id"]] = {
            "baseline_snapshot": group["baseline_snapshot"],
            "baseline_samples": group["baseline_window"]["samples"],
            "native_outcomes": group["native_outcomes"],
            "base_predictions": group["base_predictions"],
        }
    (output / "full_traces_server_only.pkl").write_bytes(
        pickle.dumps(full_traces, protocol=pickle.HIGHEST_PROTOCOL)
    )
    result = {
        "protocol": SEED_RESULT_PROTOCOL,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": PROTOCOL,
        "execution_snapshot": _execution_snapshot_identity(),
        "seed": int(seed),
        "scenarios": list(bins_by_scenario),
        "action_groups": len(groups),
        "candidate_rows": bank.size,
        "native_nonreference_branches": branch_count,
        "baseline_trajectories": len(bins_by_scenario),
        "snapshot_restores": 0,
        "bank": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "groups": compact_groups,
        "baseline_runtime": baseline_runtime,
        "collision_handling": "diagnostic_not_selection_or_primary_gate",
        "full_traces_server_only": str(output / "full_traces_server_only.pkl"),
        "elapsed_sec": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "seed": result["seed"],
                "action_groups": result["action_groups"],
                "native_nonreference_branches": branch_count,
                "elapsed_sec": result["elapsed_sec"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def _merge_banks(banks: Sequence[MechanismDataset]) -> MechanismDataset:
    if not banks:
        raise ValueError("V158 fit requires development banks")
    base = banks[0]
    if any(
        bank.feature_names != base.feature_names
        or bank.context_names != base.context_names
        or set(bank.priors) != set(base.priors)
        or set(bank.targets) != set(base.targets)
        for bank in banks
    ):
        raise ValueError("V158 development bank schemas differ")
    row_keys = (
        "action_group_ids",
        "is_reference",
        "row_tls",
        "row_times",
        "candidate_states",
        "row_seeds",
        "row_scenarios",
    )
    metadata = {
        key: [value for bank in banks for value in bank.metadata[key]]
        for key in row_keys
    }
    references = []
    offset = 0
    for bank in banks:
        references.extend(
            [offset + int(value) for value in bank.metadata["reference_rows"]]
        )
        offset += bank.size
    metadata.update(
        {
            "reference_rows": references,
            "protocol": PROTOCOL,
            "native_prefix": True,
            "snapshot_restore_used": False,
            "action_group_count": sum(
                int(bank.metadata["action_group_count"]) for bank in banks
            ),
            "candidate_actions_per_group": 8,
        }
    )
    return MechanismDataset(
        feature_names=base.feature_names,
        features=np.vstack([bank.features for bank in banks]),
        context_names=base.context_names,
        context=np.vstack([bank.context for bank in banks]),
        priors={
            name: np.concatenate([bank.priors[name] for bank in banks])
            for name in base.priors
        },
        targets={
            name: np.concatenate([bank.targets[name] for bank in banks])
            for name in base.targets
        },
        domains=np.concatenate([bank.domains for bank in banks]),
        metadata=metadata,
    )


def _subset_groups(dataset: MechanismDataset, selected_groups: set[str]) -> MechanismDataset:
    groups = np.asarray(dataset.metadata["action_group_ids"], dtype=str)
    rows = np.flatnonzero(np.isin(groups, list(selected_groups)))
    if not len(rows):
        raise ValueError("V158 group subset is empty")
    index_map = {int(old): new for new, old in enumerate(rows.tolist())}
    row_keys = (
        "action_group_ids",
        "is_reference",
        "row_tls",
        "row_times",
        "candidate_states",
        "row_seeds",
        "row_scenarios",
    )
    metadata = {
        **{
            key: np.asarray(dataset.metadata[key], dtype=object)[rows].tolist()
            for key in row_keys
        },
        "reference_rows": [
            index_map[int(dataset.metadata["reference_rows"][old])] for old in rows
        ],
        "protocol": dataset.metadata.get("protocol"),
        "native_prefix": True,
        "snapshot_restore_used": False,
        "action_group_count": len(selected_groups),
        "candidate_actions_per_group": 8,
    }
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features[rows],
        context_names=dataset.context_names,
        context=dataset.context[rows],
        priors={name: np.asarray(values)[rows] for name, values in dataset.priors.items()},
        targets={name: np.asarray(values)[rows] for name, values in dataset.targets.items()},
        domains=dataset.domains[rows],
        metadata=metadata,
    )


def _arm_dataset(dataset: MechanismDataset, arm: str) -> MechanismDataset:
    if arm not in CALIBRATED_ARMS:
        raise ValueError(f"unknown V158 calibrated arm: {arm}")
    indices = {name: index for index, name in enumerate(dataset.feature_names)}
    raw = tuple(name for name in dataset.feature_names if name not in RAW_PREDICTION_FEATURES)
    raw_values = dataset.features[:, [indices[name] for name in raw]]
    target_values = np.column_stack(
        [dataset.features[:, indices[f"v157b_target_only_{field}"]] for field in PREDICTION_FIELDS]
    )
    if arm == "target_native":
        auxiliary = np.zeros_like(target_values)
    else:
        source_arm = (
            "uniform_source" if arm == "source_native" else "source_label_placebo"
        )
        auxiliary = np.column_stack(
            [dataset.features[:, indices[f"v157b_{source_arm}_{field}"]] for field in PREDICTION_FIELDS]
        )
    return MechanismDataset(
        feature_names=(*raw, *TARGET_META_FEATURES, *AUX_META_FEATURES),
        features=np.column_stack([raw_values, target_values, auxiliary]),
        context_names=dataset.context_names,
        context=dataset.context.copy(),
        priors={name: np.asarray(values).copy() for name, values in dataset.priors.items()},
        targets={name: np.asarray(values).copy() for name, values in dataset.targets.items()},
        domains=dataset.domains.copy(),
        metadata=dict(dataset.metadata),
    )


def _model_config(protocol: Mapping[str, Any]) -> PairwisePreferenceConfig:
    return PairwisePreferenceConfig(**dict(protocol["model"]["config"]))


def _fit_calibrator(protocol: Mapping[str, Any], dataset: MechanismDataset, arm: str):
    arm_data = _arm_dataset(dataset, arm)
    model = CausalReferenceResidualRegressor(
        config=_model_config(protocol),
        state_feature_names=PAIRWISE_CAUSAL_STATE_PARENTS,
        action_feature_names=(*PAIRWISE_CAUSAL_ACTION_PARENTS, *META_ACTION_FEATURES),
    )
    diagnostics = model.fit(arm_data)
    return model, diagnostics, arm_data


def _select_row(
    dataset: MechanismDataset,
    scores: np.ndarray,
    rows: np.ndarray,
    *,
    prohibit_stay_override: bool,
) -> int:
    references = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    reference_rows = rows[references[rows]]
    if reference_rows.size != 1:
        raise ValueError("V158 group requires exactly one PP reference row")
    reference = int(reference_rows[0])
    minimum = float(np.min(scores[rows]))
    tied = rows[np.abs(scores[rows] - minimum) <= TOLERANCE]
    selected = reference if reference in tied else int(tied[0])
    if float(scores[selected]) >= -TOLERANCE:
        return reference
    if prohibit_stay_override and selected != reference:
        feature_index = {name: index for index, name in enumerate(dataset.feature_names)}
        switch = feature_index["switch_indicator"]
        if (
            float(dataset.features[selected, switch]) <= TOLERANCE
            and float(dataset.features[reference, switch]) > TOLERANCE
        ):
            return reference
    return selected


def _policy_records(
    model: Any,
    dataset: MechanismDataset,
    *,
    arm: str,
    prohibit_stay_override: bool,
) -> list[dict[str, Any]]:
    arm_data = _arm_dataset(dataset, arm)
    scores = np.asarray(
        model.predict(arm_data)["control_cost"]["mean"], dtype=float
    )
    groups = np.asarray(dataset.metadata["action_group_ids"], dtype=str)
    seeds = np.asarray(dataset.metadata["row_seeds"], dtype=int)
    scenarios = np.asarray(dataset.metadata["row_scenarios"], dtype=str)
    states = np.asarray(dataset.metadata["candidate_states"], dtype=str)
    absolute = np.asarray(dataset.targets["absolute_cost"], dtype=float)
    result = []
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        reference = int(rows[np.asarray(dataset.metadata["is_reference"], dtype=bool)[rows]][0])
        selected = _select_row(
            arm_data,
            scores,
            rows,
            prohibit_stay_override=prohibit_stay_override,
        )
        result.append(
            {
                "group_id": group,
                "seed": int(seeds[rows[0]]),
                "scenario": str(scenarios[rows[0]]),
                "selected_row": int(selected),
                "reference_row": int(reference),
                "selected_state": str(states[selected]),
                "reference_state": str(states[reference]),
                "different_action": selected != reference,
                "predicted_advantage": max(float(-scores[selected]), 0.0),
                "actual_advantage": float(absolute[reference] - absolute[selected]),
                "selected_cost": float(absolute[selected]),
                "phase_pressure_cost": float(absolute[reference]),
            }
        )
    return result


def _seed_costs(records: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    return {
        int(seed): float(
            np.mean([row["selected_cost"] for row in records if int(row["seed"]) == seed])
        )
        for seed in sorted({int(row["seed"]) for row in records})
    }


def _pp_seed_costs(records: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    return {
        int(seed): float(
            np.mean(
                [row["phase_pressure_cost"] for row in records if int(row["seed"]) == seed]
            )
        )
        for seed in sorted({int(row["seed"]) for row in records})
    }


def _paired_summary(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keys = tuple(sorted(set(left) & set(right)))
    if set(keys) != set(left) or set(keys) != set(right) or not keys:
        raise ValueError("V158 paired comparison seed roster changed")
    differences = np.asarray([float(left[key]) - float(right[key]) for key in keys])
    rng = np.random.default_rng(int(seed))
    sampled = differences[
        rng.integers(0, len(differences), size=(int(draws), len(differences)))
    ].mean(axis=1)
    return {
        "seed_count": len(keys),
        "seeds": list(keys),
        "mean_difference": float(np.mean(differences)),
        "paired_bootstrap_95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "improved_seeds": int(np.count_nonzero(differences < -TOLERANCE)),
        "worse_seeds": int(np.count_nonzero(differences > TOLERANCE)),
        "equal_seeds": int(np.count_nonzero(np.abs(differences) <= TOLERANCE)),
        "per_seed": [
            {"seed": key, "difference": float(value)}
            for key, value in zip(keys, differences, strict=True)
        ],
    }


def _gate(
    protocol: Mapping[str, Any], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    spec = dict(protocol["oof_gate"])
    requirements = {
        "source_minus_target": float(spec["source_minus_target_mean_maximum"]),
        "source_minus_placebo": float(spec["source_minus_placebo_mean_maximum"]),
        "source_minus_phase_pressure": float(
            spec["source_minus_phase_pressure_mean_maximum"]
        ),
    }
    checks = {}
    for name, maximum in requirements.items():
        row = comparisons[name]
        checks[name] = {
            "mean_passed": float(row["mean_difference"]) <= maximum,
            "bootstrap_upper_passed": float(row["paired_bootstrap_95"][1])
            < float(spec["paired_bootstrap_upper_95_maximum"]),
            "seed_wins_passed": int(row["improved_seeds"])
            >= int(spec["minimum_improved_seeds"]),
        }
        checks[name]["passed"] = all(checks[name].values())
    return {
        "passed": all(row["passed"] for row in checks.values()),
        "checks": checks,
        "specification": spec,
    }


def fit_development(
    *, protocol_path: Path, collection_root: Path, output: Path
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    if output.exists():
        raise FileExistsError("refusing to overwrite V158 fit output")
    seeds = tuple(int(value) for value in protocol["development"]["seeds"])
    cells = []
    banks = []
    for seed in seeds:
        result_path = Path(collection_root) / f"seed_{seed}" / "result.json"
        result = _read_json(result_path)
        if (
            result.get("protocol") != SEED_RESULT_PROTOCOL
            or result.get("status") != "PASS"
            or int(result.get("seed", -1)) != seed
            or int(result.get("action_groups", -1)) != 10
            or int(result.get("candidate_rows", -1)) != 80
            or int(result.get("native_nonreference_branches", -1)) != 70
            or int(result.get("snapshot_restores", -1)) != 0
            or result.get("execution_snapshot") != _execution_snapshot_identity()
        ):
            raise ValueError(f"seed {seed}: V158 collection result is invalid")
        bank_path = Path(result["bank"])
        if _sha256(bank_path) != result["bank_sha256"]:
            raise ValueError(f"seed {seed}: V158 native bank identity changed")
        cells.append(result)
        banks.append(load_mechanism_dataset(bank_path))
    data = _merge_banks(banks)
    groups = tuple(dict.fromkeys(data.metadata["action_group_ids"]))
    if (
        data.size != 800
        or len(groups) != 100
        or any(data.metadata["action_group_ids"].count(group) != 8 for group in groups)
        or not np.all(np.isfinite(data.targets["interval_cost"]))
    ):
        raise ValueError("V158 merged native B100 is incomplete")
    output.mkdir(parents=True)
    bank_path = output / "native_b100.npz"
    save_mechanism_dataset(bank_path, data)
    oof_records = {arm: [] for arm in CALIBRATED_ARMS}
    folds = []
    all_groups = set(groups)
    for fold_index, heldout_seeds in enumerate(protocol["crossfit"]["folds"]):
        heldout = {
            group
            for group in groups
            if int(group.split(":seed", 1)[1].split(":", 1)[0])
            in set(map(int, heldout_seeds))
        }
        training = all_groups - heldout
        if len(heldout) != 20 or len(training) != 80:
            raise ValueError("V158 cross-fit seed fold changed")
        train_data = _subset_groups(data, training)
        heldout_data = _subset_groups(data, heldout)
        diagnostics = {}
        for arm in CALIBRATED_ARMS:
            model, diagnostics[arm], _ = _fit_calibrator(
                protocol, train_data, arm
            )
            records = _policy_records(
                model,
                heldout_data,
                arm=arm,
                prohibit_stay_override=True,
            )
            if len(records) != 20:
                raise ValueError("V158 OOF arm does not cover every held-out group")
            oof_records[arm].extend(records)
        folds.append(
            {
                "fold": fold_index,
                "heldout_seeds": list(map(int, heldout_seeds)),
                "training_group_count": len(training),
                "heldout_group_count": len(heldout),
                "fit_diagnostics": diagnostics,
            }
        )
    if any(len(rows) != 100 for rows in oof_records.values()):
        raise ValueError("V158 OOF coverage is incomplete")
    seed_costs = {arm: _seed_costs(rows) for arm, rows in oof_records.items()}
    pp_costs = _pp_seed_costs(oof_records["source_native"])
    gate_spec = protocol["oof_gate"]
    comparison_inputs = {
        "source_minus_target": (seed_costs["source_native"], seed_costs["target_native"]),
        "source_minus_placebo": (seed_costs["source_native"], seed_costs["placebo_native"]),
        "source_minus_phase_pressure": (seed_costs["source_native"], pp_costs),
    }
    comparisons = {
        name: _paired_summary(
            left,
            right,
            draws=int(gate_spec["bootstrap_draws"]),
            seed=int(gate_spec["bootstrap_seed"]),
        )
        for name, (left, right) in comparison_inputs.items()
    }
    authorization = _gate(protocol, comparisons)
    full_models = {}
    full_diagnostics = {}
    for arm in CALIBRATED_ARMS:
        full_models[arm], full_diagnostics[arm], _ = _fit_calibrator(
            protocol, data, arm
        )
    bundle_path = output / "calibrators.pkl"
    bundle_payload = {
        "protocol": CALIBRATOR_BUNDLE_PROTOCOL,
        "parent_protocol": PROTOCOL,
        "base_runtime_bundle_sha256": protocol["v157b_runtime"]["bundle_sha256"],
        "threshold": 0.0,
        "no_stay_override_of_pp_switch": True,
        "arms": full_models,
        "feature_contract": {
            "state": tuple(PAIRWISE_CAUSAL_STATE_PARENTS),
            "action": (*PAIRWISE_CAUSAL_ACTION_PARENTS, *META_ACTION_FEATURES),
            "target_prediction_features": TARGET_META_FEATURES,
            "auxiliary_prediction_features": AUX_META_FEATURES,
        },
        "development_seeds": seeds,
        "development_bank_sha256": _sha256(bank_path),
    }
    bundle_path.write_bytes(pickle.dumps(bundle_payload, protocol=pickle.HIGHEST_PROTOCOL))
    round_trip = pickle.loads(bundle_path.read_bytes())
    if (
        round_trip.get("protocol") != CALIBRATOR_BUNDLE_PROTOCOL
        or tuple(round_trip.get("arms", ())) != CALIBRATED_ARMS
    ):
        raise ValueError("V158 calibrator bundle round trip failed")
    oof_path = output / "oof_server_only.json"
    _write_json(oof_path, {"records": oof_records, "folds": folds})
    result = {
        "protocol": FIT_RESULT_PROTOCOL,
        "status": "PASS" if authorization["passed"] else "OOF_GATE_FAIL",
        "scientific_status": (
            "reserve_authorized" if authorization["passed"] else "remediation_closed_at_oof"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": PROTOCOL,
        "execution_snapshot": _execution_snapshot_identity(),
        "development": {
            "seeds": list(seeds),
            "action_groups": len(groups),
            "candidate_rows": data.size,
            "native_nonreference_branches": sum(
                int(cell["native_nonreference_branches"]) for cell in cells
            ),
            "baseline_trajectories": sum(
                int(cell["baseline_trajectories"]) for cell in cells
            ),
            "snapshot_restores": 0,
        },
        "oof": {
            "fold_count": len(folds),
            "unit": "whole_seed",
            "threshold": 0.0,
            "seed_costs": {
                arm: {str(key): value for key, value in rows.items()}
                for arm, rows in {**seed_costs, "phase_pressure": pp_costs}.items()
            },
            "mean_costs": {
                arm: float(np.mean(list(rows.values())))
                for arm, rows in {**seed_costs, "phase_pressure": pp_costs}.items()
            },
            "comparisons": comparisons,
            "gate": authorization,
        },
        "full_fit_diagnostics": full_diagnostics,
        "artifacts": {
            "native_b100": {
                "path": str(bank_path),
                "sha256": _sha256(bank_path),
                "size_bytes": bank_path.stat().st_size,
            },
            "calibrators": {
                "path": str(bundle_path),
                "sha256": _sha256(bundle_path),
                "size_bytes": bundle_path.stat().st_size,
                "protocol": CALIBRATOR_BUNDLE_PROTOCOL,
            },
            "oof_server_only": str(oof_path),
        },
        "reserve_authorized": bool(authorization["passed"]),
        "claim_boundary": protocol["claim_boundary"],
        "elapsed_sec": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "reserve_authorized": result["reserve_authorized"],
                "mean_costs": result["oof"]["mean_costs"],
                "comparisons": comparisons,
                "elapsed_sec": result["elapsed_sec"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def _augment_runtime_dataset(
    dataset: MechanismDataset,
    *,
    vectors: Mapping[str, Mapping[str, np.ndarray]],
    arm: str,
) -> MechanismDataset:
    target = vectors["target_only"]
    if arm == "target_native":
        auxiliary = {field: np.zeros(dataset.size, dtype=float) for field in PREDICTION_FIELDS}
    else:
        key = "uniform_source" if arm == "source_native" else "source_label_placebo"
        auxiliary = vectors[key]
    additions = [target[field] for field in PREDICTION_FIELDS] + [
        auxiliary[field] for field in PREDICTION_FIELDS
    ]
    return MechanismDataset(
        feature_names=(*dataset.feature_names, *META_ACTION_FEATURES),
        features=np.column_stack([dataset.features, *additions]),
        context_names=dataset.context_names,
        context=dataset.context.copy(),
        priors={name: np.asarray(values).copy() for name, values in dataset.priors.items()},
        targets={name: np.asarray(values).copy() for name, values in dataset.targets.items()},
        domains=dataset.domains.copy(),
        metadata=dict(dataset.metadata),
    )


class CalibratedRuntimeOriginator:
    """Score every feasible action with one frozen native calibrator."""

    selection_layer = "v158_native_calibrated_runtime"

    def __init__(self, *, base_models: Mapping[str, Any], calibrator: Any, arm: str) -> None:
        if arm not in CALIBRATED_ARMS or tuple(base_models) != BASE_ARMS:
            raise ValueError("V158 runtime arm roster changed")
        self.base_models = dict(base_models)
        self.calibrator = calibrator
        self.arm = str(arm)
        self.decisions = 0
        self.overrides = 0
        self.no_stay_vetoes = 0

    def select(self, dataset: MechanismDataset, *, reference_index: int) -> V157CActionDecision:
        vectors = {
            arm: _prediction_vectors(model, dataset)
            for arm, model in self.base_models.items()
        }
        augmented = _augment_runtime_dataset(dataset, vectors=vectors, arm=self.arm)
        scores = np.asarray(
            self.calibrator.predict(augmented)["control_cost"]["mean"], dtype=float
        )
        rows = np.arange(dataset.size, dtype=int)
        selected = _select_row(
            augmented, scores, rows, prohibit_stay_override=True
        )
        reference = int(reference_index)
        raw_minimum = int(np.argmin(scores))
        if selected == reference and raw_minimum != reference:
            index = {name: i for i, name in enumerate(dataset.feature_names)}
            switch = index["switch_indicator"]
            if (
                float(scores[raw_minimum]) < -TOLERANCE
                and float(dataset.features[raw_minimum, switch]) <= TOLERANCE
                and float(dataset.features[reference, switch]) > TOLERANCE
            ):
                self.no_stay_vetoes += 1
        states = tuple(str(value) for value in dataset.metadata["candidate_states"])
        tls_id = str(dataset.metadata["row_tls"][0])
        time_sec = float(dataset.metadata["row_times"][0])
        self.decisions += 1
        if selected != reference:
            self.overrides += 1
        return V157CActionDecision(
            reference_index=reference,
            proposed_index=raw_minimum,
            selected_index=selected,
            learned_differs=selected != reference,
            eligible=selected != reference,
            priority=max(float(-scores[selected]), 0.0),
            rejection=("fixed_zero_threshold_or_no_stay_veto" if selected == reference else None),
            arm=self.arm,
            tls_id=tls_id,
            time_sec=time_sec,
            proposed_state=states[raw_minimum],
            selected_state=states[selected],
            reference_state=states[reference],
            predicted_score=float(scores[selected]),
            reference_score=float(scores[reference]),
            forced_reference=selected == reference,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "overrides": self.overrides,
            "no_stay_vetoes": self.no_stay_vetoes,
        }


def _compact_full_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "mean_queue",
        "mean_queue_per_lane",
        "p90_queue_per_lane",
        "mean_tripinfo_waiting_time",
        "mean_tripinfo_depart_delay",
        "system_vehicle_hours",
        "active_vehicle_hours",
        "pending_vehicle_hours",
        "departed",
        "arrived",
        "active_vehicles_at_horizon",
        "pending_vehicles_at_horizon",
        "starting_teleports",
        "ending_teleports",
        "collision_events",
        "collision_incidents",
        "emergency_stops",
        "fixed_horizon_sample_seconds",
        "controlled_lane_count",
        "accepted_intervention_trace",
        "residual_deployment",
    )
    return {key: metrics.get(key) for key in keys}


def run_reserve_seed(
    *,
    protocol_path: Path,
    fit_result_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    calibrator_bundle_path: Path,
    calibrator_bundle_sha256: str,
    seed: int,
    scratch_root: Path,
    output: Path,
) -> dict[str, Any]:
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3

    started = time.monotonic()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    fit = _read_json(fit_result_path)
    if (
        fit.get("protocol") != FIT_RESULT_PROTOCOL
        or fit.get("reserve_authorized") is not True
        or fit.get("status") != "PASS"
    ):
        raise ValueError("V158 OOF gate did not authorize reserve execution")
    if int(seed) not in tuple(int(value) for value in protocol["reserve"]["seeds"]):
        raise ValueError("V158 reserve seed is outside the frozen roster")
    if output.exists() or scratch_root.exists():
        raise FileExistsError("refusing to overwrite V158 reserve output or scratch")
    if _sha256(manifest_path) != protocol["network_manifest"]["sha256"]:
        raise ValueError("V158 reserve manifest identity changed")
    if (
        _sha256(runtime_bundle_path) != str(runtime_bundle_sha256)
        or str(runtime_bundle_sha256) != protocol["v157b_runtime"]["bundle_sha256"]
    ):
        raise ValueError("V158 reserve base model identity changed")
    if _sha256(calibrator_bundle_path) != str(calibrator_bundle_sha256):
        raise ValueError("V158 calibrator bundle identity changed")
    if str(calibrator_bundle_sha256) != str(
        fit.get("artifacts", {}).get("calibrators", {}).get("sha256", "")
    ):
        raise ValueError("V158 calibrator does not match the authorized fit result")
    payload = pickle.loads(calibrator_bundle_path.read_bytes())
    if (
        payload.get("protocol") != CALIBRATOR_BUNDLE_PROTOCOL
        or tuple(payload.get("arms", ())) != CALIBRATED_ARMS
        or payload.get("base_runtime_bundle_sha256") != str(runtime_bundle_sha256)
    ):
        raise ValueError("V158 calibrator bundle contract changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    scenario = str(protocol["reserve"]["scenario"])
    sumocfg = _scenario_paths(manifest_path)[scenario]
    base_models = _load_arm_models(runtime_bundle_path, runtime_bundle_sha256)
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("V158 requires SUMO 1.22.0")
    scratch_root.mkdir(parents=True)
    output.mkdir(parents=True)
    arms = {}
    for arm in ALL_RESERVE_ARMS:
        tripinfo = scratch_root / f"{arm}.xml"
        if arm == "phase_pressure":
            try:
                metrics = v3.evaluate_policy_v3(
                    sumo_api=api,
                    sumocfg=sumocfg,
                    scenario=scenario,
                    policy="phase_pressure",
                    models=None,
                    duration_sec=int(protocol["reserve"]["duration_sec"]),
                    control_interval_sec=CONTROL_INTERVAL_SEC,
                    warmup_sec=0,
                    seed=int(seed),
                    tripinfo_output=tripinfo,
                )
            finally:
                tripinfo.unlink(missing_ok=True)
            originator_diagnostics = None
        else:
            originator = CalibratedRuntimeOriginator(
                base_models=base_models,
                calibrator=payload["arms"][arm],
                arm=arm,
            )
            metrics = _evaluate_originator(
                api=api,
                v3=v3,
                sumocfg=sumocfg,
                scenario=scenario,
                seed=int(seed),
                duration_sec=int(protocol["reserve"]["duration_sec"]),
                originator=originator,
                observer=None,
                tripinfo_path=tripinfo,
                residual_coordination_mode=str(
                    protocol["reserve"]["residual_coordination_mode"]
                ),
                cooldown_intervals=int(
                    protocol["reserve"]["residual_cooldown_intervals"]
                ),
            )
            originator_diagnostics = originator.diagnostics()
        compact = _compact_full_metrics(metrics)
        checks = {
            "evaluator_complete": metrics.get("ok") is True,
            "full_horizon": int(metrics.get("fixed_horizon_sample_seconds", -1))
            == int(protocol["reserve"]["duration_sec"]),
            "no_teleports": int(metrics.get("starting_teleports", -1)) == 0
            and int(metrics.get("ending_teleports", -1)) == 0,
            "finite_primary_metric": bool(
                np.isfinite(float(metrics.get("mean_queue_per_lane", np.nan)))
            ),
        }
        arms[arm] = {
            "status": "PASS" if all(checks.values()) else "INVALID",
            "checks": checks,
            "metrics": compact,
            "originator": originator_diagnostics,
        }
    status = "PASS" if all(row["status"] == "PASS" for row in arms.values()) else "INVALID"
    result = {
        "protocol": RESERVE_SEED_PROTOCOL,
        "status": status,
        "parent_protocol": PROTOCOL,
        "execution_snapshot": _execution_snapshot_identity(),
        "seed": int(seed),
        "scenario": scenario,
        "duration_sec": int(protocol["reserve"]["duration_sec"]),
        "arms": arms,
        "calibrator_bundle_sha256": str(calibrator_bundle_sha256),
        "collision_handling": "diagnostic_not_primary_gate",
        "elapsed_sec": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "status": status,
                "seed": int(seed),
                "mean_queue_per_lane": {
                    arm: row["metrics"]["mean_queue_per_lane"]
                    for arm, row in arms.items()
                },
                "elapsed_sec": result["elapsed_sec"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def aggregate_reserve(
    *, protocol_path: Path, reserve_root: Path, output: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    if output.exists():
        raise FileExistsError("refusing to overwrite V158 reserve aggregate")
    seeds = tuple(int(value) for value in protocol["reserve"]["seeds"])
    rows = []
    costs = {arm: {} for arm in ALL_RESERVE_ARMS}
    for seed in seeds:
        path = Path(reserve_root) / f"seed_{seed}" / "result.json"
        row = _read_json(path)
        if (
            row.get("protocol") != RESERVE_SEED_PROTOCOL
            or row.get("status") != "PASS"
            or int(row.get("seed", -1)) != seed
            or set(row.get("arms", ())) != set(ALL_RESERVE_ARMS)
            or row.get("execution_snapshot") != _execution_snapshot_identity()
        ):
            raise ValueError(f"seed {seed}: invalid V158 reserve result")
        rows.append(row)
        for arm in ALL_RESERVE_ARMS:
            costs[arm][seed] = float(
                row["arms"][arm]["metrics"][protocol["reserve"]["primary_metric"]]
            )
    gate_spec = protocol["oof_gate"]
    comparisons = {
        "source_minus_target": _paired_summary(
            costs["source_native"], costs["target_native"],
            draws=int(gate_spec["bootstrap_draws"]), seed=int(gate_spec["bootstrap_seed"]),
        ),
        "source_minus_placebo": _paired_summary(
            costs["source_native"], costs["placebo_native"],
            draws=int(gate_spec["bootstrap_draws"]), seed=int(gate_spec["bootstrap_seed"]),
        ),
        "source_minus_phase_pressure": _paired_summary(
            costs["source_native"], costs["phase_pressure"],
            draws=int(gate_spec["bootstrap_draws"]), seed=int(gate_spec["bootstrap_seed"]),
        ),
    }
    gate = _gate(protocol, comparisons)
    result = {
        "protocol": RESERVE_AGGREGATE_PROTOCOL,
        "status": "PASS" if gate["passed"] else "SCIENTIFIC_FAIL",
        "scientific_status": (
            "jinan_native_prefix_source_calibration_confirmed"
            if gate["passed"]
            else "final_jinan_native_prefix_remediation_failed"
        ),
        "parent_protocol": PROTOCOL,
        "execution_snapshot": _execution_snapshot_identity(),
        "seeds": list(seeds),
        "primary_metric": protocol["reserve"]["primary_metric"],
        "mean_costs": {
            arm: float(np.mean(list(values.values()))) for arm, values in costs.items()
        },
        "per_seed_costs": {
            arm: {str(seed): value for seed, value in values.items()}
            for arm, values in costs.items()
        },
        "comparisons": comparisons,
        "gate": gate,
        "collision_events": {
            arm: int(
                sum(row["arms"][arm]["metrics"]["collision_events"] for row in rows)
            )
            for arm in ALL_RESERVE_ARMS
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(output, result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect-seed")
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--conversion-root", type=Path, required=True)
    collect.add_argument("--runtime-bundle", type=Path, required=True)
    collect.add_argument("--runtime-bundle-sha256", required=True)
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--scratch-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--reuse-complete", action="store_true")
    fit = subparsers.add_parser("fit")
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--collection-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--reuse-complete", action="store_true")
    reserve = subparsers.add_parser("run-reserve-seed")
    reserve.add_argument("--protocol", type=Path, required=True)
    reserve.add_argument("--fit-result", type=Path, required=True)
    reserve.add_argument("--manifest", type=Path, required=True)
    reserve.add_argument("--conversion-root", type=Path, required=True)
    reserve.add_argument("--runtime-bundle", type=Path, required=True)
    reserve.add_argument("--runtime-bundle-sha256", required=True)
    reserve.add_argument("--calibrator-bundle", type=Path, required=True)
    reserve.add_argument("--calibrator-bundle-sha256", required=True)
    reserve.add_argument("--seed", type=int, required=True)
    reserve.add_argument("--scratch-root", type=Path, required=True)
    reserve.add_argument("--output", type=Path, required=True)
    reserve.add_argument("--reuse-complete", action="store_true")
    aggregate = subparsers.add_parser("aggregate-reserve")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--reserve-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--reuse-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "collect-seed":
        existing = _prepare_directory_run(
            output=args.output,
            scratch=args.scratch_root,
            reuse_complete=args.reuse_complete,
            protocol=SEED_RESULT_PROTOCOL,
            statuses={"PASS"},
            seed=args.seed,
        )
        if existing is not None:
            print(json.dumps({"status": "REUSED", "seed": args.seed}), flush=True)
            return 0
        result = collect_development_seed(
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            conversion_root=args.conversion_root,
            runtime_bundle_path=args.runtime_bundle,
            runtime_bundle_sha256=args.runtime_bundle_sha256,
            seed=args.seed,
            scratch_root=args.scratch_root,
            output=args.output,
        )
        return 0 if result["status"] == "PASS" else 2
    if args.command == "fit":
        existing = _prepare_directory_run(
            output=args.output,
            scratch=None,
            reuse_complete=args.reuse_complete,
            protocol=FIT_RESULT_PROTOCOL,
            statuses={"PASS", "OOF_GATE_FAIL"},
        )
        if existing is not None:
            print(
                json.dumps(
                    {
                        "status": "REUSED",
                        "fit_status": existing["status"],
                        "reserve_authorized": existing["reserve_authorized"],
                    }
                ),
                flush=True,
            )
            return 0
        result = fit_development(
            protocol_path=args.protocol,
            collection_root=args.collection_root,
            output=args.output,
        )
        return 0 if result["status"] in {"PASS", "OOF_GATE_FAIL"} else 2
    if args.command == "run-reserve-seed":
        existing = _prepare_directory_run(
            output=args.output,
            scratch=args.scratch_root,
            reuse_complete=args.reuse_complete,
            protocol=RESERVE_SEED_PROTOCOL,
            statuses={"PASS"},
            seed=args.seed,
        )
        if existing is not None:
            print(json.dumps({"status": "REUSED", "seed": args.seed}), flush=True)
            return 0
        result = run_reserve_seed(
            protocol_path=args.protocol,
            fit_result_path=args.fit_result,
            manifest_path=args.manifest,
            conversion_root=args.conversion_root,
            runtime_bundle_path=args.runtime_bundle,
            runtime_bundle_sha256=args.runtime_bundle_sha256,
            calibrator_bundle_path=args.calibrator_bundle,
            calibrator_bundle_sha256=args.calibrator_bundle_sha256,
            seed=args.seed,
            scratch_root=args.scratch_root,
            output=args.output,
        )
        return 0 if result["status"] == "PASS" else 2
    if args.reuse_complete:
        existing = _existing_result(
            args.output,
            protocol=RESERVE_AGGREGATE_PROTOCOL,
            statuses={"PASS", "SCIENTIFIC_FAIL"},
        )
        if existing is not None:
            print(
                json.dumps(
                    {"status": "REUSED", "aggregate_status": existing["status"]}
                ),
                flush=True,
            )
            return 0
        args.output.unlink(missing_ok=True)
    result = aggregate_reserve(
        protocol_path=args.protocol,
        reserve_root=args.reserve_root,
        output=args.output,
    )
    return 0 if result["status"] in {"PASS", "SCIENTIFIC_FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
