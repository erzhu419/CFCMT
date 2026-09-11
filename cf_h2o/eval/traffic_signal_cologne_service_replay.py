"""Read-only lane and action observations during the original Cologne rigid run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any, Mapping

import numpy as np

from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import _rigid_score
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    RUNTIME_POLICY,
    StateConditionedSourceUtilityOriginator,
    _runtime_models,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import evaluate_policy_v3
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.right_of_way_context import (
    net_file_from_sumocfg, read_network_right_of_way_context,
)


PROTOCOL = "tsc-v154-cologne-rigid-read-only-service-replay-v2"
TLS_ID = "cluster_357187_359543"
BEGIN_SEC = 25200.0
END_SEC = 26820.0
OBSERVE_BEGIN_SEC = 25380.0
CANDIDATE_FEATURES = (
    "green_q", "red_q", "service_pressure", "green_down_occ",
    "switch_indicator", "clearance_fraction",
)
CANDIDATE_COLUMNS = ("index", "phase_state", "rigid_score", *CANDIDATE_FEATURES)


def observe_lane(
    api: Any, lane_id: str, previous_ids: set[str] | None, *, incoming: bool
) -> tuple[dict[str, Any], set[str]]:
    """Use getters only; sampled lane exits include lane changes and departures."""
    ids = set(api.lane.getLastStepVehicleIDs(lane_id))
    row: dict[str, Any] = {
        "vehicles": len(ids),
        "halting": int(api.lane.getLastStepHaltingNumber(lane_id)),
        "occupancy_fraction": float(api.lane.getLastStepOccupancy(lane_id)),
        "mean_speed_mps": float(api.lane.getLastStepMeanSpeed(lane_id)),
        "sampled_lane_leaves": len(previous_ids - ids) if previous_ids is not None else None,
        "sampled_lane_enters": len(ids - previous_ids) if previous_ids is not None else None,
    }
    if not incoming:
        return row, ids
    next_tls: dict[str, list[Any]] = {}
    for vehicle in sorted(ids):
        next_tls[vehicle] = next((
            [int(link), float(distance), str(state)]
            for tls, link, distance, state in api.vehicle.getNextTLS(vehicle)
            if str(tls) == TLS_ID
        ), [])
    row["next_tls_link_counts"] = dict(Counter(
        str(value[0]) for value in next_tls.values() if value
    ))
    if ids:
        head = max(sorted(ids), key=lambda vehicle: api.vehicle.getLanePosition(vehicle))
        route = tuple(api.vehicle.getRoute(head))
        route_index = int(api.vehicle.getRouteIndex(head))
        row["head"] = {
            "id": head,
            "position_m": float(api.vehicle.getLanePosition(head)),
            "speed_mps": float(api.vehicle.getSpeed(head)),
            "waiting_sec": float(api.vehicle.getWaitingTime(head)),
            "next_tls": next_tls[head],
            "next_route_edge": route[route_index + 1] if route_index + 1 < len(route) else None,
        }
    else:
        row["head"] = None
    return row, ids


def add_selected_demand(lanes: dict[str, Any], selected_state: str) -> None:
    for lane in lanes.values():
        if "next_tls_link_counts" not in lane:
            continue
        lane["selected_green_demand"] = sum(
            count for link, count in lane["next_tls_link_counts"].items()
            if selected_state[int(link)] in "Gg"
        )
        head = lane["head"]
        lane["head_selected_signal"] = (
            selected_state[int(head["next_tls"][0])]
            if head is not None and head["next_tls"] else None
        )


class ObservedRigidOriginator(StateConditionedSourceUtilityOriginator):
    """Delegate every action to the original originator and retain extra getters."""

    def attach_observer(self, api: Any, snapshot_path: Path | None = None) -> None:
        self.observation_api = api
        self.snapshot_path = snapshot_path
        self.saved_snapshot: dict[str, Any] | None = None
        self.service_samples: list[dict[str, Any]] = []
        self.lane_catalog: dict[str, Any] = {}
        self.movement_catalog: list[dict[str, Any]] = []
        self.previous_lane_ids: dict[str, set[str]] = {}
        self.current_service_sample: dict[str, Any] | None = None

    def prepare_interval(self, *, states, executors, routing_graph, control_interval_sec):
        super().prepare_interval(
            states=states, executors=executors, routing_graph=routing_graph,
            control_interval_sec=control_interval_sec,
        )
        if set(states) != {TLS_ID} or control_interval_sec != 10:
            raise ValueError("Service replay requires the original single Cologne TLS")
        api = self.observation_api
        now = float(api.simulation.getTime())
        self.current_service_sample = None
        if not OBSERVE_BEGIN_SEC <= now < END_SEC:
            return
        info = states[TLS_ID].info
        incoming = set(info.incoming_lanes)
        if now == 26700.0 and self.snapshot_path is not None:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            api.simulation.saveState(str(self.snapshot_path))
            context_path = self.snapshot_path.with_name(self.snapshot_path.name + ".executors.json")
            context_path.write_text(json.dumps({
                "time_sec": now, "before_action_request": True,
                "save_state_rng": True,
                "executors": {name: executor.snapshot() for name, executor in executors.items()},
                "runtime_policy": RUNTIME_POLICY, "residual_coordination_mode": "direct",
                "residual_cooldown_intervals": 0,
            }, separators=(",", ":")) + "\n")
            self.saved_snapshot = {
                "time_sec": now, "sumo_state_path": str(self.snapshot_path),
                "executor_context_path": str(context_path),
                "remote_scratch_only": True, "before_action_request": True,
            }
        if not self.lane_catalog:
            outgoing = set()
            for index, connections in enumerate(info.controlled_links):
                for from_lane, to_lane, via in connections:
                    outgoing.add(str(to_lane))
                    self.movement_catalog.append({
                        "link": index, "from_lane": str(from_lane),
                        "to_lane": str(to_lane), "via": str(via),
                    })
            self.lane_catalog = {
                lane: {"incoming": lane in incoming, "length_m": float(api.lane.getLength(lane))}
                for lane in sorted(incoming | outgoing)
            }
        lanes = {}
        for lane in self.lane_catalog:
            lanes[lane], self.previous_lane_ids[lane] = observe_lane(
                api, lane, self.previous_lane_ids.get(lane), incoming=lane in incoming
            )
            if lane in incoming:
                lanes[lane]["queue_proxy"] = float(states[TLS_ID].q_by_lane[lane])
        executor = executors[TLS_ID].snapshot()
        executor.pop("audit")
        self.current_service_sample = {
            "time_sec": now,
            "active_inserted_vehicles": int(api.vehicle.getIDCount()),
            "pending_insertion_vehicles": len(api.simulation.getPendingVehicles()),
            "executor_before": executor,
            "lanes": lanes,
            "selected_phase_state": None,
            "pressure_phase_state": None,
            "candidates": None,
        }
        self.service_samples.append(self.current_service_sample)

    def select(self, contrast, *, reference_index):
        # Returning this same decision preserves the parent's tie-break and action.
        decision = super().select(contrast, reference_index=reference_index)
        sample = self.current_service_sample
        if sample is None:
            return decision
        augmented = self._augment_contrast(contrast, reference_index=reference_index)
        scores = _rigid_score(self.payload["rigid_model"], augmented)
        states = list(contrast.metadata["candidate_states"])
        indices = [augmented.feature_names.index(name) for name in CANDIDATE_FEATURES]
        sample["candidates"] = [
            [i, str(state), float(scores[i]), *[
                float(augmented.features[i, feature]) for feature in indices
            ]]
            for i, state in enumerate(states)
        ]
        sample["selected_phase_state"] = str(states[decision.selected_index])
        sample["pressure_phase_state"] = str(states[decision.reference_index])
        sample["selected_index"] = int(decision.selected_index)
        sample["pressure_index"] = int(decision.reference_index)
        if float(scores[decision.selected_index]) != float(decision.predicted_score):
            raise RuntimeError("Observed score differs from original rigid selection")
        add_selected_demand(sample["lanes"], sample["selected_phase_state"])
        return decision


def compare_replay_prefix(reference: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Require every old trace field to match; new diagnostic fields may be added."""
    expected = [row for row in reference["metrics"]["accepted_intervention_trace"]
                if float(row["time_sec"]) < END_SEC]
    actual = list(metrics.get("accepted_intervention_trace", ()))
    mismatches: list[str] = []

    def compare(old, new, path):
        if isinstance(old, Mapping):
            if not isinstance(new, Mapping):
                mismatches.append(path)
            else:
                for key, value in old.items():
                    if key not in new:
                        mismatches.append(f"{path}.{key}")
                    else:
                        compare(value, new[key], f"{path}.{key}")
        elif old != new:
            mismatches.append(path)

    if len(expected) != len(actual):
        mismatches.append("trace_length")
    for index, (old, new) in enumerate(zip(expected, actual)):
        compare(old, new, f"trace[{index}]")
    if not expected:
        mismatches.append("empty_reference")
    return {
        "passed": not mismatches,
        "comparison": "exact_equality_of_all_retained_original_fields_before_26820",
        "expected_rows": len(expected), "actual_rows": len(actual),
        "first_time_sec": expected[0]["time_sec"] if expected else None,
        "last_time_sec": expected[-1]["time_sec"] if expected else None,
        "mismatch_count": len(mismatches), "first_mismatches": mismatches[:10],
    }


def launch_inputs(launch: Mapping[str, Any]) -> dict[str, str]:
    signature = "CFCMT/v150l/state-conditioned-source-closed-loop-v3/smoke/cologne/cologne1/41242/rigid_target_only"
    task = next(row for row in launch["task_specs"] if row["signature"] == signature)
    tokens = shlex.split(task["cmd"])
    options = {key: tokens[tokens.index(key) + 1] for key in (
        "--manifest", "--conversion-root", "--scenario", "--city", "--seed",
        "--arm", "--duration-sec", "--warmup-sec", "--control-interval-sec",
        "--prediction-horizon-sec", "--runtime-model", "--runtime-model-sha256",
    )}
    expected = {"--scenario": "cologne1", "--city": "cologne", "--seed": "41242",
                "--arm": "rigid_target_only", "--duration-sec": "3600",
                "--warmup-sec": "60", "--control-interval-sec": "10",
                "--prediction-horizon-sec": "450"}
    if any(options[key] != value for key, value in expected.items()):
        raise ValueError("Original Cologne runtime arguments changed")
    options["reference_result"] = str(Path(task["result_dir"]) / "result.json")
    return options


def run_replay(launch_path: Path, tripinfo: Path, snapshot_path: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    options = launch_inputs(json.loads(launch_path.read_text()))
    reference = json.loads(Path(options["reference_result"]).read_text())
    if (reference["city"], reference["scenario"], reference["seed"], reference["arm"]) != (
        "cologne", "cologne1", 41242, "rigid_target_only"
    ):
        raise ValueError("Replay reference belongs to another rollout")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = options["--conversion-root"]
    manifest = load_traffic_signal_manifest(Path(options["--manifest"]))
    sumocfg = Path(manifest.sumocfgs["cologne1"]).resolve()
    context = read_network_right_of_way_context(net_file_from_sumocfg(sumocfg))
    originator = ObservedRigidOriginator.load(
        Path(options["--runtime-model"]),
        expected_sha256=options["--runtime-model-sha256"], expected_city="cologne",
        network_context=context, arm="rigid_target_only",
    )
    api = load_libsumo()
    if snapshot_path is not None and "CFCMT_SCRATCH" not in snapshot_path.parts:
        raise ValueError("The optional state snapshot must remain in remote CFCMT_SCRATCH")
    originator.attach_observer(api, snapshot_path)
    if tripinfo.exists():
        raise FileExistsError(tripinfo)
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario="cologne1", policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=450),
            duration_sec=END_SEC - BEGIN_SEC, control_interval_sec=10, warmup_sec=60,
            seed=41242, tripinfo_output=tripinfo, residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0,
        )
    finally:
        tripinfo.unlink(missing_ok=True)
    prefix = compare_replay_prefix(reference, metrics)
    # The simulation stops at 26820; its final pre-action sample is at 26810.
    sample_times = [row["time_sec"] for row in originator.service_samples]
    samples_complete = sample_times == list(np.arange(OBSERVE_BEGIN_SEC, END_SEC, 10))
    endpoint_keys = (
        "ok", "error", "departed", "arrived", "active_vehicles_at_horizon",
        "pending_vehicles_at_horizon", "demand_population_at_horizon",
        "completion_ratio", "departure_service_ratio", "mean_queue",
        "pending_vehicle_hours", "system_vehicle_hours", "collision_events",
        "collision_incidents", "starting_teleports", "ending_teleports",
        "phase_execution_audit", "guard_audit",
    )
    return {
        "protocol": PROTOCOL,
        "status": "PASS" if metrics.get("ok") and prefix["passed"] and samples_complete else "FAIL",
        "city": "cologne", "scenario": "cologne1", "seed": 41242,
        "arm": "rigid_target_only", "tls_id": TLS_ID,
        "elapsed_sec": time.monotonic() - started,
        "sumo_version": libsumo_version(),
        "launch_record": str(launch_path), "reference_result": options["reference_result"],
        "original_runtime_arguments": options,
        "simulation_window_sec": [BEGIN_SEC, END_SEC],
        "observation_window_sec": [OBSERVE_BEGIN_SEC, END_SEC],
        "observation_end_exclusive": True, "sample_count": len(sample_times),
        "sample_times_complete": samples_complete,
        "replay_prefix": prefix,
        "endpoint": {key: metrics[key] for key in endpoint_keys if key in metrics},
        "lane_catalog": originator.lane_catalog,
        "movement_catalog": originator.movement_catalog,
        "saved_snapshot": originator.saved_snapshot,
        "candidate_columns": list(CANDIDATE_COLUMNS),
        "samples": originator.service_samples,
        "observation_semantics": {
            "actions": "unchanged original originator return values and original executor",
            "sampled_lane_leaves": "previous sampled lane IDs minus current; includes lane changes and is not arrival throughput",
            "head": "frontmost current vehicle by lane position",
            "head_next_tls": "[link index for this TLS, distance in metres, current signal character]",
            "next_tls_link_counts": "current lane vehicles grouped by their next movement at this TLS",
            "null_candidates": "original runtime did not ask the originator to select at this interval",
            "pending": "SUMO pending insertion vehicles, separate from inserted active vehicles",
            "scope": "single recorded failing trajectory; no policy repair or efficacy test",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--tripinfo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-at-26700", type=Path)
    args = parser.parse_args()
    result = run_replay(args.launch_record, args.tripinfo, args.state_at_26700)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "sample_count", "replay_prefix", "elapsed_sec",
    )}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
