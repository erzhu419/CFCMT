#!/usr/bin/env python3
"""Summarize retained V150L Cologne results without running SUMO."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROTOCOL = "cologne-rigid-service-failure-retained-summary-v1"
TRACE_LIMIT = 256
ARMS = (
    "phase_pressure",
    "rigid_target_only",
    "selected_source_corrected_rigid",
    "same_capacity_placebo_corrected_rigid",
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/"
    "state_conditioned_source_closed_loop_smoke_v3_paired_safety/"
    "cologne/cologne1/seed_41242"
)


def _sample(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "time_sec", "tls_id", "selected_phase_state", "prior_phase_state",
        "current_green_state_before", "executor_mode_before",
        "green_elapsed_sec_before", "override_kind", "request_accepted",
        "execution_effective", "total_vehicles", "mean_speed", "mean_occupancy",
    )
    result = {key: row[key] for key in fields}
    # The trace's total_queue is the state estimator's composite queue proxy.
    result["queue_proxy"] = row["total_queue"]
    result["originator"] = row.get("target_originator")
    return result


def _run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    return {
        "tls_id": first["tls_id"],
        "selected_phase_state": first["selected_phase_state"],
        "start_time_sec": first["time_sec"],
        "end_time_sec": last["time_sec"],
        "sampled_span_sec": last["time_sec"] - first["time_sec"],
        "sample_count": len(rows),
        "first_total_vehicles": first["total_vehicles"],
        "last_total_vehicles": last["total_vehicles"],
        "minimum_mean_speed": min(row["mean_speed"] for row in rows),
        "maximum_mean_speed": max(row["mean_speed"] for row in rows),
        "first_mean_speed": first["mean_speed"],
        "last_mean_speed": last["mean_speed"],
        "last_queue_proxy": last["total_queue"],
        "last_green_elapsed_sec_before": last["green_elapsed_sec_before"],
        "last_current_green_state_before": last["current_green_state_before"],
        "same_phase_request_count": sum(
            row["override_kind"] == "stay" for row in rows
        ),
    }


def sampled_runs(
    trace: list[dict[str, Any]], interval_sec: float, *, stationary: bool = False
) -> list[dict[str, Any]]:
    """Group consecutive recorded samples, never bridge an unrecorded interval."""
    runs: list[dict[str, Any]] = []
    tls_ids = sorted({row["tls_id"] for row in trace})
    for tls_id in tls_ids:
        current: list[dict[str, Any]] = []
        for row in sorted(
            (row for row in trace if row["tls_id"] == tls_id),
            key=lambda row: row["time_sec"],
        ):
            eligible = not stationary or (
                row["mean_speed"] == 0.0 and row["total_vehicles"] > 0
            )
            consecutive = not current or (
                row["time_sec"] - current[-1]["time_sec"] == interval_sec
                and row["selected_phase_state"] == current[-1]["selected_phase_state"]
            )
            if current and (not eligible or not consecutive):
                runs.append(_run(current))
                current = []
            if eligible:
                current.append(row)
        if current:
            runs.append(_run(current))
    return sorted(runs, key=lambda row: (row["start_time_sec"], row["tls_id"]))


def summarize_arm(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    trace = metrics["accepted_intervention_trace"]
    phase = metrics["phase_execution_audit"]
    originator = payload.get("originator_diagnostics") or {}
    guard = metrics["guard_audit"]
    interval = float(payload["control_interval_sec"])
    endpoint_keys = (
        "demand_population_at_horizon", "departed", "arrived",
        "pending_vehicles_at_horizon", "active_vehicles_at_horizon",
        "departure_service_ratio", "completion_ratio", "throughput_ratio",
        "mean_tripinfo_waiting_time", "mean_completed_tripinfo_waiting_time",
        "mean_tripinfo_depart_delay", "tripinfo_count", "tripinfo_unfinished_count",
        "pending_vehicle_hours", "active_vehicle_hours", "system_vehicle_hours",
        "mean_queue", "mean_queue_proxy", "collision_incidents", "collision_events",
    )
    endpoints = {key: metrics[key] for key in endpoint_keys}
    demand = metrics["demand_population_at_horizon"]
    endpoints["pending_fraction_of_demand"] = metrics["pending_vehicles_at_horizon"] / demand
    endpoints["uncompleted_fraction_of_demand"] = (
        metrics["pending_vehicles_at_horizon"] + metrics["active_vehicles_at_horizon"]
    ) / demand
    endpoints["arrived_plus_active_plus_pending"] = sum(
        metrics[key] for key in (
            "arrived", "active_vehicles_at_horizon", "pending_vehicles_at_horizon"
        )
    )
    sampled = sampled_runs(trace, interval)
    stationary = sampled_runs(trace, interval, stationary=True)
    last_time = trace[-1]["time_sec"] if trace else None
    stationary_suffix = next(
        (run for run in reversed(stationary) if run["end_time_sec"] == last_time), None
    )
    total_accepted = int(guard.get("accepted_overrides", 0))
    phase_seconds = sum(phase[key] for key in (
        "green_seconds", "yellow_seconds", "all_red_seconds"
    ))
    source_rows = [row for row in trace if (
        row.get("target_originator") or {}
    ).get("source_changed_rigid_action", False)]
    return {
        "arm": payload["arm"],
        "endpoints": endpoints,
        "phase_execution_audit": phase,
        "green_fraction_of_executor_time": phase["green_seconds"] / phase_seconds,
        "same_phase_fraction_of_requests": phase["same_phase_requests"] / phase["requests"],
        "originator_diagnostics": originator,
        "full_run_guard_counts": {key: guard.get(key, 0) for key in (
            "decisions", "accepted_overrides", "executed_switch_overrides",
            "executed_stay_overrides", "rejected_executor", "prior_agreements",
        )},
        "controlled_tls_count": metrics["residual_deployment"].get(
            "intervention_graph", {}
        ).get("tls_count"),
        "trace_coverage": {
            "retained_rows": len(trace),
            "capacity": TRACE_LIMIT,
            "capacity_reached": len(trace) == TRACE_LIMIT,
            "unretained_accepted_overrides": total_accepted - len(trace),
            "first_time_sec": trace[0]["time_sec"] if trace else None,
            "last_time_sec": last_time,
            "source_changes_in_full_run": originator.get("source_changed_rigid_action_count"),
            "source_changes_in_retained_override_rows": len(source_rows),
            "population": "first_256_pressure_override_proposals_at_execution",
            "complete_decision_trace": False,
        },
        "retained_kind_counts": dict(Counter(row["override_kind"] for row in trace)),
        "retained_selected_phase_counts": dict(Counter(
            row["selected_phase_state"] for row in trace
        )),
        "sampled_selected_phase_runs": sampled,
        "zero_speed_sample_runs": stationary,
        "stationary_suffix_of_retained_trace": stationary_suffix,
        "last_retained_sample": _sample(trace[-1]) if trace else None,
        "retained_source_changes": [_sample(row) for row in source_rows],
    }


def summarize_results(input_root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    identities = set()
    for arm in ARMS:
        path = input_root / arm / "result.json"
        payload = json.loads(path.read_text())
        if payload["arm"] != arm:
            raise ValueError(f"Result arm does not match its directory: {path}")
        identities.add((payload["city"], payload["scenario"], payload["seed"]))
        arms[arm] = summarize_arm(payload)
        arms[arm]["source_result"] = str(path.resolve())
    if len(identities) != 1:
        raise ValueError("Arm results do not share one city/scenario/seed")
    city, scenario, seed = next(iter(identities))
    return {
        "protocol": PROTOCOL,
        "city": city, "scenario": scenario, "seed": seed,
        "arms": arms,
        "metric_semantics": {
            "trace_total_queue": "composite q_by_lane proxy, not a halted vehicle count",
            "trace_mean_speed": "unweighted mean of controlled incoming lane mean speeds",
            "request_accepted_false_for_stay": "same phase returns false without an executor rejection",
            "throughput_ratio": "arrived/departed; pending vehicles excluded",
            "completion_ratio": "arrived/(departed+pending); full due demand denominator",
            "tripinfo_population": "inserted vehicles including unfinished; pending vehicles excluded",
        },
        "limitations": [
            "Pressure agreements are not traced and pressure-only has no intervention trace.",
            "The first 256 override rows truncate the learned policies before the simulation horizon.",
            "Zero mean speed is a sequence of lane-state samples, not measured stop-line discharge.",
            "Retained aggregates cannot distinguish red approach starvation, green lane blockage, or upstream lane blocking.",
            "PhasePressure has collisions; its service totals do not authorize it as a safe deployment policy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_results(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "arms": {arm: {
            "arrived": row["endpoints"]["arrived"],
            "pending": row["endpoints"]["pending_vehicles_at_horizon"],
            "stationary_suffix": row["stationary_suffix_of_retained_trace"],
        } for arm, row in summary["arms"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
