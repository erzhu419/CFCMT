"""Run strict microscopic admission for an ETH five-city package."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import _collision_incident_key
from cf_h2o.sumo_runtime import (
    libsumo_version,
    load_libsumo,
    runtime_metadata,
    sumo_state_directory,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.data.package_eth_city_sumo import (
    MICRO_CONFIG_PROTOCOL,
    PROTOCOL as PACKAGE_PROTOCOL,
)
from scripts.data.repair_eth_boston_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_REMEDIATED_PACKAGE_PROTOCOL,
    validate_remediated_package_manifest,
)
from scripts.data.repair_eth_boston_uncontrolled_merge import (
    PACKAGE_PROTOCOL as BOSTON_V8_PACKAGE_PROTOCOL,
    validate_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_ramp_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V9_PACKAGE_PROTOCOL,
    validate_ramp_merge_manifest,
)
from scripts.data.repair_eth_boston_permissive_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V10_PACKAGE_PROTOCOL,
    validate_permissive_merge_manifest,
)
from scripts.data.repair_eth_boston_protected_uncontrolled_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V11_PACKAGE_PROTOCOL,
    validate_protected_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_joined_tls_uncontrolled_merge import (
    PACKAGE_PROTOCOL as BOSTON_V12_PACKAGE_PROTOCOL,
    validate_joined_tls_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_systemic_right_of_way import (
    PACKAGE_PROTOCOL as BOSTON_V13_PACKAGE_PROTOCOL,
    validate_systemic_right_of_way_manifest,
)
from scripts.data.repair_eth_boston_systemic_uncontrolled_major import (
    PACKAGE_PROTOCOL as BOSTON_V14_PACKAGE_PROTOCOL,
    validate_systemic_uncontrolled_major_manifest,
)
from scripts.data.repair_eth_boston_tls_yellow_clearance import (
    PACKAGE_PROTOCOL as BOSTON_V15_PACKAGE_PROTOCOL,
    validate_tls_yellow_clearance_manifest,
)
from scripts.data.repair_eth_boston_implicit_no_tls_major import (
    PACKAGE_PROTOCOL as BOSTON_V16_PACKAGE_PROTOCOL,
    validate_implicit_no_tls_major_manifest,
)
from scripts.data.repair_eth_boston_no_tls_major_response import (
    PACKAGE_PROTOCOL as BOSTON_V17_PACKAGE_PROTOCOL,
    validate_no_tls_major_response_manifest,
)
from scripts.data.repair_eth_boston_controlled_shared_receiving_yield import (
    PACKAGE_PROTOCOL as BOSTON_V18_PACKAGE_PROTOCOL,
    validate_controlled_shared_receiving_yield_manifest,
)


PROTOCOL = "eth-five-city-strict-libsumo-microscopic-admission-v1"
DIAGNOSTIC_TRACE_PROTOCOL = "eth-city-read-only-vehicle-trace-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_package(
    package_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_city_code: str,
) -> tuple[Path, dict[str, Any], str]:
    package_root = package_root.resolve()
    manifest_path = package_root / "package_manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "ETH package manifest identity changed: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = _read_json(manifest_path)
    package_protocol = manifest.get("protocol")
    if package_protocol == BOSTON_V18_PACKAGE_PROTOCOL:
        validate_controlled_shared_receiving_yield_manifest(manifest)
    elif package_protocol == BOSTON_V17_PACKAGE_PROTOCOL:
        validate_no_tls_major_response_manifest(manifest)
    elif package_protocol == BOSTON_V16_PACKAGE_PROTOCOL:
        validate_implicit_no_tls_major_manifest(manifest)
    elif package_protocol == BOSTON_V15_PACKAGE_PROTOCOL:
        validate_tls_yellow_clearance_manifest(manifest)
    elif package_protocol == BOSTON_V14_PACKAGE_PROTOCOL:
        validate_systemic_uncontrolled_major_manifest(manifest)
    elif package_protocol == BOSTON_V13_PACKAGE_PROTOCOL:
        validate_systemic_right_of_way_manifest(manifest)
    elif package_protocol == BOSTON_V12_PACKAGE_PROTOCOL:
        validate_joined_tls_uncontrolled_merge_manifest(manifest)
    elif package_protocol == BOSTON_V11_PACKAGE_PROTOCOL:
        validate_protected_uncontrolled_merge_manifest(manifest)
    elif package_protocol == BOSTON_V10_PACKAGE_PROTOCOL:
        validate_permissive_merge_manifest(manifest)
    elif package_protocol == BOSTON_V9_PACKAGE_PROTOCOL:
        validate_ramp_merge_manifest(manifest)
    elif package_protocol == BOSTON_V8_PACKAGE_PROTOCOL:
        validate_uncontrolled_merge_manifest(manifest)
    elif package_protocol == BOSTON_REMEDIATED_PACKAGE_PROTOCOL:
        validate_remediated_package_manifest(manifest)
    elif package_protocol != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected ETH package protocol: {manifest.get('protocol')}")
    city_code = str(manifest.get("city_code", ""))
    if city_code != expected_city_code.upper():
        raise ValueError(
            f"ETH package city changed: {city_code} != {expected_city_code.upper()}"
        )
    if manifest.get("passed") is not True or not all(
        bool(value) for value in dict(manifest.get("gates", {})).values()
    ):
        raise ValueError("ETH package did not pass frozen static admission")
    demand = dict(manifest.get("demand_inventory", {}))
    network = dict(manifest.get("network_inventory", {}))
    if int(demand.get("trip_count", 0)) <= 0:
        raise ValueError("ETH package has no audited trips")
    if int(network.get("traffic_light_count", 0)) <= 0:
        raise ValueError("ETH package has no audited traffic lights")
    strict = dict(manifest.get("microscopic_instantiation", {}))
    expected_strict = {
        "protocol": MICRO_CONFIG_PROTOCOL,
        "mode": "microscopic",
        "backend_required_by_protocol": "libsumo",
        "step_length_sec": 1,
        "time_to_teleport_sec": -1,
        "collision_check_junctions": True,
        "collision_action": "warn",
        "ignore_route_errors": False,
        "max_depart_delay_sec": -1,
    }
    observed_strict = {key: strict.get(key) for key in expected_strict}
    if observed_strict != expected_strict:
        raise ValueError(
            f"ETH strict execution changed: {observed_strict} != {expected_strict}"
        )
    config = package_root / str(strict.get("config", ""))
    if not config.is_file():
        raise FileNotFoundError(f"ETH strict SUMO config is missing: {config}")
    return config, manifest, manifest_sha256


def _final_summary_step(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SUMO summary output is missing: {path}")
    final: dict[str, str] | None = None
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "step":
            final = dict(element.attrib)
        element.clear()
    if final is None:
        raise ValueError(f"SUMO summary output contains no step: {path}")
    integer_fields = (
        "loaded",
        "inserted",
        "running",
        "waiting",
        "ended",
        "arrived",
        "collisions",
        "teleports",
    )
    parsed: dict[str, Any] = {"time": float(final.get("time", 0.0))}
    for field in integer_fields:
        parsed[field] = int(float(final.get(field, 0)))
    return parsed


def _vehicle_collision_snapshot(sumo_api: Any, vehicle_id: str) -> dict[str, Any]:
    present = str(vehicle_id) in set(sumo_api.vehicle.getIDList())
    snapshot: dict[str, Any] = {"vehicle_id": str(vehicle_id), "present": present}
    if not present:
        return snapshot
    route = tuple(str(edge) for edge in sumo_api.vehicle.getRoute(vehicle_id))
    route_index = int(sumo_api.vehicle.getRouteIndex(vehicle_id))
    snapshot.update(
        {
            "road_id": str(sumo_api.vehicle.getRoadID(vehicle_id)),
            "lane_id": str(sumo_api.vehicle.getLaneID(vehicle_id)),
            "lane_position": float(sumo_api.vehicle.getLanePosition(vehicle_id)),
            "speed": float(sumo_api.vehicle.getSpeed(vehicle_id)),
            "route_index": route_index,
            "previous_route_edge": route[route_index - 1] if route_index > 0 else None,
            "current_route_edge": (
                route[route_index] if 0 <= route_index < len(route) else None
            ),
            "next_route_edge": (
                route[route_index + 1]
                if 0 <= route_index + 1 < len(route)
                else None
            ),
            "route": list(route),
        }
    )
    return snapshot


def _json_diagnostic_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_diagnostic_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_diagnostic_value(item) for item in value]
    return repr(value)


def _diagnostic_vehicle_call(
    vehicle_api: Any, method: str, vehicle_id: str, *args: Any
) -> Any:
    function = getattr(vehicle_api, method, None)
    if function is None:
        return {"unavailable": method}
    try:
        return _json_diagnostic_value(function(vehicle_id, *args))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _vehicle_diagnostic_snapshot(
    sumo_api: Any, vehicle_id: str, *, present_ids: set[str]
) -> dict[str, Any]:
    if str(vehicle_id) not in present_ids:
        return {"vehicle_id": str(vehicle_id), "present": False}
    vehicle = sumo_api.vehicle
    scalar_methods = (
        "getAcceleration",
        "getActionStepLength",
        "getAllowedSpeed",
        "getApparentDecel",
        "getDecel",
        "getEmergencyDecel",
        "getLaneChangeMode",
        "getLaneID",
        "getLanePosition",
        "getLateralLanePosition",
        "getLength",
        "getMinGap",
        "getRoadID",
        "getRouteIndex",
        "getSignals",
        "getSpeed",
        "getSpeedMode",
        "getSpeedWithoutTraCI",
        "getTau",
    )
    values = {
        method.removeprefix("get").replace("TraCI", "Traci"): (
            _diagnostic_vehicle_call(vehicle, method, str(vehicle_id))
        )
        for method in scalar_methods
    }
    values.update(
        {
            "LeaderWithin100m": _diagnostic_vehicle_call(
                vehicle, "getLeader", str(vehicle_id), 100.0
            ),
            "FollowerWithin100m": _diagnostic_vehicle_call(
                vehicle, "getFollower", str(vehicle_id), 100.0
            ),
            "LaneChangeStateLeft": _diagnostic_vehicle_call(
                vehicle, "getLaneChangeState", str(vehicle_id), 1
            ),
            "LaneChangeStateRight": _diagnostic_vehicle_call(
                vehicle, "getLaneChangeState", str(vehicle_id), -1
            ),
            "NextTLS": _diagnostic_vehicle_call(
                vehicle, "getNextTLS", str(vehicle_id)
            ),
        }
    )
    return {"vehicle_id": str(vehicle_id), "present": True, **values}


def _diagnostic_trace_sample(
    sumo_api: Any,
    *,
    time_sec: float,
    vehicle_ids: Sequence[str],
    lane_ids: Sequence[str],
) -> dict[str, Any]:
    present_ids = {str(value) for value in sumo_api.vehicle.getIDList()}
    vehicles_by_lane: dict[str, list[str]] = {}
    selected_ids = {str(value) for value in vehicle_ids}
    for lane_id in lane_ids:
        values = [
            str(value) for value in sumo_api.lane.getLastStepVehicleIDs(str(lane_id))
        ]
        vehicles_by_lane[str(lane_id)] = values
        selected_ids.update(values)
    return {
        "time_sec": float(time_sec),
        "vehicle_ids_by_lane": vehicles_by_lane,
        "vehicles": [
            _vehicle_diagnostic_snapshot(
                sumo_api, vehicle_id, present_ids=present_ids
            )
            for vehicle_id in sorted(selected_ids)
        ],
    }


def _collision_lane_topology(network_path: Path, lane_id: str) -> dict[str, Any]:
    edge_id, lane_index = str(lane_id).rsplit("_", 1)
    root = ET.parse(network_path).getroot()
    edges = [edge for edge in root.iter("edge") if edge.get("id") == edge_id]
    if len(edges) != 1:
        raise ValueError(f"collision lane edge matches changed: {edge_id}={len(edges)}")
    junction_id = str(edges[0].get("from", ""))
    incoming = [
        dict(connection.attrib)
        for connection in root.iter("connection")
        if connection.get("to") == edge_id
        and connection.get("toLane") == lane_index
    ]
    tls_ids = sorted(
        {
            str(connection["tl"])
            for connection in incoming
            if connection.get("tl")
        }
    )
    return {
        "lane_id": str(lane_id),
        "edge_id": edge_id,
        "lane_index": lane_index,
        "junction_id": junction_id,
        "incoming_connections": incoming,
        "traffic_light_ids": tls_ids,
    }


def _traffic_light_collision_snapshot(
    sumo_api: Any, topology: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "traffic_light_id": tls_id,
            "program_id": str(sumo_api.trafficlight.getProgram(tls_id)),
            "phase_index": int(sumo_api.trafficlight.getPhase(tls_id)),
            "state": str(sumo_api.trafficlight.getRedYellowGreenState(tls_id)),
            "next_switch_sec": float(sumo_api.trafficlight.getNextSwitch(tls_id)),
        }
        for tls_id in topology["traffic_light_ids"]
    ]


def _collision_sample(
    collision: Any,
    *,
    time_sec: float,
    sumo_api: Any | None = None,
    network_path: Path | None = None,
    topology_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {"repr": repr(collision), "time_sec": time_sec}
    for field in ("collider", "victim", "type", "lane", "pos"):
        value = getattr(collision, field, None)
        if value is not None:
            sample[field] = value
    if sumo_api is not None:
        participant_ids = tuple(
            dict.fromkeys(
                str(sample[field])
                for field in ("collider", "victim")
                if sample.get(field) not in (None, "")
            )
        )
        sample["participants"] = [
            _vehicle_collision_snapshot(sumo_api, vehicle_id)
            for vehicle_id in participant_ids
        ]
    if network_path is not None and sample.get("lane"):
        try:
            lane_id = str(sample["lane"])
            cache = topology_cache if topology_cache is not None else {}
            topology = cache.get(lane_id)
            if topology is None:
                topology = _collision_lane_topology(network_path, lane_id)
                cache[lane_id] = topology
            sample["lane_topology"] = topology
            if sumo_api is not None:
                sample["traffic_light_state"] = _traffic_light_collision_snapshot(
                    sumo_api, topology
                )
        except Exception as exc:
            sample["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return sample


def _full_admission_checks(
    *,
    expected_vehicle_count: int,
    expected_traffic_lights: int,
    observed: dict[str, Any],
    summary: dict[str, Any] | None,
    error: str | None,
) -> dict[str, bool]:
    final_summary = summary or {}
    return {
        "runtime_exception_free": error is None,
        "all_vehicles_completed_before_cap": (
            observed.get("termination_reason") == "all_vehicles_completed"
        ),
        "exact_vehicle_demand_loaded": (
            int(final_summary.get("loaded", -1)) == expected_vehicle_count
        ),
        "exact_vehicle_demand_departed": (
            int(observed.get("departed_vehicles", -1)) == expected_vehicle_count
        ),
        "exact_vehicle_demand_arrived": (
            int(observed.get("arrived_vehicles", -1)) == expected_vehicle_count
            and int(final_summary.get("arrived", -1)) == expected_vehicle_count
        ),
        "no_running_vehicles": (
            int(observed.get("final_active_vehicles", -1)) == 0
            and int(final_summary.get("running", -1)) == 0
        ),
        "no_pending_vehicles": (
            int(observed.get("final_pending_vehicles", -1)) == 0
            and int(final_summary.get("waiting", -1)) == 0
        ),
        "no_expected_vehicles": int(observed.get("final_min_expected", -1)) == 0,
        "zero_collisions": (
            int(observed.get("unique_collision_incidents", -1)) == 0
            and int(final_summary.get("collisions", -1)) == 0
        ),
        "zero_teleports": (
            int(observed.get("starting_teleports", -1)) == 0
            and int(observed.get("ending_teleports", -1)) == 0
            and int(final_summary.get("teleports", -1)) == 0
        ),
        "exact_controllable_tls": (
            int(observed.get("controllable_tls_count", -1))
            == expected_traffic_lights
        ),
    }


def _operational_preflight_checks(
    *,
    expected_traffic_lights: int,
    observed: dict[str, Any],
    error: str | None,
) -> dict[str, bool]:
    return {
        "runtime_exception_free": error is None,
        "operational_horizon_reached": (
            observed.get("termination_reason") == "operational_horizon_reached"
        ),
        "zero_collisions_during_preflight": (
            int(observed.get("unique_collision_incidents", -1)) == 0
        ),
        "zero_starting_and_ending_teleports_during_preflight": (
            int(observed.get("starting_teleports", -1)) == 0
            and int(observed.get("ending_teleports", -1)) == 0
        ),
        "exact_controllable_tls": (
            int(observed.get("controllable_tls_count", -1))
            == expected_traffic_lights
        ),
    }


def run_eth_city_admission(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    expected_city_code: str,
    expected_sumo_version: str,
    operational_horizon_sec: int | None = None,
    progress_interval_sec: int = 1800,
    state_sample_interval_sec: int = 60,
    diagnostic_start_time_sec: float | None = None,
    diagnostic_vehicle_ids: Sequence[str] = (),
    diagnostic_lane_ids: Sequence[str] = (),
    diagnostic_maximum_samples: int = 200,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, package, package_manifest_sha256 = _validate_package(
        package_root,
        expected_manifest_sha256=expected_package_manifest_sha256,
        expected_city_code=expected_city_code,
    )
    demand = dict(package["demand_inventory"])
    network = dict(package["network_inventory"])
    strict = dict(package["microscopic_instantiation"])
    network_path = package_root.resolve() / str(strict["network_file"])
    expected_vehicle_count = int(demand["trip_count"])
    expected_traffic_lights = int(network["traffic_light_count"])
    seed = int(strict["seed"])
    maximum_time_sec = float(strict["end_sec"])
    begin_time_sec = float(strict["begin_sec"])
    if operational_horizon_sec is not None and operational_horizon_sec <= 0:
        raise ValueError("operational horizon must be positive")
    diagnostic_enabled = diagnostic_start_time_sec is not None
    if diagnostic_enabled and int(diagnostic_maximum_samples) <= 0:
        raise ValueError("diagnostic maximum samples must be positive")
    if not diagnostic_enabled and (diagnostic_vehicle_ids or diagnostic_lane_ids):
        raise ValueError("diagnostic vehicle/lane ids require a diagnostic start time")
    mode = (
        "operational_preflight"
        if operational_horizon_sec is not None
        else "full_admission"
    )
    stop_time_sec = (
        min(maximum_time_sec, begin_time_sec + float(operational_horizon_sec))
        if operational_horizon_sec is not None
        else maximum_time_sec
    )
    actual_sumo_version = libsumo_version()
    if actual_sumo_version != expected_sumo_version:
        raise RuntimeError(
            f"SUMO version mismatch: {actual_sumo_version} != {expected_sumo_version}"
        )

    sumo_api = load_libsumo()
    loaded_vehicle_events = 0
    departed_vehicles = 0
    arrived_vehicles = 0
    starting_teleports = 0
    ending_teleports = 0
    collision_incidents: set[tuple[str, str, str, str]] = set()
    collision_samples: list[dict[str, Any]] = []
    collision_topology_cache: dict[str, dict[str, Any]] = {}
    collision_lane_counts: Counter[str] = Counter()
    max_sampled_active_vehicles = 0
    max_sampled_pending_vehicles = 0
    initial_min_expected = -1
    final_min_expected = -1
    final_active_vehicles = -1
    final_pending_vehicles = -1
    controllable_tls_count = -1
    final_time_sec = begin_time_sec
    termination_reason: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    next_progress = begin_time_sec + float(progress_interval_sec)
    next_state_sample = begin_time_sec
    diagnostic_trace: list[dict[str, Any]] = []

    with sumo_state_directory(prefix=f"cfcmt-eth-{expected_city_code.lower()}-{seed}-") as scratch:
        summary_path = scratch / "summary.xml"
        try:
            arguments = [
                "sumo",
                "-c",
                str(config),
                "--seed",
                str(seed),
                "--summary-output",
                str(summary_path),
                "--summary-output.period",
                str(state_sample_interval_sec),
                "--no-step-log",
                "true",
            ]
            sumo_api.start(arguments)
            initial_min_expected = int(sumo_api.simulation.getMinExpectedNumber())
            controllable_tls_count = len(sumo_api.trafficlight.getIDList())

            while True:
                minimum_expected = int(sumo_api.simulation.getMinExpectedNumber())
                final_time_sec = float(sumo_api.simulation.getTime())
                if operational_horizon_sec is None and minimum_expected == 0:
                    active_vehicles = int(sumo_api.vehicle.getIDCount())
                    pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
                    if active_vehicles == 0 and pending_vehicles == 0:
                        termination_reason = "all_vehicles_completed"
                        break
                if final_time_sec + 1e-9 >= stop_time_sec:
                    termination_reason = (
                        "operational_horizon_reached"
                        if operational_horizon_sec is not None
                        else "fixed_completion_cap_reached"
                    )
                    break

                sumo_api.simulationStep()
                final_time_sec = float(sumo_api.simulation.getTime())
                loaded_vehicle_events += int(sumo_api.simulation.getLoadedNumber())
                departed_vehicles += int(sumo_api.simulation.getDepartedNumber())
                arrived_vehicles += int(sumo_api.simulation.getArrivedNumber())
                starting_teleports += int(
                    sumo_api.simulation.getStartingTeleportNumber()
                )
                ending_teleports += int(
                    sumo_api.simulation.getEndingTeleportNumber()
                )
                if (
                    diagnostic_enabled
                    and final_time_sec + 1e-9 >= float(diagnostic_start_time_sec)
                    and len(diagnostic_trace) < int(diagnostic_maximum_samples)
                ):
                    diagnostic_trace.append(
                        _diagnostic_trace_sample(
                            sumo_api,
                            time_sec=final_time_sec,
                            vehicle_ids=diagnostic_vehicle_ids,
                            lane_ids=diagnostic_lane_ids,
                        )
                    )
                for collision in tuple(sumo_api.simulation.getCollisions()):
                    incident = _collision_incident_key(collision)
                    if incident not in collision_incidents:
                        collision_lane_counts[incident[3]] += 1
                        if len(collision_samples) < 12:
                            collision_samples.append(
                                _collision_sample(
                                    collision,
                                    time_sec=final_time_sec,
                                    sumo_api=sumo_api,
                                    network_path=network_path,
                                    topology_cache=collision_topology_cache,
                                )
                            )
                    collision_incidents.add(incident)

                if collision_incidents:
                    termination_reason = "collision_gate_irreversibly_failed"
                    break
                if starting_teleports or ending_teleports:
                    termination_reason = "teleport_gate_irreversibly_failed"
                    break

                if final_time_sec + 1e-9 >= next_state_sample:
                    active_vehicles = int(sumo_api.vehicle.getIDCount())
                    pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
                    max_sampled_active_vehicles = max(
                        max_sampled_active_vehicles, active_vehicles
                    )
                    max_sampled_pending_vehicles = max(
                        max_sampled_pending_vehicles, pending_vehicles
                    )
                    next_state_sample += float(state_sample_interval_sec)
                if final_time_sec + 1e-9 >= next_progress:
                    print(
                        "ETH_CITY_ADMISSION_PROGRESS "
                        f"city={expected_city_code.upper()} mode={mode} "
                        f"time={final_time_sec:.0f}/{stop_time_sec:.0f} "
                        f"vehicles={departed_vehicles}/{arrived_vehicles} "
                        f"expected={minimum_expected}",
                        flush=True,
                    )
                    next_progress += float(progress_interval_sec)

            final_min_expected = int(sumo_api.simulation.getMinExpectedNumber())
            final_active_vehicles = int(sumo_api.vehicle.getIDCount())
            final_pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if termination_reason is None:
                termination_reason = "runtime_exception"
        finally:
            try:
                sumo_api.close()
            except Exception:
                pass
        try:
            summary = _final_summary_step(summary_path)
        except Exception as exc:
            if error is None:
                error = f"{type(exc).__name__}: {exc}"

    observed = {
        "initial_min_expected": initial_min_expected,
        "loaded_vehicle_event_total": loaded_vehicle_events,
        "departed_vehicles": departed_vehicles,
        "arrived_vehicles": arrived_vehicles,
        "starting_teleports": starting_teleports,
        "ending_teleports": ending_teleports,
        "unique_collision_incidents": len(collision_incidents),
        "collision_samples": collision_samples,
        "unique_collision_incidents_by_lane": dict(
            sorted(collision_lane_counts.items())
        ),
        "max_sampled_active_vehicles": max_sampled_active_vehicles,
        "max_sampled_pending_vehicles": max_sampled_pending_vehicles,
        "state_sample_interval_sec": state_sample_interval_sec,
        "final_active_vehicles": final_active_vehicles,
        "final_pending_vehicles": final_pending_vehicles,
        "final_min_expected": final_min_expected,
        "controllable_tls_count": controllable_tls_count,
        "final_time_sec": final_time_sec,
        "termination_reason": termination_reason,
        "summary": summary,
        "diagnostic_trace": {
            "protocol": DIAGNOSTIC_TRACE_PROTOCOL,
            "enabled": diagnostic_enabled,
            "read_only": True,
            "start_time_sec": diagnostic_start_time_sec,
            "requested_vehicle_ids": [str(value) for value in diagnostic_vehicle_ids],
            "requested_lane_ids": [str(value) for value in diagnostic_lane_ids],
            "maximum_samples": int(diagnostic_maximum_samples),
            "sample_count": len(diagnostic_trace),
            "truncated": len(diagnostic_trace) >= int(diagnostic_maximum_samples),
            "samples": diagnostic_trace,
        },
    }
    if operational_horizon_sec is None:
        checks = _full_admission_checks(
            expected_vehicle_count=expected_vehicle_count,
            expected_traffic_lights=expected_traffic_lights,
            observed=observed,
            summary=summary,
            error=error,
        )
        operational_preflight_passed = None
        scientific_admission_passed = all(checks.values())
    else:
        checks = _operational_preflight_checks(
            expected_traffic_lights=expected_traffic_lights,
            observed=observed,
            error=error,
        )
        operational_preflight_passed = all(checks.values())
        scientific_admission_passed = False
    return {
        "experiment": "traffic_signal_eth_city_admission",
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "passed": scientific_admission_passed,
        "scientific_admission_passed": scientific_admission_passed,
        "operational_preflight_passed": operational_preflight_passed,
        "checks": checks,
        "error": error,
        "setting": {
            "package_root": str(package_root.resolve()),
            "package_manifest_sha256": package_manifest_sha256,
            "sumocfg": str(config),
            "city_code": expected_city_code.upper(),
            "backend": "libsumo",
            "expected_sumo_version": expected_sumo_version,
            "seed": seed,
            "begin_time_sec": begin_time_sec,
            "maximum_time_sec": maximum_time_sec,
            "stop_time_sec": stop_time_sec,
            "operational_horizon_sec": operational_horizon_sec,
            "expected_vehicle_count": expected_vehicle_count,
            "expected_traffic_light_count": expected_traffic_lights,
            "time_based_teleportation_disabled": True,
            "junction_collision_checks_enabled": True,
            "read_only_diagnostic_trace_enabled": diagnostic_enabled,
        },
        "observed": observed,
        "package": {
            "protocol": package["protocol"],
            "scenario": package["scenario"],
            "city_code": package["city_code"],
            "demand_inventory": demand,
            "network_inventory": network,
        },
        "runtime": runtime_metadata(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "Operational preflight never authorizes microscopic admission or "
            "controller evaluation. Only a passing full_admission does so."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-city-code", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--operational-horizon-sec", type=int)
    parser.add_argument("--progress-interval-sec", type=int, default=1800)
    parser.add_argument("--state-sample-interval-sec", type=int, default=60)
    parser.add_argument("--diagnostic-start-time-sec", type=float)
    parser.add_argument("--diagnostic-vehicle-id", action="append", default=[])
    parser.add_argument("--diagnostic-lane-id", action="append", default=[])
    parser.add_argument("--diagnostic-maximum-samples", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_eth_city_admission(
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        expected_city_code=args.expected_city_code,
        expected_sumo_version=args.expected_sumo_version,
        operational_horizon_sec=args.operational_horizon_sec,
        progress_interval_sec=args.progress_interval_sec,
        state_sample_interval_sec=args.state_sample_interval_sec,
        diagnostic_start_time_sec=args.diagnostic_start_time_sec,
        diagnostic_vehicle_ids=args.diagnostic_vehicle_id,
        diagnostic_lane_ids=args.diagnostic_lane_id,
        diagnostic_maximum_samples=args.diagnostic_maximum_samples,
    )
    atomic_write_json(args.out, payload)
    execution_passed = (
        payload["operational_preflight_passed"]
        if payload["mode"] == "operational_preflight"
        else payload["scientific_admission_passed"]
    )
    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "execution_passed": execution_passed,
                "scientific_admission_passed": payload[
                    "scientific_admission_passed"
                ],
                "checks": payload["checks"],
            }
        )
    )
    return 0 if execution_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
