"""Run the frozen full-day InTAS microscopic safety admission."""

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


PROTOCOL = "intas-full-day-strict-libsumo-safety-admission-v1"
PACKAGE_PROTOCOL = "cfcmt-intas-full-day-sumo122-package-v1"
SCENARIO = "intas_ingolstadt_full_day"
SEED = 5107
MAXIMUM_TIME_SEC = 108_000.0
EXPECTED_CAR_VEHICLES = 185_923
EXPECTED_BUS_VEHICLES = 1_581
EXPECTED_PERSONS = 254
EXPECTED_TRAFFIC_LIGHTS = 98


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
) -> tuple[Path, dict[str, Any], str]:
    package_root = package_root.resolve()
    manifest_path = package_root / "package_manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "InTAS package manifest identity changed: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected InTAS package protocol: {manifest.get('protocol')}")
    if manifest.get("scenario") != SCENARIO or manifest.get("passed") is not True:
        raise ValueError("InTAS package did not pass frozen static admission")
    if manifest.get("static_failures") != []:
        raise ValueError("InTAS package contains a failed static gate")
    if int(manifest.get("workers", -1)) != 8:
        raise ValueError("InTAS package was not audited with the frozen 8 workers")

    cars = dict(manifest.get("car_demand_inventory", {}))
    buses = dict(manifest.get("bus_demand_inventory", {}))
    pedestrians = dict(manifest.get("pedestrian_demand_inventory", {}))
    network = dict(manifest.get("network_inventory", {}))
    expected = {
        "car_vehicle_count": EXPECTED_CAR_VEHICLES,
        "bus_vehicle_count": EXPECTED_BUS_VEHICLES,
        "person_count": EXPECTED_PERSONS,
        "traffic_light_count": EXPECTED_TRAFFIC_LIGHTS,
    }
    observed = {
        "car_vehicle_count": int(cars.get("vehicle_count", -1)),
        "bus_vehicle_count": int(buses.get("expanded_vehicle_count", -1)),
        "person_count": int(pedestrians.get("person_count", -1)),
        "traffic_light_count": int(network.get("traffic_light_count", -1)),
    }
    if observed != expected:
        raise ValueError(f"InTAS frozen inventory changed: {observed} != {expected}")

    strict = dict(manifest.get("strict_execution", {}))
    frozen_execution = {
        "backend": "libsumo",
        "sumo_version": "1.22.0",
        "seed": SEED,
        "begin_sec": 0,
        "end_sec": int(MAXIMUM_TIME_SEC),
        "step_length_sec": 0.1,
        "time_to_teleport_sec": -1,
        "max_depart_delay_sec": -1,
        "demand_scale": 1.0,
        "collision_check_junctions": True,
        "collision_action": "warn",
    }
    strict_observed = {key: strict.get(key) for key in frozen_execution}
    if strict_observed != frozen_execution:
        raise ValueError(
            f"InTAS strict execution changed: {strict_observed} != {frozen_execution}"
        )
    config = package_root / str(strict.get("config", ""))
    if not config.is_file():
        raise FileNotFoundError(f"InTAS strict SUMO config is missing: {config}")
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


def _collision_sample(collision: Any, *, time_sec: float) -> dict[str, Any]:
    sample: dict[str, Any] = {"repr": repr(collision), "time_sec": time_sec}
    for field in ("collider", "victim", "type", "lane", "pos"):
        value = getattr(collision, field, None)
        if value is not None:
            sample[field] = value
    return sample


def _admission_checks(
    *,
    expected_vehicle_count: int,
    expected_person_count: int,
    expected_traffic_lights: int,
    observed: dict[str, Any],
    summary: dict[str, Any] | None,
    error: str | None,
) -> dict[str, bool]:
    final_summary = summary or {}
    return {
        "runtime_exception_free": error is None,
        "all_entities_completed_before_cap": (
            observed.get("termination_reason") == "all_entities_completed"
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
        "exact_person_demand_departed": (
            int(observed.get("departed_persons", -1)) == expected_person_count
        ),
        "exact_person_demand_arrived": (
            int(observed.get("arrived_persons", -1)) == expected_person_count
        ),
        "no_running_vehicles": (
            int(observed.get("final_active_vehicles", -1)) == 0
            and int(final_summary.get("running", -1)) == 0
        ),
        "no_running_persons": int(observed.get("final_active_persons", -1)) == 0,
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


def run_intas_admission(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    expected_sumo_version: str,
    progress_interval_sec: int = 1800,
    state_sample_interval_sec: int = 60,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, package, package_manifest_sha256 = _validate_package(
        package_root,
        expected_manifest_sha256=expected_package_manifest_sha256,
    )
    expected_vehicle_count = EXPECTED_CAR_VEHICLES + EXPECTED_BUS_VEHICLES
    actual_sumo_version = libsumo_version()
    if actual_sumo_version != expected_sumo_version:
        raise RuntimeError(
            f"SUMO version mismatch: {actual_sumo_version} != {expected_sumo_version}"
        )

    sumo_api = load_libsumo()
    loaded_vehicle_events = 0
    departed_vehicles = 0
    arrived_vehicles = 0
    departed_persons = 0
    arrived_persons = 0
    starting_teleports = 0
    ending_teleports = 0
    collision_incidents: set[tuple[str, str, str, str]] = set()
    collision_samples: list[dict[str, Any]] = []
    collision_lane_counts: Counter[str] = Counter()
    max_sampled_active_vehicles = 0
    max_sampled_active_persons = 0
    max_sampled_pending_vehicles = 0
    initial_min_expected = -1
    final_min_expected = -1
    final_active_vehicles = -1
    final_active_persons = -1
    final_pending_vehicles = -1
    controllable_tls_count = -1
    final_time_sec = 0.0
    termination_reason: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    next_progress = float(progress_interval_sec)
    next_state_sample = 0.0

    with sumo_state_directory(prefix=f"cfcmt-intas-{SEED}-") as scratch:
        summary_path = scratch / "summary.xml"
        try:
            arguments = [
                "sumo",
                "-c",
                str(config),
                "--seed",
                str(SEED),
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
                if minimum_expected == 0:
                    active_vehicles = int(sumo_api.vehicle.getIDCount())
                    active_persons = int(sumo_api.person.getIDCount())
                    pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
                    if active_vehicles == 0 and active_persons == 0 and pending_vehicles == 0:
                        termination_reason = "all_entities_completed"
                        break
                if final_time_sec + 1e-9 >= MAXIMUM_TIME_SEC:
                    termination_reason = "fixed_completion_cap_reached"
                    break

                sumo_api.simulationStep()
                final_time_sec = float(sumo_api.simulation.getTime())
                loaded_vehicle_events += int(sumo_api.simulation.getLoadedNumber())
                departed_vehicles += int(sumo_api.simulation.getDepartedNumber())
                arrived_vehicles += int(sumo_api.simulation.getArrivedNumber())
                departed_persons += int(sumo_api.simulation.getDepartedPersonNumber())
                arrived_persons += int(sumo_api.simulation.getArrivedPersonNumber())
                starting_teleports += int(sumo_api.simulation.getStartingTeleportNumber())
                ending_teleports += int(sumo_api.simulation.getEndingTeleportNumber())
                for collision in tuple(sumo_api.simulation.getCollisions()):
                    incident = _collision_incident_key(collision)
                    if incident not in collision_incidents:
                        collision_lane_counts[incident[3]] += 1
                        if len(collision_samples) < 12:
                            collision_samples.append(
                                _collision_sample(collision, time_sec=final_time_sec)
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
                    active_persons = int(sumo_api.person.getIDCount())
                    pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
                    max_sampled_active_vehicles = max(
                        max_sampled_active_vehicles, active_vehicles
                    )
                    max_sampled_active_persons = max(
                        max_sampled_active_persons, active_persons
                    )
                    max_sampled_pending_vehicles = max(
                        max_sampled_pending_vehicles, pending_vehicles
                    )
                    next_state_sample += float(state_sample_interval_sec)
                if final_time_sec + 1e-9 >= next_progress:
                    print(
                        "INTAS_ADMISSION_PROGRESS "
                        f"time={final_time_sec:.0f}/{MAXIMUM_TIME_SEC:.0f} "
                        f"vehicles={departed_vehicles}/{arrived_vehicles} "
                        f"persons={departed_persons}/{arrived_persons} "
                        f"expected={minimum_expected}",
                        flush=True,
                    )
                    next_progress += float(progress_interval_sec)

            final_min_expected = int(sumo_api.simulation.getMinExpectedNumber())
            final_active_vehicles = int(sumo_api.vehicle.getIDCount())
            final_active_persons = int(sumo_api.person.getIDCount())
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
        "departed_persons": departed_persons,
        "arrived_persons": arrived_persons,
        "starting_teleports": starting_teleports,
        "ending_teleports": ending_teleports,
        "unique_collision_incidents": len(collision_incidents),
        "collision_samples": collision_samples,
        "unique_collision_incidents_by_lane": dict(
            sorted(collision_lane_counts.items())
        ),
        "max_sampled_active_vehicles": max_sampled_active_vehicles,
        "max_sampled_active_persons": max_sampled_active_persons,
        "max_sampled_pending_vehicles": max_sampled_pending_vehicles,
        "state_sample_interval_sec": state_sample_interval_sec,
        "final_active_vehicles": final_active_vehicles,
        "final_active_persons": final_active_persons,
        "final_pending_vehicles": final_pending_vehicles,
        "final_min_expected": final_min_expected,
        "controllable_tls_count": controllable_tls_count,
        "final_time_sec": final_time_sec,
        "termination_reason": termination_reason,
        "summary": summary,
    }
    checks = _admission_checks(
        expected_vehicle_count=expected_vehicle_count,
        expected_person_count=EXPECTED_PERSONS,
        expected_traffic_lights=EXPECTED_TRAFFIC_LIGHTS,
        observed=observed,
        summary=summary,
        error=error,
    )
    return {
        "experiment": "traffic_signal_intas_full_day_admission",
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "error": error,
        "setting": {
            "package_root": str(package_root.resolve()),
            "package_manifest_sha256": package_manifest_sha256,
            "sumocfg": str(config),
            "backend": "libsumo",
            "expected_sumo_version": expected_sumo_version,
            "seed": SEED,
            "maximum_time_sec": MAXIMUM_TIME_SEC,
            "expected_car_vehicle_count": EXPECTED_CAR_VEHICLES,
            "expected_bus_vehicle_count": EXPECTED_BUS_VEHICLES,
            "expected_vehicle_count": expected_vehicle_count,
            "expected_person_count": EXPECTED_PERSONS,
            "expected_traffic_light_count": EXPECTED_TRAFFIC_LIGHTS,
            "time_based_teleportation_disabled": True,
            "junction_collision_checks_enabled": True,
        },
        "observed": observed,
        "package": {
            "protocol": package["protocol"],
            "scenario": package["scenario"],
            "car_demand_inventory": package["car_demand_inventory"],
            "bus_demand_inventory": package["bus_demand_inventory"],
            "pedestrian_demand_inventory": package[
                "pedestrian_demand_inventory"
            ],
            "network_inventory": package["network_inventory"],
        },
        "runtime": runtime_metadata(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--progress-interval-sec", type=int, default=1800)
    parser.add_argument("--state-sample-interval-sec", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_intas_admission(
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        expected_sumo_version=args.expected_sumo_version,
        progress_interval_sec=args.progress_interval_sec,
        state_sample_interval_sec=args.state_sample_interval_sec,
    )
    atomic_write_json(args.out, payload)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
