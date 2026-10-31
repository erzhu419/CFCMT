"""Run the frozen full-day Paderborn microscopic safety admission."""

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


PROTOCOL = "paderborn-full-day-strict-libsumo-safety-admission-v1"
PACKAGE_PROTOCOL = "cfcmt-zenodo-paderborn-full-day-sumo122-package-v1"
SCENARIO = "paderborn_full_day"


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
            "Paderborn package manifest identity changed: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected Paderborn package protocol: {manifest.get('protocol')}")
    if manifest.get("scenario") != SCENARIO or manifest.get("passed") is not True:
        raise ValueError("Paderborn package did not pass frozen static admission")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise ValueError("Paderborn package contains a failed static gate")
    route_inventory = dict(manifest.get("route_inventory", {}))
    if int(route_inventory.get("vehicle_count", -1)) != 203_387:
        raise ValueError("Paderborn package does not contain all 203,387 vehicles")
    config = package_root / str(
        dict(manifest.get("microscopic_instantiation", {})).get("config", "")
    )
    if not config.is_file():
        raise FileNotFoundError(f"Paderborn strict SUMO config is missing: {config}")
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
    expected_minimum_tls: int,
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
        "exact_demand_loaded": (
            int(final_summary.get("loaded", -1)) == expected_vehicle_count
        ),
        "exact_demand_departed": (
            int(observed.get("departed", -1)) == expected_vehicle_count
        ),
        "exact_demand_arrived": (
            int(observed.get("arrived", -1)) == expected_vehicle_count
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
        "minimum_controllable_tls": (
            int(observed.get("controllable_tls_count", -1))
            >= expected_minimum_tls
        ),
    }


def run_paderborn_admission(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    expected_sumo_version: str,
    seed: int,
    maximum_time_sec: float,
    expected_minimum_tls: int,
    progress_interval_sec: int = 1800,
    state_sample_interval_sec: int = 60,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, package, package_manifest_sha256 = _validate_package(
        package_root,
        expected_manifest_sha256=expected_package_manifest_sha256,
    )
    expected_vehicle_count = int(package["route_inventory"]["vehicle_count"])
    actual_sumo_version = libsumo_version()
    if actual_sumo_version != expected_sumo_version:
        raise RuntimeError(
            f"SUMO version mismatch: {actual_sumo_version} != {expected_sumo_version}"
        )

    sumo_api = load_libsumo()
    loaded = 0
    departed = 0
    arrived = 0
    starting_teleports = 0
    ending_teleports = 0
    collision_incidents: set[tuple[str, str, str, str]] = set()
    collision_samples: list[dict[str, Any]] = []
    collision_lane_counts: Counter[str] = Counter()
    max_sampled_active_vehicles = 0
    max_sampled_pending_vehicles = 0
    initial_min_expected = -1
    final_min_expected = -1
    final_active_vehicles = -1
    final_pending_vehicles = -1
    controllable_tls_count = -1
    final_time_sec = 0.0
    termination_reason: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    next_progress = float(progress_interval_sec)
    next_state_sample = 0.0

    with sumo_state_directory(prefix=f"cfcmt-paderborn-{seed}-") as scratch:
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
                if minimum_expected == 0:
                    termination_reason = "all_vehicles_completed"
                    break
                if final_time_sec + 1e-9 >= maximum_time_sec:
                    termination_reason = "fixed_completion_cap_reached"
                    break

                sumo_api.simulationStep()
                final_time_sec = float(sumo_api.simulation.getTime())
                loaded += int(sumo_api.simulation.getLoadedNumber())
                departed += int(sumo_api.simulation.getDepartedNumber())
                arrived += int(sumo_api.simulation.getArrivedNumber())
                starting_teleports += int(
                    sumo_api.simulation.getStartingTeleportNumber()
                )
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
                    active = int(sumo_api.vehicle.getIDCount())
                    pending = len(sumo_api.simulation.getPendingVehicles())
                    max_sampled_active_vehicles = max(
                        max_sampled_active_vehicles, active
                    )
                    max_sampled_pending_vehicles = max(
                        max_sampled_pending_vehicles, pending
                    )
                    next_state_sample += float(state_sample_interval_sec)
                if final_time_sec + 1e-9 >= next_progress:
                    print(
                        "PADERBORN_ADMISSION_PROGRESS "
                        f"time={final_time_sec:.0f}/{maximum_time_sec:.0f} "
                        f"departed={departed} arrived={arrived} "
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
        "loaded_event_total": loaded,
        "departed": departed,
        "arrived": arrived,
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
    }
    checks = _admission_checks(
        expected_vehicle_count=expected_vehicle_count,
        expected_minimum_tls=expected_minimum_tls,
        observed=observed,
        summary=summary,
        error=error,
    )
    return {
        "experiment": "traffic_signal_paderborn_full_day_admission",
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
            "seed": seed,
            "maximum_time_sec": maximum_time_sec,
            "expected_vehicle_count": expected_vehicle_count,
            "expected_minimum_tls": expected_minimum_tls,
            "time_based_teleportation_disabled": True,
            "junction_collision_checks_enabled": True,
        },
        "observed": observed,
        "package": {
            "protocol": package["protocol"],
            "scenario": package["scenario"],
            "route_inventory": package["route_inventory"],
            "network_inventory": package["network_inventory"],
            "external_tls_inventory": package["external_tls_inventory"],
        },
        "runtime": runtime_metadata(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--seed", type=int, default=5057)
    parser.add_argument("--maximum-time-sec", type=float, default=108000.0)
    parser.add_argument("--expected-minimum-tls", type=int, default=128)
    parser.add_argument("--progress-interval-sec", type=int, default=1800)
    parser.add_argument("--state-sample-interval-sec", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_paderborn_admission(
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        expected_sumo_version=args.expected_sumo_version,
        seed=args.seed,
        maximum_time_sec=args.maximum_time_sec,
        expected_minimum_tls=args.expected_minimum_tls,
        progress_interval_sec=args.progress_interval_sec,
        state_sample_interval_sec=args.state_sample_interval_sec,
    )
    atomic_write_json(args.out, payload)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
