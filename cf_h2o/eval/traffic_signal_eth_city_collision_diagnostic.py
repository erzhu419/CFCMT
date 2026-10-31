"""Reproduce one admitted-package collision with a bounded read-only trace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_eth_city_admission import (
    DIAGNOSTIC_TRACE_PROTOCOL,
    run_eth_city_admission,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


PROTOCOL = "eth-city-collision-read-only-diagnostic-v1"


def collision_diagnostic_checks(
    admission: Mapping[str, Any],
    *,
    expected_collision_time_sec: float,
    expected_lane_id: str,
    expected_collider_id: str,
    expected_victim_id: str,
    expected_trace_start_time_sec: float,
) -> dict[str, bool]:
    observed = dict(admission.get("observed", {}))
    samples = list(observed.get("collision_samples", ()))
    collision = dict(samples[0]) if len(samples) == 1 else {}
    trace = dict(observed.get("diagnostic_trace", {}))
    trace_samples = list(trace.get("samples", ()))
    return {
        "admission_failed_only_at_collision_gate": (
            admission.get("error") is None
            and admission.get("operational_preflight_passed") is False
            and observed.get("termination_reason")
            == "collision_gate_irreversibly_failed"
        ),
        "one_collision_reproduced": int(
            observed.get("unique_collision_incidents", -1)
        )
        == 1,
        "collision_time_reproduced": float(
            observed.get("final_time_sec", -1.0)
        )
        == float(expected_collision_time_sec)
        and float(collision.get("time_sec", -1.0))
        == float(expected_collision_time_sec),
        "collision_identity_reproduced": (
            str(collision.get("lane", "")) == str(expected_lane_id)
            and str(collision.get("collider", "")) == str(expected_collider_id)
            and str(collision.get("victim", "")) == str(expected_victim_id)
        ),
        "read_only_trace_contract": (
            trace.get("protocol") == DIAGNOSTIC_TRACE_PROTOCOL
            and trace.get("enabled") is True
            and trace.get("read_only") is True
            and float(trace.get("start_time_sec", -1.0))
            == float(expected_trace_start_time_sec)
            and trace.get("truncated") is False
        ),
        "trace_covers_collision_step": (
            bool(trace_samples)
            and float(dict(trace_samples[0]).get("time_sec", -1.0))
            == float(expected_trace_start_time_sec)
            and float(dict(trace_samples[-1]).get("time_sec", -1.0))
            == float(expected_collision_time_sec)
        ),
    }


def run_collision_diagnostic(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    expected_city_code: str,
    expected_sumo_version: str,
    operational_horizon_sec: int,
    expected_collision_time_sec: float,
    expected_lane_id: str,
    expected_collider_id: str,
    expected_victim_id: str,
    trace_start_time_sec: float,
    trace_lane_ids: Sequence[str],
    trace_vehicle_ids: Sequence[str],
    diagnostic_maximum_samples: int,
) -> dict[str, Any]:
    admission = run_eth_city_admission(
        package_root=package_root,
        expected_package_manifest_sha256=expected_package_manifest_sha256,
        expected_city_code=expected_city_code,
        expected_sumo_version=expected_sumo_version,
        operational_horizon_sec=operational_horizon_sec,
        progress_interval_sec=900,
        state_sample_interval_sec=60,
        diagnostic_start_time_sec=trace_start_time_sec,
        diagnostic_vehicle_ids=trace_vehicle_ids,
        diagnostic_lane_ids=trace_lane_ids,
        diagnostic_maximum_samples=diagnostic_maximum_samples,
    )
    checks = collision_diagnostic_checks(
        admission,
        expected_collision_time_sec=expected_collision_time_sec,
        expected_lane_id=expected_lane_id,
        expected_collider_id=expected_collider_id,
        expected_victim_id=expected_victim_id,
        expected_trace_start_time_sec=trace_start_time_sec,
    )
    return {
        "experiment": "traffic_signal_eth_city_collision_diagnostic",
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "expected_collision": {
            "time_sec": float(expected_collision_time_sec),
            "lane_id": str(expected_lane_id),
            "collider_id": str(expected_collider_id),
            "victim_id": str(expected_victim_id),
        },
        "admission_result": admission,
        "claim_boundary": (
            "Passing means the previously observed collision and bounded read-only "
            "trace were reproduced. It does not admit the Boston package."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-city-code", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--operational-horizon-sec", type=int, required=True)
    parser.add_argument("--expected-collision-time-sec", type=float, required=True)
    parser.add_argument("--expected-lane-id", required=True)
    parser.add_argument("--expected-collider-id", required=True)
    parser.add_argument("--expected-victim-id", required=True)
    parser.add_argument("--trace-start-time-sec", type=float, required=True)
    parser.add_argument("--trace-lane-id", action="append", default=[])
    parser.add_argument("--trace-vehicle-id", action="append", default=[])
    parser.add_argument("--diagnostic-maximum-samples", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_collision_diagnostic(
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        expected_city_code=args.expected_city_code,
        expected_sumo_version=args.expected_sumo_version,
        operational_horizon_sec=args.operational_horizon_sec,
        expected_collision_time_sec=args.expected_collision_time_sec,
        expected_lane_id=args.expected_lane_id,
        expected_collider_id=args.expected_collider_id,
        expected_victim_id=args.expected_victim_id,
        trace_start_time_sec=args.trace_start_time_sec,
        trace_lane_ids=args.trace_lane_id,
        trace_vehicle_ids=args.trace_vehicle_id,
        diagnostic_maximum_samples=args.diagnostic_maximum_samples,
    )
    atomic_write_json(args.out, payload)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
