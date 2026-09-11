from __future__ import annotations

from pathlib import Path

from cf_h2o.eval.traffic_signal_eth_city_admission import DIAGNOSTIC_TRACE_PROTOCOL
from cf_h2o.eval.traffic_signal_eth_city_collision_diagnostic import (
    collision_diagnostic_checks,
)
from scripts.cluster.launch_eth_boston_v18_collision_diagnostic import (
    EXPECTED_COLLIDER_ID,
    EXPECTED_COLLISION_LANE,
    EXPECTED_COLLISION_TIME_SEC,
    EXPECTED_VICTIM_ID,
    TRACE_START_TIME_SEC,
    build_spec,
)


def test_collision_diagnostic_requires_exact_reproduction_and_trace() -> None:
    admission = {
        "error": None,
        "operational_preflight_passed": False,
        "observed": {
            "termination_reason": "collision_gate_irreversibly_failed",
            "unique_collision_incidents": 1,
            "final_time_sec": EXPECTED_COLLISION_TIME_SEC,
            "collision_samples": [
                {
                    "time_sec": EXPECTED_COLLISION_TIME_SEC,
                    "lane": EXPECTED_COLLISION_LANE,
                    "collider": EXPECTED_COLLIDER_ID,
                    "victim": EXPECTED_VICTIM_ID,
                }
            ],
            "diagnostic_trace": {
                "protocol": DIAGNOSTIC_TRACE_PROTOCOL,
                "enabled": True,
                "read_only": True,
                "start_time_sec": TRACE_START_TIME_SEC,
                "truncated": False,
                "samples": [
                    {"time_sec": TRACE_START_TIME_SEC},
                    {"time_sec": EXPECTED_COLLISION_TIME_SEC},
                ],
            },
        },
    }
    checks = collision_diagnostic_checks(
        admission,
        expected_collision_time_sec=EXPECTED_COLLISION_TIME_SEC,
        expected_lane_id=EXPECTED_COLLISION_LANE,
        expected_collider_id=EXPECTED_COLLIDER_ID,
        expected_victim_id=EXPECTED_VICTIM_ID,
        expected_trace_start_time_sec=TRACE_START_TIME_SEC,
    )
    assert all(checks.values())
    admission["observed"]["collision_samples"][0]["victim"] = "changed"
    assert collision_diagnostic_checks(
        admission,
        expected_collision_time_sec=EXPECTED_COLLISION_TIME_SEC,
        expected_lane_id=EXPECTED_COLLISION_LANE,
        expected_collider_id=EXPECTED_COLLIDER_ID,
        expected_victim_id=EXPECTED_VICTIM_ID,
        expected_trace_start_time_sec=TRACE_START_TIME_SEC,
    )["collision_identity_reproduced"] is False


def test_collision_diagnostic_spec_is_bounded_and_libsumo_backed() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/data/boston_v18"),
        expected_package_manifest_sha256="abc",
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    command = spec["cmd"]
    assert "traffic_signal_eth_city_collision_diagnostic" in command
    assert f"--trace-start-time-sec {TRACE_START_TIME_SEC}" in command
    assert f"--expected-collision-time-sec {EXPECTED_COLLISION_TIME_SEC}" in command
    assert command.count("--trace-lane-id") == 2
    assert command.count("--trace-vehicle-id") == 2
    assert "traci" not in command.lower()
    assert spec["allowed_nodes"]
    assert spec["result_dir"] == "/remote/result"
