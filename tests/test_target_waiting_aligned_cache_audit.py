from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_target_waiting_aligned_cache_audit import (
    _validate_protocol_contract,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    TrafficSignalScenarioSpec,
)
from scripts.cluster.run_tsc_target_waiting_aligned_cache_audit import (
    build_remote_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_target_pure_waiting_adaptation_cache.json"
        ).read_text(encoding="utf-8")
    )


def _manifest(protocol: dict) -> TrafficSignalBenchmarkManifest:
    scenarios = [
        scenario
        for assigned in protocol["node_scenario_assignments"].values()
        for scenario in assigned
    ]
    return TrafficSignalBenchmarkManifest(
        path=Path("manifest.json"),
        version=2,
        protocol="test",
        scenarios=tuple(
            TrafficSignalScenarioSpec(
                scenario=scenario,
                city_group=("los_angeles" if scenario == "la_1x4" else "jinan"),
                suite="test",
                sumocfg=Path(f"{scenario}.sumocfg"),
                provenance="test",
                tls_count=1,
                demand_count=1,
            )
            for scenario in scenarios
        ),
    )


def test_target_waiting_cache_protocol_matches_full_external_manifest() -> None:
    protocol = _protocol()
    _validate_protocol_contract(protocol, _manifest(protocol))


def test_target_waiting_cache_protocol_rejects_one_scenario_shortcut() -> None:
    protocol = _protocol()
    protocol["node_scenario_assignments"]["node001"] = ["jinan_3x4_real"]
    with pytest.raises(ValueError, match="duplicates"):
        _validate_protocol_contract(protocol, _manifest(_protocol()))


def test_target_waiting_cache_audit_command_uses_full_protocol() -> None:
    command = build_remote_command(
        snapshot=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        cache_root=Path("/remote/cache"),
        task_root=Path("/remote/tasks"),
        remote_out=Path("/remote/audit.json"),
        read_workers=32,
    )
    assert "traffic_signal_target_waiting_aligned_cache_audit" in command
    assert "traffic_signal_tsc_v114_target_pure_waiting_adaptation_cache.json" in command
    assert "--read-workers 32" in command
