from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_source_waiting_aligned_cache_audit import (
    _scenario_seed_audit,
    _validate_protocol_contract,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    TrafficSignalScenarioSpec,
)
from scripts.cluster.run_tsc_source_waiting_aligned_cache_audit import (
    build_remote_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest(protocol: dict) -> TrafficSignalBenchmarkManifest:
    scenarios = [
        scenario
        for job in protocol["node_jobs"].values()
        for scenario in job["scenarios"]
    ]
    scenarios = list(dict.fromkeys(scenarios))
    group_by_scenario = {
        "atlanta_1x5": "atlanta",
        "cologne1": "cologne",
        "cologne3": "cologne",
        "cologne8": "cologne",
        "hangzhou_bc_tyc": "hangzhou",
        "hangzhou_kn_hz": "hangzhou",
        "hangzhou_qc_yn": "hangzhou",
        "hangzhou_sb_sx": "hangzhou",
        "hangzhou_4x4": "hangzhou",
        "hangzhou_4x4_hetero": "hangzhou",
        "ingolstadt1": "ingolstadt",
        "ingolstadt7": "ingolstadt",
        "ingolstadt21": "ingolstadt",
        "manhattan_28x7": "new_york",
        "grid4x4": "resco_synthetic",
        "arterial4x4": "resco_synthetic",
        "saltlake_400s_200w_q1_weekday_peak": "salt_lake_city",
        "saltlake_state_university_q1_weekday_peak": "salt_lake_city",
    }
    return TrafficSignalBenchmarkManifest(
        path=Path("manifest.json"),
        version=2,
        protocol="test",
        scenarios=tuple(
            TrafficSignalScenarioSpec(
                scenario=scenario,
                city_group=group_by_scenario[scenario],
                suite="test",
                sumocfg=Path(f"{scenario}.sumocfg"),
                provenance="test",
                tls_count=1,
                demand_count=1,
            )
            for scenario in scenarios
        ),
    )


def _protocol() -> dict:
    return json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_source_pure_waiting_cache.json"
        ).read_text(encoding="utf-8")
    )


def test_source_waiting_cache_protocol_matches_full_manifest() -> None:
    protocol = _protocol()
    _validate_protocol_contract(protocol, _manifest(protocol))


def test_source_waiting_cache_protocol_accepts_manhattan_shard_override() -> None:
    protocol = json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_source_pure_waiting_cache_v2.json"
        ).read_text(encoding="utf-8")
    )
    _validate_protocol_contract(protocol, _manifest(protocol))


def test_source_waiting_cache_protocol_rejects_duplicate_scenario() -> None:
    protocol = _protocol()
    protocol["node_jobs"]["node006"]["scenarios"].append("manhattan_28x7")
    with pytest.raises(ValueError, match="duplicated"):
        _validate_protocol_contract(protocol, _manifest(_protocol()))


def test_scenario_seed_audit_requires_unique_action_keys_and_full_tls() -> None:
    common = {
        "rows": 2,
        "groups": 1,
        "behavior_trace": "trace",
        "eligible_opportunities": 9,
        "controllable_tls": {"tls0", "tls1"},
        "covered_tls": {"tls0", "tls1"},
        "full_horizon_equivalence_passed": True,
        "candidate_groups": 1,
        "retained_groups": 1,
        "behavior_collision_steps": 0,
        "passed": True,
    }
    good = [
        {**common, "shard": 0, "action_keys": [("g0", "a0")]},
        {**common, "shard": 1, "action_keys": [("g1", "a0")]},
    ]
    audit = _scenario_seed_audit(
        scenario="grid4x4",
        city_group="resco_synthetic",
        seed=2027,
        shard_count=2,
        items=good,
    )
    assert audit["passed"] is True

    duplicated = [good[0], {**good[1], "action_keys": [("g0", "a0")]}]
    audit = _scenario_seed_audit(
        scenario="grid4x4",
        city_group="resco_synthetic",
        seed=2027,
        shard_count=2,
        items=duplicated,
    )
    assert audit["passed"] is False


def test_scenario_seed_audit_accepts_complete_native_unsafe_exclusion() -> None:
    item = {
        "shard": 0,
        "rows": 0,
        "groups": 0,
        "action_keys": [],
        "behavior_trace": "trace",
        "eligible_opportunities": 10,
        "controllable_tls": {"tls0"},
        "covered_tls": set(),
        "candidate_groups": 8,
        "retained_groups": 0,
        "behavior_collision_steps": 4,
        "full_horizon_equivalence_passed": True,
        "passed": True,
    }
    audit = _scenario_seed_audit(
        scenario="ingolstadt21",
        city_group="ingolstadt",
        seed=2027,
        shard_count=1,
        items=[item],
        excluded=True,
    )
    assert audit["passed"] is True
    assert audit["training_admitted"] is False


def test_source_waiting_cache_audit_command_uses_frozen_protocol() -> None:
    command = build_remote_command(
        snapshot=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        cache_root=Path("/remote/cache"),
        task_root=Path("/remote/tasks"),
        native_safety_audit=Path("/remote/native_safety.json"),
        remote_out=Path("/remote/audit.json"),
        read_workers=32,
    )
    assert "traffic_signal_source_waiting_aligned_cache_audit" in command
    assert "traffic_signal_tsc_v114_source_pure_waiting_admission.json" in command
    assert "--native-safety-audit /remote/native_safety.json" in command
    assert "--read-workers 32" in command
    assert "CFCMT_SOURCE_ROOT=/remote/snapshot" in command
