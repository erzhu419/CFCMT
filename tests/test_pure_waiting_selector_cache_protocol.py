from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (
    _validate_protocol_contract,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    TrafficSignalScenarioSpec,
)
from scripts.cluster.launch_tsc_pure_waiting_selector_development_cache import (
    build_specs,
)
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import (
    SEEDS,
    _require_selector_cache_audit,
)
from scripts.cluster.run_tsc_pure_waiting_selector_cache_audit import (
    build_remote_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v116_pure_waiting_"
            "selector_development_cache.json"
        ).read_text(encoding="utf-8")
    )


def _manifest() -> TrafficSignalBenchmarkManifest:
    return TrafficSignalBenchmarkManifest(
        path=Path("manifest.json"),
        version=2,
        protocol="test",
        scenarios=(
            TrafficSignalScenarioSpec(
                scenario="jinan_3x4_real",
                city_group="jinan",
                suite="test",
                sumocfg=Path("jinan.sumocfg"),
                provenance="test",
                tls_count=12,
                demand_count=1,
            ),
        ),
    )


def test_selector_cache_protocol_covers_all_22_seeds_and_704_shards() -> None:
    protocol = _protocol()
    _validate_protocol_contract(protocol, _manifest())
    assert protocol["expected_seed_count"] == 22
    assert protocol["expected_cache_file_count"] == 704


def test_selector_cache_protocol_rejects_duplicate_seed() -> None:
    protocol = _protocol()
    protocol["node_seed_assignments"]["node006"][-1] = 80314
    with pytest.raises(ValueError, match="assignment order"):
        _validate_protocol_contract(protocol, _manifest())


def test_selector_cache_launcher_limits_each_node_to_20_workers() -> None:
    protocol = _protocol()
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        conversion_root=Path(protocol["conversion_input"]["root"]),
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
    )
    assert len(specs) == 6
    assert all(spec["cpu"] == 20 for spec in specs)
    assert all("--rollout-prefix-horizons-sec 450" in spec["cmd"] for spec in specs)
    assert all("--counterfactual-cost-mode halted_queue" in spec["cmd"] for spec in specs)


def test_selector_cache_launcher_can_submit_disjoint_node_subset() -> None:
    protocol = _protocol()
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        conversion_root=Path(protocol["conversion_input"]["root"]),
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
        selected_nodes=("node006",),
    )
    assert [spec["require_node"] for spec in specs] == ["node006"]
    assert "--seeds 74744 74374 50079" in specs[0]["cmd"]


def test_selector_cache_audit_command_uses_v116_protocol() -> None:
    command = build_remote_command(
        snapshot=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        cache_root=Path("/remote/cache"),
        task_root=Path("/remote/tasks"),
        remote_out=Path("/remote/audit.json"),
        read_workers=32,
    )
    assert "traffic_signal_pure_waiting_selector_cache_audit" in command
    assert "traffic_signal_tsc_v116_pure_waiting_selector" in command
    assert "--read-workers 32" in command


def test_v116_selector_launcher_requires_complete_cache_audit(tmp_path) -> None:
    path = tmp_path / "audit.json"
    payload = {
        "protocol": "tsc-v116-pure-waiting-selector-cache-audit-v1",
        "status": "PASS",
        "decision": "authorize_v116_pure_waiting_nested_source_selection",
        "gate": {"passed": True},
        "scenario": "jinan_3x4_real",
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "shards_per_seed": 32,
        "cache_file_count": len(SEEDS) * 32,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _require_selector_cache_audit(path)["status"] == "PASS"

    payload["cache_file_count"] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        _require_selector_cache_audit(path)
