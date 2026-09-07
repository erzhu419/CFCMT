from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.launch_tsc_target_waiting_aligned_adaptation_cache import (
    _parse_physical_node_map,
    build_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_target_waiting_cache_specs_preserve_full_city_b100_pool() -> None:
    protocol = json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_target_pure_waiting_adaptation_cache.json"
        ).read_text(encoding="utf-8")
    )
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        conversion_root=Path("/remote/conversion"),
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
    )
    assert len(specs) == 4
    assert {spec["require_node"] for spec in specs} == {
        "node001",
        "node003",
        "node004",
        "node006",
    }
    assert all(spec["cpu"] == 20 for spec in specs)
    assert all("--seeds 5057 6067" in spec["cmd"] for spec in specs)
    assert all("--counterfactual-horizon-intervals 45" in spec["cmd"] for spec in specs)
    assert all("--rollout-prefix-horizons-sec 450" in spec["cmd"] for spec in specs)
    scenarios = [
        value
        for values in protocol["node_scenario_assignments"].values()
        for value in values
    ]
    assert set(scenarios) == {
        "la_1x4",
        "jinan_3x4_real",
        "jinan_3x4_real_2000",
        "jinan_3x4_real_2500",
    }
    subset = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        conversion_root=Path("/remote/conversion"),
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
        selected_nodes=("node006",),
    )
    assert len(subset) == 1
    assert subset[0]["require_node"] == "node006"
    assert "--scenarios jinan_3x4_real_2000" in subset[0]["cmd"]


def test_target_waiting_cache_can_remap_logical_shards_without_changing_outputs() -> None:
    protocol = json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_target_pure_waiting_adaptation_cache.json"
        ).read_text(encoding="utf-8")
    )
    selected = ("node001", "node003", "node004")
    mapping = _parse_physical_node_map(
        ("node001=node004", "node003=node005", "node004=node006"),
        selected_nodes=selected,
    )
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        conversion_root=Path("/remote/conversion"),
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
        selected_nodes=selected,
        physical_node_overrides=mapping,
    )
    assert [spec["require_node"] for spec in specs] == [
        "node005",
        "node006",
        "node004",
    ]
    assert {spec["result_dir"] for spec in specs} == {
        "/remote/tasks/node001",
        "/remote/tasks/node003",
        "/remote/tasks/node004",
    }
