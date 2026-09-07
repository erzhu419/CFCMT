import json
from pathlib import Path

from scripts.cluster.launch_tsc_external_hierarchical_offline_cache import (
    NODES,
)


def test_hierarchical_offline_cache_uses_all_six_nodes_once() -> None:
    assert NODES == (
        "node001",
        "node002",
        "node003",
        "node004",
        "node005",
        "node006",
    )
    assert len(NODES) == len(set(NODES))


def test_frozen_protocol_has_rectangular_offline_matrix() -> None:
    path = Path(
        "cf_h2o/config/traffic_signal_tsc_v44_external_v9_hierarchical_guard_confirmation.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    seeds = protocol["fresh_offline_confirmation"]["seeds"]
    scenarios = [
        scenario
        for values in protocol["city_scenarios"].values()
        for scenario in values
    ]

    assert len(seeds) == 6
    assert len(scenarios) == 4
    assert 6 * 4 * 16 == 384
