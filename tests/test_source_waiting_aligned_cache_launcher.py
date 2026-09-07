from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.launch_tsc_source_waiting_aligned_cache import build_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_waiting_cache_specs_cover_all_scenarios_once() -> None:
    protocol = json.loads(
        (
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v114_source_pure_waiting_cache.json"
        ).read_text(encoding="utf-8")
    )

    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol=protocol,
        remote_cache_root=Path("/remote/cache"),
        remote_task_root=Path("/remote/tasks"),
    )

    assert len(specs) == 6
    assert {spec["require_node"] for spec in specs} == {
        f"node00{index}" for index in range(1, 7)
    }
    assert all(spec["cpu"] == 20 for spec in specs)
    assert all("--counterfactual-horizon-intervals 45" in spec["cmd"] for spec in specs)
    assert all("--rollout-prefix-horizons-sec 450" in spec["cmd"] for spec in specs)
    scenarios = [
        value
        for job in protocol["node_jobs"].values()
        for value in job["scenarios"]
    ]
    assert len(set(scenarios)) == 18
    scenario_seeds = {
        (scenario, seed)
        for job in protocol["node_jobs"].values()
        for scenario in job["scenarios"]
        for seed in job["seeds"]
    }
    assert len(scenario_seeds) == 54
    assert {
        tuple(job["seeds"])
        for job in list(protocol["node_jobs"].values())[:3]
    } == {(2027,), (3037,), (4047,)}
