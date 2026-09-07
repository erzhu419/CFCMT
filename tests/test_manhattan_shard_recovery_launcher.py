from __future__ import annotations

from pathlib import Path

from scripts.cluster.launch_tsc_pure_waiting_manhattan_shards import (
    NODES,
    assign_pairs,
    build_specs,
)


def _protocol() -> dict:
    return {
        "collection": {
            "seeds": [2027, 3037, 4047],
            "collection_shards_per_seed": 16,
            "scenario_collection_shards_per_seed": {"manhattan_28x7": 64},
            "duration_sec": 3600,
            "control_interval_sec": 10,
            "warmup_sec": 60,
            "max_focal_tls": 4,
            "counterfactual_horizon_intervals": 45,
            "behavior_policy": "phase_pressure",
            "counterfactual_cost_mode": "halted_queue",
            "rollout_prefix_horizons_sec": [450],
            "sumo_version": "1.22.0",
        }
    }


def test_assign_pairs_balances_complete_matrix_without_overlap() -> None:
    pairs = [(seed, shard) for seed in (2027, 3037, 4047) for shard in range(16)]
    assignments = assign_pairs(pairs)
    flattened = [pair for node in NODES for pair in assignments[node]]
    assert sorted(flattened) == sorted(pairs)
    assert len(flattened) == len(set(flattened)) == 48
    assert {len(assignments[node]) for node in NODES} == {8}


def test_specs_pin_each_assignment_and_use_at_most_twenty_workers() -> None:
    assignments = assign_pairs(
        [(seed, shard) for seed in (2027, 3037, 4047) for shard in range(64)]
    )
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        protocol=_protocol(),
        assignments=assignments,
        remote_cache_root=Path("/results/cache"),
        remote_task_root=Path("/results/tasks"),
    )
    assert len(specs) == 6
    assert {spec["require_node"] for spec in specs} == set(NODES)
    assert all(spec["cpu"] == 20 for spec in specs)
    assert all("--collection-shards 64" in spec["cmd"] for spec in specs)
    assert all("--counterfactual-cost-mode halted_queue" in spec["cmd"] for spec in specs)
