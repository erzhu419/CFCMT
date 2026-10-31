from __future__ import annotations

from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    _node_assignments,
)


def test_node_assignments_cover_22_seeds_once_and_balance_workers() -> None:
    seeds = list(range(22))
    assignments = _node_assignments(seeds)
    flattened = [seed for values in assignments.values() for seed in values]
    assert flattened == seeds
    assert sorted(len(values) for values in assignments.values()) == [3, 3, 4, 4, 4, 4]
    assert max(len(values) * 32 for values in assignments.values()) == 128
