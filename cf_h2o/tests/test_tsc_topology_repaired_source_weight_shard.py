from scripts.cluster.run_tsc_topology_repaired_source_weight_shard import (
    all_identities,
    shard_identities,
)


def test_source_weight_shards_are_complete_and_disjoint():
    seeds = (11, 13)
    weights = (0.0, 0.5, 1.0)
    expected = set(all_identities(seeds, weights))
    shards = [
        set(
            shard_identities(
                seeds,
                weights,
                shard_index=index,
                shard_count=5,
            )
        )
        for index in range(5)
    ]
    assert set().union(*shards) == expected
    assert sum(len(values) for values in shards) == len(expected)
