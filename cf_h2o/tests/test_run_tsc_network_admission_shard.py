from scripts.cluster.run_tsc_network_admission_shard import build_jobs


def test_admission_shards_are_disjoint_and_complete() -> None:
    protocol = {
        "scenarios": ["a", "b", "c"],
        "seeds": [1, 2, 3, 4],
    }
    shards = [
        build_jobs(protocol=protocol, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    flattened = [row for shard in shards for row in shard]

    assert len(flattened) == 12
    assert len(set(flattened)) == 12
    assert set(flattened) == {
        (scenario, seed)
        for scenario in protocol["scenarios"]
        for seed in protocol["seeds"]
    }
