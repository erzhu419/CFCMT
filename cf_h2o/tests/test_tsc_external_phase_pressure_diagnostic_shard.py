from scripts.cluster.run_tsc_external_phase_pressure_diagnostic_shard import (
    shard_identities,
)


def test_phase_pressure_shards_are_complete_and_disjoint():
    expected = {
        (scenario, seed)
        for scenario in (
            "la_1x4",
            "jinan_3x4_real",
            "jinan_3x4_real_2000",
            "jinan_3x4_real_2500",
        )
        for seed in (11, 13)
    }
    shards = [
        set(shard_identities((11, 13), shard_index=index, shard_count=3))
        for index in range(3)
    ]
    assert set().union(*shards) == expected
    assert sum(len(values) for values in shards) == len(expected)
