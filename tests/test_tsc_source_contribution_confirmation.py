from __future__ import annotations

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import METHOD_POLICY
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (
    POLICY as TARGET_ONLY_POLICY,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_confirmation import (
    all_rollout_identities,
)
from scripts.cluster.audit_tsc_external_source_contribution_confirmation import (
    _exact_one_sided_sign_p,
)
from scripts.cluster.freeze_tsc_external_source_contribution_confirmation import (
    SEED_COUNT,
    _seed_values,
    generate_seeds,
)
from scripts.cluster.run_tsc_external_source_contribution_confirmation_shard import (
    shard_identities,
)


def _protocol() -> dict[str, object]:
    seeds = list(range(1_000_000, 1_000_000 + SEED_COUNT))
    return {
        "confirmation": {
            "city_scenarios": {
                "los_angeles": ["la_1x4"],
                "jinan": [
                    "jinan_3x4_real",
                    "jinan_3x4_real_2000",
                    "jinan_3x4_real_2500",
                ],
            },
            "seeds": seeds,
            "policies": [METHOD_POLICY, TARGET_ONLY_POLICY],
            "matrix_size": 512,
        }
    }


def test_v91_seed_generator_is_deterministic_and_disjoint() -> None:
    excluded = {1_000_001, 1_000_002, 8081, 9091}
    first = generate_seeds(excluded=excluded)
    second = generate_seeds(excluded=excluded)

    assert first == second
    assert len(first) == len(set(first)) == SEED_COUNT
    assert not set(first).intersection(excluded)


def test_v91_seed_exclusion_reads_only_seed_fields() -> None:
    payload = {
        "adaptation_seeds": [5057, 6067],
        "nested": {"evaluation_seed": 8171, "horizon_sec": 3600},
        "bootstrap": {"seed": 20260901, "replicates": 20000},
        "non_seed_values": [10, 20],
        "counterfactual_collection_shards_per_scenario_seed": 16,
        "seed_7079_metrics_must_not_change_any_choice": True,
    }

    assert _seed_values(payload) == {5057, 6067, 8171, 20260901}


def test_v91_six_shards_cover_all_512_identities_once() -> None:
    protocol = _protocol()
    identities = all_rollout_identities(protocol)
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert len(identities) == len(set(identities)) == 512
    assert [len(shard) for shard in shards] == [86, 86, 85, 85, 85, 85]
    assert len(flattened) == len(set(flattened)) == 512
    assert set(flattened) == set(identities)


def test_v91_sign_test_uses_improvement_tail() -> None:
    assert _exact_one_sided_sign_p(improved=64, total=64) == 2**-64
    assert _exact_one_sided_sign_p(improved=32, total=64) > 0.5
