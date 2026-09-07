from __future__ import annotations

from cf_h2o.eval.traffic_signal_external_topology_veto_closed_loop_development import (
    CANDIDATE_KEY,
    PHASE_POLICY,
    all_rollout_identities,
    development_candidate,
)
from scripts.cluster.run_tsc_external_topology_veto_closed_loop_development_shard import (
    shard_identities,
)


def _protocol() -> dict:
    return {
        "development": {
            "candidate": {
                "key": CANDIDATE_KEY,
                "runtime_policy": (
                    "causal_group_normalized_rigid_advantage_contrast_"
                    "hierarchical_guard"
                ),
                "coordination_mode": "direct",
                "cooldown_intervals": 44,
                "max_simultaneous_overrides": 1,
            },
            "city_scenarios": {
                "jinan": [
                    "jinan_3x4_real",
                    "jinan_3x4_real_2000",
                    "jinan_3x4_real_2500",
                ],
                "los_angeles": ["la_1x4"],
            },
            "seeds": [80314, 88625, 27178, 77524, 63775, 50945],
        }
    }


def test_v78_matrix_is_paired_and_evenly_sharded() -> None:
    protocol = _protocol()
    identities = all_rollout_identities(protocol)
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]

    assert len(identities) == len(set(identities)) == 48
    assert {identity[3] for identity in identities} == {PHASE_POLICY, CANDIDATE_KEY}
    assert all(len(shard) == 8 for shard in shards)
    assert set().union(*(set(shard) for shard in shards)) == set(identities)
    assert all(
        set(shards[left]).isdisjoint(shards[right])
        for left in range(6)
        for right in range(left + 1, 6)
    )


def test_v78_execution_candidate_is_fixed_before_rollout() -> None:
    candidate = development_candidate(_protocol())

    assert candidate.key == "cfcmt_topology_veto_cd450"
    assert candidate.cooldown_intervals == 44
    assert candidate.max_simultaneous_overrides == 1
    assert candidate.execution_trust_region.global_cooldown_intervals == 44
