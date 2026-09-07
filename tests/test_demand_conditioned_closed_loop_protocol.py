from __future__ import annotations

from copy import deepcopy

import pytest

from cf_h2o.eval.traffic_signal_external_demand_conditioned_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    all_method_rollout_identities,
    development_candidate,
)
from scripts.cluster.run_tsc_external_demand_conditioned_closed_loop_development_shard import (
    shard_identities,
)


def _protocol() -> dict:
    return {
        "development": {
            "candidate": {
                "key": "cfcmt_demand_conditioned_veto_cd450",
                "runtime_policy": HIERARCHICAL_RUNTIME_POLICY,
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
            "seeds": [52282, 41242, 63792, 80722, 75638, 61337, 73385, 81189],
        }
    }


def test_v71_identity_matrix_and_six_shards_are_exact() -> None:
    protocol = _protocol()
    identities = all_method_rollout_identities(protocol)
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]

    assert len(identities) == len(set(identities)) == 32
    assert sorted(len(rows) for rows in shards) == [5, 5, 5, 5, 6, 6]
    assert {row for rows in shards for row in rows} == set(identities)
    assert sum(len(rows) for rows in shards) == len(identities)


def test_v71_candidate_contract_rejects_post_freeze_cooldown_change() -> None:
    protocol = _protocol()
    candidate = development_candidate(protocol)
    assert candidate.cooldown_intervals == 44
    assert candidate.execution_trust_region.global_cooldown_intervals == 44

    changed = deepcopy(protocol)
    changed["development"]["candidate"]["cooldown_intervals"] = 29
    with pytest.raises(ValueError, match="candidate changed"):
        development_candidate(changed)
