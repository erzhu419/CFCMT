from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (
    all_rollout_identities,
    policy_candidates,
    scenario_city,
)
from scripts.cluster.run_tsc_external_policy_horizon_search_shard import (
    shard_identities,
)
from scripts.cluster.audit_tsc_external_policy_horizon_adaptation import (
    summarize_candidate,
)


PROTOCOL = Path(
    "cf_h2o/config/traffic_signal_tsc_v34_external_policy_horizon_adaptation.json"
)


def _protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_policy_horizon_grid_is_frozen_and_unique():
    protocol = _protocol()
    for city in protocol["city_scenarios"]:
        candidates = policy_candidates(protocol, city)
        assert len(candidates) == 28
        assert len({candidate.key for candidate in candidates}) == 28
        assert candidates[0].key == "phase_pressure"
    identities = all_rollout_identities(protocol)
    assert len(identities) == 224
    assert len(set(identities)) == 224


def test_policy_horizon_scenario_city_mapping():
    protocol = _protocol()
    assert scenario_city(protocol, "la_1x4") == "los_angeles"
    assert scenario_city(protocol, "jinan_3x4_real_2500") == "jinan"


def test_policy_horizon_shards_partition_the_matrix():
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    assert sorted(len(shard) for shard in shards) == [37, 37, 37, 37, 38, 38]
    flattened = [identity for shard in shards for identity in shard]
    assert len(flattened) == 224
    assert len(set(flattened)) == 224


def test_policy_horizon_selector_uses_seed_level_city_means():
    candidate = {
        "key": "candidate",
        "family": "cfcmt_pressure_regularized_direct",
        "blend_weight": 0.5,
        "risk_multiplier": 0.0,
        "min_context_trust": 0.25,
        "cooldown_intervals": 3,
    }
    paired = [
        {
            "city": "city",
            "scenario": "scenario",
            "seed": seed,
            "candidate": "candidate",
            "method_mean_waiting_time": method,
            "phase_mean_waiting_time": 100.0,
            "executed_overrides": 2,
            "method_safety": {"ending_teleports": 0, "collision_incidents": 0},
            "phase_safety": {"ending_teleports": 0, "collision_incidents": 0},
        }
        for seed, method in ((1, 98.0), (2, 99.0))
    ]
    summary = summarize_candidate(
        city="city",
        candidate=candidate,
        paired_rows=paired,
        scenarios=("scenario",),
        seeds=(1, 2),
        selection_rule={
            "minimum_city_mean_relative_improvement": 0.005,
            "maximum_seed_level_city_mean_relative_regression": 0.01,
            "minimum_executed_interventions_per_scenario_seed": 1,
        },
    )
    assert summary["feasible"] is True
    assert summary["city_mean_relative_delta"] == pytest.approx(-0.015)
