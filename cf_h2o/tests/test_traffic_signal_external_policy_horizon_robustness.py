from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_external_policy_horizon_robustness import (
    METHOD,
    all_rollout_identities,
    scenario_city,
)
from scripts.cluster.run_tsc_external_policy_horizon_robustness_shard import (
    shard_identities,
)
from scripts.cluster.audit_tsc_external_policy_horizon_robustness import (
    bootstrap_mean_interval,
)


PROTOCOL = Path(
    "cf_h2o/config/traffic_signal_tsc_v35_external_policy_horizon_robustness.json"
)


def _protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_robustness_matrix_is_frozen_and_unique():
    protocol = _protocol()
    identities = all_rollout_identities(protocol)
    assert len(identities) == 64
    assert len(set(identities)) == 64
    assert {identity[3] for identity in identities} == {METHOD, "phase_pressure"}


def test_robustness_scenario_city_mapping():
    protocol = _protocol()
    assert scenario_city(protocol, "la_1x4") == "los_angeles"
    assert scenario_city(protocol, "jinan_3x4_real") == "jinan"


def test_robustness_shards_partition_matrix():
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    assert sorted(len(shard) for shard in shards) == [10, 10, 11, 11, 11, 11]
    identities = [identity for shard in shards for identity in shard]
    assert len(identities) == 64
    assert len(set(identities)) == 64


def test_robustness_bootstrap_is_deterministic():
    first = bootstrap_mean_interval(
        [-0.02, -0.01, 0.0, 0.01], replicates=1000, seed=7, confidence=0.95
    )
    second = bootstrap_mean_interval(
        [-0.02, -0.01, 0.0, 0.01], replicates=1000, seed=7, confidence=0.95
    )
    assert first == second
    assert first["lower"] == pytest.approx(-0.015)
    assert first["upper"] == pytest.approx(0.005)
