from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_uncertainty_horizon_closed_loop_development import (
    _intervention_alignment,
    all_rollout_identities,
    development_candidates,
)
from scripts.cluster.audit_tsc_external_uncertainty_horizon_closed_loop import (
    select_candidate,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_closed_loop_development import (
    CANDIDATE_SPECS,
    DIAGNOSTIC_METHOD_KEYS,
    METHOD_KEYS,
    PRIMARY_METHOD_KEYS,
)
from scripts.cluster.run_tsc_external_uncertainty_horizon_closed_loop_shard import (
    shard_identities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v83_external_v9_"
    "uncertainty_horizon_closed_loop_development.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v83_matrix_and_candidate_horizons_are_exact() -> None:
    protocol = _protocol()
    candidates = development_candidates(protocol)
    identities = all_rollout_identities(protocol)

    assert len(identities) == len(set(identities)) == 110
    assert len({identity[2] for identity in identities}) == 22
    assert tuple(candidate.key for candidate in candidates) == METHOD_KEYS
    assert PRIMARY_METHOD_KEYS == METHOD_KEYS
    assert DIAGNOSTIC_METHOD_KEYS == ()
    assert all(candidate.selection_eligible_closed_loop for candidate in candidates)
    assert tuple(candidate.artifact_key for candidate in candidates) == tuple(
        row[1] for row in CANDIDATE_SPECS
    )
    assert tuple(candidate.prediction_horizon_sec for candidate in candidates) == (
        60,
        60,
        120,
        120,
    )
    assert tuple(
        candidate.execution_trust_region.global_cooldown_intervals
        for candidate in candidates
    ) == (5, 5, 11, 11)
    assert all(candidate.cooldown_intervals == 0 for candidate in candidates)
    assert all(
        candidate.execution_trust_region.min_priority == 0.0
        for candidate in candidates
    )
    assert protocol["environment"]["prediction_horizons_sec"] == [60, 120]
    assert protocol["development"]["matrix_size"] == 110
    assert protocol["artifact"]["risk_multiplier"] == 0.25
    assert protocol["artifact"]["minimum_context_trust"] == 0.10


def test_v83_six_shards_partition_every_identity_once() -> None:
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [19, 19, 18, 18, 18, 18]
    assert len(flattened) == len(set(flattened)) == 110
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v83_sealed_seed_sets_remain_disjoint() -> None:
    development = _protocol()["development"]
    active = set(development["seeds"])
    confirmatory = set(development["sealed_confirmatory_seeds"])
    prospective = set(development["sealed_prospective_seeds"])

    assert (len(active), len(confirmatory), len(prospective)) == (22, 32, 8)
    assert not active.intersection(confirmatory)
    assert not active.intersection(prospective)
    assert not confirmatory.intersection(prospective)


def _summary(
    *, policy: str, delta: float, upper: float, overrides: int, horizon: int
) -> dict[str, object]:
    return {
        "policy": policy,
        "selection_eligible_closed_loop": True,
        "feasible": True,
        "mean_relative_delta": delta,
        "bootstrap": {"ci95": [delta - 0.01, upper]},
        "total_executed_overrides": overrides,
        "prediction_horizon_sec": horizon,
    }


def test_v83_selector_uses_frozen_order() -> None:
    worse_mean = _summary(
        policy="uh_compact_h60",
        delta=-0.01,
        upper=-0.001,
        overrides=1,
        horizon=60,
    )
    better_mean = _summary(
        policy="uh_state_action_h120",
        delta=-0.02,
        upper=-0.001,
        overrides=100,
        horizon=120,
    )
    assert select_candidate((worse_mean, better_mean))["policy"] == (
        "uh_state_action_h120"
    )

    shorter = _summary(
        policy="uh_compact_h60",
        delta=-0.02,
        upper=-0.001,
        overrides=10,
        horizon=60,
    )
    longer = _summary(
        policy="uh_compact_h120",
        delta=-0.02,
        upper=-0.001,
        overrides=10,
        horizon=120,
    )
    assert select_candidate((longer, shorter))["policy"] == "uh_compact_h60"


def test_v83_intervention_alignment_uses_executed_requests() -> None:
    metrics = {
        "accepted_intervention_trace": [
            {"time_sec": 10.0, "request_accepted": False},
            {"time_sec": 20.0, "request_accepted": True},
            {"time_sec": 80.0, "request_accepted": True},
            {"time_sec": 200.0, "request_accepted": True},
        ]
    }
    h60 = _intervention_alignment(metrics, expected_horizon_sec=60)
    h120 = _intervention_alignment(metrics, expected_horizon_sec=120)

    assert h60["accepted_request_count"] == 3
    assert h60["minimum_observed_gap_sec"] == 60.0
    assert h60["passed"] is True
    assert h120["passed"] is False
