from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_multihorizon_estimand_aligned_closed_loop_development import (
    all_rollout_identities,
    development_candidates,
)
from scripts.cluster.audit_tsc_external_multihorizon_estimand_aligned_closed_loop import (
    select_diagnostic_candidate,
    select_primary_candidate,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (
    PRIMARY_ARTIFACT_KEY,
    SECONDARY_ARTIFACT_KEY,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_estimand_aligned_closed_loop_development import (
    DIAGNOSTIC_CANDIDATE_SPECS,
    DIAGNOSTIC_METHOD_KEYS,
    METHOD_KEYS,
    PRIMARY_CANDIDATE_SPECS,
    PRIMARY_METHOD_KEYS,
)
from scripts.cluster.run_tsc_external_multihorizon_estimand_aligned_closed_loop_shard import (
    shard_identities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v81_external_v9_"
    "multihorizon_estimand_aligned_closed_loop_development.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v81_matrix_and_primary_diagnostic_roles_are_exact() -> None:
    protocol = _protocol()
    candidates = development_candidates(protocol)
    identities = all_rollout_identities(protocol)

    assert len(identities) == 198
    assert len(set(identities)) == 198
    assert len({identity[2] for identity in identities}) == 22
    assert tuple(candidate.key for candidate in candidates) == METHOD_KEYS
    primary_count = len(PRIMARY_METHOD_KEYS)
    assert tuple(candidate.key for candidate in candidates[:primary_count]) == PRIMARY_METHOD_KEYS
    assert tuple(candidate.key for candidate in candidates[primary_count:]) == DIAGNOSTIC_METHOD_KEYS
    assert {candidate.artifact_key for candidate in candidates[:primary_count]} == {
        PRIMARY_ARTIFACT_KEY
    }
    assert {candidate.artifact_key for candidate in candidates[primary_count:]} == {
        SECONDARY_ARTIFACT_KEY
    }
    assert all(candidate.selection_eligible_closed_loop for candidate in candidates[:primary_count])
    assert not any(
        candidate.selection_eligible_closed_loop for candidate in candidates[primary_count:]
    )
    expected_specs = (*PRIMARY_CANDIDATE_SPECS, *DIAGNOSTIC_CANDIDATE_SPECS)
    assert tuple(candidate.cooldown_intervals for candidate in candidates) == (0,) * 8
    assert tuple(
        candidate.execution_trust_region.global_cooldown_intervals
        for candidate in candidates
    ) == tuple(row[1] for row in expected_specs)
    assert tuple(
        candidate.execution_trust_region.min_priority for candidate in candidates
    ) == tuple(row[2] for row in expected_specs)
    assert protocol["environment"]["prediction_horizon_sec"] == 60
    assert protocol["development"]["matrix_size"] == 198


def test_v81_six_shards_partition_every_identity_once() -> None:
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [33, 33, 33, 33, 33, 33]
    assert len(flattened) == 198
    assert len(set(flattened)) == 198
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v81_sealed_seed_sets_remain_disjoint() -> None:
    development = _protocol()["development"]
    active = set(development["seeds"])
    confirmatory = set(development["sealed_confirmatory_seeds"])
    prospective = set(development["sealed_prospective_seeds"])

    assert (len(active), len(confirmatory), len(prospective)) == (22, 32, 8)
    assert not active.intersection(confirmatory)
    assert not active.intersection(prospective)
    assert not confirmatory.intersection(prospective)


def _summary(*, policy: str, eligible: bool, delta: float) -> dict[str, object]:
    return {
        "policy": policy,
        "selection_eligible_closed_loop": eligible,
        "feasible": True,
        "mean_relative_delta": delta,
        "bootstrap": {"ci95": [delta - 0.01, delta + 0.01]},
        "total_executed_overrides": 10,
        "cooldown_intervals": 0,
        "global_cooldown_intervals": 5,
        "min_priority": 0.05,
    }


def test_v81_selector_never_promotes_better_diagnostic_to_primary() -> None:
    primary = _summary(policy="mh_compact_g5_p005", eligible=True, delta=-0.01)
    diagnostic = _summary(
        policy="mh_state_action_g5_p005", eligible=False, delta=-0.10
    )

    assert select_primary_candidate((primary, diagnostic))["policy"] == "mh_compact_g5_p005"
    assert (
        select_diagnostic_candidate((primary, diagnostic))["policy"]
        == "mh_state_action_g5_p005"
    )


def test_v81_diagnostic_only_success_cannot_select_primary() -> None:
    diagnostic = _summary(
        policy="mh_state_action_g5_p005", eligible=False, delta=-0.10
    )

    assert select_primary_candidate((diagnostic,)) is None
    assert select_diagnostic_candidate((diagnostic,)) is not None
