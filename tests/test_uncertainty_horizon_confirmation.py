from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_uncertainty_horizon_confirmation import (
    all_rollout_identities,
    confirmation_candidates,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_confirmation import (
    ARTIFACT_KEY,
    METHOD_KEY,
)
from scripts.cluster.run_tsc_external_uncertainty_horizon_confirmation_shard import (
    shard_identities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v84_external_v9_"
    "uncertainty_horizon_confirmation.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v84_confirmation_freezes_one_h120_candidate() -> None:
    protocol = _protocol()
    candidates = confirmation_candidates(protocol)
    identities = all_rollout_identities(protocol)

    assert len(candidates) == 1
    assert candidates[0].key == METHOD_KEY
    assert candidates[0].artifact_key == ARTIFACT_KEY
    assert candidates[0].prediction_horizon_sec == 120
    assert candidates[0].execution_trust_region.global_cooldown_intervals == 11
    assert candidates[0].execution_trust_region.min_priority == 0.0
    assert len(identities) == len(set(identities)) == 64
    assert protocol["confirmation"]["seed_count"] == 32
    assert protocol["confirmation"]["matrix_size"] == 64
    assert protocol["scientific_status"]["candidate_count"] == 1
    assert protocol["scientific_status"]["threshold_or_architecture_tuning_in_v84"] is False


def test_v84_six_shards_partition_every_identity_once() -> None:
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [11, 11, 11, 11, 10, 10]
    assert len(flattened) == len(set(flattened)) == 64
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v84_confirmatory_seeds_are_untouched_partition() -> None:
    protocol = _protocol()
    partition = json.loads(Path(protocol["partition"]["path"]).read_text())
    confirmatory = set(protocol["confirmation"]["seeds"])
    prospective = set(partition["prospective"]["seeds"])
    v83 = json.loads(Path(protocol["confirmation_parent"]["v83_protocol"]["path"]).read_text())
    development = set(v83["development"]["seeds"])

    assert len(confirmatory) == 32
    assert confirmatory == set(partition["confirmatory"]["seeds"])
    assert not confirmatory.intersection(development)
    assert not confirmatory.intersection(prospective)
    assert protocol["prospective_partition"]["remains_sealed"] is True
