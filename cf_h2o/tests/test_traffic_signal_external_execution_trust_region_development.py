from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_execution_trust_region_development import (
    all_rollout_identities,
    trust_region_candidates,
)
from scripts.cluster.run_tsc_external_execution_trust_region_development_shard import (
    shard_identities,
)


PROTOCOL = Path(
    "cf_h2o/config/traffic_signal_tsc_v37_external_execution_trust_region_development.json"
)
ADAPTATION = Path(
    "cf_h2o/config/traffic_signal_tsc_v34_external_policy_horizon_adaptation.json"
)


def test_execution_trust_region_matrix_is_frozen_and_blinded():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    adaptation = json.loads(ADAPTATION.read_text(encoding="utf-8"))
    candidates = trust_region_candidates(protocol, adaptation)
    identities = all_rollout_identities(protocol, adaptation)
    assert len(candidates) == 24
    assert len({candidate.key for candidate in candidates}) == 24
    assert len(identities) == len(set(identities)) == 240
    development = {identity[2] for identity in identities}
    validation = set(protocol["new_untouched_validation"]["seeds"])
    prospective = set(protocol["prospective_confirmation_reservation"]["seeds"])
    assert development.isdisjoint(validation)
    assert development.isdisjoint(prospective)
    assert validation.isdisjoint(prospective)
    assert {identity[0] for identity in identities} == {"los_angeles"}


def test_execution_trust_region_grid_contains_sparse_and_strong_guards():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    adaptation = json.loads(ADAPTATION.read_text(encoding="utf-8"))
    candidates = trust_region_candidates(protocol, adaptation)
    assert all(candidate.coordination_mode == "sparse" for candidate in candidates)
    assert any(not candidate.execution_trust_region.enabled for candidate in candidates)
    assert any(
        candidate.execution_trust_region.max_simultaneous_overrides == 1
        and candidate.execution_trust_region.min_priority == 0.5
        and candidate.execution_trust_region.min_total_queue == 10.0
        and candidate.execution_trust_region.max_mean_speed == 7.0
        for candidate in candidates
    )


def test_execution_trust_region_matrix_splits_evenly_across_six_nodes():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    adaptation = json.loads(ADAPTATION.read_text(encoding="utf-8"))
    shards = [
        shard_identities(
            protocol,
            adaptation,
            shard_index=index,
            shard_count=6,
        )
        for index in range(6)
    ]
    assert [len(shard) for shard in shards] == [40] * 6
    flattened = [identity for shard in shards for identity in shard]
    assert len(flattened) == len(set(flattened)) == 240
