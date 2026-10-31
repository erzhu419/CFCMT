from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_latent_closed_loop_development import (
    all_rollout_identities,
    development_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v72_external_v9_"
    "waiting_aligned_latent_closed_loop_development.json"
)


def test_frozen_latent_closed_loop_matrix_is_exact() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    candidates = development_candidates(protocol)
    identities = all_rollout_identities(protocol)

    assert len(identities) == 88
    assert len(set(identities)) == 88
    assert {identity[0] for identity in identities} == {"jinan"}
    assert {identity[1] for identity in identities} == {"jinan_3x4_real"}
    assert len({identity[2] for identity in identities}) == 22
    assert tuple(candidate.cooldown_intervals for candidate in candidates) == (
        0,
        2,
        5,
    )
    assert all(
        candidate.runtime_policy == HIERARCHICAL_RUNTIME_POLICY
        and candidate.execution_trust_region.max_simultaneous_overrides == 1
        and candidate.execution_trust_region.global_cooldown_intervals == 0
        for candidate in candidates
    )


def test_frozen_latent_closed_loop_sealed_seeds_are_disjoint() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    development = protocol["development"]
    active = set(development["seeds"])
    confirmatory = set(development["sealed_confirmatory_seeds"])
    prospective = set(development["sealed_prospective_seeds"])

    assert len(active) == 22
    assert len(confirmatory) == 32
    assert len(prospective) == 8
    assert not active.intersection(confirmatory)
    assert not active.intersection(prospective)
    assert not confirmatory.intersection(prospective)
