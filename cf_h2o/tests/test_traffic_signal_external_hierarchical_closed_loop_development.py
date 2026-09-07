import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    HIERARCHICAL_RUNTIME_POLICY,
    PHASE_POLICY,
    all_rollout_identities,
    development_candidates,
)


def _protocol() -> dict:
    return json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v46_external_v9_hierarchical_closed_loop_development.json"
        ).read_text(encoding="utf-8")
    )


def test_closed_loop_development_matrix_is_complete_and_unique() -> None:
    protocol = _protocol()
    identities = all_rollout_identities(protocol)

    assert len(identities) == len(set(identities)) == 416
    assert {row[3] for row in identities} == set(protocol["development"]["policies"])
    assert sum(row[3] == PHASE_POLICY for row in identities) == 32


def test_closed_loop_candidates_keep_frozen_hierarchical_runtime() -> None:
    candidates = development_candidates(_protocol())

    assert len(candidates) == 12
    assert all(row.runtime_policy == HIERARCHICAL_RUNTIME_POLICY for row in candidates)
    assert all(row.execution_trust_region.enabled for row in candidates)
