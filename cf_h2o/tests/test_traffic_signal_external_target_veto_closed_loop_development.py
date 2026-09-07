import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_target_veto_closed_loop_development import (
    all_method_rollout_identities,
    development_candidates,
)


def test_target_veto_development_matrix_excludes_reused_phase_rollouts() -> None:
    parent = json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v47_external_v9_long_horizon_target_veto.json"
        ).read_text(encoding="utf-8")
    )
    protocol = {
        "development": {
            **parent["closed_loop_development"],
            "method_only_matrix_size": 96,
        }
    }
    candidates = development_candidates(protocol)
    identities = all_method_rollout_identities(protocol)

    assert len(candidates) == 3
    assert len(identities) == 96
    assert all(policy != "phase_pressure" for _, _, _, policy in identities)
    assert {candidate.cooldown_intervals for candidate in candidates} == {11, 29, 44}
    assert all(
        candidate.execution_trust_region.max_simultaneous_overrides == 1
        for candidate in candidates
    )
