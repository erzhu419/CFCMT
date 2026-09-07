import json
from pathlib import Path

from scripts.cluster.freeze_tsc_external_v9_hierarchical_closed_loop_protocol import (
    execution_candidates,
)


def test_hierarchical_closed_loop_grid_is_execution_only_and_unique() -> None:
    parent = json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v44_external_v9_hierarchical_guard_confirmation.json"
        ).read_text(encoding="utf-8")
    )

    candidates = execution_candidates(parent)

    assert len(candidates) == 12
    assert len({row["key"] for row in candidates}) == 12
    assert all("blend_weight" not in row for row in candidates)
    assert all(
        row["runtime_policy"].endswith("_hierarchical_guard")
        for row in candidates
    )
    assert all(
        row["cooldown_intervals"]
        == row["execution_trust_region"]["global_cooldown_intervals"]
        for row in candidates
    )
