from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_historical_action_identifiability as v133
from scripts.cluster.launch_tsc_historical_action_identifiability import (
    SIGNATURE,
    build_spec,
)


def _toy_layout() -> dict[str, np.ndarray]:
    groups = []
    states = []
    actual = []
    eligible = []
    references = []
    group_seeds = []
    for seed in range(22):
        for interval in range(5):
            group = f"jinan:seed{seed}:{interval}:tls0"
            for state, value, pressure_safe in (
                ("reference", 0.0, True),
                ("better", -0.1, True),
                ("unsafe", -0.2, False),
            ):
                groups.append(group)
                states.append(state)
                actual.append(value)
                eligible.append(pressure_safe)
                references.append(state == "reference")
            group_seeds.append(seed)
    return {
        "groups": np.asarray(groups, dtype=str),
        "candidate_states": np.asarray(states, dtype=str),
        "actual": np.asarray(actual, dtype=float),
        "eligible": np.asarray(eligible, dtype=bool),
        "policy_rows": np.arange(len(groups), dtype=int).reshape(-1, 3),
        "references": np.asarray(references, dtype=bool),
        "group_seeds": np.asarray(group_seeds, dtype=int),
    }


def test_history_key_removes_only_seed() -> None:
    assert (
        v133._history_group_key("jinan:seed80314:7:tls0")
        == "jinan:seed*:7:tls0"
    )


def test_outer_fold_uses_history_and_not_heldout_label() -> None:
    values = _toy_layout()
    fold = v133._outer_fold(0, all_seeds=tuple(range(22)), **values)
    primary = fold["arms"][v133.CONSENSUS_ARM]
    assert 0 not in fold["development_seeds"]
    assert fold["support_qualified_group_count"] == 5
    assert primary["heldout_intervention_count"] == 5
    assert np.isclose(primary["heldout_value"], -0.1)

    changed = {key: value.copy() for key, value in values.items()}
    heldout_rows = np.repeat(values["group_seeds"] == 0, 3)
    changed["actual"][heldout_rows & (changed["candidate_states"] == "better")] = 0.4
    changed_fold = v133._outer_fold(
        0,
        all_seeds=tuple(range(22)),
        **changed,
    )
    changed_primary = changed_fold["arms"][v133.CONSENSUS_ARM]
    assert changed_primary["heldout_intervention_count"] == 5
    assert np.isclose(changed_primary["heldout_value"], 0.4)


def test_v133_launcher_uses_no_source_artifact() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="a" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
        fold_workers=20,
    )
    assert "historical_action_identifiability" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert "node004" not in spec["allowed_nodes"]
