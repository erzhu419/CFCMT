from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_local_mechanism_surrogate_oracle as v134
from scripts.cluster.launch_tsc_local_mechanism_surrogate_oracle import (
    SIGNATURE,
    build_spec,
)


def test_surrogate_proposals_prefer_proxy_improving_safe_action() -> None:
    policy_rows = np.arange(12, dtype=int).reshape(4, 3)
    scores = np.tile(np.asarray([0.0, -0.5, -1.0]), 4)
    actual = np.tile(np.asarray([0.0, -0.1, -0.2]), 4)
    eligible = np.tile(np.asarray([True, True, False]), 4)
    references = np.tile(np.asarray([True, False, False]), 4)
    proposals = v134._surrogate_proposals(
        scores,
        actual,
        policy_rows,
        eligible,
        references,
    )
    assert np.array_equal(proposals["selected_rows"], policy_rows[:, 1])
    assert np.all(proposals["proposed"])
    assert np.allclose(proposals["actual_selected_delta"], -0.1)


def test_familywise_selection_accepts_consistent_surrogate(monkeypatch) -> None:
    monkeypatch.setattr(v134, "MINIMUM_PROFILE_INTERVENTIONS", 1)
    folds = []
    for seed in range(22):
        folds.append(
            {
                "heldout_seed": seed,
                "arms": {
                    "good": {
                        "heldout_value": -0.1,
                        "heldout_intervention_count": 2,
                        "heldout_harmful_intervention_fraction": 0.0,
                        "deployment_enabled": True,
                        "surrogate": "queue",
                        "retention_fraction": 0.1,
                    },
                    "bad": {
                        "heldout_value": 0.1,
                        "heldout_intervention_count": 2,
                        "heldout_harmful_intervention_fraction": 1.0,
                        "deployment_enabled": True,
                        "surrogate": "red",
                        "retention_fraction": 0.1,
                    },
                },
            }
        )
    gate, summaries = v134._familywise_selection(folds)
    assert gate["passed"] is True
    assert gate["selected_profile"] == "good"
    assert summaries["good"]["eligible"] is True
    assert summaries["bad"]["eligible"] is False


def test_v134_launcher_uses_no_source_artifact() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="a" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
    )
    assert "local_mechanism_surrogate_oracle" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert "node004" not in spec["allowed_nodes"]
