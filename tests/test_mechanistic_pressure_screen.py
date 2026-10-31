from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_mechanistic_pressure_screen as v131
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_mechanistic_pressure_screen import (
    SIGNATURE,
    build_spec,
)


def _toy_groups() -> tuple[MechanismDataset, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    actual = []
    groups = []
    references = []
    group_seeds = []
    for seed in range(22):
        for group_index in range(4):
            group = f"toy:seed{seed}:group{group_index}"
            rows = (
                (10.0, 10.0, 0.0, 0.0, 10.0, 0.0),
                (9.0, 0.0, 0.0, 1.0, 11.0, 1.0),
                (8.0, 0.0, 0.0, 1.0, 10.5, 0.5),
            )
            for action, row in enumerate(rows):
                features.append(row)
                actual.append((0.0, -0.1, 0.2)[action])
                groups.append(group)
                references.append(action == 0)
            group_seeds.append(seed)
    dataset = MechanismDataset(
        feature_names=(
            "green_q",
            "green_down_q",
            "green_down_occ",
            "switch_indicator",
            "service_pressure",
            "delta_service_pressure",
        ),
        features=np.asarray(features, dtype=float),
        context_names=("context",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={},
        targets={"interval_cost": np.asarray(actual, dtype=float)},
        domains=np.asarray(["jinan"] * len(features)),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
        },
    )
    policy_rows = np.arange(dataset.size, dtype=int).reshape(-1, 3)
    return (
        dataset,
        np.asarray(actual, dtype=float),
        policy_rows,
        np.asarray(group_seeds, dtype=int),
    )


def test_rule_proposals_prefer_reference_on_score_tie() -> None:
    dataset, actual, policy_rows, _ = _toy_groups()
    proposals = v131._rule_proposals(
        dataset,
        actual,
        policy_rows,
        np.ones(dataset.size, dtype=bool),
        np.asarray(dataset.metadata["is_reference"], dtype=bool),
        PressurePolicySpec(),
    )
    assert not np.any(proposals["proposed"])
    assert np.array_equal(proposals["selected_rows"], policy_rows[:, 0])


def test_outer_fold_selects_and_holds_out_mechanistic_rule(monkeypatch) -> None:
    dataset, actual, policy_rows, group_seeds = _toy_groups()
    spec = PressurePolicySpec(downstream_queue_weight=1.0)
    monkeypatch.setattr(v131, "RULE_SPECS", (spec,))
    proposals = v131._rule_proposals(
        dataset,
        actual,
        policy_rows,
        np.ones(dataset.size, dtype=bool),
        np.asarray(dataset.metadata["is_reference"], dtype=bool),
        spec,
    )
    fold = v131._outer_fold(
        0,
        all_seeds=tuple(range(22)),
        proposals_by_rule={spec.key: proposals},
        group_seeds=group_seeds,
    )
    arm = fold["arms"][v131.MODEL_ARM]
    assert 0 not in fold["training_seeds"]
    assert 0 not in fold["calibration_seeds"]
    assert arm["training_profile"]["policy_spec"]["key"] == spec.key
    assert arm["calibration_profile"]["enabled"] is True
    assert arm["heldout_intervention_count"] == 4
    assert np.isclose(arm["heldout_value"], -0.1)


def test_v131_launcher_uses_no_source_artifact() -> None:
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
    assert "traffic_signal_mechanistic_pressure_screen" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert "node004" not in spec["allowed_nodes"]
