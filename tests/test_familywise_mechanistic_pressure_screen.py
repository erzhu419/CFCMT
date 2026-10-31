from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_familywise_mechanistic_pressure_screen as v132
from cf_h2o.eval.traffic_signal_mechanistic_pressure_screen import _rule_proposals
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_familywise_mechanistic_pressure_screen import (
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
        for group_index in range(40):
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
        metadata={"action_group_ids": groups, "is_reference": references},
    )
    return (
        dataset,
        np.asarray(actual, dtype=float),
        np.arange(dataset.size, dtype=int).reshape(-1, 3),
        np.asarray(group_seeds, dtype=int),
    )


def test_max_t_is_deterministic_and_correlation_aware() -> None:
    generator = np.random.default_rng(7)
    base = generator.normal(size=21)
    values = np.vstack((base, base, -base, 0.5 * base))
    first = v132._max_t_critical(values, alpha=0.05, draws=2_000, random_seed=4)
    second = v132._max_t_critical(values, alpha=0.05, draws=2_000, random_seed=4)
    assert first == second
    assert 1.6 < first < 3.0


def test_outer_fold_selects_without_reading_heldout_labels(monkeypatch) -> None:
    dataset, actual, policy_rows, group_seeds = _toy_groups()
    spec = PressurePolicySpec(downstream_queue_weight=1.0)
    monkeypatch.setattr(v132, "RULE_SPECS", (spec,))
    monkeypatch.setattr(v132, "RETENTION_FRACTIONS", (1.0,))
    proposals = _rule_proposals(
        dataset,
        actual,
        policy_rows,
        np.ones(dataset.size, dtype=bool),
        np.asarray(dataset.metadata["is_reference"], dtype=bool),
        spec,
    )
    fold = v132._outer_fold(
        0,
        all_seeds=tuple(range(22)),
        proposals_by_rule={spec.key: proposals},
        group_seeds=group_seeds,
        multiplier_draws=2_000,
    )
    arm = fold["arms"][v132.MODEL_ARM]
    assert 0 not in fold["development_seeds"]
    assert arm["development_profile"]["selected"] is True
    assert arm["heldout_intervention_count"] == 40
    assert np.isclose(arm["heldout_value"], -0.1)

    changed = {key: np.asarray(value).copy() for key, value in proposals.items()}
    changed["actual_selected_delta"][group_seeds == 0] = 0.4
    changed_fold = v132._outer_fold(
        0,
        all_seeds=tuple(range(22)),
        proposals_by_rule={spec.key: changed},
        group_seeds=group_seeds,
        multiplier_draws=2_000,
    )
    changed_arm = changed_fold["arms"][v132.MODEL_ARM]
    assert changed_arm["development_profile"] == arm["development_profile"]
    assert np.isclose(changed_arm["heldout_value"], 0.4)


def test_v132_launcher_uses_no_source_artifact() -> None:
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
    assert "familywise_mechanistic_pressure_screen" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert "node004" not in spec["allowed_nodes"]
