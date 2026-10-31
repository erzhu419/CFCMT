from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_causal_pairwise_source_prior_feasibility import (
    SOURCE_PRIOR_FEATURE,
    TARGET_PRIOR_FEATURE,
    _outer_partition,
    _outer_fold,
    _permute_group_channel,
    _policy_proposals,
    _ranker_dataset,
    _select_calibration_profile,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_causal_pairwise_source_prior_feasibility import (
    build_spec,
)


def test_outer_partition_keeps_fit_calibration_and_test_disjoint() -> None:
    seeds = tuple(range(22))
    training, calibration = _outer_partition(seeds, 7)
    assert len(training) == 16
    assert len(calibration) == 5
    assert not set(training) & set(calibration)
    assert 7 not in training
    assert 7 not in calibration
    assert set(training) | set(calibration) | {7} == set(seeds)


def test_group_channel_placebo_preserves_whole_action_vectors() -> None:
    channel = np.arange(24, dtype=float)
    rows = np.arange(24).reshape(6, 4)
    seeds = np.asarray([10, 10, 10, 20, 20, 20])
    placebo = _permute_group_channel(channel, rows, seeds)
    assert not np.array_equal(placebo, channel)
    for seed in (10, 20):
        groups = np.flatnonzero(seeds == seed)
        expected = sorted(tuple(row) for row in channel[rows[groups]])
        observed = sorted(tuple(row) for row in placebo[rows[groups]])
        assert observed == expected


def test_ranker_dataset_keeps_only_pairwise_parents_and_matched_channels() -> None:
    contrast = MechanismDataset(
        feature_names=("total_q", "green_q", "unused_dense_feature"),
        features=np.asarray([[1.0, 0.0, 9.0], [1.0, 2.0, 8.0]]),
        context_names=("context",),
        context=np.zeros((2, 1)),
        priors={},
        targets={"prefix_mean_cost_450s": np.asarray([0.0, -1.0])},
        domains=np.asarray(["jinan", "jinan"]),
        metadata={
            "action_group_ids": np.asarray(["g", "g"]),
            "is_reference": np.asarray([True, False]),
        },
    )
    dataset = _ranker_dataset(
        contrast,
        np.asarray([0.0, -0.5]),
        np.asarray([0.0, -0.2]),
        np.asarray([0.0, -0.1]),
    )
    assert dataset.feature_names == (
        "total_q",
        "green_q",
        "target_b500_score",
        "atlanta_b500_incremental_score",
    )
    assert dataset.features.shape == (2, 4)
    np.testing.assert_allclose(dataset.targets["interval_cost"], [0.0, -0.5])


def test_policy_proposal_never_selects_pressure_degrading_action() -> None:
    predicted = np.asarray([0.0, -2.0, -1.0, 0.0, -3.0, -2.0])
    actual = np.asarray([0.0, -1.0, -0.2, 0.0, -2.0, -0.1])
    rows = np.asarray([[0, 1, 2], [3, 4, 5]])
    eligible = np.asarray([True, False, True, True, False, True])
    references = np.asarray([True, False, False, True, False, False])
    proposals = _policy_proposals(
        predicted, actual, rows, eligible, references
    )
    np.testing.assert_array_equal(proposals["selected_rows"], [2, 5])
    np.testing.assert_allclose(proposals["actual_selected_delta"], [-0.2, -0.1])


def test_calibration_profile_requires_replicated_negative_seed_value() -> None:
    seeds = np.repeat(np.arange(5), 20)
    advantage = np.tile(np.linspace(0.01, 0.20, 20), 5)
    proposals = {
        "proposed": np.ones(100, dtype=bool),
        "predicted_advantage": advantage,
        "actual_selected_delta": np.full(100, -0.2),
    }
    profile = _select_calibration_profile(proposals, seeds, tuple(range(5)))
    assert profile["enabled"] is True
    assert profile["selected_calibration_summary"]["intervention_count"] >= 20
    assert profile["selected_calibration_summary"]["improving_seed_fraction"] == 1.0


def test_calibration_profile_falls_back_when_interventions_are_harmful() -> None:
    seeds = np.repeat(np.arange(5), 20)
    proposals = {
        "proposed": np.ones(100, dtype=bool),
        "predicted_advantage": np.tile(np.linspace(0.01, 0.20, 20), 5),
        "actual_selected_delta": np.full(100, 0.2),
    }
    profile = _select_calibration_profile(proposals, seeds, tuple(range(5)))
    assert profile["enabled"] is False
    assert profile["predicted_advantage_threshold"] is None
    json.dumps(profile, allow_nan=False)


def test_v127_launcher_freezes_worker_and_completion_contract() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        prediction_artifact_remote=Path("/evidence/predictions.pkl"),
        prediction_artifact_sha256="a" * 64,
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
        fold_workers=20,
    )
    command = spec["cmd"]
    assert "traffic_signal_causal_pairwise_source_prior_feasibility" in command
    assert "--cache-workers 20" in command
    assert "--fold-workers 20" in command
    assert command.endswith("printf 'TASK_DONE\\n'")
    assert spec["signature"].endswith("v2-json-safe")
    assert spec["cpu"] == 20
    assert spec["ram_mb"] == 24_576
    assert spec["allowed_nodes"] == [
        "node001",
        "node002",
        "node003",
        "node005",
        "node006",
    ]
    assert "require_node" not in spec


def test_outer_fold_runs_pairwise_fit_calibration_and_heldout_policy() -> None:
    seeds = tuple(range(7))
    feature_rows = []
    target_rows = []
    group_ids = []
    references = []
    row_seeds = []
    for seed in seeds:
        for group_index in range(2):
            group = f"toy:seed{seed}:group{group_index}"
            for action in range(2):
                feature_rows.append([float(group_index + 1), float(action), 0.0, 0.0])
                target_rows.append(0.0 if action == 0 else -0.1)
                group_ids.append(group)
                references.append(action == 0)
                row_seeds.append(seed)
    base = MechanismDataset(
        feature_names=(
            "total_q",
            "green_q",
            TARGET_PRIOR_FEATURE,
            SOURCE_PRIOR_FEATURE,
        ),
        features=np.asarray(feature_rows),
        context_names=("context",),
        context=np.zeros((len(feature_rows), 1)),
        priors={},
        targets={"interval_cost": np.asarray(target_rows)},
        domains=np.asarray(["jinan"] * len(feature_rows)),
        metadata={
            "action_group_ids": np.asarray(group_ids),
            "is_reference": np.asarray(references),
        },
    )
    result = _outer_fold(
        0,
        datasets={arm: base for arm in ("target_only", "source_aligned", "source_placebo")},
        all_seeds=seeds,
        row_seeds=np.asarray(row_seeds),
        eligible=np.ones(len(feature_rows), dtype=bool),
    )
    assert result["heldout_seed"] == 0
    assert len(result["training_seeds"]) == 1
    assert len(result["calibration_seeds"]) == 5
    assert set(result["arms"]) == {"target_only", "source_aligned", "source_placebo"}
    json.dumps(result, allow_nan=False)
