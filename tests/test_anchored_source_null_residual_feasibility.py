from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility import (
    _center_on_reference,
    _outer_partition,
    _permute_group_channel,
    _policy_proposals,
    _residual_dataset,
    _select_calibration_profile,
    _source_null_gate,
)
from cf_h2o.traffic_signal.action_ranker import (
    CausalReferenceResidualRegressor,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_anchored_source_null_residual_feasibility import (
    build_spec,
)


def _toy_dataset(group_count: int = 8) -> MechanismDataset:
    features = []
    targets = []
    groups = []
    references = []
    for group_index in range(group_count):
        group = f"toy:seed{group_index}:group0"
        for action in range(3):
            features.append([float(group_index + 1), float(action)])
            targets.append(0.0 if action == 0 else 0.1 - 0.1 * action)
            groups.append(group)
            references.append(action == 0)
    return MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray(features, dtype=float),
        context_names=("context",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(["jinan"] * len(features)),
        metadata={
            "action_group_ids": np.asarray(groups),
            "is_reference": np.asarray(references),
        },
    )


def test_reference_residual_model_uses_causal_reference_differences() -> None:
    dataset = _toy_dataset()
    model = CausalReferenceResidualRegressor(
        state_feature_names=("total_q",),
        action_feature_names=("green_q",),
    )
    fit = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]
    reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    assert fit["residual_protocol"] == "causal_reference_contrast_residual_v1"
    assert fit["state_feature_names"] == ["total_q"]
    assert fit["action_feature_names"] == ["green_q"]
    assert fit["candidate_row_count"] == 16
    np.testing.assert_allclose(prediction["mean"][reference], 0.0)
    np.testing.assert_allclose(prediction["uncertainty"][reference], 0.0)


def test_center_and_residual_dataset_preserve_actual_target() -> None:
    dataset = _toy_dataset(group_count=2)
    rows = np.arange(dataset.size).reshape(2, 3)
    references = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    raw = np.asarray([3.0, 2.0, 1.0, -1.0, -0.5, -2.0])
    prior = _center_on_reference(raw, rows, references)
    np.testing.assert_allclose(prior, [0.0, -1.0, -2.0, 0.0, 0.5, -1.0])
    actual = np.asarray(dataset.targets["interval_cost"], dtype=float)
    residual = _residual_dataset(dataset, actual, prior)
    np.testing.assert_allclose(
        residual.targets["interval_cost"] + prior,
        actual,
    )


def test_outer_partition_keeps_eleven_fit_ten_calibration_one_test() -> None:
    seeds = tuple(range(22))
    training, calibration = _outer_partition(seeds, 7)
    assert len(training) == 11
    assert len(calibration) == 10
    assert not set(training) & set(calibration)
    assert set(training) | set(calibration) | {7} == set(seeds)


def test_placebo_preserves_whole_action_vectors_within_seed() -> None:
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


def test_source_null_gate_selects_only_replicated_source_gain() -> None:
    seeds = tuple(range(10))
    target = {
        "enabled": True,
        "selected_calibration_values": {str(seed): 0.0 for seed in seeds},
    }
    source = {
        "enabled": True,
        "selected_calibration_values": {str(seed): -0.002 for seed in seeds},
    }
    selected = _source_null_gate(source, target, seeds)
    assert selected["selected_source"] is True
    unstable = {
        "enabled": True,
        "selected_calibration_values": {
            str(seed): (-0.002 if seed < 5 else 0.002) for seed in seeds
        },
    }
    rejected = _source_null_gate(unstable, target, seeds)
    assert rejected["selected_source"] is False


def test_tied_candidate_is_not_proposed_and_disabled_profile_is_strict_json() -> None:
    predicted = np.zeros(30, dtype=float)
    actual = np.zeros(30, dtype=float)
    rows = np.arange(30).reshape(10, 3)
    references = np.zeros(30, dtype=bool)
    references[rows[:, 1]] = True
    proposals = _policy_proposals(
        predicted,
        actual,
        rows,
        np.ones(30, dtype=bool),
        references,
    )
    assert not np.any(proposals["proposed"])
    profile = _select_calibration_profile(proposals, np.arange(10), tuple(range(10)))
    assert profile["enabled"] is False
    assert profile["predicted_advantage_threshold"] is None
    json.dumps(profile, allow_nan=False)


def test_v128_launcher_freezes_source_null_completion_contract() -> None:
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
    assert "anchored_source_null_residual_feasibility" in spec["cmd"]
    assert spec["cmd"].endswith("printf 'TASK_DONE\n'")
    assert spec["signature"].endswith("reference-residual-b500-v1")
    assert spec["cpu"] == 20
    assert spec["allowed_nodes"] == [
        "node001",
        "node002",
        "node003",
        "node005",
        "node006",
    ]
