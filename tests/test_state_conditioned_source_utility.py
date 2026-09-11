import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES,
    DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
    STATE_UTILITY_FEATURES,
    UtilityGateConfig,
    UtilityRecords,
    _fold_and_group_balanced_weights,
    apply_dense_pressure_pairwise_utility_gate,
    apply_utility_gate,
    build_dense_pressure_pairwise_utility_records,
    build_utility_records,
    fit_dense_pressure_pairwise_utility_gate,
    fit_utility_gate,
    pressure_align_candidate_scores,
    pressure_pairwise_candidate_scores,
    source_blind_candidate_scores,
)


def _dataset(states: list[float]) -> MechanismDataset:
    rows = []
    groups = []
    for index, state in enumerate(states):
        values = np.zeros(len(STATE_UTILITY_FEATURES), dtype=float)
        values[0] = state
        rows.extend([values.copy(), values.copy()])
        groups.extend([f"g{index}", f"g{index}"])
    size = len(rows)
    return MechanismDataset(
        feature_names=STATE_UTILITY_FEATURES,
        features=np.asarray(rows, dtype=float),
        context_names=("context",),
        context=np.zeros((size, 1), dtype=float),
        priors={},
        targets={},
        domains=np.full(size, "target", dtype=object),
        metadata={
            "action_group_ids": groups,
            "is_reference": [index % 2 == 1 for index in range(size)],
        },
    )


def test_pressure_alignment_keeps_only_rigid_or_reference_actions() -> None:
    dataset = _dataset([0.0, 1.0])
    rigid = np.asarray([0.0, 2.0, 0.0, 2.0])
    candidates = {
        "source|queue_service": np.asarray([2.0, 0.0, 2.0, 0.0]),
        "source|spillback": np.asarray([0.0, 2.0, 0.0, 2.0]),
    }

    projected, audit = pressure_align_candidate_scores(
        dataset, rigid, candidates
    )

    assert np.array_equal(
        projected["source|queue_service"], candidates["source|queue_service"]
    )
    assert np.array_equal(
        projected["source|spillback"], candidates["source|spillback"]
    )
    assert audit["protocol"] == PRESSURE_ALIGNED_CANDIDATE_PROTOCOL
    assert audit["pressure_aligned_change_group_count"] == 2
    assert audit["off_reference_change_rejected_group_count"] == 0


def test_pressure_alignment_replaces_third_action_with_exact_rigid_scores() -> None:
    dataset = _dataset([0.0])
    dataset = MechanismDataset(
        feature_names=dataset.feature_names,
        features=np.vstack((dataset.features, dataset.features[:1])),
        context_names=dataset.context_names,
        context=np.vstack((dataset.context, dataset.context[:1])),
        priors={},
        targets={},
        domains=np.asarray(["target"] * 3, dtype=object),
        metadata={
            "action_group_ids": ["g0", "g0", "g0"],
            "is_reference": [False, True, False],
        },
    )
    rigid = np.asarray([0.0, 2.0, 1.0])
    third_action = np.asarray([2.0, 1.0, 0.0])

    projected, audit = pressure_align_candidate_scores(
        dataset,
        rigid,
        {"source|queue_service": third_action},
    )

    assert np.array_equal(projected["source|queue_service"], rigid)
    assert audit["changed_rigid_group_count"] == 1
    assert audit["pressure_aligned_change_group_count"] == 0
    assert audit["off_reference_change_rejected_group_count"] == 1


def test_pressure_pairwise_candidate_uses_only_reference_or_rigid() -> None:
    dataset = _dataset([0.0])
    dataset = MechanismDataset(
        feature_names=dataset.feature_names,
        features=np.vstack((dataset.features, dataset.features[:1])),
        context_names=dataset.context_names,
        context=np.vstack((dataset.context, dataset.context[:1])),
        priors={},
        targets={},
        domains=np.asarray(["target"] * 3, dtype=object),
        metadata={
            "action_group_ids": ["g0", "g0", "g0"],
            "is_reference": [False, True, False],
        },
    )
    rigid = np.asarray([0.0, 2.0, 1.0])
    candidates = {
        "source|queue_service": np.asarray([2.0, -1.0, -3.0]),
        "source|spillback": np.asarray([0.0, 2.0, -2.0]),
    }

    projected, audit = pressure_pairwise_candidate_scores(
        dataset, rigid, candidates
    )

    assert int(np.argmin(projected["source|queue_service"])) == 1
    assert np.array_equal(projected["source|spillback"], rigid)
    assert audit["protocol"] == PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL
    assert audit["proposed_phase_group_count"] == 1


def test_state_gate_selects_candidate_only_in_beneficial_state() -> None:
    states = [float(index % 2) for index in range(32)]
    dataset = _dataset(states)
    rigid = np.tile([0.0, 1.0], len(states))
    candidate = np.tile([1.0, 0.0], len(states))
    actual = np.concatenate(
        [np.asarray([0.0, -1.0 if state else 1.0]) for state in states]
    )
    records = build_utility_records(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        actual=actual,
        fold_id=0,
    )
    records = type(records)(
        features=records.features,
        cost_delta=records.cost_delta,
        group_ids=records.group_ids,
        candidate_keys=records.candidate_keys,
        fold_ids=np.asarray([index % 4 for index in range(len(states))]),
    )
    gate = fit_utility_gate(
        records,
        config=UtilityGateConfig(
            ridge_alpha=0.01,
            error_quantile=0.5,
            minimum_records=8,
            minimum_predicted_gain=0.0,
        ),
    )

    test = _dataset([0.0, 1.0])
    score, audit = apply_utility_gate(
        test,
        np.tile([0.0, 1.0], 2),
        {"source|queue_service": np.tile([1.0, 0.0], 2)},
        gate,
    )

    assert np.array_equal(score[:2], np.asarray([0.0, 1.0]))
    assert np.array_equal(score[2:], np.asarray([1.0, 0.0]))
    assert audit["selected_group_count"] == 1


def test_insufficient_records_preserve_rigid_score_exactly() -> None:
    dataset = _dataset([0.0, 1.0])
    rigid = np.tile([0.0, 1.0], 2)
    candidate = np.tile([1.0, 0.0], 2)
    actual = np.asarray([0.0, 1.0, 0.0, -1.0])
    records = build_utility_records(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        actual=actual,
        fold_id=0,
    )
    gate = fit_utility_gate(records)

    score, audit = apply_utility_gate(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        gate,
    )

    assert gate.model is None
    assert np.array_equal(score, rigid)
    assert audit["selected_group_count"] == 0


def test_insufficient_fold_training_records_disable_gate() -> None:
    states = [float(index % 2) for index in range(12)]
    dataset = _dataset(states)
    rigid = np.tile([0.0, 1.0], len(states))
    candidate = np.tile([1.0, 0.0], len(states))
    actual = np.tile([0.0, -1.0], len(states))
    records = build_utility_records(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        actual=actual,
        fold_id=0,
    )
    records = type(records)(
        features=records.features,
        cost_delta=records.cost_delta,
        group_ids=records.group_ids,
        candidate_keys=records.candidate_keys,
        fold_ids=np.asarray([0] * 10 + [1] + [2]),
    )
    gate = fit_utility_gate(
        records,
        config=UtilityGateConfig(minimum_records=12),
    )

    score, audit = apply_utility_gate(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        gate,
    )

    assert gate.model is None
    assert gate.diagnostics["reason"] == "insufficient_fold_training_records"
    assert gate.diagnostics["fold_training_record_counts"] == {0: 2, 1: 11, 2: 11}
    assert np.array_equal(score, rigid)
    assert audit["selected_group_count"] == 0


def test_state_features_must_not_change_across_candidate_actions() -> None:
    dataset = _dataset([0.0])
    dataset.features[1, 0] = 1.0

    try:
        build_utility_records(
            dataset,
            np.asarray([0.0, 1.0]),
            {"source|queue_service": np.asarray([1.0, 0.0])},
            actual=np.asarray([0.0, -1.0]),
            fold_id=0,
        )
    except ValueError as exc:
        assert "changes across candidate actions" in str(exc)
    else:
        raise AssertionError("action-dependent state feature was accepted")


def test_dense_pairwise_records_do_not_require_source_action_disagreement() -> None:
    dataset = _dataset([0.0, 1.0])
    rigid = np.tile([0.0, 1.0], 2)
    candidate = np.asarray([0.0, 2.0, 1.0, 0.0])
    actual = np.asarray([0.0, 1.0, 0.0, -1.0])

    records = build_dense_pressure_pairwise_utility_records(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        actual=actual,
        fold_id=0,
    )

    assert records.features.shape[0] == 2
    assert np.array_equal(records.cost_delta, np.asarray([1.0, -1.0]))
    assert records.features[0, 1] < 0.0
    assert records.features[1, 1] > 0.0


def test_dense_pairwise_gate_selects_only_beneficial_reference_action() -> None:
    states = [float(index % 2) for index in range(64)]
    dataset = _dataset(states)
    rigid = np.tile([0.0, 1.0], len(states))
    candidate = np.concatenate(
        [
            np.asarray([1.0, 0.0]) if state else np.asarray([0.0, 2.0])
            for state in states
        ]
    )
    actual = np.concatenate(
        [
            np.asarray([0.0, -1.0]) if state else np.asarray([0.0, 1.0])
            for state in states
        ]
    )
    records = build_dense_pressure_pairwise_utility_records(
        dataset,
        rigid,
        {"source|queue_service": candidate},
        actual=actual,
        fold_id=0,
    )
    records = type(records)(
        features=records.features,
        cost_delta=records.cost_delta,
        group_ids=records.group_ids,
        candidate_keys=records.candidate_keys,
        fold_ids=np.asarray([index % 4 for index in range(len(states))]),
    )
    gate = fit_dense_pressure_pairwise_utility_gate(
        records,
        config=UtilityGateConfig(
            ridge_alpha=0.01,
            error_quantile=0.5,
            minimum_records=16,
            minimum_predicted_gain=0.0,
        ),
    )

    test = _dataset([0.0, 1.0])
    test_rigid = np.tile([0.0, 1.0], 2)
    test_candidate = np.asarray([0.0, 2.0, 1.0, 0.0])
    score, audit = apply_dense_pressure_pairwise_utility_gate(
        test,
        test_rigid,
        {"source|queue_service": test_candidate},
        gate,
    )

    assert int(np.argmin(score[:2])) == 0
    assert int(np.argmin(score[2:])) == 1
    assert audit["selected_group_count"] == 1
    assert gate.diagnostics["record_protocol"] == (
        DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL
    )


def test_source_blind_candidates_preserve_rigid_scores_exactly() -> None:
    rigid = np.asarray([0.0, 1.0, 2.0])
    candidates = source_blind_candidate_scores(
        rigid,
        ["a|queue_service", "b|spillback"],
    )

    assert set(candidates) == {"a|queue_service", "b|spillback"}
    assert all(np.array_equal(value, rigid) for value in candidates.values())
    assert all(value is not rigid for value in candidates.values())


def test_cross_city_utility_weights_balance_folds_and_groups() -> None:
    groups = np.asarray(["a", "a", "b", "c", "d", "d", "d", "e"])
    folds = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])

    weights = _fold_and_group_balanced_weights(groups, folds)

    assert np.isclose(np.sum(weights[folds == 0]), np.sum(weights[folds == 1]))
    assert np.isclose(weights[0] + weights[1], weights[2])
    assert np.isclose(weights[4] + weights[5] + weights[6], weights[3])
    assert np.isclose(weights[4] + weights[5] + weights[6], weights[7])


def test_dense_gate_reports_cross_city_fold_balancing() -> None:
    width = len(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES)
    records = UtilityRecords(
        features=np.vstack(
            [np.zeros((24, width)), np.ones((72, width))]
        ),
        cost_delta=np.concatenate([np.full(24, -1.0), np.full(72, 1.0)]),
        group_ids=np.asarray(
            [f"city0:g{i}" for i in range(24)]
            + [f"city1:g{i}" for i in range(72)]
        ),
        candidate_keys=np.full(96, "source|queue_service", dtype=object),
        fold_ids=np.asarray(
            [index % 3 for index in range(24)]
            + [3 + index % 3 for index in range(72)]
        ),
    )

    gate = fit_dense_pressure_pairwise_utility_gate(
        records,
        config=UtilityGateConfig(minimum_records=24),
        balance_folds=True,
    )

    assert gate.model is not None
    assert gate.diagnostics["weighting_protocol"] == "equal_fold_then_equal_group"
