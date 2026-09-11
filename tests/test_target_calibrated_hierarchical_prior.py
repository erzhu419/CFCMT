from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_target_calibrated_hierarchical_prior as module

from cf_h2o.eval.traffic_signal_hierarchical_mechanism_prior import (
    PRIOR_PROTOCOL,
    _candidate_key,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_target_calibrated_hierarchical_prior import (
    FEATURE_BINDING,
    METHOD_PROTOCOL,
    RESULT_PROTOCOL,
    SELECTOR_PROTOCOL,
    _TargetFold,
    _candidate_diagnostics,
    _fit_target_folds,
    _selector_audit,
    aggregate_results,
)


def _fold(index: int, *, passing: bool) -> _TargetFold:
    key = _candidate_key("queue_service", 0.005)
    rejected = _candidate_key("spillback", 0.005)
    rigid = np.asarray([1.0, 1.0])
    source = np.asarray([0.98, 0.98]) if passing else rigid.copy()
    return _TargetFold(
        fold_index=index,
        heldout_groups=(f"group-{index}-a", f"group-{index}-b"),
        rigid_values=rigid,
        source_values={key: source, rejected: np.asarray([1.1, 1.1])},
        placebo_values={
            key: np.asarray([1.0, 1.0]),
            rejected: np.asarray([1.0, 1.0]),
        },
        zero_values={
            key: np.asarray([0.995, 0.995]),
            rejected: np.asarray([1.0, 1.0]),
        },
        source_interventions={key: 1, rejected: 1},
    )


def test_nested_target_selector_admits_source_specific_candidate() -> None:
    folds = [_fold(index, passing=True) for index in range(5)]
    selector = _selector_audit(
        folds,
        inner_folds={index: [row for row in folds if row.fold_index != index]
                     for index in range(5)},
    )

    assert selector["admitted"] is True
    assert selector["candidate"] == _candidate_key("queue_service", 0.005)
    assert selector["nested_selector_intervention_group_count"] == 5
    assert len(selector["nested_selector_folds"]) == 5
    assert all(selector["checks"].values())


def test_nested_target_selector_rejects_no_gain() -> None:
    folds = [_fold(index, passing=False) for index in range(5)]
    selector = _selector_audit(
        folds,
        inner_folds={index: [row for row in folds if row.fold_index != index]
                     for index in range(5)},
    )

    assert selector["admitted"] is False
    assert selector["checks"]["nested_mean_gain_vs_rigid"] is False


@pytest.mark.parametrize("workers", [1, 3])
def test_nested_fits_exclude_outer_labels_from_candidate_selection(
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    groups = tuple(f"group-{index:03d}" for index in range(100))
    labels = {group: 0.0 for group in groups}
    calls = []
    key = _candidate_key("queue_service", 0.005)

    def prediction(problem, fold):
        training = tuple(fold["training_groups"])
        heldout = tuple(fold["heldout_groups"])
        calls.append((training, heldout))
        # A fitted candidate explicitly depends on every training label.
        fitted = sum(problem.labels[group] for group in training)
        values = np.full(len(heldout), fitted)
        return _TargetFold(
            fold_index=fold["fold_index"],
            heldout_groups=heldout,
            rigid_values=np.ones(len(heldout)),
            source_values={key: values},
            placebo_values={key: np.ones(len(heldout))},
            zero_values={key: np.ones(len(heldout))},
            source_interventions={key: len(heldout)},
        )

    monkeypatch.setattr(module, "_fold_prediction", prediction)
    problem = SimpleNamespace(selected_groups=groups, labels=labels, city="test")
    outer, inner = _fit_target_folds(problem, workers=workers)
    assert len(calls) == 25
    assert sum(len(training) == 80 for training, _ in calls) == 5
    assert sum(len(training) == 60 for training, _ in calls) == 20
    for outer_fold in outer:
        outer_heldout = set(outer_fold.heldout_groups)
        assert len(inner[outer_fold.fold_index]) == 4
        for inner_fold in inner[outer_fold.fold_index]:
            matching_training = [
                set(training)
                for training, heldout in calls
                if len(training) == 60
                and heldout == inner_fold.heldout_groups
                and not set(training) & outer_heldout
            ]
            assert matching_training == [
                set(groups) - outer_heldout - set(inner_fold.heldout_groups)
            ]

    # Perturb all outer-fold-0 labels. They must affect neither inner fitted
    # candidates nor their selection diagnostics for that outer fold.
    before = _candidate_diagnostics(inner[0])
    for group in outer[0].heldout_groups:
        labels[group] = 10.0
    changed_outer, changed_inner = _fit_target_folds(problem, workers=workers)
    assert _candidate_diagnostics(changed_inner[0]) == before
    assert _candidate_diagnostics(changed_outer[1:]) != _candidate_diagnostics(
        outer[1:]
    )


def test_selector_uses_inner_refits_instead_of_cached_full_oof() -> None:
    outer = [_fold(index, passing=True) for index in range(5)]
    key = _candidate_key("queue_service", 0.005)
    alternative = _candidate_key("spillback", 0.005)
    inner = {}
    for index in range(5):
        rows = [_fold(other, passing=True) for other in range(5) if other != index]
        for row in rows:
            row.source_values[key][:] = 1.1
            row.source_values[alternative][:] = 0.9
        inner[index] = rows

    selector = _selector_audit(outer, inner_folds=inner)

    assert selector["candidate"] == key
    assert all(row["selected_candidate"] == alternative
               for row in selector["nested_selector_folds"])
    assert selector["admitted"] is False


def _result(city: str, *, admitted: bool) -> dict:
    rigid = {str(seed): 1.0 for seed in SOURCE_SEEDS}
    source = {
        str(seed): value - (0.02 if admitted else 0.0)
        for seed, value in rigid.items()
    }
    placebo = {
        str(seed): value + (0.01 if admitted else 0.0)
        for seed, value in source.items()
    }
    zero = {
        str(seed): value + (0.015 if admitted else 0.0)
        for seed, value in source.items()
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "feature_binding": FEATURE_BINDING,
            "prior_protocol": PRIOR_PROTOCOL,
            "selector_protocol": SELECTOR_PROTOCOL,
            "selector": {"admitted": admitted},
        },
        "information_budget": {
            "target_selector_label_group_count": 100,
            "target_evaluation_label_count": 0,
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_target_calibrated_source_prior": {"seed_values": source},
            "selected_matched_placebo_prior": {"seed_values": placebo},
            "selected_zero_centred_prior": {"seed_values": zero},
            "forced_target_calibrated_source_prior": {"seed_values": source},
        },
    }


def test_v153a_aggregate_passes_two_admissions_and_exact_fallback() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["feature_binding"] == "stored_feature_names"
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v153a_source_prior_closed_loop_development"
    )


def test_v153a_aggregate_rejects_matched_placebo_tie() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]
    for index in range(2):
        rows[index]["arm_summaries"]["selected_matched_placebo_prior"] = deepcopy(
            rows[index]["arm_summaries"][
                "selected_target_calibrated_source_prior"
            ]
        )

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["checks"][
        "mean_selected_effect_beats_matched_placebo"
    ] is False


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_v153a_v3_aggregate_rejects_previous_results(version: str) -> None:
    rows = [_result(city, admitted=False) for city in EXPECTED_CITY_GROUPS]
    rows[0]["protocol"] = (
        f"tsc-v153a-target-calibrated-hierarchical-prior-target-{version}"
    )

    with pytest.raises(ValueError, match="seven valid B100"):
        aggregate_results(rows)


@pytest.mark.parametrize("binding", [None, "training_column_indices"])
def test_v153a_v3_aggregate_requires_named_feature_binding(binding) -> None:
    rows = [_result(city, admitted=False) for city in EXPECTED_CITY_GROUPS]
    rows[0]["method"]["feature_binding"] = binding

    with pytest.raises(ValueError, match="seven valid B100"):
        aggregate_results(rows)
