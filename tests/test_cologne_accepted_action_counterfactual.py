"""Check temporal cost vectors without running SUMO or fitting a model."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def compare_vectors(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/data"))
    return importlib.import_module("run_cologne_accepted_action_counterfactual").compare_vectors


def test_identical_full_outcome_vector_is_accepted(compare_vectors):
    vector = list(range(450)) + [0.] * 10
    result = compare_vectors(vector, vector)
    assert result["passed"]
    assert result["left_count"] == result["right_count"] == 460
    assert result["max_absolute_difference"] == 0.


@pytest.mark.parametrize("left,right", [([], []), ([1.] * 450, [1.] * 449)])
def test_empty_or_truncated_vector_is_not_equivalent(compare_vectors, left, right):
    result = compare_vectors(left, right)
    assert result["passed"] is False
    assert result["left_count"] == len(left)
    assert result["right_count"] == len(right)


def test_same_mean_with_shifted_first_interval_is_not_equivalent(compare_vectors):
    left = [0., 0., 1., 1., 2., 2., 3., 3., 4., 4.]
    right = left[1:] + left[:1]
    assert sum(left) == sum(right)
    result = compare_vectors(left, right)
    assert result["passed"] is False
    assert result["max_absolute_difference"] == 4.


def test_replay_terminal_difference_is_not_hidden_by_identical_costs(compare_vectors):
    original = [1.] * 450 + [0.] * 10
    changed = list(original)
    changed[-1] = .625
    result = compare_vectors(original, changed)
    assert result["passed"] is False
    assert result["max_absolute_difference"] == .625
