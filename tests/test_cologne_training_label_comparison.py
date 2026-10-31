import json

import numpy as np
import pytest

from scripts.data.cologne_training_label_comparison import compare_labels


def test_ranking_reversal_tracks_both_pp_signs_and_the_best_action():
    result = compare_labels([1, 2, 3, 2, 5, 6, 7, 8], [3, 2, 1, 2, 5, 6, 7, 8], 1)
    assert result["mean_absolute_cost_error"] == 0.5
    assert result["max_absolute_cost_error"] == 2
    assert result["cost_deltas"] == [2, 0, -2, 0, 0, 0, 0, 0]
    assert result["stored_advantages"][:4] == [1, 0, -1, 0]
    assert result["native_advantages"][:4] == [-1, 0, 1, 0]
    assert result["strict_sign_flip_indices"] == [0, 2]
    assert result["any_advantage_sign_change_indices"] == [0, 2]
    assert result["stored_optimal_indices"] == [0]
    assert result["native_optimal_indices"] == [2]
    assert result["optimal_set_changed"] and result["disjoint_optimal_sets"]
    assert result["any_cost_changed"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_tie_preservation_and_tie_creation_are_not_strict_reversals():
    stored = [1, 1, 2, 3, 4, 5, 6, 7]
    unchanged = compare_labels(stored, stored, 1)
    assert unchanged["stored_optimal_indices"] == unchanged["native_optimal_indices"] == [0, 1]
    assert unchanged["strict_sign_flip_indices"] == unchanged["any_advantage_sign_change_indices"] == []
    assert not unchanged["optimal_set_changed"] and not unchanged["any_cost_changed"]

    result = compare_labels(stored, [1, 1, 1, 3, 4, 5, 6, 7], 1)
    assert result["strict_sign_flip_indices"] == []
    assert result["any_advantage_sign_change_indices"] == [2]
    assert result["native_optimal_indices"] == [0, 1, 2]
    assert result["optimal_set_changed"] and not result["disjoint_optimal_sets"]


def test_tolerance_applies_to_cost_changes_advantage_signs_and_optimal_ties():
    tolerance = 2 ** -20
    stored = [0, tolerance, 2 * tolerance, 1, 2, 3, 4, 5]
    native = [0, -tolerance, 2 * tolerance, 1, 2, 3, 4, 5]
    result = compare_labels(stored, native, 0, tolerance)
    assert result["strict_sign_flip_indices"] == result["any_advantage_sign_change_indices"] == []
    assert result["stored_optimal_indices"] == result["native_optimal_indices"] == [0, 1]
    assert result["any_cost_changed"]  # Cost moved by twice tolerance, despite remaining tied to PP.

    native[1] = -1.5 * tolerance
    result = compare_labels(stored, native, 0, tolerance)
    assert result["strict_sign_flip_indices"] == []
    assert result["any_advantage_sign_change_indices"] == [1]
    assert result["native_optimal_indices"] == [1]

    native = [0, 0.5 * tolerance, 2 * tolerance, 1, 2, 3, 4, 5]
    assert not compare_labels(stored, native, 0, tolerance)["any_cost_changed"]


@pytest.mark.parametrize("invalid,side", [
    ([1] * 7, "stored"),
    ([[1] * 8], "native"),
    (np.ones((8, 1)), "stored"),
    ([1] * 7 + [float("nan")], "native"),
])
def test_rejects_incomplete_nonscalar_or_nonfinite_costs(invalid, side):
    with pytest.raises(ValueError, match="eight finite scalar costs"):
        compare_labels(invalid if side == "stored" else [1] * 8,
                       invalid if side == "native" else [1] * 8, 0)


@pytest.mark.parametrize("reference", [-1, 8, 1.5])
def test_rejects_reference_outside_the_candidate_roster(reference):
    with pytest.raises(ValueError, match="reference_index"):
        compare_labels([1] * 8, [1] * 8, reference)
