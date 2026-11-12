from collections import Counter
from copy import deepcopy

import pytest

from scripts.cluster import run_tsc_native_calibration_budget as budget


def test_cyclic_subsets_are_nested_balanced_and_exclude_outer_seeds():
    seeds = list(range(10))
    folds = [seeds[i:i + 2] for i in range(0, 10, 2)]
    cells = list(budget.nested_cells(seeds, folds))
    assert len(cells) * 2 == 160  # Both arms fit; the 80-group endpoint is reused.
    assert {cell["budget"] for cell in cells} == {20, 40}
    for fold, heldout in enumerate(folds):
        selected = [cell for cell in cells if cell["fold"] == fold]
        for rotation in range(8):
            small, large = [cell for cell in selected if cell["rotation"] == rotation]
            assert small["training_seeds"] == large["training_seeds"][:2]
            assert not set(large["training_seeds"]) & set(heldout)
            assert small["heldout_seeds"] == large["heldout_seeds"] == heldout
        for size, repeats in ((20, 2), (40, 4)):
            counts = Counter(seed for cell in selected if cell["budget"] == size
                             for seed in cell["training_seeds"])
            assert counts == Counter({seed: repeats for seed in seeds if seed not in heldout})


def records(seed, gains):
    return [dict(seed=seed, group_id=f"{seed}:{i}",
                 scenario="base" if i < 4 else "mid" if i < 7 else "high",
                 phase_pressure_cost=10., selected_cost=10. - gain,
                 actual_advantage=gain, different_action=gain != 0)
            for i, gain in enumerate(gains)]


def cells_with_gains(even, odd):
    return [dict(budget=20, rotation=r, heldout_seeds=[100, 200],
                 records=records(100, odd if r % 2 else even)
                         + records(200, odd if r % 2 else even))
            for r in range(8)]


def test_outcomes_are_averaged_before_nonlinear_harm_not_as_ensemble():
    cells = cells_with_gains([-4.] * 10, [4.] * 10)
    summary = budget.aggregate_cells(cells, [100, 200])
    assert summary["mean"]["cost"] == 10.
    assert summary["mean"]["harm"] == summary["mean"]["benefit"] == 2.
    assert summary["mean"]["harmful"] == 5.
    assert summary["independent_seed_count"] == 2  # Not 16 seed/rotation entries.
    assert len(summary["per_seed"]) == 2
    assert summary["subset_count_per_seed"] == 8
    assert summary["variation_across_fixed_subsets"][100]["cost"] == dict(min=6., max=14., std=4.)


def test_harmful_fraction_is_ratio_of_expected_counts():
    cells = cells_with_gains([-1.] + [0.] * 9, [-1.] * 5 + [1.] * 5)
    summary = budget.aggregate_cells(cells, [100, 200])
    assert summary["mean"]["harmful"] == 3.
    assert summary["mean"]["overrides"] == 5.5
    assert summary["harmful_fraction"] == pytest.approx(6 / 11)
    assert summary["harmful_fraction"] != .75  # Mean of per-rotation fractions.
    endpoint = [dict(budget=80, rotation=None, reused=True, heldout_seeds=[100, 200],
                     records=records(100, [0.] * 10) + records(200, [0.] * 10))]
    frozen = budget.aggregate_cells(endpoint, [100, 200])
    assert frozen["subset_count_per_seed"] == 1
    assert frozen["harmful_fraction"] is None


@pytest.mark.parametrize("remaining", [1, 7])
def test_small_budget_missing_rotations_cannot_be_treated_as_endpoint(remaining):
    cells = cells_with_gains([-1.] * 10, [1.] * 10)
    with pytest.raises(ValueError):
        budget.aggregate_cells(cells[:remaining], [100, 200])


def test_duplicate_rotation_and_dropped_validation_group_are_rejected():
    cells = cells_with_gains([-1.] * 10, [1.] * 10)
    duplicate = deepcopy(cells)
    duplicate[-1]["rotation"] = 0
    with pytest.raises(ValueError, match="rotation"):
        budget.aggregate_cells(duplicate, [100, 200])
    cells[3]["records"] = cells[3]["records"][1:]
    with pytest.raises(ValueError, match="group"):
        budget.aggregate_cells(cells, [100, 200])


def test_demand_summary_preserves_the_same_group_roster_per_rotation():
    cells = cells_with_gains([-1.] * 4 + [0.] * 6, [1.] * 4 + [0.] * 6)
    summary = budget.aggregate_cells(cells, [100, 200], scenario="base")
    assert summary["mean"]["overrides"] == 4.
    assert summary["mean"]["harmful"] == 2.
    assert summary["mean"]["harm"] == .5
    assert summary["independent_seed_count"] == 2
