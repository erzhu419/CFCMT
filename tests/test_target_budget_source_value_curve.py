from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np
import pytest

import cf_h2o.eval.traffic_signal_target_budget_source_value_curve as curve
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
    _policy_arrays,
    nested_leave_one_seed_source_selection,
    select_city_adaptation_groups_for_budget,
    validate_nested_budget_groups,
)
from scripts.cluster.launch_tsc_target_budget_source_value_curve import (
    SEEDS,
    SIGNATURE,
    build_spec,
)


class _Dataset:
    def __init__(self, groups: list[str]) -> None:
        self.metadata = {"action_group_ids": groups}


def test_budget_selection_is_balanced_and_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        curve,
        "_group_seed_v3",
        lambda group: int(str(group).split(":seed", 1)[1].split(":", 1)[0]),
    )
    monkeypatch.setattr(
        curve,
        "_coverage_first_group_order_v3",
        lambda _dataset, candidates, **_kwargs: sorted(candidates),
    )
    bank = {
        scenario: _Dataset(
            [
                f"{scenario}:seed{seed}:g{index:03d}"
                for seed in curve.ADAPTATION_SEEDS
                for index in range(100)
            ]
        )
        for scenario in ("a", "b", "c")
    }
    selected = {}
    for budget in (25, 50, 100):
        groups, audit = select_city_adaptation_groups_for_budget(
            bank,
            city="jinan",
            scenarios=("a", "b", "c"),
            budget=budget,
        )
        selected[budget] = groups
        assert len(groups) == budget
        counts = audit["selected_group_count_by_seed"]
        assert abs(counts["5057"] - counts["6067"]) <= 1
    validate_nested_budget_groups(selected, expected_b100=selected[100])


def test_budget_validation_rejects_changed_frozen_b100() -> None:
    with pytest.raises(ValueError, match="V123 B100 groups"):
        validate_nested_budget_groups(
            {25: ("a",), 50: ("a", "b"), 100: ("a", "b", "c")},
            expected_b100=("other",),
        )


def test_nested_source_selection_never_uses_heldout_seed_for_choice() -> None:
    candidates = {
        "a": {
            "1": {"mean_normalized_delta_vs_phase_pressure": 100.0},
            "2": {"mean_normalized_delta_vs_phase_pressure": -2.0},
            "3": {"mean_normalized_delta_vs_phase_pressure": -2.0},
        },
        "b": {
            "1": {"mean_normalized_delta_vs_phase_pressure": 0.0},
            "2": {"mean_normalized_delta_vs_phase_pressure": 1.0},
            "3": {"mean_normalized_delta_vs_phase_pressure": 1.0},
        },
    }
    target = {
        seed: {"mean_normalized_delta_vs_phase_pressure": 0.5}
        for seed in ("1", "2", "3")
    }
    result = nested_leave_one_seed_source_selection(candidates, target)
    fold1 = next(row for row in result["folds"] if row["heldout_seed"] == 1)
    assert fold1["selected_candidate"] == "a"
    assert fold1["source_policy_delta"] == 100.0


def test_paired_bootstrap_is_deterministic() -> None:
    first = _paired_bootstrap([-0.3, -0.2, 0.1, -0.1])
    second = _paired_bootstrap([-0.3, -0.2, 0.1, -0.1])
    assert first == second
    assert np.isclose(first["mean"], -0.125)


def test_vectorized_policy_rows_match_ragged_fallback() -> None:
    actual = np.asarray([0.0, -0.2, 0.1, 0.0, 0.4, -0.1])
    score = np.asarray([0.0, -0.3, 0.2, 0.0, 0.1, -0.2])
    references = np.asarray([True, False, False, True, False, False])
    ragged = (np.asarray([0, 1, 2]), np.asarray([3, 4, 5]))
    matrix = np.vstack(ragged)
    expected = _policy_arrays(actual, score, ragged, references)
    observed = _policy_arrays(actual, score, matrix, references)
    assert np.array_equal(observed[0], expected[0])
    assert np.array_equal(observed[1], expected[1])


def test_v123_launcher_freezes_full_budget_curve_and_selector() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/source"),
        source_cache_audit_remote=Path("/audit/source.json"),
        target_cache_root=Path("/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/audit/selector.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/results/v123"),
        node="node005",
        cache_workers=20,
        fit_workers=20,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20
    assert spec["ram_mb"] == 98304
    assert "cf_h2o.eval.traffic_signal_target_budget_source_value_curve" in tokens
    seed_start = tokens.index("--selector-seeds") + 1
    seed_stop = tokens.index("--selector-collection-shards")
    assert [int(value) for value in tokens[seed_start:seed_stop]] == list(SEEDS)
    assert tokens[tokens.index("--fit-workers") + 1] == "20"
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")
