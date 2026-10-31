from types import SimpleNamespace

import numpy as np
import pytest

import cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting as v140


SELECTOR_SEEDS = (
    22208,
    27178,
    31657,
    34310,
    46535,
    47023,
    50079,
    50744,
    50945,
    51400,
    52367,
    61481,
    61994,
    63775,
    73412,
    74374,
    74744,
    77524,
    80314,
    84579,
    84751,
    88625,
)


def test_seed_partition_is_exact_label_independent_and_disjoint() -> None:
    adaptation, evaluation = v140.split_adaptation_evaluation_seeds(
        tuple(reversed(SELECTOR_SEEDS))
    )
    assert adaptation == SELECTOR_SEEDS[:5]
    assert evaluation == SELECTOR_SEEDS[5:]
    assert not set(adaptation) & set(evaluation)
    with pytest.raises(ValueError, match="exactly 22 unique"):
        v140.split_adaptation_evaluation_seeds(SELECTOR_SEEDS[:-1])
    with pytest.raises(ValueError, match="exactly 22 unique"):
        v140.split_adaptation_evaluation_seeds((*SELECTOR_SEEDS[:-1], SELECTOR_SEEDS[0]))


def test_total_budget_groups_are_balanced_and_strictly_nested(monkeypatch) -> None:
    adaptation = SELECTOR_SEEDS[:5]
    groups = [
        f"jinan:seed{seed}:{index:04d}:tls"
        for seed in adaptation
        for index in range(240)
    ]
    dataset = SimpleNamespace(metadata={"action_group_ids": groups})
    monkeypatch.setattr(
        v140,
        "_coverage_first_group_order_v3",
        lambda _dataset, candidates, **_kwargs: tuple(sorted(candidates)),
    )
    selected = {}
    for budget in v140.TARGET_BUDGETS:
        chosen, audit = v140.select_total_budget_groups(
            dataset,
            adaptation_seeds=adaptation,
            budget=budget,
        )
        selected[budget] = chosen
        assert len(chosen) == budget
        quotas = list(audit["seed_quotas"].values())
        assert max(quotas) - min(quotas) <= 1
        assert {v140.parse_action_group_seed(group) for group in chosen} == set(adaptation)
    v140.validate_nested_total_budgets(selected)


def test_nonnegative_weights_recover_signal_inside_simplex() -> None:
    groups = []
    references = []
    row_seeds = []
    source = []
    for seed_index, seed in enumerate(SELECTOR_SEEDS[:5]):
        for group_index in range(4):
            group = f"jinan:seed{seed}:{group_index}:tls"
            groups.extend((group, group))
            references.extend((False, True))
            row_seeds.extend((seed, seed))
            source.append((1.0 + 0.1 * seed_index, 0.2 + 0.05 * group_index))
            source.append((0.0, 0.0))
    source_matrix = np.asarray(source, dtype=float)
    target = np.zeros(len(groups), dtype=float)
    expected = np.asarray([0.6, 0.25], dtype=float)
    actual = target + source_matrix @ expected
    fitted, diagnostics = v140.fit_nonnegative_source_weights(
        actual=actual,
        target_score=target,
        source_residuals=source_matrix,
        groups=np.asarray(groups, dtype=str),
        references=np.asarray(references, dtype=bool),
        row_seeds=np.asarray(row_seeds, dtype=int),
        fit_seeds=SELECTOR_SEEDS[:5],
        ridge_l2=0.0,
    )
    assert np.allclose(fitted, expected, atol=1e-6)
    assert np.all(fitted >= 0.0)
    assert float(np.sum(fitted)) <= 1.0
    assert diagnostics["source_null_weight"] == pytest.approx(0.15)


def test_placebo_moves_whole_action_blocks_within_seed_stratum() -> None:
    groups = np.asarray(
        [
            "jinan:seed22208:0:tls",
            "jinan:seed22208:0:tls",
            "jinan:seed22208:0:tls",
            "jinan:seed22208:1:tls",
            "jinan:seed22208:1:tls",
            "jinan:seed22208:1:tls",
        ],
        dtype=str,
    )
    references = np.asarray([False, False, True, False, False, True], dtype=bool)
    rank_signal = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    source = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [3.0, 30.0], [4.0, 40.0], [0.0, 0.0]]
    )
    placebo = v140.permute_source_residuals_by_seed_group(
        source,
        groups,
        references,
        rank_signal,
    )
    assert np.array_equal(placebo[:2], source[3:5])
    assert np.array_equal(placebo[3:5], source[:2])
    assert np.array_equal(placebo[references], source[references])


def test_nested_gate_never_fits_the_heldout_seed(monkeypatch) -> None:
    seeds = (1, 2, 3, 4, 5)
    groups = np.asarray(
        [f"jinan:seed{seed}:0:tls" for seed in seeds for _ in range(2)],
        dtype=str,
    )
    references = np.asarray([False, True] * len(seeds), dtype=bool)
    row_seeds = np.repeat(np.asarray(seeds, dtype=int), 2)
    policy_rows = np.arange(2 * len(seeds), dtype=int).reshape(len(seeds), 2)
    group_seeds = np.asarray(seeds, dtype=int)
    actual = np.asarray([-0.01, 0.0] * len(seeds), dtype=float)
    target = np.asarray([1.0, 0.0] * len(seeds), dtype=float)
    aligned = np.asarray([[-2.0], [0.0]] * len(seeds), dtype=float)
    placebo = np.zeros_like(aligned)
    calls = []

    def fake_fit(*, source_residuals, fit_seeds, **_kwargs):
        calls.append(tuple(int(seed) for seed in fit_seeds))
        weight = 0.0 if np.allclose(source_residuals, 0.0) else 1.0
        return np.asarray([weight]), {"fit_seeds": list(fit_seeds)}

    monkeypatch.setattr(v140, "fit_nonnegative_source_weights", fake_fit)
    gate = v140.nested_source_weight_gate(
        actual=actual,
        target_score=target,
        source_residuals=aligned,
        placebo_residuals=placebo,
        groups=groups,
        references=references,
        row_seeds=row_seeds,
        policy_rows=policy_rows,
        group_seeds=group_seeds,
        adaptation_seeds=seeds,
    )
    for fold_index, heldout in enumerate(seeds):
        assert heldout not in calls[2 * fold_index]
        assert heldout not in calls[2 * fold_index + 1]
        assert len(calls[2 * fold_index]) == 4
    assert gate["aligned_authorized"] is True
    assert gate["placebo_authorized"] is False
    assert gate["aligned_full_weights_effective"] == [1.0]
    assert gate["placebo_full_weights_effective"] == [0.0]


def test_selector_arm_seed_keys_are_json_objects() -> None:
    summary = v140._selector_arm_summary({2: -0.1, 1: 0.0})
    assert summary["seed_values"] == {"1": 0.0, "2": -0.1}
