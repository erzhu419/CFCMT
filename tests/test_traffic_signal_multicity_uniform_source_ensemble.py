from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble as v143
import cf_h2o.eval.traffic_signal_target_budget_source_value_curve as curve


def test_target_model_keeps_jinan_default_but_accepts_explicit_city() -> None:
    parameters = signature(curve._target_model).parameters
    assert parameters["target_domain"].default == "jinan"


def test_city_covariate_signature_is_scenario_equal_and_label_free() -> None:
    def dataset(offset: float, rows: int):
        return SimpleNamespace(
            feature_names=("state",),
            features=np.arange(rows, dtype=float)[:, None] + offset,
            context_names=("lanes",),
            context=np.full((rows, 1), 2.0 + offset),
            size=rows,
            priors=np.full((rows, 1), 999.0),
            targets=np.full((rows, 1), -999.0),
        )

    signature = v143._city_covariate_signature(
        {"small": dataset(0.0, 2), "large": dataset(10.0, 10)},
        ("small", "large"),
    )
    expected = (np.mean(np.arange(2)) + np.mean(np.arange(10) + 10.0)) / 2.0
    assert signature["feature_mean"] == pytest.approx([expected])
    assert signature["context_mean"] == pytest.approx([7.0])
    assert signature["information_boundary"].endswith("no_priors_or_targets")


def test_city_b25_selection_balances_strata_and_retains_evaluation(monkeypatch) -> None:
    scenarios = ("scenario_a", "scenario_b")
    groups = [
        f"{scenario}:seed{seed}:{index:03d}:tls"
        for scenario in scenarios
        for seed in v143.SOURCE_SEEDS
        for index in range(10)
    ]
    dataset = SimpleNamespace(metadata={"action_group_ids": groups})
    monkeypatch.setattr(
        v143,
        "_coverage_first_group_order_v3",
        lambda _dataset, candidates, **_kwargs: tuple(sorted(candidates)),
    )
    selected, audit = v143.select_city_budget_groups(
        dataset, city="city", scenarios=scenarios
    )
    assert len(selected) == v143.TARGET_BUDGET
    assert all(row["evaluation_group_count"] > 0 for row in audit["strata"].values())
    quotas = list(audit["scenario_seed_quotas"].values())
    assert max(quotas) - min(quotas) <= 1


def test_city_b25_selection_rejects_a_stratum_without_evaluation(monkeypatch) -> None:
    groups = [
        f"scenario:seed{seed}:{index:03d}:tls"
        for seed in v143.SOURCE_SEEDS
        for index in range(9)
    ]
    dataset = SimpleNamespace(metadata={"action_group_ids": groups})
    monkeypatch.setattr(
        v143,
        "_coverage_first_group_order_v3",
        lambda _dataset, candidates, **_kwargs: tuple(sorted(candidates)),
    )
    with pytest.raises(ValueError, match="cannot retain evaluation groups"):
        v143.select_city_budget_groups(
            dataset, city="city", scenarios=("scenario",)
        )


def test_scenario_seed_summary_weights_units_not_group_counts() -> None:
    scenarios = ("scenario_a", "scenario_b")
    unit_values = {
        ("scenario_a", 2027): -0.6,
        ("scenario_a", 3037): -0.3,
        ("scenario_a", 4047): 0.0,
        ("scenario_b", 2027): 0.2,
        ("scenario_b", 3037): 0.4,
        ("scenario_b", 4047): 0.6,
    }
    unique_groups = []
    actual = []
    score = []
    references = []
    rows = []
    cursor = 0
    for unit_index, ((scenario, seed), value) in enumerate(unit_values.items()):
        for group_index in range(unit_index + 1):
            unique_groups.append(f"{scenario}:seed{seed}:{group_index}:tls")
            actual.extend((value, 0.0))
            score.extend((-1.0, 0.0))
            references.extend((False, True))
            rows.append((cursor, cursor + 1))
            cursor += 2
    summary = v143._scenario_seed_policy_summary(
        actual=np.asarray(actual),
        score=np.asarray(score),
        policy_rows=np.asarray(rows, dtype=int),
        references=np.asarray(references, dtype=bool),
        unique_groups=unique_groups,
        scenarios=scenarios,
    )
    assert summary["mean"] == pytest.approx(np.mean(list(unit_values.values())))
    assert summary["seed_values"]["2027"] == pytest.approx((-0.6 + 0.2) / 2)


def test_reference_policy_score_selects_only_reference_rows() -> None:
    references = np.asarray((False, True, False, True), dtype=bool)
    score = v143._reference_policy_score(references)
    assert score.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_label_free_intervention_diagnostic_accepts_ragged_groups() -> None:
    diagnostic = v143._label_free_intervention_diagnostic(
        np.asarray([-1.0, 0.0, 0.2, -0.4, 0.0]),
        (
            np.asarray([0, 1]),
            np.asarray([2, 3, 4]),
        ),
        np.asarray([False, True, False, False, True]),
    )
    assert diagnostic["group_count"] == 2
    assert diagnostic["intervention_fraction"] == pytest.approx(1.0)
    assert diagnostic["information_boundary"].endswith("reference_only")


def test_scenario_seed_summary_accepts_mixed_action_group_sizes() -> None:
    actual = np.asarray([0.2, 0.0, -0.3, 0.0, 0.4, 0.1, 0.0])
    score = np.asarray([-1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0])
    references = np.asarray([False, True, False, True, False, False, True])
    policy_rows = (
        np.asarray([0, 1]),
        np.asarray([2, 3, 4]),
        np.asarray([5, 6]),
    )
    groups = [
        "scenario:seed2027:000:tls",
        "scenario:seed3037:000:tls",
        "scenario:seed4047:000:tls",
    ]
    summary = v143._scenario_seed_policy_summary(
        actual=actual,
        score=score,
        policy_rows=policy_rows,
        references=references,
        unique_groups=groups,
        scenarios=("scenario",),
    )
    assert summary["mean"] == pytest.approx(0.0)
    assert all(
        value == pytest.approx(1.0)
        for value in summary["scenario_seed_intervention_fraction"].values()
    )
