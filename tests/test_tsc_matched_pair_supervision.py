"""Check paired labels, effective mass and reference scores without fitting."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster import run_tsc_matched_pair_supervision as runner


STATE = ("state",)
ACTION = tuple(f"action_{i}" for i in range(8))


def training_bank():
    domains = np.repeat(["jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"], [32, 24, 24])
    features, absolute, residual, row_domains = [], [], [], []
    metadata = {key: [] for key in ("action_group_ids", "is_reference", "candidate_states", "row_seeds", "row_scenarios")}
    for group, domain in enumerate(domains):
        reference = group % 8
        cost = 1 + group / 100 + (1 + group % 3) * np.arange(8) ** 2 / 1000
        features.extend(np.column_stack((np.full(8, group), np.eye(8))))
        absolute.extend(cost)
        residual.extend(cost - cost[reference])
        row_domains.extend([domain] * 8)
        for action in range(8):
            for key, value in dict(action_group_ids=f"g{group}", is_reference=action == reference,
                                   candidate_states=f"action{action}", row_seeds=group // 10,
                                   row_scenarios=domain).items():
                metadata[key].append(value)
    return MechanismDataset(feature_names=(*STATE, *ACTION), features=np.asarray(features),
                            context_names=("context",), context=np.zeros((640, 1)), priors={},
                            targets={"absolute_cost": np.asarray(absolute), "interval_cost": np.asarray(residual)},
                            domains=np.asarray(row_domains), metadata=metadata)


def test_all_oriented_pairs_keep_raw_cost_scale_and_reverse_weights():
    data = training_bank()
    design = runner.build_pair_design(data, STATE, ACTION)
    assert design["X"].shape == (4480, 9)
    assert design["y"].shape == (4480,)
    for group in range(80):
        rows = np.flatnonzero(np.asarray(design["group_ids"]) == f"g{group}")
        assert len(rows) == 56
        differences = design["X"][rows, 1:]
        pairs = [(int(np.argmax(delta)), int(np.argmin(delta))) for delta in differences]
        assert set(pairs) == {(a, b) for a in range(8) for b in range(8) if a != b}
        np.testing.assert_array_equal(design["X"][rows, 0], np.full(56, group))
        cost = data.targets["absolute_cost"][8 * group:8 * group + 8]
        expected = np.array([cost[a] - cost[b] for a, b in pairs])
        np.testing.assert_allclose(design["y"][rows], expected, rtol=0, atol=1e-15)
        for arm in ("S", "A"):
            by_pair = dict(zip(pairs, design["weights"][arm][rows]))
            assert all(weight == by_pair[b, a] for (a, b), weight in by_pair.items())
        assert np.count_nonzero(design["weights"]["S"][rows]) == 14
        assert np.count_nonzero(design["weights"]["A"][rows]) == 56
        reference = group % 8
        np.testing.assert_array_equal(design["weights"]["S"][rows] > 0,
                                      [a == reference or b == reference for a, b in pairs])


def test_weight_mass_is_matched_by_domain_and_group_without_row_renormalization():
    design = runner.build_pair_design(training_bank(), STATE, ACTION)
    for arm in ("S", "A"):
        weights = design["weights"][arm]
        assert weights.sum() == pytest.approx(560)
        for domain, count in (("jinan_3x4_real", 32), ("jinan_3x4_real_2000", 24), ("jinan_3x4_real_2500", 24)):
            mask = np.asarray(design["domains"]) == domain
            assert weights[mask].sum() == pytest.approx(560 / 3)
            for group in set(np.asarray(design["group_ids"])[mask]):
                selected = np.asarray(design["group_ids"]) == group
                assert weights[selected].sum() == pytest.approx(560 / (3 * count))
        assert weights.mean() == pytest.approx(0.125)


def test_pair_scores_cancel_even_terms_and_anchor_each_actual_pp_row():
    data = training_bank()
    coefficients = np.array([0.03, -0.02, 0.05, -0.01, 0.06, 0.04, -0.03, 0.02])

    class AnalyticalPredictor:
        def predict(self, X):
            delta = X[:, 1:]
            return 10 + 3 * X[:, 0] + 2 * (delta @ coefficients) + 5 * (delta ** 2).sum(axis=1)

    scores = runner.pair_scores(AnalyticalPredictor(), data, STATE, ACTION)
    expected = np.concatenate([2 * (coefficients - coefficients[group % 8]) for group in range(80)])
    np.testing.assert_allclose(scores, expected, rtol=0, atol=5e-14)
    np.testing.assert_array_equal(scores[np.asarray(data.metadata["is_reference"])], np.zeros(80))


def test_pair_labels_reject_mismatched_reference_residuals():
    data = training_bank()
    targets = {key: values.copy() for key, values in data.targets.items()}
    targets["interval_cost"][1] += 0.01
    with pytest.raises(ValueError):
        runner.build_pair_design(replace(data, targets=targets), STATE, ACTION)


def test_cached_record_validation_uses_identity_and_native_cost_not_fold_row_ids():
    data = training_bank()
    records = []
    for group in range(80):
        reference, selected = 8 * group + group % 8, 8 * group + (group + 1) % 8
        cost = data.targets["absolute_cost"]
        records.append(dict(group_id=f"g{group}", seed=data.metadata["row_seeds"][reference],
                            scenario=data.metadata["row_scenarios"][reference], selected_row=-100, reference_row=999999,
                            selected_state=data.metadata["candidate_states"][selected],
                            reference_state=data.metadata["candidate_states"][reference],
                            selected_cost=cost[selected], phase_pressure_cost=cost[reference],
                            actual_advantage=cost[reference] - cost[selected], different_action=True))
    runner.validate_records(data, records[::-1])
    changed = deepcopy(records)
    changed[0]["selected_cost"] += 0.01
    with pytest.raises(ValueError):
        runner.validate_records(data, changed)
    with pytest.raises(ValueError):
        runner.validate_records(data, records[:-1])
