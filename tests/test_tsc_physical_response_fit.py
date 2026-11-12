"""Focused protocol checks using fake fits; no simulation or trained estimator."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import fit_tsc_physical_response_trial as fit


def protocol():
    source, target = list(range(101, 105)), list(range(201, 211))
    cases = [{"case_id": f"{city}_{seed}_{slot}", "city": city, "seed": seed}
             for city, seeds in (("cologne", source), ("jinan", target)) for seed in seeds for slot in range(2)]
    return dict(protocol=fit.PROTOCOL, source_seeds=source, target_seeds=target, cases=cases,
        response_horizon_sec=10, cost_horizon_sec=450, max_new_fits=30,
        model={"config": fit.MODEL_CONFIG, "sklearn_version": "1.7.2"},
        weighting={"total_weight": 1536, "source_fraction": .5}, placebo_seed=20261106,
        analysis={"bootstrap_draws": 10000, "bootstrap_seed": 20261106})


def group(case):
    movements = 16 if case["city"] == "cologne" else 12
    X = np.zeros((8, movements, len(fit.FEATURE_NAMES)))
    X[:, :, 0] = case["seed"]  # Test marker: identify exactly which seeds reach fit.
    X[:, :, 1] = np.arange(8)[:, None]
    Y = np.arange(8)[:, None, None] * np.ones((8, movements, 2))
    Y[:, :, 1] += np.arange(movements)[None, :]
    return {**case, "X": X, "Y": Y, "reference_index": 3,
            "movement_ids": [f"m{i}" for i in range(movements)],
            "candidate_states": [f"phase{i}" for i in range(8)]}


def write_collection(root, spec):
    for case in spec["cases"]:
        data = group(case)
        directory = root / f"seed_{case['seed']}" / case["case_id"]
        directory.mkdir(parents=True)
        receipt = {**case, "status": "COMPLETE", "branches": [
            {"candidate_index": i, "status": "COMPLETE", "checks": {"same_movement_roster": True}}
            for i in [3, 0, 1, 2, 4, 5, 6, 7]]}  # Collector executes the nonzero PP index first.
        payload = {**data, "X": data["X"].tolist(), "Y": data["Y"].tolist(),
            "feature_names": list(fit.FEATURE_NAMES), "label_names": list(fit.LABEL_NAMES),
            "costs_450": [999999.] * 8, "network_preaction": {"future_poison": 1e50},
            "branches": [{"candidate_index": i, "preaction": {"movement_ids": data["movement_ids"]}} for i in range(8)]}
        (directory / "result.json").write_text(json.dumps(receipt))
        (directory / "group_data_server_only.json").write_text(json.dumps(payload))


def test_whole_seed_folds_and_training_mass():
    spec = protocol()
    groups = [group(case) for case in spec["cases"]]
    source = [g for g in groups if g["city"] == "cologne"]
    placebo, _ = fit.placebo_labels(source)
    heldout_ids = []
    for train, test, heldout in fit.folds(groups, spec["target_seeds"]):
        assert not {g["seed"] for g in train} & set(heldout)
        assert len(train) == 16 and len(test) == 4
        heldout_ids.extend(g["case_id"] for g in test)
        for arm in fit.ARMS:
            X, Y, weights, audit = fit.training_design(train, source, arm, placebo)
            assert len(X) == len(Y) == (1536 if arm == "target_only" else 2560)
            assert weights.sum() == pytest.approx(1536)
            assert not set(X[:, 0]) & set(heldout)
            assert audit["target_weight"] == (1536 if arm == "target_only" else 768)
            assert audit["source_weight"] == (0 if arm == "target_only" else 768)
            if arm == "target_only":
                assert not set(X[:, 0]) & set(spec["source_seeds"])
                np.testing.assert_array_equal(weights, np.ones(1536))
    assert len(heldout_ids) == len(set(heldout_ids)) == 20


def test_placebo_moves_whole_action_vectors_and_preserves_pp():
    source = [group(case) for case in protocol()["cases"][:8]]
    labels, mapping = fit.placebo_labels(source)
    changed = 0
    for g in source:
        value = labels[g["case_id"]]
        np.testing.assert_array_equal(value[3], g["Y"][3])
        assert set(mapping[g["case_id"]].values()) == set(range(8)) - {3}
        for action, donor in mapping[g["case_id"]].items():
            np.testing.assert_array_equal(value[int(action)], g["Y"][donor])
        changed += int(np.count_nonzero(value != g["Y"]))
    assert changed > 0


def test_contrast_metric_uses_predicted_pp_and_ignores_common_response_offset():
    g = group(protocol()["cases"][-1])
    g["X"][:, :, fit.FEATURE_NAMES.index("internal_vehicle_count")] = 2
    errors, prediction = fit.response_error(g, g["Y"] + 7)
    assert errors["contrast_mae"] == 0 and errors["raw_count_mae"] == 7
    assert np.all(prediction[3] > 0)  # Raw PP is never forced to zero.
    assert errors["prediction_balance_audit"] == {
        "movement_action_count": 96,
        "receiving_exceeds_initial_internal_plus_entry_count": 72}
    np.testing.assert_array_equal(prediction, g["Y"] + 7)  # Audit never projects predictions.
    altered = prediction.copy()
    altered[0, :, 0] += 14
    errors, _ = fit.response_error(g, altered)
    assert errors["contrast_mae"] == pytest.approx(1)
    assert errors["output_contrast_mae"][fit.LABEL_NAMES[0]] == pytest.approx(2)


def test_complete_mock_pipeline_and_missing_group_prevents_any_fit(tmp_path, monkeypatch):
    spec = protocol()
    fit.validate_spec(spec)
    spec_path = tmp_path / "protocol.json"
    spec_path.write_text(json.dumps(spec))
    root = tmp_path / "collection"
    write_collection(root, spec)
    calls = []

    class FakeModel:
        def __init__(self, **config):
            assert config == fit.MODEL_CONFIG

        def fit(self, X, y, sample_weight):
            calls.append((set(X[:, 0]), len(X), sample_weight.sum(), X.shape[1]))
            assert X.shape[1] == len(fit.FEATURE_NAMES)

        def predict(self, X):
            return -np.ones(len(X))

    monkeypatch.setattr(fit, "HistGradientBoostingRegressor", FakeModel)
    result = fit.run(SimpleNamespace(spec=spec_path, collection_root=root, output=tmp_path / "analysis"))
    assert len(calls) == result["new_fits"] == 30
    assert result["status"] == "HOLD_RESPONSE" and not result["network_readout_preparation_supported"]
    assert result["mean_contrast_mae"]["source_target"] == result["mean_contrast_mae"]["target_only"]
    for arm in fit.ARMS:
        assert result["prediction_balance_audit"][arm] == {
            "movement_action_count": 1920,
            "receiving_exceeds_initial_internal_plus_entry_count": 0}
    for fold in range(5):
        heldout = set(spec["target_seeds"][fold * 2:fold * 2 + 2])
        assert all(not marker & heldout for marker, *_ in calls[fold * 6:fold * 6 + 6])
    saved = json.loads((tmp_path / "analysis/oof_server_only.json").read_text())
    assert len(saved["records"]) == 60
    assert all(np.count_nonzero(row["predicted_counts"]) == 0 for row in saved["records"])
    missing = spec["cases"][-1]
    (root / f"seed_{missing['seed']}" / missing["case_id"] / "result.json").unlink()
    held = fit.run(SimpleNamespace(spec=spec_path, collection_root=root, output=tmp_path / "missing"))
    assert held["status"] == "HOLD_INPUT" and held["new_fits"] == 0 and len(calls) == 30


def test_pp_first_receipt_preserves_candidate_order_and_rejects_duplicate(tmp_path):
    spec = protocol()
    write_collection(tmp_path, spec)
    loaded = fit.load_groups(spec, tmp_path)
    for actual, case in zip(loaded, spec["cases"]):
        expected = group(case)
        assert actual["reference_index"] == 3
        np.testing.assert_array_equal(actual["X"], expected["X"])
        np.testing.assert_array_equal(actual["Y"], expected["Y"])
    case = spec["cases"][0]
    path = tmp_path / f"seed_{case['seed']}" / case["case_id"] / "result.json"
    receipt = json.loads(path.read_text())
    assert receipt["branches"][0]["candidate_index"] == 3
    receipt["branches"][1]["candidate_index"] = 3
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="incomplete native action group"):
        fit.load_groups(spec, tmp_path)


def test_movement_reordering_and_future_feature_columns_are_rejected(tmp_path):
    spec = protocol()
    root = tmp_path / "collection"
    write_collection(root, spec)
    case = spec["cases"][0]
    path = root / f"seed_{case['seed']}" / case["case_id"] / "group_data_server_only.json"
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed["branches"][7]["preaction"]["movement_ids"].reverse()
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ordered focal movements"):
        fit.load_groups(spec, root)
    original["feature_names"][-1] = "actual_future_green_seconds"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="roster differs"):
        fit.load_groups(spec, root)
