"""Fixed-split and coverage-gate checks with fake fits, never real estimators."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import fit_tsc_joint_response_trial as fit


def protocol():
    source, target = list(range(101, 105)), list(range(201, 211))
    cases = [{"case_id": f"{city}_{seed}_{slot}", "city": city, "seed": seed, "tls_id": "tls"}
             for city, seeds in (("hangzhou", source), ("jinan", target)) for seed in seeds for slot in range(2)]
    return dict(protocol=fit.PROTOCOL, source_seeds=source, target_seeds=target, cases=cases,
        response_horizon_sec=10, cost_horizon_sec=450, max_new_fits=30,
        feature_names=list(fit.FEATURE_NAMES),
        model={"config": fit.MODEL_CONFIG, "sklearn_version": "1.7.2"},
        weighting={"total_weight": 1536, "source_fraction": .5}, placebo_seed=20261106,
        analysis={"bootstrap_draws": 10000, "bootstrap_seed": 20261106})


def group(case):
    X = np.zeros((8, 12, len(fit.FEATURE_NAMES)))
    X[:, :, 0] = case["seed"]  # A test marker, not a production feature.
    X[:, :, 1] = np.arange(8)[:, None]
    raw_X = np.zeros((8, 12, len(fit.RAW_FEATURE_NAMES)))
    Y = np.arange(8)[:, None, None] * np.ones((8, 12, 2))
    Y[:, :, 1] += np.arange(12)[None, :]
    return {**case, "X": X, "raw_X": raw_X, "Y": Y, "reference_index": 3,
            "movement_ids": [f"m{i}" for i in range(12)],
            "candidate_states": [f"phase{i}" for i in range(8)]}


def coverage(spec, status="PASS_SOURCE_COVERAGE"):
    return dict(protocol=fit.PROTOCOL, status=status, case_ids=[c["case_id"] for c in spec["cases"]])


def write_collection(root, spec):
    for case in spec["cases"]:
        data = group(case)
        directory = root / f"seed_{case['seed']}" / case["case_id"]
        directory.mkdir(parents=True)
        receipt = {**case, "status": "COMPLETE", "resolved_checkpoint_sec": 600, "branches": [
            {"candidate_index": i, "status": "COMPLETE", "checks": {"same_movement_roster": True}}
            for i in [3, 0, 1, 2, 4, 5, 6, 7]]}
        payload = {**{k: v for k, v in data.items() if k not in {"X", "raw_X", "Y"}},
            "case": case, "complete": True, "X": data["raw_X"].tolist(), "Y": data["Y"].tolist(),
            "feature_names": list(fit.RAW_FEATURE_NAMES), "costs_450": [999999.] * 8,
            "branches": [{"candidate_index": i, "preaction": {"movement_ids": data["movement_ids"]}}
                         for i in range(8)]}
        graph = dict(protocol=fit.INTERFACE_PROTOCOL, case_id=case["case_id"], focal_tls=case["tls_id"],
            reference_index=3, information_time_sec=600, fixture_seed=case["seed"],
            movement_nodes=[{"movement_id": m} for m in data["movement_ids"]],
            candidate_views=[{"candidate_index": i} for i in range(8)])
        (directory / "result.json").write_text(json.dumps(receipt))
        (directory / "group_data_server_only.json").write_text(json.dumps(payload))
        (directory / "joint_interface_server_only.json").write_text(json.dumps(graph))


def fake_features(graph):
    return group({"seed": graph["fixture_seed"]})["X"]


def test_whole_seed_folds_and_same_training_mass_with_twelve_source_movements():
    spec = protocol()
    groups = [group(case) for case in spec["cases"]]
    source = [g for g in groups if g["city"] == "hangzhou"]
    placebo, _ = fit.placebo_labels(source)
    heldout_ids = []
    for train, test, heldout in fit.folds(groups, spec["target_seeds"]):
        assert not {g["seed"] for g in train} & set(heldout)
        assert len(train) == 16 and len(test) == 4
        heldout_ids.extend(g["case_id"] for g in test)
        for arm in fit.ARMS:
            X, Y, weights, audit = fit.training_design(train, source, arm, placebo)
            assert len(X) == len(Y) == (1536 if arm == "target_only" else 2304)
            assert weights.sum() == pytest.approx(1536)
            assert not set(X[:, 0]) & set(heldout)
            assert audit["target_weight"] == (1536 if arm == "target_only" else 768)
            assert audit["source_weight"] == (0 if arm == "target_only" else 768)
            assert audit["source_rows"] == (0 if arm == "target_only" else 768)
            if arm == "target_only":
                assert not set(X[:, 0]) & set(spec["source_seeds"])
                np.testing.assert_array_equal(weights, np.ones(1536))
            else:
                np.testing.assert_array_equal(weights[:1536], np.full(1536, .5))
                np.testing.assert_array_equal(weights[1536:], np.ones(768))
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


def test_contrast_metric_and_raw_internal_count_balance_are_unchanged():
    g = group(protocol()["cases"][-1])
    g["raw_X"][:, :, fit.RAW_FEATURE_NAMES.index("internal_vehicle_count")] = 2
    errors, prediction = fit.response_error(g, g["Y"] + 7)
    assert errors["contrast_mae"] == 0 and errors["raw_count_mae"] == 7
    assert errors["prediction_balance_audit"] == {
        "movement_action_count": 96,
        "receiving_exceeds_initial_internal_plus_entry_count": 72}
    np.testing.assert_array_equal(prediction, g["Y"] + 7)
    altered = prediction.copy(); altered[0, :, 0] += 14
    errors, _ = fit.response_error(g, altered)
    assert errors["contrast_mae"] == pytest.approx(1)


def test_coverage_hold_prevents_even_loading_training_labels(tmp_path, monkeypatch):
    spec = protocol(); spec_path = tmp_path / "protocol.json"; spec_path.write_text(json.dumps(spec))
    decision = tmp_path / "coverage.json"; decision.write_text(json.dumps(coverage(spec, "HOLD_SOURCE_COVERAGE")))
    monkeypatch.setattr(fit, "load_groups", lambda *args: pytest.fail("HOLD must not load labels"))
    args = SimpleNamespace(spec=spec_path, coverage_decision=decision,
                           collection_root=tmp_path / "absent", output=tmp_path / "hold")
    result = fit.run(args)
    assert result["status"] == "HOLD_SOURCE_COVERAGE" and result["new_fits"] == 0
    wrong = coverage(spec); wrong["case_ids"][0] = "different_roster"
    decision.write_text(json.dumps(wrong)); args.output = tmp_path / "wrong"
    result = fit.run(args)
    assert result["status"] == "HOLD_INPUT" and result["new_fits"] == 0


def test_complete_mock_pipeline_keeps_30_fits_and_raw_analytic_inputs(tmp_path, monkeypatch):
    spec = protocol(); fit.validate_spec(spec)
    spec_path = tmp_path / "protocol.json"; spec_path.write_text(json.dumps(spec))
    decision = tmp_path / "coverage.json"; decision.write_text(json.dumps(coverage(spec)))
    root = tmp_path / "collection"; write_collection(root, spec)
    monkeypatch.setattr(fit, "build_joint_features", fake_features)
    calls, analytic_calls = [], []

    class FakeModel:
        def __init__(self, **config):
            assert config == fit.MODEL_CONFIG

        def fit(self, X, y, sample_weight):
            calls.append((set(X[:, 0]), len(X), sample_weight.sum(), X.shape[1]))
            assert X.shape[1] == len(fit.FEATURE_NAMES)

        def predict(self, X):
            return -np.ones(len(X))

    def analytic(X):
        analytic_calls.append(X.shape)
        assert X.shape == (96, len(fit.RAW_FEATURE_NAMES)) and np.count_nonzero(X) == 0
        return np.zeros((len(X), 2))

    monkeypatch.setattr(fit, "HistGradientBoostingRegressor", FakeModel)
    monkeypatch.setattr(fit, "analytic_baseline", analytic)
    args = SimpleNamespace(spec=spec_path, coverage_decision=decision,
                           collection_root=root, output=tmp_path / "analysis")
    result = fit.run(args)
    assert len(calls) == result["new_fits"] == 30 and len(analytic_calls) == 20
    assert result["status"] == "HOLD_RESPONSE" and not result["network_readout_preparation_supported"]
    for fold in range(5):
        heldout = set(spec["target_seeds"][fold * 2:fold * 2 + 2])
        assert all(not marker & heldout for marker, *_ in calls[fold * 6:fold * 6 + 6])
    saved = json.loads((tmp_path / "analysis/oof_server_only.json").read_text())
    assert len(saved["records"]) == 60
    assert all(np.count_nonzero(row["predicted_counts"]) == 0 for row in saved["records"])
    missing = spec["cases"][-1]
    (root / f"seed_{missing['seed']}" / missing["case_id"] / "result.json").unlink()
    args.output = tmp_path / "missing"
    held = fit.run(args)
    assert held["status"] == "HOLD_INPUT" and held["new_fits"] == 0 and len(calls) == 30


def test_loader_rejects_graph_movement_misalignment_and_raw_future_columns(tmp_path, monkeypatch):
    spec = protocol(); write_collection(tmp_path, spec)
    monkeypatch.setattr(fit, "build_joint_features", fake_features)
    loaded = fit.load_groups(spec, tmp_path)
    assert len(loaded) == 28 and all(g["X"].shape[1] == 12 for g in loaded)
    case = spec["cases"][0]; directory = tmp_path / f"seed_{case['seed']}" / case["case_id"]
    graph_path = directory / "joint_interface_server_only.json"; original = graph_path.read_text()
    graph = json.loads(original); graph["movement_nodes"].reverse(); graph_path.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="ordered movement/action roster"):
        fit.load_groups(spec, tmp_path)
    graph_path.write_text(original)
    path = directory / "group_data_server_only.json"; data = json.loads(path.read_text())
    data["feature_names"][-1] = "actual_future_green_seconds"; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="roster differs"):
        fit.load_groups(spec, tmp_path)
