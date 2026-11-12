"""Check native transfer labels, mass, placebo and selected costs without fits."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster import run_tsc_native_direct_transfer as runner


STATE = ("state",)
ACTION = ("action", "switch_indicator")


def bank(source=False):
    domains = (["cologne1"] * 195 if source else
               ["jinan_3x4_real"] * 4 + ["jinan_3x4_real_2000"] * 3 + ["jinan_3x4_real_2500"] * 3)
    if not source:
        domains *= 10
    features, absolute, physical, row_domains = [], [], [], []
    meta = {key: [] for key in ("action_group_ids", "is_reference", "row_tls", "row_times",
                                "candidate_states", "row_seeds", "row_scenarios")}
    factor = 3600 if source else 64800
    for group, domain in enumerate(domains):
        ref = group % 8
        actions = np.arange(8, dtype=float)
        cost = 1 + group / 1000 + (group % 3 + 1) * actions ** 2 / 10000
        features.extend(np.column_stack((np.full(8, group), actions, np.ones(8))))
        absolute.extend(cost)
        physical.extend((cost - cost[ref]) * factor)
        row_domains.extend([domain] * 8)
        seed = [2027, 3037, 4047][group % 3] if source else runner.v158.DEVELOPMENT_SEEDS[group // 10]
        for action in range(8):
            values = dict(action_group_ids=f"{'source' if source else 'target'}:{group}",
                          is_reference=action == ref, row_tls="junction", row_times=300,
                          candidate_states=f"state{action}", row_seeds=seed, row_scenarios=domain)
            for key, value in values.items():
                meta[key].append(value)
    return MechanismDataset(feature_names=(*STATE, *ACTION), features=np.asarray(features),
                            context_names=("context",), context=np.zeros((len(features), 1)), priors={},
                            targets={"absolute_cost": np.asarray(absolute), "interval_cost": np.asarray(physical)},
                            domains=np.asarray(row_domains), metadata=meta)


def first_training_fold():
    target = bank()
    folds = runner.target_folds(target, [list(fold) for fold in runner.v158.CROSSFIT_FOLDS])
    return runner.target_subset(target, folds[0][0])


def test_design_preserves_physical_residuals_and_differences_only_action_columns():
    data = first_training_fold()
    design = runner.build_design(data, STATE, ACTION)
    assert design["X"].shape == (640, 3)
    assert len(design["candidate_rows"]) == 560
    np.testing.assert_array_equal(design["y"], data.targets["interval_cost"])
    np.testing.assert_array_equal(design["X"][:, 0], data.features[:, 0])
    for rows in runner.group_rows(data).values():
        ref = rows[np.asarray(data.metadata["is_reference"])[rows]][0]
        np.testing.assert_allclose(design["X"][rows, 1:], data.features[rows, 1:] - data.features[ref, 1:])
    changed = {name: values.copy() for name, values in data.targets.items()}
    changed["interval_cost"][np.flatnonzero(design["reference"])[0]] = 1
    with pytest.raises(ValueError, match="PP-relative"):
        runner.build_design(replace(data, targets=changed), STATE, ACTION)


def test_strict_target_and_source_arms_match_mass_without_mean_row_renormalization():
    target = runner.build_design(first_training_fold(), STATE, ACTION)
    source = runner.build_design(bank(source=True), STATE, ACTION)
    strict = runner.sample_weights(target)
    mixed = runner.sample_weights(target, source)
    assert strict["diagnostics"]["source_rows"] == 0
    assert mixed["diagnostics"]["source_rows"] == 1365
    assert len(strict["target"]) == len(mixed["target"]) == 560
    assert strict["target"].sum() == pytest.approx(560)
    assert mixed["target"].sum() == pytest.approx(280)
    assert mixed["source"].sum() == pytest.approx(280)
    for info, mass in [(strict["diagnostics"], 560), (mixed["diagnostics"], 280)]:
        assert info["target_domain_groups"] == runner.DOMAIN_GROUPS
        assert all(value == pytest.approx(mass / 3) for value in info["target_per_domain_weight"].values())
    for group in set(source["group_ids"]):
        mask = source["group_ids"][source["candidate_rows"]] == group
        assert mixed["source"][mask].sum() == pytest.approx(280 / 195)
    assert np.r_[mixed["target"], mixed["source"]].mean() != pytest.approx(1)


def test_placebo_preserves_each_group_label_distribution_and_pp_zero():
    source = runner.build_design(bank(source=True), STATE, ACTION)
    original = source["y"].copy()
    placebo = runner.permute_source_labels(source)
    np.testing.assert_array_equal(placebo, runner.permute_source_labels(source))
    np.testing.assert_array_equal(source["y"], original)
    assert np.count_nonzero(placebo != original) > 0
    np.testing.assert_array_equal(placebo[source["reference"]], np.zeros(195))
    for group in set(source["group_ids"]):
        rows = np.flatnonzero((source["group_ids"] == group) & ~source["reference"])
        np.testing.assert_array_equal(np.sort(placebo[rows]), np.sort(original[rows]))


def test_selector_keeps_native_cost_units_reference_ties_and_stay_veto():
    data = runner.target_subset(bank(), {"target:1"})  # PP is row1, not row0.
    scores = np.array([-64.8, 999., 0., 0., 0., 0., 0., 0.])
    records, native_scores = runner.policy_records(data, scores)
    assert records[0]["selected_state"] == "state0"
    assert records[0]["predicted_advantage"] == pytest.approx(.001)
    assert native_scores[1] == 0
    assert records[0]["selected_cost"] == data.targets["absolute_cost"][0]
    # A tiny physical negative score is below the existing native-unit tolerance.
    scores[0] = -1e-9
    assert runner.policy_records(data, scores)[0][0]["selected_state"] == "state1"
    # Best candidate is a stay while PP switches: return PP, not second-best switch.
    data.features[0, 2] = 0
    scores[0], scores[2] = -100, -50
    assert runner.policy_records(data, scores)[0][0]["selected_state"] == "state1"


def test_folds_hold_out_whole_original_seeds_and_cover_every_group_once():
    data = bank()
    heldout_sets = []
    for training, heldout in runner.target_folds(data, [list(fold) for fold in runner.v158.CROSSFIT_FOLDS]):
        assert len(training) == 80 and len(heldout) == 20 and not training & heldout
        train = runner.target_subset(data, training)
        test = runner.target_subset(data, heldout)
        assert not set(train.metadata["row_seeds"]) & set(test.metadata["row_seeds"])
        heldout_sets.extend(heldout)
    assert len(heldout_sets) == len(set(heldout_sets)) == 100


def test_frozen_protocol_and_complete_output_path_without_real_fits(tmp_path, monkeypatch):
    spec = json.loads((Path(__file__).resolve().parents[1] / "experiments/tsc_native_direct_transfer_20261106/protocol.json").read_text())
    runner.validate_spec(spec)  # Descriptive protocol fields must not become fit kwargs.
    spec["features"].update(state_feature_count=len(STATE), action_feature_count=len(ACTION))
    spec_path, inputs_path = tmp_path / "protocol.json", tmp_path / "inputs.json"
    spec_path.write_text(json.dumps(spec))
    inputs_path.write_text("{}")
    monkeypatch.setitem(sys.modules, "tsc_native_transfer_data", SimpleNamespace(load_native_inputs=lambda *_: (
        bank(source=True), bank(), {"state_feature_names": STATE, "action_feature_names": ACTION})))
    calls = []

    class NoFitModel:
        def __init__(self, **config):
            assert config == runner.MODEL_CONFIG

        def fit(self, X, y, sample_weight):
            calls.append((len(X), float(sample_weight.sum())))

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(runner, "HistGradientBoostingRegressor", NoFitModel)
    result = runner.run(SimpleNamespace(spec=spec_path, source_inputs=inputs_path,
                                        out_dir=tmp_path / "analysis", history_result=None))
    assert calls == [(560, pytest.approx(560)), (1925, pytest.approx(560)), (1925, pytest.approx(560))] * 5
    assert result["new_fits"] == 15  # Mock-call accounting; no estimator was trained.
    assert result["status"] == "HOLD_DEVELOPMENT"
    assert result["mean_costs"]["target_only"] == result["mean_costs"]["source_target"] == result["mean_costs"]["phase_pressure"]
    saved = json.loads((tmp_path / "analysis/oof_server_only.json").read_text())
    assert len(saved["fits"]) == 15
    assert all(len(rows) == 100 for rows in saved["records"].values())
