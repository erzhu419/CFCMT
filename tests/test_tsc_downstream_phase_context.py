from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.cluster import fit_tsc_predecision_history as history
from scripts.cluster import run_tsc_downstream_phase_context as runner
from test_tsc_native_action_timeline import frozen_bank as runtime_contrast_fixture
from test_tsc_predecision_history_fit import fixture_bank


def prepared_bank():
    data, histories = fixture_bank()
    # Use the actual runtime contrast schema, including reference_* and delta_*.
    contrast = runtime_contrast_fixture()[-1]
    missing = tuple(n for n in contrast.feature_names if n not in data.feature_names)
    extra = np.arange(data.size * len(missing), dtype=float).reshape(data.size, -1)
    data = replace(data, feature_names=(*data.feature_names, *missing),
                   features=np.column_stack((data.features, extra)))
    data = history.join_history(data, histories)
    packets = []
    for group, rows in runner.group_rows(data).items():
        i = rows[0]
        packets.append(dict(group_id=group, seed=data.metadata["row_seeds"][i],
            scenario=data.metadata["row_scenarios"][i], checkpoint_sec=data.metadata["row_times"][i],
            tls_id=data.metadata["row_tls"][i],
            candidate_states=[data.metadata["candidate_states"][r] for r in rows],
            features=(5 + np.arange(8)[:, None] * np.arange(1, 7)[None, :]).tolist()))
    return data, packets


def test_join_preserves_full_contrast_history_labels_and_raw_six_columns():
    data, packets = prepared_bank()
    joined = runner.join_features(data, packets[::-1])
    assert joined.feature_names == (*data.feature_names, *runner.FEATURE_NAMES)
    assert any(n.startswith("reference_") for n in data.feature_names)
    assert any(n.startswith("delta_") for n in data.feature_names)
    assert set(history.HISTORY_NAMES) <= set(data.feature_names)
    np.testing.assert_array_equal(joined.features[:, :len(data.feature_names)], data.features)
    np.testing.assert_array_equal(joined.features[:8, -6:], packets[0]["features"])
    for key in data.targets:
        np.testing.assert_array_equal(joined.targets[key], data.targets[key])
    assert joined.metadata == data.metadata


def test_join_rejects_missing_group_and_candidate_reordering():
    data, packets = prepared_bank()
    with pytest.raises(ValueError, match="complete one-to-one"):
        runner.join_features(data, packets[:-1])
    packets[0]["candidate_states"].reverse()
    with pytest.raises(ValueError, match="candidate order"):
        runner.join_features(data, packets)


def test_mapped_only_and_floating_roundoff_do_not_satisfy_phase_support():
    data, packets = prepared_bank()
    for packet in packets:
        values = np.zeros((8, 6))
        values[:, 0] = np.arange(8) / 8
        values[:, 1:] = np.arange(8)[:, None] * 1e-16
        packet["features"] = values.tolist()
    joined = runner.join_features(data, packets)
    result = runner.input_support(joined, history.seed_folds(joined, history.v158.CROSSFIT_FOLDS))
    assert not result["passed"] and result["phase_informative_groups"] == 0
    assert result["all_groups"]["ds_mapped_fraction"]["nonzero_groups"] == 100
    assert result["all_groups"]["ds_protected_fraction"]["max_abs_action_minus_pp"] > 0


def test_phase_support_must_exist_in_every_fixed_training_fold():
    data, packets = prepared_bank()
    for packet in packets:
        packet["features"] = np.zeros((8, 6)).tolist()
    packets[0]["features"][1][1] = 0.5  # Sole support is held out by fold 0.
    joined = runner.join_features(data, packets)
    folds = history.seed_folds(joined, history.v158.CROSSFIT_FOLDS)
    result = runner.input_support(joined, folds)
    assert result["phase_informative_groups"] == 1
    assert result["training_fold_informative_groups"] == [0, 1, 1, 1, 1]
    assert not result["passed"]
    packets[20]["features"][1][1] = 0.5  # A seed in the next held-out pair.
    result = runner.input_support(runner.join_features(data, packets), folds)
    assert result["passed"] and result["training_fold_informative_groups"] == [1, 1, 2, 2, 2]


def records_for_bank(data):
    records = []
    for group, rows in runner.group_rows(data).items():
        reference, selected = rows[0], rows[1]
        absolute = data.targets["absolute_cost"]
        records.append(dict(group_id=group, seed=data.metadata["row_seeds"][reference],
            scenario=data.metadata["row_scenarios"][reference],
            selected_row=-100, reference_row=999999,  # Deliberately meaningless outside its fold.
            selected_state=data.metadata["candidate_states"][selected],
            reference_state=data.metadata["candidate_states"][reference],
            selected_cost=absolute[selected], phase_pressure_cost=absolute[reference],
            actual_advantage=absolute[reference] - absolute[selected], different_action=True))
    return records


def test_cached_records_align_by_group_and_state_not_fold_local_row():
    data, _ = prepared_bank()
    records = records_for_bank(data)
    runner.validate_records(data, records[::-1])
    with pytest.raises(ValueError, match="all original groups exactly once"):
        runner.validate_records(data, records[:-1])
    changed = deepcopy(records)
    changed[0]["selected_cost"] += 0.01
    with pytest.raises(ValueError, match="selected_cost differs"):
        runner.validate_records(data, changed)


def test_true_residual_design_differences_new_raw_columns_once():
    data, packets = prepared_bank()
    augmented = runner.join_features(data, packets)
    arm = runner.v158._arm_dataset(augmented, "target_native")
    state = (*runner.v158.PAIRWISE_CAUSAL_STATE_PARENTS, *history.HISTORY_STATE_NAMES)
    action = (*runner.v158.PAIRWISE_CAUSAL_ACTION_PARENTS, *runner.v158.META_ACTION_FEATURES,
              *history.HISTORY_ACTION_NAMES, *runner.FEATURE_NAMES)
    model = runner.CausalReferenceResidualRegressor(state_feature_names=state, action_feature_names=action)
    model._resolve_features(arm)
    design, reference = model._design(arm)
    assert len(model.state_feature_names) == 35 and len(model.action_feature_names) == 68
    assert design.shape == (800, 103) and np.count_nonzero(reference) == 100
    raw = np.asarray(packets[0]["features"])
    np.testing.assert_array_equal(design[:8, -6:], raw - raw[0])
    np.testing.assert_array_equal(augmented.features[:8, -6:], raw)
    assert np.all(augmented.features[0, -6:] == 5)  # Input wasn't pre-differenced.
