"""Exercise unit conversion and PP identity, without fits or simulation."""
from dataclasses import replace

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster import tsc_native_transfer_data as adapter


def bank(city):
    names = (*adapter.PAIRWISE_CAUSAL_STATE_PARENTS,
             *adapter.PAIRWISE_CAUSAL_ACTION_PARENTS, "v157b_prediction")
    features = np.zeros((8, len(names)))
    features[:, names.index("total_q")] = 12
    # A tie verifies the native controller's stable first-maximum convention.
    features[:, names.index("green_q")] = [1, 2, 9, 9, 4, 5, 6, 7]
    costs = np.arange(8, dtype=float) / 100 + 1
    group = "cologne1:seed2027:g0" if city == "cologne" else "jinan:g0"
    metadata = {
        "action_group_ids": [group] * 8,
        "candidate_states": [f"phase{i}" for i in range(8)],
        "row_tls": ["tls0"] * 8, "row_times": [300] * 8,
    }
    if city == "cologne":
        targets = {"interval_cost": costs, "prefix_mean_cost_450s": costs.copy()}
        metadata["scenario"] = "cologne1"
    else:
        targets = {"absolute_cost": costs, "interval_cost": costs - costs[2]}
        metadata.update(row_seeds=[123] * 8, row_scenarios=["jinan"] * 8,
                        is_reference=[False, False, True, False, False, False, False, False])
    return MechanismDataset(names, features, ("unused",), np.zeros((8, 1)), {}, targets,
                            np.asarray([city] * 8), metadata)


def convert(data, city):
    return adapter.canonicalize(data, city, 8 if city == "cologne" else 144,
                                adapter.PAIRWISE_CAUSAL_STATE_PARENTS,
                                adapter.PAIRWISE_CAUSAL_ACTION_PARENTS)


def test_common_features_use_all_physical_parents_and_exclude_predictions():
    states, actions = adapter.common_feature_names(bank("cologne"), bank("jinan"))
    assert len(states) == 23 and len(actions) == 40
    canonical, _ = convert(bank("cologne"), "cologne")
    assert canonical.feature_names == (*states, *actions)
    assert "v157b_prediction" not in canonical.feature_names


@pytest.mark.parametrize("city,factor", [("cologne", 3600), ("jinan", 64800)])
def test_absolute_and_existing_residual_labels_become_physical_pp_deltas(city, factor):
    original = bank(city)
    canonical, audit = convert(original, city)
    absolute = original.targets["interval_cost" if city == "cologne" else "absolute_cost"]
    np.testing.assert_allclose(canonical.targets["interval_cost"], (absolute - absolute[2]) * factor)
    np.testing.assert_array_equal(canonical.targets["absolute_cost"], absolute)
    assert canonical.metadata["reference_rows"] == [2] * 8
    assert np.flatnonzero(canonical.metadata["is_reference"]).tolist() == [2]
    assert audit["physical_cost_factor"] == factor
    assert canonical.targets["interval_cost"][2] == 0


def test_jinan_reference_and_residual_disagreement_stop_adaptation():
    data = bank("jinan")
    metadata = dict(data.metadata, is_reference=[True] + [False] * 7)
    with pytest.raises(ValueError, match="stored Jinan PP"):
        convert(replace(data, metadata=metadata), "jinan")
    targets = dict(data.targets, interval_cost=data.targets["interval_cost"] + 1)
    with pytest.raises(ValueError, match="candidate-minus-PP"):
        convert(replace(data, targets=targets), "jinan")


def test_pre_action_state_cannot_contain_candidate_dependent_values():
    data = bank("cologne")
    features = data.features.copy()
    features[1, data.feature_names.index("total_q")] += 1
    with pytest.raises(ValueError, match="pre-action state varies"):
        convert(replace(data, features=features), "cologne")
