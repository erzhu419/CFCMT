from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.cluster import fit_tsc_predecision_history as fit
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def fixture_bank():
    names = (*fit.v158.PAIRWISE_CAUSAL_STATE_PARENTS, *fit.v158.PAIRWISE_CAUSAL_ACTION_PARENTS,
             *fit.v158.RAW_PREDICTION_FEATURES)
    metadata = {name: [] for name in ("action_group_ids", "is_reference", "reference_rows", "row_tls",
                                    "row_times", "candidate_states", "row_seeds", "row_scenarios")}
    histories, rows, targets = [], [], []
    for seed in fit.v158.DEVELOPMENT_SEEDS:
        for slot in range(10):
            scenario = "jinan_3x4_real" + ("" if slot < 4 else "_2000" if slot < 7 else "_2500")
            group = f"{scenario}:seed{seed}:slot{slot}:t300:tls"
            start = len(rows)
            for action in range(8):
                values = np.zeros(len(names))
                values[names.index("total_q")] = slot + 1
                values[names.index("green_q")] = action
                values[names.index("switch_indicator")] = action > 0
                rows.append(values)
                targets.append(action * (-0.01 if slot % 2 else 0.01))
                for key, value in dict(action_group_ids=group, is_reference=action == 0, reference_rows=start,
                                       row_tls="tls", row_times=300, candidate_states=f"state{action}",
                                       row_seeds=seed, row_scenarios=scenario).items():
                    metadata[key].append(value)
            history = np.zeros((8, 28))
            history[:, :12] = slot
            history[:, 12:] = np.arange(8)[:, None]
            histories.append(dict(group_id=group, seed=seed, scenario=scenario, checkpoint_sec=300, tls_id="tls",
                                  candidate_states=[f"state{action}" for action in range(8)],
                                  history_times=list(range(240, 300, 10)), feature_names=list(fit.HISTORY_NAMES),
                                  features=history.tolist(), parity={"static_features_exact": True}))
    data = MechanismDataset(feature_names=names, features=np.asarray(rows), context_names=("context",),
                            context=np.zeros((800, 1)), priors={}, targets={"interval_cost": np.asarray(targets),
                            "absolute_cost": 1 + np.asarray(targets)}, domains=np.asarray(["jinan"] * 800),
                            metadata=metadata)
    return data, histories


def test_history_join_preserves_labels_and_aligns_independent_of_record_order():
    data, histories = fixture_bank()
    joined = fit.join_history(data, histories[::-1])
    np.testing.assert_array_equal(joined.targets["interval_cost"], data.targets["interval_cost"])
    np.testing.assert_array_equal(joined.features[:, :data.features.shape[1]], data.features)
    np.testing.assert_array_equal(joined.features[:8, -28:], histories[0]["features"])
    assert joined.metadata == data.metadata


@pytest.mark.parametrize("change,match", [
    (lambda row: row["candidate_states"].reverse(), "candidate order"),
    (lambda row: row["history_times"].__setitem__(-1, 300), "history times"),
    (lambda row: row["feature_names"].pop(), "feature names"),
    (lambda row: row.__setitem__("seed", 999), "seed does not match"),
    (lambda row: row["features"][1].__setitem__(0, 99), "shared by all candidate actions"),
    (lambda row: row["parity"].__setitem__("static_features_exact", False), "checkpoint parity"),
])
def test_rejects_misalignment_leakage_or_invalid_replay(change, match):
    data, histories = fixture_bank()
    change(histories[0])
    with pytest.raises(ValueError, match=match):
        fit.join_history(data, histories)


def test_join_rejects_missing_group_instead_of_filling_with_zero():
    data, histories = fixture_bank()
    with pytest.raises(ValueError, match="one-to-one complete"):
        fit.join_history(data, histories[:-1])


def test_whole_seed_folds_never_split_related_actions_or_demands():
    data, _ = fixture_bank()
    folds = fit.seed_folds(data, fit.v158.CROSSFIT_FOLDS)
    group_seed = dict(zip(data.metadata["action_group_ids"], data.metadata["row_seeds"]))
    observed = []
    for training, heldout in folds:
        assert not {group_seed[group] for group in training} & {group_seed[group] for group in heldout}
        assert len(training) == 80 and len(heldout) == 20
        observed.extend(heldout)
    assert len(observed) == len(set(observed)) == 100
    changed = deepcopy(data.metadata)
    changed["row_seeds"][1] = fit.v158.DEVELOPMENT_SEEDS[1]
    with pytest.raises(ValueError, match="crosses simulator seeds"):
        fit.seed_folds(replace(data, metadata=changed), fit.v158.CROSSFIT_FOLDS)
    with pytest.raises(ValueError, match="fold roster"):
        fit.seed_folds(data, fit.v158.CROSSFIT_FOLDS[::-1])


def test_model_uses_declared_history_and_rejects_silently_dropped_feature():
    data, histories = fixture_bank()
    joined = fit.join_history(data, histories)
    selected = set(joined.metadata["action_group_ids"][:80])
    subset = fit.v158._subset_groups(joined, selected)
    for enabled in (False, True):
        _, diagnostics = fit.fit_model(subset, enabled)
        assert diagnostics["state_feature_count"] == 35
        assert diagnostics["action_feature_count"] == 62
        assert set(fit.HISTORY_NAMES) <= set(diagnostics["state_feature_names"] + diagnostics["action_feature_names"])
    incomplete = replace(subset, feature_names=subset.feature_names[:-1], features=subset.features[:, :-1])
    with pytest.raises(ValueError, match="missing declared features"):
        fit.fit_model(incomplete, True)


def test_behavior_reports_harm_even_with_positive_net_gain():
    rows = [dict(actual_advantage=value, different_action=value != 0) for value in (0, 0.2, -0.1)]
    summary = fit.behavior(rows)
    assert summary["harmful_overrides"] == 1 and summary["harmful_fraction"] == 0.5
    assert summary["raw_harm_sum"] == 0.1 and summary["raw_net_gain_mean"] > 0
