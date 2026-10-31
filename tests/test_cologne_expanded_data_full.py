from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from cf_h2o.traffic_signal import action_ranker as ranker
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.data import fit_cologne_expanded_data_full as fit


def prior_fixture():
    root = Path(__file__).parents[1] / 'cf_h2o/results/cluster/tsc_v155k_cologne_expanded_data_oof_20260910'
    previous_protocol = json.loads((root / 'protocol_v1.json').read_text())
    previous = json.loads((root / 'fit_v1/result.json').read_text())
    protocol = {**previous_protocol, 'protocol': fit.PROTOCOL}
    return protocol, previous_protocol, previous


def contrast_fixture(protocol):
    groups = protocol['original_group_ids'] + protocol['additional_group_ids']
    names = tuple(reversed(protocol['original_feature_names']))
    target = np.tile([0., -.1, .2, .15, -.05, .05, .3, -.2], 195)
    features = np.random.default_rng(15).normal(size=(1560, 29))
    features[:, names.index('delta_service_pressure')] = target * 10.
    return MechanismDataset(feature_names=names, features=features,
        context_names=('density',), context=np.repeat(np.arange(195)/195, 8)[:, None],
        targets={'interval_cost': target}, priors={}, domains=np.full(1560, 'cologne1'),
        metadata={'action_group_ids': np.repeat(groups, 8).tolist(), 'is_reference': np.tile([True]+[False]*7, 195).tolist(),
                  'reference_rows': np.repeat(np.arange(0, 1560, 8), 8).tolist()})


def test_one_full_hgb_fit_and_serialized_original_classes_preserve_raw_scores(tmp_path, monkeypatch):
    protocol, previous_protocol, previous = prior_fixture()
    fit.validate_k_binding(protocol, previous_protocol, previous)
    contrast = contrast_fixture(protocol)
    calls = []
    original_fit = ranker.HistGradientBoostingRegressor.fit
    def spy(estimator, features, targets, sample_weight=None):
        calls.append((features.copy(), targets.copy(), sample_weight.copy()))
        return original_fit(estimator, features, targets, sample_weight=sample_weight)
    monkeypatch.setattr(ranker.HistGradientBoostingRegressor, 'fit', spy)
    with threadpool_limits(limits=1):
        model, diagnostics = fit.fit_full_model(contrast, protocol)
        path = tmp_path / 'model.pkl'
        fit.serialize_model(model, contrast, protocol, path)
        payload = pickle.loads(path.read_bytes())
        restored = payload['rigid_model']
        predicted = restored.predict(contrast)['control_cost']['mean']
        np.testing.assert_array_equal(predicted, model.predict(contrast)['control_cost']['mean'])
    assert len(calls) == 1 and calls[0][0].shape == (1365, 29)
    mask = ~np.asarray(contrast.metadata['is_reference'])
    np.testing.assert_array_equal(calls[0][1], contrast.targets['interval_cost'][mask])
    np.testing.assert_array_equal(calls[0][2], diagnostics['server_only_sample_weights'])
    assert diagnostics['actual_group_count'] == 195 and diagnostics['rows'] == 1560
    assert restored.anchor_model is restored.correction_model
    assert type(restored.anchor_model) is ranker.PairwiseActionAdvantageRegressor
    assert restored.anchor_model.feature_names == tuple(protocol['original_feature_names'])
    assert payload['protocol'] == fit.MODEL_PROTOCOL and payload['target_budget'] == 195
    assert payload['selected_groups'] == protocol['original_group_ids'] + protocol['additional_group_ids']
    assert payload['sample_weight_source_protocol'] == 'group_action_range_v1'
    assert np.all(predicted[~mask] == 0.)


@pytest.mark.parametrize('change', ['threshold', 'feature_order', 'shard_binding', 'criterion'])
def test_changed_k_inputs_or_unqualified_result_are_rejected(change):
    protocol, previous_protocol, previous = prior_fixture()
    protocol = deepcopy(protocol)
    if change == 'threshold':
        protocol['threshold'] = .1
    elif change == 'feature_order':
        protocol['original_feature_names'].reverse()
    elif change == 'shard_binding':
        protocol['collection_shards'][0]['bank'] = 'replacement.npz'
    else:
        previous['metrics']['model']['mean_cost'] = previous['metrics']['original_a']['mean_cost']
    with pytest.raises(ValueError, match='qualifying K'):
        fit.validate_k_binding(protocol, previous_protocol, previous)


def test_incomplete_or_reordered_full_group_roster_never_reaches_fit(monkeypatch):
    protocol, _, _ = prior_fixture()
    contrast = contrast_fixture(protocol)
    contrast.metadata['action_group_ids'][0:8] = [protocol['original_group_ids'][1]]*8
    monkeypatch.setattr(fit, 'fit_raw_target_anchor', lambda *args, **kwargs: pytest.fail('Invalid roster reached fitting'))
    with pytest.raises(ValueError, match='frozen row order'):
        fit.fit_full_model(contrast, protocol)
