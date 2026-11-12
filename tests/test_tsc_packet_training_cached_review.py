"""Synthetic cached-rebuild regressions; no native start or real optimizer."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.tsc_target_packet_training_20261109 import training_data as data
from test_tsc_packet_training_data import fixture, persist


def test_cached_JSON_seed_keys_reset_derived_counts_without_reestimating(tmp_path, monkeypatch):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'train'
    persist(directory, frames, actions)
    expected = data.build_training_data({1: directory}, protocol, static, seeds)
    laws, reservoirs, sampling, history = data.estimate_laws_and_sample({1: directory}, protocol, static)
    laws, reservoirs, sampling, history = json.loads(json.dumps([laws, reservoirs, sampling, history]))
    assert set(reservoirs) == {'1'}
    for stat in sampling['1'].values():
        stat.update(retained=123, allzero_excluded=456, allzero_excluded_positive_labels=789)
    cached_inputs = deepcopy((laws, reservoirs, sampling, history))

    def forbidden_estimation(*args, **kwargs):
        raise AssertionError('A cache rebuild must not estimate history laws again')

    monkeypatch.setattr(data, 'estimate_laws_and_sample', forbidden_estimation)
    actual = data.build_from_cached({1: directory}, protocol, static, seeds,
                                   laws, reservoirs, sampling, history)
    assert actual['sampling'] == expected['sampling']
    assert actual['record_references'] == expected['record_references']
    assert actual['laws'] == expected['laws'] and actual['history'] == expected['history']
    assert (laws, reservoirs, sampling, history) == cached_inputs
    for head in data.HEADS:
        for name in expected['heads'][head]:
            np.testing.assert_array_equal(actual['heads'][head][name], expected['heads'][head][name])


def test_cached_rebuild_reconsiders_old_allzero_rows_from_full_reservoir(tmp_path, monkeypatch):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'train'
    persist(directory, frames, actions)
    laws, reservoirs, sampling, history = data.estimate_laws_and_sample({1: directory}, protocol, static)
    original_candidate = data.candidate_features

    def old_allzero(*args):
        candidate = original_candidate(*args)
        candidate['yield_risk'] = False
        return candidate

    monkeypatch.setattr(data, 'candidate_features', old_allzero)
    previous = data.build_from_cached({1: directory}, protocol, static, seeds,
                                     laws, reservoirs, sampling, history)
    assert previous['record_references']['yield_entry'] == []
    assert previous['sampling']['1']['yield_entry']['allzero_excluded_positive_labels'] == 1

    monkeypatch.setattr(data, 'candidate_features', original_candidate)
    restored = data.build_from_cached({1: directory}, protocol, static, seeds,
                                     laws, reservoirs, previous['sampling'], history)
    assert len(restored['record_references']['yield_entry']) == 1
    assert restored['record_references']['yield_entry'][0]['y'] == 1
    assert restored['sampling']['1']['yield_entry']['sampled'] == 1
    assert restored['sampling']['1']['yield_entry']['retained'] == 1
    assert restored['sampling']['1']['yield_entry']['allzero_excluded'] == 0
    assert restored['sampling']['1']['yield_entry']['allzero_excluded_positive_labels'] == 0


@pytest.mark.parametrize('head', ['start', 'yield_entry', 'lane_change'])
def test_zero_prior_candidate_cannot_rescue_an_unsupported_label(head):
    from experiments.tsc_target_packet_training_20261109.corrected_fit import check_zero_support

    X = np.zeros((1, 2, 38))
    X[:, :, 0] = 1.
    heads = {name: dict(X=X.copy(), prior=np.array([[1., 0.]]),
                       risk=np.ones((1, 2), dtype=bool), y=np.array([0.]))
             for name in data.HEADS}
    heads['motion'].update(upper=np.ones((1, 2)), a=np.full((1, 2), 2.), b=np.full((1, 2), 4.5))
    heads['lane_change'].update(y=np.array([1.]), mask=np.ones((1, 2, 3), dtype=bool))
    refs = {name: [dict(seed=1, time_sec=0, packet_id='p', y=row['y'][0])]
            for name, row in heads.items()}
    if head == 'lane_change':
        heads[head]['y'][0] = 2.
        heads[head]['mask'][0, 0] = [False, True, False]
    else:
        heads[head]['y'][0] = 1.
        heads[head]['risk'][0, 0] = False
    refs[head][0]['y'] = float(heads[head]['y'][0])
    original = deepcopy(heads)
    rejected = check_zero_support(heads, refs)
    assert rejected['unsupported_total'] == 1
    for name, row in original.items():
        for key, value in row.items():
            np.testing.assert_array_equal(heads[name][key], value)

    # Changing actual positive-prior support admits the row; no epsilon or
    # renormalization of the zero-prior candidate can do this instead.
    if head == 'lane_change':
        heads[head]['mask'][0, 0, 2] = True
    else:
        heads[head]['risk'][0, 0] = True
    admitted = check_zero_support(heads, refs)
    assert admitted['unsupported_total'] == 0
    assert np.isfinite(admitted['initial_loss'])
