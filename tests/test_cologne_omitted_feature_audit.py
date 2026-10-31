from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_cologne_oof_neighborhood import b100_case


@pytest.fixture
def case(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    helper = importlib.import_module('cologne_omitted_feature_audit')
    neighborhood = importlib.import_module('cologne_oof_neighborhood')
    original, records, model_features, folds = b100_case()
    previous = neighborhood.audit_neighborhood(original, records, model_features, folds)
    names = list(helper.OMITTED_FEATURE_NAMES)
    values = np.zeros((original.size, 13))
    values[:, 0] = [5. if group.startswith('cologne1:seed2027:') else 0.
                    for group in original.metadata['action_group_ids']]
    candidates = np.tile(np.arange(8), 100)
    for family in range(4):
        values[:, 1+3*family] = candidates*(family+1)
        values[:, 3+3*family] = candidates*(family+1)
    contrast = replace(original, feature_names=(*original.feature_names, *names),
                       features=np.column_stack((original.features, values)))
    return helper, contrast, previous, folds, names


def strip_feature_values(queries):
    return [{**{key: value for key, value in query.items() if key not in ('feature_values', 'neighbors')},
             'neighbors': [{key: value for key, value in neighbor.items() if key != 'feature_values'}
                           for neighbor in query['neighbors']]} for query in queries]


def test_adds_actual_thirteen_columns_without_changing_frozen_neighbor_order(case):
    helper, contrast, previous, folds, names = case
    before = deepcopy(previous)
    result = helper.audit_omitted_features(contrast, previous, folds, names)
    assert result['query_count'] == 48 and result['pair_count'] == 144
    assert all(result['checks'].values())
    assert strip_feature_values(result['queries']) == previous['queries'] == before['queries']
    assert previous == before
    query = result['queries'][0]
    assert query['feature_values'] == [5., 1., 0., 1., 2., 0., 2., 3., 0., 3., 4., 0., 4.]
    assert query['neighbors'][0]['feature_values'] == [0., 1., 0., 1., 2., 0., 2., 3., 0., 3., 4., 0., 4.]


def test_fold_stats_exclude_heldout_seed_and_reference_rows_and_keep_constants_raw(case):
    helper, contrast, previous, folds, names = case
    result = helper.audit_omitted_features(contrast, previous, folds, names)
    assert [row['fit_row_count'] for row in result['folds']] == [462, 469, 469]
    fold = result['folds'][0]
    assert fold['training_seeds'] == [3037, 4047] and fold['training_reference_rows'] == 0
    assert fold['feature_min'][1] == 1. and fold['feature_max'][1] == 7.
    assert fold['feature_mean'][1] == 4. and fold['feature_std'][1] == 2.
    assert names[0] in fold['constant_features']
    assert fold['feature_min'][0] == fold['feature_max'][0] == fold['feature_std'][0] == 0.
    heldout = np.asarray([group.startswith('cologne1:seed2027:') for group in contrast.metadata['action_group_ids']])
    contrast.features[heldout, -13:] += 100.
    changed = helper.audit_omitted_features(contrast, previous, folds, names)
    assert changed['folds'][0] == fold
    assert strip_feature_values(changed['queries']) == previous['queries']
    assert changed['queries'][0]['feature_values'][0] == 105.


@pytest.mark.parametrize('mismatch', ['selected_index', 'reference_index', 'raw_gain'])
def test_frozen_query_and_neighbor_action_or_label_mismatches_are_rejected(case, mismatch):
    helper, contrast, previous, folds, names = case
    changed = deepcopy(previous)
    if mismatch == 'selected_index':
        changed['queries'][0]['selected_index'] = 8
    elif mismatch == 'reference_index':
        changed['queries'][0]['neighbors'][0]['reference_index'] = 2
    else:
        changed['queries'][0]['neighbors'][0]['raw_gain'] += .05
    with pytest.raises(ValueError, match='differs from the bank'):
        helper.audit_omitted_features(contrast, changed, folds, names)
