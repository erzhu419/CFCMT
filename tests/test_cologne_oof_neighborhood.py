from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np
import pytest

from cf_h2o.traffic_signal.action_ranker import CAUSAL_RIGID_RANKING_PARENTS
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


@pytest.fixture
def neighborhood(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('cologne_oof_neighborhood')


def b100_case():
    names = tuple(reversed(CAUSAL_RIGID_RANKING_PARENTS))
    features, groups, states, references, deltas, records = [], [], [], [], [], []
    folds = []
    for seed_index, (seed, count, query_count) in enumerate(((2027, 34, 17), (3037, 33, 17), (4047, 33, 14))):
        folds.append({'heldout_seed': seed, 'fit_group_count': 100-count, 'heldout_group_count': count})
        for interval in range(1, count+1):
            group = f'cologne1:seed{seed}:{interval}:tls'
            group_delta = np.asarray([0., [-.2, .1, 0.][interval % 3]*(1+seed_index*.1), -.03, .01, .02, .03, -.01, .04])
            costs = 1. + group_delta
            selected = 1 if interval <= query_count else 0
            records.append({'group_id': group, 'seed': seed, 'selected_index': selected, 'reference_index': 0,
                'candidate_states': [f'phase{candidate}' for candidate in range(8)], 'label_costs': costs.tolist(),
                'different_action': selected != 0, 'predicted_advantage': .2 if selected else 0.,
                'actual_advantage': float(-group_delta[selected])})
            for candidate in range(8):
                row = np.zeros(len(names))
                row[names.index('total_q')] = candidate if candidate else 10000.
                row[names.index('total_veh')] = interval
                row[names.index('mean_speed')] = 5. if seed == 2027 else 0.
                features.append(row)
                groups.append(group)
                states.append(f'phase{candidate}')
                references.append(candidate == 0)
                deltas.append(group_delta[candidate])
    contrast = MechanismDataset(feature_names=names, features=np.asarray(features), context_names=(),
        context=np.empty((800, 0)), priors={}, targets={'interval_cost': np.asarray(deltas)},
        domains=np.full(800, 'cologne'), metadata={'action_group_ids': groups,
            'candidate_states': states, 'is_reference': references})
    return contrast, records, list(CAUSAL_RIGID_RANKING_PARENTS), folds


def test_training_only_scaling_and_three_distinct_neighbor_groups(neighborhood):
    case = b100_case()
    result = neighborhood.audit_neighborhood(*case)
    assert result['query_count'] == 48 and all(result['checks'].values())
    assert [row['query_count'] for row in result['folds']] == [17, 17, 14]
    assert [row['fit_row_count'] for row in result['folds']] == [462, 469, 469]
    assert [row['group_id'] for row in result['queries']] == [row['group_id'] for row in case[1] if row['different_action']]
    fold = result['folds'][0]
    q_index = case[2].index('total_q')
    assert fold['feature_mean'][q_index] == 4.
    assert fold['feature_std'][q_index] == 2.
    assert fold['feature_min'][q_index] == 1. and fold['feature_max'][q_index] == 7.
    assert fold['training_seeds'] == [3037, 4047] and fold['training_reference_rows'] == 0
    query = result['queries'][0]
    assert query['nearest_distance'] == 0.
    assert [row['group_id'] for row in query['neighbors']] == [
        'cologne1:seed3037:1:tls', 'cologne1:seed4047:1:tls', 'cologne1:seed3037:2:tls']
    assert all(row['selected_index'] == 1 and row['reference_index'] == 0 for row in query['neighbors'])
    assert query['outcome'] == 'harmful' and query['actual_gain'] == -.1
    assert query['outside_training_range'] == [
        {'feature': 'mean_speed', 'value': 5., 'minimum': 0., 'maximum': 0., 'training_constant': True}]


def test_heldout_features_do_not_enter_fold_scaling_and_changed_labels_do_not_choose_neighbors(neighborhood):
    contrast, records, names, folds = b100_case()
    original = neighborhood.audit_neighborhood(contrast, records, names, folds)
    heldout = np.asarray([group.startswith('cologne1:seed2027:') for group in contrast.metadata['action_group_ids']])
    contrast.features[heldout, contrast.feature_names.index('total_q')] += 100.
    changed_features = neighborhood.audit_neighborhood(contrast, records, names, folds)
    assert changed_features['folds'][0] == original['folds'][0]
    contrast, records, names, folds = b100_case()
    contrast.targets['interval_cost'][1] = -.4
    records[0]['label_costs'][1] = .6
    records[0]['actual_advantage'] = .4
    changed_labels = neighborhood.audit_neighborhood(contrast, records, names, folds)
    assert changed_labels['queries'][0]['neighbors'] == original['queries'][0]['neighbors']
    assert changed_labels['queries'][0]['outcome'] == 'beneficial'
    assert changed_labels['queries'][0]['actual_gain'] == .4


def test_tied_candidate_distances_use_group_then_original_candidate_order(neighborhood):
    contrast, records, names, folds = b100_case()
    # Equalize candidate features in each training group, leaving reference rows excluded.
    for start in range(0, contrast.size, 8):
        contrast.features[start+1:start+8, contrast.feature_names.index('total_q')] = 1.
    result = neighborhood.audit_neighborhood(contrast, records, names, folds)
    query = result['queries'][0]
    assert [row['group_id'] for row in query['neighbors'][:2]] == ['cologne1:seed3037:1:tls', 'cologne1:seed4047:1:tls']
    assert [row['selected_index'] for row in query['neighbors']] == [1, 1, 1]
    assert query['nearest_distance'] == 0.


def test_cached_label_and_fold_mismatches_are_rejected(neighborhood):
    contrast, records, names, folds = b100_case()
    changed = deepcopy(records)
    changed[0]['actual_advantage'] += .01
    with pytest.raises(ValueError, match='cached OOF label'):
        neighborhood.audit_neighborhood(contrast, changed, names, folds)
    changed_folds = deepcopy(folds)
    changed_folds[0]['fit_group_count'] -= 1
    with pytest.raises(ValueError, match='frozen fit fold'):
        neighborhood.audit_neighborhood(contrast, records, names, changed_folds)
