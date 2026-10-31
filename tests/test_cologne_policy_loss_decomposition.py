from copy import deepcopy

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.cologne_policy_loss_decomposition import decompose_policy_loss


def policy_case():
    groups = [f'cologne1:seed2027:{index}:tls' for index in range(4)]
    states = [f'phase{index}' for index in range(8)]
    costs = np.asarray([
        [.1, 4., 2., 2., 5., 6., 7., 8.],
        [.1, 2., 3., 1., 5., 6., 7., 8.],
        [.1, 2., 1., 3., 5., 6., 7., 8.],
        [.1, 1., 2., 3., 5., 6., 7., 8.],
    ])
    records = []
    for index, selected in enumerate([2, 1, 1, 3]):
        scores = np.ones(8)
        scores[selected] = 0.
        records.append(group_record(groups[index], states, costs[index], scores, 3, selected))
    features = np.tile(np.column_stack(([1., 2., 3., 10., 4., 5., 6., 7.],
                                        [0., 1., 1., 0., 1., 1., 1., 1.])), (4, 1))
    bank = MechanismDataset(feature_names=('green_q', 'switch_indicator'), features=features,
        context_names=(), context=np.empty((32, 0)), priors={},
        targets={key: costs.ravel().copy() for key in ('interval_cost', 'prefix_mean_cost_450s')},
        domains=np.full(32, 'cologne1'), metadata={
            'action_group_ids': np.repeat(groups, 8).tolist(), 'candidate_states': states * 4})
    return records, bank, groups


def test_allowed_oracle_excludes_vetoed_stay_keeps_pp_and_decomposes_all_categories():
    records, bank, groups = policy_case()
    original = deepcopy(records)
    result = decompose_policy_loss(records[::-1], bank, groups)
    assert result['group_count'] == 4 and all(result['checks'].values())
    assert records == original
    rows = result['rows']
    assert [row['group_id'] for row in rows] == groups
    assert [row['category'] for row in rows] == [
        'achieves_label_optimum', 'unnecessary_override_when_pp_optimal',
        'suboptimal_switch_when_benefit_available', 'missed_beneficial_switch']
    assert all(row['allowed_indices'] == [1, 2, 3, 4, 5, 6, 7] for row in rows)
    assert rows[0]['oracle_indices'] == [2, 3]
    assert rows[0]['oracle_cost'] == 2. and rows[0]['regret'] == 0.
    assert [row['deployed_index'] for row in rows] == [2, 1, 1, 3]
    assert [row['realized_gain'] for row in rows] == [0., -1., 1., 0.]
    assert [row['regret'] for row in rows] == [0., 1., 1., 2.]
    for row in rows:
        assert row['pp_to_oracle_gain'] == row['realized_gain'] + row['regret']


def test_threshold_tolerance_keeps_the_actual_proposal_without_reranking():
    records, bank, groups = policy_case()
    records[3]['selected_index'] = 1
    records[3]['actual_advantage'] = 2.
    for predicted, deployed in [(0., 3), (1e-12, 3), (2e-12, 1)]:
        records[3]['predicted_advantage'] = predicted
        row = decompose_policy_loss(records, bank, groups)['rows'][3]
        assert row['selected_index'] == 1 and row['deployed_index'] == deployed
    records[3]['predicted_advantage'] = .2
    assert decompose_policy_loss(records, bank, groups, threshold=.2)['rows'][3]['deployed_index'] == 3


def test_cost_ties_within_tolerance_are_all_retained():
    records, bank, groups = policy_case()
    for target in bank.targets.values():
        target[2] += 5e-13
    records[0]['label_costs'][2] += 5e-13
    records[0]['actual_advantage'] = 2. - records[0]['label_costs'][2]
    row = decompose_policy_loss(records, bank, groups)['rows'][0]
    assert row['oracle_indices'] == [2, 3]
    assert row['category'] == 'achieves_label_optimum'


@pytest.mark.parametrize('mismatch', ['states', 'labels', 'reference', 'selected', 'native_cost', 'forbidden_stay'])
def test_original_action_and_label_alignment_mismatches_are_rejected(mismatch):
    records, bank, groups = policy_case()
    if mismatch == 'states':
        records[0]['candidate_states'][0:2] = records[0]['candidate_states'][1::-1]
    elif mismatch == 'labels':
        records[0]['label_costs'][1] += .2
    elif mismatch == 'reference':
        records[0]['reference_index'] = 2
    elif mismatch == 'selected':
        records[0]['selected_index'] = 1
    elif mismatch == 'native_cost':
        bank.targets['interval_cost'][1] += .2
    else:
        records[0]['selected_index'] = 0
        records[0]['actual_advantage'] = 1.9
    with pytest.raises(ValueError):
        decompose_policy_loss(records, bank, groups)
